"""校验层：语义等价聚类 + 跨源投票 + 是否追问用户。

判定标准是语义等价，不是字符串相等 —— `\\frac{1}{2}x` 和 `\\frac{x}{2}` 必须算同一个。
"""

from __future__ import annotations

import difflib
import threading

import sympy

from .latex import parse, to_expr
from .types import Candidate, Cluster, ParsedExpr, VerificationResult

_SIMPLIFY_TIMEOUT = 3.0


def _simplify_is_zero(diff, timeout: float = _SIMPLIFY_TIMEOUT) -> bool:
    """带超时的化简判零。简化卡死时线程被丢弃，不让整个请求挂住。"""
    box: dict[str, bool] = {}

    def work() -> None:
        try:
            box["v"] = sympy.simplify(diff) == 0
        except Exception:  # noqa: BLE001
            box["v"] = False

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout)
    return box.get("v", False)


def equivalent(a: ParsedExpr, b: ParsedExpr) -> bool:
    ea, eb = to_expr(a), to_expr(b)
    if ea is None or eb is None:
        return False
    if ea == eb:
        return True
    try:
        diff = sympy.together(ea - eb)
    except Exception:  # noqa: BLE001
        return False
    if diff == 0:
        return True
    if not diff.free_symbols:
        try:
            return abs(complex(diff.evalf())) < 1e-9
        except Exception:  # noqa: BLE001
            return False
    return _simplify_is_zero(diff)


def _ambiguity_hint(chosen: str, other: str) -> str:
    """找出两个候选的差异片段，供交互层高亮。"""
    sm = difflib.SequenceMatcher(None, chosen, other)
    bits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        a, b = chosen[i1:i2], other[j1:j2]
        if not a and not b:
            continue
        bits.append(f"{a or '∅'} ↔ {b or '∅'}")
    return "；".join(bits[:3])


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def cluster(cands: list[Candidate]) -> tuple[list[Cluster], list[Candidate]]:
    """按语义等价分组。解析不了的进入 rejected，不参与投票。"""
    parsed: list[tuple[Candidate, ParsedExpr]] = []
    rejected: list[Candidate] = []
    for c in cands:
        pe = parse(c.latex)
        if pe.ok:
            parsed.append((c, pe))
        else:
            c.note = c.note or f"无法解析（{pe.error.split(':')[0]}）"
            rejected.append(c)

    uf = _UF(len(parsed))
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            if equivalent(parsed[i][1], parsed[j][1]):
                uf.union(i, j)

    buckets: dict[int, Cluster] = {}
    for idx, (c, pe) in enumerate(parsed):
        root = uf.find(idx)
        # 用簇里第一条"原始候选"做代表，而不是 pe.normalized。
        # 规整版只该出现在解析环节：把它当 chosen 传给界面的候选项单选组，
        # 会因为和原始字符串对不上而被 Gradio 判为"值不在选项里"，整页报错。
        buckets.setdefault(root, Cluster(latex=(c.latex or "").strip())).members.append(c)

    clusters = sorted(
        buckets.values(),
        key=lambda cl: (cl.size, sum(m.conf or 0.5 for m in cl.members)),
        reverse=True,
    )
    return clusters, rejected


def _vendor(source: str) -> str:
    """把 "simpletex-turbo" / "qwen3-vl-plus" 归一成厂商名。

    判"是否独立"要看厂商，不是看模型名后缀：同一家的两个版本
    训练数据和失效模式高度重合，凑不出独立旁证。
    """
    s = (source or "unknown").lower()
    for v in ("simpletex", "qwen", "gpt", "claude", "glm", "gemini", "fixture", "manual", "user"):
        if s.startswith(v):
            return v
    return s


def verify(cands: list[Candidate], review_threshold: float = 0.6) -> VerificationResult:
    raw = list(cands)
    cands = [c for c in cands if (c.latex or "").strip()]
    if not cands:
        notes = [f"{c.source}：{c.note}" for c in raw if c.note]
        return VerificationResult(
            verdict="failed",
            reasons=["没有拿到任何识别结果"] + notes[:3],
        )

    clusters, rejected = cluster(cands)
    total = len(cands)

    if not clusters:
        return VerificationResult(
            verdict="failed",
            rejected=rejected,
            reasons=["所有候选都无法解析为数学表达式"],
        )

    best = clusters[0]
    conf = best.size / total
    sources = sorted({(c.source or "unknown") for c in cands})
    vendors = sorted({_vendor(s) for s in sources})
    independent = len(vendors) >= 2

    vr = VerificationResult(
        verdict="ok",
        chosen=best.latex,
        confidence=conf,
        clusters=clusters,
        rejected=rejected,
        sources=sources,
        independent=independent,
    )

    # 同一模型采样多次得到的一致，是自一致性，不是交叉验证。
    # 单源时错误模式高度相关，两次都对得上并不能说明没认错——
    # 所以这里必须把置信度压下来，否则就成了"用一个模型自证清白"。
    if not independent:
        cap = 0.8
        if conf > cap:
            conf = cap
            vr.confidence = cap
        vr.reasons.append(
            f"仅 {'、'.join(sources)} 一个来源，一致度是自一致性，没有独立旁证"
        )

    single = best.members[0]
    if independent and single.conf is not None and single.conf < review_threshold:
        vr.verdict = "review"
        vr.reasons.append(f"识别器自评置信度仅 {single.conf:.2f}")

    if len(clusters) >= 2:
        second = clusters[1]
        if second.size == best.size:
            vr.verdict = "review"
            vr.reasons.append(
                f"两派各 {best.size} 票，无法裁决：{_ambiguity_hint(best.latex, second.latex)}"
            )
        elif conf < review_threshold:
            vr.verdict = "review"
            vr.reasons.append(
                f"最大一致簇仅占 {conf:.0%}：{_ambiguity_hint(best.latex, second.latex)}"
            )
        else:
            vr.reasons.append(
                f"少数派意见已记录：{_ambiguity_hint(best.latex, second.latex)}"
            )

    if rejected:
        names = "、".join(sorted({c.source for c in rejected}))
        vr.reasons.append(f"{names} 的结果无法解析，已排除")
        if best.size == 1:
            vr.verdict = "review"
            vr.reasons.append("唯一可用候选没有旁证，建议人工确认")

    if not vr.reasons:
        vr.reasons.append(f"{best.size}/{total} 个候选语义一致")
    return vr
