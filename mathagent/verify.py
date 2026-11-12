"""校验层：语义等价聚类 + 跨源投票 + 是否追问用户。

判定标准是语义等价，不是字符串相等 —— `\\frac{1}{2}x` 和 `\\frac{x}{2}` 必须算同一个。
"""

from __future__ import annotations

import difflib

import sympy

from . import guard
from .latex import parse, systems_equivalent, to_expr
from .types import Candidate, Cluster, ParsedExpr, VerificationResult

_SIMPLIFY_TIMEOUT = 3.0

# 两个常数差多少才算同一个。取绝对值，不取相对值：手写数字错一位就是量级差，
# 相对容差会把 1200 和 1212 也算成一样，那正是最该抓住的错。
_NUM_TOL = 1e-9


def _simplify_is_zero(diff, timeout: float = _SIMPLIFY_TIMEOUT) -> bool:
    """带超时的化简判零。

    化简卡死时把子进程杀掉，而不是只丢下一条还在算的线程——线程杀不掉，
    它会一直握着 GIL，把「我放弃了」变成「整个页面都慢了」。
    """
    return bool(guard.run("simplify_zero", diff, timeout=timeout, default=False))


def equivalent(a: ParsedExpr, b: ParsedExpr) -> bool:
    """语义等价。

    方程组单独走一路：它的"值"不是把各条方程相加能表示的。求和会把约束抵消
    （`{x+y=3, x-y=1}` 与 `{x+y=30, x-y=-26}` 都是 `2x-4`），两个解不同的题
    就成了一簇，投票还会把这个当成"两个来源一致"。
    """
    sa = a.kind == "system"
    sb = b.kind == "system"
    if sa or sb:
        # 方程组和单方程不是一回事，不能互认
        return bool(sa and sb and systems_equivalent(a.obj or [], b.obj or []))

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
            return abs(complex(diff.evalf())) < _NUM_TOL
        except Exception:  # noqa: BLE001
            return False
    if _simplify_is_zero(diff):
        return True
    if a.kind == "equation" and b.kind == "equation":
        # `x=1` 与 `1=x`、`2x=2` 是同一个方程：零点相同，只是两边同乘了一个
        # 非零常数。压成 lhs-rhs 之后差的是个倍数，不是零，所以上面的判零
        # 抓不到。不认这一条，两个模型把同一行读成同一个方程的两种写法，
        # 会被当成"两个来源在打架"——该收敛的地方反而判了 review。
        # 只对方程这么做：表达式 `x` 和 `2x` 是两回事。
        return bool(guard.run("scaled", ea, eb, timeout=_SIMPLIFY_TIMEOUT, default=False))
    return False


def _ambiguity_hint(chosen: str, other: str) -> str:
    """找出两个候选的差异片段，供交互层高亮。

    只差一处时给片段（`x-y=1` 与 `x-y=-1` → `∅ ↔ -`，一眼看出少了个负号）；
    差好几处时片段反而读不懂（`∅ ↔ 1.2\\times1；∅ ↔ ^{-9}`），整句摊开更好读。
    """
    sm = difflib.SequenceMatcher(None, chosen, other)
    bits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        a, b = chosen[i1:i2], other[j1:j2]
        if not a and not b:
            continue
        bits.append(f"{a or '∅'} ↔ {b or '∅'}")
    if len(bits) != 1:
        return f"{chosen.strip()} ↔ {other.strip()}"
    return bits[0]


def _clusters(parsed: list[tuple[Candidate, ParsedExpr]]) -> list[tuple[Cluster, ParsedExpr]]:
    """按语义等价分组，**每个候选只跟簇的代表比**。

    以前用并查集把"两两等价"传递起来，这对符号严格等价没问题（它本来就是
    等价关系），对数近似相等是错的：0、6e-10、1.2e-9 两两相邻都在容差内，
    首尾差 1.2e-9 超了，并查集却把三个并成一簇，三票一致。
    改成对代表比之后，簇里的任何一个成员都跟代表在容差内，
    "一致"这句话才真的成立。

    代价是顺序相关：谁先来谁当代表。候选来自不同模型、没有先后之分，
    这点随机性比"静默合并"划算。
    """
    out: list[tuple[Cluster, ParsedExpr]] = []
    for c, pe in parsed:
        for cl, rep in out:
            try:
                if equivalent(rep, pe):
                    cl.members.append(c)
                    break
            except Exception:  # noqa: BLE001
                continue
        else:
            # 用簇里第一条"原始候选"做代表，而不是 pe.normalized。
            # 规整版只该出现在解析环节：把它当 chosen 传给界面的候选项单选组，
            # 会因为和原始字符串对不上而被 Gradio 判为"值不在选项里"，整页报错。
            out.append((Cluster(latex=(c.latex or "").strip(), members=[c]), pe))
    return out


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

    clusters = [
        cl for cl, _ in sorted(
            _clusters(parsed),
            key=lambda pair: (
                pair[0].size,
                sum(m.conf or 0.5 for m in pair[0].members),
            ),
            reverse=True,
        )
    ]
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
    # 旁证要按**赢家成员**的厂商算，不能看全场的厂商。qwen 两票说 x=1、
    # glm 一票说 x=2 时，全场的厂商数是 2，可赢家只有 qwen 一家——
    # 把反对票所在的厂商当成赢家的独立旁证，等于自己给自己背书。
    win_sources = sorted({(c.source or "unknown") for c in best.members})
    win_vendors = sorted({_vendor(s) for s in win_sources})
    independent = len(win_vendors) >= 2

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
            f"一致的候选都来自 {'、'.join(win_sources)}，"
            "同一家模型的一致是自一致性，没有独立旁证"
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
        elif not independent:
            # 赢家连第二个厂商都没有，却已经有另一派给了不同答案——
            # 票数占多不等于有人给它作证，这种一律交给人看。
            vr.verdict = "review"
            vr.reasons.append(
                f"赢家没有独立旁证，另一派却给出了不同答案："
                f"{_ambiguity_hint(best.latex, second.latex)}"
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
