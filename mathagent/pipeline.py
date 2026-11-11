"""流水线编排：识别 -> 校验 -> 求解。

一句话原则：**识别可以不确定，但不能装作确定。**
verdict 为 review 时不往下硬算，把选择权交回用户。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from . import config
from .explain import explain as explain_steps
from .latex import parse
from .perceive import FixtureRecognizer, Recognizer, build_ensemble, page_note
from .route import detect
from .solve import solve as solve_parsed
from .types import Candidate, ParsedExpr, PipelineResult, SolveResult, VerificationResult
from .verify import verify as verify_candidates

RUNS = config.ROOT / "runs"

NO_KEY_HINT = (
    "还没有可用的识别通道。请在项目根目录的 .env 里填写 "
    "SIMPLETEX_TOKEN 或 DASHSCOPE_API_KEY，保存后点界面上的「测试连接」。"
)


def _collect(image, recognizers: list[Recognizer], k: int) -> list[Candidate]:
    cands: list[Candidate] = []
    for rec in recognizers:
        try:
            cands.extend(rec.recognize(image, k=k))
        except Exception as exc:  # noqa: BLE001
            cands.append(Candidate(latex="", source=rec.name, note=f"{type(exc).__name__}: {exc}"))
    return cands


def _longest(cands: list[Candidate]) -> str:
    """候选里最长的一条。

    整页题被读成一段中文时，投票往往全军覆没（判「无法解析」），chosen 为空。
    这时候最长的候选反而最能说明「这拍的到底是什么」。
    """
    texts = [(c.latex or "").strip() for c in cands]
    return max(texts, key=len, default="")


def run(
    image: str | Path | None = None,
    latex: str | None = None,
    k: int = 2,
    allow_fixture: bool = True,
    force_fixture: bool = False,
    confirmed: bool = False,
    save: bool = True,
) -> PipelineResult:
    """force_fixture=True 时只用内置夹具，不花钱也不受模型随机性影响，
    演示和回归测试用它才能得到可复现的结果。"""
    timings: dict[str, float] = {}
    note = ""

    t0 = time.time()
    if latex and latex.strip():
        cands = [Candidate(latex=latex.strip(), source="manual" if not confirmed else "user")]
    elif image:
        recs = [] if force_fixture else build_ensemble(k=k)
        if not recs:
            if not allow_fixture:
                raise ValueError(NO_KEY_HINT)
            fix = FixtureRecognizer()
            if Path(image).stem not in fix.data:
                raise ValueError(
                    "未配置任何密钥，且这张图不在内置演示样例里，无法识别。" + NO_KEY_HINT
                )
            recs = [fix]
        note = page_note(image)
        cands = _collect(image, recs, k)
    else:
        raise ValueError("image 和 latex 至少给一个")
    timings["perceive"] = time.time() - t0

    t0 = time.time()
    if confirmed:
        v = VerificationResult(
            verdict="ok",
            chosen=latex or "",
            confidence=1.0,
            reasons=["用户已确认识别结果"],
        )
    else:
        v = verify_candidates(cands)
    timings["verify"] = time.time() - t0

    # 入口判断：这不是算式的话，后面的求解一个字都不该做。
    # 硬算下去只会把「投错了门」包装成「解析失败」，用户还得自己猜原因。
    #
    # 注意要拿**所有候选里最长的那条**兜底：整页题的文字在投票阶段往往被
    # 判成「无法解析」，chosen 会是空串——只盯着 chosen 就永远看不出这是几何题。
    route_text = v.chosen or (latex or "") or _longest(cands)
    route = detect(route_text)

    t0 = time.time()
    pe = parse(route_text)
    timings["parse"] = time.time() - t0

    if route.kind != "formula":
        v.verdict = "routed"
        v.reasons.insert(0, route.note)

    t0 = time.time()
    sr = solve_parsed(pe) if pe.ok else SolveResult(kind=pe.kind, error=pe.error)
    timings["solve"] = time.time() - t0

    # 认得出但算不出，多半是识别错了，降级为待确认
    if v.verdict == "ok" and pe.ok and not sr.ok and sr.kind != "expression":
        v.verdict = "review"
        v.confidence = min(v.confidence, 0.5)
        v.reasons.append(f"能解析但求不出解（{sr.error}），可能是符号识别有误")

    # 讲解只对"已经算对"的结果做，算错的时候讲得越顺越有害
    t0 = time.time()
    ex = None
    if sr.ok and sr.steps and config.explain_enabled():
        ex = explain_steps(pe, sr)
    timings["explain"] = time.time() - t0

    result = PipelineResult(
        recognition=cands,
        verification=v,
        parsed=pe,
        solved=sr,
        timings=timings,
        explain=ex,
        image_note=note,
        route="" if route.kind == "formula" else route.kind,
        route_note=route.note,
        route_hints=route.hints,
    )
    if save:
        _save(result, image)
    return result


def _save(result: PipelineResult, image) -> None:
    try:
        RUNS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        payload = result.to_dict()
        payload["image"] = str(image) if image else None
        (RUNS / f"{stamp}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001
        pass


def resolve(latex: str) -> PipelineResult:
    """用户在界面上点选/改好式子后，从这一步继续。"""
    return run(latex=latex, confirmed=True)
