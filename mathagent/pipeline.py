"""流水线编排：识别 -> 校验 -> 求解。

一句话原则：**识别可以不确定，但不能装作确定。**
verdict 为 review 时不往下硬算，把选择权交回用户。
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from . import config
from .explain import explain as explain_steps
from .latex import parse
from .perceive import (
    Deadline,
    FixtureRecognizer,
    Recognizer,
    budget_for,
    build_ensemble,
    call_recognize,
    page_note,
)
from .route import detect
from .solve import solve as solve_parsed
from .types import Candidate, ParsedExpr, PipelineResult, SolveResult, VerificationResult
from .verify import verify as verify_candidates

RUNS = config.ROOT / "runs"
log = logging.getLogger(__name__)

NO_KEY_HINT = (
    "还没有可用的识别通道。请在项目根目录的 .env 里填写 "
    "SIMPLETEX_TOKEN 或 DASHSCOPE_API_KEY，保存后点界面上的「测试连接」。"
)


def _one_source(rec: Recognizer, image, k: int,
                deadline: Deadline | None = None) -> list[Candidate]:
    """单个通道识别。异常就地转成带 note 的候选，不让一路故障拖垮整页。"""
    try:
        return call_recognize(rec, image, k, deadline)
    except Exception as exc:  # noqa: BLE001
        return [Candidate(latex="", source=rec.name, note=f"{type(exc).__name__}: {exc}")]


def _collect(image, recognizers: list[Recognizer], k: int,
             deadline: Deadline | None = None) -> list[Candidate]:
    """多个识别通道**并发**跑。

    这里原来是串行 for 循环，于是「SimpleTex 8 秒 + 通义 12 秒」要等 20 秒。
    两条通道互不依赖、连的是两家不同的服务，串行纯属白等——实测 perceive
    最长 21.7 秒，并发后这个时间由最慢的一条决定。

    候选顺序必须稳定（投票和展示都依赖它），所以用 map 而不是 as_completed，
    返回顺序与 recognizers 一致。k 次采样仍留在各通道内部串行：
    同一个模型连发 k 个请求更容易撞 429，换来的是退避重试，不划算。
    """
    if not recognizers:
        return []
    if len(recognizers) == 1:
        return _one_source(recognizers[0], image, k, deadline)
    with ThreadPoolExecutor(max_workers=len(recognizers), thread_name_prefix="perceive") as pool:
        groups = list(pool.map(lambda r: _one_source(r, image, k, deadline), recognizers))
    return [c for group in groups for c in group]


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
        # 单图识别看不到行数（这里还没切行），按 2 行的量给预算。
        # 它只约束"最多等多久"，正常请求远用不到——真撞上了说明服务在慢，
        # 这时候快速降级比让页面转两分钟更诚实。
        dl = Deadline(budget_for(2))
        cands = _collect(image, recs, k, dl)
        # 超时要能上报，否则"慢"和"认不出"在日志里长得一模一样
        timed_out = sum(1 for c in cands if "BudgetExhausted" in (c.note or ""))
        if timed_out:
            timings["perceive_timed_out"] = float(timed_out)
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

    # 认得出但算不出，多半是识别错了，降级为待确认。
    # 但「超出声明范围」不算：那是我们自己的边界，不是这张图的问题，
    # 标成"待核对"会让学生以为是自己的式子有问题。
    if v.verdict == "ok" and pe.ok and not sr.ok and sr.kind != "expression" and not sr.out_of_scope:
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
    """把这次运行落盘，方便事后复盘。

    落盘失败不能影响主流程（学生对"结果算出来了但存不下来"没兴趣），
    但也不能像以前那样 `except: pass`——真出问题时日志里一片空白，
    只能靠猜。这里留一条 warning 就够定位了。
    """
    try:
        RUNS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        payload = result.to_dict()
        payload["image"] = str(image) if image else None
        (RUNS / f"{stamp}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("运行记录写入失败（不影响结果）：%s: %s", type(exc).__name__, exc)
        return
    _prune_runs()


RUN_KEEP = 500  # 只留最近这么多次运行


def _prune_runs(keep: int = RUN_KEEP) -> int:
    """删掉最旧的运行记录。平时只做一次目录列举，超过上限才真删。

    不清的话这个目录会一直涨（调 prompt 时一天几百次），最后拖慢列举，
    也没有人真的会翻三个月前的某次运行。
    """
    try:
        files = list(RUNS.glob("*.json"))
        if len(files) <= keep:
            return 0
        files.sort(key=lambda p: p.stat().st_mtime)
        removed = 0
        for p in files[: len(files) - keep]:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
        return removed
    except Exception:  # noqa: BLE001
        return 0


def resolve(latex: str) -> PipelineResult:
    """用户在界面上点选/改好式子后，从这一步继续。"""
    return run(latex=latex, confirmed=True)
