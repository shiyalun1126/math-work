"""题目归档：一条事件一行 json，按时间排。

和 ``cases.py`` 的区别，一句话：**cases 回答「我缺哪个动作」，这里回答「我什么时候做了什么」。**

三个理由说明为什么另起一个文件，而不是把 cases 改一改：

1. ``cases`` 只收判错的。看过答案就走的那些题（搜题场景）根本不会进去，
   而用户要归档的恰恰包括它们。
2. 两种读法对存储的要求不一样。cases 是聚合视图（按错因分组、算占比），
   越少越好；这里是流水视图（按时间翻），一条不能少。
3. cases 已经被「错题本」页签依赖，动它会牵一片，收益不抵风险。

## 两个区是怎么来的

同一份数据，两种读法：

* **题目记录**：全部事件，按时间倒序。回答"我什么时候搜过什么、改过哪一页"。
* **待检查步骤**：把所有记录里的 ``pending`` 摊平，按时间倒序。
  回答"哪几步我现在还欠一个交代"。

``pending`` 有两个来源，都带 ``by`` 字段，界面上一眼能看出是谁标的：

* ``by="system"``——系统自己没把握（这一行没认准、题面读错、判不出来）。
  系统从不假装有把握，所以这类必须摊到台面上。
* ``by="manual"``——你自己点「这步回头再看」。

**时间存的是写入时刻，不是显示时算出来的。** 用户要的就是"这条是几点进来的"，
时间戳必须跟着记录走；否则文件一搬、时区一变，整个归档的时间线就全错了。
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config

HISTORY: Path = config.ROOT / "data" / "history.jsonl"

# 事件类型。名字用中文直接当界面标签，避免"kind=grade"这种内部术语漏到界面上。
KINDS = {
    "search": "搜题",
    "grade": "批改",
    "check": "手动标",
}

# 结论的字面说法。空字符串给搜题用——搜题没有"对不对"，只有"看过了"。
VERDICTS = {
    "pass": "全对",
    "partial": "部分对",
    "wrong": "有错",
    "fake": "伪推导",
    "gap": "有断点",
    "unread": "没读准",
    "unknown": "没判定",
}

# 双击按钮会连着触发两次。同一道题 3 秒内重复落两条，时间线上就是噪声。
# 不做更长的窗口：隔一分钟再看一遍同一道题，那是真有两次，不该合并。
_DEDUPE_SECONDS = 3


# --------------------------------------------------------------------------- 写

def safe(fn, *args, **kwargs):
    """跑一次归档写操作，失败只留痕、不往上抛。

    用户在页面上要的是这道题的答案，不是"记录没写进去"。所以这里不弹错，
    但也不静默：往 stderr 留一行，翻日志时能看出归档少了哪一条。

    返回被调函数的结果，失败时返回 ``None``——调用方本来就只是"顺手记一笔"，
    不需要区分这两种情况。
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[history] 归档失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def _now() -> datetime:
    return datetime.now()


def _read_raw() -> list[dict]:
    if not HISTORY.exists():
        return []
    out: list[dict] = []
    for line in HISTORY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue          # 半行（上次写到一半断电）不该让整个归档读不出来
        if isinstance(rec, dict) and rec.get("id"):
            out.append(rec)
    return out


def _write_raw(recs: list[dict]) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY.with_suffix(".jsonl.tmp")
    # 先写临时文件再换过去：mark_done 会重写整个文件，
    # 中途被打断的话，直接覆盖会让归档少一半。
    tmp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + ("\n" if recs else ""),
        encoding="utf-8",
    )
    tmp.replace(HISTORY)


def _recent_duplicate(kind: str, problem: str, now: datetime) -> dict | None:
    recs = _read_raw()
    for rec in reversed(recs[-3:]):
        if rec.get("kind") != kind or (rec.get("problem") or "") != (problem or ""):
            continue
        try:
            ts = datetime.fromisoformat(rec.get("ts", ""))
        except ValueError:
            continue
        if abs((now - ts).total_seconds()) <= _DEDUPE_SECONDS:
            return rec
    return None


def record(
    kind: str,
    problem: str,
    *,
    source: str = "text",
    image: str = "",
    verdict: str = "",
    error_type: str = "",
    headline: str = "",
    n_steps: int = 0,
    pending: list[dict] | None = None,
    detail: dict | None = None,
) -> dict:
    """落一条事件。重复点击在 3 秒内会被吞掉，返回已存在的那条。"""
    if kind not in KINDS:
        raise ValueError(f"未知事件类型 {kind!r}，只能是 {'/'.join(KINDS)}")
    now = _now()
    dup = _recent_duplicate(kind, problem, now)
    if dup is not None:
        return dup

    rec = {
        "id": "h" + uuid.uuid4().hex[:9],
        "ts": now.isoformat(timespec="seconds"),
        "kind": kind,
        "problem": (problem or "").strip(),
        "source": source,
        "image": image or "",
        "verdict": verdict or "",
        "error_type": error_type or "",
        "headline": headline or "",
        "n_steps": int(n_steps or 0),
        "pending": list(pending or []),
        "detail": dict(detail or {}),
    }
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


# ------------------------------------------------------------------- 结果快照
#
# 列表只需要几个字（时间/题目/结论），详情要的是"那次到底判了什么"。
# 两者放在同一条记录里，是因为它们同生同死：拆成两个文件就多一个可能对不上的东西。
#
# 代价是 jsonl 会变大（一次整页批改约 4~6 KB）。列表页每次都全读这个文件，
# 所以快照必须**裁剪**：行数封顶、每个字段截断。不裁的话一千条就是几十兆，
# 点一下「归档」要等半秒。真到那一步再拆文件，现在不值得。

_MAX_LINES = 40
_MAX_STEPS = 20


def _clip(text: Any, n: int) -> str:
    """按**字符**截，不按字节。中文一个字算一个，截出来不会切坏半个字。"""
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def snapshot_check(check: Any) -> dict:
    """一次批改的快照：逐行写了什么、判成什么、为什么。

    ``counter``（反例）留着——它是"凭什么说这步不对"的唯一证据，
    回头再看时最想找的就是它。
    """
    lines = []
    for s in getattr(check, "steps", []) or []:
        kind = getattr(s, "kind", "") or ""
        verdict = getattr(s, "verdict", "") or ""
        if kind == "note":
            continue
        lines.append({
            "i": int(getattr(s, "idx", 0) or 0),
            "t": _clip(getattr(s, "latex", "") or getattr(s, "raw", ""), 90),
            "v": verdict,
            "k": kind,
            "r": getattr(s, "role", "") or "",
            "why": _clip(getattr(s, "reason", ""), 70),
            "c": _clip(getattr(s, "counter", ""), 70),
        })
    return {
        "headline": _clip(getattr(check, "headline", ""), 180),
        "advice": _clip(getattr(check, "advice", ""), 400),
        "expected": [_clip(x, 60) for x in (getattr(check, "expected", []) or [])[:6]],
        "got": [_clip(x, 60) for x in (getattr(check, "got", []) or [])[:6]],
        "lines": lines[:_MAX_LINES],
        "n_more": max(0, len(lines) - _MAX_LINES),
    }


def snapshot_answer(ans: Any) -> dict:
    """一次搜题的快照：答案、解集、推导步骤。"""
    topic = getattr(ans, "topic", None)
    steps = []
    for st in (getattr(ans, "steps", []) or [])[:_MAX_STEPS]:
        if isinstance(st, dict):
            steps.append({
                "t": _clip(st.get("title", ""), 80),
                "x": _clip(st.get("latex", ""), 160),
            })
        else:
            steps.append({"t": "", "x": _clip(st, 160)})
    return {
        "answer": _clip(getattr(ans, "answer_latex", ""), 200),
        "solutions": [_clip(x, 60) for x in (getattr(ans, "solutions", []) or [])[:8]],
        "topic": _clip(getattr(topic, "label", ""), 40),
        "steps": steps,
        "branches": [_clip(x, 120) for x in (getattr(ans, "branches", []) or [])[:6]],
    }


def snapshot_solved(solved: Any) -> dict:
    """完整版搜题链路的快照。和 ``snapshot_answer`` 是同一种形状，只是源对象不同：

    学生版走 ``equations.Answer``，完整版走 ``SolveResult``（通用求解，不只方程）。
    详情面板只认形状，所以两者都摊成 ``{answer, solutions, steps}``。
    """
    steps = []
    for st in (getattr(solved, "steps", []) or [])[:_MAX_STEPS]:
        if isinstance(st, dict):
            steps.append({
                "t": _clip(st.get("title", ""), 80),
                "x": _clip(st.get("latex", ""), 160),
            })
        else:
            steps.append({
                "t": _clip(getattr(st, "title", ""), 80),
                "x": _clip(getattr(st, "latex", ""), 160),
            })
    return {
        "answer": "",
        "solutions": [_clip(x, 60) for x in (getattr(solved, "solutions", []) or [])[:8]],
        "topic": _clip(getattr(solved, "kind", ""), 40),
        "steps": steps,
    }



def _why_unclear(step: Any) -> str:
    """把内部的 verdict 翻译成"为什么这步需要你看一眼"。"""
    if getattr(step, "role", "") == "problem_misread":
        return "题面那行读得不对，整页结论不可靠"
    if getattr(step, "kind", "") == "unparsed":
        return "这一行读不出完整式子"
    reason = (getattr(step, "reason", "") or "").strip()
    if reason:
        return reason[:60]
    if getattr(step, "verdict", "") == "suspect":
        return "系统没把握，判定被降级"
    return "这一步没判出来"


def pending_from_check(check: Any) -> list[dict]:
    """从一次批改里抽出「系统自己没把握」的步骤。

    只收 unknown / suspect / 题面读错这三类。``wrong`` 不收——那是**确定的结论**，
    已经进错题本了，再塞进待检查区只会让用户以为"系统在怀疑自己"。
    """
    out: list[dict] = []
    for step in getattr(check, "steps", []) or []:
        verdict = getattr(step, "verdict", "")
        misread = getattr(step, "role", "") == "problem_misread"
        if not (verdict in ("unknown", "suspect") or misread):
            continue
        if getattr(step, "kind", "") == "note":
            continue
        out.append({
            "i": int(getattr(step, "idx", 0) or 0),
            "where": f"第 {int(getattr(step, 'idx', 0) or 0)} 步",
            "text": (getattr(step, "raw", "") or getattr(step, "latex", "") or "").strip()[:80],
            "why": _why_unclear(step),
            "by": "system",
            "done": False,
        })
    return out


def record_grade(check: Any, *, source: str = "text", image: str = "") -> dict:
    """一次批改落一条。pending 和 detail 都从 check 里自动抽，调用方不用自己判断。"""
    return record(
        "grade",
        getattr(check, "problem", ""),
        source=source,
        image=image or getattr(check, "problem_image", ""),
        verdict=getattr(check, "grade", ""),
        error_type=getattr(check, "error_type", ""),
        headline=getattr(check, "headline", ""),
        n_steps=len(getattr(check, "steps", []) or []),
        pending=pending_from_check(check),
        detail=snapshot_check(check),
    )


def record_search(
    problem: str, *, source: str = "text", image: str = "", detail: dict | None = None
) -> dict:
    """一次搜题落一条。``detail`` 传 ``snapshot_answer(ans)``，没有就留空。"""
    return record("search", problem, source=source, image=image, detail=detail)


def add_pending(problem: str, text: str, why: str = "") -> dict:
    """手动标一步要复查。独立成条——纸质本子上的题也能记。

    ``text`` 是"哪一步"，人话写的（"第 3 行"/"判别式那步"），直接当位置显示。
    没有它就说"整题"：用户没指明哪一步时，要复查的确实是整道题。
    """
    where = (text or "").strip()[:24] or "整题"
    return record(
        "check",
        problem,
        pending=[{
            "i": 0,
            "where": where,
            "text": "",
            "why": (why or "").strip()[:60] or "你自己标的，回头再看",
            "by": "manual",
            "done": False,
        }],
    )


def mark_done(keys: list[str]) -> int:
    """把待检查项标为已复查。``keys`` 形如 ``h1a2b3c4d:3``（记录 id + 步号）。"""
    want = {k for k in keys if k}
    if not want:
        return 0
    recs = _read_raw()
    hit = 0
    for rec in recs:
        for item in rec.get("pending") or []:
            key = f"{rec['id']}:{item.get('i', 0)}"
            if key in want and not item.get("done"):
                item["done"] = True
                hit += 1
    if hit:
        _write_raw(recs)
    return hit


def reopen(keys: list[str]) -> int:
    """标错了能撤回——一个只会单向推进的状态机，用户不敢点。"""
    want = {k for k in keys if k}
    if not want:
        return 0
    recs = _read_raw()
    hit = 0
    for rec in recs:
        for item in rec.get("pending") or []:
            if f"{rec['id']}:{item.get('i', 0)}" in want and item.get("done"):
                item["done"] = False
                hit += 1
    if hit:
        _write_raw(recs)
    return hit


# --------------------------------------------------------------------------- 读

def _clock(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%m-%d %H:%M")
    except ValueError:
        return (ts or "")[:16]


def _short(text: str, n: int = 34) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _verdict_word(rec: dict) -> str:
    kind = rec.get("kind", "")
    if kind == "search":
        return "看过"
    if kind == "check":
        return "待查"
    return VERDICTS.get(rec.get("verdict", ""), rec.get("verdict", "") or "—")


def _newest_first() -> list[dict]:
    """按时间倒序。同一秒内，后写进来的排前面。

    ``ts`` 只到秒，连着做两次操作会撞在同一秒上。只按 ``ts`` 排序是**稳定排序**，
    撞秒时保持文件原序，等于"先做的排在上面"——和"新的在上"正好反了。
    所以把文件里的位置也当第二排序键，并且一起倒过来。
    """
    recs = _read_raw()
    order = sorted(
        range(len(recs)),
        key=lambda i: (recs[i].get("ts", ""), i),
        reverse=True,
    )
    return [recs[i] for i in order]


def listing(limit: int = 200) -> tuple[list[list], list[str]]:
    """区一 + 每行对应的记录 id。

    表格只能传二维数组，点行回来的是行号，**行号和记录的对应关系必须由这里给**。
    让 ``rows()`` 和它各排一次序是错的：两边只要有一处过滤条件不同，
    点第 3 行看到的就是第 4 条。所以只保留这一份实现。
    """
    recs = _newest_first()
    out, ids = [], []
    for rec in recs[:limit]:
        pend = [p for p in (rec.get("pending") or []) if not p.get("done")]
        note = rec.get("error_type") or ""
        if pend:
            note = f"{note}／{len(pend)} 步待查" if note else f"{len(pend)} 步待查"
        out.append([
            _clock(rec.get("ts", "")),
            KINDS.get(rec.get("kind", ""), rec.get("kind", "")),
            _short(rec.get("problem", "")),
            _verdict_word(rec),
            note or rec.get("headline", "")[:30] or "—",
        ])
        ids.append(rec.get("id", ""))
    return out, ids


def rows(limit: int = 200) -> list[list]:
    """区一：全部事件，新的在前。"""
    return listing(limit)[0]


def by_id(key: str) -> dict | None:
    """按 id 取一条完整记录。找不到返回 None（文件被人手工删过行）。"""
    for rec in _read_raw():
        if rec.get("id") == key:
            return rec
    return None


def fill_missing_detail(rec: dict | None) -> dict | None:
    """给没有快照的旧记录补一份明细，补得起就写回文件。

    **搜题补得起**：题面本身就存着，``equations.answer`` 按同一道题再算一遍就行。
    **批改补不了**：当时逐行怎么判的过程没落盘，"重算"没有输入——不能凭空造一份。

    补出来的明细带 ``how="recomputed"``，界面据此说明它**不是当时那份**：
    库里的代码可能已经改过，答案和当时未必完全一样。写回文件是为了只算一次，
    不然每点一次都要重解一道题。
    """
    if not rec or rec.get("detail") or rec.get("kind") != "search":
        return rec

    problem = (rec.get("problem") or "").strip()
    detail: dict = {"how": "recomputed"}
    if problem:
        try:
            # 只走"解析 → 求解"这两步，不走 pipeline.run：
            # 后者会调讲解模型（要花钱、要等网络），还会往 runs/ 里再写一份运行记录。
            # 补一条旧记录不该有这些副作用，答案和步骤本地就能算。
            from .latex import parse
            from .solve import solve

            pe = parse(problem)
            if not pe.ok:
                detail["failed"] = True
                detail["why"] = f"题目本身解析不了：{pe.error}"[:120]
            else:
                sr = solve(pe)
                detail.update(snapshot_solved(sr))
                if not sr.ok:
                    detail["why"] = (getattr(sr, "error", "") or "求解没结果")[:120]
        except Exception as exc:  # noqa: BLE001
            # 解析器抛错也算"没补出来"，别让一条旧记录把整个页面点崩。
            detail["failed"] = True
            detail["why"] = f"{type(exc).__name__}: {exc}"[:120]
    else:
        detail["failed"] = True
        detail["why"] = "这条记录没存题目，没法重算"
    if not (detail.get("answer") or detail.get("solutions")):
        detail["failed"] = True

    out = dict(rec, detail=detail)
    try:
        recs = _read_raw()
        for i, old in enumerate(recs):
            if old.get("id") == rec.get("id"):
                recs[i] = out
                break
        else:
            return out  # 文件里已经没有这条了（被手工删行），不写回
        _write_raw(recs)
    except OSError:
        pass  # 写不进去也照样把算出来的显示出来，只是下次还得再算一遍
    return out


def pick_md(index: Any, limit: int = 200) -> str:
    """表格第 ``index`` 行 → 那一条的详情。

    行号和 id 的对应关系只在 ``listing()`` 里存在，所以这个映射也只写一份：
    两个界面各抄一遍，迟早有一边用了不同的 limit。
    """
    if isinstance(index, (list, tuple)):
        index = index[0] if index else None
    _, ids = listing(limit=limit)
    try:
        key = ids[int(index)]
    except (TypeError, ValueError, IndexError):
        return "### 没读到这一行\n\n刷新一下再点。"
    return detail_md(fill_missing_detail(by_id(key)))


def pending_rows(include_done: bool = False) -> list[list]:
    """区二：把 pending 摊平。默认只看还没复查的。"""
    out = []
    for rec in _newest_first():
        for item in rec.get("pending") or []:
            done = bool(item.get("done"))
            if done and not include_done:
                continue
            out.append([
                f"{rec['id']}:{item.get('i', 0)}",           # 第 0 列是隐藏的 key
                _clock(rec.get("ts", "")),
                _short(rec.get("problem", ""), 24),
                # 老数据没有 where 字段：系统标的按步号补，手动的按整题算
                item.get("where") or (f"第 {item['i']} 步" if item.get("i") else "整题"),
                _short(item.get("text", ""), 26) or "—",
                item.get("why", "") or "—",
                "系统" if item.get("by") == "system" else "我",
                "已复查" if done else "待查",
            ])
    return out


PENDING_HEADERS = ["key", "时间", "题目", "位置", "这一步", "为什么", "谁标的", "状态"]
LIST_HEADERS = ["时间", "动作", "题目", "结论", "备注"]


def summary_md(hint: str = "") -> str:
    """摘要。``hint`` 是空态时告诉用户"去哪做一次"，措辞要跟着页面叫法走：

    `app.py` 那边叫「搜题 · 求解」，学生版叫「对答案」——同一份数据，
    两个页面的按钮名不一样，硬写死一句会让另一边的人找不着。
    """
    recs = _read_raw()
    if not recs:
        where = hint or "「搜题 · 求解」／「批改过程」"
        return (
            "### 归档还是空的\n\n"
            f"在{where}里做一次，这里就会按时间记下来。\n\n"
            "两类东西分开放：**题目记录**是流水（我什么时候做了什么），"
            "**待检查步骤**是欠账（哪几步还等我回头看）。"
        )
    todo = sum(1 for r in recs for p in (r.get("pending") or []) if not p.get("done"))
    done = sum(1 for r in recs for p in (r.get("pending") or []) if p.get("done"))
    search = sum(1 for r in recs if r.get("kind") == "search")
    grade = sum(1 for r in recs if r.get("kind") == "grade")
    first, last = recs[0].get("ts", ""), recs[-1].get("ts", "")
    span = f"{_clock(first)} ~ {_clock(last)}" if first and last else "—"

    lines = [
        f"### 共 {len(recs)} 条（搜题 {search} · 批改 {grade}），最早到最近：{span}",
        "",
    ]
    if todo:
        lines.append(f"还有 **{todo}** 步待检查" + (f"，已复查 {done} 步。" if done else "。"))
    else:
        lines.append("待检查都清空了。" + (f"（复查过 {done} 步）" if done else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------- 详情

# 行判定翻成人话。和 VERDICTS 分开：那个是"整份的结论"，这个是"单行的命运"。
LINE_WORDS = {
    "ok": "站得住",
    "partial": "漏解",
    "wrong": "不成立",
    "unknown": "没判定",
    "note": "说明",
}

KIND_WORDS = {
    "problem": "题面",
    "assign": "答案",
    "unparsed": "没读出",
    "prose": "文字",
}

# 求解器内部的 kind 是英文的（equation / system / expression），
# 详情页是给人读的，不能直接把 "equation" 端上去。
TOPIC_WORDS = {
    "equation": "方程",
    "system": "方程组",
    "expression": "代数式",
    "inequality": "不等式",
    "param": "含参方程",
}


def _tex(text: Any) -> str:
    """能当公式渲染的包上 `$`，带中文的按原文放着。

    题面通常是 `2x+y=4` 这种纯式子，包起来好看；但拍照批改时 `problem`
    也可能是"解这个方程"一整句话，包进 `$` 会变成一串斜体乱码。

    `\\text{或}` 这种是 LaTeX **内部**的中文，不能当"这句话是中文"的证据——
    先把它剔掉再判断，否则 `x=2 \\quad\\text{或}\\quad x=3` 会被当成中文裸着显示。
    """
    s = str(text or "").strip()
    if not s:
        return "—"
    if s.endswith("…"):
        # 被快照裁剪过的串不能包 `$`：括号已经缺了一个，MathJax 会整段报错，
        # 而不是像别处那样降级成纯文本。宁可不排公式，也要让它读得下去。
        return s
    bare = re.sub(r"\\text\{[^}]*\}", "", s)
    return s if re.search(r"[\u4e00-\u9fff]", bare) else f"${s}$"


def _cell(text: Any) -> str:
    """表格单元格：`|` 不转义会把一行拆成两列。"""
    return str(text or "—").replace("|", "\\|") or "—"


def detail_md(rec: dict | None) -> str:
    """把一条记录摊开成能读的一页。

    返回 Markdown 而不是 HTML：两个界面各自有排版，但"一行一步、左边数字"
    这种结构用 Markdown 表格写一次就够，两边 inherited 同一套样式。
    """
    if not rec:
        return "### 这条记录找不到了\n\n归档文件被人改过，或者那条已经被删掉。"

    when = _clock(rec.get("ts", ""))
    kind = KINDS.get(rec.get("kind", ""), rec.get("kind", ""))
    problem = rec.get("problem", "")
    head = [f"### {when} · {kind}", "", f"**{_tex(problem)}**", ""]
    detail = rec.get("detail") or {}

    if not detail:
        # 走到这里说明绕过了 ``pick_md``（它总会先试着补一份）。所以说的是
        # "还能做什么"，而不是"就这样了"。
        tail = (
            "> 它是加「明细」之前存下的，当时只留了结论。"
            "搜题的答案可以按题目重算，点表格那一行就能看到。"
            if rec.get("kind") == "search"
            else "> 它是加「明细」之前存下的，当时只留了结论。"
            "批改的过程当时没落盘，补不回来；以后新做的记录能点开看逐行怎么判的。"
        )
        head += [
            f"这条记录的结论是 **{_verdict_word(rec)}**"
            + (f"——{rec.get('headline')}" if rec.get("headline") else "。"),
            "",
            tail,
            "",
        ]
        # 批改的逐行判定补不回来，但"哪几步没认准"当时就单独存了 pending，
        # 这部分不重算也能列出来，比一句空话有用。
        pend = rec.get("pending") or []
        if pend:
            head += ["**当时没认准、等你看的步骤：**", ""]
            for item in pend[:12]:
                where = item.get("where") or (f"第 {item.get('i')} 步" if item.get("i") else "整题")
                mark = "已复查" if item.get("done") else "待查"
                head.append(f"- {where}：{_tex(item.get('text', ''))}（{mark}）")
            if len(pend) > 12:
                head.append(f"- 还有 {len(pend) - 12} 步，在下面「还等你回头看」那一区。")
            head.append("")
        return "\n".join(head)

    recomputed = detail.get("how") == "recomputed" and not detail.get("failed")
    if recomputed:
        head += ["> 当时只留了结论，没有明细；下面是按这道题重算的，和当时那次可能有出入。", ""]

    if rec.get("kind") == "search":
        head += _search_body(detail)
    else:
        head += _grade_body(detail)
    if rec.get("source") == "photo":
        # 拍照进来的记录，用户点开第一反应是"让我再瞅一眼那张卷子"。
        # 图是 Gradio 的临时文件，重启就没了，与其让人找，不如直说。
        head += [
            "> 原图是当时上传的临时文件，已经清掉了。题目是当时识别出来的。"
            if not recomputed
            else "> 原图已经清掉了。题目是当时识别出来的，上面的答案和步骤是刚重算的。",
            "",
        ]
    return "\n".join(head)


def _search_body(d: dict) -> list[str]:
    out: list[str] = []
    if d.get("failed"):
        out += ["按这条的题目重算了一次，没得到答案——题目写法可能不完整，或者当时就没解出来。", ""]
        if d.get("why"):
            out += [f"> {d['why']}", ""]
        return out

    answer = (d.get("answer") or "").strip()
    sols = d.get("solutions") or []
    # 快照会按字符裁剪，被裁过的串末尾带 `…`。这种半截 LaTeX 既不能包 `$`
    # （MathJax 会报缺括号），摆出来也只是 `\frac{- b - \sqrt{…` 一堆乱码，
    # 所以干脆指到下面「怎么来的」——那里是一条一条完整的。
    cut = any(str(x).rstrip().endswith("…") for x in ([answer] if answer else []) + list(sols))
    if answer and not cut:
        out += [f"答案：**{_tex(answer)}**", ""]
    elif len(sols) > 2:
        # 含参方程动辄六种情形，join 成一行会横着排出屏幕外
        out += ["答案：", ""] + [f"- {_tex(x)}" for x in sols] + [""]
    elif sols and not cut:
        out += [f"答案：**{'，'.join(_tex(x) for x in sols)}**", ""]
    elif cut:
        out += ["答案太长，一行放不下（含参要按情形分开）；下面「怎么来的」里逐条列了。", ""]
    else:
        out += ["这道题没解出来，没留下答案。", ""]

    if d.get("topic"):
        topic = d["topic"]
        out += [f"题型：{TOPIC_WORDS.get(topic, topic)}", ""]
    if d.get("branches"):
        out += ["**含参，分情况：**", ""]
        out += [f"- {_tex(b)}" for b in d["branches"]]
        out.append("")

    steps = d.get("steps") or []
    if steps:
        out += ["**怎么来的：**", ""]
        for n, st in enumerate(steps, 1):
            title = (st.get("t") or "").strip()
            body = _tex(st.get("x", ""))
            out.append(f"{n}. {title + ' —— ' if title else ''}{body}")
        out.append("")
    return out


def _grade_body(d: dict) -> list[str]:
    out: list[str] = []
    if d.get("headline"):
        out += [f"结论：**{d['headline']}**", ""]

    lines = d.get("lines") or []
    if lines:
        out += ["| 行 | 写的是 | 判定 |", "|:--:|---|---|"]
        for ln in lines:
            word = LINE_WORDS.get(ln.get("v", ""), ln.get("v") or "—")
            # 题面那一行 verdict 是 note，光写"说明"看不出它是题目本身
            if ln.get("r") == "problem" or ln.get("k") == "problem":
                word = "题面"
            elif ln.get("k") in KIND_WORDS and ln.get("k") != "equation":
                word = f"{KIND_WORDS[ln['k']]} · {word}"
            out.append(f"| {ln.get('i', '')} | {_cell(_tex(ln.get('t', '')))} | {_cell(word)} |")
        out.append("")
        if d.get("n_more"):
            out += [f"> 后面还有 {d['n_more']} 行没列出来（快照封顶 40 行）。", ""]

    # 只挑有问题的行说原因：全列出来会把这一页撑成一张长表，而 ok 的行没有话要说。
    notes = []
    for ln in lines:
        if ln.get("v") in ("ok", "note") or not (ln.get("why") or ln.get("c")):
            continue
        pos = f"第 {ln.get('i')} 行"
        why = (ln.get("why") or "").strip()
        counter = (ln.get("c") or "").strip()
        msg = why or "这一步不成立"
        if counter:
            msg += f"（反例：{_tex(counter)}）"
        notes.append(f"- **{pos}**：{msg}")
    if notes:
        out += ["**为什么：**", ""] + notes + [""]

    if d.get("expected") or d.get("got"):
        exp = "，".join(_tex(x) for x in (d.get("expected") or []))
        got = "，".join(_tex(x) for x in (d.get("got") or []))
        if exp or got:
            out += [f"应有：{exp or '—'}　实际：{got or '—'}", ""]

    if d.get("advice"):
        out += [f"> {d['advice']}", ""]
    return out
