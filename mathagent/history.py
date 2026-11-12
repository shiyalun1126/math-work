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
    }
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


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
    """一次批改落一条。pending 自动从步骤里抽，调用方不用自己判断。"""
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
    )


def record_search(problem: str, *, source: str = "text", image: str = "") -> dict:
    return record("search", problem, source=source, image=image)


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


def rows(limit: int = 200) -> list[list]:
    """区一：全部事件，新的在前。"""
    recs = sorted(_read_raw(), key=lambda r: r.get("ts", ""), reverse=True)
    out = []
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
    return out


def pending_rows(include_done: bool = False) -> list[list]:
    """区二：把 pending 摊平。默认只看还没复查的。"""
    out = []
    for rec in sorted(_read_raw(), key=lambda r: r.get("ts", ""), reverse=True):
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
