"""错题本：一条 case 一行 json，全部追加到 data/cases.jsonl。

为什么不是数据库：这个项目没有并发、没有多用户，一张 jsonl 就能跑。
真正花时间的不是存，是**归因**——同一道题错两次，和错在两个不同的地方，
是两件不同的事，前者要重学，后者要重练。所以每条 case 都带 error_type，
统计按错因分，不按题型分。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from . import config
from .types import WorkCheck

CASES: Path = config.ROOT / "data" / "cases.jsonl"

# 错因排序：越靠前越该先补。和"题型"排序不同，它回答的是"我缺哪个动作"。
SEVERITY = {
    "伪推导": 0,
    "移项/变号错": 1,
    "漏乘漏除": 2,
    "变形不等价": 3,
    "增根": 4,
    "漏解": 5,
    "代回不成立": 6,
    "答案错": 7,
    "未写答案": 8,
    "": 9,
}


def _row(rec: dict) -> dict:
    return {
        "id": rec.get("id", ""),
        "时间": (rec.get("ts") or "")[:16].replace("T", " "),
        "判定": rec.get("grade", ""),
        "错因": rec.get("error_type", "") or "—",
        "题目": rec.get("problem", ""),
        "学生答案": "，".join(rec.get("got") or []) or "—",
        "标准答案": "，".join(rec.get("expected") or []) or "—",
        "一句话": rec.get("headline", ""),
    }


def add(check: WorkCheck, problem_image: str = "") -> dict:
    """把一次批改落成一条错题记录。判 pass 的不进错题本——那不是错题。"""
    rec = {
        "id": uuid.uuid4().hex[:10],
        "ts": datetime.now().isoformat(timespec="seconds"),
        "problem": check.problem,
        "problem_ok": check.problem_ok,
        "grade": check.grade,
        "error_type": check.error_type,
        "headline": check.headline,
        "advice": check.advice,
        "first_error": check.first_error,
        "expected": list(check.expected),
        "got": list(check.got),
        "problem_image": problem_image or check.problem_image,
        "steps": [s.to_dict() for s in check.steps],
    }
    CASES.parent.mkdir(parents=True, exist_ok=True)
    with CASES.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load() -> list[dict]:
    if not CASES.exists():
        return []
    out: list[dict] = []
    for line in CASES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def rows() -> list[list]:
    """给界面表格用。按"该先补的"排序，不是按时间。"""
    recs = load()
    # 先按时间新→旧，再按错因稳定排序：同一类里最新的排前面
    recs.sort(key=lambda r: r.get("ts", ""), reverse=True)
    recs.sort(key=lambda r: SEVERITY.get(r.get("error_type", ""), 9))
    return [[c for c in _row(r).values()] for r in recs]


def headers() -> list[str]:
    return list(_row({}).keys())


def stats() -> list[tuple[str, int]]:
    """按错因计数。比"按题型"有用：题型只说明你学到哪，错因说明你缺什么。"""
    counts: dict[str, int] = {}
    for rec in load():
        key = rec.get("error_type") or "未归类"
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], SEVERITY.get(kv[0], 9)))


def same_type(problem: str) -> list[dict]:
    """找出同一道题以前错过几次。这是"复发"信号，比新错更值得看。"""
    key = (problem or "").strip()
    if not key:
        return []
    return [r for r in load() if (r.get("problem") or "").strip() == key]


def summary_md() -> str:
    recs = load()
    if not recs:
        return (
            "### 错题本还是空的\n\n"
            "在「批改过程」页签里批完一份，点「存进错题本」，这里才会有内容。\n\n"
            "存进来的不是「题目」，是**你错在哪一步**——所以统计按错因分，不按题型。"
        )
    st = stats()
    total = len(recs)
    top = st[0] if st else ("—", 0)
    lines = [f"### 共 {total} 条，最集中的错因是「{top[0]}」（{top[1]} 条）", ""]
    lines.append("| 错因 | 条数 | 占比 |")
    lines.append("|---|---|---|")
    for name, n in st:
        lines.append(f"| {name} | {n} | {n / total:.0%} |")
    return "\n".join(lines)
