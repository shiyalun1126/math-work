"""批改链路评测入口。

    F:/PyCharm/python.exe -m eval.run_eval            # 全跑
    F:/PyCharm/python.exe -m eval.run_eval --suite work

真值来自 eval/work_cases.jsonl 与 eval/photo_cases.jsonl，按数学规范人工标注，
**不是**从 grade() 的输出反推的。跑出来的不一致会原样列进报告，由人裁定：
是标注写错了、还是引擎有 bug。悄悄改真值去迎合实现，这套评测就没有意义了。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import metrics as M  # noqa: E402
from eval.report import render  # noqa: E402
from mathagent.photocheck import LineRead, _apply_uncertainty  # noqa: E402
from mathagent.workcheck import grade  # noqa: E402

HERE = Path(__file__).parent
WORK_CASES = HERE / "work_cases.jsonl"
PHOTO_CASES = HERE / "photo_cases.jsonl"
OUT_DIR = ROOT / "reports"

GRADES = ["pass", "fake", "partial", "wrong", "gap", "unknown", "unread"]
BAD_GRADES = ("wrong", "fake", "partial")

# 错因的"族"。首轮评测暴露出标注员写「答案错」、引擎写「代回不成立」——
# 两者指的是同一个失败模式，但标签不同。严格一致率如实报，不靠改真值抹平；
# 族级一致率用来说明归因方向对不对。定义见 docs/批改链路.md §6。
ERROR_FAMILY = {
    "伪推导": "伪推导",
    "移项/变号错": "变形", "漏乘漏除": "变形", "变形不等价": "变形",
    "增根": "变形",
    "漏解": "不全", "未写答案": "不全",
    "答案错": "答案", "代回不成立": "答案",
}


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{i} 不是合法 JSON: {exc}") from exc
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else ""


def run_work() -> tuple[list[dict], dict]:
    cases = _load(WORK_CASES)
    records: list[dict] = []
    for c in cases:
        t0 = time.time()
        truth_grade = c["grade"]
        truth_first = int(c.get("first_error", -1))
        truth_type = c.get("error_type", "") or ""
        try:
            r = grade(c["work"], problem=c.get("problem", "") or "")
            pred_grade, pred_first, pred_type = r.grade, r.first_error, r.error_type
            err = ""
        except Exception as exc:  # noqa: BLE001
            pred_grade, pred_first, pred_type = "error", -2, ""
            err = f"{type(exc).__name__}: {exc}"
        rec = {
            "id": c["id"],
            "group": c["group"],
            "note": c.get("note", ""),
            "work": c["work"],
            "truth_grade": truth_grade,
            "pred_grade": pred_grade,
            "truth_first": truth_first,
            "pred_first": pred_first,
            "truth_type": truth_type,
            "pred_type": pred_type,
            "grade_hit": int(pred_grade == truth_grade),
            "first_applicable": int(truth_first >= 0),
            "first_hit": int(truth_first >= 0 and pred_first == truth_first),
            "type_applicable": int(bool(truth_type)),
            "type_hit": int(bool(truth_type) and pred_type == truth_type),
            "family_hit": int(
                bool(truth_type)
                and ERROR_FAMILY.get(truth_type) == ERROR_FAMILY.get(pred_type)
            ),
            "false_accuse": int(truth_grade == "pass" and pred_grade in BAD_GRADES),
            "miss_applicable": int(truth_grade in BAD_GRADES),
            "let_through": int(truth_grade in BAD_GRADES and pred_grade == "pass"),
            "fake_applicable": int(truth_grade == "fake"),
            "fake_hit": int(truth_grade == "fake" and pred_grade == "fake"),
            "error": err,
            "elapsed": round(time.time() - t0, 4),
        }
        records.append(rec)

    def bits(key: str, where=None) -> list[int]:
        return [r[key] for r in (where or records) if isinstance(r[key], int)]

    def m(key: str, where=None) -> dict:
        return M.metric([r[key] for r in (where or records)])

    false_pool = [r for r in records if r["truth_grade"] == "pass"]
    miss_pool = [r for r in records if r["miss_applicable"]]
    fake_pool = [r for r in records if r["fake_applicable"]]
    first_pool = [r for r in records if r["first_applicable"]]
    type_pool = [r for r in records if r["type_applicable"]]
    fake_pred = [r for r in records if r["pred_grade"] == "fake"]

    tp = sum(r["fake_hit"] for r in records)
    fp = sum(1 for r in fake_pred if r["truth_grade"] != "fake")
    fn = sum(1 for r in fake_pool if r["pred_grade"] != "fake")

    summary = {
        "n": len(records),
        "conclusion": m("grade_hit"),
        "false_accusation": M.metric([r["false_accuse"] for r in false_pool]),
        "let_through": M.metric([r["let_through"] for r in miss_pool]),
        "first_error": M.metric([r["first_hit"] for r in first_pool]),
        "error_type": M.metric([r["type_hit"] for r in type_pool]),
        "error_family": M.metric([r["family_hit"] for r in type_pool]),
        "fake_recall": M.metric([r["fake_hit"] for r in fake_pool]),
        "fake_prf": M.prf(tp, fp, fn),
        "by_group": M.by_group(records, "grade_hit"),
        "confusion": M.confusion(
            [(r["truth_grade"], r["pred_grade"]) for r in records], GRADES
        ),
        "mismatches": [r for r in records if not r["grade_hit"]],
        "type_mismatches": [
            {"id": r["id"], "truth": r["truth_type"], "pred": r["pred_type"]}
            for r in type_pool if not r["type_hit"]
        ],
        "first_mismatches": [
            {"id": r["id"], "truth": r["truth_first"], "pred": r["pred_first"]}
            for r in first_pool if not r["first_hit"]
        ],
        "elapsed_total": round(sum(r["elapsed"] for r in records), 3),
    }
    return records, summary


def _line_reads(work: str, sure: list[bool]) -> list[LineRead]:
    texts = [x for x in work.split("\n") if x.strip()]
    out = []
    for i, text in enumerate(texts, 1):
        ok = sure[i - 1] if i - 1 < len(sure) else True
        out.append(LineRead(
            index=i,
            text=text,
            latex=text,
            verdict="ok" if ok else "review",
            confidence=1.0 if ok else 0.4,
            sources=["a", "b"] if ok else ["a"],
        ))
    return out


def run_photo() -> tuple[list[dict], dict]:
    cases = _load(PHOTO_CASES)
    records: list[dict] = []
    for c in cases:
        work = c["work"]
        before = grade(work)
        before_map = {s.idx: (s.verdict, s.counter) for s in before.steps}
        reads = _line_reads(work, c["sure"])
        _apply_uncertainty(before, reads)
        after_map = {s.idx: (s.verdict, s.counter) for s in before.steps}

        withdrawn = sorted(
            idx for idx, (v0, _) in before_map.items()
            if v0 in ("ok", "wrong", "partial")
            and after_map[idx][0] in ("suspect", "unknown")
        )
        expected = sorted(c.get("expect_withdrawn", []))
        tp = len(set(withdrawn) & set(expected))
        fp = len(set(withdrawn) - set(expected))
        fn = len(set(expected) - set(withdrawn))

        # 认不准的行上，不能残留任何断言（夸人和冤枉人都是装作确定）
        unsure_idx = {i for i, ok in enumerate(c["sure"], 1) if not ok}
        leftover = [
            i for i in unsure_idx
            if after_map.get(i, ("", ""))[0] in ("ok", "wrong", "partial")
        ]
        # 撤回的断言必须把反例一起作废，否则表里会留一个"证据"
        stale_counter = [
            i for i in withdrawn
            if before_map[i][1] and after_map[i][1]
        ]
        rec = {
            "id": c["id"],
            "note": c.get("note", ""),
            "expect_grade": c.get("expect_grade", ""),
            "pred_grade": before.grade,
            "grade_hit": int(before.grade == c.get("expect_grade", "")),
            "expect_withdrawn": expected,
            "withdrawn": withdrawn,
            "tp": tp, "fp": fp, "fn": fn,
            "leftover": leftover,
            "stale_counter": stale_counter,
            "no_false_certainty": int(not leftover),
            "counter_cleared": int(not stale_counter),
            "error_type": before.error_type,
        }
        records.append(rec)

    if not records:
        return [], {}
    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    summary = {
        "n": len(records),
        "withdraw_prf": M.prf(tp, fp, fn),
        "no_false_certainty": M.metric([r["no_false_certainty"] for r in records]),
        "counter_cleared": M.metric([r["counter_cleared"] for r in records]),
        "grade_guard": M.metric([r["grade_hit"] for r in records]),
        "mismatches": [r for r in records if r["fp"] or r["fn"] or r["leftover"] or not r["grade_hit"]],
    }
    return records, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["work", "photo", "all"], default="all")
    ap.add_argument("--seed", type=int, default=M.DEFAULT_SEED)
    args = ap.parse_args()

    M.DEFAULT_SEED = args.seed
    OUT_DIR.mkdir(exist_ok=True)
    work_records, work_sum = ([], {}) if args.suite == "photo" else run_work()
    photo_records, photo_sum = ([], {}) if args.suite == "work" else run_photo()

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "n_boot": M.N_BOOT,
        "cases_sha": {"work": _sha(WORK_CASES), "photo": _sha(PHOTO_CASES)},
        "work": {"records": work_records, "summary": work_sum},
        "photo": {"records": photo_records, "summary": photo_sum},
    }
    (OUT_DIR / "eval_work.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = render(payload)
    (OUT_DIR / "eval_work.md").write_text(md, encoding="utf-8")

    if work_sum:
        c = work_sum["conclusion"]
        fa = work_sum["false_accusation"]
        fr = work_sum["fake_recall"]
        print(f"结论准确率 {M.pct(c['value'], c['ci'])} ({c['hit']}/{c['n']})")
        print(f"冤枉率     {M.pct(fa['value'], fa['ci'])} ({fa['hit']}/{fa['n']})")
        print(f"伪推导召回 {M.pct(fr['value'], fr['ci'])} ({fr['hit']}/{fr['n']})")
        print(f"首错定位   {M.pct(work_sum['first_error']['value'])}")
        print(f"错因严格   {M.pct(work_sum['error_type']['value'])}")
        print(f"错因族级   {M.pct(work_sum['error_family']['value'])}")
        print(f"不一致     {len(work_sum['mismatches'])} 条")
    if photo_sum:
        p = photo_sum["withdraw_prf"]
        print(f"撤回 P/R   {M.pct(p['precision'])} / {M.pct(p['recall'])}")
        print(f"无假确定   {M.pct(photo_sum['no_false_certainty']['value'])}")
    print(f"报告       {OUT_DIR / 'eval_work.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
