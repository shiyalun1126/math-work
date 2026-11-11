"""评测层自身的测试 + 评测发现的 bug 的回归护栏。

前两轮评测抓到过两个单元测试漏掉的真 bug，这里把它们钉死：
  · 「解得 x=2 或 x=3」只认出 x=3，导致把全对的过程冤枉成漏解
  · 「x^2=x」被当成赋值语句，整道题的题面作废
"""

from __future__ import annotations

import json
from pathlib import Path

from eval import metrics as M
from eval.run_eval import GRADES, WORK_CASES, PHOTO_CASES, run_work, run_photo
from mathagent.workcheck import grade


def test_ratio_zero_denominator_is_none():
    # 「没有这类样本」和「一个都没抓到」不能都写成 0
    assert M.ratio(0, 0) is None
    assert M.ratio(3, 4) == 0.75


def test_bootstrap_ci_is_reproducible():
    bits = [1] * 8 + [0] * 2
    assert M.bootstrap_ci(bits) == M.bootstrap_ci(bits)
    lo, hi = M.bootstrap_ci(bits)
    assert 0 <= lo <= 0.8 <= hi <= 1


def test_seed_is_read_at_call_time(monkeypatch):
    """`run_eval --seed` 必须真的影响重采样，不能只在报告里记个数。"""
    bits = [1] * 63 + [0] * 37
    monkeypatch.setattr(M, "DEFAULT_SEED", 7)
    a = M.bootstrap_ci(bits)
    monkeypatch.setattr(M, "DEFAULT_SEED", 8)
    b = M.bootstrap_ci(bits)
    assert a != b, "模块级种子改了却没生效，说明又被默认参数提前绑定了"


def test_pct_and_prf_and_confusion():
    assert M.pct(0.625, (0.513, 0.731)) == "62.5% [51.3, 73.1]"
    assert M.pct(None) == "—"
    assert M.prf(3, 1, 1)["f1"] == 0.75
    cm = M.confusion([("pass", "wrong"), ("pass", "pass")], ["pass", "wrong"])
    assert cm["rows"]["pass"] == [1, 1]


def test_by_group_splits_rather_than_averages():
    recs = [
        {"group": "易", "grade_hit": 1},
        {"group": "易", "grade_hit": 1},
        {"group": "难", "grade_hit": 0},
    ]
    g = M.by_group(recs, "grade_hit")
    assert g["易"]["value"] == 1.0
    assert g["难"]["value"] == 0.0


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_work_truth_file_is_wellformed():
    cases = _read_jsonl(WORK_CASES)
    assert len(cases) >= 40
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "ID 必须唯一，否则报告对不上条"
    for c in cases:
        assert c["grade"] in GRADES, c["id"]
        assert isinstance(c["first_error"], int), c["id"]
        # 标了首错 ⇒ 一定判有错；反过来不成立：答案多写一个根也是 wrong，
        # 但没有任何一步推导「不成立」，不该硬凑首错行号（D04）。
        if c["first_error"] >= 0:
            assert c["grade"] in ("fake", "wrong"), c["id"]
        if c["grade"] == "fake":
            assert c["first_error"] >= 0, f"{c['id']}: 伪推导必然有一处不成立"
        assert c["group"] and c["note"], c["id"]


def test_photo_truth_file_is_wellformed():
    cases = _read_jsonl(PHOTO_CASES)
    assert len(cases) >= 9
    for c in cases:
        lines = [x for x in c["work"].split("\n") if x.strip()]
        assert len(c["sure"]) == len(lines), f"{c['id']}: sure 长度要和行数对齐"
        assert not any(i > len(lines) for i in c["expect_withdrawn"]), c["id"]


def test_eval_summary_guardrails():
    """护栏：不许重新冤枉人，不许漏掉伪推导。数字变了要回来修，不是改测试。"""
    records, s = run_work()
    assert len(records) == len(_read_jsonl(WORK_CASES))
    assert all(r["pred_grade"] in GRADES or r["pred_grade"] == "error" for r in records)
    assert not any(r["error"] for r in records), "评测里不允许有未捕获异常"
    assert s["false_accusation"]["hit"] == 0, "把全对的过程判错是最不能接受的回归"
    assert s["fake_recall"]["value"] == 1.0, "伪推导是项目唯一的差异化能力，必须全抓"
    assert s["error_family"]["value"] >= s["error_type"]["value"]


def test_photo_withdraw_never_over_claims():
    records, s = run_photo()
    assert records
    assert s["withdraw_prf"]["precision"] == 1.0, "撤回了不该撤的断言也是装作确定"
    assert s["no_false_certainty"]["value"] == 1.0
    assert s["counter_cleared"]["value"] == 1.0, "撤回断言必须带走反例"


def test_bug_chinese_prefix_keeps_both_roots():
    """F02：说明行里的「解得」不能吃掉第二个根。"""
    r = grade("x^2-5x+6=0\n(x-2)(x-3)=0\n解得 x=2 或 x=3",
              problem="x^2-5x+6=0")
    assert r.grade == "pass", r.headline
    assert len(r.got) == 2, r.got


def test_bug_self_reference_is_equation_not_assign():
    """C04：`x^2=x` 是方程，不是「x = x^2」的赋值。"""
    r = grade("x^2=x\nx=1", problem="x^2=x")
    assert r.grade == "partial", r.headline
    assert r.error_type == "漏解", r.error_type


def test_inferred_answer_does_not_claim_student_wrote_it():
    """C03：答案是系统从最后一行解出来的，不能说「答案也对」。"""
    r = grade("x^2-5x+6=0\n(x-2)(x-3)=0", problem="x^2-5x+6=0")
    assert r.grade == "pass"
    assert "答案没单独写" in r.headline
    assert any("按最后一个等式" in n for n in r.notes)
