"""照片评测入口的「可复算」与「未执行」边界（D08）。

原图涉及学生隐私，不入库；但评审要能对一份入库的脱敏快照复算指标。
另外这脚本以前在"一张都没测"时 return 0，日志看着像评测通过——这里钉死退出码。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import run_photo_eval as rpe  # noqa: E402

HEADER = ("id,group,priority,truth_grade,truth_first,truth_type,baseline,"
          "problem,lines\n")


def write_shotlist(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """(id, 真值档, 基线档)；W20 这类 na 页不参与指标。"""
    body = "".join(f"{i},正确过程,P0,{t},-1,,{b},x+1=2,1\n" for i, t, b in rows)
    p = tmp_path / "shotlist.csv"
    p.write_text(HEADER + body, encoding="utf-8")
    return p


def rec(wid: str, truth: str, pred: str, baseline: str = "", unsure: int = 0) -> dict:
    return {
        "id": wid, "group": "正确过程", "priority": "P0",
        "image": f"data/inbox/{wid}.jpg", "truth_grade": truth,
        "truth_first": -1, "truth_type": "", "baseline": baseline or truth,
        "pred_grade": pred, "pred_first": -1, "pred_type": "",
        "n_lines": 3, "n_unsure": unsure, "n_unread": unsure,
        "text": "x+1=2\nx=1", "lines": [], "false_certainty": False,
        "notes": [], "elapsed": 0.1,
    }


@pytest.fixture()
def env(tmp_path, monkeypatch):
    shot = write_shotlist(tmp_path, [("W01", "pass", "pass"), ("W05", "fake", "fake")])
    out = tmp_path / "reports"
    monkeypatch.setattr(rpe, "SHOTLIST", shot)
    monkeypatch.setattr(rpe, "INBOX", tmp_path / "inbox")   # 一张照片都没有
    monkeypatch.setattr(rpe, "RESULTS", tmp_path / "photo_results.jsonl")
    monkeypatch.setattr(rpe, "OUT_DIR", out)
    return tmp_path


def test_no_snapshot_is_not_run_not_success(env, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_photo_eval.py"])
    assert rpe.main() == 2
    out = capsys.readouterr().out
    assert "未执行" in out
    # 不写报告：一份"未执行"不该留下一份看起来像结果的文件
    assert not (env / "reports" / "eval_photo_real.md").exists()


def test_missing_results_path_is_not_run(env, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "run_photo_eval.py", "--results", str(env / "nope.jsonl")])
    assert rpe.main() == 2
    assert "未执行" in capsys.readouterr().out


def test_explicit_jsonl_snapshot_recomputes_report(env, monkeypatch, capsys):
    snap = env / "golden.jsonl"
    snap.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                              for r in [rec("W01", "pass", "pass"),
                                        rec("W05", "fake", "fake")]) + "\n",
                    encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_photo_eval.py", "--results", str(snap)])
    assert rpe.main() == 0
    md = (env / "reports" / "eval_photo_real.md").read_text(encoding="utf-8")
    assert "结果快照" in md and "golden.jsonl" in md
    payload = json.loads((env / "reports" / "eval_photo_real.json").read_text(
        encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["results_source"] == str(snap)
    assert payload["summary"]["n"] == 2
    assert payload["summary"]["conclusion"]["hit"] == 2


def test_explicit_json_payload_from_previous_run(env, monkeypatch):
    """本脚本自己写出的 .json（{summary, records}）也要能直接喂回去。"""
    prev = env / "prev.json"
    prev.write_text(json.dumps({"records": [rec("W01", "pass", "wrong")]},
                               ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "run_photo_eval.py", "--results", str(prev), "--smoke"])
    # --smoke 优先用内置假数据，不能悄悄把 --results 混进去
    assert rpe.main() == 0
    payload = json.loads((env / "reports" / "_smoke_photo_real.json").read_text(
        encoding="utf-8"))
    assert payload["smoke"] is True
    assert payload["results_source"] == "(内置冒烟数据)"
    assert all(r["text"] == "(冒烟数据)" for r in payload["records"])

    monkeypatch.setattr(sys, "argv", ["run_photo_eval.py", "--results", str(prev)])
    assert rpe.main() == 0
    payload = json.loads((env / "reports" / "eval_photo_real.json").read_text(
        encoding="utf-8"))
    assert payload["summary"]["n"] == 1
    assert payload["summary"]["conclusion"]["hit"] == 0


def test_load_results_ignores_broken_lines(env):
    p = env / "half.jsonl"
    p.write_text('{"id": "a", "truth_grade": "pass", "pred_grade": "pass"}\n'
                 "not json\n\n", encoding="utf-8")
    got = rpe.load_results(p)
    assert [r["id"] for r in got] == ["a"]
    assert rpe.load_results(env / "missing.jsonl") == []
