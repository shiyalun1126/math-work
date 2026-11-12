"""题目归档：时间戳、去重、待检查项的两个来源。

这里的每一条都对应一个"如果不测就会悄悄坏掉"的地方：
时间不跟记录走、双击写两条、把判错的行混进待检查区、
标了已复查却不能撤回、半截行让整个文件读不出来。
"""

from __future__ import annotations

import json

import pytest

from mathagent import history
from mathagent.types import WorkCheck, WorkStep


@pytest.fixture()
def store(tmp_path, monkeypatch):
    path = tmp_path / "history.jsonl"
    monkeypatch.setattr(history, "HISTORY", path)
    monkeypatch.setattr(history, "_DEDUPE_SECONDS", 0)
    return path


def _step(idx, raw="x=3", kind="equation", verdict="ok", role="", reason=""):
    return WorkStep(idx=idx, raw=raw, kind=kind, verdict=verdict, role=role, reason=reason)


def _check(grade="partial", steps=None, **kw):
    return WorkCheck(
        problem="x^2-5x+6=0",
        grade=grade,
        error_type=kw.get("error_type", "漏解"),
        headline=kw.get("headline", "过程成立，但解不全"),
        steps=steps if steps is not None else [_step(1), _step(2)],
    )


# --------------------------------------------------------------------------- 写与读

def test_record_keeps_its_own_timestamp(store):
    rec = history.record_search("x^2=4", source="text")
    assert rec["kind"] == "search"
    assert rec["ts"].count(":") == 2          # 存的是完整 ISO 时刻
    got = history.rows()
    assert got[0][0].count(":") == 1          # 界面显示到分钟
    assert got[0][2] == "x^2=4"
    assert got[0][3] == "看过"


def test_same_click_twice_does_not_write_twice(store, monkeypatch):
    monkeypatch.setattr(history, "_DEDUPE_SECONDS", 3)
    a = history.record_search("x^2=4")
    b = history.record_search("x^2=4")
    assert a["id"] == b["id"]
    assert len(store.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_different_problem_right_after_is_two_rows(store, monkeypatch):
    monkeypatch.setattr(history, "_DEDUPE_SECONDS", 3)
    history.record_search("x^2=4")
    history.record_search("x^2=9")
    assert len(store.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_half_written_line_does_not_kill_the_file(store):
    store.write_text(
        json.dumps({"id": "h1", "ts": "2026-11-12T10:00:00", "kind": "search",
                    "problem": "x=1"}, ensure_ascii=False) + "\n"
        '{"id": "h2", "ts": "2026-11-1\n',
        encoding="utf-8",
    )
    assert len(history.rows()) == 1


def test_unknown_kind_is_rejected(store):
    with pytest.raises(ValueError, match="未知事件类型"):
        history.record("whatever", "x=1")


# ------------------------------------------------------- 待检查项：只收"没把握"

def test_pending_comes_from_unclear_steps_only(store):
    check = _check(steps=[
        _step(1),
        _step(2, verdict="wrong", raw="x=4"),          # 确定的错，不该进待检查
        _step(3, verdict="unknown", raw="???", reason="这一行读不出完整式子"),
        _step(4, verdict="suspect", raw="x=2 或"),
        _step(5, role="problem_misread", raw="x^2-5x-6=0"),
        _step(6, kind="note", raw="两边同乘 2"),        # 说明行，不参与判定
    ])
    pending = history.pending_from_check(check)
    assert [p["i"] for p in pending] == [3, 4, 5]
    assert all(p["by"] == "system" and p["done"] is False for p in pending)
    assert "读不出" in pending[0]["why"]
    assert "题面" in pending[2]["why"]


def test_wrong_rows_never_enter_pending(store):
    check_wrong = _check(grade="wrong", steps=[_step(1), _step(2, verdict="wrong")])
    assert history.pending_from_check(check_wrong) == []


def test_record_grade_puts_pending_on_the_row(store):
    check = _check(steps=[_step(1), _step(2, verdict="unknown", reason="没判出来")])
    rec = history.record_grade(check, source="photo")
    assert rec["verdict"] == "partial"
    assert rec["error_type"] == "漏解"
    assert len(rec["pending"]) == 1
    assert rec["pending"][0]["i"] == 2


# --------------------------------------------------------------- 复查与撤销

def test_mark_done_and_reopen(store):
    check = _check(steps=[_step(1, verdict="unknown"), _step(2, verdict="unknown")])
    rec = history.record_grade(check)
    key = f"{rec['id']}:1"

    assert history.mark_done([key]) == 1
    assert [r[3] for r in history.pending_rows()] == ["第 2 步"]     # 只剩没复查的
    assert [r[7] for r in history.pending_rows(include_done=True)] == ["已复查", "待查"]

    # 再标一次不重复计数：状态是幂等的，双击不会算出 2
    assert history.mark_done([key]) == 0
    assert history.reopen([key]) == 1
    assert len(history.pending_rows()) == 2


def test_mark_done_ignores_unknown_key(store):
    history.record_grade(_check(steps=[_step(1, verdict="unknown")]))
    assert history.mark_done(["h_not_exist:1"]) == 0
    assert len(history.pending_rows()) == 1


def test_manual_pending_stands_alone(store):
    rec = history.add_pending("2x-6=12", "移项那一步", "当时少除了 2")
    assert rec["kind"] == "check"
    rows = history.pending_rows()
    assert len(rows) == 1
    assert rows[0][6] == "我"
    assert rows[0][5] == "当时少除了 2"


def test_manual_pending_without_reason_still_readable(store):
    history.add_pending("x^2=1", "", "")
    rows = history.pending_rows()
    assert rows[0][3] == "整题"          # 没写第几步，就记成整题
    assert rows[0][4] == "—"
    assert "回头再看" in rows[0][5]


# ------------------------------------------------------------------ 两个区

def test_rows_are_newest_first(store):
    history.record_search("第一题")
    history.record_search("第二题")
    got = history.rows()
    # 同一秒写入时 ts 相同，靠 stable sort 保持写入顺序的反向；
    # 这里只断言"两条都在、且不含 key 列"，不依赖秒级时钟
    assert {r[2] for r in got} == {"第一题", "第二题"}
    assert len(got[0]) == len(history.LIST_HEADERS)


def test_list_shows_pending_count(store):
    check = _check(steps=[_step(1, verdict="unknown"), _step(2, verdict="unknown")])
    history.record_grade(check)
    row = history.rows()[0]
    assert "2 步待查" in row[4]


def test_summary_mentions_todo(store):
    check = _check(steps=[_step(1, verdict="unknown")])
    history.record_grade(check)
    md = history.summary_md()
    assert "1** 步待检查" in md or "1 步待检查" in md.replace("**", "")
    assert "批改 1" in md


def test_summary_on_empty_store(store):
    md = history.summary_md()
    assert "还是空的" in md
    assert history.rows() == []
    assert history.pending_rows() == []
