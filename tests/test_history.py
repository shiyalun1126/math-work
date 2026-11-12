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


# --------------------------------------------------------------------------- 详情快照
#
# 「点开那一条」靠的是写入时存下的快照。没存的话点开只有一句结论，
# 而"当时到底是哪一行错了"恰恰是最想回看的东西。

def test_grade_records_snapshot(store):
    check = _check(steps=[
        _step(1, raw="x^2-5x+6=0", verdict="note", reason="题面"),
        _step(2, raw="(x-2)(x-3)=0"),
        _step(3, raw="x=2", kind="assign"),
    ])
    history.record_grade(check)
    rec = history._read_raw()[0]
    lines = rec["detail"]["lines"]
    assert [ln["i"] for ln in lines] == [1, 2, 3]
    assert lines[0]["why"] == "题面"
    assert rec["detail"]["headline"]


def test_search_keeps_detail(store):
    history.record_search("x^2-5x+6=0", detail={"solutions": ["x=2", "x=3"], "steps": []})
    rec = history._read_raw()[0]
    assert rec["detail"]["solutions"] == ["x=2", "x=3"]


def test_snapshot_is_capped(store):
    """整页 40 行以上的批改不常见，但真出现时不能把文件撑爆。"""
    steps = [_step(i, raw=f"x={i}") for i in range(1, 60)]
    history.record_grade(_check(steps=steps))
    d = history._read_raw()[0]["detail"]
    assert len(d["lines"]) == 40
    assert d["n_more"] == 19


def test_detail_md_for_old_record(store):
    """加快照之前存下的记录要老实说"只有结论"，不能显示成空白页。"""
    rec = history.record("grade", "x^2-5x+6=0", verdict="partial", headline="解不全")
    md = history.detail_md(rec)
    assert "只留了结论" in md
    assert "解不全" in md


def test_detail_md_for_missing_record(store):
    assert "找不到" in history.detail_md(None)


def test_detail_md_grade_table(store):
    history.record_grade(_check(steps=[
        # 真实数据里题面那行是 kind=equation + role=problem + verdict=note，
        # 只按 verdict 翻会显示成"说明"，看不出它是题目本身
        _step(1, raw="x^2-5x+6=0", verdict="note", role="problem", reason="题面"),
        _step(2, raw="(x-2)(x-3)=0"),
    ]))
    md = history.detail_md(history._read_raw()[0])
    assert "| 行 | 写的是 | 判定 |" in md
    assert "题面" in md and "站得住" in md


def test_detail_md_answers_question(store):
    history.record_search(
        "x^2-5x+6=0",
        detail={"solutions": ["x=2", "x=3"], "steps": [{"t": "原式", "x": "x^2-5x+6=0"}]},
    )
    md = history.detail_md(history._read_raw()[0])
    assert "答案" in md and "x=2" in md
    assert "怎么来的" in md


def test_tex_keeps_chinese_out_of_math(store):
    """中文句子包进 `$` 会变成一串斜体乱码。"""
    assert history._tex("解这个方程") == "解这个方程"
    assert history._tex("x^2-5x+6=0") == "$x^2-5x+6=0$"


def test_tex_tolerates_chinese_inside_text(store):
    """`\\text{或}` 是 LaTeX 内部的中文，整条仍要当公式渲染。"""
    got = history._tex(r"x=2 \quad\text{或}\quad x=3")
    assert got.startswith("$") and got.endswith("$")


def test_cell_escapes_pipe(store):
    assert history._cell("a|b") == "a\\|b"


def test_listing_ids_line_up(store):
    """行号和记录 id 必须是同一次排序的结果，否则点 A 看到 B。"""
    for p in ("a", "b", "c"):
        history.record_search(p)
    rows, ids = history.listing()
    assert len(rows) == len(ids) == 3
    assert [history.by_id(k)["problem"] for k in ids] == [r[2] for r in rows]


def test_same_second_newest_first(store):
    """ts 只到秒，同一秒里连做两次时，后做的要排在上面。"""
    history.record("search", "first")
    history.record("search", "second")
    assert [r[2] for r in history.rows()] == ["second", "first"]


def test_pick_md_maps_row_to_record(store):
    """行号到记录的唯一映射在这里，两个界面都靠它。"""
    history.record_search("first")
    history.record_search("second")
    assert "second" in history.pick_md(0)
    assert "first" in history.pick_md(1)
    # 前端传过来的可能是 [row, col]
    assert "second" in history.pick_md([0, 3])


def test_pick_md_out_of_range(store):
    assert "没读到这一行" in history.pick_md(9)
    assert "没读到这一行" in history.pick_md(None)


# --------------------------------------------------------------------------- 给旧记录补答案
#
# 归档的前几版只存结论词，点开是一句"当时只留了结论"。搜题补得起（题目还在），
# 批改补不了（过程没落盘）；这两种情况都不能让页面崩或撒谎。

def test_pick_fills_answer_for_old_search(store):
    history.record_search("x^2-5x+6=0")
    md = history.pick_md(0)
    assert "答案" in md
    assert "x = 2" in md and "x = 3" in md
    # 补出来的要写回文件，不然每点一次都得重解一道题
    assert history._read_raw()[0]["detail"]["how"] == "recomputed"


def test_pick_does_not_recompute_twice(store, monkeypatch):
    history.record_search("x^2-5x+6=0")
    history.pick_md(0)

    def boom(*args, **kwargs):
        raise AssertionError("已有明细就不该再算一次")

    monkeypatch.setattr(history, "snapshot_solved", boom)
    assert "x = 2" in history.pick_md(0)


def test_pick_survives_unsolvable(store):
    """题目现在也解不出来时，老实说，而不是抛异常把页面点崩。"""
    history.record_search("解这个方程")
    md = history.pick_md(0)
    assert "没得到答案" in md
    assert history._read_raw()[0]["detail"]["failed"] is True


def test_recomputed_detail_says_it_is_recomputed(store):
    """答案是按现在的代码重算的，可能和当时那次不一样，得说清楚。"""
    history.record_search("x^2-5x+6=0")
    assert "重算" in history.pick_md(0)


def test_topic_word_is_chinese(store):
    """详情页是给人读的，不能把 equation 这种内部名字端上去。"""
    history.record_search("x^2-5x+6=0")
    md = history.pick_md(0)
    assert "题型：方程" in md
    assert "题型：equation" not in md


def test_old_grade_lists_its_pending(store):
    """逐行判定补不回来，但"哪几步没认准"当时就单独存了。"""
    history.record(
        "grade", "2x+y=4", verdict="unread",
        pending=[{"i": 2, "text": "x+3y=7", "why": "没认准", "by": "system", "done": False}],
    )
    md = history.pick_md(0)
    assert "批改的过程当时没落盘" in md
    assert "第 2 步" in md and "x+3y=7" in md


def test_old_grade_is_not_recomputed(store):
    """批改没有输入可重算，不能拿题面去猜一份假的逐行判定。"""
    history.record("grade", "x^2-5x+6=0", verdict="pass", headline="全对")
    rec = history._read_raw()[0]
    assert not history.fill_missing_detail(rec).get("detail")
