"""学生版的归档接入：两个动作各记一条，第三个切换能翻出来。

为什么单独测这一层：`test_history.py` 守的是存储，`app.py` 那边守的是多题型界面。
学生版是另一个入口，它的 `run_answer` / `run_check` 各写一遍记账调用——
漏写一个，页面上不报错，只是归档里少了一半记录，这种事只有测试能发现。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mathagent import history  # noqa: E402


@pytest.fixture()
def app(monkeypatch, tmp_path):
    """把归档落到临时文件，再把学生版模块拿进来。

    必须先进模块、再改路径：`app_student` 在导入时就会渲染一次归档区。
    顺序反了的话，导入那一下会读真实 `data/history.jsonl`，
    测试断言就会跟着本机历史跑。
    """
    import app_student

    monkeypatch.setattr(history, "HISTORY", tmp_path / "history.jsonl")
    return app_student


def _kind(recs, want):
    return [r for r in recs if r.get("kind") == want]


def test_switch_is_three_way(app):
    """点「归档」只亮那一个按钮，另外两块必须收起来。"""
    ups = app.switch("hist")
    assert len(ups) == 6, "按钮 3 个 + 面板 3 块的顺序一变，界面就串位"
    variants = [u.get("variant") for u in ups[:3]]
    visible = [u.get("visible") for u in ups[3:]]
    assert variants == ["secondary", "secondary", "primary"]
    assert visible == [False, False, True]


def test_switch_back_to_answer(app):
    ups = app.switch("answer")
    assert [u.get("variant") for u in ups[:3]] == ["primary", "secondary", "secondary"]
    assert [u.get("visible") for u in ups[3:]] == [True, False, False]


def test_answer_records_once(app):
    app.run_answer("x^2-5x+6=0", None, True)
    recs = history._read_raw()
    assert len(_kind(recs, "search")) == 1
    # 记的是待解的方程，不是界面上的原始输入
    assert recs[0]["problem"] == "x^2-5x+6=0"
    assert recs[0]["source"] == "text"


def test_check_records_once(app):
    app.run_check("", "x^2-5x+6=0\n(x-2)(x-3)=0\nx=2", None)
    recs = history._read_raw()
    grades = _kind(recs, "grade")
    assert len(grades) == 1
    assert grades[0]["verdict"], "批改记录必须带结论，否则时间线上只有一个空壳"
    assert grades[0]["n_steps"] >= 2


def test_both_actions_share_one_timeline(app):
    app.run_answer("3x+6=15", None, True)
    app.run_check("", "x+2=5\nx=3", None)
    rows = history.rows()
    assert len(rows) == 2
    assert {r[1] for r in rows} == {"搜题", "批改"}


def test_hist_html_escapes_problem(app, monkeypatch):
    """题目是用户输入，直接拼进 HTML 会被当成标签。"""
    monkeypatch.setattr(
        history, "rows",
        lambda limit=60: [["11-12 15:00", "搜题", "x<3 且 y>1", "看过", "—"]],
    )
    out = app._hist_html()
    assert "x&lt;3" in out and "x<3" not in out


def test_hist_html_empty_points_somewhere(app, monkeypatch):
    monkeypatch.setattr(history, "rows", lambda limit=60: [])
    out = app._hist_html()
    assert "还没有记录" in out
    assert "对答案" in out, "空态要告诉用户回哪个按钮做一次"


def test_pend_html_marks_manual(app, monkeypatch):
    app.__dict__  # 只为让 lint 知道 app 被用到
    history.add_pending("3x+6=15", "第 2 行", "移项方向没想通")
    out = app._pend_html()
    assert "第 2 行" in out
    assert "我标的" in out
    assert "移项方向没想通" in out


def test_pend_html_empty_explains(app, monkeypatch):
    monkeypatch.setattr(history, "pending_rows", lambda include_done=False: [])
    out = app._pend_html()
    assert "没有欠着的步骤" in out


def test_hist_state_shape(app):
    """返回 5 项，顺序要和 outputs 对齐。"""
    state = app._hist_state("试一下")
    assert len(state) == 5
    assert isinstance(state[0], str) and state[0].startswith("###")
    assert state[4] == "试一下"


def test_pend_choices_key_is_record_key(app):
    rec = history.add_pending("3x+6=15", "第 3 行", "")
    label, key = app.pend_choices()[0]
    assert key == f"{rec['id']}:0"
    assert "3x+6=15" in label and "第 3 行" in label


def test_archived_swallows_but_logs(app, capsys, monkeypatch):
    """归档写不进去时不能连答案一起吞掉——但也不能一声不吭。"""

    def boom(*a, **k):
        raise OSError("磁盘满了")

    app._archived(boom, 1)
    err = capsys.readouterr().err
    assert "归档失败" in err and "磁盘满了" in err
