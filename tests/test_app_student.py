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
    # 点开要有东西可看：解集和步骤都在快照里
    assert recs[0]["detail"].get("solutions"), "没存解集，点开就是一张空页"
    assert recs[0]["detail"].get("steps")


def test_check_records_once(app):
    app.run_check("", "x^2-5x+6=0\n(x-2)(x-3)=0\nx=2", None)
    recs = history._read_raw()
    grades = _kind(recs, "grade")
    assert len(grades) == 1
    assert grades[0]["verdict"], "批改记录必须带结论，否则时间线上只有一个空壳"
    assert grades[0]["n_steps"] >= 2
    # 逐行判定也要落盘，不然点开只剩一句结论
    assert len(grades[0]["detail"].get("lines") or []) >= 2


def test_both_actions_share_one_timeline(app):
    app.run_answer("3x+6=15", None, True)
    app.run_check("", "x+2=5\nx=3", None)
    rows = history.rows()
    assert len(rows) == 2
    assert {r[1] for r in rows} == {"搜题", "批改"}


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


class _Pick:
    """假的 SelectData。`_on_pick` 只读 `index`，鸭子类型够用。"""

    def __init__(self, row: int):
        self.index = [row, 2]


def test_pick_opens_that_row(app):
    """点第几行就要出第几条。行号错位是这类界面最典型的 bug。"""
    history.record_search("x^2-5x+6=0", detail={"solutions": ["x=2"]})
    history.record_search("3x+6=15", detail={"solutions": ["x=3"]})
    newest = app._on_pick(_Pick(0))
    older = app._on_pick(_Pick(1))
    assert "3x+6=15" in newest
    assert "x^2-5x+6=0" in older


def test_pick_out_of_range_does_not_raise(app):
    history.record_search("x^2-5x+6=0")
    assert "没读到这一行" in app._on_pick(_Pick(99))


def test_hist_state_shape(app):
    """返回 6 项，顺序要和 outputs 对齐。"""
    state = app._hist_state("试一下")
    assert len(state) == 6
    assert isinstance(state[0], str) and state[0].startswith("###")
    assert state[4] == "试一下"
    assert "点上面" in state[5], "刷新后详情区要退回提示语，不能留着上一条"


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


# --------------------------------------------------------------------------- 先给题目，再给答案
#
# 拍照进来时，屏幕上只有答案是不够的：学生先要能确认"系统把我的式子读成了什么"，
# 才能判断这个答案是不是在答他那道题。

def test_answer_starts_with_the_problem(app):
    out = app.run_answer("x^2-5x+6=0", None, True)
    assert out[0].startswith("### 题目")
    assert "x^2-5x+6=0" in out[0]
    # 题目在前、答案在后，顺序错了这一段就白做
    assert "### 答案" in out[2]


def test_problem_says_low_confidence(app):
    """照片认得不准时，题目下面要提醒先核对，而不是闷着头给答案。"""
    class _Ans:
        latex = "2x+y=4"

    md = app._problem_md(_Ans(), {"confidence": 0.55})
    assert "55%" in md
    assert "对一下" in md or "看一遍" in md


def test_problem_typed_has_no_photo_note(app):
    class _Ans:
        latex = "x^2-5x+6=0"

    md = app._problem_md(_Ans(), None, typed=True, raw="x^2-5x+6=0")
    assert "照片" not in md


def test_problem_falls_back_to_raw_text(app):
    """解析失败时 ans.latex 是空的，至少把用户打的字摆出来。"""
    class _Ans:
        latex = ""

    md = app._problem_md(_Ans(), None, typed=True, raw="解这个方程")
    assert "解这个方程" in md


def test_problem_without_anything_says_so(app):
    class _Ans:
        latex = ""

    assert "没读出一个完整的式子" in app._problem_md(_Ans(), None)
