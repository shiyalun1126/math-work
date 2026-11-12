"""方程题型：抽题、判范围、给答案。

这里测的是三件最容易出错、也最容易伤人（冤枉学生）的事：
  1. 题干带中文时能不能抽出方程——抽不出就会答"解析失败"；
  2. 恒等式和无解能不能分开——分不开就会把同解变形判成"无解方程"；
  3. 出了覆盖范围会不会老实说——不说就会拿复数根去答初中题。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mathagent import equations as E  # noqa: E402
from mathagent.latex import parse  # noqa: E402
from mathagent.solve import solve  # noqa: E402


# ---------------------------------------------------------------- 抽题

@pytest.mark.parametrize(
    "text, want",
    [
        ("x^2-5x+6=0", "x^2-5x+6=0"),
        ("解方程：3x+6=15", "3x+6=15"),
        ("解下列方程：x^2=2x", "x^2=2x"),
        ("求方程 2x-6=12 的解", "2x-6=12"),
        (r"\text{解方程：}x^{2}-5x+6=0", r"x^{2}-5x+6=0"),
    ],
)
def test_extract_strips_chinese_instruction(text, want):
    assert E.extract(text).latex == want


def test_extract_joins_multiple_equations_into_system():
    eq = E.extract("解方程组：x+y=3, x-y=1")
    assert eq.topic.kind == "system"
    assert "y" in eq.topic.vars
    ans = E.answer("解方程组：x+y=3, x-y=1", explain=False)
    assert ans.ok and "x = 2" in ans.answer_latex and "y = 1" in ans.answer_latex


def test_extract_says_so_when_no_equation():
    eq = E.extract("今天天气不错")
    assert eq.latex == ""
    assert eq.topic.coverage == "out"


# ---------------------------------------------------------------- 范围

@pytest.mark.parametrize(
    "latex, kind",
    [
        ("3x+6=15", "linear"),
        ("x^2-5x+6=0", "quadratic"),
        (r"\begin{cases} x+y=3 \\ x-y=1 \end{cases}", "system"),
        ("(x+1)/(x-1)=2", "rational"),
        ("2(x-3)=2x-6", "identity"),
        ("x^3-1=0", "cubic"),
        ("\\sin x=0", "nonpoly"),
        ("x^2+y^2=1", "nonlinear"),
    ],
)
def test_classify(latex, kind):
    assert E.classify(parse(latex)).kind == kind


def test_out_of_scope_is_refused_not_answered():
    ans = E.answer("x^3-1=0", explain=False)
    assert not ans.ok
    assert "3 次方程" in ans.error
    assert ans.answer_latex == ""      # 复数根一个都不许漏出来


# ---------------------------------------------------------------- 答题

def test_quadratic_answer_has_two_roots_and_checks():
    ans = E.answer("x^2-5x+6=0", explain=False)
    assert ans.ok
    assert ans.answer_latex == r"x = 2，x = 3"
    assert len(ans.checks) == 2
    assert all("✓" in c for c in ans.checks)


def test_identity_is_not_no_solution():
    """批改里最常见的一步：两边是同解变形。判成"无解"就是冤枉。"""
    for latex in ("x=x", "2(x-3)=2x-6"):
        ans = E.answer(latex, explain=False)
        assert ans.ok, latex
        assert ans.topic.kind == "identity"
        assert "任意实数" in ans.answer_latex


def test_rational_drops_extraneous_root():
    """(x^2-1)/(x-1)=0 解出 ±1，但 x=1 让分母为 0——必须剔除，只留 -1。"""
    ans = E.answer("(x^2-1)/(x-1)=0", explain=False)
    assert ans.topic.kind == "rational"
    assert ans.answer_latex == "x = -1"
    assert any("增根" in d for d in ans.dropped)


def test_identity_fraction_excludes_the_pole():
    """(x²-1)/(x-1)=x+1 是恒等式，但 x=1 处两边都没定义。

    约分把分母约掉了，定义域没跟着消失。这一支以前直接回"任意实数"，
    等于把一个代不回去的点算进答案——分式题的答案错得最典型的一种。
    """
    ans = E.answer("(x^2-1)/(x-1)=x+1", explain=False)
    assert ans.ok
    assert "任意实数" in ans.answer_latex
    assert "1" in ans.answer_latex and "≠" in ans.answer_latex, ans.answer_latex
    assert any("定义域" in (s.get("title") or "") for s in ans.steps)


def test_plain_identity_still_says_any_real():
    """没有分母的恒等式不许被上一条误伤。"""
    ans = E.answer("2(x-3)=2x-6", explain=False)
    assert ans.ok
    assert ans.answer_latex == "x 取任意实数", ans.answer_latex


def test_empty_solution_set_is_an_answer_not_a_failure():
    ans = E.answer("1/(x-2)=0", explain=False)
    assert ans.ok
    assert "无解" in ans.answer_latex


def test_no_real_root_says_so():
    ans = E.answer("x^2+1=0", explain=False)
    assert ans.ok
    assert "无实根" in ans.answer_latex


def test_two_unknown_single_equation_is_edge_not_silently_solved():
    ans = E.answer("x+y=3", explain=False)
    assert ans.topic.coverage == "edge"
    assert "不唯一" in ans.topic.note


# ---------------------------------------------------------------- 含参数

@pytest.mark.parametrize(
    "latex, n_branch",
    [("ax=2", 2), ("(a-1)x=1", 2), ("mx^2+2x+1=0", 4)],
)
def test_param_equation_is_split_into_branches(latex, n_branch):
    """漏一支就是错答案——所以答案必须是分情形的表，不是一句话。"""
    ans = E.answer(latex, explain=False)
    assert ans.topic.kind == "param"
    assert ans.ok
    assert len(ans.branches) == n_branch
    assert ans.branches[0].startswith("①")


def test_param_without_split_says_so():
    """解里带参数但不用分情形时，不能还挂着「按参数分情形讨论」。"""
    ans = E.answer("x^2-2mx+m^2-1=0", explain=False)
    assert ans.topic.kind == "param"
    assert not ans.branches
    assert "不需要分情形" in ans.note


def test_two_unknown_equation_without_param_name_stays_relation():
    """x²+y²=1 里 x、y 都是未知数，不能当成「含参方程」。"""
    assert E.classify(parse("x^2+y^2=1")).kind == "nonlinear"


# ---------------------------------------------------------------- 答案缓存

def test_same_equation_hits_cache_second_time():
    first = E.answer("5x-3=12", explain=False)
    second = E.answer("5x-3=12", explain=False)
    assert second.cached
    assert second.answer_latex == first.answer_latex == "x = 3"


def test_cache_key_is_canonical_not_raw_text():
    """同一个方程换一种写法，应该命中同一份答案。"""
    E.answer("3x^2-12=0", explain=False)
    again = E.answer("3 x^{2} - 12 = 0", explain=False)
    assert again.cached
    assert again.answer_latex


def test_out_of_scope_is_not_cached(monkeypatch):
    """算不了的东西不该进缓存——否则以后修好了还会拿到旧的「不给」。"""
    ans = E.answer("x^4-1=0", explain=False)
    assert not ans.ok
    hit = E._ANSWER_CACHE.get(E.CACHE_VERSION, parse("x^4-1=0").sympy_str, "plain")
    assert hit is None


# ---------------------------------------------------------------- 应用题列式

def _fake_model(monkeypatch, content: str):
    monkeypatch.setattr(E.config, "has_llm", lambda: True)
    monkeypatch.setattr(
        E._CACHE, "remember",
        lambda fn, *parts: {"ok": True, "text": content, "note": ""},
    )


def test_word_problem_is_recognized(monkeypatch):
    assert E.looks_like_word_problem("小明买3支笔共花18元，每支笔多少元？")
    assert not E.looks_like_word_problem("x^2-5x+6=0")
    assert not E.looks_like_word_problem("ok")


def test_word_problem_setup_passes_answer_to_cas(monkeypatch):
    """模型只许列方程；答案必须由 CAS 算、并且代回验算过。"""
    _fake_model(monkeypatch, "3x = 18")
    ans = E.answer("小明买3支笔共花18元，每支笔多少元？", explain=False)
    assert ans.ok
    assert ans.latex == "3x = 18"
    assert ans.answer_latex == "x = 6"
    assert ans.provided is False                      # 界面要标明"方程是模型列的"
    assert "模型" in ans.note
    assert ans.checks and "✓" in ans.checks[0]


def test_model_output_that_is_not_an_equation_is_refused(monkeypatch):
    _fake_model(monkeypatch, "每支笔 6 元")
    eq = E.setup_equation("小明买3支笔共花18元，每支笔多少元？")
    assert eq.latex == ""
    assert eq.topic.coverage == "out"


def test_model_saying_unknown_is_respected(monkeypatch):
    _fake_model(monkeypatch, "UNKNOWN")
    eq = E.setup_equation("这段文字没有数量关系")
    assert eq.latex == ""
    assert "列不出" in eq.topic.label or "列不出" in eq.note
