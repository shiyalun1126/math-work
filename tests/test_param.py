"""含参讨论的测试。

「含参」这件事的判据不是题里出现了几个字母，而是**答案要不要分情形**。
以前 ax²+bx+c=0 直接套求根公式，等于默认 a≠0，a=0 那一支悄悄没了——
学生拿到的答案看着对，实际上漏了一整类情形。

这里守五件事：
  1. 降次那一支必须出现（a=0 时方程是一次方程，不是「无解」）
  2. 判别式的三支（Δ>0、Δ=0、Δ<0）都要有，且 Δ<0 叫「无实根」不叫「无解」
  3. 只靠整数取样试不到的情形（m=9/8 时才 Δ=0）也得能找到见证
  4. 不可能成立的分支（系数恒为 1 时的 A=0）要被删掉，不能写进答案
  5. 没有真参数、或未知数不是 x 时，别硬套分类讨论
"""

from __future__ import annotations

from mathagent.latex import parse
from mathagent.solve import solve


def run(tex: str):
    return solve(parse(tex))


def cases(sr) -> str:
    assert sr.ok, sr.error
    assert sr.kind == "param", sr.kind
    return sr.solutions[0]


def test_linear_keeps_the_degenerate_branch():
    """ax=b：三支都要有——a≠0 有一解、a=0 且 b=0 任意解、a=0 且 b≠0 无解。"""
    text = cases(run(r"ax=b"))
    assert r"a \ne 0" in text, text
    assert "b = 0" in text and "b \\ne 0" in text, text
    assert "任意实数" in text, text
    assert "无解" in text, text


def test_quadratic_keeps_the_degraded_case():
    """ax²+(a-1)x-1=0：a=0 时降成一次方程，这一支不能丢。"""
    text = cases(run(r"ax^2+(a-1)x-1=0"))
    assert r"a \ne 0" in text, text
    assert "a = 0" in text, text
    # 两个根是 x=-1 与 x=1/a
    assert "x = -1" in text, text
    assert r"\frac{1}{a}" in text, text


def test_discriminant_three_ways():
    """判别式三种情形各一条，且 Δ<0 要说「无实根」而不是「无解」。"""
    text = cases(run(r"ax^2+2x+1=0"))
    assert r"\Delta > 0" in text and r"\Delta = 0" in text and r"\Delta < 0" in text, text
    assert "无实根" in text, text
    assert "无解" not in text, "二次方程没有实根，不能说方程无解"


def test_exact_witness_beyond_the_grid():
    """mx²-3x+2=0 的 Δ=0 只出现在 m=9/8，靠整数取样试不到。"""
    sr = run(r"mx^2-3x+2=0")
    text = cases(sr)
    assert r"\Delta = 0" in text, text
    assert "9/8" in sr.steps[-1].latex, "见证里应该有 m=9/8"


def test_impossible_branch_is_dropped():
    """x²+2x+m=0 的二次项系数恒为 1，不存在「A=0」这一支。"""
    text = cases(run(r"x^2+2x+m=0"))
    assert "A = 0" not in text, text
    assert "1 = 0" not in text, text


def test_no_discussion_when_coefficients_are_plain():
    """x²=x 只有一个字母，就是普通求解，不该冒出「情形一」。"""
    sr = run(r"x^2=x")
    assert sr.kind == "equation"
    assert sr.solutions == ["x = 0", "x = 1"]


def test_unknown_is_x_not_the_first_letter():
    """x²-2mx+m²-1=0 要解 x，不能因为 m 排在前头就去解 m。"""
    sr = run(r"x^2-2mx+m^2-1=0")
    assert sr.ok
    assert all(s.startswith("x =") for s in sr.solutions), sr.solutions


def test_every_branch_is_verified():
    """最后一步验算，列出每支用的见证值。"""
    sr = run(r"ax^2+bx+c=0")
    assert "验算" in sr.steps[-1].title, sr.steps[-1].title
    # 六个情形各有见证
    assert sr.steps[-1].latex.count("\\quad") >= 4, sr.steps[-1].latex


def test_general_quadratic_has_all_six_cases():
    """ax²+bx+c=0 是课本上那张表：A≠0 三支 + A=0 三支。"""
    text = cases(run(r"ax^2+bx+c=0"))
    assert "任意实数" in text, text
    assert "b \\ne 0" in text and "c \\ne 0" in text, text
