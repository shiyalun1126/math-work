"""配方法推导：`ax²+bx+c=0` → 求根公式。

写这层的理由是「光给六支结论回答不了求根公式怎么来的」。
但推导比结论更容易写成一段读起来很顺、某一步其实是凑的东西，
所以这里的用例重点不在措辞，而在**每一步真的同解**、以及
**不该出推导的时候别硬出**（数字系数的题套模板会出现「两边除以 1」
和 `- -5 ± √1` 这种）。
"""

import sympy

from mathagent.derive import complete_square
from mathagent.equations import answer

x, a, b, c, m = sympy.symbols("x a b c m")


def test_general_quadratic_has_seven_steps():
    d = complete_square(a * x**2 + b * x + c, 0)
    assert d is not None and d.ok, getattr(d, "error", "")
    assert len(d.steps) == 7


def test_final_step_is_quadratic_formula():
    d = complete_square(a * x**2 + b * x + c, 0)
    last = d.steps[-1].latex
    assert r"\pm" in last and "sqrt" in last and "2 a" in last


def test_every_algebraic_step_is_equivalent():
    """前五步必须是同一个方程的等价变形——这是敢把它给学生看的前提。"""
    d = complete_square(a * x**2 + b * x + c, 0)
    assert d.checks
    assert "等价变形" in d.checks[0]
    assert "均为 0" in d.checks[1]


def test_note_names_the_actual_coefficient():
    """`mx²+2x+1` 的前提是 m≠0，不能把 a 写死进去。"""
    d = complete_square(m * x**2 + 2 * x + 1, 0)
    assert "m ≠ 0" in d.note
    assert "a ≠ 0" not in d.note


def test_note_has_no_python_power_syntax():
    """`a**2` 是 Python 写法，印在界面上像代码，得写成 a²。"""
    a2 = sympy.Symbol("a") ** 2
    d = complete_square(a2 * x**2 + a * x + 1, 0)
    assert d is not None and d.ok
    assert "**" not in d.note, d.note
    assert "a²" in d.note


def test_no_linear_term_uses_direct_square_root():
    """缺一次项时配方是空转（加上 0、右边通分），教材用的是直接开平方。"""
    d = complete_square(a * x**2 - b, 0)
    assert d is not None and d.ok, getattr(d, "error", "")
    assert len(d.steps) == 3
    assert d.steps[-1].title.startswith("两边开平方")
    assert "缺一次项" in d.steps[-1].title
    # 不应该出现「加上一次项系数一半的平方」那种加了 0 的步骤
    assert all("一半的平方" not in s.title for s in d.steps)


def test_numeric_coefficient_gets_no_derivation():
    """数字系数套这个模板只会得到空转步骤和双负号，不出。"""
    assert complete_square(x**2 - 5 * x + 6, 0) is None


def test_non_quadratic_gets_no_derivation():
    assert complete_square(x**3 - 1, 0) is None
    assert complete_square(2 * x + 1, 0) is None


def test_answer_carries_derivation():
    ans = answer("ax^2+bx+c=0", explain=False)
    assert ans.ok
    assert len(ans.branches) == 6
    assert len(ans.derivation) == 7
    assert ans.derivation[0]["title"] == "原方程"


def test_derivation_survives_cache():
    """缓存要连推导一起存。只存答案不存推导，第二次打开就少一章——
    这是最容易被当成「有时有有时没有」的怪现象。"""
    first = answer("ax^2+bx+c=0", explain=False)
    second = answer("ax^2+bx+c=0", explain=False)
    assert second.cached
    assert second.derivation == first.derivation
