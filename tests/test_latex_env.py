"""方程组的不同 LaTeX 写法都要能读。

根因不是识别：两个模型都读对了，一个用 aligned、一个用带逗号的 cases，
解析器两种都吃不下，投票层于是给出「所有候选都无法解析」，
页面显示成做不了。而 `parse` 对 `1)\\left\\{\\begin{aligned}x+y&=6`
还报 ok=True，把 `1)` 的 1 和右边的 6 配成 `1=6`——假成功。

这些用例都是从那次真实故障里反推出来的，不是凭空构造。
"""

from mathagent.equations import answer, extract
from mathagent.latex import canonical_env, parse

# 用户截图里的原样，题号、左括号、aligned、句末句点一个不少
SHOT = (
    "1)\\left\\{\\begin{aligned}x+y&=6,\\\\y+z&=8,\\\\x+z&=10.\\end{aligned}\\right."
)


def test_shot_parses_as_three_equations():
    pe = parse(SHOT)
    assert pe.ok, pe.error
    assert pe.kind == "system"
    assert len(pe.obj) == 3


def test_shot_extracts_and_solves():
    eq = extract(SHOT)
    assert eq.topic.coverage == "in"
    assert eq.topic.label == "三元一次方程组"
    a = answer(SHOT, explain=False)
    assert a.ok, a.topic.note
    assert a.solutions and "x = 4" in a.solutions[0]


def test_aligned_forms_are_canonicalized():
    forms = {
        "题号+left右点": SHOT,
        "裸 aligned": "\\begin{aligned}x+y&=6\\\\y+z&=8\\\\x+z&=10\\end{aligned}",
        "array 带列格式": "\\begin{array}{ll}x+y=6\\\\y+z=8\\\\x+z=10\\end{array}",
        "带逗号的 cases": "1) \\begin{cases} x + y = 6, \\\\ y + z = 8, \\\\ x + z = 10. \\end{cases}",
    }
    for name, s in forms.items():
        got = canonical_env(s)
        assert got.startswith("\\begin{cases}"), (name, got)
        assert got.count("\\\\") == 2, (name, got)
        pe = parse(s)
        assert pe.ok and pe.kind == "system", (name, pe.error)


def test_standard_cases_untouched():
    s = "\\begin{cases}x+y=6\\\\y+z=8\\\\x+z=10\\end{cases}"
    assert canonical_env(s) == s


def test_enum_prefix_removed():
    assert parse("1) x^{2}-5x+6=0").ok
    assert canonical_env("1) x^{2}-5x+6=0") == "x^{2}-5x+6=0"


def test_enum_prefix_does_not_eat_coefficients():
    """题号判据必须带分隔标点，否则 2x 的 2、2.5 的 2. 都会被当成题号。"""
    # 2.5x 的小数点不能被当成分隔点
    assert canonical_env("2.5x+1=0") == "2.5x+1=0"
    # 行首系数不能被剥
    assert canonical_env("2x+3=5") == "2x+3=5"
    pe = parse("2x+3=5")
    assert pe.ok and pe.obj == [(parse("2x+3=5").obj[0][0], 5)]
    assert str(pe.obj[0][0]) == "2*x + 3"


def test_single_equation_untouched():
    s = "x^{2}-5x+6=0"
    assert canonical_env(s) == s
    pe = parse(s)
    assert pe.ok and pe.kind == "equation"


def test_set_notation_not_mangled():
    """集合的 \\left\\{ \\right\\} 不能因为剥壳被削掉一半。"""
    s = "\\left\\{ x \\mid x>0 \\right\\}"
    assert canonical_env(s) == s
