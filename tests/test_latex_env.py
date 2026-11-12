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


def test_incomplete_cases_row_is_refused_with_its_row_number():
    """cases 里有一行不是等式，整题不能解。

    以前是 `[e for e in eqs if len(e) == 2]` 一过滤，读不出的行直接消失，
    剩下那行还能解出一个"答案"——把一道残缺的输入悄悄换成另一道题。
    错在哪也看不出来，比报错糟得多。
    """
    s = "\\begin{cases}x=1\\\\x+2\\end{cases}"
    pe = parse(s)
    assert not pe.ok
    assert "第 2 行" in pe.error, pe.error


def test_incomplete_row_inside_aligned_is_also_refused():
    """aligned 里的说明行/分段条件同样不能当成方程悄悄丢掉。"""
    s = "\\begin{aligned}x=1\\\\\\text{所以}\\end{aligned}"
    pe = parse(s)
    assert not pe.ok
    assert "行" in pe.error


def test_system_with_a_stray_note_is_refused():
    """`x=1, 说明` 这种混着一段话的输入，不能只取前半截。"""
    pe = parse("x=1, y=2, 所以")
    assert not pe.ok
    assert "段" in pe.error or "行" in pe.error


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


# ---------------------------------------------------------------- 手打根号
#
# `sqrt(x+1)` 不写反斜杠、不用花括号，被 latex2sympy 读成 `q*r*s*t*(x+1)`：
# 不报错，只是凭空多出四个"参数"，无理方程于是被判成「含参数方程（a、m…）」。
# 学生用手机手打公式就是这么写的，这条比方程组那次更难发现——它不失败，只是答得不对。


def test_sqrt_without_backslash():
    pe = parse("sqrt(x+1)=x-1")
    assert pe.ok, pe.error
    assert str(pe.obj[0][0]) == "sqrt(x + 1)"


def test_sqrt_variants_agree():
    """三种打发号都得读成同一个式子，否则同一道题换个写法就变一道题。"""
    want = str(parse("\\sqrt{x+1}").obj[0])
    for s in ("sqrt(x+1)", "\\sqrt(x+1)", "\\sqrt{x+1}", "√(x+1)"):
        pe = parse(s)
        assert pe.ok, (s, pe.error)
        assert str(pe.obj[0]) == want, (s, str(pe.obj[0]))


def test_radical_not_mistaken_for_parameter():
    """判成「含参数」是最坏的错法：答案里会冒出 a、m 分支，而题里根本没有字母。"""
    eq = extract("sqrt(x+1)=x-1")
    assert eq.topic.kind == "radical", eq.topic.label
    assert eq.topic.coverage == "edge"


def test_radical_solved_and_extraneous_root_dropped():
    a = answer("sqrt(x+1)=x-1", explain=False)
    assert a.ok, a.topic.note
    # 平方后 x=3 与 x=-1 都出现，-1 让左边 = 0、右边 = -2，必须剔掉
    assert a.solutions == ["x = 3"], a.solutions
    assert "x = 3" in " ".join(a.checks)


def test_double_radical():
    a = answer("sqrt(x+2)+sqrt(x-1)=3", explain=False)
    assert a.ok, a.topic.note
    assert a.solutions == ["x = 2"], a.solutions


# ---------------------------------------------------------------- 题目尾注
#
# `ax^{2}+bx+c=0\left(a,b,c\text{均为常数}\right)` 是课本上最标准的一行。
# 尾注被当成式子的一部分，整串解析失败，界面说「题面里没有可以解析的等式」——
# 而 discuss() 早就算出了六支答案。识别没毛病，是解析把注解吃进去了。

NOTE_RAW = r"ax^{2}+bx+c=0\left(a,b,c\text{均为常数}\right)"


def test_note_stripped():
    from mathagent.latex import normalize

    assert normalize(NOTE_RAW) == "ax^{2}+bx+c=0"
    assert normalize("ax^2+bx+c=0 (a,b,c为常数)") == "ax^2+bx+c=0"


def test_definition_domain_kept():
    """括号里是条件（x>0）就不能删，删了等于丢条件。"""
    from mathagent.latex import normalize

    assert normalize("x^2=4 (x>0)") == "x^2=4 (x>0)"
    assert normalize("(a+b)^2=4") == "(a+b)^2=4"


def test_general_quadratic_from_text():
    """a、b、c 当参数，答案必须是分情形表，而且第一支就是求根公式。"""
    a = answer(NOTE_RAW, explain=False)
    assert a.ok, a.error
    assert a.topic.kind == "param"
    assert len(a.branches) == 6, a.branches
    assert "sqrt" in a.branches[0] and "2 a" in a.branches[0]


def test_general_quadratic_keeps_degenerate_branches():
    """a=0 那一支不能丢——丢了就是漏解，这是含参题最容易错的地方。"""
    a = answer("ax^2+bx+c=0", explain=False)
    joined = " ".join(a.branches)
    assert "a = 0" in joined and "b \\ne 0" in joined
