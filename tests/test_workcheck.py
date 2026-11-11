"""批改引擎的边界测试。

重点不是"能不能判对"，是**判错的时候说不说得清**：
每条 ✗ 都必须带一个具体反例，否则用户没法核对，这条判定就不该出现。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mathagent import cases  # noqa: E402
from mathagent.workcheck import grade, split_lines  # noqa: E402

PASS_WORK = """x^2-5x+6=0
(x-2)(x-3)=0
x=2 或 x=3"""

# 移项错了一步，后面将错就错，最后把答案抄对了。这一类最该抓。
FAKE_WORK = """x^2-5x+6=0
x^2-6=5x
x^2-5x=6
x=2 或 x=3"""

MISS_WORK = """x^2-5x+6=0
(x-2)(x-3)=0
x=2"""

SIGN_WORK = """2x-6=12
2x=12
x=6"""


def test_pass():
    r = grade(PASS_WORK)
    assert r.grade == "pass", r.headline
    assert r.first_error < 0
    assert [s.verdict for s in r.steps].count("ok") == 2


def test_fake_derivation():
    """过程不成立但答案对：必须判 fake，不能判 pass。"""
    r = grade(FAKE_WORK)
    assert r.grade == "fake", r.grade
    assert r.first_error == 2
    assert r.error_type == "伪推导"


def test_missing_root():
    r = grade(MISS_WORK)
    assert r.grade == "partial", r.grade
    assert r.error_type == "漏解"
    assert "3" in r.headline


def test_sign_error_has_counterexample():
    r = grade(SIGN_WORK)
    assert r.grade == "wrong", r.grade
    bad = [s for s in r.steps if s.verdict == "wrong"]
    assert bad, "应当至少有一行判不成立"
    # 判错必须带得走：要么有反例，要么说清了差多少
    assert bad[0].counter or "常数" in bad[0].reason, bad[0]


def test_answer_not_satisfying_problem():
    r = grade("x^2-5x+6=0\nx=5")
    assert r.grade == "wrong", r.grade
    assert "≠ 0" in r.steps[-1].counter


def test_prose_line_is_not_judged():
    """夹着人话的行：式子照判，人话当声明留着，不参与算对错。"""
    r = grade("x^2-5x+6=0\n两边同乘2得 2x^2-10x+12=0\nx=2 或 x=3")
    assert r.grade == "pass", r.headline
    said = [s for s in r.steps if s.declared]
    assert said, "「两边同乘2」应当被识别成学生声明的动作"
    assert "两边同乘" in said[0].reason, said[0].reason
    # 说明行本身不产生 ✓ / ✗ 之外的判定
    assert all(s.verdict in ("ok", "note") for s in r.steps)


def test_problem_can_be_given_separately():
    r = grade(PASS_WORK, problem="x^2-5x+6=0")
    assert r.grade == "pass"
    assert r.steps[0].role == "problem", "题面要对上过程里那一行，不能显示成'第一处推导'"


def test_no_formula():
    r = grade("这道题我不会")
    assert r.grade == "gap"
    assert "没有一道能读出来" in r.headline


def test_split_lines_drops_blank():
    assert split_lines("a\n\n  b  \n") == ["a", "b"]


def test_case_store_roundtrip():
    tmp = Path(tempfile.mkdtemp()) / "cases.jsonl"
    old, cases.CASES = cases.CASES, tmp
    try:
        r = grade(SIGN_WORK)
        rec = cases.add(r)
        assert rec["id"]
        loaded = cases.load()
        assert len(loaded) == 1
        assert loaded[0]["error_type"] == r.error_type
        st = dict(cases.stats())
        assert st.get(r.error_type) == 1
        row = cases.rows()[0]
        assert len(row) == len(cases.headers())
    finally:
        cases.CASES = old


def test_stats_survive_broken_line():
    tmp = Path(tempfile.mkdtemp()) / "cases.jsonl"
    old, cases.CASES = cases.CASES, tmp
    try:
        tmp.write_text('{"error_type":"漏解"}\nnot json\n', encoding="utf-8")
        assert len(cases.load()) == 1
    finally:
        cases.CASES = old


def test_multivariate_equation_does_not_crash():
    """多元式子会让 solve 返回 And，list(...) 曾直接抛 TypeError 崩掉整页批改。

    这类题在真实作业里存在（方程组、圆的方程），崩一次后面全批不了。
    解集算不出来只能跳过，不能顺势判学生错。
    """
    r = grade("x^2+y^2=1\nx=1\ny=0")
    assert r.grade in ("wrong", "unknown", "unread"), r.headline
    assert r.steps, "至少要给出逐行结果，不能整个崩掉"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("批改引擎测试全部通过")


# ------------------------------------------------------------------ 方程组
#
# 单方程是递推链，方程组是条件集。前两行是并列的题给条件，谁也不是从谁推出来的。
# 下面这批用例锁住这件事：一份**正确**的方程组解答，不能因为"第二行接不上第一行"
# 被判错。这是最容易伤到学生的一类误判——越老实抄题越容易中。


def test_system_treats_parallel_conditions_as_given():
    """x+y=5 后面接 x-y=1 是并列条件，不是"对不上上一步"。"""
    w = grade(["x+y=5", "x-y=1", "2x=6", "x=3", "y=2"])
    assert w.grade == "pass"
    step2 = next(s for s in w.steps if s.idx == 2)
    assert step2.verdict == "ok"
    assert "条件" in step2.reason


def test_system_elimination_step_is_accepted():
    """两式相加得到 2x=6：联立后解集不变，成立。"""
    w = grade(["x+y=5", "2x=6", "x=3", "y=2"], problem="x+y=5, x-y=1")
    assert w.grade == "pass"


def test_system_wrong_condition_is_caught_when_problem_given():
    """题面给了 x-y=1，学生抄成 x-y=3，与题面联立无解，必须判错。"""
    w = grade(["x+y=5", "x-y=3", "2x=8", "x=4", "y=1"], problem="x+y=5, x-y=1")
    assert w.grade == "wrong"
    assert w.first_error == 2


def test_system_only_x_solved_is_partial():
    """只解出 x 没解出 y，是漏解，不是全对。"""
    w = grade(["x+y=5", "2x=6", "x=3"], problem="x+y=5, x-y=1")
    assert w.grade == "partial"
    assert "x" in w.headline


def test_system_wrong_answer_is_not_pass():
    """x=4 代回 x+y=5、x-y=1 不成立，不能判过。"""
    w = grade(["x+y=5", "2x=6", "x=4", "y=2"], problem="x+y=5, x-y=1")
    assert w.grade != "pass"


def test_system_bad_steps_good_answer_is_fake():
    """过程错但答案对，是伪推导——这是最该报出来的一类。"""
    w = grade(["x+y=5", "x-y=3", "2x=8", "x=3", "y=2"], problem="x+y=5, x-y=1")
    assert w.grade == "fake"


def test_system_three_variables():
    w = grade(
        ["x+y+z=6", "x-y=0", "z=2", "x=2", "y=2", "z=2"],
        problem="x+y+z=6, x-y=0, z=2",
    )
    assert w.grade == "pass"


def test_system_contradictory_problem_refuses_to_grade():
    """题面自己矛盾时不给对错结论——学生没错，是题有问题。"""
    w = grade(["x+y=1", "x+y=3"], problem="x+y=1, x+y=3")
    assert w.grade == "gap"
    assert "无解" in w.headline


def test_single_equation_still_uses_adjacent_comparison():
    """只含一个未知数时不能溜进方程组判法，否则会放过真正的变形错误。"""
    w = grade(["x^2-5x+6=0", "x^2-6=5x", "x=2 或 x=3"])
    assert w.grade in ("pass", "partial", "fake")
    assert all(s.verdict != "unknown" or s.kind == "prose" for s in w.steps)


# ------------------------------------------------- 因式分解错与并排写的多个答案
#
# 这一批来自用户真实作业：x²-5x+6=0 被拆成 (x+1)(x-6)=0。
# 判错是对的，但**理由**原来写"移项/变号错"——学生照着移项查十遍也查不出来。


def test_factorization_error_is_named_as_factorization():
    """拆括号错要说是"因式分解"，不能说成"移项/变号"。"""
    w = grade(["x^2-5x+6=0", "(x+1)(x-6)=0"], problem="x^2-5x+6=0")
    assert w.grade == "wrong"
    assert w.error_type == "因式分解错"
    step2 = next(s for s in w.steps if s.idx == 2)
    assert "常数项" in step2.reason
    assert "移项" not in step2.reason


def test_factorization_error_points_at_the_right_coefficient():
    """错的是一次项时，就该说一次项，不能总说常数项。"""
    w = grade(["x^2-5x+6=0", "(x-2)(x-4)=0"], problem="x^2-5x+6=0")
    step2 = next(s for s in w.steps if s.idx == 2)
    assert w.error_type == "因式分解错"
    assert "一次项" in step2.reason


def test_counterexample_is_written_as_substitution():
    """反例要写成"代进去左边多少、右边多少"，不能写"两边相减为 0"。"""
    w = grade(["x^2-5x+6=0", "(x+1)(x-6)=0"], problem="x^2-5x+6=0")
    step2 = next(s for s in w.steps if s.idx == 2)
    assert "代进去" in step2.counter
    assert "两边相减" not in step2.counter
    assert "|" not in step2.counter, "竖线会把 Markdown 表格切散"


def test_two_answers_separated_by_spaces_are_split():
    """`x=-1  x=6` 中间只有空格，也要认出是两个答案，不能报"读不出"。"""
    from mathagent.workcheck import _read

    line = _read(3, "x=-1  x=6")
    assert line.kind == "assign"
    assert [(s.name, v) for s, v in line.assigns] == [("x", -1), ("x", 6)]


def test_two_answers_separated_by_quad_are_split():
    """识别结果是 LaTeX 时，间隔是两个答案之间的 `\\quad`。"""
    from mathagent.workcheck import _read

    line = _read(3, r"x=-1\quad x=6")
    assert line.kind == "assign"
    assert len(line.assigns) == 2


def test_split_does_not_invent_an_answer():
    """`x=2y` 只有一个等号，切开会凭空造出 `x=2` 这种学生没写的答案。"""
    from mathagent.workcheck import _read

    line = _read(1, "x=2y")
    assert line.kind == "equation"
    assert line.assigns == []
