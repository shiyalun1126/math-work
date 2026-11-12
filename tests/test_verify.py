"""投票与等价判定。

这个文件里的每条用例都对应一个"系统会答得很确定、其实答错了"的场景。
判错的门槛本来就高，一次冤枉之后，用户就不信后面所有的"你错了"了。
"""

from __future__ import annotations

from mathagent.latex import parse
from mathagent.types import Candidate
from mathagent.verify import cluster, equivalent, verify


def _sys(*rows: str) -> str:
    return r"\begin{cases}" + r"\\".join(rows) + r"\end{cases}"


# ---------------------------------------------------------------- 方程组等价


def test_residual_sum_is_not_a_system_representation():
    """解 (2,1) 与解 (2,28) 的两个方程组不能判等价。

    以前把每条方程相加压成 `2x-4`：y 的条件一正一负正好抵消，
    两个不同的题就成了同一簇，投票还会把这个当成"两个来源一致"。
    """
    a = parse(_sys("x+y=3", "x-y=1"))       # (2, 1)
    b = parse(_sys("x+y=30", "x-y=-26"))    # (2, 28)
    assert a.ok and b.ok and a.kind == b.kind == "system"
    assert not equivalent(a, b)


def test_same_system_written_differently_still_matches():
    """系数整体乘 2 还是同一个方程组，别矫枉过正。"""
    a = parse(_sys("x+y=3", "x-y=1"))
    b = parse(_sys("2x+2y=6", "3x-3y=3"))
    assert equivalent(a, b)


def test_system_never_merges_with_a_single_equation():
    """`{x+y=3, x-y=1}` 和 `x=2` 不是一回事，即使某个式子碰巧同解。"""
    a = parse(_sys("x+y=3", "x-y=1"))
    b = parse("2x-4=0")
    assert not equivalent(a, b)


def test_reordered_rows_are_the_same_system():
    a = parse(_sys("x+y=3", "x-y=1"))
    b = parse(_sys("x-y=1", "x+y=3"))
    assert equivalent(a, b)


def test_legal_linear_combination_is_the_same_system():
    """`{2x=4, 2y=2}` 和 `{x+y=3, x-y=1}` 是同一组解。

    逐条比认不出来（既不是同一个式子，也不是差一个常数倍），
    但系数全是数字时可以比增广矩阵的行空间——行空间相同就是同一组解。
    """
    a = parse(_sys("x+y=3", "x-y=1"))
    b = parse(_sys("2x=4", "2y=2"))
    assert equivalent(a, b)


def test_contradictory_systems_are_not_merged():
    """`x+y=3` 与 `x+y=30` 同时要求是不可能的，和"有唯一解"那组不是一回事。"""
    a = parse(_sys("x+y=3", "x-y=1"))
    b = parse(_sys("x+y=3", "x+y=30"))
    assert not equivalent(a, b)


def test_collision_does_not_produce_a_confident_verdict():
    """两个来源把同一条手写方程组读成不同解，不能判成"2/2 一致"。"""
    vr = verify([
        Candidate(source="qwen3-vl-plus", latex=_sys("x+y=3", "x-y=1")),
        Candidate(source="glm-4v", latex=_sys("x+y=30", "x-y=-26")),
    ])
    assert vr.verdict == "review", vr.reasons
    assert vr.confidence <= 0.5
    assert not vr.independent


# ---------------------------------------------------------------- 旁证口径


def test_corroboration_counts_the_winner_not_the_whole_field():
    """qwen 两票说 x=1、glm 一票说 x=2：赢家只有一家，没有独立旁证。

    以前按全场厂商算，glm 那张反对票反倒成了赢家的"旁证"。
    """
    vr = verify([
        Candidate(source="qwen3-vl-plus", latex="x=1"),
        Candidate(source="qwen2.5-vl", latex="x=1"),
        Candidate(source="glm-4v", latex="x=2"),
    ])
    assert not vr.independent
    assert vr.verdict == "review", vr.reasons
    assert "没有独立旁证" in " ".join(vr.reasons)


def test_single_vendor_is_capped_and_labelled():
    vr = verify([
        Candidate(source="qwen3-vl-plus", latex="x=1"),
        Candidate(source="qwen3-vl-plus", latex="x=1"),
        Candidate(source="qwen3-vl-plus", latex="x=1"),
    ])
    assert not vr.independent
    assert vr.confidence <= 0.8
    assert "自一致性" in " ".join(vr.reasons)


def test_real_two_vendor_agreement_still_ok():
    """真有两个厂商一致时不能被这套规矩误伤。"""
    vr = verify([
        Candidate(source="qwen3-vl-plus", latex=r"\frac{1}{2}x=1"),
        Candidate(source="glm-4v", latex=r"\frac{x}{2}=1"),
    ])
    assert vr.independent and vr.verdict == "ok"
    assert vr.confidence == 1.0


# ---------------------------------------------------------------- 数值近似


def test_approximate_equality_does_not_chain():
    """0、6e-10、1.2e-9：相邻都在容差内，首尾超出——不能因此合成一簇。

    近似相等不是等价关系，并查集却假设它可传递。改成"只跟簇代表比"之后，
    簇里每个人跟代表都在容差内，"一致"这句话才站得住。
    """
    a, b, c = (parse("x=0"), parse(r"x=6\times10^{-10}"), parse(r"x=1.2\times10^{-9}"))
    assert equivalent(a, b) and equivalent(b, c)
    assert not equivalent(a, c)

    clusters, _ = cluster([
        Candidate(source="s1", latex="x=0"),
        Candidate(source="s2", latex=r"x=6\times10^{-10}"),
        Candidate(source="s3", latex=r"x=1.2\times10^{-9}"),
    ])
    sizes = sorted(cl.size for cl in clusters)
    assert sizes == [1, 2], [(cl.size, cl.latex) for cl in clusters]


# ---------------------------------------------------------------- 同一个方程的不同写法


def test_same_equation_written_the_other_way_round():
    """`x=1` 与 `1=x` 是同一个方程。

    压成 lhs-rhs 之后差的是个负号，判零抓不到。两个模型把同一行读成这两种
    写法时，会被当成"两派各一票，无法裁决"——该收敛的地方反而要人来看。
    """
    assert equivalent(parse("x=1"), parse("1=x"))


def test_same_equation_scaled_by_a_constant():
    """两边同乘一个非零常数，零点不变。"""
    assert equivalent(parse("x=1"), parse("2x=2"))
    assert equivalent(parse("x^2=1"), parse("2x^2=2"))


def test_scaling_does_not_swallow_different_equations():
    """放宽到"差一个倍数"不能把不同的方程并进来。"""
    assert not equivalent(parse("x=1"), parse("x=-1"))
    assert not equivalent(parse("x=1"), parse("x=2"))
    # 表达式不适用：x 和 2x 是两回事
    assert not equivalent(parse("x"), parse("2x"))
    # 恒等式与"x=0"也不是一回事
    assert not equivalent(parse("x=0"), parse("0=0"))
