"""变换型辅助线（平行线、角平分线、旋转、翻折）的测试。

前四种辅助线（连线、作垂线、取中点、延长）都只是「画一条线、造一个点」。
变换不一样：它的价值在于把条件搬来搬去——旋转能把散在三条边上的
PA、PB、PC 凑进同一个三角形，翻折能把折痕两侧的图形叠合。

这里的守则和别处一样：**新点的坐标必须是真算出来的**（平行线与边的交点、
角平分线与边的交点、绕点旋转、关于直线对称），算出坐标之后一切照常验算。
所以下面既测「造得对」，也测「造错了会被挡下」。

坐标直接给，只测推理层，不依赖求解器。
"""

from __future__ import annotations

import math

from geoagent import reason
from geoagent.schema import Problem
from geoagent.solve import GeoSolution


def triangle(**kw) -> Problem:
    return Problem(
        problem_text="在△ABC中，D是AB上一点。",
        points=["A", "B", "C", "D"],
        segments=[("A", "B"), ("B", "C"), ("C", "A")],
        relations=["D在AB上"],
        **kw,
    )


CO = {"A": (0.0, 0.0), "B": (4.0, 0.0), "C": (1.0, 3.0), "D": (1.6, 0.0)}


# ------------------------------------------------------------------ 平行线


def test_parallel_through_point_hits_the_opposite_side():
    """过 D 作 BC 的平行线，交点要落在 CA 上，且这条线真的与 BC 平行。"""
    plans = reason._aux_parallel(triangle(), GeoSolution(ok=True, coords=CO))
    assert plans, "没作出平行线"
    plan = plans[0]
    assert "∥" in plan.desc, plan.desc
    (n, xy), = plan.pts.items()
    # 平行：QD 与 BC 的叉积为 0
    ux, uy = xy[0] - CO["D"][0], xy[1] - CO["D"][1]
    wx, wy = CO["C"][0] - CO["B"][0], CO["C"][1] - CO["B"][1]
    assert abs(ux * wy - uy * wx) / (math.hypot(ux, uy) * math.hypot(wx, wy)) < 1e-9
    # 交点真的在 CA 上：CA 的两段加起来等于 CA
    on_ca = (math.dist(CO["C"], xy) + math.dist(xy, CO["A"])
             - math.dist(CO["C"], CO["A"]))
    assert abs(on_ca) < 1e-9
    assert plan.facts[0][0] == "para", plan.facts


# ---------------------------------------------------------------- 角平分线


def test_bisector_splits_the_angle_in_two():
    """作 ∠A 的平分线交 BC 于 H，则 ∠BAH = ∠HAC。"""
    plans = reason._aux_bisector(triangle(), GeoSolution(ok=True, coords=CO))
    assert plans, "没作出角平分线"
    plan = plans[0]
    (h, xy), = plan.pts.items()
    ba = (CO["B"][0] - CO["A"][0], CO["B"][1] - CO["A"][1])
    ca = (CO["C"][0] - CO["A"][0], CO["C"][1] - CO["A"][1])
    ha = (xy[0] - CO["A"][0], xy[1] - CO["A"][1])
    deg = lambda p, q: math.degrees(math.acos(
        (p[0] * q[0] + p[1] * q[1]) / (math.hypot(*p) * math.hypot(*q))))
    assert abs(deg(ba, ha) - deg(ha, ca)) < 1e-6, (deg(ba, ha), deg(ha, ca))
    # 交点落在 BC 上
    on_bc = (math.dist(CO["B"], xy) + math.dist(xy, CO["C"])
             - math.dist(CO["B"], CO["C"]))
    assert abs(on_bc) < 1e-9


# -------------------------------------------------------------------- 翻折


def test_reflect_puts_the_image_on_the_far_side():
    """把 D 沿 CA 翻折：折痕垂直平分 DD′。"""
    plans = reason._aux_reflect(triangle(), GeoSolution(ok=True, coords=CO))
    assert plans, "没作出对称点"
    plan = plans[0]
    kinds = {k for k, _ in plan.facts}
    assert "right" in kinds and "mid" in kinds, plan.facts
    # 对称点与 D 到折痕的距离相等
    d, img = CO["D"], None
    for n, xy in plan.pts.items():
        if math.dist(xy, d) > 1e-6:
            img = xy
    assert img is not None
    a, c = CO["A"], CO["C"]
    def _dist_to_line(p):
        ux, uy = c[0] - a[0], c[1] - a[1]
        return abs(ux * (p[1] - a[1]) - uy * (p[0] - a[0])) / math.hypot(ux, uy)
    assert abs(_dist_to_line(d) - _dist_to_line(img)) < 1e-9


# -------------------------------------------------------------------- 旋转


def equilateral_with_point():
    """正三角形 ABC 内一点 P，PA=3、PB=4、PC=5。坐标是解出来的一组真值。"""
    A, P = (3.0, 0.0), (0.0, 0.0)
    th = math.radians(-60)
    dx, dy = P[0] - A[0], P[1] - A[1]
    p2 = (A[0] + dx * math.cos(th) - dy * math.sin(th),
          A[1] + dx * math.sin(th) + dy * math.cos(th))
    u = (p2[0] / 3.0, p2[1] / 3.0)
    perp = (-u[1], u[0])
    C = (p2[0] + 4 * perp[0], p2[1] + 4 * perp[1])
    th2 = math.radians(60)
    vx, vy = C[0] - A[0], C[1] - A[1]
    B = (A[0] + vx * math.cos(th2) - vy * math.sin(th2),
         A[1] + vx * math.sin(th2) + vy * math.cos(th2))
    co = {"A": A, "B": B, "C": C, "P": P}
    p = Problem(
        problem_text="正三角形ABC内有一点P，PA=3，PB=4，PC=5，求∠APB的度数。",
        given=["△ABC是等边三角形", "PA=3", "PB=4", "PC=5"],
        goal="求∠APB的度数", points=["A", "B", "C", "P"],
        segments=[("A", "B"), ("B", "C"), ("C", "A"),
                  ("P", "A"), ("P", "B"), ("P", "C")],
    )
    s = GeoSolution(ok=True, coords=co, goal_kind="angle", goal_name="APB")
    # 先确认这组坐标真是题里那个图
    assert abs(math.dist(A, P) - 3) < 1e-6
    assert abs(math.dist(B, P) - 4) < 1e-6
    assert abs(math.dist(C, P) - 5) < 1e-6
    return p, s


def test_rotation_solves_the_3_4_5_inscribed_point():
    """旋转 60° 把 PA、PB、PC 凑成一个三角形：3²+4²=5² ⇒ 直角 ⇒ ∠APB=150°。"""
    p, s = equilateral_with_point()
    pr = reason.prove(p, s)
    assert pr.ok, pr.note
    assert "旋转" in pr.aux, pr.aux
    t = reason.render(pr, s)
    assert "勾股定理的逆定理" in t, t
    assert "∠APB = 150°" in t, t


def test_rotation_marks_itself_as_a_transform():
    """旋转出来的板书要说「辅助变换」，不能谎称「图上本来没有这条线」。"""
    p, s = equilateral_with_point()
    t = reason.render(reason.prove(p, s), s)
    assert "辅助变换" in t, t


# ------------------------------------------------------------ 假关系要挡住


def test_false_construction_is_rejected():
    """辅助线声称的平行其实不平行——这条辅助线必须整条丢弃。"""
    p = triangle()
    s = GeoSolution(ok=True, coords=CO)
    bad = reason.Aux(
        desc="过 D 作 DH ∥ BC，交 CA 于 H",
        pts={"H": (0.4, 1.2)},
        segs=(("D", "H"),),
        rels=("H在CA上",),
        # DH 的实际方向是 (-1.2, 1.2)，和 BC 根本不平行
        facts=(("para", (("D", "H"), ("A", "B"))),),
    )
    pr = reason._attempt(p, s, None, aux=bad)
    assert not pr.ok
    assert "不符" in pr.note, pr.note


def test_fact_holds_checks_each_kind():
    """验算函数本身：对的放行，错的拦下。"""
    co = {"A": (0, 0), "B": (4, 0), "C": (0, 3)}
    p = Problem(problem_text="x", points=["A", "B", "C"],
                segments=[("A", "B"), ("B", "C"), ("C", "A")])
    ctx = reason.Ctx(p, co)
    assert reason._fact_holds(ctx, "right", (reason._ang("A", "B", "C"),))
    assert not reason._fact_holds(ctx, "right", (reason._ang("B", "A", "C"),))
    assert reason._fact_holds(ctx, "eqseg", (("A", "B"), ("B", "C"))) is False
    # AC 与 AB 在 A 处成直角，显然不平行
    assert reason._fact_holds(ctx, "para", (("A", "C"), ("A", "B"))) is False
    assert reason._fact_holds(ctx, "eqang",
                              (reason._ang("B", "A", "C"),
                               reason._ang("A", "B", "C"))) is False
