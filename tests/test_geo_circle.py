"""圆与辅助线的测试。

补这一层之前，圆的规则只有一条「直径所对的圆周角是直角」，而且那条还
永远不触发——`diam` 这个事实从来没有人往库里写。圆题全靠坐标法兜底，
学生看到的是「C(0.821, 0.383)」这种东西。

这里守五件事：
  1. 圆周角、圆心角、弦切角三条角度定理都通，且方向对（不能把补角当答案）
  2. 圆内接四边形对角互补、切线长相等、垂径定理都能用
  3. 直径必须是量出来的，不能拿「圆心 + 前两个圆周点」硬认
  4. 辅助线不止「连两点」：作垂线、取中点、延长都要造得出、坐标算得对
  5. 圆题的整条链要能走通，而不是退回坐标法

为了只测推理、不被求解器干扰，这里直接给 GeoSolution 塞入已知坐标——
坐标就是一组合法的解，验算那一步照常执行。
"""

from __future__ import annotations

import math

import pytest

from geoagent import reason, solve
from geoagent.schema import Circle, Problem
from geoagent.solve import GeoSolution


def pt(deg: float, r: float = 1.0) -> tuple[float, float]:
    a = math.radians(deg)
    return (r * math.cos(a), r * math.sin(a))


def scaled(co: dict, a: str, b: str, want: float) -> dict:
    """把整张图放缩到 |ab| = want。

    题目给的数字和坐标必须一致，否则「等量代换」那条最终验算会过不去——
    而它过不去是对的：图上量出来 1.7、题上说 4，本来就有一边错了。
    """
    cur = math.dist(co[a], co[b])
    k = want / cur
    return {n: (x * k, y * k) for n, (x, y) in co.items()}


def text_of(p: Problem, s: GeoSolution) -> str:
    pr = reason.prove(p, s)
    assert pr.ok, pr.note
    return reason.render(pr, s)


# ------------------------------------------------------------------ 圆周角


def test_center_angle_is_twice_inscribed():
    """∠AOB = 80° ⇒ ∠ACB = 40°。"""
    co = {"O": (0, 0), "A": pt(0), "B": pt(80), "C": pt(200)}
    p = Problem(
        problem_text="在⊙O中，点A、B、C在圆上，∠AOB=80°，求∠ACB的度数。",
        given=["∠AOB = 80°"], goal="求∠ACB的度数",
        points=["O", "A", "B", "C"],
        segments=[("O", "A"), ("O", "B"), ("O", "C"), ("A", "B"), ("A", "C"), ("B", "C")],
        circles=[Circle("O", ["A", "B", "C"])],
    )
    s = GeoSolution(ok=True, coords=co, goal_kind="angle", goal_name="ACB")
    t = text_of(p, s)
    assert "圆周角是圆心角的一半" in t, t
    assert "∠ACB = 40°" in t, t


def test_same_arc_angles_are_equal():
    """A、B 定下一条弦，C、D 在同一个弧上，两个圆周角相等。"""
    co = {"O": (0, 0), "A": pt(0), "B": pt(90), "C": pt(180), "D": pt(250)}
    p = Problem(
        problem_text="A、B、C、D 都在⊙O上。求证：∠ACB=∠ADB。",
        goal="求证：∠ACB=∠ADB",
        points=["O", "A", "B", "C", "D"],
        segments=[("O", "A"), ("A", "C"), ("C", "B"), ("A", "D"), ("D", "B"), ("A", "B")],
        circles=[Circle("O", ["A", "B", "C", "D"])],
    )
    s = GeoSolution(ok=True, coords=co, goal_kind="prove", claim_name="∠ACB=∠ADB")
    t = text_of(p, s)
    assert "同弧所对的圆周角相等" in t, t


def test_cyclic_quadrilateral_opposite_angles_supplement():
    """圆内接四边形，∠A = 95° ⇒ ∠C = 85°。"""
    co = {"O": (0, 0), "A": pt(0), "B": pt(60), "C": pt(140), "D": pt(250)}
    p = Problem(
        problem_text="四边形ABCD内接于⊙O，∠A=95°，求∠C的度数。",
        given=["∠A=95°"], goal="求∠C的度数",
        points=["O", "A", "B", "C", "D"],
        segments=[("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("O", "A"), ("O", "B")],
        circles=[Circle("O", ["A", "B", "C", "D"])],
    )
    s = GeoSolution(ok=True, coords=co, goal_kind="angle", goal_name="C",
                    goal_arms=("B", "D"))
    t = text_of(p, s)
    assert "∠BCD = 85°" in t, t


# ------------------------------------------------------------------ 切线


def test_tangent_lengths_are_equal():
    """PA、PB 是过 P 的两条切线 ⇒ PA = PB，已知 PA = 4 就得到 PB = 4。"""
    co = {"O": (0, 0), "A": pt(60), "B": pt(-60), "P": (2.0, 0.0)}
    co = scaled(co, "P", "A", 4.0)
    p = Problem(
        problem_text="PA、PB 是⊙O的切线，A、B 为切点，PA=4，求 PB。",
        given=["PA=4"], goal="求 PB 的长",
        points=["O", "A", "B", "P"],
        segments=[("O", "A"), ("O", "B"), ("P", "A"), ("P", "B")],
        circles=[Circle("O", ["A", "B"])],
    )
    s = GeoSolution(ok=True, coords=co, goal_kind="length", goal_name="PB")
    t = text_of(p, s)
    assert "切线长相等" in t, t
    assert "PB = 4" in t, t


def test_tangent_chord_angle_carries_the_angle_to_the_circle():
    """弦切角定理：切线 CP 与弦 CB 的夹角，等于 CB 所对的圆周角 ∠BAC。"""
    c = pt(200)
    tangent = (c[0] + 2 * math.cos(math.radians(110)),
               c[1] + 2 * math.sin(math.radians(110)))
    co = {"O": (0, 0), "A": pt(0), "B": pt(100), "C": c, "P": tangent}
    p = Problem(
        problem_text="CP 是⊙O的切线，切点为 C，AB 是⊙O的直径，求∠BCP。",
        given=["∠BAC = 50°"], goal="求∠BCP的度数",
        points=["O", "A", "B", "C", "P"],
        segments=[("A", "B"), ("A", "C"), ("B", "C"), ("O", "C"), ("C", "P")],
        circles=[Circle("O", ["A", "B", "C"])],
    )
    s = GeoSolution(ok=True, coords=co, goal_kind="angle", goal_name="BCP")
    t = text_of(p, s)
    assert "弦切角" in t, t
    assert "50" in t, t


# ------------------------------------------------------------------ 垂径定理


def test_perpendicular_diameter_bisects_the_chord():
    """AB 是直径、CD ⊥ AB 于 E ⇒ CE = ED。"""
    h = math.sqrt(3) / 2
    co = {"O": (0, 0), "A": (-1, 0), "B": (1, 0),
          "C": (0.5, h), "D": (0.5, -h), "E": (0.5, 0.0)}
    co = scaled(co, "C", "E", 3.0)
    p = Problem(
        problem_text="AB 是⊙O的直径，CD⊥AB于E，CE=3，求 ED 的长。",
        given=["AB是⊙O的直径", "CE=3"], goal="求 ED 的长",
        points=["O", "A", "B", "C", "D", "E"],
        segments=[("A", "B"), ("C", "D"), ("O", "E"), ("C", "E"), ("E", "D")],
        relations=["E在AB上"],
        circles=[Circle("O", ["A", "B", "C", "D"])],
    )
    s = GeoSolution(ok=True, coords=co, goal_kind="length", goal_name="ED")
    t = text_of(p, s)
    assert "垂径定理" in t, t
    assert "ED = 3" in t, t


def test_diameter_is_measured_not_assumed():
    """三个圆周点里有对径的一对才认直径，别的一律不认。

    早期实现是 `through[0]` 和 `through[1]` 直接当直径，于是「A、B、C 在圆上」
    这种最普通的条件会被记成 A、O、B 三点共线，图形立刻自相矛盾。
    """
    co = {"O": (0, 0), "A": pt(0), "B": pt(80), "C": pt(200)}
    p = Problem(
        problem_text="在⊙O中，点A、B、C在圆上。",
        points=["O", "A", "B", "C"],
        segments=[("O", "A"), ("O", "B"), ("O", "C")],
        circles=[Circle("O", ["A", "B", "C"])],
        pos={"O": [50, 50], "A": [90, 50], "B": [57, 11], "C": [12, 36]},
    )
    groups = solve.collinear_groups(p)
    assert not any({"O", "A", "B"} == g for g in groups), groups

    # 真正的直径要认：A、C 分居 O 两侧
    p2 = Problem(
        problem_text="AB 是⊙O的直径。",
        points=["O", "A", "B"],
        segments=[("A", "B")],
        circles=[Circle("O", ["A", "B"])],
        pos={"O": [50, 50], "A": [10, 50], "B": [90, 50]},
    )
    assert any({"O", "A", "B"} == g for g in solve.collinear_groups(p2))


# ------------------------------------------------------------------ 辅助线


def _triangle() -> Problem:
    return Problem(
        problem_text="在△ABC中。",
        points=["A", "B", "C"],
        segments=[("A", "B"), ("B", "C"), ("C", "A")],
    )


def test_aux_foot_lands_on_the_base():
    """过 C 作 AB 的垂线，垂足要真的落在 AB 上，坐标是投影。"""
    co = {"A": (0, 0), "B": (4, 0), "C": (1, 3)}
    plans = reason._aux_foot(_triangle(), GeoSolution(ok=True, coords=co))
    assert plans, "没作出高"
    plan = next(x for x in plans if "过 C" in x.desc)
    assert "⊥ AB" in plan.desc, plan.desc
    (h, xy), = plan.pts.items()
    assert abs(xy[0] - 1) < 1e-6 and abs(xy[1]) < 1e-6, xy
    assert plan.rights == ((h, "C", "A"),), plan.rights


def test_aux_midpoint_is_the_average():
    """取 AB 的中点再连到 C——中线。"""
    co = {"A": (0, 0), "B": (4, 0), "C": (1, 3)}
    plans = reason._aux_mid(_triangle(), GeoSolution(ok=True, coords=co))
    assert plans, "没造出中线"
    plan = next(x for x in plans if x.mids and x.mids[0][1] == "AB")
    (m, xy), = plan.pts.items()
    assert abs(xy[0] - 2) < 1e-6 and abs(xy[1]) < 1e-6, xy
    assert f"连接 C{m}" in plan.desc, plan.desc


def test_aux_extend_meets_outside():
    """梯形两腰延长后相交——交点要在两条边的外面。"""
    co = {"A": (0, 0), "B": (4, 0), "C": (3, 2), "D": (1, 2)}
    p = Problem(
        problem_text="梯形ABCD中，AD 与 BC 不平行。",
        points=["A", "B", "C", "D"],
        segments=[("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")],
    )
    plans = reason._aux_extend(p, GeoSolution(ok=True, coords=co))
    assert plans, "没找到延长线"
    plan = plans[0]
    assert "延长" in plan.desc and "相交于" in plan.desc, plan.desc
    (g, xy), = plan.pts.items()
    assert xy[1] > 2.5, xy          # 交点在两腰上方


def test_auxiliary_line_is_reported_in_the_writeup():
    """真用了辅助线，就要写在板书最前面，并且说清作的是什么。"""
    p = Problem(
        problem_text="正方形ABCD中，E是BC上一点，F是CD上一点，BE=CF。求证：△ABE≌△BCF。",
        given=["四边形ABCD为正方形", "BE = CF"],
        goal="求证：△ABE≌△BCF",
        points=["A", "B", "C", "D", "E", "F"],
        segments=[("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")],
        relations=["E在BC上", "F在CD上"],
        pos={"A": [20, 20], "B": [80, 20], "C": [80, 80],
             "D": [20, 80], "E": [80, 50], "F": [50, 80]},
    )
    sol = solve.solve(p)
    assert sol.ok, sol.error
    pr = reason.prove(p, sol)
    assert pr.ok and pr.aux
    t = reason.render(pr, sol)
    assert "辅助线" in t and "△ABE ≌ △BCF" in t, t


# ------------------------------------------------------------------ 端到端


def test_circle_problem_is_solved_by_theorems_not_coordinates():
    """g2（直径 + 切线 + ∠A=25°）要走出「弦切角 → 平角 → 内角和」这条链。

    以前这道题退回坐标法，讲解里是「CP=0.596、BP=0.278，由余弦定理…」。
    """
    from test_geo import g2

    p = g2()
    sol = solve.solve(p)
    assert sol.ok, sol.error
    pr = reason.prove(p, sol)
    assert pr.ok, pr.note
    t = reason.render(pr, sol)
    assert "弦切角" in t, t
    assert "平角" in t, t
    assert "40" in t, t
    assert "余弦定理" not in t, t
    assert t.count("**∴**") <= 8, "步骤太碎，学生看不到主线"
