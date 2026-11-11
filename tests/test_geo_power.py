"""圆幂定理（相交弦、割线、切割线）的测试。

这三条是「四点共圆」最常用的入口，题面往往只给三个长度求第四个：
    PA·PB = PC·PD
以前没有它们，这类题只能退回坐标法，学生看到的是「P(0,0)、A(-2,0)」，
而不是「由相交弦定理得 PD = 4」。

守四件事：
  1. 点在圆内写「相交弦定理」，在圆外写「割线定理」，名字不能混
  2. 切线 + 割线走的是 PT² = PA·PB，未知量出现两次要开方，不能当除法
  3. 题给数字与图形对不上时，结论必须被丢弃——宁可说推不出来
  4. 定理名要出现在板书里，学生才知道这一步凭什么

坐标都是直接算好的合法解，只测推理、不依赖求解器。
"""

from __future__ import annotations

import math

from geoagent import reason
from geoagent.schema import Circle, Problem
from geoagent.solve import GeoSolution


def circum(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
          + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
          + (cx**2 + cy**2) * (bx - ax)) / d
    return (ux, uy)


def text_of(p: Problem, s: GeoSolution) -> str:
    pr = reason.prove(p, s)
    assert pr.ok, pr.note
    return reason.render(pr, s)


def chords(inside: bool):
    """两条弦交于 P：PA·PB = 12，PC·PD = 12。

    inside=True 时 P 在圆内（相交弦），否则 P 在圆外（两条割线）。
    """
    th = math.radians(60)
    if inside:
        P = (0.0, 0.0)
        A, B = (-2.0, 0.0), (6.0, 0.0)
        C = (-3 * math.cos(th), -3 * math.sin(th))
        D = (4 * math.cos(th), 4 * math.sin(th))
        text = "AB、CD 是⊙O的两条弦，交于点P，PA=2，PB=6，PC=3，求PD。"
        goal, name = "求 PD 的长", "PD"
        segs = [("A", "B"), ("C", "D"), ("O", "A")]
    else:
        # P 在圆外：两条割线都从 P 出发
        P = (7.0, 0.0)
        A, B = (5.0, 0.0), (2.0, 0.0)          # PA=2, PB=5 → 积 10
        C = (7 - 2.5 * math.cos(th), -2.5 * math.sin(th))
        D = (7 - 4 * math.cos(th), -4 * math.sin(th))   # PC=2.5, PD=4 → 积 10
        text = "过⊙O外一点P作两条割线PAB、PCD，PA=2，PB=5，PC=2.5，求PD。"
        goal, name = "求 PD 的长", "PD"
        segs = [("A", "B"), ("C", "D"), ("O", "A")]
    O = circum(A, B, C)
    co = {"P": P, "A": A, "B": B, "C": C, "D": D, "O": O}
    p = Problem(
        problem_text=text,
        given=[f"PA={math.dist(P, A):g}", f"PB={math.dist(P, B):g}",
               f"PC={math.dist(P, C):g}"],
        goal=goal, points=["O", "P", "A", "B", "C", "D"],
        segments=segs, circles=[Circle("O", ["A", "B", "C", "D"])],
    )
    s = GeoSolution(ok=True, coords=co, goal_kind="length", goal_name=name)
    return p, s


def test_intersecting_chords():
    """相交弦：PA=2、PB=6、PC=3 ⇒ PD=4。"""
    p, s = chords(inside=True)
    t = text_of(p, s)
    assert "相交弦定理" in t, t
    assert "PD = 4" in t, t


def test_secants_from_outside_point():
    """割线：圆外一点引两条割线，PA·PB = PC·PD ⇒ PD=4。"""
    p, s = chords(inside=False)
    t = text_of(p, s)
    assert "割线定理" in t, t
    assert "割线定理" in t and "相交弦" not in t, "点在圆外不能叫相交弦"
    assert "PD = 4" in t, t


def test_tangent_squared_equals_product():
    """切割线：PT² = PA·PB。PT=4、PA=2 ⇒ PB=8。"""
    O = (0.0, 0.0)
    R = 3.0
    P = (5.0, 0.0)
    T = (R * R / 5, math.sqrt(R * R - (R * R / 5) ** 2))
    A, B = (3.0, 0.0), (-3.0, 0.0)      # 直径方向：PA=2、PB=8
    co = {"O": O, "P": P, "T": T, "A": A, "B": B}
    p = Problem(
        problem_text="PT 是⊙O的切线，T 为切点，PAB 是割线。PT=4，PA=2，求PB。",
        given=["PT=4", "PA=2"], goal="求 PB 的长",
        points=["O", "P", "T", "A", "B"],
        segments=[("P", "T"), ("P", "A"), ("O", "T"), ("A", "B")],
        circles=[Circle("O", ["T", "A", "B"])],
    )
    s = GeoSolution(ok=True, coords=co, goal_kind="length", goal_name="PB")
    t = text_of(p, s)
    assert "切割线定理" in t, t
    assert "PT²" in t, f"切线那条应该是平方，不是 PT·PT 写成一次：{t}"
    assert "PB = 8" in t, t
    assert "PT = 4；PT = 4" not in t, "同一条边不该在同一个 ∵ 里出现两遍"


def test_bad_numbers_are_rejected():
    """题上写 PA=2，图上量出来 6 倍——这一步必须推不出来，不能凑。"""
    p, s = chords(inside=True)
    p.given = ["PA=2", "PB=6", "PC=3"]
    p.goal = "求 PD 的长"
    s.goal_name = "PD"
    # 把 P 挪到别处，圆幂等式不再成立
    bad = dict(s.coords)
    bad["P"] = (0.9, 0.4)
    pr = reason.prove(p, GeoSolution(ok=True, coords=bad,
                                     goal_kind="length", goal_name="PD"))
    assert not pr.ok, "图形对不上时不该给结论"


def test_power_of_point_facts_are_normalized():
    """PA·PB = PC·PD 与 PB·PA = PD·PC 是同一条，不能存成两条。"""
    a = reason._norm("prod", (("AB", "CD"), ("EF", "GH")))
    b = reason._norm("prod", (("GH", "EF"), ("DC", "AB")))
    assert a == b, (a, b)
