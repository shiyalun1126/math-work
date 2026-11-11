"""全等、相似、辅助线的测试。

补这一层之前，推理引擎只会「平行 → 角等 → 等角对等边」这一路。初中几何
真正的主干是另外两条：全等（SSS/SAS/ASA/AAS/HL）和相似（AA）。没有它们，
「求证 △ABE ≌ △BCF」「求证 △ADE ∽ △ABC」这类题一步都走不动——判定不了，
也就拿不到对应边、对应角。

这里守五件事：
  1. 判定要认出来，而且判定名要对（写 SAS 的地方不能报 HL）
  2. 判完全等要能取用对应边、对应角——判定本身不是目的
  3. 公共边（△ABC 与 △ADC 共用 AC）可以算条件，但不能只靠它凑数
  4. 图上少了一条线时，补一条真能走通的辅助线，并把它说出来
  5. 假结论照样推不出来
"""

from __future__ import annotations

from geoagent import reason, solve
from geoagent.schema import Mark, Problem


def square() -> Problem:
    """正方形 ABCD，E 在 BC 上、F 在 CD 上，BE = CF。

    图中画了 AE、BF，于是 △ABE 与 △BCF 都在图上：
    AB = BC（正方形）、BE = CF（已知）、∠ABE = ∠BCF = 90°——SAS。
    """
    return Problem(
        problem_text="正方形ABCD中，E是BC上一点，F是CD上一点，BE=CF。求证：△ABE≌△BCF。",
        given=["四边形ABCD为正方形", "BE = CF"],
        goal="求证：△ABE≌△BCF",
        points=["A", "B", "C", "D", "E", "F"],
        segments=[("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"),
                  ("A", "E"), ("B", "F")],
        relations=["E在BC上", "F在CD上"],
        pos={"A": [20, 20], "B": [80, 20], "C": [80, 80],
             "D": [20, 80], "E": [80, 50], "F": [50, 80]},
    )


def parallel_tri() -> Problem:
    """△ABC 中 D 在 AB 上、E 在 AC 上，DE ∥ BC。

    这是相似里最基础的一张图：同位角两次，AA 判定。
    """
    return Problem(
        problem_text="在△ABC中，D是AB上一点，E是AC上一点，DE∥BC。求证：△ADE∽△ABC。",
        given=["DE∥BC"],
        goal="求证：△ADE∽△ABC",
        points=["A", "B", "C", "D", "E"],
        segments=[("A", "B"), ("B", "C"), ("C", "A"), ("D", "E"),
                  ("A", "D"), ("A", "E")],
        relations=["D在AB上", "E在AC上"],
        pos={"A": [50, 10], "B": [15, 80], "C": [85, 80],
             "D": [32, 45], "E": [68, 45]},
    )


def isosceles() -> Problem:
    """等腰三角形，D 是底边中点，求证 ∠ADB = ∠ADC。

    三线合一的起点：△ABD 与 △ACD 共用 AD，SSS 成立，斜边/公共边那条
    规矩得认，否则这条最常考的证明反而推不出来。
    """
    return Problem(
        problem_text="在△ABC中，AB=AC，D是BC的中点。求证：∠ADB=∠ADC。",
        given=["AB=AC", "D是BC的中点"],
        goal="求证：∠ADB=∠ADC",
        points=["A", "B", "C", "D"],
        segments=[("A", "B"), ("B", "C"), ("C", "A"), ("A", "D")],
        marks=[Mark("midpoint", "D", ["BC"]), Mark("equal_length", "", ["AB", "AC"])],
        pos={"A": [50, 10], "B": [20, 80], "C": [80, 80], "D": [50, 80]},
    )


def text(p: Problem) -> str:
    sol = solve.solve(p)
    assert sol.ok, sol.error
    return reason.render(reason.prove(p, sol), sol)


def test_congruence_by_sas():
    """SAS 要报 SAS，不能说成 HL——直角确实有，但相等的两条边是直角边。"""
    t = text(square())
    assert "△ABE ≌ △BCF" in t, t
    assert "边角边（SAS）" in t, t
    assert "斜边、直角边" not in t, t


def test_congruence_gives_corresponding_parts():
    """判完全等要用得上：结论改成 AE = BF，靠“全等三角形对应边相等”。"""
    p = square()
    p.goal = "求证：AE=BF"
    p.problem_text = "正方形ABCD中，E是BC上一点，F是CD上一点，BE=CF。求证：AE=BF。"
    t = text(p)
    assert "全等三角形的对应边相等" in t, t
    assert "AE = BF" in t, t


def test_common_side_counts_as_a_condition():
    """△ABD 与 △ACD 共用 AD：这条公共边得算，而且要写出来。"""
    t = text(isosceles())
    assert "公共边" in t, t
    assert "边边边（SSS）" in t, t
    assert "∠ADB = ∠ADC" in t


def test_similarity_by_aa_from_parallel_lines():
    """DE∥BC 出两组同位角，AA 得相似。"""
    t = text(parallel_tri())
    assert "同位角" in t, t
    assert "∽" in t and "两角分别相等" in t, t


def test_similarity_gives_the_missing_length():
    """相似比落到数上：AD=2、AB=4、DE=3，则 BC=6。"""
    p = parallel_tri()
    p.given = ["DE∥BC", "AD=2", "AB=4", "DE=3"]
    p.problem_text = "在△ABC中，D是AB上一点，E在AC上，DE∥BC，AD=2，AB=4，DE=3。求BC的长。"
    p.goal = "求 BC 的长"
    t = text(p)
    assert "相似三角形对应边成比例" in t, t
    assert "BC = 6" in t, t


def test_auxiliary_line_is_found_and_named():
    """图上没画 AE、BF 时，推理会卡住——补上这两条线就能走通，得说出来。"""
    p = square()
    p.segments = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    sol = solve.solve(p)
    assert sol.ok, sol.error
    pr = reason.prove(p, sol)
    assert pr.ok, pr.note
    assert pr.aux, "补了辅助线却没说"
    t = reason.render(pr, sol)
    assert "辅助线" in t and "连接" in t, t
    assert "△ABE ≌ △BCF" in t, t


def test_false_congruence_is_rejected():
    """假结论不能凑：△ABE 与 △DCF 并不全等，推不出来就如实说推不出来。"""
    p = square()
    p.goal = "求证：△ABE≌△DCF"
    p.problem_text = "正方形ABCD中，E是BC上一点，F是CD上一点，BE=CF。求证：△ABE≌△DCF。"
    sol = solve.solve(p)
    pr = reason.prove(p, sol)
    assert not pr.ok
    assert pr.note


def test_solid_geometry_is_refused_not_guessed():
    """立体几何直接说做不了。硬按平面算，给出的会是「看着很像那么回事」的错答案。"""
    from geoagent import pipeline

    p = Problem(
        problem_text="正方体ABCD-A1B1C1D1中，求异面直线AB与CC1所成的角。",
        goal="求角", points=["A", "B"], segments=[("A", "B")],
    )
    r = pipeline.resolve(p)
    assert r.verdict == "fail"
    assert "立体几何" in r.error and "平面" in r.error
    assert r.solution.ok is False
