"""几何链路的离线测试：不调 API，全部用构造好的结构直接验证。

重点不是「能不能算对一道题」，而是「算不出来的时候会不会老实说」。
"""

from __future__ import annotations

import json
import math

import pytest

from geoagent import crosscheck, relations, render, solve
from geoagent.read import parse_json
from geoagent.schema import Circle, Mark, Problem


def g1() -> Problem:
    """直角三角形 + 斜边中点，CD = 5。"""
    return Problem(
        given=["∠C = 90°", "AC = 6", "BC = 8", "D 是 AB 的中点"],
        goal="求 CD 的长",
        points=["A", "B", "C", "D"],
        segments=[("A", "B"), ("A", "C"), ("B", "C"), ("C", "D")],
        marks=[Mark("midpoint", "D", ["AB"]), Mark("right_angle", "C", ["AC", "BC"])],
    )


def g2() -> Problem:
    """圆 + 直径 + 切线，∠P = 40°。

    注意直角标的是 ∠ACB —— 这是直径所对的圆周角，图上确实有这个直角，
    但切线真正的条件是「半径 ⊥ 切线」，只能从题面文字补出来。
    这个用例专门守这件事。
    """
    return Problem(
        given=[
            "AB是⊙O的直径",
            "点C在圆上",
            "过C作⊙O的切线，交AB的延长线于P",
            "∠A = 25°",
        ],
        goal="求∠P的度数",
        points=["A", "B", "C", "O", "P"],
        segments=[("A", "B"), ("A", "C"), ("B", "C"), ("O", "C"), ("C", "P")],
        circles=[Circle(center="O", through=["A", "B", "C"])],
        marks=[Mark("right_angle", "C", ["AC", "BC"])],
    )


# ------------------------------------------------------------------ 解析


def test_parse_json_tolerates_fence_and_prefix():
    raw = '好的，结果如下：\n```json\n{"points": ["A", "B"], "goal": "求AB"}\n```'
    obj = parse_json(raw)
    assert obj["points"] == ["A", "B"]


def test_parse_json_repairs_truncated_output():
    """模型在 max_tokens 处被截断是常态，必须能救回来。"""
    raw = '{"problem_text":"如图","points":["A","B","C"],"segments":[["A","B"]'
    obj = parse_json(raw)
    assert obj["points"] == ["A", "B", "C"]


def test_parse_json_rejects_think_only():
    with pytest.raises(ValueError):
        parse_json("<think>我正在思考这道题该怎么读……")


# ------------------------------------------------------------------ 结构


def test_normative_makes_two_channels_comparable():
    a = Problem(points=["B", "A"], segments=[("B", "A")], marks=[Mark("right_angle", "C", ["BC", "AC"])])
    b = Problem(points=["A", "B"], segments=[("A", "B")], marks=[Mark("right_angle", "C", ["AC", "BC"])])
    assert a.normative().points == b.normative().points
    assert a.normative().segments == b.normative().segments
    assert [m.key for m in a.normative().marks] == [m.key for m in b.normative().marks]


def test_mark_of_normalizes_single_letters():
    """一个模型给 ["AC","BC"]，另一个给 ["A","B"]，要归成同一个东西。"""
    p = Problem.from_dict(
        {
            "points": ["A", "B", "C"],
            "marks": [{"type": "right_angle", "at": "C", "of": ["A", "B"]}],
        }
    )
    assert p.marks[0].of == ["AB"]


# ------------------------------------------------------------------ 校验


def test_crosscheck_confirms_text_against_figure():
    checks = crosscheck.run(g1())
    levels = [c.level for c in checks]
    assert "conflict" not in levels
    titles = " ".join(c.title for c in checks)
    assert "90°" in titles
    assert "中点" in titles


def test_crosscheck_flags_wrong_right_angle():
    """题干说 C 是直角，图上却标在 A —— 必须报冲突，不能放过。"""
    p = g1()
    p.marks = [Mark("right_angle", "A", ["AB", "AC"]), Mark("midpoint", "D", ["AB"])]
    checks = crosscheck.run(p)
    assert any(c.level == "conflict" for c in checks)


def test_crosscheck_notes_figure_supplied_condition():
    """g2 的直角是图上才有的（切线垂直于半径），应提示为「补充」而不是冲突。"""
    p = Problem(
        given=["AB是⊙O的直径", "∠A = 25°"],
        goal="求∠P",
        points=["A", "B", "C", "O", "P"],
        segments=[("A", "B"), ("O", "C"), ("C", "P")],
        circles=[Circle(center="O", through=["A", "B"])],
        marks=[Mark("right_angle", "C", ["OC", "CP"])],
    )
    checks = crosscheck.run(p)
    assert any(c.level in ("info", "ok") for c in checks)
    assert not any(c.level == "conflict" for c in checks)


# ------------------------------------------------------------------ 求解


def test_solve_right_triangle_median():
    r = solve.solve(g1())
    assert r.ok, r.error
    assert r.value == pytest.approx(5.0, abs=1e-3)
    assert r.unique


def test_solve_circle_tangent_angle():
    r = solve.solve(g2())
    assert r.ok, r.error
    assert r.value == pytest.approx(40.0, abs=0.01)


def test_solve_refuses_when_underdetermined():
    """去掉 BC=8，CD 就不再唯一，必须说「定不下来」而不是随便给个数。"""
    p = g1()
    p.given = ["∠C = 90°", "AC = 6", "D 是 AB 的中点"]
    r = solve.solve(p)
    assert not r.ok
    assert "不唯一" in r.error or "条件不足" in r.error


def test_solve_detects_contradiction():
    """三个条件凑在一起是矛盾的，不能硬解出一个图形。"""
    p = g1()
    p.given = ["∠C = 90°", "AC = 6", "BC = 8", "AB = 8", "D 是 AB 的中点"]
    r = solve.solve(p)
    assert not r.ok
    assert "矛盾" in r.error


def test_solve_never_collapses_points():
    """防退化：不能把所有点缩到原点蒙混过关。"""
    r = solve.solve(g1())
    assert r.ok
    pts = list(r.coords.values())
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = ((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2) ** 0.5
            assert d > 0.5, f"第 {i} 和第 {j} 个点几乎重合：{d}"


# ------------------------------------------------------------------ 渲染


def test_render_runs_with_and_without_coords():
    img1 = render.render(g1())
    assert img1.size == (520, 520)
    r = solve.solve(g1())
    img2 = render.render(g1(), r.coords)
    assert img2.size == (520, 520)
    assert img1.tobytes() != img2.tobytes()


def test_render_handles_empty_figure():
    img = render.render(Problem())
    assert img.size == (520, 520)


# ------------------------------------------------------------------ 端到端结构


def test_pipeline_dict_is_json_serializable():
    from geoagent.pipeline import resolve

    res = resolve(g1())
    payload = json.dumps(res.to_dict(), ensure_ascii=False)
    assert "CD" in payload
    assert res.verdict == "ok"


def test_single_source_caps_confidence():
    """只有一路来源时，置信度必须被压到 75% 以下——和公式链路同一条规矩。"""
    from geoagent.pipeline import resolve

    res = resolve(g1())
    assert res.confidence <= 0.75


# ------------------------------------------------------------------ 重绘纠错


def test_repair_moves_right_angle_onto_radius_and_tangent():
    """切点处的直角必须落在「半径 ⊥ 切线」上。

    模型常把它读成两条弦的夹角（∠ACB）——那个角恰好也是 90°，
    等式检验永远发现不了，只有对着原图才看得出方框标错了位置。
    """
    p = g2()
    notes = solve.repair_marks(p)
    pairs = {"".join(sorted(s)) for m in p.marks if m.type == "right_angle" for s in m.of}
    assert pairs == {"CO", "CP"}, pairs
    assert notes and "切线" in notes[0]


def test_repair_leaves_figures_without_tangent_alone():
    p = g1()
    assert solve.repair_marks(p) == []
    assert p.marks[1].of == ["AC", "BC"]


def test_seed_from_pos_keeps_original_orientation():
    p = g2()
    p.pos = {"A": [10, 60], "B": [80, 60], "C": [45, 10], "O": [45, 60], "P": [100, 30]}
    x = solve.seed_from_pos(p, 1.0)
    d = {n: (x[2 * i], x[2 * i + 1]) for i, n in enumerate(p.points)}
    assert d["A"][0] < d["B"][0]          # A 在左、B 在右，和原图一致
    assert d["C"][1] > d["A"][1]          # 数学坐标里 C 在 AB 上方


def test_align_to_pos_rotates_without_breaking_geometry():
    coords = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}
    pos = {"A": [10, 90], "B": [90, 90], "C": [10, 10]}
    out = solve.align_to_pos(coords, pos)
    for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
        assert abs(math.dist(coords[a], coords[b]) - math.dist(out[a], out[b])) < 1e-6
    assert out["C"][1] > out["A"][1]      # 摆回原图朝向：C 在上


def test_extra_links_connect_point_on_extension():
    """「P 在 AB 的延长线上」要真的连成一条线，否则 P 会悬空。"""
    p = g2()
    pos = {"A": (0, 0), "B": (1, 0), "C": (0.5, 0.8), "O": (0.5, 0), "P": (1.6, 0)}
    links = {tuple(sorted(x)) for x in render._extra_links(p, pos)}
    assert ("B", "P") in links


def test_solve_records_goal_so_it_can_be_drawn():
    """目标角要记下两条边，重绘才知道该标哪个角。"""
    sol = solve.solve(g2())
    assert sol.ok and sol.goal_kind == "angle" and sol.goal_name == "P"
    assert sol.goal_arms and set(sol.goal_arms) == {"C", "B"}
    img = render.render(g2(), sol.coords, goal=("angle", "P", sol.goal_arms), value=sol.value)
    assert img.size == (520, 520)


# ------------------------------------------------------------------ 题干整句声明


def g5(claim: str = "CE=CF", extra: list[str] | None = None) -> Problem:
    """正方形 ABCD，DE∥AC，AE=AC，AE 与 CD 相交于 F，求证 CE=CF。

    这题暴露过三个问题，每个都有一个测试守着：
      - 「四边形 ABCD 为正方形」不在图上画任何符号，模型只把它抄进 given，
        结果正方形退化成矩形，CE 变成一段区间（实测 1.379~19.95）
      - 目标是「求证」不是「求值」，硬按求值算就报「条件不足」
      - F 的位置只写在「AE 与 CD 相交于 F」里，漏了它 CF 就不定
    """
    return Problem(
        problem_text=f"四边形ABCD为正方形，DE∥AC，AE=AC。AE与CD相交于F。求证：{claim}。",
        given=["四边形ABCD为正方形", "DE∥AC", "AE=AC"] + (extra or []),
        goal=f"求证：{claim}",
        points=["A", "B", "C", "D", "E", "F"],
        segments=[
            ("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("A", "F"),
            ("B", "C"), ("C", "D"), ("C", "E"), ("C", "F"), ("D", "E"),
        ],
        relations=["F在AE上，也在CD上"],
        pos={"A": [20, 20], "B": [20, 80], "C": [80, 80],
             "D": [80, 20], "E": [100, 50], "F": [70, 50]},
    )


def test_square_declaration_becomes_constraints():
    """「四边形ABCD为正方形」必须翻成四直角 + 四边等，不能只停在 given 里。"""
    d = relations.derive(g5())
    right = {m.at for m in d.marks if m.type == "right_angle"}
    assert right == {"A", "B", "C", "D"}, right
    assert sum(m.type == "equal_length" for m in d.marks) >= 3
    assert any("正方形" in n for n in d.notes)


def test_relations_reads_point_on_line_and_between():
    """「F 在 AE 上，也在 CD 上」要读成两条共线，而且 F 得在两端之间。"""
    d = relations.derive(g5())
    assert {"A", "E", "F"} in d.lines
    assert {"C", "D", "F"} in d.lines
    assert ("F", "A", "E") in d.between


def test_solve_proves_conclusion_instead_of_computing_a_number():
    """证明题不该给一个数，而该说结论成立——并且给出数值依据。"""
    r = solve.solve(g5())
    assert r.ok, r.error
    assert r.goal_kind == "prove"
    assert "成立" in r.answer
    assert any("CE=" in s and "CF=" in s for s in r.steps)


def test_proof_does_not_assume_what_it_should_prove():
    """待证结论被抄进 given 时不能当条件用，否则什么结论都「成立」。"""
    bad = solve.solve(g5(claim="CE=CD", extra=["CE=CD"]))
    assert not bad.ok
    assert "不符" in bad.error


def test_underdetermined_problem_still_has_coords():
    """条件不足时可以不给答案，但坐标得有——否则重绘只能画个环形示意。

    以前 coords 只在「目标算出来了」时才存，于是条件不足的题一张图都画
    不出来。可这明明是两件事：图建得出来（残差 0），只是答案定不下来。
    """
    p = g1()
    p.given = ["∠C = 90°", "AC = 6", "D 是 AB 的中点"]  # 少了 BC=8，CD 就不定
    r = solve.solve(p)
    assert not r.ok
    assert r.coords, "条件不足时也要能重绘"


def test_render_uses_pos_layout_when_no_coords():
    """没有坐标时按读到的位置摆图，绝不能退回环形布局。

    这条守的是用户最直观的那个抱怨：重绘图把 8 个点排到圆周上，
    A 在正上方、D 跑到底下、E 跑到左边，和原图毫无关系。
    """
    p = g5()
    layout = render.pos_layout(p, p.points)
    assert layout is not None
    assert layout["A"][0] < layout["C"][0]     # A 在 C 左边
    assert layout["A"][1] > layout["C"][1]     # 数学坐标里 A 在上面
    assert render.render(p).size == (520, 520)


def test_render_marks_both_sides_of_a_proof():
    r = solve.solve(g5())
    img = render.render(g5(), r.coords, goal=("compare", r.claim_name, r.claim_edges))
    assert img.size == (520, 520)
    assert r.claim_edges == ("CE", "CF")
