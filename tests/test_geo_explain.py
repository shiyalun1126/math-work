"""解题过程的测试：给学生看的东西必须是解题，不是系统日志。

用户先后提了三条意见：
  1. 「让你给解题过程，不是把题干条件再罗列一遍」
  2. 「你不知道他们都是多长，这都是你假设的」——坐标法设 AB=1 是凭空来的数字
  3. 「几何题尽量还是用几何思维，谁平行，哪个角相等，用推理的形式」

所以现在证明题和常规求值题都先走 `reason.py` 的定理推理，坐标法退为兜底。
这里守四件事：
  1. 输出是 ∵∴ 推理，每条结论挂着定理名
  2. 长度符号化（a、√2a），不出现「设 AB = 1 所以 E 在 (0.366, 1.366)」
  3. 坐标法仍可用（force_coords），但只在推不出来时出场
  4. 解不出来就不编
"""

from __future__ import annotations

from geoagent import explain, reason, solve
from geoagent.schema import Mark, Problem


def g1() -> Problem:
    """直角三角形 + 斜边中点，求 CD。几何推理两步就该出结果。"""
    return Problem(
        given=["∠C = 90°", "AC = 6", "BC = 8", "D 是 AB 的中点"],
        goal="求 CD 的长",
        points=["A", "B", "C", "D"],
        segments=[("A", "B"), ("A", "C"), ("B", "C"), ("C", "D")],
        marks=[Mark("midpoint", "D", ["AB"]), Mark("right_angle", "C", ["AC", "BC"])],
    )


def g4() -> Problem:
    """直角三角形给两条直角边，求斜边。"""
    return Problem(
        given=["∠C = 90°", "AC = 3", "BC = 4"],
        goal="求 AB 的长",
        points=["A", "B", "C"],
        segments=[("A", "B"), ("A", "C"), ("B", "C")],
        marks=[Mark("right_angle", "C", ["AC", "BC"])],
    )


def g5() -> Problem:
    """正方形 ABCD，DE∥AC，AE=AC，F 为 AE 与 CD 的交点，求证 CE=CF。"""
    return Problem(
        problem_text="四边形ABCD为正方形，DE∥AC，AE=AC。AE与CD相交于F。求证：CE=CF。",
        given=["四边形ABCD为正方形", "DE∥AC", "AE=AC"],
        goal="求证：CE=CF",
        points=["A", "B", "C", "D", "E", "F"],
        segments=[
            ("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("A", "F"),
            ("B", "C"), ("C", "D"), ("C", "E"), ("C", "F"), ("D", "E"),
        ],
        relations=["F在AE上，也在CD上"],
        pos={"A": [20, 20], "B": [20, 80], "C": [80, 80],
             "D": [80, 20], "E": [100, 50], "F": [70, 50]},
    )


def test_proof_is_geometric_reasoning_not_coordinates():
    """证明题要走定理，不走坐标——用户原话是「用几何思维，谁平行，哪个角相等」。"""
    sol = solve.solve(g5())
    assert sol.ok, sol.error
    t = explain.write(g5(), sol).text
    for must in ("∵", "∴", "内错角", "等角对等边", "解题过程"):
        assert must in t, f"推理链里缺「{must}」：\n{t}"
    for forbid in ("建系", "(0.", "√[(1"):
        assert forbid not in t, f"坐标法混进了证明过程：{forbid}\n{t}"
    assert "CE = CF" in t


def test_lengths_are_symbolic_never_invented():
    """题目没给长度时，长度必须写成 a 的倍数，不能出现「设 AB = 1」。"""
    t = explain.write(g5(), solve.solve(g5())).text
    assert "都用 **a** 表示" in t, t
    assert "√2a" in t
    assert "设 AB = 1" not in t


def test_every_step_carries_a_theorem():
    """每一步都得说清用了哪条定理，不能只有「∴ 某个数」。"""
    t = explain.write(g5(), solve.solve(g5())).text
    steps = [x for x in t.split("\n") if x.strip().startswith("**∴**")]
    assert len(steps) >= 10, t
    for s in steps:
        assert "（" in s and "）" in s, f"这步没写依据：{s}"


def test_proof_names_parallel_when_it_uses_it():
    """用到了 DE∥AC，就得在链条里说出来；不能只写角度数值。"""
    t = explain.write(g5(), solve.solve(g5())).text
    assert "平行" in t or "∥" in t, t


def test_short_problem_stays_short():
    """g4 一步勾股就够，不该为了「详细」凑出十几步。"""
    t = explain.write(g4(), solve.solve(g4())).text
    assert "勾股定理" in t
    assert t.count("**∴**") <= 3, t
    assert "AB = 5" in t


def test_median_theorem_for_right_triangle():
    """g1 两步：勾股求斜边，斜边中线等于斜边一半。"""
    t = explain.write(g1(), solve.solve(g1())).text
    assert "勾股定理" in t
    assert "斜边" in t and "中点" in t
    assert "CD = 5" in t
    assert t.count("**∴**") <= 4, t


def test_coords_still_work_as_fallback():
    """坐标法没删，只是退到兜底；强制走它时仍然能出完整过程。"""
    p = g1()
    t = explain.write(p, solve.solve(p), force_coords=True).text
    for must in ("解题过程", "建系", "坐标", "\\sqrt", "结论"):
        assert must in t, f"兜底过程里缺「{must}」：\n{t}"
    for forbid in ("读入", "残差", "补全：", "初值"):
        assert forbid not in t, f"系统日志「{forbid}」混进了解题过程：\n{t}"


def test_base_edge_is_never_the_goal():
    """坐标法里，求 AB 就不能拿 AB 当 x 轴，否则「设 AB = 5」等于先写答案。"""
    p = g4()
    sol = solve.solve(p)
    assert sol.ok, sol.error
    t = explain.write(p, sol, force_coords=True).text
    assert "AC = 3 是题目给的" in t, t
    assert "AB = 5 是题目给的" not in t


def test_coords_are_centered_and_right_way_up():
    """A 在原点，基准边落在 x 轴上，图不能上下颠倒。"""
    p = g1()
    sol = solve.solve(p)
    X, origin, base, base_len = explain._frame(p, sol.coords)
    assert origin == "A" and base == "C" and base_len == 6
    assert X["A"] == (0.0, 0.0)
    assert abs(X["C"][1]) < 1e-9
    assert X["B"][1] > 0


def test_no_solution_means_no_story():
    """解不出来就不编解题过程——但也不能一言不发。

    以前这里返回空字符串，界面于是退回 `sol.steps`（「读入 5 个点」「不成立」
    「不成立」）。那是排查日志，学生看完只知道「它没做出来」，不知道该怎么办。
    现在改成诊断：不给假推导，但说清读到了什么、卡在哪、先核对哪三件事。
    """
    p = Problem(points=["A", "B"], segments=[("A", "B")], goal="求 AB 的长")
    e = explain.write(p, solve.GeoSolution())
    assert not e.ok
    # 不能有「因为…所以…」的假推导，也不能有结论
    assert "余弦定理" not in e.text
    assert "#### 结论" not in e.text
    # 但要把话说清楚
    assert "没做出来" in e.text
    assert "卡在哪" in e.text
    assert e.note


def test_engine_never_fabricates_a_false_conclusion():
    """假结论推不出来，不能硬凑。推理引擎的底线是「推不出就说推不出」。"""
    p = g5()
    p.goal = "求证：CE=CD"
    p.problem_text = "四边形ABCD为正方形，DE∥AC，AE=AC。AE与CD相交于F。求证：CE=CD。"
    sol = solve.solve(p)
    pr = reason.prove(p, sol)
    assert not pr.ok
    assert pr.note
