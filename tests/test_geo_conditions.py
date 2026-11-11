"""条件解析的三条底线。

这三个 bug 的共同点：**报错信息都把责任推给用户**（「多半是读错了，请核对
识别结果」），而实际上读得没错，是系统自己没把条件用上。用户照着提示去
重拍一张，还是同样的结果——这类错误比崩溃更难查。

1. `∠PBA = ∠PDA` 这种「只给相等、不给度数」的条件，既不进数值约束，也不
   进知识库。少一条约束，那个点就变成自由的，结论于是「不成立」。
2. 待证的结论和条件长得一模一样（`求证：∠PAB = ∠PCB`），扫条件时被
   一起扫进去，等于用结论证明结论——永远成立，也永远看不出错在哪。
3. 「P 在 ABCD 内部」被当成修饰语丢掉。P 跑到形外后，同一组条件照样成立，
   但那时量到的是补角，两个度数加起来正好 180°。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoagent import explain, pipeline, solve
from geoagent.schema import Problem


def parallelogram() -> Problem:
    """截图里那道题：P 在平行四边形 ABCD 内，∠PBA = ∠PDA，求证 ∠PAB = ∠PCB。"""
    return Problem(
        problem_text="设P是平行四边形ABCD内部的一点，且∠PBA=∠PDA。求证：∠PAB=∠PCB。",
        given=["四边形ABCD是平行四边形", "∠PBA=∠PDA", "P在平行四边形ABCD内部"],
        goal="求证：∠PAB=∠PCB",
        points=["A", "B", "C", "D", "P"],
        segments=[
            ("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"),
            ("P", "A"), ("P", "B"), ("P", "C"), ("P", "D"),
        ],
        # 原图：A 左上、B 左下、C 右下、D 右上、P 居中偏右
        pos={"A": (30, 25), "B": (12, 70), "C": (70, 80), "D": (88, 30), "P": (55, 50)},
    )


def test_angle_equal_condition_reaches_the_model():
    """∠PBA = ∠PDA 必须进约束，否则 P 是自由点。"""
    pairs = solve._angle_pairs_given(parallelogram())
    assert ("PBA", "PDA") in pairs


def test_claim_is_not_taken_as_a_condition():
    """待证的 ∠PAB = ∠PCB 不能被当条件——那是循环论证。

    `used` 是「本题用到了哪些条件」，它会把结论列进去，说明求解时真的
    按结论建了约束，于是怎么解都成立。
    """
    res = pipeline.resolve(parallelogram())
    # 只看「条件：」这一行。目标、结论两行本来就该出现结论，不能混为一谈。
    cond = next(s for s in res.solution.steps if s.startswith("条件："))
    assert "∠PBA = ∠PDA" in cond
    assert "∠PAB = ∠PCB" not in cond


def test_problem_is_solved():
    res = pipeline.resolve(parallelogram())
    assert res.solution.ok, res.solution.error
    assert "成立" in res.solution.answer


def test_inside_condition_keeps_the_point_inside():
    """漏掉内部约束时，P 会跑到形外，量到的是补角（和恒为 180°）。"""
    res = pipeline.resolve(parallelogram())
    X = res.solution.coords
    assert X, "没有坐标"
    assert explain._inside_poly(X, "P", ["A", "B", "C", "D"]), (
        f"P 跑到形外了：{X}"
    )


def test_redraw_keeps_the_original_layout():
    """重绘图要长得像原图，否则用户没法拿它对照原图找错。

    只比形状不比大小：两边各自按自身跨度归一，再看点距的偏差。实测过的
    两种塌法——平行四边形被压成一条线、B 从 A 的左下翻到右上——都会让
    这个偏差明显变大。
    """
    p = parallelogram()
    res = pipeline.resolve(p)
    X = res.solution.coords
    # 阈值放宽到 0.15：原图 pos 是目测值，本身就不是精确的平行四边形
    # （实测 A(30,25)、B(12,70)、D(88,30) 画出来的 AB 和 DC 差 5 个点），
    # 这里要抓的是「塌」而不是「有点不像」。
    import numpy as np

    score = solve.shape_score(p, {k: np.array(v) for k, v in X.items()}, 1.0)
    assert score < 0.15, f"重绘形状偏离原图太多（score={score:.3f}）：{X}"

    # 没塌：AB 和 AD 不能共线（压扁时叉积趋 0）
    ab = (X["B"][0] - X["A"][0], X["B"][1] - X["A"][1])
    ad = (X["D"][0] - X["A"][0], X["D"][1] - X["A"][1])
    area = abs(ab[0] * ad[1] - ab[1] * ad[0])
    assert area > 0.2, f"平行四边形被压扁了（面积 {area:.3f}）：{X}"

    # 象限关系也得对：D 在 A 的右边，B 在 A 的下方
    assert X["D"][0] > X["A"][0], "D 跑到 A 左边了"
    assert X["B"][1] < X["A"][1], "B 跑到 A 上方了"


def test_angle_claim_gets_an_angle_conclusion():
    """角相等题不能套用边相等的讲法。"""
    res = pipeline.resolve(parallelogram())
    text = res.explain.text
    assert res.explain.ok
    assert "两个角相等" in text
    assert "余弦定理" in text
    assert "∠PAB = ∠PCB 成立" in text
    # 条件表里只能出现题给的角相等，不能出现结论
    table = text.split("逐条核对题干条件")[1].split("####")[0]
    assert "∠PBA = ∠PDA" in table
    assert "∠PAB = ∠PCB" not in table


def test_failure_speaks_human_not_log():
    """解不出来时给诊断，不给日志。

    漏掉 ∠PBA = ∠PDA 就复现了截图里的情形：结论「不成立」。以前界面会把
    「读入 5 个点」「不成立 / 不成立 / 不成立」倒出来。
    """
    p = parallelogram()
    p.problem_text = "设P是平行四边形ABCD内部的一点。求证：∠PAB=∠PCB。"
    p.given = ["四边形ABCD是平行四边形", "P在平行四边形ABCD内部"]
    res = pipeline.resolve(p)

    assert not res.solution.ok
    text = res.explain.text
    assert not res.explain.ok
    assert "没做出来" in text and "卡在哪" in text
    # 日志式的说法一个都不能漏到界面上
    for log in ("读入", "补全：", "残差"):
        assert log not in text, f"日志词语漏出来了：{log}"
    # 诊断要指向真正的原因
    assert "只给相等、不给度数" in text
