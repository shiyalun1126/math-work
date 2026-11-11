"""三元一次方程组：方程题型的覆盖面与步骤质量。

守两件事：
1. 三元题不能被叫成"二元"（标签按未知数个数走）；
2. 三元题的步骤必须是学生能看的那种（消元、代入、回代），
   而不是 CAS 的「方程 / 解得 / 代回验证 0=0」。
"""

from __future__ import annotations

import pytest

from mathagent import equations

CAS_MARKS = {"代回验证"}  # 出现这几个字说明退回了机器步骤


def steps_text(ans) -> str:
    return "\n".join(f"{s['title']}|{s['latex']}" for s in ans.steps)


def assert_student_steps(ans) -> None:
    assert ans.steps, "没有生成任何步骤"
    titles = [s["title"] for s in ans.steps]
    assert titles[0].startswith("题目给的三个条件"), f"第一步应是列出条件，实际：{titles[0]}"
    assert not (CAS_MARKS & set(titles)), f"退回了 CAS 步骤：{titles}"
    assert any("抵消" in t or "代" in t for t in titles), f"看不到消元或代入：{titles}"
    # 最后一步是校验，而且要真的说成立
    assert "成立" in ans.steps[-1]["latex"], ans.steps[-1]


CASES = {
    # 三条式子都含 z，两两相减正好各定住一个未知数
    "标准": (r"\begin{cases}x+y+z=6\\x-y+z=2\\2x+y-z=1\end{cases}", r"x = 1，y = 2，z = 3"),
    # 三角和：每个未知数只出现在两条式子里，消元只能得一条，要借原式补第二条
    "三角和": (r"\begin{cases}x+y=3\\y+z=5\\x+z=4\end{cases}", r"x = 1，y = 2，z = 3"),
    # 系数对不上，要两边各乘一个数
    "需倍乘": (
        r"\begin{cases}2x+3y+z=1\\x-2y+2z=3\\3x+y-z=2\end{cases}",
        r"x = 1，y = - \frac{1}{2}，z = \frac{1}{2}",
    ),
    # 分数系数
    "分数": (
        r"\begin{cases}\frac{x}{2}+y=4\\y+z=6\\x-z=0\end{cases}",
        r"x = 4，y = 2，z = 4",
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_system3_solves_and_teaches(name):
    src, want = CASES[name]
    ans = equations.answer(src, explain=False)

    assert ans.ok, f"{name} 没解出来：{ans.error}"
    assert ans.solutions == [want], f"{name} 解错了：{ans.solutions}"
    assert ans.topic.label == "三元一次方程组", f"{name} 标签不对：{ans.topic.label}"
    assert_student_steps(ans)


def test_system2_still_labelled_two():
    """加了三元不能把二元的标签带偏。"""
    ans = equations.answer(r"\begin{cases}x+y=5\\x-y=1\end{cases}", explain=False)
    assert ans.ok
    assert ans.topic.label == "二元一次方程组"
    assert ans.solutions == ["x = 3，y = 2"]


def test_unknown_count_drives_label():
    """四个未知数也不该被硬叫成两元或三元。"""
    src = r"\begin{cases}a+b=1\\a+c=2\\a+d=3\\b+c=4\end{cases}"
    ans = equations.answer(src, explain=False)
    # 解不出来也无所谓，标签必须按实际未知数个数走
    assert ans.topic.label.startswith("四元"), ans.topic.label


def test_underdetermined_system_does_not_crash():
    """两个方程三个未知数：解不唯一，但不能崩，也不能假装给出了唯一解。"""
    ans = equations.answer(r"\begin{cases}x+y+z=6\\x-y=0\end{cases}", explain=False)
    assert ans.topic.label == "三元一次方程组"
    # 参数化解里应当还留着字母，而不是编出一组数
    joined = " ".join(ans.solutions)
    assert "z" in joined, joined
