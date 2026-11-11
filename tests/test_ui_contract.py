"""界面契约测试：组件的 value 必须落在 choices 里。

真踩过这个坑——`Cluster` 存的是规整后的 LaTeX，候选框列的是原始字符串，
两者对不上时 Gradio 6 在 preprocess 阶段直接抛 Error，而且一抛就把
页面上所有输出组件都刷成红色"错误"，看起来像全线崩了，实际只是单选组
判了个值不合法。这条测试专门守这个边界。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mathagent.types import (  # noqa: E402
    Candidate,
    ParsedExpr,
    PipelineResult,
    SolveResult,
)
from mathagent.verify import verify  # noqa: E402

# 两路对同一道定积分的不同写法：一个是规整版，一个带多余空格
CANDS = [
    Candidate(latex=r"\int_{0}^{2} \sqrt{2x - x^{2}} \, dx", source="qwen3-vl"),
    Candidate(latex=r"\int_0^2 \sqrt{2x - x^2} dx", source="glm-4.6v-flash"),
]


def _result(chosen_override: str | None = None) -> PipelineResult:
    v = verify(list(CANDS))
    if chosen_override is not None:
        v.chosen = chosen_override
    return PipelineResult(
        recognition=list(CANDS),
        verification=v,
        parsed=ParsedExpr(latex=v.chosen, normalized=v.chosen, kind="expression", ok=True),
        solved=SolveResult(ok=True, kind="expression"),
    )


def test_chosen_keeps_original_text():
    """聚类不能改写原文，否则 chosen 就回不到候选列表里。"""
    v = verify(list(CANDS))
    raws = {c.latex for c in CANDS}
    assert v.chosen in raws, f"chosen={v.chosen!r} 不是任何原始候选，界面必崩"


def test_radio_value_in_choices():
    """正常路径：value 必须能在 choices 的取值里找到。"""
    import app

    u = app.cand_update(_result())
    values = [c[1] for c in u.get("choices", [])]
    assert u.get("value") in values, f"value={u.get('value')!r} 不在 {values!r}"


def test_radio_survives_foreign_value():
    """极端路径：上游塞了个不在候选里的值时，要自动补成选项而不是抛错。"""
    import app

    foreign = r"\int_{0}^{2} \sqrt{2x - x^{2}} dx"  # 规整版，故意与原文不同
    u = app.cand_update(_result(chosen_override=foreign))
    values = [c[1] for c in u.get("choices", [])]
    assert foreign in values, "应把当前式作为选项补进去"
    assert u.get("value") in values, "补完之后 value 仍必须合法"


if __name__ == "__main__":
    test_chosen_keeps_original_text()
    test_radio_value_in_choices()
    test_radio_survives_foreign_value()
    print("界面契约测试全部通过")
