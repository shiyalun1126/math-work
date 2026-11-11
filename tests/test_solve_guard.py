"""求解的止损边界：该算的算，不该算的当场说清楚。

这里的用例守的是两件事：
1. 已知会炸的形态（嵌套根式）不能再进 SymPy 空转；
2. 判「超出范围」不等于判「识别错了」——前者不该让学生以为是自己写错。
"""

from __future__ import annotations

import time

import pytest

from mathagent.latex import parse
from mathagent.solve import solve

NESTED = r"y=\sqrt[3]{\sqrt{x^{2}+1}+x}+\sqrt[3]{x-\sqrt{x^{2}+1}}"


def test_nested_radical_is_out_of_scope():
    """这道题实测让 SymPy 跑了 79.7 秒，必须在预判阶段就拦下。"""
    pe = parse(NESTED)
    t0 = time.time()
    sr = solve(pe)
    dt = time.time() - t0

    assert sr.out_of_scope is True
    assert sr.ok is False
    assert "根号" in sr.error
    assert dt < 1.0, f"预判应该几乎瞬间返回，实际 {dt:.2f}s"


def test_ordinary_equations_unaffected():
    """拦住病态输入，不能顺手把正常题也拦了。"""
    for src, expect in (
        (r"x^{2}-5x+6=0", ["x = 2", "x = 3"]),
        (r"x^{2}-2mx+m^{2}-1=0", ["x = m - 1", "x = m + 1"]),
    ):
        sr = solve(parse(src))
        assert sr.out_of_scope is False
        assert sr.ok is True
        assert sr.solutions == expect

    # 单层根式属于范围内的题，不能被"嵌套"判据误伤
    sr = solve(parse(r"\sqrt{x+1}=2"))
    assert sr.out_of_scope is False
    assert sr.ok is True


def test_timeout_returns_scope_exit(monkeypatch):
    """预判挡不住的花样，由超时兜底，且退出时带明确原因而不是空结果。"""
    monkeypatch.setenv("SOLVE_TIMEOUT", "0.01")
    sr = solve(parse(r"\int \frac{1}{\sqrt{a^{2}-x^{2}}} dx"))

    assert sr.out_of_scope is True
    assert sr.ok is False
    assert "停" in sr.error or "秒" in sr.error


def test_out_of_scope_reported_in_dict():
    """落盘的 JSON 要能区分「算不出」和「超范围」，否则事后没法复盘。"""
    d = solve(parse(NESTED)).to_dict()
    assert d["out_of_scope"] is True
    assert d["ok"] is False


@pytest.mark.parametrize("bad", ["", r"\frac{1}{"])
def test_broken_input_still_reports_error(bad):
    sr = solve(parse(bad))
    assert sr.ok is False
    assert sr.error
