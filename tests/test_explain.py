"""讲评层的拦截规则测试（不调接口，纯本地）。

这层是整个项目"算与讲分离"的落点：大模型只讲思路，一旦写了
自己算的等式就必须拦下。拦错（误伤合法讲解）同样算 bug，
所以正例负例一起测。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mathagent.explain import _math_part, _refs, check  # noqa: E402
from mathagent.latex import parse  # noqa: E402
from mathagent.solve import solve  # noqa: E402

FAILED = 0


def expect(cond: bool, label: str) -> None:
    global FAILED
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILED += 1


def main() -> int:
    pe = parse("x^2-5x+6=0")
    sr = solve(pe)
    pool = [pe.latex] + [s.latex for s in sr.steps] + sr.solutions
    refs = _refs(pe, sr)

    print("片段抽取")
    expect(_math_part("把它拆成 (x-2)(x-3)=0 就容易看出两根") == "(x-2)(x-3)=0", "从中文里抠出算式")

    print("\n应放行（合法讲法）")
    allow = [
        "把它拆成 (x-2)(x-3)=0 就容易看出两根。",
        "两边同时除以 a，得 x=3。",
        "原式可以写成 1=1，显然成立。",
        "这道题考的是因式分解，先凑两个数相乘为积、相加为一次项系数。",
    ]
    for t in allow:
        ok, why = check(t, pool, refs)
        expect(ok, f"放行：{t[:22]}　{why}")

    print("\n应拦截（自己算了 / 编了答案）")
    block = [
        "得到 2+2=5，所以答案错了。",
        "所以两根是 x=4 和 x=5。",
        "因为 \\frac{1}{2}=0.5，所以答案是 3。",
    ]
    for t in block:
        ok, _ = check(t, pool, refs)
        expect(not ok, f"拦截：{t[:22]}")

    print("\n无参考式时（refs 为空）仍能拦住凭空等式")
    ok, _ = check("把它拆成 (x-2)(x-3)=0 就容易看出两根。", pool, [])
    expect(not ok, "没有原式参照时不放行")

    print("\n" + ("全部通过" if FAILED == 0 else f"{FAILED} 项未通过"))
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
