"""冒烟测试：解析 / 校验 / 求解三段，不依赖任何 API。

跑法：F:/PyCharm/python.exe tests/test_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mathagent.latex import normalize, parse  # noqa: E402
from mathagent.verify import verify  # noqa: E402
from mathagent.solve import solve  # noqa: E402
from mathagent.types import Candidate  # noqa: E402

CASES = [
    (r"x^2-5x+6=0", "equation", True),
    (r"\frac{x+1}{x-2}=3", "equation", True),
    (r"\int x e^x dx", "expression", True),
    (r"\frac{1}{2}x", "expression", True),
    (r"\begin{cases} x+y=3 \\ x-y=1 \end{cases}", "system", True),
    (r"\times^2-3x+2=0", "unknown", False),
    (r"x^2-3x+2=", "equation", True),
    ("", "unknown", False),
]


def test_parse():
    """用 assert 而不是 return：pytest 只看断言，`return False` 也会被记成通过。

    以前这里写的是 `return ok == len(CASES)`，于是就算整张表全错，
    测试仍然是绿的——这种"假绿"比没有测试更危险。
    """
    bad = []
    for tex, kind, expect_ok in CASES:
        pe = parse(tex)
        good = pe.ok == expect_ok and (not expect_ok or pe.kind == kind)
        flag = "PASS" if good else "FAIL"
        print(f"[{flag}] {tex!r:46} ok={pe.ok} kind={pe.kind:10} {pe.error[:50]}")
        if not good:
            bad.append((tex, pe.ok, pe.kind, pe.error))
    print(f"parse {len(CASES) - len(bad)}/{len(CASES)}\n")
    assert not bad, f"解析不符合预期：{bad}"


def test_normalize():
    got = normalize(r"\left( x+1 \right) \div 2 = 3。")
    want = "( x+1 ) / 2 = 3"
    print(f"normalize: {got!r} -> {'PASS' if got == want else 'FAIL ' + want!r}\n")
    assert got == want, f"归一化结果 {got!r}，期望 {want!r}"


def test_verify_vote():
    """三源两派：两个说同一个式子，一个说错了。应判 ok 且置信度 2/3 以上。"""
    cands = [
        Candidate(r"x^2-5x+6=0", "simpletex", 0.93),
        Candidate(r"x^{2}-5x+6=0", "qwen3-vl", 0.90),
        Candidate(r"x^2-5x+8=0", "qwen3-vl", 0.55),
    ]
    v = verify(cands)
    print(f"投票: verdict={v.verdict} conf={v.confidence:.2f} chosen={v.chosen!r}")
    print(f"      簇数={len(v.clusters)} 理由={v.reasons}")
    assert v.verdict in ("ok", "review"), v.verdict
    assert v.chosen.replace(" ", "") in (r"x^2-5x+6=0", r"x^{2}-5x+6=0"), v.chosen
    print("vote PASS\n")


def test_review_when_split():
    """1:1 平票必须判 review，而不是挑一个。"""
    cands = [
        Candidate(r"x^2-3x+2=0", "simpletex", 0.8),
        Candidate(r"x^2-3\times2=0", "qwen3-vl", 0.7),
    ]
    v = verify(cands)
    print(f"平票: verdict={v.verdict} conf={v.confidence:.2f} 理由={v.reasons}")
    # 这条断言以前不存在：函数只 print 不 assert，所以平票判错也一直是绿的。
    assert v.verdict == "review", f"1:1 平票应判 review，实际 {v.verdict}"
    print("review PASS\n")


def test_solve():
    expect = {
        r"x^2-5x+6=0": {"x = 2", "x = 3"},
        r"\begin{cases} x+y=3 \\ x-y=1 \end{cases}": {"x = 2，y = 1"},
    }
    for tex in (r"x^2-5x+6=0", r"\frac{x+1}{x-2}=3", r"\int x e^x dx",
                r"\begin{cases} x+y=3 \\ x-y=1 \end{cases}"):
        pe = parse(tex)
        sr = solve(pe)
        sols = ", ".join(sr.solutions) or sr.error
        print(f"solve {tex!r:44} ok={sr.ok} -> {sols}")
        for st in sr.steps:
            print(f"        · {st.title}: {st.latex}")
        assert sr.ok, f"{tex} 应该能解出来，实际：{sr.error}"
        if tex in expect:
            assert set(sr.solutions) == expect[tex], f"{tex} 解集是 {sr.solutions}"
    print()


if __name__ == "__main__":
    for fn in (test_normalize, test_parse, test_verify_vote, test_review_when_split, test_solve):
        fn()
        print(f"{fn.__name__} 通过")
    print("全部通过")
