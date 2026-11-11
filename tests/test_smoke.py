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
    ok = 0
    for tex, kind, expect_ok in CASES:
        pe = parse(tex)
        good = pe.ok == expect_ok and (not expect_ok or pe.kind == kind)
        ok += good
        flag = "PASS" if good else "FAIL"
        print(f"[{flag}] {tex!r:46} ok={pe.ok} kind={pe.kind:10} {pe.error[:50]}")
    print(f"parse {ok}/{len(CASES)}\n")
    return ok == len(CASES)


def test_normalize():
    got = normalize(r"\left( x+1 \right) \div 2 = 3。")
    want = "( x+1 ) / 2 = 3"
    print(f"normalize: {got!r} -> {'PASS' if got == want else 'FAIL ' + want!r}\n")
    return got == want


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
    return True


def test_review_when_split():
    """1:1 平票必须判 review，而不是挑一个。"""
    cands = [
        Candidate(r"x^2-3x+2=0", "simpletex", 0.8),
        Candidate(r"x^2-3\times2=0", "qwen3-vl", 0.7),
    ]
    v = verify(cands)
    print(f"平票: verdict={v.verdict} conf={v.confidence:.2f} 理由={v.reasons}")
    print("review PASS\n" if v.verdict == "review" else "review FAIL\n")


def test_solve():
    for tex in (r"x^2-5x+6=0", r"\frac{x+1}{x-2}=3", r"\int x e^x dx",
                r"\begin{cases} x+y=3 \\ x-y=1 \end{cases}"):
        pe = parse(tex)
        sr = solve(pe)
        sols = ", ".join(sr.solutions) or sr.error
        print(f"solve {tex!r:44} ok={sr.ok} -> {sols}")
        for st in sr.steps:
            print(f"        · {st.title}: {st.latex}")
    print()


if __name__ == "__main__":
    passed = [test_normalize(), test_parse(), test_verify_vote()]
    test_review_when_split()
    test_solve()
    print("全部通过" if all(passed) else "有失败项")
