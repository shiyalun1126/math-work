"""含参讨论：把「关于 x 的方程」按参数分情形讨论。

为什么单独写一层。原来的求解器默认系数不为零——`ax²+bx+c=0` 直接套求根
公式，`a=0` 那一支悄悄丢了。学生拿到的答案看着对，实际上漏了一整类情形。
教学里这叫「含参分类讨论」，是中考压轴和高中必修的固定题型，漏一支就是
错答案，不是「不够详细」。

做法分三步：

  1. **按次数分情形**：二次项系数 A=0 与 A≠0，后者才是二次方程
  2. **按判别式分情形**：Δ>0 / Δ=0 / Δ<0
  3. **每一支都要有见证**：给参数找一组真取值，代进去确认这个情形真的
     存在、解真的成立。找不到取值的分支（比如 A=1 时还谈 A=0）直接删掉，
     不写进答案——这和几何那边「推不出来就说推不出来」是同一条规矩。

于是答案不是一句「x = …」，而是一张分情形的表。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import sympy
from sympy import Poly

from .latex import to_latex
from .types import Step

# 主元的候选。都不在时退化成「次数最高的那个符号」。
MAIN = ("x", "y", "z", "t", "u", "v", "w")

# 见证用的取值。取分数是有意的：整数试不出「系数为零导致降次」那些支。
SAMPLES = [sympy.Integer(k) for k in (-3, -2, -1, 0, 1, 2, 3)] + [
    sympy.Rational(1, 2), sympy.Rational(-1, 2),
]

TOL = 1e-6

_OP_LATEX = {"≠": r"\ne", "=": "=", ">": ">", "<": "<"}


@dataclass
class Cond:
    """一个条件：`expr op 0`。disp 是展示用的写法（Δ 要写成 Δ 而不是它的值）。"""

    expr: object
    op: str
    disp: object = None

    def _shown(self):
        return self.expr if self.disp is None else self.disp

    def has_param(self) -> bool:
        return bool(getattr(self.expr, "free_symbols", set()))

    def latex(self) -> str:
        return f"{_fmt(self._shown())} {_OP_LATEX[self.op]} 0"

    def plain(self) -> str:
        # 标题是给人读的纯文本，\Delta 要还原成 Δ，不能把 LaTeX 命令直接甩出来
        s = sympy.sstr(self._shown()).replace(r"\Delta", "Δ")
        return f"{s} {self.op} 0"

    def under(self, sym, val) -> "Cond":
        """把某个参数取定之后的条件（「a=0 时 a-1≠0」这种就化成常数了）。"""
        return Cond(sympy.simplify(self.expr.subs(sym, val)), self.op,
                    None if self.disp is None else sympy.simplify(self.disp.subs(sym, val)))


@dataclass
class Leaf:
    """一个情形。checks 是判定条件，sols 是这个情形下的解。"""

    checks: list
    conds: list = field(default_factory=list)
    sols: list = field(default_factory=list)
    kind: str = "roots"                          # roots / all / none
    none_text: str = "无解"


@dataclass
class Discussion:
    main: str = ""
    params: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    solutions: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _fmt(v) -> str:
    return to_latex(v)


def _paren(e) -> str:
    """和式、负数放进平方或乘法里必须加括号，否则 a-1 的平方会印成 a - 1²。"""
    s = _fmt(e)
    if getattr(e, "is_Atom", False):
        return rf"\left({s}\right)" if e.is_number and e.could_extract_minus_sign() else s
    return s if not e.is_Add else rf"\left({s}\right)"


def _pick_main(expr) -> tuple | None:
    """挑主元和参数。x 优先，其次次数最高的符号。"""
    syms = sorted(expr.free_symbols, key=str)
    if len(syms) < 2:
        return None
    for name in MAIN:
        for s in syms:
            if str(s) == name:
                return s, [t for t in syms if t != s]
    best, deg = syms[0], -1
    for s in syms:
        try:
            d = sympy.degree(expr, s)
        except Exception:  # noqa: BLE001
            d = 0
        if d > deg:
            best, deg = s, d
    return best, [t for t in syms if t != best]


def _witness(checks: list, params: list):
    """给这些条件找一组真实取值。找不到就说明这个情形根本不存在。

    「Δ=0」这类条件的解常常是 9/8 这种不在取样表上的数。只靠整数和两个
    半整数去试，「m=9/8 时重根」那一支就会被误判成不可能，白白漏掉。
    所以先把每个等式条件对每个参数解出来，把解填进取值池。
    """
    if not params:
        try:
            return {} if all(bool(c) for c in checks) else None
        except TypeError:
            return None

    pools: list[list] = []
    for p in params:
        pool = list(SAMPLES)
        for c in checks:
            try:
                sols = sympy.solve(c, p)
            except Exception:  # noqa: BLE001
                continue
            # 不等式条件会返回 And(...) 这种布尔对象，不能当列表遍历
            if not isinstance(sols, (list, tuple)):
                continue
            for s in sols:
                if not isinstance(s, sympy.Expr) or getattr(s, "free_symbols", None):
                    continue          # 还带着别的参数，等别的分支去试
                try:
                    pool.append(sympy.nsimplify(s))
                except Exception:  # noqa: BLE001
                    continue
        pools.append(pool[:16])

    for combo in itertools.product(*pools):
        sub = dict(zip(params, combo))
        try:
            if all(bool(c.subs(sub)) for c in checks):
                return sub
        except (TypeError, ValueError):
            continue
    return None


def _ok_residual(expr, main, sol, sub) -> bool:
    """解代回原方程，残差要真的接近 0。算不出数就算没通过。"""
    try:
        r = expr.subs(main, sol).subs(sub)
        return abs(complex(sympy.N(r, 20))) < TOL
    except Exception:  # noqa: BLE001
        return False


def _ok_all(expr, sub) -> bool:
    """「任意实数都是解」：整个式子恒等于 0 才成立。"""
    try:
        return sympy.simplify(expr.subs(sub)) == 0
    except Exception:  # noqa: BLE001
        return False


def _ok_none(expr, main, sub) -> bool:
    """「无解」：代进去之后剩下一个不为零的常数。"""
    try:
        p = Poly(sympy.expand(expr.subs(sub)), main)
    except Exception:  # noqa: BLE001
        return False
    if p.degree() == 0:
        return sympy.simplify(p.as_expr()) != 0
    return True


def _conds_latex(conds: list) -> str:
    return r" \text{ 且 } ".join(c.latex() for c in conds if c.has_param())


def _conds_plain(conds: list) -> str:
    return " 且 ".join(c.plain() for c in conds if c.has_param())


def _sub_text(sub: dict) -> str:
    if not sub:
        return "取一组特值"
    return "、".join(f"{k}={v}" for k, v in sub.items())


def _leaf_latex(lf: Leaf, main) -> str:
    if lf.kind == "all":
        return rf"{_fmt(main)} \text{{ 为任意实数}}"
    if lf.kind == "none":
        return rf"\text{{{lf.none_text}}}"
    return r" \text{ 或 } ".join(f"{_fmt(main)} = {_fmt(s)}" for s in lf.sols)


def _plain(title: str, latex: str) -> Step:
    return Step(title=title, latex=latex)


def _roots_of(expr, main, A, B, C):
    """先试 sympy 能不能给出一组干净的根，不行再用求根公式。"""
    try:
        got = [sympy.simplify(g) for g in sympy.solve(expr, main)]
        if got and all(g.is_real is not False for g in got):
            return got
    except Exception:  # noqa: BLE001
        pass
    disc = sympy.expand(B**2 - 4 * A * C)
    return [sympy.simplify((-B + sympy.sqrt(disc)) / (2 * A)),
            sympy.simplify((-B - sympy.sqrt(disc)) / (2 * A))]


def discuss(lhs, rhs) -> Discussion | None:
    """把方程按参数分类讨论。没有参数、或不是至多二次的多项式，返回 None。"""
    try:
        num, _den = sympy.fraction(sympy.together(lhs - rhs))
        expr = sympy.expand(num)
    except Exception:  # noqa: BLE001
        return None

    got = _pick_main(expr)
    if not got:
        return None
    main, params = got

    try:
        poly = Poly(expr, main)
        deg = poly.degree()
    except Exception:  # noqa: BLE001
        return None
    if deg < 1 or deg > 2:
        return None

    d = Discussion(main=str(main), params=[str(p) for p in params])
    d.steps.append(_plain("原方程", f"{_fmt(lhs)} = {_fmt(rhs)}"))
    if sympy.simplify(lhs - rhs) != sympy.simplify(expr):
        d.steps.append(_plain("去分母整理", f"{_fmt(expr)} = 0"))

    if deg == 1:
        A, B = poly.all_coeffs()
        rhs_c = sympy.simplify(-B)          # A·x = rhs_c，比 A·x + B = 0 好读
        leaves = [
            Leaf([sympy.Ne(A, 0)], [Cond(A, "≠")],
                 [sympy.simplify(rhs_c / A)]),
            Leaf([sympy.Eq(A, 0), sympy.Eq(rhs_c, 0)],
                 [Cond(A, "="), Cond(rhs_c, "=")], [], kind="all"),
            Leaf([sympy.Eq(A, 0), sympy.Ne(rhs_c, 0)],
                 [Cond(A, "="), Cond(rhs_c, "≠")], [], kind="none"),
        ]
    else:
        A, B, C = poly.all_coeffs()
        disc = sympy.expand(B**2 - 4 * A * C)
        d.steps.append(_plain(
            "判别式",
            rf"\Delta = {_paren(B)}^2 - 4 \cdot {_paren(A)} \cdot {_paren(C)}"
            rf" = {_fmt(disc)}",
        ))
        roots = _roots_of(expr, main, A, B, C)
        dcond = Cond(disc, ">", sympy.Symbol(r"\Delta"))
        leaves = [
            Leaf([sympy.Ne(A, 0), sympy.Gt(disc, 0)],
                 [Cond(A, "≠"), dcond], list(roots)),
            Leaf([sympy.Ne(A, 0), sympy.Eq(disc, 0)],
                 [Cond(A, "≠"), Cond(disc, "=", sympy.Symbol(r"\Delta"))],
                 [sympy.simplify(-B / (2 * A))]),
            # Δ<0 和「一次方程无解」是两回事，说法分开
            Leaf([sympy.Ne(A, 0), sympy.Lt(disc, 0)],
                 [Cond(A, "≠"), Cond(disc, "<", sympy.Symbol(r"\Delta"))],
                 [], kind="none", none_text="无实根"),
            Leaf([sympy.Eq(A, 0), sympy.Ne(B, 0)],
                 [Cond(A, "="), Cond(B, "≠")],
                 [sympy.simplify(-C / B)]),
            Leaf([sympy.Eq(A, 0), sympy.Eq(B, 0), sympy.Eq(C, 0)],
                 [Cond(A, "="), Cond(B, "="), Cond(C, "=")], [], kind="all"),
            Leaf([sympy.Eq(A, 0), sympy.Eq(B, 0), sympy.Ne(C, 0)],
                 [Cond(A, "="), Cond(B, "="), Cond(C, "≠")], [], kind="none"),
        ]
        # 「a=0 时，方程退化成 (a-1)x-1=0」——把 a=0 代进后继条件，
        # 不然答案里会留下一句「a=0 且 a-1≠0」，看着像没化简。
        # 只改展示，判定条件原样保留：见证还得靠它们。
        if isinstance(A, sympy.Symbol):
            for lf in leaves[3:]:
                lf.conds = [lf.conds[0]] + [c.under(A, 0) for c in lf.conds[1:]]
                # 解也要跟着取定：a=0 时答案是 -1，不该还挂着 a
                lf.sols = [sympy.simplify(s.subs(A, 0)) for s in lf.sols]

    kept: list[tuple[Leaf, dict]] = []
    for lf in leaves:
        sub = _witness(lf.checks, params)
        if sub is None:
            # 找不到取值，说明这些条件不可能同时成立（A 是常数 1 时就不存在
            # 「A=0」这种情形），不写进答案
            d.dropped.append(f"情形 {_conds_plain(lf.conds)}："
                             "不存在满足条件的参数取值")
            continue
        if lf.kind == "roots":
            if not lf.sols or not all(_ok_residual(expr, main, s, sub) for s in lf.sols):
                d.dropped.append(f"情形 {_conds_plain(lf.conds)}："
                                 "解代回原方程不成立，已剔除")
                continue
        elif lf.kind == "all":
            if not _ok_all(expr, sub):
                continue
        elif not _ok_none(expr, main, sub):
            continue
        kept.append((lf, sub))

    if not kept:
        return None

    # 一支都不剩「条件」时，说明系数里没有真参数，那不算分类讨论，交回普通求解
    if len(kept) == 1 and not _conds_latex(kept[0][0].conds):
        return None

    marks = "①②③④⑤⑥"
    cases = []
    for i, (lf, sub) in enumerate(kept):
        cond = _conds_latex(lf.conds)
        body = _leaf_latex(lf, main)
        d.steps.append(_plain(
            f"情形 {i + 1}：{_conds_plain(lf.conds) or '无需附加条件'}", body))
        cases.append(f"{marks[i]}\\ {cond}: {body}" if cond else f"{marks[i]}\\ {body}")

    d.solutions = [r" \quad ".join(cases)]
    d.steps.append(_plain(
        "验算（取下面的参数值代回，各式均成立）",
        r" \quad ".join(f"{marks[i]}\\ {_sub_text(sub)}"
                        for i, (_, sub) in enumerate(kept)),
    ))
    if deg == 2:
        d.notes.append("二次项系数含参数时，必须先讨论它是否为零——为零就降成"
                       "一次方程，直接套求根公式会把这一整支漏掉。")
    return d
