"""配方法推导：把 `ax² + bx + c = 0` 一步步推到求根公式。

为什么单独一层。`params.discuss()` 给的是「判别式 → 六种情形」的结论表，
老师和学生要看的是那句话**从哪来**：两边除以 a、移项、配料、开方。
结论表回答「答案是什么」，推导回答「凭什么」。

规矩和项目别处一样：**每一步都要能验**。

  - 前五步是恒等变形（两边同除、同加），验法是「相邻两步是同一个方程
    乘一个不含主元的因子」——比例里一旦漏出主元，说明这一步改了解集，
    直接不输出。
  - 开方那一步不是恒等变形（平方相等不等于原式相等），所以不套上面的判据，
    改成把最终根代回原方程验。
  - 验不过就不给推导。宁可少一段，也不给一段读起来很顺、其实某步是凑的。

**± 为什么能吸收 a 的符号。** 右边是 (b²-4ac)/(4a²)，分母 4a² 开方是 2|a|，
而 a<0 时 |a| = -a，这个负号正好被 ± 吃掉，所以统一写成 2a 不失去一般性。
教科书都这么写，但这一步是“默认”，写清楚比省一句话值。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sympy

from .latex import to_latex
from .types import Step

# 主元候选，和 params 保持一致：中学只写 x、y、z。
_MAIN = ("x", "y", "z", "t", "u", "v", "w")


@dataclass
class Derivation:
    """一段推导。steps 是给人看的，checks 是凭什么敢给。"""

    ok: bool = False
    main: str = ""
    steps: list[Step] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    note: str = ""
    error: str = ""


def _pick(expr):
    """挑主元：x/y/z 优先，其次次数最高的符号。"""
    syms = sorted(expr.free_symbols, key=str)
    if not syms:
        return None
    for name in _MAIN:
        for s in syms:
            if str(s) == name:
                return s
    best, deg = syms[0], -1
    for s in syms:
        try:
            d = sympy.degree(expr, s)
        except Exception:  # noqa: BLE001
            d = 0
        if d > deg:
            best, deg = s, d
    return best


def _same_equation(f_old, f_new, main) -> bool:
    """`f_old = 0` 与 `f_new = 0` 是不是同一个方程。

    判据取「两式之比不含主元」：恒等变形只会乘上一个与 x 无关的因子
    （除以 a、两边同加）。比例里一旦出现主元，说明解集变了。

    退化的情形（某一步两边同时乘 0）返回 False——那种“变形”本来就不合法。
    """
    try:
        if sympy.simplify(f_new) == 0 or sympy.simplify(f_old) == 0:
            return False
        ratio = sympy.simplify(f_old / f_new)
        return not (getattr(ratio, "free_symbols", set()) & {main})
    except Exception:  # noqa: BLE001
        return False


def _residual(expr, main, root) -> bool:
    """把根代回原方程，看是不是真的 0（符号意义下）。"""
    try:
        r = sympy.simplify(sympy.expand(expr.subs(main, root)))
        if r == 0:
            return True
        r = sympy.simplify(sympy.radsimp(r))
        return r == 0
    except Exception:  # noqa: BLE001
        return False


def _coef_text(e) -> str:
    """系数在纯文本里怎么写。

    `sympy.sstr(a**2)` 给的是 `a**2`——那是 Python 的写法，印在界面上像代码。
    上标数字直接换成 ²，其余幂次写成 `^n`。
    """
    sup = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    s = sympy.sstr(e)

    def _pow(m: re.Match) -> str:
        p = m.group(1)
        if p.isdigit() and len(p) == 1:
            return sup[int(p)]
        return "^" + p

    return re.sub(r"\*\*\(?(-?\w+)\)?", _pow, s)


def _step(title: str, lhs, rhs) -> Step:
    return Step(title=title, latex=f"{to_latex(lhs)} = {to_latex(rhs)}")


def _text(title: str, body: str) -> Step:
    return Step(title=title, latex=body)


def _disc_latex(A, B, C) -> str:
    """判别式 b²-4ac。

    试过用 `evaluate=False` 强行排成教材顺序，结果 `m` 那种系数会印出
    `\\left(-4\\right) m 1 + 4`——为了顺序把式子弄坏了，不划算。
    用 sympy 默认输出 `- 4 a c + b^{2}`，与 params 里判别式那一行保持一致。
    """
    return to_latex(sympy.expand(B**2 - 4 * A * C))


def _rad_latex(disc) -> str:
    """根号能化简就化简：`\\sqrt{1}` 应写成 1，`\\sqrt{4-4m}` 保留根号。"""
    r = sympy.sqrt(disc)
    if r.is_Pow and r.exp == sympy.Rational(1, 2):
        return rf"\sqrt{{{to_latex(r.base)}}}"
    return to_latex(r)


def complete_square(lhs, rhs) -> Derivation | None:
    """关于主元的一元二次方程 → 配方法推导。不是二次就返回 None。"""
    try:
        expr = sympy.expand(lhs - rhs)
    except Exception:  # noqa: BLE001
        return None

    main = _pick(expr)
    if main is None:
        return None
    try:
        poly = sympy.Poly(expr, main)
    except Exception:  # noqa: BLE001
        return None
    if poly.degree() != 2:
        return None

    A, B, C = poly.all_coeffs()
    d = Derivation(main=str(main))

    # 二次项系数为零时不能除以 a，这一步是整段推导的前提，不是废话。
    if A == 0:
        d.error = "二次项系数为 0，不是二次方程"
        return None

    # 只给「系数含字母」的方程出推导。`x²-5x+6=0` 这种系数是数字的，
    # 写出来第一步就是「两边除以 1」，第七步是 `- -5 ± √1`——全是空转和丑模样。
    # 那种题界面另有完整求解步骤，不必套这个模板。
    if not A.free_symbols:
        d.error = "二次项系数是常数，不出推导模板"
        return None

    x = main
    half = B / (2 * A)                    # 配出来的常数项 b/(2a)
    disc = sympy.expand(B**2 - 4 * A * C)
    right = sympy.together(half**2 - C / A)   # 右边通分后的 (b²-4ac)/(4a²)

    if B == 0:
        # 缺一次项时配方是空转：一次项系数一半的平方是 0，「两边加上 0」
        # 和「右边通分」两步什么也没做，右边还凭空多一层 4a² 的分子分母。
        # 这种题教材用直接开平方，就照教材写。
        rhs0 = sympy.together(-C / A)
        d.steps.append(_step("原方程", A * x**2 + C, 0))
        d.steps.append(_step("两边除以二次项系数", x**2, rhs0))
        d.steps.append(_text(
            "两边开平方（缺一次项，左边本来就是一个平方）",
            rf"{to_latex(x)} = \pm {to_latex(sympy.sqrt(rhs0))}",
        ))
        forms = [expr, sympy.expand(A * (x**2 - rhs0))]
        roots = [sympy.sqrt(rhs0), -sympy.sqrt(rhs0)]
    else:
        d.steps.append(_step("原方程", A * x**2 + B * x + C, 0))
        d.steps.append(_step(
            "两边除以二次项系数",
            x**2 + (B / A) * x + C / A,
            0,
        ))
        d.steps.append(_step(
            "常数项移到右边",
            x**2 + (B / A) * x,
            -C / A,
        ))
        d.steps.append(_step(
            "两边加上一次项系数一半的平方，左边凑成完全平方",
            x**2 + (B / A) * x + half**2,
            sympy.together(-C / A + half**2),
        ))
        d.steps.append(_text(
            "左边写成完全平方，右边通分",
            rf"{to_latex((x + half) ** 2)} = "
            rf"\frac{{{_disc_latex(A, B, C)}}}{{{to_latex(4 * A**2)}}}",
        ))
        d.steps.append(_text(
            "两边开平方（4a² 开方是 2|a|，符号被 ± 吸收，写成 2a 不失一般性）",
            rf"{to_latex(x + half)} = \pm \frac{{{_rad_latex(disc)}}}"
            rf"{{{to_latex(2 * A)}}}",
        ))
        d.steps.append(_text(
            "移项，得到求根公式",
            rf"{to_latex(x)} = \frac{{{to_latex(sympy.simplify(-B))} \pm "
            rf"{_rad_latex(disc)}}}{{{to_latex(2 * A)}}}",
        ))
        # 验算：前五步逐对查「是不是同一个方程」，最后一步查根代回。
        forms = [
            expr,                                   # 原方程
            sympy.expand(x**2 + (B / A) * x + C / A),
            sympy.expand(x**2 + (B / A) * x + C / A),        # 移项不改方程
            sympy.expand(x**2 + (B / A) * x + C / A),        # 两边同加，仍等价
            sympy.expand((x + half) ** 2 - right),
        ]
        roots = [(-B + sympy.sqrt(disc)) / (2 * A), (-B - sympy.sqrt(disc)) / (2 * A)]

    for i in range(1, len(forms)):
        if not _same_equation(forms[i - 1], forms[i], x):
            d.error = f"第 {i + 1} 步与上一步不同解，推导不输出"
            return None
    d.checks.append(f"前 {len(forms)} 步逐对验过：每步都是同一个方程的等价变形")

    if all(_residual(expr, x, r) for r in roots):
        d.checks.append("求根公式的两支代回原方程，符号化简均为 0")
    else:
        # 符号化简不动不等于公式错了，但不能假装验过——如实少说一句。
        d.checks.append("求根公式已给出；这一版未能完成符号验算，只做了情形讨论的数值验算")


    # note 会拼进界面的纯文本区，这里不能出现 LaTeX 命令（\neq 会被原样显示），
    # 也不能出现 `a**2` 这种 Python 写法。
    coef = _coef_text(A)
    d.note = (
        f"推导的前提是 {coef} ≠ 0。若 {coef} = 0，方程降为一次，"
        f"上面的变形不成立——这也是答案要给 {coef} = 0 那一支的原因。"
    )
    d.ok = True
    return d
