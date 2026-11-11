"""求解层：SymPy 负责算，模板负责讲。

算与讲分离不是口号：每条步骤都带着 lhs/rhs 和 check 模式，
讲完之后逐条验算，不成立的丢进 dropped，不让它出现在用户眼前。
"""

from __future__ import annotations

import sympy
from sympy import Derivative, Integral, Poly, solve as sp_solve

from .latex import to_latex
from .types import ParsedExpr, SolveResult, Step


def _symbols_of(exprs) -> list:
    syms = set()
    for e in exprs:
        syms |= set(getattr(e, "free_symbols", set()))
    return sorted(syms, key=str)


def _denominator(expr):
    try:
        _, den = sympy.fraction(sympy.together(expr))
        return den
    except Exception:  # noqa: BLE001
        return sympy.Integer(1)


def _eq_step(title: str, lhs, rhs, check: str = "none") -> Step:
    return Step(title=title, latex=f"{to_latex(lhs)} = {to_latex(rhs)}", lhs=lhs, rhs=rhs, check=check)


def _plain(title: str, latex: str) -> Step:
    return Step(title=title, latex=latex)


def _solution_set(lhs, rhs):
    syms = _symbols_of([lhs, rhs])
    if not syms:
        return None
    try:
        return set(sp_solve(sympy.expand(lhs - rhs), syms[0]))
    except Exception:  # noqa: BLE001
        return None


def check_steps(sr: SolveResult) -> SolveResult:
    """逐条验算。identity 验恒等，equiv 验与上一步同解。"""
    kept: list[Step] = []
    prev: Step | None = None
    for st in sr.steps:
        if st.check == "none" or st.lhs is None:
            kept.append(st)
            continue
        try:
            if st.check == "identity":
                ok = sympy.simplify(st.lhs - st.rhs) == 0
            else:
                if prev is None:
                    ok = True
                else:
                    s1, s2 = _solution_set(prev.lhs, prev.rhs), _solution_set(st.lhs, st.rhs)
                    if s1 is not None and s2 is not None:
                        ok = s1 == s2
                    else:
                        ok = sympy.simplify((st.lhs - st.rhs) - (prev.lhs - prev.rhs)) == 0
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            kept.append(st)
            prev = st
        else:
            sr.dropped.append(f"{st.title}: {st.latex}")
    sr.steps = kept
    return sr


def _solve_equation(lhs, rhs, sr: SolveResult) -> None:
    raw = lhs - rhs
    try:
        num, den = sympy.fraction(sympy.together(raw))
    except Exception:  # noqa: BLE001
        num, den = raw, sympy.Integer(1)
    expr = sympy.expand(num)
    syms = _symbols_of([lhs, rhs])
    sr.steps.append(_eq_step("原式", lhs, rhs, check="equiv"))

    # 化简后恒等于 0：这是恒等式，解集是全体实数，不是"无解"。
    # 两者在批改里差得很远——学生写 2(x-3)=2x-6 是同解变形，
    # 判成"无解方程"就是冤枉；反过来把无解说成恒成立，是放过一个错。
    if syms and sympy.simplify(expr) == 0:
        sr.steps.append(_plain("结论", r"\text{两边恒等，}x\text{ 取任意实数}"))
        sr.solutions = ["x 取任意实数"]
        sr.ok = True
        return

    if not syms:
        if sympy.simplify(expr) == 0:
            sr.ok = True
            sr.solutions = ["恒成立"]
            return
        # 两边都是常数还矛盾，数学上叫"无解方程"，但在手写识别场景里
        # 这几乎总是上标/符号被吃掉了（2ˣ 认成 2、x² 认成 x）。
        # 直接回"矛盾，无解"等于把识别错误包装成正确答案，必须降级追问。
        sr.ok = False
        sr.error = "式子里没有任何未知数，疑似识别时丢了上标或变量，请核对原图"
        return

    x = syms[0]
    # 两个以上符号时，未知数不能取「字母序最小的那个」——x²-2mx+m²-1=0 会被
    # 解成 m = x ± 1，答的是另一个字母。按 x 优先、其次次数最高来定主元。
    try:
        from .params import _pick_main

        got = _pick_main(sympy.expand(lhs - rhs))
        if got:
            x = got[0]
    except Exception:  # noqa: BLE001
        pass
    if len(syms) > 1:
        sols = sp_solve(expr, x, dict=True)
        sr.steps.append(_eq_step("整理", expr, 0, check="equiv"))
        sr.solutions = [f"{to_latex(x)} = {to_latex(s[x])}" for s in sols] or ["无解"]
        sr.ok = bool(sols)
        if not sols:
            sr.error = "未求出显式解"
        return

    move = _eq_step("移项整理", expr, 0, check="equiv")
    if move.latex != sr.steps[-1].latex:
        sr.steps.append(move)

    poly, deg = None, None
    try:
        poly = Poly(expr, x)
        deg = poly.degree()
    except Exception:  # noqa: BLE001
        pass

    if deg == 2:
        a, b, c = poly.all_coeffs()
        disc = sympy.simplify(b**2 - 4 * a * c)
        sr.steps.append(_plain("判别式", rf"\Delta = b^2-4ac = {to_latex(disc)}"))
        sols = sp_solve(expr, x)
        if sympy.simplify(disc).is_negative:
            sr.steps.append(_plain("结论", r"\Delta < 0，无实根"))
            sr.solutions = ["无实根"]
            sr.ok = True
            return
    else:
        sols = sp_solve(expr, x)

    kept = []
    for s in sols:
        if den != 1:
            try:
                if sympy.simplify(den.subs(x, s)) == 0:
                    sr.dropped.append(f"增根已剔除：{to_latex(s)} 使分母为 0")
                    continue
            except Exception:  # noqa: BLE001
                pass
        kept.append(s)

    # 每个根一条"代回验证 → 0 = 0"是内部记账：0 = 0 既不是过程也不是结论，
    # 学生只会问"这个 0 是哪来的"。验算在答案区用"左边 = ? 右边 = ?"单独讲，
    # 这里不再重复一遍机器残差。
    sr.solutions = [f"{to_latex(x)} = {to_latex(s)}" for s in kept] or ["无解"]
    sr.ok = bool(kept)
    if kept:
        sep = r" \quad \text{或} \quad "
        sr.steps.append(_plain("解得", sep.join(sr.solutions)))


def _solve_system(pairs, sr: SolveResult) -> None:
    eqs = [sympy.expand(l - r) for l, r in pairs]
    syms = _symbols_of([e for p in pairs for e in p])
    for l, r in pairs:
        sr.steps.append(_eq_step("方程", l, r))
    sols = sp_solve(eqs, syms, dict=True)
    sr.ok = bool(sols)
    if not sols:
        sr.error = "无解或无法求出闭式解"
        sr.solutions = ["无解"]
        return
    sr.solutions = [
        "，".join(f"{to_latex(k)} = {to_latex(v)}" for k, v in s.items()) for s in sols
    ]
    sr.steps.append(_plain("解得", r" \quad;\quad ".join(sr.solutions)))
    for s in sols:
        resid = [sympy.simplify(e.subs(s)) for e in eqs]
        sr.steps.append(
            _plain("代回验证", ",\\ ".join(f"{to_latex(r)} = 0" for r in resid))
        )
        break


def _solve_expression(expr, sr: SolveResult) -> None:
    sr.steps.append(_plain("原式", to_latex(expr)))

    if isinstance(expr, Integral):
        var = expr.limits[0][0] if expr.limits else None
        body = expr.function
        val = sympy.simplify(expr.doit())
        sr.steps.append(_plain("积分", f"{to_latex(expr)} = {to_latex(val)}"))
        if var is not None:
            sr.steps.append(
                _eq_step("验算（求导还原）", sympy.diff(val, var), body, check="identity")
            )
        sr.solutions = [to_latex(val) + " + C"]
        sr.ok = True
        return

    if isinstance(expr, Derivative):
        val = sympy.simplify(expr.doit())
        sr.steps.append(_plain("求导", f"{to_latex(expr)} = {to_latex(val)}"))
        sr.solutions = [to_latex(val)]
        sr.ok = True
        return

    val = sympy.simplify(expr)
    if val != expr:
        sr.steps.append(_eq_step("化简", expr, val, check="identity"))
    sr.solutions = [to_latex(val)]
    sr.ok = True


def solve(pe: ParsedExpr) -> SolveResult:
    sr = SolveResult(kind=pe.kind, problem_latex=pe.normalized)
    if not pe.ok:
        sr.error = f"无法解析：{pe.error}"
        return sr
    obj = getattr(pe, "obj", None)
    if not obj:
        sr.error = "解析结果为空"
        return sr
    try:
        if pe.kind == "equation":
            lhs, rhs = obj[0]
            # 含参数就先分类讨论：普通求解器会默认系数非零，把「降次」那一支
            # 悄悄丢掉——对学生来说那是漏解，不是简略。
            from .params import discuss

            disc = discuss(lhs, rhs)
            if disc is not None:
                sr.kind = "param"
                sr.steps.extend(disc.steps)
                sr.solutions = disc.solutions
                sr.ok = True
                return check_steps(sr)
            _solve_equation(lhs, rhs, sr)
        elif pe.kind == "system":
            _solve_system(obj, sr)
        else:
            _solve_expression(obj[0], sr)
    except Exception as exc:  # noqa: BLE001
        sr.ok = False
        sr.error = f"{type(exc).__name__}: {exc}"
        return sr
    return check_steps(sr)


def summary(sr: SolveResult) -> str:
    if not sr.ok:
        return sr.error or "未能求解"
    if sr.kind == "equation":
        return "解：" + "，".join(sr.solutions) if sr.solutions else "无解"
    if sr.kind == "system":
        return "解：" + "；".join(sr.solutions)
    if sr.kind == "param":
        return "分类讨论：" + "；".join(sr.solutions)
    return "结果：" + (sr.solutions[0] if sr.solutions else "")
