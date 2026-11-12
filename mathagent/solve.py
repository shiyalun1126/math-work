"""求解层：SymPy 负责算，模板负责讲。

算与讲分离不是口号：每条步骤都带着 lhs/rhs 和 check 模式，
讲完之后逐条验算，不成立的丢进 dropped，不让它出现在用户眼前。
"""

from __future__ import annotations

import sympy
from sympy import Derivative, Integral, Poly, solve as sp_solve

from . import config, guard
from .latex import to_latex
from .types import ParsedExpr, SolveResult, Step


def _symbols_of(exprs) -> list:
    syms = set()
    for e in exprs:
        syms |= set(getattr(e, "free_symbols", set()))
    return sorted(syms, key=str)


def _root_depth(e, _seen: int = 0) -> int:
    """根式嵌套层数。√(x²+1) 算 1 层，∛(√(x²+1)+x) 算 2 层。

    为什么要数这个：sympy 遇到嵌套根式会尝试一系列变换，其中某些分支
    在符号层面几乎指数级膨胀。实测 y=∛(√(x²+1)+x)+∛(x-√(x²+1))
    跑了 79.7 秒才放弃——用户看到的就是页面卡死一分多钟。
    """
    if _seen > 60:  # 病态输入别把自己递归炸了
        return 0
    deepest = 0
    for a in getattr(e, "args", ()):
        deepest = max(deepest, _root_depth(a, _seen + 1))
    if isinstance(e, sympy.Pow) and isinstance(e.exp, sympy.Rational) and e.exp.q != 1:
        deepest += 1
    return deepest


def _hard_reason(pe: ParsedExpr) -> str:
    """解析之后、动 SymPy 之前先看一眼：这题是不是根本不该硬算。

    把"我知道这题算不出来"和"我算了 80 秒然后放弃了"分开说，
    是可信度的一部分——后者用户只会以为程序坏了。
    """
    obj = pe.obj
    try:
        if pe.kind == "equation":
            exprs = [obj[0][0], obj[0][1]]
        elif pe.kind == "system":
            exprs = [e for pair in obj for e in pair]
        else:
            exprs = [obj[0]] if obj else []
    except Exception:  # noqa: BLE001
        return ""

    try:
        depth = max((_root_depth(e) for e in exprs), default=0)
    except Exception:  # noqa: BLE001
        return ""
    if depth >= 2:
        return (
            "根号里面还套着根号（嵌套根式），这类方程没有统一的初等解法，"
            "当前版本不硬解。可以先检查式子有没有抄错；"
            "如果确实是这道题，把两边平方/立方逐步去掉根号后再问我。"
        )
    return ""


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


def _original_denominators(lhs, rhs) -> list:
    """原式里真实出现的分母（不为 1 的那些）。

    约分之前必须先记下来。`(x²-1)/(x-1)` 约完是 `x+1`，那个 `x-1` 就没了，
    可定义域还在：x=1 依然代不回去。恒等式那一支尤其危险——它根本不走
    "解出来再验根"那段，分母一旦丢了，x=1 就直接混进"任意实数"里。
    """
    out = []
    for e in (lhs, rhs):
        try:
            _, den = sympy.fraction(sympy.together(e))
        except Exception:  # noqa: BLE001
            continue
        if den != 1 and getattr(den, "free_symbols", None):
            out.append(den)
    return out


def _den_zeros(dens: list, x) -> list:
    """这些分母在哪些 x 上为 0。只收解出来不含别的主元的那些。"""
    bad = []
    for den in dens:
        try:
            roots = sympy.solve(den, x)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(roots, (list, tuple)):
            continue
        for r in roots:
            if isinstance(r, sympy.Expr) and not r.free_symbols:
                bad.append(r)
    return bad


def _solve_equation(lhs, rhs, sr: SolveResult) -> None:
    raw = lhs - rhs
    try:
        num, den = sympy.fraction(sympy.together(raw))
    except Exception:  # noqa: BLE001
        num, den = raw, sympy.Integer(1)
    expr = sympy.expand(num)
    syms = _symbols_of([lhs, rhs])
    dens = _original_denominators(lhs, rhs)
    sr.steps.append(_eq_step("原式", lhs, rhs, check="equiv"))

    x = syms[0]
    # 两个以上符号时，未知数不能取「字母序最小的那个」——x²-2mx+m²-1=0 会被
    # 解成 m = x ± 1，答的是另一个字母。按 x 优先、其次次数最高来定主元。
    if syms:
        try:
            from .params import _pick_main

            got = _pick_main(sympy.expand(lhs - rhs))
            if got:
                x = got[0]
        except Exception:  # noqa: BLE001
            pass

    # 化简后恒等于 0：这是恒等式，解集是全体实数，不是"无解"。
    # 两者在批改里差得很远——学生写 2(x-3)=2x-6 是同解变形，
    # 判成"无解方程"就是冤枉；反过来把无解说成恒成立，是放过一个错。
    #
    # 但"全体实数"要按原式算，不是按约分后的式子算：
    # (x²-1)/(x-1)=x+1 在 x=1 处两边都没定义，答案得把 1 抠掉。
    if syms and sympy.simplify(expr) == 0:
        bad = _den_zeros(dens, x)
        if bad:
            holes = "、".join(f"{to_latex(v)}" for v in bad)
            sr.steps.append(_plain(
                "定义域",
                rf"\text{{原式的分母不能为 0，}}{holes}\text{{ 使分母为 0，需排除}}",
            ))
            sr.solutions = [f"{to_latex(x)} 为任意实数且 {to_latex(x)} ≠ {to_latex(bad[0])}"]
            sr.steps.append(_plain("结论", rf"{to_latex(x)} \neq {holes}\text{{ 的任意实数}}"))
        else:
            sr.steps.append(_plain("结论", rf"\text{{两边恒等，}}{to_latex(x)}\text{{ 取任意实数}}"))
            sr.solutions = [f"{to_latex(x)} 取任意实数"]
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

    # 主元已经在上面定过了（恒等式那一支也要用它写结论），这里不再重算。
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


def _solve_core(pe: ParsedExpr) -> SolveResult:
    sr = SolveResult(kind=pe.kind, problem_latex=pe.normalized)
    obj = getattr(pe, "obj", None)
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


def _scope_exit(pe: ParsedExpr, msg: str) -> SolveResult:
    sr = SolveResult(kind=pe.kind, problem_latex=pe.normalized)
    sr.ok = False
    sr.out_of_scope = True
    sr.error = msg
    return sr


def solve(pe: ParsedExpr) -> SolveResult:
    if not pe.ok:
        sr = SolveResult(kind=pe.kind, problem_latex=pe.normalized)
        sr.error = f"无法解析：{pe.error}"
        return sr
    if not getattr(pe, "obj", None):
        sr = SolveResult(kind=pe.kind, problem_latex=pe.normalized)
        sr.error = "解析结果为空"
        return sr

    # 先判"该不该算"，再谈"算得对不对"。
    hard = _hard_reason(pe)
    if hard:
        return _scope_exit(pe, hard)

    # 兜底超时：预判挡不住的情况（没见过的花样）还得有个止损点。
    # 跑在子进程里（见 guard 模块），到点直接杀掉——线程只能停止等待，
    # 那段没算完的 sympy 会继续占着 GIL，界面照样卡。
    timeout = config.solve_timeout()
    status, got = guard.try_run("solve_core", pe, timeout=timeout)

    if status == guard.TIMEOUT:
        return _scope_exit(
            pe,
            f"这题算了 {timeout:g} 秒还没出结果，我先停下来了。"
            "继续等下去也不会变快——通常是方程没有初等解法，"
            "或者某个符号被认错了。可以先核对一下式子。",
        )

    if status == guard.OK and isinstance(got, SolveResult):
        return got

    sr = SolveResult(kind=pe.kind, problem_latex=pe.normalized)
    if status == guard.ERR:
        sr.error = str(got)
    else:
        sr.error = "求解没有返回结果"
    return sr


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
