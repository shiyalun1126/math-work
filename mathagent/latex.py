"""LaTeX 归一化与解析。

解析失败不抛异常，一律返回 ok=False + error —— 上层靠这个字段决定是否追问用户，
抛异常会让"识别错了"和"解析器崩了"混在一起，没法排查。
"""

from __future__ import annotations

import re
from typing import Any

import sympy

from .route import detect
from .types import ParsedExpr

# 全角 -> 半角（学生用手机输入法时很常见）
_FULLWIDTH = str.maketrans(
    {
        "（": "(",
        "）": ")",
        "＝": "=",
        "＋": "+",
        "－": "-",
        "＊": "*",
        "／": "/",
        "，": ",",
        "。": ".",
        "：": ":",
        "｛": "{",
        "｝": "}",
        "　": " ",
        "×": r"\times ",
        "÷": r"\div ",
    }
)

_DROP_CMDS = [
    r"\left", r"\right", r"\bigl", r"\bigr", r"\Bigl", r"\Bigr",
    r"\bigg", r"\Bigg", r"\displaystyle", r"\limits", r"\nolimits",
    r"\because", r"\therefore", r"\quad", r"\qquad", r"\;", r"\,", r"\!",
    r"\ ", r"~", r"\nonumber",
]


def _as_system(s: str) -> str:
    """把"多行方程"补成 cases 环境。

    识别器看到两行方程时，会原样输出 `x + y = 3 \\\\ x - y = 1` 而不加环境，
    归一化后就成了无法解析的字符串。学生写方程组本来就是一行一个方程，
    所以这个形态在真实数据里是常态，不是边角情况。

    只把"每行都含等号、且行内没有逗号"的形态当方程组：像
    `x=2, x=3` 这种解集不会被误包。
    """
    if "\\begin{" in s or "\x00" in s:
        return s
    parts = [p.strip() for p in re.split(r"\\\\|\n", s) if p.strip()]
    if len(parts) < 2 or len(parts) > 6:
        return s
    if not all("=" in p and "," not in p for p in parts):
        return s
    return "\\begin{cases} " + " \\\\ ".join(parts) + " \\end{cases}"


_UNICODE_MAP = {
    "∫": r"\int ",
    "∬": r"\iint ",
    "∮": r"\oint ",
    "∑": r"\sum ",
    "∏": r"\prod ",
    "√": r"\sqrt",
    "∂": r"\partial ",
    "π": r"\pi ",
    "θ": r"\theta ",
    "α": r"\alpha ",
    "β": r"\beta ",
    "γ": r"\gamma ",
    "λ": r"\lambda ",
    "Δ": r"\Delta ",
    "×": r"\times ",
    "÷": "/",
    "·": r"\cdot ",
    "≤": r"\leq ",
    "≥": r"\geq ",
    "≠": r"\neq ",
    "∞": r"\infty ",
    "±": r"\pm ",
    "→": r"\to ",
    "∈": r"\in ",
}

_SUP_CHARS = "⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ"
_SUB_CHARS = "₀₁₂₃₄₅₆₇₈₉"


def _unicode_to_latex(s: str) -> str:
    """把模型偷懒照抄的 Unicode 数学符号换回 LaTeX 命令。

    通用多模态模型（GLM 尤其明显）有时直接输出 `∫ x sin x dx` 而不写
    `\\int`，latex2sympy 只认命令，这一路候选就会整个作废——明明认对了，
    却因为格式白丢一张独立票。这层兜底专门捞这种。
    """
    if not any(ch in s for ch in _SUP_CHARS + _SUB_CHARS) and not any(
        ch in s for ch in _UNICODE_MAP
    ):
        return s

    def sup_repl(m: re.Match) -> str:
        body = "".join(
            "0123456789n"[_SUP_CHARS.index(ch)] for ch in m.group(0)
        )
        return "^{" + body + "}" if len(body) > 1 else "^" + body

    def sub_repl(m: re.Match) -> str:
        body = "".join(
            "0123456789"[_SUB_CHARS.index(ch)] for ch in m.group(0)
        )
        return "_{" + body + "}" if len(body) > 1 else "_" + body

    s = re.sub(f"[{_SUP_CHARS}]+", sup_repl, s)
    s = re.sub(f"[{_SUB_CHARS}]+", sub_repl, s)
    for k, v in _UNICODE_MAP.items():
        s = s.replace(k, v)

    # √(x+3) -> \sqrt{x+3}；√x -> \sqrt{x}
    s = re.sub(r"\\sqrt\s*\(([^()]*)\)", r"\\sqrt{\1}", s)
    s = re.sub(r"\\sqrt\s*([a-zA-Z0-9])", r"\\sqrt{\1}", s)
    return s


def _fix_functions(s: str) -> str:
    """函数名补反斜杠：模型常写 "x sin x" 而不写 "\\sin x"。

    后者会被 latex2sympy 解析成 s×i×n 四个变量相乘——认对了却算错，
    比直接认错更隐蔽，而且不会触发任何解析错误。
    """
    return re.sub(
        r"(?<![\\a-zA-Z])(arcsin|arccos|arctan|sinh|cosh|tanh|sin|cos|tan|cot|sec|csc|log|ln|exp)(?![a-zA-Z])",
        r"\\\1",
        s,
    )


def normalize(latex: str) -> str:
    s = (latex or "").strip()
    if not s:
        return ""
    s = s.translate(_FULLWIDTH)
    s = _unicode_to_latex(s)
    s = _fix_functions(s)
    s = _as_system(s)
    # 先把换行符 \\ 保护起来，否则后面清理 \ + 空格的规则会吃掉其中一个反斜杠
    s = s.replace("\\\\", "\x00")
    # 去掉 markdown/数学环境包裹
    s = re.sub(r"^\s*\$\$?", "", s)
    s = re.sub(r"\$\$?\s*$", "", s)
    s = s.replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")
    for cmd in _DROP_CMDS:
        s = re.sub(re.escape(cmd), " ", s)
    s = re.sub(r"\\label\{[^}]*\}", " ", s)
    s = s.replace("\x00", "\\\\")
    # 等价写法归一
    s = re.sub(r"\\d?frac", r"\\frac", s)
    s = re.sub(r"\\div\b", "/", s)
    # 常见 OCR 噪声：结尾的句号、等号、逗号
    s = re.sub(r"[。.,，;；:：\s]+$", "", s)
    # 结尾只有等号，视为 "=0"（学生常漏写右边）
    s = re.sub(r"=\s*$", "=0", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_top(s: str, seps: tuple[str, ...]) -> list[str]:
    """按括号深度为 0 处切分，避开 \\frac{..}{..} 内部的逗号。"""
    out, buf, depth = [], [], 0
    i = 0
    while i < len(s):
        if depth == 0:
            hit = ""
            for sep in seps:
                if s.startswith(sep, i):
                    hit = sep
                    break
            if hit:
                out.append("".join(buf))
                buf = []
                i += len(hit)
                continue
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            buf.append(s[i : i + 2])
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        buf.append(ch)
        i += 1
    out.append("".join(buf))
    return [x.strip() for x in out if x.strip()]


def _to_sympy(piece: str):
    errors = []
    for name, fn in (
        ("latex2sympy2", _via_latex2sympy),
        ("sympy.parse_latex", _via_sympy_parse_latex),
    ):
        try:
            return fn(piece), name
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}")
    raise ValueError("; ".join(errors))


def _via_latex2sympy(s: str):
    from latex2sympy2 import latex2sympy

    return latex2sympy(s)


def _via_sympy_parse_latex(s: str):
    from sympy.parsing.latex import parse_latex

    return parse_latex(s)


_CASES = re.compile(r"\\begin\{cases\}(.*?)\\end\{cases\}", re.S)


def parse(latex: str) -> ParsedExpr:
    norm = normalize(latex)
    pe = ParsedExpr(latex=latex, normalized=norm, kind="unknown")
    if not norm:
        pe.error = "空输入"
        return pe

    # 先判「这是不是算式」。整页题干被塞进公式链路时，再往下走只会得到
    # 「等号数量异常：2 个」这种谁也看不懂的话——错的不是式子，是入口。
    r = detect(norm)
    if r.kind != "formula":
        pe.kind = r.kind
        pe.error = r.note
        return pe

    try:
        body = norm
        m = _CASES.search(body)
        if m:  # 分段/方程组写法
            pieces = _split_top(m.group(1), ("\\\\",))
            eqs = [_split_top(p, ("=",)) for p in pieces]
            eqs = [e for e in eqs if len(e) == 2]
            if not eqs:
                raise ValueError("cases 环境里没有解析出等式")
            obj = [(_to_sympy(l)[0], _to_sympy(r)[0]) for l, r in eqs]
            pe.kind = "system"
            pe.obj = obj
            pe.sympy_str = str(obj)
            pe.ok = True
            return pe

        parts = _split_top(body, (",",))
        if len(parts) > 1 and all("=" in p for p in parts):
            obj = []
            for p in parts:
                l, r = _split_top(p, ("=",))[:2]
                obj.append((_to_sympy(l)[0], _to_sympy(r)[0]))
            pe.kind = "system"
            pe.obj = obj
            pe.sympy_str = str(obj)
            pe.ok = True
            return pe

        if "=" in body:
            sides = _split_top(body, ("=",))
            if len(sides) != 2:
                raise ValueError(f"等号数量异常：{len(sides) - 1} 个")
            lhs = _to_sympy(sides[0])[0]
            rhs = _to_sympy(sides[1])[0]
            pe.kind = "equation"
            pe.obj = [(lhs, rhs)]
            pe.sympy_str = f"{lhs} == {rhs}"
            pe.ok = True
            return pe

        pe.kind = "expression"
        pe.obj = [_to_sympy(body)[0]]
        pe.sympy_str = str(pe.obj[0])
        pe.ok = True
        return pe

    except Exception as exc:  # noqa: BLE001
        pe.ok = False
        pe.kind = "unknown"
        pe.error = f"{type(exc).__name__}: {exc}"
        return pe


def to_expr(pe: ParsedExpr) -> Any:
    """把解析结果压成"用于等价比较"的单个表达式。

    等式 a=b 压成 lhs-rhs，表达式原样。这样语义等价判定统一成 simplify(diff)==0。
    """
    obj = getattr(pe, "obj", None)
    if not obj:
        return None
    if pe.kind in ("equation", "system"):
        acc = 0
        for lhs, rhs in obj:
            acc += sympy.together(lhs - rhs)
        return acc
    return obj[0]


def to_latex(expr) -> str:
    try:
        return sympy.latex(expr)
    except Exception:  # noqa: BLE001
        return str(expr)
