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

    `sqrt` 是同一个坑的另一副面孔，而且更毒：`sqrt(x+1)` 被读成
    `q*r*s*t*(x+1)`——表面上没报错，只是凭空多出四个"参数"，于是
    无理方程被判成「含参数方程」。学生手打公式时几乎不会写 `\\sqrt{}`，
    这条必须兜住。
    """
    s = re.sub(
        r"(?<![\\a-zA-Z])(arcsin|arccos|arctan|sinh|cosh|tanh|sin|cos|tan|cot|sec|csc|log|ln|exp|sqrt)(?![a-zA-Z])",
        r"\\\1",
        s,
    )
    # latex2sympy 只认 `\sqrt{...}`：`\sqrt(x+1)` 和 `\sqrt2` 都会抛异常，
    # 所以补完反斜杠还得把圆括号 / 裸字符收敛成花括号。
    # `\sqrt[3]{x}` 带可选参数，不匹配 `\(`，原样留给解析器。
    s = re.sub(r"\\sqrt\s*\(([^()]*)\)", r"\\sqrt{\1}", s)
    return re.sub(r"\\sqrt\s*([a-zA-Z0-9])", r"\\sqrt{\1}", s)


# 括号里写的是题目说明，不是式子的一部分。
#
# `ax^{2}+bx+c=0\left(a,b,c\text{均为常数}\right)` 是课本上最标准的一行，
# 整串进解析器必然失败——失败得安全（不给错答案），但用户看到的是「做不了」，
# 而题本身完全能做：`discuss()` 早就算出了六支，含求根公式。
#
# 只删「括号里有中文、且没有比较符」的尾注。`(x>0)` 是定义域，删了就是丢条件；
# `(a+b)^2=4` 这种括号在开头，不匹配。
_NOTE = re.compile(
    r"(?:\s*\\left)?\s*\(\s*([^()]*?)\s*\)\s*(?:\s*\\right)?\s*[。.，,]?\s*$"
)


def _strip_note(s: str) -> str:
    m = _NOTE.search(s)
    if not m:
        return s
    body = m.group(1)
    if not re.search(r"[\u4e00-\u9fff]", body):
        return s
    if re.search(r"[=<>≤≥]", body):
        return s
    # 括号前面必须有东西，否则整串就是一句说明，删完什么都不剩
    head = s[: m.start()].strip()
    return head or s


def normalize(latex: str) -> str:
    s = (latex or "").strip()
    if not s:
        return ""
    s = s.translate(_FULLWIDTH)
    s = _unicode_to_latex(s)
    s = _strip_note(s)
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


_ENV = re.compile(
    r"\\begin\{(alignedat|aligned|array|split|gathered|cases)\}"
    r"(?:\{[^{}]*\})?(.*?)\\end\{\1\}",
    re.S,
)

# 题号：`1) x+y=6`、`（2）x=3`、`3. x=1`。
#
# 不剥的话，`1)` 里的 1 会和等号右边的数配成一个式子——
# `1)\left\{\begin{aligned}x+y&=6` 会被读成 `1=6`，还报 ok=True。
# 假成功比报错难查得多。
#
# 判据必须带上标点：只写 `^\d+` 会把 `2x+3=5` 剥成 `x+3=5`，
# 加上 `(?!\d)` 是为了别把 `2.5x` 的小数点当成题号的分隔点。
_ENV_ENUM = re.compile(r"^\s*(?:[(（]\s*\d+\s*[)）]|\d+\s*[)）.、](?!\d))\s*")


def canonical_env(text: str) -> str:
    """把「一个环境里装多条方程」的各种写法规整成标准 ``cases``。

    模型读方程组不只用 ``cases``：``aligned`` / ``array`` / ``split`` /
    ``gathered`` 都遇到过。这些环境里的 ``\\\\`` 是**排版换行**，不是方程之间的
    分隔符，可它装的确实是好几条方程。不捋平的话，换一种写法就"读不出来"，
    而投票层看到全部候选失败，会给出"没认准"——其实每个模型都读对了。

    只动结构，不动数学内容：剥题号、按排版换行拆行、去掉对齐符 ``&``、
    去掉行末逗号句号。凑不出两条就原样返回，不改单条式子的写法。
    """
    s = _ENV_ENUM.sub("", (text or "").strip())
    m = _ENV.search(s)
    if not m:
        return s

    rows: list[str] = []
    for row in re.split(r"\\\\", m.group(2)):
        row = row.replace("&", "").strip(" \t,，;；")
        row = re.sub(r"[.。]$", "", row).strip()
        if row:
            rows.append(row)
    if len(rows) < 2:
        return s
    return "\\begin{cases}" + "\\\\".join(rows) + "\\end{cases}"


def parse(latex: str) -> ParsedExpr:
    latex = canonical_env(latex)
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
            eqs = []
            for no, piece in enumerate(pieces, 1):
                sides = _split_top(piece, ("=",))
                if len(sides) != 2:
                    # 以前是 `[e for e in eqs if len(e) == 2]` 一过，读不出的行
                    # 直接消失，剩下的行还能解出 "答案"。那是把一道残缺的输入
                    # 悄悄换成另一道题——比报错糟得多，错在哪都看不出来。
                    raise ValueError(
                        f"cases 第 {no} 行不是一条完整等式（{piece}），"
                        "分段条件或缺项的行不支持，整题不求解"
                    )
                eqs.append(sides)
            if not eqs:
                raise ValueError("cases 环境里没有解析出等式")
            obj = [(_to_sympy(l)[0], _to_sympy(r)[0]) for l, r in eqs]
            pe.kind = "system"
            pe.obj = obj
            pe.sympy_str = str(obj)
            pe.ok = True
            return pe

        parts = _split_top(body, (",",))
        if len(parts) > 1 and any("=" in p for p in parts):
            # 有一段带等号、另一段不带，最可能是"方程组 + 一句说明"，
            # 也可能是分段条件。两种都不该猜，说清哪一段不成样子。
            for no, p in enumerate(parts, 1):
                if "=" not in p:
                    raise ValueError(f"第 {no} 段不是等式（{p}），整题不求解")
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

    **方程组不能走这里。** 把每条方程相加压成一个式子，会把约束抵消掉：
    `{x+y=3, x-y=1}`（解 (2,1)）与 `{x+y=30, x-y=-26}`（解 (2,28)）加起来都是
    `2x-4`，两个不同的题就被判成同一簇。方程组请用 `systems_equivalent()`。
    """
    obj = getattr(pe, "obj", None)
    if not obj:
        return None
    if pe.kind == "equation":
        acc = 0
        for lhs, rhs in obj:
            acc += sympy.together(lhs - rhs)
        return acc
    if pe.kind == "system":
        return None
    return obj[0]


def _scaled(a, b) -> bool:
    """`a=0` 和 `b=0` 是不是同一个方程乘了一个非零常数。

    `2x+2y=6` 与 `x+y=3` 是；`x+y=3` 与 `x+y=30` 不是。系数比必须是数，
    带主元就说明两条方程的零点是两回事（`x(x-1)=0` 与 `(x-1)=0`）。
    """
    try:
        if a == 0 or b == 0:
            return a == b
        ratio = sympy.simplify(a / b)
        return bool(ratio.is_number) and ratio != 0
    except Exception:  # noqa: BLE001
        return False


def systems_equivalent(oa, ob) -> bool:
    """两个方程组是不是同一个。

    为什么不能求和：`y` 的条件一正一负，相加正好抵消，剩下 `2x-4`。
    两个解不同的题会合成一簇，多源投票还会把这种"一致"当成旁证。

    先逐条比："条数相同，且每一条都能在对方那里找到只差一个非零常数倍的那条"。
    这比同解集更严：`{x+y=3, x-y=1}` 与 `{x=2, y=1}` 同解，却过不了这一关。
    严一点是有意的——判成"没认准"只是让人多看一眼，判成"一致"是两个错读
    互相背书，后者没法补救。

    系数全是数字时再补一步：把两条方程换成"合法线性组合"（`{2x=4, 2y=2}`
    和 `{x+y=3, x-y=1}` 是同一组解），逐条比认不出来。这时比较增广矩阵的行空间，
    行空间相同就是同一组解。带参数时不走这条——符号秩不可靠，宁可判"没认准"。
    """
    if not oa or not ob or len(oa) != len(ob):
        return False
    if _rows_match(oa, ob):
        return True
    return _same_row_space(oa, ob)


def _rows_match(oa, ob) -> bool:
    left = list(range(len(ob)))
    for lhs_a, rhs_a in oa:
        try:
            ea = sympy.together(sympy.expand(lhs_a - rhs_a))
        except Exception:  # noqa: BLE001
            return False
        hit = None
        for j in left:
            lhs_b, rhs_b = ob[j]
            try:
                eb = sympy.together(sympy.expand(lhs_b - rhs_b))
            except Exception:  # noqa: BLE001
                continue
            if _scaled(ea, eb):
                hit = j
                break
        if hit is None:
            return False
        left.remove(hit)
    return True


def _same_row_space(oa, ob) -> bool:
    """系数全为数字时，比较增广矩阵的行空间。

    行空间相同 ⇔ 两个线性方程组的解集相同。`{x+y=3, x-y=1}` 与
    `{2x=4, 2y=2}` 逐条比不过这一关，但它们是同一组解，该认。
    """
    try:
        ea = [sympy.expand(l - r) for l, r in oa]
        eb = [sympy.expand(l - r) for l, r in ob]
        syms = sorted(set().union(*[e.free_symbols for e in ea + eb]), key=str)
        if not syms:
            return False
        A1, b1 = sympy.linear_eq_to_matrix(ea, syms)
        A2, b2 = sympy.linear_eq_to_matrix(eb, syms)
        if any(v.free_symbols for v in list(A1) + list(b1) + list(A2) + list(b2)):
            return False
        M1 = A1.row_join(b1)
        M2 = A2.row_join(b2)
        r1, r2 = M1.rank(), M2.rank()
        stack = M1.col_join(M2)
        return r1 == r2 == stack.rank()
    except Exception:  # noqa: BLE001
        return False


def to_latex(expr) -> str:
    try:
        return sympy.latex(expr)
    except Exception:  # noqa: BLE001
        return str(expr)
