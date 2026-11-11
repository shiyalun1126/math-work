"""批改：把学生的一整套过程当成一串断言，逐步验它站不站得住。

判据只有一条：**局部有效性**。这一步能不能从上一步通过一个合法变换得到。
不是"和我算的一样"——学生用配方法、系统用求根公式，路径不同但都对，
拿全局相等去判，会误伤一题多解。

这里不调大模型。每一句"这一步不成立"都必须能给出反例：
一个满足你上一步、却不满足这一步的取值。给不出反例就说不知道，不硬判。

三种结论值得区分，恶劣程度递增：
  过程对 + 答案对   → pass
  过程对 + 答案缺   → partial（漏解）
  过程错 + 答案对   → fake（伪推导，最危险：答案不是推出来的）
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

import sympy
from sympy import Symbol, simplify, together

from .latex import parse, to_latex
from .route import cjk_count
from .solve import solve as solve_parsed
from .types import ParsedExpr, WorkCheck, WorkStep
from .verify import equivalent

_GUARD = 2.5

# 一行开头常见的连接词。学生不会写"由①②得"，但会写"所以""因此"。
# 连接词后面常跟一个冒号或逗号（"所以：(x-2)(x-3)=0"），必须一起吃掉：
# 只吃"所以"会把"："留在行首，解析器认不得这个字符，整行退化成"读不出算式"。
# 这不是识别问题——字都读对了，是清洗没跟到底。
_LEAD_WORD = re.compile(
    r"^\s*(?:所以|因此|故|则|即|可得|得到|于是|又因为|因为|又)[\s,，:：、]*"
)

# 题号/序号前缀：①②、(1)、1.、"步骤2："
_ENUM = re.compile(r"^\s*(?:[①②③④⑤⑥⑦⑧⑨⑩]|[（(]\s*\d+\s*[)）]|\d+\s*[.、)）:：])\s*")

# 设元行的引导词：「设 y=kx」「令 k=y/x」「不妨设 y=kx」。
# 这类行不是推导的一步，是**换了未知数**——后面的式子跟原题不在一个论域里，
# 拿原题条件去比必然"对不上"。齐次方程用设比值法解，整条正确过程会因此
# 从这一行起全判 ✗。所以先认出来，不硬判。
_DECLARE_LEAD = re.compile(r"^(?:不妨设|假设|设|令|取)")

# 学生写的"得分点"类型，用于给这一步贴标签（只展示，不参与判定）
_ACTIONS: list[tuple[str, str]] = [
    ("移项", "移项"),
    ("同乘", "两边同乘"),
    ("同除", "两边同除"),
    ("除以", "两边同除"),
    ("乘以", "两边同乘"),
    ("两边平方", "两边平方"),
    ("平方", "两边平方"),
    ("开方", "开方"),
    ("配方", "配方"),
    ("因式分解", "因式分解"),
    ("分解因式", "因式分解"),
    ("代入", "代入"),
    ("通分", "通分"),
    ("求导", "求导"),
    ("积分", "积分"),
    ("化简", "化简"),
    ("整理", "整理"),
    ("合并同类项", "合并同类项"),
    ("判别式", "判别式"),
]

# 「由①」「代入(2)」——学生在引用前面的某一步
_REF = re.compile(r"(?:由|代入|将|把)\s*[①②③④⑤⑥⑦⑧⑨⑩]")
_REF_NUM = re.compile(r"(?:由|代入|将|把)\s*[（(]\s*(\d+)\s*[)）]")


def _guard(fn, default=None, timeout: float = _GUARD):
    """带超时的符号运算。simplify 遇到病态表达式会卡很久，不能让它拖住整页。"""
    box: dict = {}
    done = threading.Event()

    def work() -> None:
        try:
            box["v"] = fn()
        except Exception:  # noqa: BLE001
            box["v"] = default
        finally:
            done.set()

    threading.Thread(target=work, daemon=True).start()
    done.wait(timeout)
    return box.get("v", default)


# ------------------------------------------------------------------ 切行与读行


def split_lines(text: str) -> list[str]:
    """把一大段过程切成行。整页粘贴时行尾空白和空行都得清掉。"""
    out = []
    for line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = line.strip()
        if s:
            out.append(s)
    return out


def _tidy(s: str) -> str:
    s = _ENUM.sub("", s or "").strip()
    s = _LEAD_WORD.sub("", s).strip()
    # 学生写「x=2 或 x=3」，latex2sympy 不认「或」
    s = re.sub(r"\\text\s*\{\s*或\s*\}", " 或 ", s)
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    return s.strip()


def _split_out(s: str, seps: tuple[str, ...] = ("或", ",", "，", ";", "；")) -> list[str]:
    """按顶层分隔符切开，括号里的逗号不切（\\frac{a}{b}、\\begin{cases}）。"""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
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
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        buf.append(ch)
        i += 1
    out.append("".join(buf))
    return [x.strip() for x in out if x.strip()]


# 一行里并排写的几个答案：`x=-1  x=6`。等号两边都是"字母 = 数"才算数，
# 这样 `2x=6` 里那个 `2x` 不会被误认成"变量 = 值"。
_ASSIGN_TOK = re.compile(r"([A-Za-z])\s*=\s*(-?\d+(?:\.\d+)?|[-+]?[A-Za-z](?![A-Za-z]))")


def _split_assigns(s: str) -> list[str]:
    """空格隔开的多个赋值切开来。

    学生写两个根时，常常就并排写 `x=-1  x=6`，中间不写"或"也不写逗号。
    原来只认"或/逗号"，这一行整行读不出，判定栏写"读不出"，
    可下面的结论又写"你写的：x=-1、x=6"——一边说没读出来，一边把它当答案说了。
    两种说法打架，看着就像系统在乱判。切开来，判定和结论才对得上。

    只在整个字符串**正好**由几个"字母=数"拼成时才切，中间只许夹空格/逗号：
    宁可漏切判"读不出"，也不能把 `x=2y` 这种切成 `x=2` 加半个 `y`，
    那会凭空造出一个学生没写的答案。
    """
    if s.count("=") < 2:
        return [s]
    # 识别结果是 LaTeX 时，两个答案之间常是 `\quad` 而不是空格。
    # 不先归一，正则看到的间隔符是 `\quad`，整行又切不开。
    s = re.sub(r"\\qquad\b|\\q?quad\b|\\[,;! ]|~", " ", s)
    hits = list(_ASSIGN_TOK.finditer(s))
    if len(hits) < 2:
        return [s]
    if sum(1 for _ in re.finditer("=", s)) != len(hits):
        return [s]
    pieces, last = [], 0
    for m in hits:
        if s[last : m.start()].strip(" \t,，、;；"):
            return [s]
        pieces.append(m.group(0).replace(" ", ""))
        last = m.end()
    if s[last:].strip(" \t,，、;；"):
        return [s]
    return pieces


def _as_assign(pe: ParsedExpr):
    """x = 3 这种赋值，返回 (符号, 值)；否则 None。

    必须排除自引用：`x^2 = x` 反过来看很像「x = x^2」，但它是一条方程，
    不是学生声明的答案。早期版本把它读成赋值，整道题的题面就废了
    （评测用例 C04 抓到的）。赋值要求右边不出现左边的符号。
    """
    if not pe.ok or pe.kind != "equation":
        return None
    try:
        lhs, rhs = pe.obj[0]
    except Exception:  # noqa: BLE001
        return None
    for a, b in ((lhs, rhs), (rhs, lhs)):
        if isinstance(a, Symbol):
            free = getattr(b, "free_symbols", set())
            if a in free:            # 自引用，是方程不是赋值
                continue
            if not (free - {a}):
                return (a, b)
    return None


def _declared(text: str) -> str:
    for word, name in _ACTIONS:
        if word in (text or ""):
            return name
    return ""


@dataclass
class _Line:
    idx: int
    raw: str
    kind: str = "unparsed"     # equation / assign / system / expr / prose / unparsed
    pe: ParsedExpr | None = None
    assigns: list = field(default_factory=list)  # [(Symbol, Expr)]
    text: str = ""             # prose 的原文
    declared: str = ""
    role: str = ""
    verdict: str = "unknown"
    relation: str = ""
    reason: str = ""
    counter: str = ""
    missing: list = field(default_factory=list)
    wtype: str = ""            # 错因（只在判错时有值）
    row: list = field(default_factory=list)

    @property
    def display(self) -> str:
        if self.kind == "prose":
            return self.text
        return (self.pe.normalized if self.pe else self.raw) or self.raw


def _normalize_multi(pe: ParsedExpr) -> list:
    """`x=2, x=3` 被 latex 层当成了方程组，这里纠回"多个解"。"""
    if not pe.ok or pe.kind != "system":
        return []
    got = []
    var = None
    for lhs, rhs in pe.obj:
        if not isinstance(lhs, Symbol):
            return []
        if getattr(rhs, "free_symbols", set()) - {lhs}:
            return []
        if var is not None and lhs.name != var:
            return []
        var = lhs.name
        got.append((lhs, rhs))
    return got


def _strip_prose(body: str) -> tuple[str, str]:
    """把「解得 x=2」「两边同乘2得 2x²-10x+12=0」拆成 (人话, 式子)。

    必须在按「或」切分**之前**做。否则「解得 x=2 或 x=3」会被切成
    「解得 x=2」+「x=3」，前半句解析失败，整行退化成只认出 x=3——
    学生明明写了两个根，系统却判他漏解（评测用例 F02 抓到的冤案）。
    """
    marks = list(re.finditer(r"[\u4e00-\u9fff]+", body))
    if not marks:
        return "", body
    cut = marks[-1].end()
    head = body[:cut].strip()
    tail = body[cut:].strip(" 　,，。；:：、")
    if tail and re.search(r"[=<>]", tail):
        return head, tail
    return "", body


def _read(idx: int, raw: str) -> _Line:
    body = _tidy(raw)
    if not body:
        return _Line(idx=idx, raw=raw, kind="prose", text=raw)

    # 先按「或 / 逗号」切开：`x=2 或 x=3` 是两句话，不是一道方程组。
    # 剥离人话必须**在切分之后逐段做**：整行先剥的话，「或」会被当成人话，
    # 把「x=2 或 x=3」砍成只剩 x=3。这是评测抓到的第一个回归。
    pieces = _split_out(body)
    if len(pieces) == 1:
        pieces = _split_assigns(body)
    stripped = [_strip_prose(p) for p in pieces]
    head = next((h for h, _ in stripped if h), "")
    declared = _declared(head)
    bodies = [e for _, e in stripped]

    if len(bodies) > 1:
        parsed = [parse(e) for e in bodies]
        if all(p.ok for p in parsed):
            pairs = []
            others = []
            for p in parsed:
                a = _as_assign(p)
                (pairs if a else others).append(a or p)
            names = {s.name for s, _ in pairs}
            if pairs and not others and len(names) == 1:
                return _Line(idx=idx, raw=raw, kind="assign", pe=parsed[0],
                             assigns=pairs, text=head, declared=declared)

    pe = parse(bodies[0] if len(bodies) == 1 else body)
    if pe.ok:
        multi = _normalize_multi(pe)
        if multi:
            return _Line(idx=idx, raw=raw, kind="assign", pe=pe, assigns=multi,
                         text=head, declared=declared)
        one = _as_assign(pe)
        if one:
            return _Line(idx=idx, raw=raw, kind="assign", pe=pe, assigns=[one],
                         text=head, declared=declared)
        return _Line(idx=idx, raw=raw, kind=pe.kind, pe=pe, text=head,
                     declared=declared)

    # 中文还夹在式子中间/后头，整句读不出：只当说明行，不硬猜。
    if cjk_count(body) >= 1:
        return _Line(idx=idx, raw=raw, kind="prose", text=body, declared=declared)

    return _Line(idx=idx, raw=raw, kind="unparsed", text=body)


# ------------------------------------------------------------------ 判定原语


def _main_var(exprs, hint: Symbol | None = None) -> Symbol | None:
    if hint is not None:
        return hint
    syms = set()
    for e in exprs:
        syms |= set(getattr(e, "free_symbols", set()))
    if not syms:
        return None
    named = {s for s in syms if s.name.startswith("x")}
    pool = named or syms
    return sorted(pool, key=lambda s: (len(s.name), s.name))[0]


def _diffof(pe: ParsedExpr):
    """等式压成 lhs-rhs；方程组求和。和 latex.to_expr 同构，但保留 together。"""
    if not pe or not pe.ok:
        return None
    if pe.kind == "equation":
        lhs, rhs = pe.obj[0]
        return together(lhs - rhs)
    if pe.kind == "system":
        acc = 0
        for lhs, rhs in pe.obj:
            acc += together(lhs - rhs)
        return acc
    if pe.kind == "expression":
        return together(pe.obj[0])
    return None


def _solutions(pe: ParsedExpr, var: Symbol | None = None):
    d = _diffof(pe)
    if d is None:
        return var, None
    var = _main_var([d], hint=var)
    if var is None:
        return None, None
    got = _guard(lambda: sympy.solve(sympy.together(d), var))
    if got is None:
        return var, None
    return var, got


def _residual(pe: ParsedExpr, var: Symbol, val):
    """把 val 代进 pe 的每一条等式，返回残差列表。全 0 才算成立。"""
    if not pe or not pe.ok:
        return None
    pairs = pe.obj if pe.kind in ("equation", "system") else []
    out = []
    for lhs, rhs in pairs:
        r = _guard(lambda: simplify((lhs - rhs).subs(var, val)))
        out.append(r)
    return out


def _eq(a, b) -> bool:
    if a is None or b is None:
        return False
    v = _guard(lambda: simplify(a - b))
    if v is None:
        return False
    try:
        return bool(v == 0)
    except Exception:  # noqa: BLE001
        return False


def _num(x) -> str:
    """给人读的数：写 `1/2`，不写 `\\frac{1}{2}`。

    这些字符串会进 markdown 表格和 HTML 徽标，LaTeX 在那里不渲染，
    只会露出反斜杠。真要公式的地方走 WorkStep.latex，那里才是 LaTeX。
    """
    try:
        if getattr(x, "is_number", False):
            return str(x)
        return sympy.sstr(x)
    except Exception:  # noqa: BLE001
        return str(x)


def _scale_word(ratio) -> str:
    """把倍率说成人话：2 -> 同乘 2，1/2 -> 同除 2。"""
    try:
        if ratio.is_Rational and abs(ratio) < 1 and ratio != 0:
            return f"两边同除 {_num(1 / ratio)}"
    except Exception:  # noqa: BLE001
        pass
    return f"两边同乘 {_num(ratio)}"


def _one_pair(pe: ParsedExpr):
    """单条等式才取 (左, 右)；方程组没有"左右两边"这回事，返回 None。"""
    if pe is None or not pe.ok:
        return None
    pairs = pe.obj if pe.kind in ("equation", "system") else []
    if len(pairs) != 1:
        return None
    return pairs[0]


def _side_val(pair, var, val):
    """把 val 代进一条等式的左右两边，各自算成一个数。"""
    if pair is None:
        return None, None
    lhs, rhs = pair
    a = _guard(lambda: simplify(lhs.subs(var, val)))
    b = _guard(lambda: simplify(rhs.subs(var, val)))
    return a, b


def _counterexample(prev: ParsedExpr, cur: ParsedExpr, var: Symbol) -> str:
    """找一个满足上一步、却不满足这一步的取值。找不到就空串。

    说法必须是学生自己能验算的那种：**把数代进去，左右两边看等不等**。
    原来写的是"上一步两边相减为 0，这一步却是 -12 ≠ 0"——"两边相减"是
    把等式搬成 f(x)=0 的机器动作，中学生没这么算过，看到 -12 也不知道
    它是从哪冒出来的。改成"代进去左边是多少、右边是多少"，学生拿笔就能核。
    """
    _, sols = _solutions(prev, var)
    probes = list(sols or [])
    if not probes:
        probes = [sympy.Integer(0), sympy.Integer(1), sympy.Integer(-1),
                  sympy.Rational(1, 2), sympy.Integer(2), sympy.Integer(3)]
    for s in probes:
        if not getattr(s, "is_number", False):
            continue
        r = _residual(prev, var, s)
        if not r or any(v is None or v != 0 for v in r):
            continue  # 上一步本身就不成立，换一个
        r2 = _residual(cur, var, s)
        if r2 and any(v is not None and v != 0 for v in r2):
            bad = next(v for v in r2 if v is not None and v != 0)
            pair = _one_pair(cur)
            lv, rv = _side_val(pair, var, s)
            if lv is not None and rv is not None:
                # 不用 $\left. ... \right|_{x=2}$ 那种写法：竖线在 Markdown 表格里
                # 会被当成列分隔符，整张表会散架。学生要的也只是两个数。
                return (
                    f"把 {var.name}={_num(s)} 代进去：上一步成立；"
                    f"这一步左边算出来是 {_num(lv)}，右边是 {_num(rv)}，不相等"
                )
            return (
                f"把 {var.name}={_num(s)} 代进去，这一步算出来是 "
                f"{_num(simplify(bad))}，不等于 0"
            )
    return ""


def _guess_action(prev: ParsedExpr, cur: ParsedExpr, var: Symbol) -> str:
    """推断这一步做了什么变换。推不出来就返回空——不硬猜。"""
    d1, d2 = _diffof(prev), _diffof(cur)
    if d1 is None or d2 is None:
        return ""
    diff = _guard(lambda: simplify(d2 - d1))
    if diff is not None and diff == 0:
        return "整理（写法变了，两边相减仍相等）"
    # 先看倍数：两边同乘/同除是最常见的一种合法变形，
    # 而且它的差分往往不是常数，放到后面就会被误说成"移项"。
    ratio = _guard(lambda: simplify(d2 / d1))
    if ratio is not None and not (getattr(ratio, "free_symbols", set()) - {var}):
        if ratio != 0 and ratio != 1:
            return _scale_word(ratio)
    if diff is not None and not (getattr(diff, "free_symbols", set()) - {var}):
        if diff.is_number and var is not None:
            try:
                sign = "加" if diff > 0 else "减"
                return f"两边{sign} {_num(sympy.Abs(diff))}（移项/合并同类项）"
            except Exception:  # noqa: BLE001
                pass
        return "两边同加减一个式子（移项/合并同类项）"
    return ""


_TERM_NAME = {0: "常数项", 1: "一次项", 2: "二次项", 3: "三次项"}


def _term_at(k: int, var: Symbol) -> str:
    name = _TERM_NAME.get(k)
    if name:
        return name
    return f"{var.name}^{k} 项"


def _poly_of(pe: ParsedExpr, var: Symbol):
    """等式整理成关于 var 的多项式。方程组、非多项式都返回 None。"""
    pair = _one_pair(pe)
    if pair is None or var is None:
        return None
    lhs, rhs = pair
    try:
        return sympy.Poly(sympy.expand(lhs - rhs), var)
    except Exception:  # noqa: BLE001
        return None


def _is_factored(pe: ParsedExpr, var: Symbol) -> bool:
    """这一步是不是"把二次式写成两个括号相乘"。

    判据看的是**写法**，不是值：`(x-6)(x+1)` 是 Mul，而且至少一个因子含 var。
    只看 `is_Mul` 不够——`2x(x-3)` 也是，但那还是因式分解，一样该认。
    """
    pair = _one_pair(pe)
    if pair is None or var is None:
        return False
    body = pair[0] - pair[1]
    if not body.is_Mul:
        return False
    return sum(1 for f in body.args if var in f.free_symbols) >= 1 and any(
        f.is_Add for f in body.args
    )


def _factor_diag(prev: ParsedExpr, cur: ParsedExpr, var: Symbol) -> str:
    """把二次式拆括号拆错时，指出**是哪一项**对不上。

    这一步真的不是"移项变号"。学生做的是因式分解，错在常数项：
    `x²-5x+6` 拆成 `(x+1)(x-6)`，乘开是 `x²-5x-6`——一次项 -5x 是对的，
    常数项从 +6 变成了 -6。说成"移项/变号"会把学生引到完全无关的地方去找错，
    他检查十遍移项也不会发现问题。这是判错理由，不能含糊。
    """
    if var is None or not _is_factored(cur, var):
        return ""
    p1 = _poly_of(prev, var)
    p2 = _poly_of(cur, var)
    if p1 is None or p2 is None or p1.degree() < 2 or p1.degree() != p2.degree():
        return ""
    c1, c2 = p1.all_coeffs()[::-1], p2.all_coeffs()[::-1]
    bad = [k for k in range(len(c1)) if not _eq(c1[k], c2[k])]
    if not bad:
        return ""
    if len(bad) == 1:
        k = bad[0]
        head = (
            f"这一步是因式分解，{_term_at(k, var)}对不上："
            f"乘开是 ${to_latex(c2[k])}$，原式是 ${to_latex(c1[k])}$"
        )
    else:
        # 两项以上都错时逐个点名反而啰嗦，直接把两个整式并排：
        # `(x-2)(x-4)` 乘开是 x²-6x+8，一眼看出比原式多了什么。
        names = "、".join(_term_at(k, var) for k in bad)
        head = (
            f"这一步是因式分解，{names}都对不上："
            f"乘开是 ${to_latex(sympy.expand(p2.as_expr()))}$，"
            f"原式是 ${to_latex(sympy.expand(p1.as_expr()))}$"
        )
    # 能分解成整数括号时，直接给出该写成的样子——比"再想想"有用得多。
    _, sols = _solutions(prev, var)
    roots = [s for s in (sols or []) if getattr(s, "is_rational", False)]
    if len(roots) == 2 and len(set(roots)) == 2:
        shapes = "".join(
            rf"\left({var.name}{'' if r == 0 else ('-' if r > 0 else '+')}"
            rf"{'' if abs(r) == 1 else _num(abs(r))}\right)"
            for r in sorted(roots)
        )
        head += f"；${shapes}=0$ 才等于原式"
    return head


def _wrong_type(prev: ParsedExpr, cur: ParsedExpr, var) -> str:
    """错在哪：差一个常数 → 移项符号反了；差一个倍数 → 漏乘漏除。"""
    fact = _factor_diag(prev, cur, var)
    if fact:
        return fact
    d1, d2 = _diffof(prev), _diffof(cur)
    if d1 is None or d2 is None:
        return ""
    diff = _guard(lambda: simplify(d2 - d1))
    if diff is not None and getattr(diff, "is_number", False) and diff != 0:
        return f"两边差了一个常数 {_num(sympy.Abs(diff))}，典型的移项变号/漏乘错误"
    ratio = _guard(lambda: simplify(d2 / d1))
    if (
        ratio is not None
        and not getattr(ratio, "free_symbols", set())
        and ratio not in (0, 1)
    ):
        return f"两边差了 {_num(ratio)} 倍，像是漏乘或漏除"
    return ""


def _with_declared(decl: str, action: str) -> str:
    """把学生嘴上说的和式子上实际发生的对上。

    对不上不判错——推断本身是启发式的，拿一个猜出来的结论去否定学生，
    就成了另一种"说得很顺但是错的"。只并列摆出来，让人自己看。
    """
    if not decl:
        return action
    if not action:
        return f"你写的是「{decl}」。这一步具体怎么变的，系统推不出来，只验了结果"
    if decl[:2] in action or action[:2] in decl:
        return f"「{decl}」—— {action}"
    return f"你写的是「{decl}」，按式子的变化看更像「{action}」"


def _as_sol_list(s):
    """把 solve 的返回统一成解列表；算不出来的返回 None。

    sympy.solve 对多元、含参或带条件的式子会返回 And / ConditionSet /
    EmptySet 之类，其中一部分根本不可迭代。拿它去 list() 会抛 TypeError，
    整页批改直接崩。这里宁可返回 None 让调用方跳过，也不猜。
    """
    if s is None:
        return None
    try:
        return list(s)
    except TypeError:
        return None


def _relate(prev: ParsedExpr, cur: ParsedExpr, var: Symbol | None) -> tuple[str, str, list]:
    """相邻两式的关系：(relation, 人话, 漏掉的解)。"""
    d1, d2 = _diffof(prev), _diffof(cur)
    if d1 is None or d2 is None:
        return "unknown", "这一段读不出算式，跳过", []

    same = _guard(lambda: simplify(d2 - d1))
    if same is not None and same == 0:
        act = _guess_action(prev, cur, var) if var is not None else ""
        return "equiv", act or "等价改写（两边相减仍相等）", []

    v1, s1 = _solutions(prev, var)
    v2, s2 = _solutions(cur, var)
    var = var or v1 or v2

    # solve 对多元/含参式子会返回 And、ConditionSet 这类非序列对象，
    # list(...) 会直接抛 TypeError。解集算不出来只能跳过，不能顺势判学生错。
    s1s, s2s = _as_sol_list(s1), _as_sol_list(s2)
    if s1s is None or s2s is None:
        return "unknown", "这一步的解集算不出来，跳过", []

    if len(s1s) == len(s2s) and all(
        any(_eq(a, b) for b in s2s) for a in s1s
    ):
        act = _guess_action(prev, cur, var) if var is not None else ""
        return "same_solution", act or "同解变形（解集没变）", []
    # 这一步的解都在上一步里，但上一步还有别的解 → 丢了
    lost = [a for a in s1s if not any(_eq(a, b) for b in s2s)]
    if lost and all(any(_eq(b, a) for a in s1s) for b in s2s):
        return "lost", "这一步只保留了一部分解", lost
    extra = [b for b in s2s if not any(_eq(b, a) for a in s1s)]
    if extra and all(any(_eq(a, b) for b in s2s) for a in s1s):
        return "gain", "这一步多出了上一步没有的解（增根）", extra

    if var is not None:
        cex = _counterexample(prev, cur, var)
        if cex:
            return "wrong", "这一步和上一步不同解", []
    return "wrong", "这一步对不上上一步", []


# ------------------------------------------------------------------ 线性方程组
#
# 单方程是一条**递推链**：每一行都从上一步变出来，拿相邻两行比解集就够。
# 方程组不是链，是**条件集**：前两行是题给的并列条件，谁也不是从谁推出来的。
#
# 拿递推去判会出这种事：`x+y=5` 后面接 `x-y=1`，系统说"这一步对不上上一步"，
# 于是一份完全正确的方程组解答被判成 wrong。这不是学生错，是模型错——
# 而这类错最伤信任：学生越是老老实实一步一行抄题，越容易被判错。
#
# 所以方程组另走一条路：维护一个"已知条件集"，每多一行就联立一次，看解集怎么变。
#   不变        → 有效推导（消元、移项、同乘）
#   变小且非空  → 开头那几行是并列条件，后面的是在解出具体值
#   变空        → 与条件矛盾，这一步不成立
# 判据是联立后的解集，不是两行各自长什么样，所以谁先谁后、怎么变形都不影响。


def _pairs_of(pe) -> list[tuple]:
    """一条断言拆成若干 (lhs, rhs)。方程组就是多条并列。"""
    if pe is None or not pe.ok or pe.kind not in ("equation", "system"):
        return []
    return list(pe.obj)


def _symbols_of(pairs) -> set:
    got: set = set()
    for lhs, rhs in pairs:
        got |= (lhs - rhs).free_symbols
    return got


def _is_linear(pairs, vars_) -> bool:
    """每个变量都不超过一次。非线性走原来的相邻比较，不在这里硬套。"""
    if not vars_:
        return False
    for lhs, rhs in pairs:
        try:
            p = sympy.Poly(sympy.expand(lhs - rhs), *vars_)
        except (sympy.PolynomialError, TypeError, ValueError):
            return False
        if p.total_degree() > 1:
            return False
    return True


def _linsolve(pairs, vars_):
    """线性方程组的解集（可能带自由变量）。算不出来返 None——那是不知道，不是错。"""
    eqs = [sympy.expand(lhs - rhs) for lhs, rhs in pairs]
    try:
        s = sympy.linsolve(eqs, vars_)
    except Exception:  # noqa: BLE001
        return None
    if s == sympy.EmptySet:
        return sympy.EmptySet
    return s


def _row_of(x) -> list:
    """linsolve 吐出来的是 sympy.Tuple，不是 Python tuple，不能直接解包/遍历。"""
    if isinstance(x, sympy.Tuple):
        return list(x.args)
    if isinstance(x, (tuple, list)):
        return list(x)
    return [x]


def _is_point(sol, vars_) -> bool:
    """解集是不是唯一一组具体数值。欠定（含自由变量）就不算。"""
    if sol is None or sol is sympy.EmptySet:
        return False
    try:
        rows = list(sol)
    except TypeError:
        return False
    if len(rows) != 1:
        return False
    return all(not (sympy.expand(v).free_symbols & set(vars_)) for v in _row_of(rows[0]))


def _sol_show(sol, vars_) -> str:
    """解集说成人话。参数解也照写，不假装它是几个数。"""
    if sol is None:
        return "算不出来"
    if sol is sympy.EmptySet:
        return "无解"
    parts = []
    for row in sorted(sol, key=lambda t: sympy.sstr(t)):
        cells = _row_of(row)
        parts.append("，".join(f"{v.name} = {_num(x)}" for v, x in zip(vars_, cells)))
    return "；".join(p for p in parts if p) or "任意解"


def _pair_same(a, b) -> bool:
    """两条等式是不是同一个方程（等式两边相减之差为 0）。"""
    try:
        return bool(sympy.simplify((a[0] - a[1]) - (b[0] - b[1])) == 0)
    except Exception:  # noqa: BLE001
        return False


def _mark_declares(items: list[_Line], ppe) -> tuple[set[int], set]:
    """挑出「设了新符号」的行，返回 (这些行的序号, 新符号集合)。

    只认真正引入**题面没有的符号**的行：`设 y=kx` 的 k 是新的，算；
    `设 x=2` 只是给原变量赋值，本来就是正常的一步，不算。

    为什么要单独认：设元之后的式子（`x²+2kx²+k²x²=0`、`1+2k+k²=0`）跟原题
    不在同一个论域里，拿原题条件验算必然"对不上"。这不是学生写错了，是
    批改这边换了参照系——按项目那条"识别有把握才判错"，这里必须标待确认。
    """
    known = _symbols_of(_pairs_of(ppe))
    ids: set[int] = set()
    fresh: set = set()
    for it in items:
        if it.pe is None or not it.pe.ok:
            continue
        if not _DECLARE_LEAD.match((it.text or "").strip()):
            continue
        new = _symbols_of(_pairs_of(it.pe)) - known
        if new:
            ids.add(it.idx)
            fresh |= new
    return ids, fresh


def _looks_like_system(items, ppe) -> bool:
    """这一段是在解方程组，还是解单方程。

    两个未知数以上、且全都是一次，才算。一元二次那种（含参也算多项）不进来——
    它的相邻两步就是等价变形，用递推判得挺准，没必要换路径。
    """
    pairs = _pairs_of(ppe)
    vars_ = _symbols_of(pairs)
    lines = [it for it in items if it.pe is not None and it.pe.ok]
    for it in lines:
        vars_ |= _symbols_of(_pairs_of(it.pe))
    vars_ = sorted(vars_, key=lambda s: s.name)
    if len(vars_) < 2:
        return False

    allp = list(pairs)
    for it in lines:
        allp.extend(_pairs_of(it.pe))
    if not _is_linear(allp, vars_):
        return False

    if ppe.kind == "system":
        return True
    return any(it.kind == "system" for it in lines) or len(lines) >= 2


def _grade_linear_system(items, res: WorkCheck, ppe, t0: float) -> WorkCheck:
    """方程组的判法：条件集 + 联立。"""
    orig = _pairs_of(ppe)
    facts = list(orig)
    vars_ = _symbols_of(facts)
    for it in items:
        if it.pe is not None and it.pe.ok:
            vars_ |= _symbols_of(_pairs_of(it.pe))
        for sym, _ in it.assigns:
            vars_.add(sym)
    vars_ = sorted(vars_, key=lambda s: s.name)

    base_sol = _linsolve(facts, vars_)
    res.problem = res.problem or ppe.normalized
    res.expected = [_sol_show(base_sol, vars_)] if base_sol is not None else []

    # 题面自己就矛盾时，往下每一步都会"联立后无解"，一路判错——可学生没错，
    # 是题抄错了或题目本身无解。这种情况不给对错结论。
    if base_sol == sympy.EmptySet:
        for it in items:
            it.verdict = "unknown"
            it.reason = "题面本身无解，无法逐行验算"
            it.row = ["跳过", "—"]
            it.role = it.role or ""
        res.grade = "gap"
        res.headline = "题面本身无解，没法批改"
        res.advice = (
            "给出的几个方程互相矛盾（例如 x+y=1 和 x+y=3 同时成立）。"
            "先核对题目有没有抄错，再批改过程。"
        )
        res.steps = [_to_step(it) for it in items]
        res.timings["total"] = time.time() - t0
        return res

    res.notes.append("这是方程组，按「与前面的条件联立后解集怎么变」逐步判，不要求相邻两行同解")

    open_zone = True       # 还在"题给条件"区：这之前的行列都是抄题
    first_err = -1
    got: list[tuple] = []

    for it in items:
        if it.role == "problem":
            it.verdict = "note"
            it.reason = "题面"
            it.row = ["题面", "—"]
            continue

        if it.kind == "prose":
            it.verdict = "note"
            it.reason = "说明行，不参与验算"
            it.row = ["说明", it.declared or "—"]
            continue

        if it.kind == "unparsed":
            it.verdict = "unknown"
            it.reason = "这一行读不出算式，跳过"
            it.row = ["读不出", "跳过"]
            continue

        pairs = [(s, v) for s, v in it.assigns] if it.kind == "assign" else _pairs_of(it.pe)
        if not pairs:
            it.verdict = "unknown"
            it.reason = "这一行读不出算式，跳过"
            it.row = ["读不出", "跳过"]
            continue

        new = _linsolve(facts + pairs, vars_)
        if base_sol is None or new is None:
            it.verdict = "unknown"
            it.reason = "这一步的解集算不出来，跳过"
            it.row = ["跳过", "—"]
            continue

        # 答案按学生写下的记，不管成不成立——"答案对但过程不通"要判得出来，
        # 靠的正是把这两件事分开记。
        if it.kind == "assign":
            got.extend(it.assigns)

        if new == sympy.EmptySet:
            it.verdict = "wrong"
            it.counter = "与前面的条件联立后无解"
            it.reason = "把这一步和前面的条件放在一起，没有任何一组数能满足"
            it.row = ["✗ 不成立", it.counter]
            if first_err < 0:
                first_err = it.idx
            continue

        if new == base_sol:
            it.verdict = "ok"
            it.relation = "equiv"
            # 解集没变有两种来路，说成同一句话会骗人：
            # 抄题（本来就在条件里）和推导（消元/移项得来）。
            dup = all(any(_pair_same(p, q) for q in facts) for p in pairs)
            if it.kind == "assign":
                body = "，".join(f"{s.name}={_num(v)}" for s, v in it.assigns)
                it.reason = f"{body} 代回前面的条件成立"
            else:
                it.reason = _with_declared(
                    it.declared,
                    "与题面条件一致（照抄，不是推导）" if dup
                    else "与前几式联立，解集没变（消元/移项/同乘）",
                )
            it.row = ["✓ 成立", it.reason]
            open_zone = False
            continue

        # 解集变小：要么是题给的并列条件，要么是解出了一部分。
        alone = _linsolve(pairs, vars_)
        if open_zone and alone is not None and alone != sympy.EmptySet and not _is_point(alone, vars_):
            # 这一行自己解出来是"一条线"，联立后才缩小——这是在抄题，不是推导。
            it.verdict = "ok"
            it.reason = "题给条件：与上一行并列，不是从上一步推出来的"
            it.row = ["✓ 条件", it.reason]
            facts = facts + pairs
            base_sol = new
            res.notes.append(
                f"第 {it.idx} 行当作了题给条件（和上一行并列）。"
                "如果题干另给了方程组，把它填到上面的题面框里，判得更准"
            )
            continue

        if _is_point(new, vars_) and not _is_point(base_sol, vars_):
            it.verdict = "ok"
            it.relation = "same_solution"
            it.reason = "与前几式联立后解出唯一解：" + _sol_show(new, vars_)
            it.row = ["✓ 成立", it.reason]
            facts = facts + pairs
            base_sol = new
            open_zone = False
            continue

        # 剩下的：比前面的条件更严，会把前面的解丢掉
        it.verdict = "wrong"
        it.relation = "lost"
        it.counter = f"这一步只剩 {_sol_show(new, vars_)}"
        it.reason = "这一步比前面的条件更严，会丢掉前面成立的那些解"
        it.row = ["✗ 丢解", it.counter]
        if first_err < 0:
            first_err = it.idx

    # 答案核验：学生写的值能不能满足**原题条件**（不是被 assign 扩充过的 facts，
    # 否则自己写的答案代回去当然成立）
    res.got = [f"{s.name}={_num(v)}" for s, v in got]
    have = {s.name for s, _ in got}
    answer_ok = False
    if len(have) >= len(vars_):
        sub = {s: v for s, v in got}
        ok_all = True
        for lhs, rhs in orig:
            r = _guard(lambda: simplify((lhs - rhs).subs(sub)))
            if r is None or r != 0:
                ok_all = False
                break
        answer_ok = ok_all

    res.first_error = first_err
    has_wrong = first_err >= 0

    if not got:
        res.grade = "wrong" if has_wrong else "partial"
        res.error_type = "未写答案" if not has_wrong else "变形不等价"
        res.headline = "过程里没读到最终答案"
        res.advice = (
            "把答案写成 x=…、y=… 的形式，系统才能和标准解对照。"
            + (f"这道题的标准解是 {'；'.join(res.expected)}。" if res.expected else "")
        )
    elif has_wrong and answer_ok:
        res.grade = "fake"
        res.error_type = "伪推导"
        res.headline = "过程有一步不成立，但最终答案是对的"
        res.advice = (
            f"第 {first_err} 步开始推不通，答案却是对的——答案不是从这一步往下推出来的。"
            "考试按步骤给分，这种写法会被扣过程分。"
        )
    elif has_wrong:
        res.grade = "wrong"
        res.error_type = "变形不等价"
        res.headline = f"第 {first_err} 步开始不成立"
        res.advice = (
            "从标 ✗ 的那一步往下的推导都不能采信。"
            "解方程组时两式相减最容易错号，先把这一步改对再往下推。"
        )
    elif answer_ok:
        res.grade = "pass"
        res.headline = "过程逐行成立，答案也对"
        res.advice = "每一步都和前面的条件合得上，解也满足两个方程。"
    elif len(have) < len(vars_):
        res.grade = "partial"
        res.error_type = "漏解"
        res.headline = "过程成立，但只解出了 " + "、".join(sorted(have))
        res.advice = (
            f"这个方程组有 {'、'.join(v.name for v in vars_)} 两个未知数，"
            "两个都要解出来并写进答案。"
        )
    else:
        res.grade = "wrong"
        res.error_type = "答案错"
        res.headline = "过程看着通，但答案代回原方程不成立"
        res.advice = "把解分别代回原来的每个方程验一遍，看是哪一个不满足。"

    if first_err > 0:
        for it in items:
            if it.idx > first_err and it.verdict == "ok":
                it.reason = f"局部成立；第 {first_err} 步已错，这一步不影响结论的可靠性"
                it.row = ["✓ 局部", it.reason]

    res.steps = [_to_step(it) for it in items]
    res.timings["total"] = time.time() - t0
    return res


# ------------------------------------------------------------------ 主流程


def grade(
    work: str | list[str],
    problem: str = "",
    problem_image: str = "",
) -> WorkCheck:
    """批改一整套过程。

    work 可以是整段文本（按行切），也可以是行列表。
    problem 为空时，把第一条可读的等式行当题面。
    """
    t0 = time.time()
    lines = split_lines(work) if isinstance(work, str) else [x for x in work if (x or "").strip()]
    res = WorkCheck(problem=problem.strip(), problem_image=problem_image)

    if not lines and not res.problem:
        res.grade = "gap"
        res.headline = "没有可批改的内容"
        res.advice = "贴入学生的解题过程，一行一步；或者至少给一道题。"
        return res

    items = [_read(i + 1, ln) for i, ln in enumerate(lines)]

    # 题面：显式给了就用它，否则拿第一条等式行
    ppe: ParsedExpr | None = parse(res.problem) if res.problem else None
    if ppe is not None and not ppe.ok:
        res.notes.append(f"题面读不出算式（{ppe.error}），只做过程内部自检")
        ppe = None
    if ppe is None:
        first = next(
            (it for it in items if it.kind in ("equation", "system") and it.pe and it.pe.ok),
            None,
        )
        if first is not None:
            ppe = first.pe
            first.role = "problem"
            res.problem = first.display
    if ppe is None:
        res.grade = "gap"
        res.headline = "整段过程里没有一道能读出来的式子"
        res.advice = (
            "批改要有可验算的对象。纯文字的答案（“…所以答案是 2”）请写成 x=2 这样的式子，"
            "或者把题面单独贴到上面的输入框。"
        )
        res.steps = [_to_step(it) for it in items]
        res.timings["total"] = time.time() - t0
        return res

    res.problem = res.problem or (ppe.normalized if ppe.ok else "")
    res.problem_ok = True

    # 先认出"设了新符号"的行。它们和原题不在一个论域，不能拿去跟题面比。
    declare_ids, introduced = _mark_declares(items, ppe)

    # 题面单独给了的话，过程里那一行还没被认出来。不标出来，它会显示成
    # "第一处推导，没有上一步可比"——把题面说成推导，整张表就看着像错的。
    #
    # 两级匹配，先严后宽：
    #   1) 字面归一后完全相同 —— 这才像"照抄题目"；
    #   2) 同解集 —— 题目被换了个等价写法抄下来。
    # 顺序不能反。`x+y=5, x-y=1` 的题面下，`2x=6` 与它同解，宽松匹配会把
    # 学生的**推导**当成题面，那一步就永远不参与判定了。
    if not any(it.role == "problem" for it in items):
        want = (ppe.normalized or "").replace(" ", "")
        cands = [
            it for it in items
            if it.pe is not None and it.pe.ok and it.kind in ("equation", "system")
        ]
        hit = next((it for it in cands if (it.pe.normalized or "").replace(" ", "") == want), None)
        if hit is None:
            # 题面是多方程时，字面永远对不上整串，只能对上其中一条。
            # 要求"这一行的每条等式都在题面里"，不能反过来——反过来会把
            # 学生会解出来的那一步（`2x=6`）当成题面。
            want_pairs = _pairs_of(ppe)
            for it in cands:
                got = _pairs_of(it.pe)
                if got and all(any(_pair_same(p, q) for q in want_pairs) for p in got):
                    hit = it
                    break
        if hit is None:
            hit = next(
                (it for it in cands if _guard(lambda: equivalent(it.pe, ppe), False)), None
            )
        if hit is not None:
            hit.role = "problem"

    # 方程组不是递推链，是条件集，换个判法。
    # 放在题面标记完之后：那一步已经把"哪一行是抄题"认出来了，方程组要用。
    if _looks_like_system(items, ppe):
        return _grade_linear_system(items, res, ppe, t0)

    # 独立求一遍标准解，用来对照最终答案；顺便拿到"正确做法"
    sr = solve_parsed(ppe)
    res.expected = list(sr.solutions)
    var, exp_vals = _solutions(ppe)

    # 逐行判
    base: _Line | None = None          # 推导基准：最近一条非赋值的等式
    got: list = []                     # 学生给出的解
    first_err = -1

    for it in items:
        if it.role == "problem":
            it.verdict = "note"
            it.reason = "题面"
            it.row = ["题面", "—"]
            base = it
            continue

        if it.idx in declare_ids:
            names = "、".join(sorted(s.name for s in introduced))
            it.verdict = "unknown"
            it.reason = (
                f"这里设了新符号 {names}，等于换了个未知数往下算。"
                "系统的验算都以原题条件为准，这一段不硬判"
            )
            it.row = ["待确认", f"设了 {names}"]
            continue

        if it.kind == "prose":
            it.verdict = "note"
            it.reason = "说明行，不参与验算"
            it.row = ["说明", it.declared or "—"]
            continue

        if it.kind == "unparsed":
            it.verdict = "unknown"
            it.reason = "这一行读不出算式，跳过"
            it.row = ["读不出", "跳过"]
            continue

        # 用了新设符号的行（`1+2k+k²=0`、`k=-1`）：只有换回原题变量才谈得上验算。
        # 这里不判错，也不让它当基准——否则下一行会拿"含 k 的式子"跟它比，
        # 又冒出一串假错。
        if introduced and (_symbols_of(_pairs_of(it.pe)) & introduced):
            it.verdict = "unknown"
            it.reason = "这一步用的是新设的符号，系统的验算都以原题变量为准，这里不硬判"
            it.row = ["待确认", "含新设符号"]
            continue

        if it.kind == "assign":
            if base is None or base.pe is None:
                it.verdict = "unknown"
                it.reason = "前面没有可对照的式子，无法判断这个解对不对"
                it.row = ["无基准", "跳过"]
                continue
            bad = []
            for sym, val in it.assigns:
                # 学生写下的答案先记下来，不管它成不成立——
                # "答案对但过程不通"要判得出来，靠的正是这两件事分开记。
                got.append((sym, val))
                r = _residual(base.pe, sym, val)
                if not r:
                    bad.append((sym, val, "代不进去"))
                elif any(v is None for v in r):
                    bad.append((sym, val, "算不出来"))
                elif any(v != 0 for v in r):
                    nz = next(v for v in r if v != 0)
                    bad.append((sym, val, f"代回得 {_num(simplify(nz))} ≠ 0"))
            if bad:
                sym, val, why = bad[0]
                it.verdict = "wrong"
                it.counter = f"{sym.name}={_num(val)} {why}"
                it.reason = f"把 {sym.name}={_num(val)} 代回上一步，等式不成立"
                it.row = ["✗ 不成立", it.counter]
                if first_err < 0:
                    first_err = it.idx
            else:
                it.verdict = "ok"
                body = "，".join(f"{s.name}={_num(v)}" for s, v in it.assigns)
                it.reason = f"{body} 满足上一步那个式子"
                it.row = ["✓ 成立", body]
            continue

        # equation / expr：和基准比
        if base is None or base.pe is None:
            it.verdict = "unknown"
            it.reason = "第一处推导，没有上一步可比，默认它本身成立"
            it.row = ["基准", "—"]
            base = it
            continue

        rel, why, lost = _relate(base.pe, it.pe, var)
        it.relation = rel
        if var is None:
            var = _solutions(base.pe)[0]

        if rel in ("equiv", "same_solution"):
            it.verdict = "ok"
            it.reason = _with_declared(it.declared, why)
            it.row = ["✓ 成立", it.reason]
        elif rel == "lost":
            it.verdict = "partial"
            it.missing = lost
            it.reason = "这一步只写了一部分解：" + "、".join(_num(x) for x in lost)
            it.row = ["△ 漏解", "丢了 " + "、".join(_num(x) for x in lost)]
        elif rel == "gain":
            it.verdict = "wrong"
            it.reason = "这一步多出了上一步没有的解：" + "、".join(_num(x) for x in lost)
            it.counter = "增根 " + "、".join(_num(x) for x in lost)
            it.row = ["✗ 增根", it.counter]
            if first_err < 0:
                first_err = it.idx
        else:
            it.verdict = "wrong"
            wt = _wrong_type(base.pe, it.pe, var) if var is not None else ""
            it.wtype = wt
            it.reason = wt or why or "这一步和上一步不同解"
            it.counter = _counterexample(base.pe, it.pe, var) if var is not None else ""
            it.row = ["✗ 不成立", it.counter or it.reason]
            if first_err < 0:
                first_err = it.idx

        base = it

    # 答案：优先用 assign 行收集到的解；没有就用最后一条推导式的解集
    inferred_answer = False
    if not got:
        last = next(
            (x for x in reversed(items)
             if x.kind in ("equation", "system") and x.role != "problem"),
            None,
        )
        if last is not None and last.pe is not None:
            _, vals = _solutions(last.pe, var)
            if vals and var is not None:
                got = [(var, v) for v in vals]
                inferred_answer = True
                res.notes.append("过程里没有写成 x=… 的答案，按最后一个等式的解集判定")
    res.got = [f"{s.name}={_num(v)}" for s, v in got]

    # 总判定
    res.first_error = first_err
    has_wrong = first_err >= 0
    lost_any = any(x.verdict == "partial" for x in items)

    if exp_vals is None:
        res.grade = "unknown"
        res.headline = "过程内部检查完了，但题面求不出标准答案，无法判定对错"
        res.advice = (
            "该查的每一步都查了：标 △ / ✗ 的就是有问题的地方。"
            "只是这道题系统给不出标准解，所以不给「你对了/错了」的结论。"
        )
    else:
        missing = [e for e in exp_vals if not any(_eq(e, v) for _, v in got)]
        extra = [v for _, v in got if not any(_eq(v, e) for e in exp_vals)]
        covered = bool(got) and not missing and not extra

        if not got and exp_vals:
            missing = list(exp_vals)

        if not got:
            res.grade = "wrong" if has_wrong else "partial"
            res.error_type = "未写答案" if not has_wrong else _errtype(items, first_err)
            res.headline = "过程里没读到最终答案"
            res.advice = (
                "把答案写成 x=… 的形式（多个解用「或」隔开），系统才能和标准解对照。"
                f"这道题的标准解是 {'、'.join(res.expected)}。"
            )
        elif has_wrong and covered:
            res.grade = "fake"
            res.headline = "过程有一步不成立，但最终答案是对的"
            res.error_type = "伪推导"
            res.advice = (
                f"第 {first_err} 步开始推不通，答案却是对的——这类情况最常见的原因是"
                "答案不是从这一步往下推出来的（抄的、或者中途改了思路没改过程）。"
                "考试按步骤给分，这种写法会被扣过程分。"
            )
        elif has_wrong:
            res.grade = "wrong"
            res.headline = f"第 {first_err} 步开始不成立"
            res.error_type = _errtype(items, first_err)
            res.advice = (
                "从标 ✗ 的那一步往下的推导都不能采信，即使后面某步看起来对了。"
                "先把这一步改对，再往下重推。"
            )
        elif missing or lost_any:
            res.grade = "partial"
            miss = missing if missing else [
                x for it in items if it.verdict == "partial" for x in it.missing
            ]
            res.error_type = "漏解"
            shown = "、".join(f"{var.name}={_num(m)}" for m in miss) if miss else "—"
            res.headline = f"过程成立，但解不全，少了 {shown}"
            res.advice = (
                "解方程时，「两边同除」「开平方」「两边约掉一个因式」都会丢解；"
                "开方要带 ±，约分前先确认那个因式能不能为 0。"
            )
        elif covered:
            res.grade = "pass"
            if declare_ids:
                # 结论对、中间没验，这两件事必须分开说。
                # 只写"过程逐行成立"是假的：换元那几行系统根本没看。
                where = "、".join(str(i) for i in sorted(declare_ids))
                res.headline = f"换回原变量后的结论成立；设元那一段（第 {where} 步起）没有验算"
                res.advice = (
                    "设元等于换了个未知数，那几行的代数变换系统不比——它只认原题里的变量。"
                    "系统能保证的是最后换回来的结论跟原题一致，中间 k 算得对不对要自己再看一遍。"
                )
            elif inferred_answer:
                # 学生根本没写答案，是他从最后一行式子解出来的。说「答案也对」
                # 是把系统补的东西算到学生头上，这种便宜不能占（评测 C03）。
                res.headline = "过程逐行成立；答案没单独写，按最后一行式子解出的解集与标准解一致"
                res.advice = (
                    "每一步都能从上一步推出来。答案没写成 x=… 的形式，"
                    "这里是从过程最后一行解出来的——作业里还是把答案单独写出来更稳妥。"
                )
            else:
                res.headline = "过程逐行成立，答案也对"
                res.advice = "每一步都能从上一步推出来，没有跳步。"
        elif exp_vals and got:
            res.grade = "wrong"
            res.error_type = "答案错"
            res.headline = "过程看着通，但答案和标准解不一致"
            res.advice = (
                "标准解是 " + "、".join(f"{var.name}={_num(e)}" for e in exp_vals) +
                "。检查最后一步是不是抄错了，或者解的整理形式出了问题。"
            )
        else:
            res.grade = "unknown"
            res.headline = "过程检查完了，没能判定最终答案"
            res.advice = "把最终答案写成 x=… 的形式，系统才能和标准解对照。"

    if sr.ok and sr.steps:
        res.notes.append("标准解法见下方「正确做法」，那是独立求出来的，不依赖学生的过程")

    # 设元那一段没验算，得说出来。不说的话，表格里几行"待确认"会让人
    # 以为系统坏了；说清楚是"换了未知数、坐标系变了"，才知道怎么看这张表。
    if declare_ids:
        names = "、".join(sorted(s.name for s in introduced))
        where = "、".join(str(i) for i in sorted(declare_ids))
        res.notes.append(
            f"第 {where} 步设了新符号 {names}，之后换元得到的式子跟原题不在同一个"
            "未知数里，这一段没有验算；换回原变量之后的结论仍然照常判。"
        )

    # 第一步错之后，后面每一步即使"局部成立"也不能说明结论可靠。
    # 不标出来的话，表格里一片 ✓ 会让人以为只有一处小错。
    if first_err > 0:
        for it in items:
            if it.idx > first_err and it.verdict == "ok":
                it.reason = (
                    f"局部成立，但第 {first_err} 步已经错了，"
                    f"这里推得通也不能说明结论没受影响。{it.reason}"
                )
                it.row = ["✓ 局部", it.reason]

    res.steps = [_to_step(it) for it in items]
    res.timings["total"] = time.time() - t0
    return res


def _errtype(items: list[_Line], idx: int) -> str:
    it = next((x for x in items if x.idx == idx), None)
    if it is None:
        return ""
    if it.kind == "assign":
        return "代回不成立"
    if it.relation == "gain":
        return "增根"
    if it.wtype:
        # "因式分解"必须排在"常数"前面：拆括号的错法描述里也有"常数项"三个字，
        # 先命中"常数"就会把它说成移项变号，指错方向。
        if "因式分解" in it.wtype:
            return "因式分解错"
        return "移项/变号错" if "常数" in it.wtype else (
            "漏乘漏除" if "倍" in it.wtype else "变形不等价"
        )
    return "变形不等价"


def _to_step(it: _Line) -> WorkStep:
    latex = ""
    if it.pe is not None and it.pe.ok:
        latex = it.pe.normalized
    elif it.kind == "assign" and it.assigns:
        latex = "，".join(f"{s.name}={_num(v)}" for s, v in it.assigns)
    return WorkStep(
        idx=it.idx,
        raw=it.raw,
        kind=it.kind,
        role=it.role,
        latex=latex or it.raw,
        verdict=it.verdict,
        relation=it.relation,
        reason=it.reason or ("说明行，不参与验算" if it.kind == "prose" else ""),
        counter=it.counter,
        missing=[_num(x) for x in it.missing],
        declared=it.declared,
        row=it.row,
    )
