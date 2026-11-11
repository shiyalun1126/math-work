"""方程类：先把一个题型做透。

这个模块只回答两件事，其余一律往外推：

  1. 这道题（题目照片 / 贴进来的题干）是什么方程，答案是多少；
  2. 这份过程（照片 / 逐行文字）在方程这一类里站不站得住。

**为什么要先划范围。** 会解三次方程不等于能批改三次方程：判"这一步不成立"
要给出反例，反例要求解集可算、可比较。范围划不清，最典型的坏结果是
拿复数根去答初中题，或者拿"和我算的不一样"去冤枉一个用配方法的学生。
所以这里的 `Topic` 自带 coverage（in / edge / out），出了范围就明说，
不硬给结论——批改场景里，一句错的"你错了"比"我看不了"贵得多。

**题目 → 方程** 靠规则，不靠模型：学生抄的题目九成是"解方程："加一条式子，
把中文剥掉就行。只有应用题（通篇叙述、没有等号）才请模型列式，而且
模型只准输出方程，求解与验算仍然走 SymPy——答案必须能被代回验算，
不能被模型"讲"出来。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import sympy
from sympy import Poly, Symbol

from . import config
from .cache import Cache
from .latex import parse, to_latex
from .solve import solve as solve_parsed
from .solve import summary as solve_summary
from .types import ParsedExpr, SolveResult

# ---------------------------------------------------------------- 题型范围

# 这几类是这个工具的"包票"：答案、步骤、验算都按同一套规矩给。
COVERED = {
    "linear": "一元一次方程",
    "quadratic": "一元二次方程",
    "system": "二元一次方程组",
}

# 能算，但要多说一句的：答案对，边界要交代。
EDGE = {
    "rational": "分式方程（自动验增根）",
    "radical": "无理方程（两边平方后验根）",
    "param": "含参数方程（按参数分类讨论）",
    "identity": "恒等式（解集为全体实数）",
    "no-solution": "无解方程（解集为空）",
}

# 明确不做。不是算不出来，是算了也不敢判——见模块开头。
OUT = {
    "cubic": "三次及以上方程",
    "nonpoly": "非多项式方程（三角、指数、对数）",
    "nonlinear-system": "非线性方程组",
    "not-equation": "不是方程",
}

MAX_DEGREE = 2

# 中学书写里"未知数"基本只用这三个；其他符号（a、m、k…）按参数看。
_UNKNOWN_NAMES = ("x", "y", "z")

# 缓存版本号。判据一改就必须升版本，否则会拿旧结论回答新问题——
# 这种错最隐蔽：结果看起来正常，只是它是上一版算法算的。
CACHE_VERSION = "eq-answer-v3"  # v3：步骤里去掉"代回验证 0 = 0"，旧缓存的步骤得作废

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_TEXTLABEL = re.compile(r"\\text\{[^{}]*\}")
_CN_PUNCT = "，。；：、？！、“”‘’（）《》【】…—\u3000"
_EQ_SPLIT = re.compile(r"[,;]|\\quad|\\qquad|\\\\")

# 题干里常见的指令词。不剥掉的话，"解方程：3x+6=15" 整串进解析器必然失败，
# 用户看到"解析失败"会以为系统连这么简单的题都不会——其实是没读懂人话。
_INSTRUCTION = re.compile(
    r"^\s*(?:请|试|麻烦)?\s*"
    r"(?:解下列|解下面的|解这个|求解|解)(?:一元)?(?:一次|二次)?\s*(?:方程|方程组|等式)"
    r"(?:组)?\s*[:：]?\s*"
)

WORD_HINTS = (
    "多少", "几", "每", "共", "总", "平均", "倍", "比", "多", "少", "元", "支",
    "本", "个", "人", "小时", "千米", "米", "千克", "亩", "件", "页", "天",
)


@dataclass
class Topic:
    """这道题属于哪一类、在不在覆盖范围内。"""

    kind: str = "not-equation"      # linear / quadratic / system / rational / ...
    label: str = "不是方程"
    coverage: str = "out"           # in / edge / out
    degree: int = -1
    vars: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def inside(self) -> bool:
        return self.coverage in ("in", "edge")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "coverage": self.coverage,
            "degree": self.degree,
            "vars": self.vars,
            "note": self.note,
        }


def _degree(expr, x) -> int:
    try:
        return int(Poly(sympy.expand(expr), x).degree())
    except Exception:  # noqa: BLE001
        return -1


def _has_var(expr) -> bool:
    return bool(getattr(expr, "free_symbols", set()))


def classify(pe: ParsedExpr) -> Topic:
    """按解出来的式子判类型。判类型不调模型，也不看题面文字。"""
    if not pe.ok:
        return Topic("not-equation", OUT["not-equation"], "out", -1, [], pe.error or "解析失败")

    if pe.kind == "system":
        pairs = pe.obj or []
        vars_: list[str] = []
        linear = True
        for lhs, rhs in pairs:
            vars_ += [str(s) for s in sorted((lhs - rhs).free_symbols, key=str)]
            for s in (lhs - rhs).free_symbols:
                if _degree(lhs - rhs, s) > 1:
                    linear = False
        vars_ = sorted(set(vars_))
        if len(vars_) < 2:
            return Topic("equation", "一元方程", "edge", -1, vars_,
                         "只看到一个未知数，按单方程处理")
        if not linear:
            return Topic("nonlinear-system", OUT["nonlinear-system"], "out", -1, vars_,
                         "方程组里有二次项，解可能不止一组，本工具不判")
        return Topic("system", COVERED["system"], "in", 1, vars_, "")

    if pe.kind != "equation":
        return Topic("not-equation", OUT["not-equation"], "out", -1, [],
                     "这是一条式子，不是方程——没有等号，就没有「解」可求。")

    lhs, rhs = pe.obj[0]
    expr = lhs - rhs
    vars_ = [str(s) for s in sorted(expr.free_symbols, key=str)]
    try:
        num, den = sympy.fraction(sympy.together(expr))
    except Exception:  # noqa: BLE001
        num, den = expr, sympy.Integer(1)

    den_has_var = _has_var(den)
    rad_has_var = bool(num.atoms(sympy.Pow)) and any(
        (isinstance(p.exp, sympy.Rational) and p.exp.q > 1 and _has_var(p.base))
        for p in num.atoms(sympy.Pow)
    )

    if not vars_:
        if sympy.simplify(expr) == 0:
            return Topic("identity", EDGE["identity"], "edge", 0, [], "两边恒等")
        return Topic("no-solution", EDGE["no-solution"], "edge", 0, [],
                     "两边都是常数且不相等，方程无解")

    # 恒等式要先判：2(x-3)=2x-6 化简后是 0，用次数判会被当成"非多项式"，
    # 而它其实是同解变形——批改里这是最常见、也最容易被误判成错的一步。
    if sympy.simplify(expr) == 0:
        return Topic("identity", EDGE["identity"], "edge", 0, vars_, "两边恒等")

    # 谁是主元、谁是参数：只有一个 x/y/z 时，它就是主元，其余符号是参数。
    # 这条判据来自中学的书写惯例——「关于 x 的方程，a 是参数」。
    # x²+y²=1 不走这条：x、y 都是未知数，那是圆的方程，不是含参方程。
    core = [v for v in vars_ if v in _UNKNOWN_NAMES]
    if len(core) >= 2:
        x0 = Symbol(core[0])
        if _degree(num, x0) > 1:
            return Topic("nonlinear", "含两个未知数且非一次", "out", -1, vars_,
                         "两个未知数要成组出现才能定解；单独一条是关系式，不是方程题")
        return Topic("relation", "二元一次关系式", "edge", 1, vars_,
                     "一个方程两个未知数，解不唯一——下面给的是用另一个未知数表示的形式")

    if len(core) == 1:
        main = Symbol(core[0])
        params = [v for v in vars_ if v != core[0]]
    else:
        main = Symbol(vars_[0])
        params = vars_[1:]

    deg = _degree(num, main)

    if deg < 0:
        return Topic("nonpoly", OUT["nonpoly"], "out", -1, vars_,
                     "化简后不是多项式（三角 / 指数 / 对数这类），本工具不做")
    if deg == 0:
        return Topic("no-solution", EDGE["no-solution"], "edge", 0, vars_,
                     "化简后不含未知数且不恒等，方程无解")
    if deg > MAX_DEGREE:
        return Topic("cubic", OUT["cubic"], "out", deg, vars_,
                     f"这是 {deg} 次方程，超出本工具覆盖的一次 / 二次范围")
    if den_has_var:
        return Topic("rational", EDGE["rational"], "edge", deg, vars_,
                     "分母含未知数，解完要代回验根（增根会被剔除）")
    if rad_has_var:
        return Topic("radical", EDGE["radical"], "edge", deg, vars_,
                     "含根号，两边平方可能引入增根，已逐根代回验算")
    if params:
        # 含参：答案是一张分情形的表，不是一句「x = …」。漏一支就是错答案，
        # 所以这里不按「一次 / 二次」笼统答，交给 params.discuss() 逐支讨论。
        return Topic("param", EDGE["param"], "edge", deg, vars_,
                     f"含参数 {'、'.join(params)}，答案按参数分情形讨论；"
                     "每一支都取了实际参数值代回验过")
    if deg == 1:
        return Topic("linear", COVERED["linear"], "in", 1, vars_, "")
    return Topic("quadratic", COVERED["quadratic"], "in", 2, vars_, "")


# ---------------------------------------------------------------- 题目抽取


def _pieces(text: str) -> list[str]:
    """把题干切成"数学片段"，中文和中文标点当分隔符。

    不能直接对整句跑解析器："解方程：3x+6=15" 里 4 个汉字足够让
    latex2sympy2 崩掉。中文本来就不属于式子，切掉是保语义的。
    """
    t = _TEXTLABEL.sub(" ", text or "")
    t = re.sub(r"\$+", " ", t)
    t = _CJK.sub("|", t)
    for ch in _CN_PUNCT:
        t = t.replace(ch, "|")
    out = []
    for part in t.split("|"):
        part = part.strip(" \t\r\n,.;:·")
        if part and "=" in part:
            out.append(part)
    return out


def _split_multi(piece: str) -> list[str]:
    """一段里塞了多条方程（"x+y=3, x-y=1"）时再拆一次。"""
    if parse(piece).ok:
        return [piece]
    subs = [p.strip() for p in _EQ_SPLIT.split(piece) if p.strip()]
    if len(subs) > 1 and all(parse(s).ok for s in subs):
        return subs
    return [piece]


def find_equations(text: str) -> list[str]:
    """从题干里抽出所有能解析的方程，按出现顺序。"""
    out: list[str] = []
    for piece in _pieces(_INSTRUCTION.sub(" ", (text or "").strip())):
        for sub in _split_multi(piece):
            if parse(sub).ok:
                out.append(sub)
    return out


def as_system(eqs: list[str]) -> str:
    body = r" \\ ".join(e.strip() for e in eqs)
    return rf"\begin{{cases}} {body} \end{{cases}}"


def looks_like_word_problem(text: str) -> bool:
    """通篇是叙述、没有等号，但看得出在问一个数——这才值得请模型列式。"""
    t = (text or "").strip()
    if not t or find_equations(t):
        return False
    n_cjk = sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff")
    if n_cjk < 6:
        return False
    return any(h in t for h in WORD_HINTS) or "?" in t or "？" in t


@dataclass
class Equation:
    """从一段题目里抽出来的方程，连同它是怎么来的。"""

    latex: str = ""
    topic: Topic = field(default_factory=Topic)
    source: str = "text"      # text / image / llm
    provided: bool = True     # True=题面本来就有方程；False=模型从叙述里列出来的
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "latex": self.latex,
            "topic": self.topic.to_dict(),
            "source": self.source,
            "provided": self.provided,
            "note": self.note,
        }


def extract(text: str, *, source: str = "text") -> Equation:
    """题干 → 方程。抽不到就如实说抽不到，不猜。"""
    raw = (text or "").strip()
    if not raw:
        return Equation("", Topic("not-equation", "空", "out", -1, [], "没有输入"), source)

    stripped = _INSTRUCTION.sub(" ", raw).strip()
    if parse(stripped).ok:
        eq = stripped
    else:
        eqs = find_equations(raw)
        if not eqs:
            return Equation(
                "", Topic("not-equation", OUT["not-equation"], "out", -1, [],
                          "题面里没有可以解析的等式"), source,
                note="这段文字里找不到方程。若是应用题，请点「让它先列方程」。",
            )
        if len(eqs) == 1:
            eq = eqs[0]
        else:
            jointed = as_system(eqs)
            if parse(jointed).ok:
                eq = jointed
            else:
                eq = max(eqs, key=len)
                return Equation(eq, classify(parse(eq)), source,
                                note=f"题面里读到 {len(eqs)} 条等式，按最长的一条求解")

    pe = parse(eq)
    return Equation(eq, classify(pe), source, note="" if pe.ok else (pe.error or ""))


# ---------------------------------------------------------------- 应用题列式

SETUP_SYSTEM = (
    "你是中学数学老师，任务只有一个：把应用题翻译成方程。"
    "只输出 LaTeX，不要输出任何解释、不要输出答案、不要写单位。"
    r"未知数统一用 x；需要两个未知数时用 x 和 y，并写成方程组 "
    r"\begin{cases} ... \\ ... \end{cases}。"
    r"分数写成 \frac{分子}{分母}，乘号省略，等号两边都要写。"
    "拿不准就输出 UNKNOWN，不要编一个看起来像的方程。"
)

_CACHE = Cache(subdir="equation_setup")


def setup_equation(text: str, *, use_cache: bool = True) -> Equation:
    """叙述题 → 方程。模型只负责翻译，求解和验算照样走符号层。

    为什么敢用模型：它输出的是一条方程，不是答案。方程能不能站住，
    后面由 SymPy 解、代回验算，答案由我们给；模型说错也只是列式错，
    并且界面上会标明"方程是模型列的，请核对"。
    """
    import requests

    prompt = f"题目：{text.strip()}"
    key = ("setup", config.llm_model(), prompt)

    def _call() -> dict:
        r = requests.post(
            f"{config.llm_base().rstrip('/')}/chat/completions",
            json={
                "model": config.llm_model(),
                "messages": [
                    {"role": "system", "content": SETUP_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 200,
            },
            headers={"Authorization": f"Bearer {config.llm_key()}"},
            timeout=40,
        )
        if r.status_code != 200:
            return {"ok": False, "text": "", "note": f"HTTP {r.status_code}: {r.text[:120]}"}
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return {"ok": True, "text": content or "", "note": ""}

    if not config.has_llm():
        return Equation("", Topic("not-equation", "未列式", "out", -1, [],
                                  "没有配置模型，无法从叙述里列方程"), "llm",
                        provided=False, note="请在 .env 里配置 DASHSCOPE_API_KEY 或 LLM_API_KEY")

    try:
        payload = _CACHE.remember(_call, *key) if use_cache else _call()
    except Exception as exc:  # noqa: BLE001
        return Equation("", Topic("not-equation", "列式失败", "out", -1, [], str(exc)), "llm",
                        provided=False, note=f"调用模型失败：{type(exc).__name__}")

    if not payload.get("ok"):
        return Equation("", Topic("not-equation", "列式失败", "out", -1, [],
                                  payload.get("note", "")), "llm", provided=False,
                        note=payload.get("note", ""))

    content = (payload.get("text") or "").strip()
    if "UNKNOWN" in content.upper():
        return Equation("", Topic("not-equation", "列不出来", "out", -1, [],
                                  "模型认为这道题列不出方程"), "llm", provided=False,
                        note="这道题可能不是方程题，或者条件不全。")

    eq = extract(content, source="llm")
    if not eq.latex:
        return Equation("", Topic("not-equation", "列式失败", "out", -1, [],
                                  "模型输出不像方程"), "llm", provided=False,
                        note=f"模型的输出没能转成方程：{content[:80]}")

    eq.provided = False
    eq.note = "方程由题面文字转写（模型列式），请核对是否符合题意"
    return eq


# ---------------------------------------------------------------- 答题

@dataclass
class Answer:
    """一道方程题的答案，以及它凭什么可信。"""

    ok: bool = False
    text: str = ""                  # 原始题干
    latex: str = ""                 # 真正求解的方程
    topic: Topic = field(default_factory=Topic)
    solutions: list[str] = field(default_factory=list)
    answer_latex: str = ""
    steps: list[dict] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    branches: list[str] = field(default_factory=list)   # 含参方程的分情形答案
    explain: str = ""
    note: str = ""
    source: str = "text"
    provided: bool = True
    cached: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "text": self.text,
            "latex": self.latex,
            "topic": self.topic.to_dict(),
            "solutions": self.solutions,
            "answer_latex": self.answer_latex,
            "branches": self.branches,
            "checks": self.checks,
            "dropped": self.dropped,
            "explain": self.explain,
            "note": self.note,
            "source": self.source,
            "provided": self.provided,
            "cached": self.cached,
            "error": self.error,
            "steps": self.steps,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
        }


_ANSWER_CACHE = Cache(subdir="equation_answer")

# 分支之间用 "\quad ①" 隔开（params.discuss 的写法）。
_BRANCH_SEP = re.compile(r"\s*\\quad\s*(?=[①②③④⑤⑥⑦⑧⑨⑩])")


def _branches(solutions: list[str]) -> list[str]:
    text = " ".join(solutions).strip()
    if not text:
        return []
    parts = [p.strip() for p in _BRANCH_SEP.split(text) if p.strip()]
    if len(parts) == 1:
        parts = [p.strip() for p in re.split(r"；", text) if p.strip()]
    return parts


def _from_cache(ans: Answer, hit: dict) -> None:
    ans.ok = bool(hit.get("ok"))
    ans.solutions = list(hit.get("solutions") or [])
    ans.steps = list(hit.get("steps") or [])
    ans.checks = list(hit.get("checks") or [])
    ans.dropped = list(hit.get("dropped") or [])
    ans.branches = list(hit.get("branches") or [])
    ans.answer_latex = hit.get("answer_latex") or ""
    ans.explain = hit.get("explain") or ""


def _fix_param_note(ans: Answer) -> None:
    """解里带参数但不必分情形时，把「按参数分情形讨论」改成实话。

    缓存命中也得走这一步：缓存存的是答案，不存这句话，两边的口径要一致。
    """
    if ans.topic.kind == "param" and not ans.branches:
        names = "、".join(ans.topic.vars or [])
        ans.note = f"解里带着参数 {names}，但不需要分情形讨论"


def _lin_show(expr, var) -> str:
    """把 `5y - 10` 写成学生习惯的 `5y = 10`。

    `5y - 10 = 0` 也对，但没人这么写；教材上一条横线移到右边，
    写成等式才看得出"下一步该除几"。
    """
    try:
        poly = Poly(sympy.expand(expr), var)
        coef = poly.coeff_monomial(var)
        const = poly.coeff_monomial(1)
        return f"{to_latex(sympy.expand(coef * var))} = {to_latex(sympy.expand(-const))}"
    except Exception:  # noqa: BLE001
        return f"{to_latex(sympy.expand(expr))} = 0"


def _system_steps(pairs, syms) -> list[dict]:
    """二元一次方程组的解法，写成初中生看得懂的几步。

    为什么不用 CAS 原生步骤：那是**内部记账**，不是教学过程。
    - 它给两条 "方程"，学生看不出这两条是题目给的两个条件；
    - "解得" 把中间的消元过程整个吃掉了——那恰恰是这道题唯一要学的东西；
    - "代回验证 0 = 0" 是残差，只有写程序的人认得，学生看到只会困惑。

    两条路线，按学生自己的选择顺序排：
      1. 有系数是 ±1 的项 → **代入法**（学生最先想到的路）；
      2. 某一项系数正好相同或相反 → **加减消元**（说清哪一项抵消了）。
    挑不出干净路线的（要两边乘系数那种），返回空，宁可退回通用步骤也不硬编。
    中间每一步都是真算出来的，不是套话。
    """
    if len(pairs) != 2 or len(syms) != 2:
        return []
    x, y = syms
    co = []
    for lhs, rhs in pairs:
        try:
            p = Poly(sympy.expand(lhs - rhs), x, y)
        except Exception:  # noqa: BLE001
            return []
        co.append((p.coeff_monomial(x), p.coeff_monomial(y), -p.coeff_monomial(1)))
    (a1, b1, c1), (a2, b2, c2) = co
    if 0 in (a1, b1, a2, b2):
        return []                      # 缺项时"相加消元"的说法会不准确

    cond = {
        "title": "题目给的两个条件（①②）",
        "latex": rf"{to_latex(pairs[0][0])} = {to_latex(pairs[0][1])} \quad ①"
                 rf"\qquad {to_latex(pairs[1][0])} = {to_latex(pairs[1][1])} \quad ②",
    }

    # 能直接相消时（某项系数正好相同或相反），加减比代入更省事，先走那条路。
    # 顺序反了会自相矛盾：讲评说"相加最快"，步骤却绕去代入。
    can_cancel = any(
        k1 == -k2 or (k1 == k2 and k1 != 0) for k1, k2 in ((b1, b2), (a1, a2))
    )

    # ---- 路线 1：代入法（有 ±1 系数）
    for i in (() if can_cancel else (0, 1)):
        a, b, c = co[i]
        j = 1 - i
        aj, bj, cj = co[j]
        for which in (0, 1):
            k, m = (a, b) if which == 0 else (b, a)
            if abs(k) != 1:
                continue
            var, other = (x, y) if which == 0 else (y, x)
            expr_var = sympy.expand((c - m * other) / k)      # var = expr_var
            e_j = aj * x + bj * y - cj
            res = sympy.expand(e_j.subs(var, expr_var))        # 只剩 other
            try:
                sols = sympy.solve(res, other)
            except Exception:  # noqa: BLE001
                continue
            if len(sols) != 1:
                continue
            val_other = sympy.nsimplify(sols[0])
            val_var = sympy.simplify(expr_var.subs(other, val_other))
            xv, yv = (val_var, val_other) if which == 0 else (val_other, val_var)
            num = "①" if i == 0 else "②"
            num_j = "②" if i == 0 else "①"
            return [
                cond,
                {
                    "title": f"由 {num}式把 {to_latex(var)} 单独放到一边",
                    "latex": f"{to_latex(var)} = {to_latex(expr_var)}",
                },
                {
                    "title": f"把 {to_latex(var)} = {to_latex(expr_var)} 代进 {num_j}式，"
                             f"消掉 {to_latex(var)}",
                    "latex": rf"{_lin_show(res, other)} \quad\Rightarrow\quad "
                             f"{to_latex(other)} = {to_latex(val_other)}",
                },
                {
                    "title": f"把 {to_latex(other)} = {to_latex(val_other)} 代回，求 {to_latex(var)}",
                    "latex": f"{to_latex(var)} = {to_latex(val_var)}",
                },
                {
                    "title": f"检查：把 {to_latex(x)} = {to_latex(xv)}、{to_latex(y)} = {to_latex(yv)} 分别代回 ① 和 ②",
                    "latex": r"\text{两个方程左右两边都相等，答案成立}",
                },
            ]

    # ---- 路线 2：加减消元（某一项系数相同或相反）
    pick = None
    for var, k1, k2 in ((y, b1, b2), (x, a1, a2)):
        if k1 == -k2:
            pick = (var, "+", "相加")
            break
        if k1 == k2:
            pick = (var, "-", "相减")
            break
    if pick is None:
        # 两项系数都不相同：两边各乘一个数，把系数凑成一样，再加减。
        # 这是教材里的标准做法，只是比前两条路线多一步，不该因此退回机器步骤。
        best = None
        for var, k1, k2 in ((y, b1, b2), (x, a1, a2)):
            L = sympy.ilcm(abs(k1), abs(k2))
            m1, m2 = L // abs(k1), L // abs(k2)
            if best is None or m1 + m2 < best[0]:
                best = (m1 + m2, var, k1, k2, m1, m2)
        _, var, k1, k2, m1, m2 = best
        e1 = sympy.expand(m1 * (a1 * x + b1 * y - c1))
        e2 = sympy.expand(m2 * (a2 * x + b2 * y - c2))
        res = sympy.expand(e1 - e2) if k1 * k2 > 0 else sympy.expand(e1 + e2)
        alone = x if var is y else y
        try:
            sols = sympy.solve(res, alone)
        except Exception:  # noqa: BLE001
            return []
        if len(sols) != 1:
            return []
        val = sympy.nsimplify(sols[0])
        other = y if alone is x else x
        oa, ob = (a1, b1) if alone is x else (b1, a1)
        other_val = sympy.simplify((c1 - oa * val) / ob)
        xv, yv = (val, other_val) if alone is x else (other_val, val)
        word = "相减" if k1 * k2 > 0 else "相加"
        sign = "-" if k1 * k2 > 0 else "+"
        return [
            cond,
            {
                "title": f"两个方程里 {to_latex(var)} 的系数不一样，先各乘一个数凑成一样",
                "latex": rf"①式 \times {m1}：{to_latex(sympy.expand(m1 * (a1 * x + b1 * y)))}"
                         rf" = {to_latex(m1 * c1)} \qquad "
                         rf"②式 \times {m2}：{to_latex(sympy.expand(m2 * (a2 * x + b2 * y)))}"
                         rf" = {to_latex(m2 * c2)}",
            },
            {
                "title": f"①式×{m1} {word} ②式×{m2}：{to_latex(var)} 这一项正好抵消",
                "latex": rf"{_lin_show(res, alone)} \quad\Rightarrow\quad "
                         rf"{to_latex(alone)} = {to_latex(val)}",
            },
            {
                "title": f"把 {to_latex(alone)} = {to_latex(val)} 代回 ①式，求 {to_latex(other)}",
                "latex": rf"{to_latex(sympy.expand(oa * val + ob * other))} = {to_latex(c1)}"
                         rf"\quad\Rightarrow\quad {to_latex(other)} = {to_latex(other_val)}",
            },
            {
                "title": f"检查：把 {to_latex(x)} = {to_latex(xv)}、{to_latex(y)} = {to_latex(yv)}"
                         f" 分别代回①②两式",
                "latex": r"\text{两个方程左右两边都相等，答案成立}",
            },
        ]
    gone, sign, word = pick
    if sign == "+":
        ea, eb, ec = a1 + a2, b1 + b2, c1 + c2
    else:
        ea, eb, ec = a1 - a2, b1 - b2, c1 - c2
    alone = x if gone is y else y
    ka, kb = (ea, eb) if alone is x else (eb, ea)
    if ka == 0 or ec % ka != 0:
        return []
    val = ec / ka
    other = y if alone is x else x
    oa, ob = (a1, b1) if alone is x else (b1, a1)
    other_val = (c1 - oa * val) / ob
    if other_val.q != 1:
        return []
    xv, yv = (val, other_val) if alone is x else (other_val, val)

    mid = sympy.expand(ka * alone - ec)
    return [
        cond,
        {
            "title": f"①式 {word} ②式：{to_latex(gone)} 这一项正好抵消",
            "latex": rf"{_lin_show(mid, alone)} \quad\Rightarrow\quad "
                     f"{to_latex(alone)} = {to_latex(val)}",
        },
        {
            "title": f"把 {to_latex(alone)} = {to_latex(val)} 代回 ①式，求 {to_latex(other)}",
            "latex": rf"{to_latex(sympy.expand(oa * val + ob * other))} = {to_latex(c1)}"
                     rf"\quad\Rightarrow\quad {to_latex(other)} = {to_latex(other_val)}",
        },
        {
            "title": f"检查：把 {to_latex(x)} = {to_latex(xv)}、{to_latex(y)} = {to_latex(yv)} 分别代回 ① 和 ②",
            "latex": r"\text{两个方程左右两边都相等，答案成立}",
        },
    ]



def _checks(pe: ParsedExpr, sr: SolveResult) -> list[str]:
    """逐根代回原式，把"我验过"变成看得见的一行字。"""
    out: list[str] = []
    if pe.kind != "equation" or not pe.ok or sr.kind == "param":
        return out          # 含参的分支答案已经由 discuss 取实际参数值验过
    lhs, rhs = pe.obj[0]
    syms = sorted((lhs - rhs).free_symbols, key=str)
    if not syms:
        return out
    # 主元优先取 x/y/z：x²-2ax+a²=0 里的 a 是参数，拿 a 代回验的是另一个字母。
    core = [s for s in syms if str(s) in _UNKNOWN_NAMES]
    x = core[0] if core else syms[0]
    for s in sr.solutions:
        sol = s.split("=", 1)[-1].strip()
        try:
            val = parse(sol)
            sub = sympy.sympify(val.sympy_str) if val.ok else None
            if sub is None:
                continue
            left = sympy.simplify(lhs.subs(x, sub))
            right = sympy.simplify(rhs.subs(x, sub))
            if sympy.simplify(left - right) == 0:
                out.append(
                    rf"把 {s} 代回原方程：${to_latex(left)} = {to_latex(right)}$，两边相等 ✓"
                )
        except Exception:  # noqa: BLE001
            continue
    return out


def answer(
    text: str,
    *,
    source: str = "text",
    allow_setup: bool = True,
    explain: bool = True,
) -> Answer:
    """题目 → 答案。这是"搜答案"那条路的全部逻辑。"""
    t0 = time.time()
    ans = Answer(text=text.strip(), source=source)
    eq = extract(text, source=source)

    if not eq.latex and allow_setup and looks_like_word_problem(text):
        eq = setup_equation(text)
        ans.provided = False
    ans.latex = eq.latex
    ans.topic = eq.topic
    ans.note = eq.note
    ans.provided = eq.provided

    if not eq.latex:
        ans.error = eq.topic.note or "没能从题面里读出一个方程"
        ans.timings["total"] = time.time() - t0
        return ans

    pe = parse(eq.latex)
    if not pe.ok:
        ans.error = f"方程转不成语法树：{pe.error}"
        ans.timings["total"] = time.time() - t0
        return ans

    if eq.topic.coverage == "out":
        # 出范围就到此为止：能算也不给。给了就会有人拿复数根对初中的答案，
        # 而且"算得出"会被误读成"这个题型我支持"。
        ans.error = eq.topic.note or OUT["not-equation"]
        ans.timings["total"] = time.time() - t0
        return ans

    # 缓存：键用方程的标准形（`x^2-5x+6=0` 和 `x^{2} - 5x + 6 = 0` 是同一道题），
    # 不用题面文字——同一道题被抄两遍措辞不同，也该命中。讲评是否生成也进键：
    # 没有讲评的那一份不能顶替有讲评的。
    want_explain = bool(explain and config.explain_enabled())
    ckey = (CACHE_VERSION, pe.sympy_str or eq.latex,
            "explain" if want_explain else "plain")
    hit = _ANSWER_CACHE.get(*ckey)
    if hit:
        _from_cache(ans, hit)
        _fix_param_note(ans)
        ans.cached = True
        ans.timings["cache"] = 0.0
        ans.timings["total"] = time.time() - t0
        return ans

    t1 = time.time()
    sr = solve_parsed(pe)
    ans.timings["solve"] = time.time() - t1
    ans.solutions = list(sr.solutions)
    ans.steps = [{"title": s.title, "latex": s.latex} for s in sr.steps]
    # 方程组换成学生版的步骤：CAS 的"方程/解得/0=0"是内部记账，不是教学步骤。
    if eq.topic.kind == "system" and sr.ok and pe.kind == "system":
        syms = sorted(
            set().union(*(set((lhs - rhs).free_symbols) for lhs, rhs in pe.obj)),
            key=lambda s: s.name,
        )
        human = _system_steps(list(pe.obj), syms)
        if human:
            ans.steps = human
    ans.dropped = list(sr.dropped)
    ans.checks = _checks(pe, sr)
    # 解集为空是结论，不是失败：1/(x-2)=0 是真的无解，不是"我没算出来"。
    # 只有 CAS 自己报了错，才算失败。
    empty_set = (not sr.ok and not sr.error and sr.solutions == ["无解"])
    ans.ok = bool(sr.ok) or empty_set
    ans.error = sr.error
    if sr.kind == "param":
        # 含参的答案是一张分情形的表。压成一行 "分类讨论：①…②…" 读不动，
        # 界面上必须一支一行——漏掉的那一支就在这种地方被漏看。
        ans.branches = _branches(sr.solutions)
        ans.answer_latex = ""
    elif sr.ok:
        ans.answer_latex = solve_summary(sr).replace("解：", "")
    elif empty_set:
        ans.answer_latex = r"\text{无解（解集为空）}"

    if ans.topic.kind == "param" and not ans.branches:
        # 解里带参数但不用分情形（x²-2mx+m²-1=0 直接就是 x=m±1）。
        # 这时候还挂着"按参数分情形讨论"就是空话，得改口。
        _fix_param_note(ans)

    if want_explain and sr.ok:
        try:
            from .explain import explain as explain_steps

            ex = explain_steps(pe, sr)
            ans.explain = ex.text if ex.ok else ""
        except Exception:  # noqa: BLE001
            pass

    ans.timings["total"] = time.time() - t0
    if ans.ok:
        _ANSWER_CACHE.put(
            {
                "ok": ans.ok,
                "solutions": ans.solutions,
                "steps": ans.steps,
                "checks": ans.checks,
                "dropped": ans.dropped,
                "branches": ans.branches,
                "answer_latex": ans.answer_latex,
                "explain": ans.explain,
            },
            *ckey,
        )
    return ans


def answer_image(
    image: str | Path,
    *,
    allow_setup: bool = True,
    explain: bool = True,
) -> tuple[Answer, dict]:
    """题目照片 → 答案。识别走已有链路，抽方程和求解走本模块。"""
    from .pipeline import run as pipeline_run

    t0 = time.time()
    r = pipeline_run(image=str(image), k=2, save=False)
    text = (r.verification.chosen or "").strip()
    if not text:
        text = max((c.latex or "" for c in r.recognition), key=len, default="")
    ans = answer(text, source="image", allow_setup=allow_setup, explain=explain)
    ans.timings["perceive"] = round(time.time() - t0 - ans.timings.get("total", 0.0), 2)
    meta = {
        "verdict": r.verification.verdict,
        "confidence": round(r.verification.confidence, 2),
        "sources": sorted({c.source for c in r.recognition if c.latex}),
        "read": text,
        "route": r.route,
    }
    return ans, meta
