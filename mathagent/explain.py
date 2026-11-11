"""讲解层：大模型只负责把已验算的步骤讲成人话，不参与任何计算。

为什么要有这一层：SymPy 给出的步骤是准确的，但读起来像机器笔记
（"移项整理：x^2-5x+6 = 0"）。加一段思路说明能显著提升可读性。

为什么它必须受约束：LLM 讲着讲着就会自己算一步，然后算错。
所以这里做两件事——
  1. 提示词禁止输出公式和结果，只允许描述方法；
  2. 输出后回扫一遍，任何等式都要么能在已验证的步骤里找到出处，
     要么两边确实恒等，否则整段讲解丢弃。
"讲解被丢弃"是可以显示给用户的状态，不是异常。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests

from . import config
from .latex import normalize, parse
from .types import ParsedExpr, SolveResult


@dataclass
class ExplainResult:
    ok: bool = False
    text: str = ""
    model: str = ""
    elapsed: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "text": self.text,
            "model": self.model,
            "elapsed": round(self.elapsed, 2),
            "note": self.note,
        }


SYSTEM = (
    "你是一位中学数学老师，正在给学生讲评。"
    "你只能说思路和方法，不能做任何计算，也不能给出新的公式或答案——"
    "公式和答案由解题器提供，你重复或改写它们就算出错。"
    "**不要引用题目里的任何数字或系数**，需要指代时用"
    "「一次项系数」「常数项」这样的说法代替：说「和为一次项系数的相反数」是安全的，"
    "说「和为 5」一旦符号记反就是错的，而这种口误没人会替你检查。"
    "要求：2 到 4 句中文，口语化，说清这道题在考什么、关键一步为什么这么走、哪里最容易错。"
    "不要使用 LaTeX，不要写美元符号，不要分点，不要复述题目。"
)

FEWSHOT = (
    "示例输出：这道题的关键是把含未知数的项都移到一边，先化成标准形式再判断。"
    "注意常数项移过去要变号，这一步最容易漏掉负号。"
)


def _known(segment: str, pool: list[str]) -> bool:
    s = re.sub(r"\s+", "", normalize(segment))
    return any(s and s in re.sub(r"\s+", "", normalize(p)) for p in pool)


def _balanced(segment: str) -> bool | None:
    """两边都是常数且相等 → True；都是常数但不等 → False；含未知数 → None。"""
    pe = parse(segment)
    if not pe.ok or pe.kind != "equation":
        return None
    try:
        import sympy

        lhs, rhs = pe.obj[0]
        d = sympy.simplify(lhs - rhs)
        if d.free_symbols:
            return None
        return bool(d == 0)
    except Exception:  # noqa: BLE001
        return None


def _refs(pe: ParsedExpr, sr: SolveResult) -> list:
    """把题目原式和各步骤化成 f(x)=0 形式，用于判断"同一个方程的不同写法"。"""
    import sympy

    out = []
    try:
        obj = getattr(pe, "obj", None)
        if obj and pe.kind == "equation":
            lhs, rhs = obj[0]
            out.append(sympy.expand(lhs - rhs))
    except Exception:  # noqa: BLE001
        pass
    for st in sr.steps:
        if getattr(st, "lhs", None) is not None and getattr(st, "rhs", None) is not None:
            try:
                out.append(sympy.expand(st.lhs - st.rhs))
            except Exception:  # noqa: BLE001
                continue
    return out


def _same_equation(segment: str, refs: list) -> bool:
    """这句话里的等式是否只是原式/某一步的等价改写。

    只做"差一个非零常数倍"这一条判断，因为讲评里最常见的合法说法
    就是因式分解后重写方程，比如把 x^2-5x+6=0 说成 (x-2)(x-3)=0。
    这一条不成立就一律丢弃——宁可少讲，也不能放一句算错的过去。
    """
    pe = parse(segment)
    if not pe.ok or pe.kind != "equation":
        return False
    try:
        import sympy

        a = sympy.expand(pe.obj[0][0] - pe.obj[0][1])
    except Exception:  # noqa: BLE001
        return False
    for b in refs:
        try:
            if a == b or a == -b:
                return True
            if a == 0 or b == 0:
                continue
            ratio = sympy.simplify(a / b)
            if not ratio.free_symbols and ratio != 0:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


_MATH_RUN = re.compile(r"[0-9A-Za-z()\[\]{}+\-*/^=.',\s]{3,}")


def _math_part(seg: str) -> str:
    """从"把它拆成 (x-2)(x-3)=0 就好看了"里抠出 `(x-2)(x-3)=0`。

    正则整句捕获会把中文叙述一起带进来，直接解析必然失败，
    于是合法讲解被当成错误丢弃——这是误伤，比放过更糟。
    """
    runs = [m.group(0).strip() for m in _MATH_RUN.finditer(seg)]
    cands = [r for r in runs if "=" in r]
    if not cands:
        return ""
    return max(cands, key=len)


def check(text: str, pool: list[str], refs: list | None = None) -> tuple[bool, str]:
    """回扫讲解文本。宁可丢弃，也不放一条自己算过的讲解过去。"""

    t = (text or "").strip()
    if not t:
        return False, "讲解为空"
    if "$" in t or "\\" in t:
        return False, "讲解里出现了 LaTeX，已丢弃（公式必须由解题器给）"
    if len(t) > 400:
        return False, "讲解过长，已丢弃"

    for m in re.finditer(r"[^。；，、\s][^。；，、]*=[^。；，、]*", t):
        seg = m.group(0).strip()
        math = _math_part(seg) or seg
        if _known(math, pool) or _known(seg, pool):
            continue
        if _balanced(math) is True:
            continue
        if refs and _same_equation(math, refs):
            continue
        return False, f"讲解里出现了无法验证的等式「{seg}」，已丢弃"
    return True, ""


def explain(pe: ParsedExpr, sr: SolveResult) -> ExplainResult:
    if not config.has_llm():
        return ExplainResult(note="未配置讲解模型，跳过")

    pool = [pe.latex, pe.sympy_str] + [s.latex for s in sr.steps] + sr.solutions
    titles = "、".join(s.title for s in sr.steps) or "（无）"
    user = (
        f"题型：{sr.kind or pe.kind}\n"
        f"学生写的式子（LaTeX，仅供你理解，不要复述）：{pe.latex}\n"
        f"解题器已经算出的步骤标题：{titles}\n"
        f"解题器已经给出的答案：{'；'.join(sr.solutions) or '（无）'}\n"
        f"被剔除的步骤：{'；'.join(sr.dropped) or '（无）'}\n\n"
        f"{FEWSHOT}\n现在请写这段讲评。"
    )

    t0 = time.time()
    try:
        r = requests.post(
            f"{config.llm_base().rstrip('/')}/chat/completions",
            json={
                "model": config.llm_model(),
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
                "max_tokens": 260,
            },
            headers={"Authorization": f"Bearer {config.llm_key()}"},
            timeout=40,
        )
    except requests.RequestException as exc:
        return ExplainResult(note=f"网络不通：{exc}", model=config.llm_model())

    dt = time.time() - t0
    if r.status_code != 200:
        return ExplainResult(
            note=f"HTTP {r.status_code}：{r.text[:120]}", model=config.llm_model(), elapsed=dt
        )
    try:
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    except Exception as exc:  # noqa: BLE001
        return ExplainResult(note=f"返回结构异常：{exc}", model=config.llm_model(), elapsed=dt)

    ok, why = check(text, pool, _refs(pe, sr))
    return ExplainResult(
        ok=ok,
        text=text.strip() if ok else "",
        model=config.llm_model(),
        elapsed=dt,
        note="" if ok else why,
    )
