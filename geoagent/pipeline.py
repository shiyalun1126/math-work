"""几何链路编排：读图 → 交叉校验 → 求解 → 重绘。

和 mathagent.pipeline 的关系：两条独立链路，不共中间表示。
公式是「转写 + 符号求解」，几何是「结构化 + 坐标求解」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import crosscheck, explain, render, solve
from .crosscheck import Check
from .explain import GeoExplain
from .read import readers
from .schema import Problem
from .solve import GeoSolution


@dataclass
class GeoResult:
    problem: Problem | None = None
    others: list[Problem] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    solution: GeoSolution = field(default_factory=GeoSolution)
    verdict: str = "fail"       # ok / review / fail
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
    error: str = ""
    explain: GeoExplain = field(default_factory=GeoExplain)

    def to_dict(self) -> dict:
        return {
            "problem": self.problem.to_dict() if self.problem else None,
            "others": [o.to_dict() for o in self.others],
            "checks": [{"level": c.level, "title": c.title, "detail": c.detail} for c in self.checks],
            "solution": self.solution.to_dict(),
            "explain": self.explain.to_dict(),
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "notes": self.notes,
            "error": self.error,
        }


def judge(checks: list[Check], sol: GeoSolution, n_sources: int) -> tuple[str, float, list[str]]:
    """定档位和置信度。规则全部写死在这里，不藏在别处。"""
    notes: list[str] = []
    conflicts = [c for c in checks if c.level == "conflict"]
    warns = [c for c in checks if c.level == "warn"]

    conf = 0.45
    if sol.ok:
        conf += 0.20
        if sol.unique:
            conf += 0.10
    if not conflicts:
        conf += 0.10
    if not warns:
        conf += 0.05
    if n_sources >= 2:
        conf += 0.10
        notes.append("两路独立来源都读到了图")
    else:
        notes.append("只有一路在读图，没有旁证")

    verdict = "ok"
    if conflicts:
        verdict = "review"
        notes.append(f"有 {len(conflicts)} 处互相矛盾的说法，需要你确认")
    if sol.ok and not sol.unique:
        verdict = "review"
    if not sol.ok:
        verdict = "review" if sol.residual < 1e-4 else "fail"
        if sol.error:
            notes.append(sol.error)

    # 单源封顶，和公式链路保持同一条规矩：没有交叉验证就不给高置信
    if n_sources < 2:
        conf = min(conf, 0.75)
    return verdict, min(conf, 0.95), notes


# 立体几何的关键词。这一整类题不在本链路的射程内：读图会把正方体读成
# 一堆平面上的点，坐标求解再把它们按 2D 约束凑起来，凑出来的"图形"和第
# 一问问的"线面角"没有半点关系。与其交一坨算得很像那么回事的错答案，
# 不如直说做不了——用户拿着错答案去写作业才是真事故。
SOLID_HINTS = (
    "正方体", "长方体", "棱柱", "棱锥", "棱台", "圆柱", "圆锥",
    "三视图", "异面直线", "二面角", "母线", "线面角", "面面角",
)


def solid_hints(p: Problem) -> str:
    t = "".join([p.problem_text or "", *p.given, p.goal or ""])
    return "、".join(w for w in SOLID_HINTS if w in t)


def resolve(problem: Problem, others: list[Problem] | None = None) -> GeoResult:
    """只做校验 + 求解，不重新识别。界面里用户改完结构后走这条路。"""
    others = others or []
    hits = solid_hints(problem)
    if hits:
        res = GeoResult(problem=problem, others=others, verdict="fail")
        res.error = (
            f"这是一道立体几何题（题干里有「{hits}」）。当前链路只处理平面几何："
            "读图与坐标求解都按二维建模，硬算只会给出错的答案。"
        )
        res.solution.error = res.error   # 界面靠这一条把「算不出来」显示出来
        return res

    # 先按题干补全结构：「点 C 在圆上」这类条件只写在文字里，不补的话
    # 下面判断切线时根本找不到切点，标记也就修不了。
    # 证明题的结论（CE=CF）不能当条件用，补全时就要排掉，否则就是循环论证。
    claim = solve.parse_claim(problem)
    skip = {frozenset({claim.a, claim.b})} if claim and claim.kind == "len_equal" else set()
    problem, _ = solve.enrich_from_text(problem, skip_equal=skip)
    res = GeoResult(problem=problem, others=others)
    # 再修标记：切点处的直角常被读成两条弦的夹角，重绘时就会标错角。
    # 修完再校验，否则校验看到的是错的、画图用的是对的，两边不一致。
    repaired = solve.repair_marks(problem)
    res.checks = crosscheck.run(problem, others)
    res.solution = solve.solve(problem)
    # 解题过程放在求解之后生成：它只负责把已经解出来的东西讲清楚，
    # 不参与求解，也就不会把讲解里的口误变成答案。
    res.explain = explain.write(res.problem, res.solution)
    n_src = 1 + len(others)
    res.verdict, res.confidence, notes = judge(res.checks, res.solution, n_src)
    res.notes = repaired + notes
    return res


def run(image: str | Path) -> GeoResult:
    """完整链路：识别 → 校验 → 求解。"""
    rs = readers()
    if not rs:
        return GeoResult(error="没有可用的读图通道，请先在 .env 里配置密钥")

    probs: list[Problem] = []
    notes: list[str] = []
    errs: list[str] = []
    for r in rs:
        p, note = r.read(image)
        if p is None:
            errs.append(f"{r.name}：{note}")
            continue
        if note:
            notes.append(note)
        probs.append(p)

    if not probs:
        return GeoResult(error="所有读图通道都失败了。" + "；".join(errs), notes=notes)

    res = resolve(probs[0], probs[1:])
    res.notes = notes + res.notes
    if errs:
        res.notes.append("部分通道失败：" + "；".join(errs))
    return res


def figure_coords(res: GeoResult) -> tuple[dict | None, str]:
    """重绘用哪套坐标。返回 (坐标, 依据)。

    这里的选择标准不是「哪套更准」，而是「哪套更像用户手里那张图」——
    重绘的唯一用途就是让人一眼核对我读到的点线对不对。

      exact  ：图形被条件唯一确定，用解出的真实坐标，比例是准的
      pos    ：条件不足或有自由度，用模型读到的各点位置，形状未必严格，
               但它就是「系统理解的样子」，对照原图最直观
      approx ：连位置都没读到，只能用解出的坐标示意
    """
    s = res.solution
    p = res.problem
    if p is None:
        return None, "none"

    pos = {k: v for k, v in (p.pos or {}).items() if k in set(p.points)}
    if s.ok and s.unique and s.coords:
        return s.coords, "exact"
    if pos and set(pos) >= set(p.points):
        # pos 是图像坐标（y 向下），渲染那边按数学坐标处理，先翻过来
        return {k: (v[0], -v[1]) for k, v in pos.items()}, "pos"
    if s.coords:
        return s.coords, "approx"
    return None, "none"


def figure_of(res: GeoResult, size: int = 520, original: str | None = None):
    """给界面用的重绘图；原图给了就并排。"""
    if res.problem is None:
        return None
    coords, _ = figure_coords(res)
    return render.render(
        res.problem, coords, size=size, goal=goal_of(res), value=res.solution.value
    )


def goal_of(res: GeoResult) -> tuple | None:
    """重绘时要在图上标出来的目标量。解不出来就不标。"""
    s = res.solution
    # 结论不成立时不标：图上写着「求证 CE = CD」而下面写着「不成立」，
    # 自相矛盾，反倒像是系统在自说自话。
    if s.goal_kind == "prove" and s.claim_edges and s.ok:
        return ("compare", s.claim_name, s.claim_edges)
    if not s.ok or not s.goal_kind:
        return None
    return (s.goal_kind, s.goal_name, s.goal_arms)



def summary_md(res: GeoResult) -> str:
    """一句话结论 + 校验清单。"""
    if res.problem is None:
        return f"**失败**：{res.error}"
    tag = {"ok": "可以算", "review": "需要你核对", "fail": "算不出来"}[res.verdict]
    lines = [f"**{tag}**（置信度 {res.confidence:.0%}）"]
    if res.solution.ok:
        lines.append(f"\n### {res.solution.answer}\n")
    elif res.solution.error:
        lines.append(f"\n> {res.solution.error}\n")
    lines.append("**校验**")
    lines.append(crosscheck.summary_md(res.checks))
    return "\n".join(lines)
