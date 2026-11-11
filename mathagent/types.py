"""模块间传递的数据结构。所有接口只走这里定义的对象。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .explain import ExplainResult


@dataclass
class Candidate:
    """一个识别来源给出的一条候选。"""

    latex: str
    source: str  # simpletex / qwen3-vl / mock
    conf: float | None = None
    elapsed: float = 0.0
    note: str = ""
    # 主模型限流、临时换到备用模型读出来的候选。备用模型多为免费的小模型，
    # 对着中文说明行会硬凑出一条公式（实测 W04 把"两边同时乘2得"读成
    # \frac{AB}{BC}=\frac{AC}{BD}）。这种候选可以参与投票，但平票时不该
    # 压过主模型——否则一条幻觉就把整行判成"待核对"。
    weak: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedExpr:
    """LaTeX 解析结果。ok=False 时 error 说明原因，不抛异常。"""

    latex: str
    normalized: str
    kind: str  # equation / system / expression / unknown
    sympy_str: str = ""
    ok: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Cluster:
    """语义等价的一组候选。"""

    latex: str
    members: list[Candidate] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def sources(self) -> list[str]:
        return sorted({m.source for m in self.members})


@dataclass
class VerificationResult:
    """校验层结论。verdict 只有三档，交互层据此决定是否追问。"""

    verdict: str  # ok / review / failed
    chosen: str = ""
    confidence: float = 0.0
    clusters: list[Cluster] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    rejected: list[Candidate] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    independent: bool = False  # 是否拿到了两个及以上独立来源

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "chosen": self.chosen,
            "confidence": round(self.confidence, 3),
            "independent": self.independent,
            "sources": self.sources,
            "reasons": self.reasons,
            "clusters": [
                {"latex": c.latex, "size": c.size, "sources": c.sources}
                for c in self.clusters
            ],
            "rejected": [c.to_dict() for c in self.rejected],
        }


@dataclass
class Step:
    """一条解题步骤。

    check 决定事后怎么验：
      identity —— 声明是恒等式，验 simplify(lhs-rhs)==0
      equiv    —— 声明与上一步同解，验解集相同（移项、通分这类变形）
      none     —— 定义式或结论，不做验算
    """

    title: str
    latex: str
    lhs: Any = None
    rhs: Any = None
    check: str = "none"


@dataclass
class SolveResult:
    ok: bool = False
    kind: str = ""
    problem_latex: str = ""
    solutions: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "kind": self.kind,
            "problem_latex": self.problem_latex,
            "solutions": self.solutions,
            "dropped": self.dropped,
            "error": self.error,
            "steps": [
                {"title": s.title, "latex": s.latex} for s in self.steps
            ],
        }


@dataclass
class WorkStep:
    """学生过程里的一行，以及对它的判定。

    kind 说明这一行是什么：
      problem   —— 题面
      equation  —— 一个等式（推导）
      assign    —— 形如 x=3 的赋值（答案）
      expr      —— 没有等号的式子
      prose     —— 纯文字说明（"两边同乘2"），不参与判定
      unparsed  —— 读不出算式

    verdict 说明这一行站不站得住：
      ok / partial（漏解）/ wrong / unknown / note（说明行）
    """

    idx: int
    raw: str
    kind: str
    role: str = ""
    latex: str = ""
    verdict: str = "unknown"
    relation: str = ""
    reason: str = ""
    counter: str = ""                      # 反例：一个满足上一步却不满足这一步的取值
    missing: list[str] = field(default_factory=list)
    declared: str = ""                     # 学生在文字里声明的操作
    row: list[str] = field(default_factory=list)  # 界面上"判定"列要显示的字

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkCheck:
    """一次批改的结论。grade 是总判定，steps 是逐行证据。"""

    problem: str = ""
    problem_ok: bool = False
    grade: str = "unknown"   # pass / wrong / partial / fake / gap / unknown
    headline: str = ""
    steps: list[WorkStep] = field(default_factory=list)
    first_error: int = -1
    expected: list[str] = field(default_factory=list)
    got: list[str] = field(default_factory=list)
    error_type: str = ""
    advice: str = ""
    notes: list[str] = field(default_factory=list)
    problem_image: str = ""
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def n_ok(self) -> int:
        return sum(1 for s in self.steps if s.verdict == "ok")

    @property
    def n_checked(self) -> int:
        return sum(1 for s in self.steps if s.verdict != "note" and s.kind != "prose")

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "problem_ok": self.problem_ok,
            "grade": self.grade,
            "headline": self.headline,
            "first_error": self.first_error,
            "expected": self.expected,
            "got": self.got,
            "error_type": self.error_type,
            "advice": self.advice,
            "notes": self.notes,
            "problem_image": self.problem_image,
            "n_ok": self.n_ok,
            "n_checked": self.n_checked,
            "steps": [s.to_dict() for s in self.steps],
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
        }


@dataclass
class PipelineResult:
    recognition: list[Candidate]
    verification: VerificationResult
    parsed: ParsedExpr
    solved: SolveResult
    timings: dict[str, float] = field(default_factory=dict)
    explain: "ExplainResult | None" = None
    image_note: str = ""
    # 该走哪条链路："" 正常（公式），"geometry" / "prose" 说明投错了入口
    route: str = ""
    route_note: str = ""
    route_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recognition": [c.to_dict() for c in self.recognition],
            "verification": self.verification.to_dict(),
            "parsed": self.parsed.to_dict(),
            "solved": self.solved.to_dict(),
            "explain": self.explain.to_dict() if self.explain else None,
            "image_note": self.image_note,
            "route": self.route,
            "route_note": self.route_note,
            "route_hints": self.route_hints,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
        }
