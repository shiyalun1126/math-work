"""几何题的数据结构。

为什么不复用 LaTeX：几何题的中间态不是一串符号，而是一张关系网。
硬塞进 LaTeX（模型真的会这么做——把整段中文题干都塞进去）以后，
下游的符号解析器只能报错，什么也做不了。

这里定的结构刻意窄：只收「能从图上直接看出来的东西」——有哪些点、
哪些线段、哪些标记。凡是需要推理才能得到的（比如「所以这是等腰」），
一律不进结构，留给求解器去推。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# 标记类型：只认这几种。模型爱自创（"angle_bisector"、"perp"），
# 一律映射进来，映射不上就丢弃并记录，免得下游拿着不认识的值空转。
MARK_TYPES = {
    "right_angle": "right_angle",
    "perp": "right_angle",
    "perpendicular": "right_angle",
    "直角": "right_angle",
    "equal_length": "equal_length",
    "equal": "equal_length",
    "等长": "equal_length",
    "parallel": "parallel",
    "平行": "parallel",
    "midpoint": "midpoint",
    "中点": "midpoint",
}


@dataclass
class Mark:
    """图上画出来的标记。这是几何题最值钱的信息——它常常比题面更直接。"""

    type: str
    at: str = ""            # 标记所在点，如直角顶点 C、中点 D
    of: list[str] = field(default_factory=list)  # 涉及的线段名，如 ["AC", "BC"]

    def to_dict(self) -> dict:
        return {"type": self.type, "at": self.at, "of": list(self.of)}

    @property
    def key(self) -> tuple:
        return (self.type, self.at, tuple(sorted(self.of)))


@dataclass
class Circle:
    center: str = ""
    through: list[str] = field(default_factory=list)
    radius: float | None = None

    def to_dict(self) -> dict:
        return {"center": self.center, "through": list(self.through), "radius": self.radius}


@dataclass
class Problem:
    """一道几何题的读图结果。"""

    problem_text: str = ""
    given: list[str] = field(default_factory=list)
    goal: str = ""
    points: list[str] = field(default_factory=list)
    segments: list[tuple[str, str]] = field(default_factory=list)
    circles: list[Circle] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    # 各点在图形区域内的近似相对位置，0–100，左上角为原点、y 向下。
    # 精度无所谓，只用来给数值求解一个「和原图朝向一致」的初值：
    # 没有它，解出来的图和原图可能左右镜像，对照着看会以为认错了。
    pos: dict[str, tuple[float, float]] = field(default_factory=dict)
    source: str = ""      # 哪个通道读出来的
    raw: str = ""         # 原始返回，出问题好回溯

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        return {
            "problem_text": self.problem_text,
            "given": list(self.given),
            "goal": self.goal,
            "points": list(self.points),
            "segments": [list(s) for s in self.segments],
            "circles": [c.to_dict() for c in self.circles],
            "marks": [m.to_dict() for m in self.marks],
            "relations": list(self.relations),
            "pos": {k: [round(v[0], 1), round(v[1], 1)] for k, v in self.pos.items()},
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict, source: str = "", raw: str = "") -> "Problem":
        """宽松解析：模型给的 JSON 只有一半字段符合预期是常态。"""
        pts = [str(p).strip() for p in (d.get("points") or []) if str(p).strip()]

        segs: list[tuple[str, str]] = []
        for s in d.get("segments") or []:
            pair = _as_pair(s)
            if pair:
                segs.append(pair)

        circles = []
        for c in d.get("circles") or []:
            if not isinstance(c, dict):
                continue
            circles.append(
                Circle(
                    center=str(c.get("center") or "").strip(),
                    through=[str(x).strip() for x in (c.get("through") or [])],
                    radius=_as_float(c.get("radius")),
                )
            )

        marks = []
        for m in d.get("marks") or []:
            if not isinstance(m, dict):
                continue
            t = MARK_TYPES.get(str(m.get("type") or "").strip().lower())
            if not t:
                continue
            at = str(m.get("at") or "").strip()
            marks.append(Mark(type=t, at=at, of=_norm_of(m.get("of"), at)))

        return cls(
            problem_text=str(d.get("problem_text") or ""),
            given=[str(g) for g in (d.get("given") or []) if str(g).strip()],
            goal=str(d.get("goal") or ""),
            points=pts,
            segments=segs,
            circles=circles,
            marks=marks,
            relations=[str(r) for r in (d.get("relations") or []) if str(r).strip()],
            pos=_as_pos(d.get("pos")),
            source=source,
            raw=raw,
        )

    # ---- 派生 ----
    def normative(self) -> "Problem":
        """整理成规范形式：点排序、线段去重且端点有序、标记去重。

        两路模型给出的点序、线段方向常常不同（AB / BA，先 C 后 A），
        不规范化就没法判断两路是不是真的读到同一个东西。
        """
        pts = sorted(set(self.points))
        segs = sorted({tuple(sorted(s)) for s in self.segments if len(s) == 2})
        seen = set()
        marks = []
        for m in self.marks:
            k = m.key
            if k in seen:
                continue
            seen.add(k)
            marks.append(Mark(type=m.type, at=m.at, of=sorted(m.of)))
        marks.sort(key=lambda m: (m.type, m.at, m.of))
        return Problem(
            problem_text=self.problem_text.strip(),
            given=[g.strip() for g in self.given],
            goal=self.goal.strip(),
            points=pts,
            segments=list(segs),
            circles=self.circles,
            marks=marks,
            relations=self.relations,
            pos=dict(self.pos),
            source=self.source,
            raw=self.raw,
        )

    def summary(self) -> str:
        return (
            f"{len(self.points)} 点 / {len(self.segments)} 段 / "
            f"{len(self.marks)} 标记 / {len(self.given)} 已知"
        )


def _norm_of(raw: Any, at: str) -> list[str]:
    """把标记涉及的东西统一成「线段名」列表。

    两个模型写法完全不同：一个给 ["AC", "BC"]，另一个给 ["A", "B"]。
    不统一，跨源比对时同一件事会被判成两件事。
    """
    items = raw if isinstance(raw, list) else ([raw] if raw else [])
    singles: list[str] = []
    names: list[str] = []
    for x in items:
        pair = _as_pair(x)
        if pair:
            names.append("".join(pair))
            continue
        s = str(x).strip()
        if not s:
            continue
        if len(s) == 1 and s.isalpha():
            singles.append(s)
        else:
            names.append(s)
    # ["A","B"] 这种是「两端点」，拼成一条线段名
    for i in range(0, len(singles) - 1, 2):
        names.append(singles[i] + singles[i + 1])
    return names


def _as_pair(x: Any) -> tuple[str, str] | None:
    """把 ["A","B"] / "AB" 统一成 (A, B)。"""
    if isinstance(x, (list, tuple)) and len(x) == 2:
        a, b = str(x[0]).strip(), str(x[1]).strip()
        return (a, b) if a and b else None
    s = str(x).strip()
    if len(s) == 2 and s.isalpha():
        return (s[0], s[1])
    return None


def _as_pos(raw: Any) -> dict[str, tuple[float, float]]:
    """解析点的近似位置。

    模型给的形态很不统一：{"A":[10,60]}、{"A":"10,60"}、
    [{"name":"A","pos":[10,60]}] 都见过。统一成 0–100 的相对坐标，
    超范围的就夹住——它只当初始朝向用，离谱的值也不该把求解带跑。
    """
    out: dict[str, tuple[float, float]] = {}

    def put(name: Any, val: Any) -> None:
        n = str(name).strip()
        if len(n) != 1 or not n.isalpha():
            return
        x = y = None
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            x, y = _as_float(val[0]), _as_float(val[1])
        elif isinstance(val, str):
            parts = [t for t in re.split(r"[,\s]+", val.strip()) if t]
            if len(parts) >= 2:
                x, y = _as_float(parts[0]), _as_float(parts[1])
        elif isinstance(val, dict):
            x, y = _as_float(val.get("x")), _as_float(val.get("y"))
        if x is None or y is None:
            return
        out[n] = (max(0.0, min(100.0, x)), max(0.0, min(100.0, y)))

    if isinstance(raw, dict):
        for k, v in raw.items():
            put(k, v)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                put(item.get("name") or item.get("point"), item.get("pos") or item)
    return out


def _as_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
