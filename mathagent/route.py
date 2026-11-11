"""判断一张图（或一段识别结果）该走哪条链路。

公式链路只认算式。几何题是一整页「题干 + 图形」，误投到公式页签时，
原来的做法是硬解析——把整段中文塞进表达式，最后报一句「等号数量异常：2 个」。
用户看到的是「系统坏了」，实际上是「投错了门」，这两件事必须分开说。

判据只有一条硬要求：**必须是成句的中文**（否则纯算式里的 ∠ 就会被误伤，
`\\angle A + \\angle B = 180^\\circ` 是个正儿八经的算式，不是几何题）。
在成句的前提下再看有没有几何特征词。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 一个字一个分。权重按「这句话是不是几何题独有」定：
# 「求证」「如图」几乎不出现在别处，给 3；「中点」「相似」别的题也可能有，给 1。
HINTS: list[tuple[str, float]] = [
    ("求证", 3), ("如图", 3), ("四边形", 3), ("三角形", 3), ("相交于", 3),
    ("∥", 3), ("⊥", 3), ("\\parallel", 3), ("\\perp", 3),
    ("证明", 2), ("正方形", 2), ("矩形", 2), ("菱形", 2), ("梯形", 2),
    ("平行四边形", 2), ("圆心", 2), ("直径", 2), ("切线", 2), ("交点", 2),
    ("角平分线", 2), ("延长线", 2), ("全等", 2), ("圆周角", 2), ("△", 2),
    ("\\triangle", 2), ("\\angle", 1),
    ("相交", 1), ("中点", 1), ("中线", 1), ("半径", 1), ("垂线", 1),
    ("相似", 1), ("∠", 1),
]

# 成句的中文才算「题干」。一个算式里顶多出现 `\text{cm}`、`\text{度}` 这种单位。
PROSE_CJK = 6

# 命中项要显示给人看，别把 LaTeX 命令名直接甩到界面上
SHOW = {"\\parallel": "∥", "\\perp": "⊥", "\\triangle": "△", "\\angle": "∠"}


@dataclass
class Route:
    """走哪条链路，以及为什么。"""

    kind: str = "formula"          # formula / geometry / prose
    score: float = 0.0
    hints: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def is_geometry(self) -> bool:
        return self.kind == "geometry"


def cjk_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))


def plain(latex: str) -> str:
    """把识别出的 LaTeX 还原成人能读的几行字。

    整页题被读成 `\\text{...}\\\\\\text{...}` 这种串，照原样贴到界面上，
    用户只会看到一堆反斜杠，反而更怀疑「它到底读到了什么」。
    """
    s = latex or ""
    # `\text{...}` 会套娃：题干整句里面还有 `\text{ 与 }`。正则不能跨层匹配，
    # 所以一层一层剥，剥到没变化为止。
    for _ in range(6):
        outer = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
        if outer == s:
            break
        s = outer
    for a, b in (
        ("\\parallel", "∥"), ("\\perp", "⊥"), ("\\angle", "∠"),
        ("\\triangle", "△"), ("\\circ", "°"), ("\\\\", "\n"),
    ):
        s = s.replace(a, b)
    s = re.sub(r"\\(?:left|right|quad|qquad|,|;|:|!)", " ", s)
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n", s).strip()


def detect(text: str) -> Route:
    """看这段识别结果像什么。

    两段式：先看是不是成句中文，再看有没有几何特征。
    只判「是不是算式」而不判「是什么题」的话，界面照样说不出该去哪儿。
    """
    text = text or ""
    n = cjk_count(text)
    if n < PROSE_CJK:
        return Route(kind="formula")

    hits: list[str] = []
    score = 0.0
    for word, w in HINTS:
        if word in text:
            hits.append(SHOW.get(word, word))
            score += w

    if score >= 3:
        shown = "」「".join(hits[:5])
        return Route(
            kind="geometry",
            score=score,
            hints=hits,
            note=(
                f"这是一道几何题，不是算式：正文有 {n} 个汉字，"
                f"出现了「{shown}」等题干特征。"
            ),
        )

    return Route(
        kind="prose",
        score=score,
        hints=hits,
        note=(
            f"这是一段题目文字（{n} 个汉字），不是算式。"
            "公式链路只处理已经列好的式子，读不懂应用题的叙述。"
        ),
    )
