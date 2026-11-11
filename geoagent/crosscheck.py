"""图文交叉校验。

几何题的优势在于「同一件事往往说两遍」：题干写「∠C = 90°」，图上还画个
直角小方块；题干写「D 是 AB 中点」，图上还画等长标记。两处对不上，
就说明至少有一处读错了——这是公式题没有的纠错信号。

三条校验：
  1. 单份结果内部一致（线段端点、标记位置有没有落在点集里）
  2. 题干文字 ↔ 图上标记（互相印证 / 图形补充 / 存疑）
  3. 两路模型之间（分歧才需要人看）

判定刻意保守：宁可报「存疑」也不轻易报「冲突」。误报冲突会让人不再相信提示，
那这套机制就白做了。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import Problem

# 级别 → (符号, 排序权重)
LEVELS = {
    "conflict": ("冲突", 0),
    "warn": ("存疑", 1),
    "info": ("补充", 2),
    "ok": ("一致", 3),
}


@dataclass
class Check:
    level: str
    title: str
    detail: str = ""

    @property
    def mark(self) -> str:
        return LEVELS.get(self.level, ("?", 9))[0]


@dataclass
class Facts:
    """题干文字里能被机械核对的事实。"""

    right_angles: list[str]
    midpoints: list[tuple[str, str, str]]  # (中点, 端点1, 端点2)
    diameters: list[tuple[str, str]]
    tangents: list[str]
    lengths: list[tuple[str, float]]


def text_facts(given: list[str]) -> Facts:
    ra: list[str] = []
    mp: list[tuple[str, str, str]] = []
    dia: list[tuple[str, str]] = []
    tan: list[str] = []
    ln: list[tuple[str, float]] = []

    for s in given:
        t = s.replace(" ", "").replace("　", "")
        # 直角：∠C=90° / ∠ACB=90° / ∠C为直角
        for m in re.finditer(r"∠([A-Z]{1,3})\s*[=＝]\s*90", t):
            ra.append(m.group(1))
        for m in re.finditer(r"∠([A-Z]{1,3})\s*(?:是|为)\s*直角", t):
            ra.append(m.group(1))
        # 中点：D是AB的中点 / D为AB中点 / AB的中点为D
        m = re.search(r"([A-Z])\s*(?:是|为)\s*([A-Z])([A-Z])\s*的?中点", t)
        if m:
            mp.append((m.group(1), m.group(2), m.group(3)))
        m = re.search(r"([A-Z])([A-Z])\s*的?中点\s*(?:是|为)\s*([A-Z])", t)
        if m:
            mp.append((m.group(3), m.group(1), m.group(2)))
        # 直径：AB是⊙O的直径 / AB为直径
        if "直径" in t:
            m = re.search(r"([A-Z])([A-Z])(?:是|为)", t)
            if m:
                dia.append((m.group(1), m.group(2)))
        # 切线
        if "切线" in t:
            tan.append(s)
        # 定长：AC=6 / AC=6.5
        for m in re.finditer(r"([A-Z])([A-Z])\s*[=＝]\s*(\d+(?:\.\d+)?)", t):
            ln.append((m.group(1) + m.group(2), float(m.group(3))))
    return Facts(ra, mp, dia, tan, ln)


def figure_midpoints(p: Problem) -> list[str]:
    """图上被标为中点的点：显式 midpoint 标记，或等长标记标出的共享端点。"""
    out: set[str] = set()
    for m in p.marks:
        if m.type == "midpoint" and m.at:
            out.add(m.at)
        elif m.type == "equal_length" and len(m.of) == 2:
            a, b = m.of
            # AD 与 DB 等长，且首尾相接于 D ⇒ D 是 AB 中点
            if len(a) == 2 and len(b) == 2 and a[1] == b[0] and a[0] != b[1]:
                out.add(a[1])
    return sorted(out)


def _has_segment(p: Problem, name: str) -> bool:
    pair = tuple(sorted(name))
    return any(tuple(sorted(s)) == pair for s in p.segments)


def internal_checks(p: Problem) -> list[Check]:
    """单份结果的内部一致性：线段端点、标记位置是否落在点集内。"""
    out: list[Check] = []
    pts = set(p.points)
    if not pts:
        return [Check("warn", "没有读到任何点", "图里可能没有几何图形，或者模型什么都没认出来")]

    bad_segs = [s for s in p.segments if not set(s) <= pts]
    if bad_segs:
        out.append(
            Check(
                "warn",
                f"{len(bad_segs)} 条线段的端点不在点集里",
                "、".join("".join(s) for s in bad_segs) + f"；已读到的点：{'、'.join(p.points)}",
            )
        )

    bad_marks = [m for m in p.marks if m.at and m.at not in pts]
    if bad_marks:
        out.append(
            Check(
                "warn",
                f"{len(bad_marks)} 个标记的位置不在点集里",
                "、".join(f"{m.type}@{m.at}" for m in bad_marks),
            )
        )

    for c in p.circles:
        if c.center and c.center not in pts:
            out.append(Check("warn", f"圆心 {c.center} 不在点集里", "圆周或圆心可能有一处漏读"))

    if not out:
        out.append(
            Check("ok", "结构自洽", f"{len(p.points)} 个点、{len(p.segments)} 条线段都对得上")
        )
    return out


def text_vs_figure(p: Problem) -> list[Check]:
    """题干文字与图上标记的互相印证。"""
    tf = text_facts(p.given)
    out: list[Check] = []

    fig_ra = sorted({m.at for m in p.marks if m.type == "right_angle" and m.at})

    # --- 直角 ---
    if tf.right_angles and fig_ra:
        txt = {a[0] if len(a) > 1 else a for a in tf.right_angles}
        both = txt & set(fig_ra)
        if both:
            out.append(
                Check("ok", f"∠{sorted(both)[0]} = 90°",
                      "题干这么写，图上也有直角标记，两处印证")
            )
        else:
            out.append(
                Check("conflict", "直角的说法对不上",
                      f"题干说 ∠{tf.right_angles[0]}=90°，图上的直角却标在 "
                      f"{'、'.join(fig_ra)}——必有一处读错")
            )
    elif tf.right_angles and not fig_ra:
        out.append(
            Check("warn", "图上没找到直角标记",
                  f"题干称 ∠{tf.right_angles[0]}=90°，但图上没有直角符号。"
                  "可能本来就没画，也可能漏读了——按题干走，但结果要打个折")
        )
    elif fig_ra and not tf.right_angles:
        out.append(
            Check("info", f"{'、'.join(fig_ra)} 处有直角标记",
                  "题干没直接给出这个直角，它是图上补充的条件，往往是解题的关键一步")
        )

    # --- 中点 ---
    fig_mp = figure_midpoints(p)
    if tf.midpoints and fig_mp:
        txt = {m[0] for m in tf.midpoints}
        both = txt & set(fig_mp)
        if both:
            x = sorted(both)[0]
            seg = next(("".join(sorted((m[1], m[2]))) for m in tf.midpoints if m[0] == x), "")
            out.append(
                Check("ok", f"{x} 是 {seg} 的中点" if seg else f"{x} 是中点",
                      "题干这么写，图上也有等长标记，两处印证")
            )
        else:
            out.append(
                Check("conflict", "中点的说法对不上",
                      f"题干说 {sorted(txt)[0]} 是中点，图上标的却是 "
                      f"{'、'.join(fig_mp)}——必有一处读错")
            )
    elif tf.midpoints and not fig_mp:
        out.append(
            Check("warn", "图上没找到等长标记",
                  f"题干称 {tf.midpoints[0][0]} 是中点，但图上没画等长符号。"
                  "按题干走，但要注意这可能只是题干信息，图中并无对应")
        )
    elif fig_mp and not tf.midpoints:
        out.append(
            Check("info", f"{'、'.join(fig_mp)} 处是等长的中点",
                  "图上画了等长标记，题干没提——这是图给出的额外条件")
        )

    # --- 圆心与直径 ---
    if tf.diameters:
        a, b = tf.diameters[0]
        if p.circles:
            ring = set(p.circles[0].through or []) | set(p.points)
            miss = [x for x in (a, b) if x not in ring]
            if miss:
                out.append(
                    Check("warn", f"直径端点 {'、'.join(miss)} 没在圆上",
                          f"题干说 {a}{b} 是直径，但图上圆没有经过这两个点")
                )
        elif p.circles == []:
            out.append(Check("warn", "题干提到直径，但没读到圆",
                             "圆可能被漏读了，或者这句话被读错"))

    # --- 切线 → 图上应有垂直 ---
    if tf.tangents and fig_ra:
        out.append(
            Check("ok", "切线与直角标记对得上",
                  "切线垂直于过切点的半径，图上的直角标记正是这个关系")
        )
    elif tf.tangents and not fig_ra:
        out.append(
            Check("info", "题干提到切线，但图上直角没标出来",
                  "切线蕴含「半径 ⊥ 切线」，这个垂直关系可以补进条件里用")
        )

    # --- 定长 ---
    for name, val in tf.lengths:
        if not _has_segment(p, name):
            out.append(
                Check("warn", f"{name} 在图中没画出来",
                      f"题干给了 {name} = {val}，但线段列表里没有 {name}——可能有线漏读了")
            )

    return out


def cross_channel(primary: Problem, other: Problem) -> list[Check]:
    """两路模型之间的差异。

    只把「一路有、另一路没有」记成补充/存疑，不当成冲突：
    漏读远多于错读，把漏读当冲突会天天误报。真正危险的只有
    「同一处给出互相排斥的说法」，比如直角标在了不同的顶点。
    """
    if primary.source == other.source:
        return []
    out: list[Check] = []

    a_pts, b_pts = set(primary.points), set(other.points)
    if a_pts - b_pts:
        out.append(Check("info", "第二路少读了点", f"{'、'.join(sorted(a_pts - b_pts))}"))
    if b_pts - a_pts:
        out.append(Check("warn", "第二路读出了额外的点",
                         f"{'、'.join(sorted(b_pts - a_pts))}——可能是真的漏读，也可能是幻觉"))

    a_ra = {m.at for m in primary.marks if m.type == "right_angle"}
    b_ra = {m.at for m in other.marks if m.type == "right_angle"}
    if a_ra and b_ra and a_ra != b_ra:
        out.append(
            Check("conflict", "两路对直角位置的判断不一致",
                  f"{primary.source} 认为在 {'、'.join(sorted(a_ra))}，"
                  f"{other.source} 认为在 {'、'.join(sorted(b_ra))}")
        )

    a_mp = set(figure_midpoints(primary))
    b_mp = set(figure_midpoints(other))
    if a_mp and b_mp and a_mp != b_mp:
        out.append(
            Check("conflict", "两路对中点的判断不一致",
                  f"{primary.source}: {'、'.join(sorted(a_mp))} / "
                  f"{other.source}: {'、'.join(sorted(b_mp))}")
        )

    a_len = dict(text_facts(primary.given).lengths)
    b_len = dict(text_facts(other.given).lengths)
    for k in set(a_len) & set(b_len):
        if abs(a_len[k] - b_len[k]) > 1e-6:
            out.append(
                Check("conflict", f"{k} 的长度两路读得不一样",
                      f"{primary.source} 读 {a_len[k]:g}，{other.source} 读 {b_len[k]:g}")
            )

    if not out:
        out.append(Check("ok", "两路读图一致", "点集、直角、中点、定长都对得上"))
    return out


def run(primary: Problem, others: list[Problem] | None = None) -> list[Check]:
    """全部校验，按严重程度排序。"""
    checks: list[Check] = []
    checks += internal_checks(primary)
    checks += text_vs_figure(primary)
    for o in others or []:
        checks += cross_channel(primary, o)
    checks.sort(key=lambda c: LEVELS.get(c.level, (None, 9))[1])
    return checks


def summary_md(checks: list[Check]) -> str:
    """给界面用的一小段 markdown。"""
    if not checks:
        return "（没有校验项）"
    lines = []
    for c in checks:
        lines.append(f"- **{c.mark}**｜{c.title}" + (f"　—　{c.detail}" if c.detail else ""))
    return "\n".join(lines)
