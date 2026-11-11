"""把题干里的文字关系编译成几何结构。

为什么必须有这一步：读图模型只会把「图上画着什么」写进 points / segments /
marks——看得见的线、看得见的直角符号。可几何题的条件有一半在文字里，
而且恰恰是不画符号的那一半最要命：

    四边形 ABCD 为正方形   → 四直角 + 四边等
    DE ∥ AC                → 一条平行
    AE = AC                → 一条等长

这三句在图上都没有任何标记，模型就把它们抄进 given 全文了事。少了它们，
求解器看到的只是一个「四条边都有直角的四边形」——那是矩形，不是正方形；
再往下 CE 就不再是一个数，而是随自由度乱跑的一段区间。实测这题的目标值
落在 1.379 ~ 19.95 之间，界面只能报「条件不足」。

本模块只做「文字 → 结构」的翻译，不做推理：正方形就是四直角加四边等，
不替它推「对角线互相垂直平分」——那些留给求解器从基本约束里自然长出来。
这样每一条约束都能对回题面上的一句话，用户核对时看得见出处。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .schema import Mark, Problem


def _name(a: str, b: str) -> str:
    """线段名规范写法：AC 和 CA 是同一条。"""
    return "".join(sorted((a, b)))


def _clean(s: str) -> str:
    return (s or "").replace(" ", "").replace("　", "").replace("\u3000", "")


@dataclass
class Derived:
    """从文字里补出来的结构。"""

    marks: list[Mark] = field(default_factory=list)
    lines: list[set[str]] = field(default_factory=list)  # 共线组
    # 「X 在 AB 上」里的三元组 (X, A, B)：X 必须落在 A、B **之间**。
    # 只约束三点共线是不够的——「AE 与 CD 相交于 F」里的 F 若跑到延长线上，
    # 共线照样成立，可那时画出来的图已经不是题里那个图了（这题的 E 就有
    # 两个解，一个 F 在线段内、一个在线段外，CE 分别是 0.732 和 2.732）。
    between: list[tuple[str, str, str]] = field(default_factory=list)
    segments: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# 图形声明的拆解表：名称 → (需要四直角, 需要四边等, 需要两组对边平行, 几个顶点)
_QUADS = {
    "正方形": (True, True, False),
    "矩形": (True, False, False),
    "长方形": (True, False, False),
    "菱形": (False, True, False),
    "平行四边形": (False, False, True),
}


def _find_quad(texts: list[str]) -> list[tuple[str, list[str]]]:
    """找出被声明成特殊四边形的四个点，返回 [(图形名, [四个顶点])]。"""
    out: list[tuple[str, list[str]]] = []
    for s in texts:
        t = _clean(s)
        for word, _ in _QUADS.items():
            # 「四边形ABCD为正方形」「ABCD是正方形」「正方形ABCD中」
            for pat in (
                rf"([A-Z]{{4}})(?:为|是)?{word}",
                rf"{word}\s*([A-Z]{{4}})",
            ):
                for m in re.finditer(pat, t):
                    quad = m.group(1)
                    if len(set(quad)) == 4 and (word, list(quad)) not in out:
                        out.append((word, list(quad)))
    return out


def _find_triangles(texts: list[str]) -> list[tuple[str, list[str]]]:
    """正三角形 / 等边三角形 / 等腰三角形。"""
    out: list[tuple[str, list[str]]] = []
    for s in texts:
        t = _clean(s)
        for word, kind in (("正三角形", "eq"), ("等边三角形", "eq"), ("等腰三角形", "iso")):
            for pat in (rf"([A-Z]{{3}})(?:为|是)?{word}", rf"{word}\s*([A-Z]{{3}})"):
                for m in re.finditer(pat, t):
                    tri = m.group(1)
                    if len(set(tri)) == 3 and (kind, list(tri)) not in out:
                        out.append((kind, list(tri)))
    return out


def _quad_marks(quad: list[str], right: bool, equal: bool, parallel: bool) -> list[Mark]:
    """把一个四边形声明拆成「直角 + 等长 + 平行」三类标记。"""
    a, b, c, d = quad
    sides = [(a, b), (b, c), (c, d), (d, a)]
    out: list[Mark] = []

    if right:
        # 每个顶点的两条邻边
        for i in range(4):
            at = quad[i]
            prev = quad[(i - 1) % 4]
            nxt = quad[(i + 1) % 4]
            out.append(Mark("right_angle", at, [_name(at, prev), _name(at, nxt)]))

    if equal:
        # 四边等 → 三条「与第一条相等」的等式就够，别写满六条
        first = _name(*sides[0])
        for x, y in sides[1:]:
            out.append(Mark("equal_length", "", [first, _name(x, y)]))

    if parallel:
        out.append(Mark("parallel", "", [_name(a, b), _name(c, d)]))
        out.append(Mark("parallel", "", [_name(b, c), _name(d, a)]))

    return out


def _find_parallel(texts: list[str]) -> list[tuple[str, str]]:
    """DE∥AC / DE//AC / DE平行于AC。"""
    out: list[tuple[str, str]] = []
    for s in texts:
        t = _clean(s)
        for m in re.finditer(r"([A-Z]{2})\s*(?:∥|//|平行于|平行)\s*([A-Z]{2})", t):
            p = (m.group(1), m.group(2))
            if p[0] != p[1] and p not in out:
                out.append(p)
    return out


def _find_perp(texts: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for s in texts:
        t = _clean(s)
        for m in re.finditer(r"([A-Z]{2})\s*(?:⊥|垂直于|垂直)\s*([A-Z]{2})", t):
            p = (m.group(1), m.group(2))
            if p[0] != p[1] and p not in out:
                out.append(p)
    return out


def _find_equal_segments(texts: list[str]) -> list[tuple[str, str]]:
    """AE=AC / AE＝AC。右边必须是字母，否则是长度（AB=6），那是另一条路。"""
    out: list[tuple[str, str]] = []
    for s in texts:
        t = _clean(s)
        for m in re.finditer(r"([A-Z]{2})\s*[=＝]\s*([A-Z]{2})", t):
            p = (m.group(1), m.group(2))
            if p[0] != p[1] and p not in out and (p[1], p[0]) not in out:
                out.append(p)
    return out


def _find_midpoints(texts: list[str]) -> list[tuple[str, str]]:
    """D是AB的中点 / D为AB中点 / AB的中点是D。"""
    out: list[tuple[str, str]] = []
    for s in texts:
        t = _clean(s)
        for m in re.finditer(r"([A-Z])\s*(?:是|为)\s*([A-Z])([A-Z])\s*的?中点", t):
            out.append((m.group(2) + m.group(3), m.group(1)))
        for m in re.finditer(r"([A-Z])([A-Z])\s*的?中点\s*(?:是|为)\s*([A-Z])", t):
            out.append((m.group(1) + m.group(2), m.group(3)))
    return out


def _find_intersections(texts: list[str]) -> list[set[str]]:
    """「AE 与 CD 相交于 F」→ {A,E,F}、{C,D,F} 两组共线。

    只从 problem_text / given 里找，不碰 relations：那是模型自己组织的
    描述性文字，实测会把「AE 与 CD 相交于 F」记成「F 是 AC 与 DE 的交点」
    ——四个字母里错了两个，还照样说得通。拿它当约束比没有约束更糟。
    """
    out: list[set[str]] = []
    for s in texts:
        t = _clean(s)
        for pat in (
            r"([A-Z]{2})(?:与|和|、)([A-Z]{2})(?:相交|交)于([A-Z])",
            r"([A-Z]{2})交([A-Z]{2})于([A-Z])",
        ):
            for m in re.finditer(pat, t):
                a, b, x = m.group(1), m.group(2), m.group(3)
                if len({a[0], a[1], x}) == 3:
                    out.append({a[0], a[1], x})
                if len({b[0], b[1], x}) == 3:
                    out.append({b[0], b[1], x})
    return out


def _find_point_on_line(texts: list[str]) -> list[tuple[str, str, str]]:
    """「F 在 AE 上」→ (F, A, E)。

    这条只从 relations 里读也有意义：它是读图提示词明确要求的写法，
    措辞固定、歧义小，比「F 是 XX 与 YY 的交点」可靠得多——后者实测
    会把 AE 与 CD 写成 AC 与 DE，四个字母错两个还照样通顺。
    """
    out: list[tuple[str, str, str]] = []
    for s in texts:
        t = _clean(s)
        # 「F 在 AE 上，也在 CD 上」：后半句不再重复那个字母，得沿用上一个点，
        # 否则只读到 AE 那一条，F 还差一维约束，CE 就又算不出确定值。
        cur = ""
        for m in re.finditer(r"(?:([A-Z])|也)?在([A-Z]{2})上", t):
            if m.group(1):
                cur = m.group(1)
            if not cur:
                continue
            seg = m.group(2)
            if cur in seg:
                continue
            item = (cur, seg[0], seg[1])
            if item not in out:
                out.append(item)
    return out


def on_segment_lines(p: Problem, tol: float = 1.05) -> list[set[str]]:
    """从「图上画了哪些线 + 各点大致位置」推出谁落在哪条边上。

    模型几乎不会写「F 在 AE 上」——在它看来那是显而易见的。可求解器不觉得：
    F 少了「在 AE 上」「在 CD 上」这两条，CE 就不是一个数而是一族数，
    整道题直接退化成「条件不足」。

    判据是图上真的连了线：F 同时连向 A 和 E，且 AE 本身也是一条线段，
    三条边长又满足 |FA| + |FE| ≈ |AE|，那 F 就在 AE 上。位置只用来判
    比例，不用来定坐标，所以目测值够用。

    容差定在 1.05 是权衡出来的：三角形重心到两个顶点的距离和约为边长的
    1.155 倍、正方形中心约 1.414 倍，本该被挡在门外；但 1.10 放得太宽，
    这题就把「F 落在 AC 上」误判进来了（AF+FC 是 1.059 倍 AC），于是
    F 被钉在一条错的对角线上，算出来 CE 和 CF 差 60%，反过来诬告条件矛盾。
    误判比漏判贵得多：漏判只是「条件不足」，误判会给一个理直气壮的错答案。
    """
    if not p.pos or len(p.pos) < 3:
        return []
    segs = {tuple(sorted(s)) for s in p.segments}
    out: list[set[str]] = []
    seen: set[frozenset] = set()

    for a, b in segs:
        if a not in p.pos or b not in p.pos:
            continue
        L = math.dist(p.pos[a], p.pos[b])
        if L < 1e-6:
            continue
        for x in p.points:
            if x in (a, b) or x not in p.pos:
                continue
            # 必须两边都真的连了线，否则只是「离得近」，不是「在边上」
            if tuple(sorted((a, x))) not in segs or tuple(sorted((b, x))) not in segs:
                continue
            d = math.dist(p.pos[a], p.pos[x]) + math.dist(p.pos[b], p.pos[x])
            if d / L > tol:
                continue
            key = frozenset({a, b, x})
            if key in seen:
                continue
            seen.add(key)
            out.append({a, b, x})
    return out


def derive(p: Problem, skip_equal: set[frozenset] | None = None) -> Derived:
    """把一道题的题干读成结构补充。

    skip_equal 是「不要当成条件」的等长关系——证明题里待证的 CE=CF 经常
    被模型抄进 given，若照着建成约束，就成了拿结论证明结论。
    """
    out = Derived()
    pts = set(p.points)
    skip_equal = skip_equal or set()

    # 文字优先级：题干原文 > 已知条件 > 模型的关系描述
    primary = [p.problem_text] + list(p.given)
    all_texts = primary + list(p.relations)

    seen = {
        (m.type, m.at, frozenset(_name(x[0], x[1]) if len(x) == 2 else x for x in m.of))
        for m in p.marks
    }

    def add_mark(m: Mark) -> None:
        # 去重键必须先把线段名归一：AC 和 CA、AD 和 DA 是同一条边，
        # 按原字符串比会把「正方形给的 A 处直角」和「图上读到的 A 处直角」
        # 当成两件事，条件是重复的，步骤里也会出现两遍。
        of = [_name(x[0], x[1]) if len(x) == 2 and x.isalpha() else x for x in m.of]
        key = (m.type, m.at, frozenset(of))
        if key in seen:
            return
        seen.add(key)
        out.marks.append(Mark(m.type, m.at, of))


    # --- 1) 特殊图形 ---
    for word, quad in _find_quad(primary):
        if not set(quad) <= pts:
            continue
        right, equal, parallel = _QUADS[word]
        for m in _quad_marks(quad, right, equal, parallel):
            add_mark(m)
        bits = []
        if right:
            bits.append("四个直角")
        if equal:
            bits.append("四条边相等")
        if parallel:
            bits.append("两组对边平行")
        out.notes.append(f"题干称 {'、'.join(quad)} 构成{word}，已按{' + '.join(bits)}建模")

    for kind, tri in _find_triangles(primary):
        if not set(tri) <= pts:
            continue
        a, b, c = tri
        pairs = [(_name(a, b), _name(b, c)), (_name(b, c), _name(c, a))]
        if kind == "eq":
            pairs.append((_name(c, a), _name(a, b)))
        for x, y in pairs:
            add_mark(Mark("equal_length", "", [x, y]))
        out.notes.append(
            f"题干称 {''.join(tri)} 是{'正三角形' if kind == 'eq' else '等腰三角形'}"
            f"，已按{'三边相等' if kind == 'eq' else '两边相等'}建模"
        )

    # --- 2) 平行 ---
    for a, b in _find_parallel(primary):
        if not (set(a) <= pts and set(b) <= pts):
            continue
        add_mark(Mark("parallel", "", [a, b]))
        out.notes.append(f"题干给 {a} ∥ {b}")

    # --- 3) 垂直 ---
    for a, b in _find_perp(primary):
        common = set(a) & set(b)
        if len(common) != 1:
            continue
        at = common.pop()
        if not (set(a) <= pts and set(b) <= pts):
            continue
        add_mark(Mark("right_angle", at, [a, b]))
        out.notes.append(f"题干给 {a} ⊥ {b}")

    # --- 4) 等长（跳过待证结论）---
    for a, b in _find_equal_segments(all_texts):
        if not (set(a) <= pts and set(b) <= pts):
            continue
        if frozenset((a, b)) in skip_equal:
            continue
        add_mark(Mark("equal_length", "", [a, b]))

    # --- 5) 中点 ---
    for seg, at in _find_midpoints(primary):
        if not set(seg) <= pts or at not in pts:
            continue
        add_mark(Mark("midpoint", at, [seg]))
        # 中点必然落在那条线段上。少了这条共线，推理层就认不出以它为顶点的
        # 三角形：三线合一那题里 D 是 BC 中点，没有 {B,C,D} 共线，
        # △ABD 根本凑不出来，SSS 也就无从谈起（实测被逼到去"连辅助线"）。
        g = {seg[0], seg[1], at}
        if g not in out.lines:
            out.lines.append(g)
        if (at, seg[0], seg[1]) not in out.between:
            out.between.append((at, seg[0], seg[1]))

    # --- 6) 交点 → 共线 ---
    for g in _find_intersections(primary):
        g2 = {x for x in g if x in pts}
        if len(g2) >= 3 and g2 not in out.lines:
            out.lines.append(g2)
            out.notes.append(f"题干给 {'、'.join(sorted(g2))} 三点共线")

    # --- 6b) 「F 在 AE 上」——提示词要求模型这么写，措辞固定，可信度高 ---
    for x, a, b in _find_point_on_line(all_texts):
        if not {x, a, b} <= pts:
            continue
        g2 = {a, b, x}
        if g2 not in out.lines:
            out.lines.append(g2)
            out.notes.append(f"读图时记下 {x} 在 {_name(a, b)} 上")
        if (x, a, b) not in out.between:
            out.between.append((x, a, b))

    # --- 7) 图上的「点在边上」 ---
    segs = {tuple(sorted(s)) for s in p.segments}
    for g in on_segment_lines(p):
        if g in out.lines:
            continue
        out.lines.append(g)
        xs = sorted(g)
        out.notes.append(f"图上 {xs[0]}–{xs[1]} 是一段直线，{xs[2]} 落在其上（由图推出）")
        for x in g:
            for y in g:
                if x < y and tuple(sorted((x, y))) not in segs:
                    out.segments.append((x, y))
                    segs.add(tuple(sorted((x, y))))

    return out
