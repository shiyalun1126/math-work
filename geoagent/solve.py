"""几何求解：把条件翻译成坐标约束，数值解出图形，再读答案。

为什么用坐标法而不是定理搜索：定理搜索要建题库和推理规则，规模无上限；
坐标法只要能把条件写成等式，绝大多数初中题都能算，而且算完能代回检验。

关键设计是「多初值一致性」：同一道题从不同起点出发各解一次，如果目标值
都收敛到同一个数，说明图形被条件唯一确定；如果收敛到不同的数，就说明
条件不足（比如只给了一个长度却要问另一个长度）——这时如实说「定不下来」，
而不是随便交一个解出去。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .schema import Circle, Problem
from . import relations


@dataclass
class GeoSolution:
    ok: bool = False
    answer: str = ""
    value: float | None = None
    unit: str = ""
    steps: list[str] = field(default_factory=list)
    residual: float = float("nan")
    unique: bool = True
    error: str = ""
    coords: dict[str, tuple[float, float]] = field(default_factory=dict)
    n_unknown: int = 0
    n_equations: int = 0
    # 目标量是什么、用哪两条边量的——重绘时要把这个角画出来，
    # 否则用户看不出来系统到底在算哪个角（g2 就因此被当成「标错了角」）。
    goal_kind: str = ""
    goal_name: str = ""
    goal_arms: tuple[str, str] | None = None
    # 证明题：结论是哪两条边/哪两个角，重绘时要把这两条边一起标出来
    claim_edges: tuple[str, str] | None = None
    claim_angles: tuple[str, str] | None = None
    claim_name: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "answer": self.answer,
            "value": self.value,
            "unit": self.unit,
            "steps": self.steps,
            "residual": None if math.isnan(self.residual) else self.residual,
            "unique": self.unique,
            "error": self.error,
            "coords": {k: list(v) for k, v in self.coords.items()},
            "goal": {
                "kind": self.goal_kind,
                "name": self.goal_name,
                "arms": list(self.goal_arms) if self.goal_arms else [],
            },
            "claim": {
                "name": self.claim_name,
                "edges": list(self.claim_edges) if self.claim_edges else [],
                "angles": list(self.claim_angles) if self.claim_angles else [],
            },
        }



# ---------------------------------------------------------------- 条件解析


@dataclass
class Constraint:
    """一条能被写成残差的几何条件。

    `soft=True` 的是「先验」而不是「条件」：它表达的是「图形大概长这样」，
    不参与「解出来了没有」的判定（硬约束残差），只影响在众多可能的解里挑
    哪一个。缺了它，条件不足的题（平行四边形、等腰、只给一个角的题）会解
    出一整族形状——实测把平行四边形压成了一条线。
    """

    kind: str
    label: str
    fn: object  # (points dict) -> list[float]
    soft: bool = False


def _pt(P: dict, name: str) -> np.ndarray | None:
    return P.get(name)


def _seg_pts(P: dict, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    if len(name) != 2:
        return None
    a, b = P.get(name[0]), P.get(name[1])
    if a is None or b is None:
        return None
    return a, b


def points_on_circle(p: Problem) -> list[str]:
    """题干里「点 X 在圆上」这类说法。

    模型经常只把画出来的两个点放进 circles[].through，漏掉这种纯文字条件。
    漏掉的后果很具体：g2 少了「C 在圆上」，C 就成了一条射线上的自由点，
    答案直接变成不唯一。
    """
    out: list[str] = []
    texts = list(p.given) + [p.problem_text]
    for s in texts:
        t = s.replace(" ", "")
        for m in re.finditer(r"点?([A-Z])在(?:⊙|圆)([A-Z])?上", t):
            out.append(m.group(1))
        for m in re.finditer(r"点?([A-Z])在圆周上", t):
            out.append(m.group(1))
    return out


def enrich_from_text(p: Problem, skip_equal: set[frozenset] | None = None) -> tuple[Problem, list[str]]:
    """用题干文字补齐图形结构。返回 (新 Problem, 补了什么)。

    图是给学生看的，文字是给阅卷人看的，两者信息并不完全重合：
    图上画了不一定写，写了不一定画。求解要的是并集。

    最贵的漏读是「四边形 ABCD 为正方形」这种整句声明——它不在图上画任何
    符号，模型就把原文抄进 given 了事。少了它，正方形退化成矩形，答案
    从一个数变成一段区间。所以这里调 relations 把整句翻译成约束。
    """
    import copy

    q = copy.deepcopy(p)
    notes: list[str] = []

    # 待证结论不能当条件用（否则就是拿结论证明结论）。上一次 enrich 可能
    # 已经把它写成等长标记了，这里连同旧的一起清掉——solve 内部会再调一次
    # enrich，两次之间的状态必须一致。
    if skip_equal:
        q.marks = [
            m
            for m in q.marks
            if not (
                m.type == "equal_length"
                and len(m.of) == 2
                and frozenset(m.of) in skip_equal
            )
        ]

    # 0) 题干里的整句声明：特殊图形、平行、等长、垂直、中点、交点
    d = relations.derive(q, skip_equal=skip_equal)
    if d.marks:
        q.marks = q.marks + d.marks
    if d.segments:
        have = {tuple(sorted(s)) for s in q.segments}
        for s in d.segments:
            key = tuple(sorted(s))
            if key not in have:
                q.segments.append(s)
                have.add(key)
    notes.extend(d.notes)

    # 1) 「点 C 在圆上」
    if q.circles:
        ring: set[str] = set()
        for c in q.circles:
            ring |= set(c.through or [])
        for x in points_on_circle(p):
            if x in q.points and x not in ring:
                q.circles[0].through.append(x)
                ring.add(x)
                notes.append(f"题干称 {x} 在圆上，已补进圆周点（图上没标出来）")

    # 2) 「AB 是 ⊙O 的直径」→ A、B 在圆上，O 是圆心
    for s in list(p.given) + [p.problem_text]:
        t = s.replace(" ", "")
        m = re.search(r"([A-Z])([A-Z])是(?:⊙|圆)([A-Z])的?直径", t)
        if not m:
            continue
        a, b, o = m.groups()
        if not {a, b} <= set(q.points):
            continue
        if not q.circles:
            q.circles.append(Circle(center=o, through=[a, b]))
            notes.append(f"题干称 {a}{b} 是 ⊙{o} 的直径，已据此建圆")
        else:
            c0 = q.circles[0]
            for x in (a, b):
                if x not in c0.through:
                    c0.through.append(x)
            if not c0.center:
                c0.center = o
        if o in q.points:
            notes.append(f"{o} 是圆心（由直径 {a}{b} 推得）")

    return q, notes


def ray_targets(p: Problem) -> list[tuple[str, str, str]]:
    """「交 AB 的延长线于 P」→ (P, A, B)：P 在 B 那一侧。

    只保证共线是不够的：P 可能落在 A 外侧，或者干脆落在 A、B 之间，
    那样解出来的 ∠P 会变成对顶角或补角。方向必须约束住。
    """
    out: list[tuple[str, str, str]] = []
    for s in list(p.given) + [p.problem_text]:
        t = s.replace(" ", "")
        for m in re.finditer(r"交([A-Z])([A-Z])的延长线于([A-Z])", t):
            out.append((m.group(3), m.group(1), m.group(2)))
        for m in re.finditer(r"([A-Z])在([A-Z])([A-Z])的延长线上", t):
            out.append((m.group(1), m.group(2), m.group(3)))
    return out


def tangent_pairs(p: Problem) -> list[tuple[str, str, str]]:
    """从「过 C 作 ⊙O 的切线，交 AB 延长线于 P」推出 (切点, 圆心, 切线上的点)。

    为什么必须从文字补：模型读图时经常把这里的直角标到别处去。g2 就标成了
    ∠ACB = 90°——那是直径所对的圆周角，虽然也对，但切线真正的条件是
    「半径 ⊥ 切线」。少了这条，切点 C 就少一维约束，答案直接不唯一。
    """
    if not p.circles:
        return []
    c0 = p.circles[0]
    o = c0.center
    if not o or o not in p.points:
        return []
    on_circle = set(c0.through or [])
    out: list[tuple[str, str, str]] = []

    for s in list(p.given) + [p.problem_text]:
        t = s.replace(" ", "")
        if "切线" not in t:
            continue
        for m in re.finditer(r"过([A-Z])(?:点)?作", t):
            tp = m.group(1)
            if tp not in on_circle:
                continue
            # 与切点相连、又不在圆上的点，才是切线方向上的点
            for a, b in p.segments:
                if tp not in (a, b):
                    continue
                other = b if a == tp else a
                if other != o and other not in on_circle and other in p.points:
                    if (tp, o, other) not in out:
                        out.append((tp, o, other))

        # 「PA、PB 是 ⊙O 的切线」：一句话给两条切线，切点是落在圆上的那一端
        m = re.search(r"((?:[A-Z]{2}[、,，和])*[A-Z]{2})是(?:⊙|圆)([A-Z])的?切线", t)
        if m and m.group(2) == o:
            for nm in re.findall(r"[A-Z]{2}", m.group(1)):
                for tp, far in ((nm[0], nm[1]), (nm[1], nm[0])):
                    if tp in on_circle and far not in on_circle and far in p.points:
                        if (tp, o, far) not in out:
                            out.append((tp, o, far))

        # 「PA 切 ⊙O 于点 A」：切点写在后面
        for m in re.finditer(r"([A-Z])([A-Z])切(?:⊙|圆)([A-Z])于(?:点)?([A-Z])", t):
            p1, p2, o2, tp = m.groups()
            if o2 != o or tp not in on_circle:
                continue
            far = p2 if p1 == tp else p1
            if far not in on_circle and far in p.points and (tp, o, far) not in out:
                out.append((tp, o, far))
    return out


def _name(a: str, b: str) -> str:
    """线段名的规范写法：两端点排序后拼接，AC 与 CA 视为同一条。"""
    return "".join(sorted((a, b)))


def repair_marks(p: Problem) -> list[str]:
    """把切点处的直角标记归到正确的那个角上，返回改了什么。

    模型的通病：图上明明画着「半径 ⊥ 切线」，它却把直角记到两条弦之间
    （g2 就写成 ∠ACB）。而 ∠ACB 恰好也是 90°——直径所对的圆周角，
    数值上完全成立，所以任何等式检验都发现不了，只有对着原图才看得出：
    直角小方块的位置不对。重绘图照着错标记画，就成了「标错了角」。

    这里只在题干明确说了切线时才更正，并且把更正过程说出来，不静默改。
    """
    notes: list[str] = []
    tans = tangent_pairs(p)
    if not tans:
        return notes

    for m in p.marks:
        if m.type != "right_angle" or not m.at:
            continue
        for tp, o, far in tans:
            if m.at != tp:
                continue
            want = sorted({_name(o, tp), _name(tp, far)})
            have = sorted({_name(s[0], s[1]) for s in (_split(s) for s in m.of) if s})
            if have == want:
                break
            old = "、".join(have) or "未指明"
            m.of = want
            notes.append(
                f"{tp} 处的直角：模型读成了 {old}，但切线要求半径 ⊥ 切线，"
                f"已按原图更正为 {want[0]} ⊥ {want[1]}"
            )
            break

    return notes


def _split(s: str) -> tuple[str, str] | None:
    t = (s or "").strip()
    return (t[0], t[1]) if len(t) == 2 and t.isalpha() else None


def _is_diameter(p: Problem, o: str, a: str, b: str) -> bool:
    """O 是不是 AB 的中点（等价于 AB 是直径）。

    先看题干有没有明说；没说就看图上三点是不是真在一条线上。两者都判不了
    就取否——少认一条直径只是少一条约束，认错一条会直接让整道题无解。
    """
    t = (p.problem_text + "".join(p.given)).replace(" ", "")
    if re.search(rf"{a}{b}是(?:⊙|圆){o}的?直径", t) or re.search(rf"{b}{a}是(?:⊙|圆){o}的?直径", t):
        return True
    if not p.pos or not all(k in p.pos for k in (o, a, b)):
        return False
    (ox, oy), (ax, ay), (bx, by) = p.pos[o], p.pos[a], p.pos[b]
    ux, uy = ax - ox, ay - oy
    vx, vy = bx - ox, by - oy
    nu, nv = math.hypot(ux, uy), math.hypot(vx, vy)
    if nu < 1e-9 or nv < 1e-9:
        return False
    if abs(ux * vy - uy * vx) / (nu * nv) > 0.08:
        return False          # 不共线
    return ux * vx + uy * vy < 0      # 反向 → O 在中间


def collinear_groups(p: Problem) -> list[set[str]]:
    """从题干里找出「三点共线」的组（延长线、在线段上这几种说法）。"""
    groups: list[set[str]] = []
    texts = list(p.given) + list(p.relations) + [p.problem_text]

    for s in texts:
        t = s.replace(" ", "")
        # 交AB的延长线于P
        for m in re.finditer(r"交([A-Z])([A-Z])的延长线于([A-Z])", t):
            groups.append({m.group(1), m.group(2), m.group(3)})
        # P在AB上 / P在AB的延长线上
        for m in re.finditer(r"([A-Z])在([A-Z])([A-Z])(?:的延长线)?上", t):
            groups.append({m.group(1), m.group(2), m.group(3)})
        # AB的延长线
        for m in re.finditer(r"([A-Z])([A-Z])的延长线", t):
            groups.append({m.group(1), m.group(2)})

    # 直径也是一条直线：A、O、B 共线（圆心在直径上）。但不能把「圆心 + 前两个
    # 圆周点」直接当直径——那只是猜。图上量得出三点大体共线、或题干明说是直径，
    # 才算。乱认直径会把条件写成自相矛盾（残差 0.5 却看不出哪条错了）。
    for c in p.circles:
        if not c.center or len(c.through) < 2:
            continue
        ring = list(dict.fromkeys(c.through))
        for i in range(len(ring)):
            for j in range(i + 1, len(ring)):
                a, b = ring[i], ring[j]
                if not _is_diameter(p, c.center, a, b):
                    continue
                g = {c.center, a, b}
                if g not in groups:
                    groups.append(g)

    # 「AE 与 CD 相交于 F」和「图上 F 落在 AE 上」这两类，交给 relations 统一处理：
    # 前者从题面读，后者从「哪些点真的连了线 + 目测位置」推。少了 F 的这两条，
    # CE 就不是一个数而是一族数（实测落在 1.379 ~ 19.95）。
    groups.extend(g for g in relations.derive(p).lines)

    # 去重、只保留点集里存在的
    pts = set(p.points)
    uniq: list[set[str]] = []
    for g in groups:
        g2 = {x for x in g if x in pts}
        if len(g2) >= 2 and g2 not in uniq:
            uniq.append(g2)
    return uniq


def _lengths(p: Problem) -> dict[str, float]:
    """题干里明确给出的长度。

    必须先把角度式子抠掉：「∠AOB=80°」里的「OB=80」长得和「AB=6」一模一样，
    不剥就会给 OB 加一条长度 80 的约束，图形直接无解。
    """
    out: dict[str, float] = {}
    for s in list(p.given) + [p.problem_text]:
        t = s.replace(" ", "")
        t = re.sub(r"∠[A-Z]{1,3}\s*[=＝]\s*\d+(?:\.\d+)?\s*°?", "", t)
        for m in re.finditer(r"([A-Z])([A-Z])\s*[=＝]\s*(\d+(?:\.\d+)?)", t):
            out[m.group(1) + m.group(2)] = float(m.group(3))
    return out


def _angles_given(p: Problem) -> list[tuple[str, float]]:
    """题干里给出的角度：∠A = 25° / ∠ACB = 25°。"""
    out: list[tuple[str, float]] = []
    for s in list(p.given) + [p.problem_text]:
        t = s.replace(" ", "")
        for m in re.finditer(r"∠([A-Z]{1,3})\s*[=＝]\s*(\d+(?:\.\d+)?)\s*°?", t):
            name, val = m.group(1), float(m.group(2))
            if name and abs(val - 90) > 1e-6:  # 90° 走垂直约束，更稳
                out.append((name, val))
    return out


def _inside_given(p: Problem) -> list[tuple[str, str]]:
    """题干里「P 是平行四边形 ABCD 内部的一点」这类说法。

    这不是修饰语，是一条实打实的约束。少掉它，那个点会跑到形外——同一组
    条件在形外照样成立，但那时量到的是这个角的**补角**，结论于是「不成立」，
    而且报出来的两个度数加起来正好 180°，看着像读错了，其实是没管住位置。
    """
    out: list[tuple[str, str]] = []
    for s in list(p.given) + [p.problem_text]:
        t = s.replace(" ", "")
        for m in re.finditer(r"([A-Z])\s*(?:在|是)\s*[^A-Z]{0,8}([A-Z]{3,4})\s*内", t):
            out.append((m.group(1), m.group(2)))
    return out


def _angle_pairs_given(p: Problem) -> list[tuple[str, str]]:
    """题干里给出的「两个角一样大」：∠PBA = ∠PDA。

    这是最容易被漏掉的一类条件——它不给度数，只给相等。漏了它，题目就少
    一条约束：那个本来被约束住的点会变成自由点，优化器随便找个位置停下，
    结论于是「不成立」。用户看到「多半是读错了」，其实读得没错，是没认。
    """
    out: list[tuple[str, str]] = []
    for s in list(p.given) + [p.problem_text]:
        t = s.replace(" ", "")
        for m in re.finditer(r"∠([A-Z]{1,3})\s*[=＝]\s*∠([A-Z]{1,3})", t):
            out.append((m.group(1), m.group(2)))
    return out


def _resolve_angle(
    name: str, arms: dict[str, set[str]]
) -> tuple[str, str, str] | None:
    """角名 → (顶点, 臂 1, 臂 2)。

    ∠PBA 的顶点是中间那个字母，这是初中定死的写法，不用猜。只有单字母的
    ∠A 才需要看图上从 A 连出去哪两条线，而且必须恰好两条——三臂以上就
    分不清说的是哪个角，宁可放弃也不猜，猜错的代价是整道题角度全错位。
    """
    if len(name) == 3:
        return name[1], name[0], name[2]
    if len(name) == 1:
        cand = sorted(arms.get(name, set()))
        if len(cand) == 2:
            return name, cand[0], cand[1]
    return None


def _vertex_arms(p: Problem, groups: list[set[str]]) -> dict[str, set[str]]:
    """每个点「视线」能到的方向：真实连线的端点 + 共线方向上的点。"""
    arms: dict[str, set[str]] = {x: set() for x in p.points}
    for a, b in p.segments:
        if a in arms and b in arms:
            arms[a].add(b)
            arms[b].add(a)
    for g in groups:
        for x in g:
            if x in arms:
                arms[x] |= g - {x}
    return arms


def _angle_at(P: dict, vertex: str, a: str, b: str) -> float:
    """角 a-vertex-b 的余弦（用余弦避免 atan2 的分支问题）。"""
    v = P[vertex]
    u1, u2 = P[a] - v, P[b] - v
    n1, n2 = np.linalg.norm(u1), np.linalg.norm(u2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 1.0
    return float(np.dot(u1, u2) / (n1 * n2))


def build_constraints(
    p: Problem, scale: float, claim: "Claim | None" = None
) -> tuple[list[Constraint], list[str]]:
    """把所有条件编译成残差函数。返回 (约束, 用到的条件说明)。

    `claim` 是待证的结论，必须传进来：结论的写法（∠PAB = ∠PCB）和条件一模
    一样，不排除就会把它当成条件建约束，等于用结论证明结论——算出来永远
    成立，而且永远看不出错在哪。
    """
    cons: list[Constraint] = []
    used: list[str] = []
    groups = collinear_groups(p)
    lengths = _lengths(p)

    # --- 定标 ---
    # 纯角度题（g2 就是：只给 ∠A=25° 和一堆垂直/切线关系）里所有条件都是
    # 尺度无关的，于是优化器会发现「把整张图缩成一个点」最省事：角度照旧
    # 成立、共线照旧成立、残差一起趋零，最后解出一个塌成一团粉尘的假图形
    # （实测点距只有 0.08，残差卡在 1e-3 下不去）。钉住一条边的长度即可，
    # 图形整体缩放不影响任何角度，答案不变。
    if not lengths and p.segments:
        a, b = p.segments[0]

        def f(P, a=a, b=b, scale=scale):
            sp = _seg_pts(P, _name(a, b))
            if sp is None:
                return []
            return [float(np.linalg.norm(sp[0] - sp[1])) / scale - 1.0]

        cons.append(Constraint("anchor", f"取 |{_name(a, b)}| 作为定标长度", f))
        used.append(f"以 {_name(a, b)} 作为长度基准")

    # --- 定长 ---
    for name, val in lengths.items():
        def f(P, name=name, val=val):
            sp = _seg_pts(P, name)
            if sp is None:
                return []
            return [(float(np.linalg.norm(sp[0] - sp[1])) - val) / scale]

        cons.append(Constraint("length", f"|{name}| = {val:g}", f))
        used.append(f"{name} = {val:g}")

    # --- 直角（图上标记 + 题干里的 90°）---
    for m in p.marks:
        if m.type != "right_angle":
            continue
        at = m.at
        ofs = [x for x in m.of if len(x) == 2 and at in x]
        if len(ofs) < 2:
            continue
        ends = [(ofs[0].replace(at, ""), ofs[1].replace(at, ""))]

        def f(P, at=at, ends=ends):
            a, b = ends[0]
            if not all(k in P for k in (at, a, b)):
                return []
            cos = _angle_at(P, at, a, b)
            return [cos]

        cons.append(Constraint("right_angle", f"∠{''.join([ends[0][0], at, ends[0][1]])} = 90°", f))
        used.append(f"{at} 处直角")

    # --- 中点 ---
    for m in p.marks:
        if m.type != "midpoint" or not m.at:
            continue
        seg = m.of[0] if m.of and len(m.of[0]) == 2 else ""
        if not seg:
            for g in groups:
                if m.at in g:
                    rest = sorted(g - {m.at})
                    if len(rest) == 2:
                        seg = "".join(rest)
                        break
        if not seg:
            continue

        def f(P, at=m.at, seg=seg):
            if not all(k in P for k in (at, seg[0], seg[1])):
                return []
            mid = (P[seg[0]] + P[seg[1]]) / 2
            d = (P[at] - mid) / scale
            return [float(d[0]), float(d[1])]

        cons.append(Constraint("midpoint", f"{m.at} 是 {seg} 的中点", f))
        used.append(f"{m.at} 中点")

    # --- 等长 ---
    # 同一条关系会从两处冒出来（图上读到的等长标记 + 题面的「AE=AC」），
    # 记一遍去重，否则步骤列表里会写两遍「AE = AC」，看着像读重了。
    seen_equal: set[frozenset] = set()
    for m in p.marks:
        if m.type != "equal_length" or len(m.of) < 2:
            continue
        a, b = m.of[0], m.of[1]
        if len(a) != 2 or len(b) != 2:
            continue
        key = frozenset({_name(a[0], a[1]), _name(b[0], b[1])})
        if key in seen_equal:
            continue
        seen_equal.add(key)

        def f(P, a=a, b=b):
            sa, sb = _seg_pts(P, a), _seg_pts(P, b)
            if sa is None or sb is None:
                return []
            la = float(np.linalg.norm(sa[0] - sa[1]))
            lb = float(np.linalg.norm(sb[0] - sb[1]))
            return [(la - lb) / scale]

        cons.append(Constraint("equal", f"|{a}| = |{b}|", f))
        used.append(f"{a} = {b}")

    # --- 平行 ---
    seen_par: set[frozenset] = set()
    for m in p.marks:
        if m.type != "parallel" or len(m.of) < 2:
            continue
        a, b = m.of[0], m.of[1]
        key = frozenset({_name(a[0], a[1]), _name(b[0], b[1])})
        if key in seen_par:
            continue
        seen_par.add(key)

        def f(P, a=a, b=b):
            sa, sb = _seg_pts(P, a), _seg_pts(P, b)
            if sa is None or sb is None:
                return []
            u, v = sa[0] - sa[1], sb[0] - sb[1]
            return [float(u[0] * v[1] - u[1] * v[0]) / (scale * scale)]

        cons.append(Constraint("parallel", f"{a} ∥ {b}", f))
        used.append(f"{a} ∥ {b}")

    # --- 共线 ---
    for g in groups:
        if len(g) < 3:
            continue
        pts = sorted(g)
        base = pts[0]
        for other in pts[1:]:
            for third in pts[2:]:
                if third == other or (base, other, third) != tuple(sorted((base, other, third))):
                    continue

                def f(P, a=base, b=other, c=third):
                    if not all(k in P for k in (a, b, c)):
                        return []
                    u, v = P[b] - P[a], P[c] - P[a]
                    return [float(u[0] * v[1] - u[1] * v[0]) / (scale * scale)]

                cons.append(Constraint("collinear", f"{base}、{other}、{third} 共线", f))
                used.append(f"{base}{other}{third} 共线")
                break

    # --- 点必须落在两点「之间」，不能跑到延长线上 ---
    # 「AE 与 CD 相交于 F」说的是线段相交。只写共线的话，F 完全可能落在
    # AE 的延长线上——共线照样成立，但那个图已经不是题里的图了：本题的
    # E 正好有两个解，一个 F 在线段内（CE=0.732），一个在外面（CE=2.732），
    # 两个都「满足条件」，答案就报成不唯一。加个内点惩罚就分开了。
    for x, a, b in relations.derive(p).between:
        if not all(k in p.points for k in (x, a, b)):
            continue

        def f(P, x=x, a=a, b=b):
            u = P[b] - P[a]
            L2 = float(np.dot(u, u))
            if L2 < 1e-18:
                # 必须返回和正常分支一样多的残差。少返回一个，雅可比的
                # 行数就跟着残差一起变，下一轮直接 broadcast 报错。
                return [0.0, 0.0]
            t = float(np.dot(P[x] - P[a], u)) / L2
            # 满足时返回 0，违反时线性惩罚：不干扰正常求解
            return [max(0.0, -t) * 8.0, max(0.0, t - 1.0) * 8.0]

        cons.append(Constraint("between", f"{x} 落在 {a}{b} 之间", f))
        used.append(f"{x} 在 {_name(a, b)} 上（线段内）")

    # --- 点在多边形内部（「P 是平行四边形 ABCD 内部的一点」）---
    # 判据是「在每条边的同一侧」：与对顶点同号即可。写成软惩罚而不是硬
    # 等式，因为它在内部时不该给求解增加任何约束力，只有跑出去才拉回来。
    for x, poly in _inside_given(p):
        verts = [v for v in poly if v in p.points]
        if len(verts) < 3 or x not in p.points or x in verts:
            continue

        def f(P, x=x, verts=verts):
            if not all(k in P for k in verts + [x]):
                return []
            out = []
            n = len(verts)
            for i in range(n):
                a, b, c = verts[i], verts[(i + 1) % n], verts[(i + 2) % n]
                u = P[b] - P[a]
                s1 = float(u[0] * (P[c] - P[a])[1] - u[1] * (P[c] - P[a])[0])
                s2 = float(u[0] * (P[x] - P[a])[1] - u[1] * (P[x] - P[a])[0])
                if abs(s1) < 1e-12:
                    out.append(0.0)
                elif s1 > 0:
                    out.append(max(0.0, -s2) / (scale * scale) * 12.0)
                else:
                    out.append(max(0.0, s2) / (scale * scale) * 12.0)
            return out

        cons.append(Constraint("inside", f"{x} 在 {poly} 内部", f))
        used.append(f"{x} 在 {poly} 内")

    # --- 圆：点都在圆上，半径作为隐含量（取第一条半径定标）---
    for c in p.circles:
        if not c.center or len(c.through) < 2:
            continue
        o, p1, p2 = c.center, c.through[0], c.through[1]

        def f(P, o=o, p1=p1, p2=p2):
            if not all(k in P for k in (o, p1, p2)):
                return []
            r1 = float(np.linalg.norm(P[p1] - P[o]))
            r2 = float(np.linalg.norm(P[p2] - P[o]))
            return [(r1 - r2) / scale]

        cons.append(Constraint("circle", f"{p1}、{p2} 都在 ⊙{o} 上", f))
        used.append(f"⊙{o} 过 {p1}{p2}")

    # 其余圆周点：到圆心等距
    for c in p.circles:
        if not c.center or len(c.through) < 2:
            continue
        o, ref = c.center, c.through[0]
        for other in c.through[2:]:
            def f(P, o=o, ref=ref, other=other):
                if not all(k in P for k in (o, ref, other)):
                    return []
                r1 = float(np.linalg.norm(P[ref] - P[o]))
                r2 = float(np.linalg.norm(P[other] - P[o]))
                return [(r1 - r2) / scale]

            cons.append(Constraint("circle", f"{other} 在 ⊙{o} 上", f))
            used.append(f"{other} 在圆上")

    # --- 切线：半径 ⊥ 切线（文字补出来的条件）---
    for tp, o, far in tangent_pairs(p):
        def f(P, tp=tp, o=o, far=far):
            if not all(k in P for k in (tp, o, far)):
                return []
            return [_angle_at(P, tp, o, far)]

        cons.append(Constraint("tangent", f"{o}{tp} ⊥ {tp}{far}（切线）", f))
        used.append(f"切线 {tp}{far} ⊥ 半径 {o}{tp}")

    # --- 两个角相等（∠PBA = ∠PDA）---
    # 必须在「角度」之前、且用同一套 arms：这是和定角平级的一类硬约束，
    # 漏掉它不是「少算了一点」，是题目本身被换成了另一道题。
    arms = _vertex_arms(p, groups)
    banned: set[frozenset] = set()
    if claim is not None and claim.kind == "angle_equal":
        banned.add(frozenset({claim.a, claim.b}))   # 待证的，不是已知的
    seen_apair: set[frozenset] = set()
    for n1, n2 in _angle_pairs_given(p):
        if frozenset({n1, n2}) in banned:
            continue
        r1, r2 = _resolve_angle(n1, arms), _resolve_angle(n2, arms)
        if r1 is None or r2 is None or r1 == r2:
            continue
        (v1, a1, b1), (v2, a2, b2) = r1, r2
        if not all(k in p.points for k in (v1, a1, b1, v2, a2, b2)):
            continue
        key = frozenset({tuple(r1), tuple(r2)})
        if key in seen_apair:
            continue
        seen_apair.add(key)

        def f(P, v1=v1, a1=a1, b1=b1, v2=v2, a2=a2, b2=b2):
            if not all(k in P for k in (v1, a1, b1, v2, a2, b2)):
                return []
            return [_angle_at(P, v1, a1, b1) - _angle_at(P, v2, a2, b2)]

        cons.append(Constraint("angle_equal", f"∠{n1} = ∠{n2}", f))
        used.append(f"∠{n1} = ∠{n2}")

    # --- 角度 ---
    seen_angle: set[tuple[str, float]] = set()
    for name, deg in _angles_given(p):
        key = (name, deg)
        if key in seen_angle:
            continue  # given 与 problem_text 会各解析一遍，去重
        seen_angle.add(key)
        if len(name) == 3:
            vertex, a, b = name[1], name[0], name[2]
        else:
            vertex = name
            cand = sorted(arms.get(vertex, set()))
            if len(cand) < 2:
                continue
            a, b = cand[0], cand[1]

        target = math.cos(math.radians(deg))

        def f(P, vertex=vertex, a=a, b=b, target=target):
            if not all(k in P for k in (vertex, a, b)):
                return []
            return [_angle_at(P, vertex, a, b) - target]

        cons.append(Constraint("angle", f"∠{name} = {deg:g}°", f))
        used.append(f"∠{name} = {deg:g}°")

    # --- 方向：「交 AB 的延长线于 P」要求 P 在 B 外侧 ---
    for tp, ta, tb in ray_targets(p):
        if not all(k in p.points for k in (tp, ta, tb)):
            continue

        def f(P, tp=tp, ta=ta, tb=tb):
            # 参数 t = (P-B)·(B-A)/|B-A|²，t>0 表示 P 在 B 外侧
            u = P[tb] - P[ta]
            L2 = float(np.dot(u, u))
            if L2 < 1e-18:
                return [0.0]
            t = float(np.dot(P[tp] - P[tb], u)) / L2
            # 违反时给线性惩罚；满足时为 0，不干扰求解
            return [max(0.0, -t) * 8.0]

        cons.append(Constraint("ray", f"{tp} 在 {ta}{tb} 的延长线上", f))
        used.append(f"{tp} 在 {tb} 外侧")

    # --- 防退化：不同的点不能挤在一起 ---
    # 不加这条，优化器会发现「把所有点缩到原点」最省事：叉积全为 0、
    # 到圆心距离全相等、角度在零向量上按 1.0 兜底，于是所有残差一起归零，
    # 解出一个看着成立的空壳。g2 实测就得到过 A=B=C 全重合的假解。
    names = list(p.points)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]

            def f(P, a=a, b=b):
                d = float(np.linalg.norm(P[a] - P[b])) / scale
                return [max(0.0, 0.08 - d) * 15.0]

            cons.append(Constraint("distinct", f"{a}、{b} 不重合", f))

    return cons, used


# ---------------------------------------------------------------- 数值求解


class Model:
    def __init__(self, points: list[str], cons: list[Constraint]):
        self.points = points
        self.index = {p: i for i, p in enumerate(points)}
        self.cons = cons
        # 硬约束排在前、软先验排在最后（由 build_constraints 保证）。记住硬
        # 约束占多少行，就能在残差向量上把它们切开：收敛判据只看硬的，
        # 挑哪个解看整体。
        probe = {p: np.zeros(2) for p in points}
        self.n_hard = 3 + sum(
            len(c.fn(probe) or []) for c in cons if not c.soft
        )

    def n(self) -> int:
        return 2 * len(self.points)

    def unpack(self, x: np.ndarray) -> dict[str, np.ndarray]:
        return {p: x[2 * i : 2 * i + 2] for p, i in self.index.items()}

    def residual(self, x: np.ndarray) -> np.ndarray:
        P = self.unpack(x)
        out: list[float] = []
        # 规范固定：消掉整体平移(2)与旋转(1)，否则雅可比始终缺秩
        out.append(x[0])
        out.append(x[1])
        if len(self.points) > 1:
            out.append(x[3])
        for c in self.cons:
            try:
                out.extend(c.fn(P))
            except Exception:
                out.extend([0.0])
        return np.asarray(out, dtype=float)

    def jacobian(self, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        base = self.residual(x)
        J = np.zeros((base.size, x.size))
        for i in range(x.size):
            dx = np.zeros_like(x)
            dx[i] = eps
            J[:, i] = (self.residual(x + dx) - base) / eps
        return J


def _gauss_newton(model: Model, x0: np.ndarray, iters: int = 200) -> tuple[np.ndarray, float]:
    """Levenberg–Marquardt 阻尼最小二乘。

    早期版本手写「高斯牛顿 + 回溯线搜索」，在 g2 上 8 个随机起点只有 1 个
    能落进解里——不是解不存在，是一步跳太空。LM 用自适应阻尼代替固定步长，
    收敛率立刻就上来了；而多起点能不能收敛，直接决定「答案唯不唯一」这个
    判断可不可信（收敛不了会误报成「算不出来」）。
    """
    x = x0.astype(float).copy()
    r = model.residual(x)
    n_hard = getattr(model, "n_hard", r.size)

    def hard(rr: np.ndarray) -> float:
        return float(np.linalg.norm(rr[:n_hard]))

    f = float(np.linalg.norm(r))
    best_x, best_r, best_f = x.copy(), r.copy(), f
    lam = 1e-3
    n = x.size

    for _ in range(iters):
        # 收敛只看硬约束。软先验的残差永远归不了零——原图位置是目测的
        # 0–100 的相对坐标，本来就不精确——把它算进判据，所有题都会「解不出来」。
        if hard(r) < 1e-13:
            break
        J = model.jacobian(x)
        A = J.T @ J
        g = J.T @ r
        diag = np.maximum(np.diag(A), 1e-12)

        improved = False
        for _ in range(12):
            try:
                dx = np.linalg.solve(A + lam * np.diag(diag), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            step = float(np.linalg.norm(dx))
            if step > 3.0:  # 再防一手：解析解偶尔会给出离谱的大步
                dx *= 3.0 / step
            trial = x + dx
            r_try = model.residual(trial)
            f_try = float(np.linalg.norm(r_try))
            if f_try < f:
                x, r, f = trial, r_try, f_try
                lam = max(lam * 0.3, 1e-9)
                improved = True
                break
            lam = min(lam * 4.0, 1e6)

        if not improved:
            break
        if f < best_f:
            best_x, best_r, best_f = x.copy(), r.copy(), f

    # 返回硬残差：调用方拿它判「解出来了没有」
    return best_x, hard(best_r)


# ---------------------------------------------------------------- 目标解析


@dataclass
class Goal:
    kind: str  # length | angle | area | prove | unknown
    name: str = ""
    target: float | None = None
    note: str = ""


@dataclass
class Claim:
    """证明题的结论。"""

    kind: str  # len_equal | angle_equal | perp | parallel | cong | sim
    a: str = ""
    b: str = ""
    raw: str = ""

    def label(self) -> str:
        if self.kind == "len_equal":
            return f"{self.a} = {self.b}"
        if self.kind == "angle_equal":
            return f"∠{self.a} = ∠{self.b}"
        if self.kind == "perp":
            return f"{self.a} ⊥ {self.b}"
        if self.kind == "parallel":
            return f"{self.a} ∥ {self.b}"
        if self.kind == "cong":
            return f"△{self.a} ≌ △{self.b}"
        if self.kind == "sim":
            return f"△{self.a} ∽ △{self.b}"
        return self.raw or "结论"


def _claim_in(text: str) -> Claim | None:
    """从一句话里抠出一个结论。先认角度，再认垂直/平行，最后才是等长——
    「∠A=∠B」里也能匹配出「A=B」这种鬼东西，顺序错了就全错。"""
    t = (text or "").replace(" ", "").replace("　", "")
    if not t:
        return None
    # 「△ABE≌△CBF」得排在最前：三字母三角形名比两字母线段更具体，
    # 让等长那条先匹上，就会把 ABE 当成一条"边"。
    m = re.search(r"△([A-Z]{3})\s*(?:≌|≡|全等于)\s*△([A-Z]{3})", t)
    if m:
        return Claim("cong", m.group(1), m.group(2), m.group(0))
    m = re.search(r"△([A-Z]{3})\s*(?:∽|∼|~|相似于)\s*△([A-Z]{3})", t)
    if m:
        return Claim("sim", m.group(1), m.group(2), m.group(0))
    m = re.search(r"∠([A-Z]{1,3})\s*[=＝]\s*∠([A-Z]{1,3})", t)
    if m:
        return Claim("angle_equal", m.group(1), m.group(2), m.group(0))
    m = re.search(r"([A-Z]{2})\s*(?:⊥|垂直于)\s*([A-Z]{2})", t)
    if m:
        return Claim("perp", m.group(1), m.group(2), m.group(0))
    m = re.search(r"([A-Z]{2})\s*(?:∥|//|平行于)\s*([A-Z]{2})", t)
    if m:
        return Claim("parallel", m.group(1), m.group(2), m.group(0))
    m = re.search(r"([A-Z]{2})\s*[=＝]\s*([A-Z]{2})", t)
    if m:
        return Claim("len_equal", m.group(1), m.group(2), m.group(0))
    return None


def parse_claim(p: Problem) -> Claim | None:
    """认出「求证……」这类证明题。

    为什么单列一条路：证明题没有数值答案。硬按「求值」处理，系统会去算
    CE 到底多长，然后因为图形有一个自由度而报「条件不足」——题明明是会做的，
    报的却是「算不出来」。这题就是这样：它压根不要一个数。

    结论的字节通常写在题面最后一句，但模型经常把「求证：CE=CF」整个省略，
    只把 CE=CF 抄进 given。所以找不到「求证」字样时，退而按位置取：
    题面最后一句 → given 倒序。顺序反了会把真条件 AE=AC 当成结论。
    """
    texts = [p.problem_text] + list(p.given) + [p.goal]
    for s in texts:
        t = (s or "").replace(" ", "").replace("　", "")
        m = re.search(r"(?:求证|证明)[:：]?\s*(.+)", t)
        if m:
            c = _claim_in(m.group(1))
            if c and set(c.a) <= set(p.points) and set(c.b) <= set(p.points):
                return c

    # goal 字段本来该写「求什么」，可它直接写成一个等式（"CE=CF"）时，
    # 那只能是待证结论——「求」后面不会跟等号。模型经常把题干压缩成
    # 这样，比整段原文还准，浪费掉可惜。
    g = (p.goal or "").replace(" ", "")
    if g and not re.search(r"求|长|度数|面积", g):
        c = _claim_in(g)
        if c and set(c.a) <= set(p.points) and set(c.b) <= set(p.points):
            return c

    sign = (p.goal or "") + (p.problem_text or "")
    if "证明" not in sign and "求证" not in sign:
        return None

    tail = [s for s in re.split(r"[。；;]", (p.problem_text or "").replace(" ", "")) if s]
    for s in reversed(tail[-2:]):
        c = _claim_in(s)
        if c and set(c.a) <= set(p.points) and set(c.b) <= set(p.points):
            return c
    for s in reversed(list(p.given)):
        c = _claim_in(s)
        if c and set(c.a) <= set(p.points) and set(c.b) <= set(p.points):
            return c
    return None


def _dist(P: dict, name: str) -> float | None:
    sp = _seg_pts(P, name)
    if sp is None:
        return None
    return float(np.linalg.norm(sp[0] - sp[1]))


def _angle_deg(P: dict, name: str, p: Problem | None = None) -> float | None:
    if len(name) == 3:
        v, a, b = name[1], name[0], name[2]
    elif p is not None and len(name) == 1:
        pair = angle_arms(p, name, P)
        if pair is None:
            return None
        v, (a, b) = name, pair
    else:
        return None
    if not all(k in P for k in (v, a, b)):
        return None
    cos = max(-1.0, min(1.0, _angle_at(P, v, a, b)))
    return math.degrees(math.acos(cos))


def _verify_triangles(P: dict, claim: Claim, tol_deg: float) -> tuple[bool, str]:
    """检验「△ABE ≌ △BCF」「△ADE ∽ △ABC」。

    字母的书写顺序就是对应关系——这是课本的规矩，也是唯一能确定的对应：
    只看两个三角形名，谁和谁对应没有别的信息来源。所以读图时把结论抄对
    顺序很重要，抄反了这里会如实报「不成立」。
    """
    ta, tb = claim.a, claim.b
    if len(ta) != 3 or len(tb) != 3:
        return False, f"{ta}、{tb} 不是三角形名"

    lens: list[tuple[str, float, float]] = []
    for i in range(3):
        n1 = ta[i] + ta[(i + 1) % 3]
        n2 = tb[i] + tb[(i + 1) % 3]
        d1, d2 = _dist(P, n1), _dist(P, n2)
        if d1 is None or d2 is None:
            return False, f"{n1} 或 {n2} 不在图上"
        lens.append((f"{n1}={d1:.4g}，{n2}={d2:.4g}", d1, d2))

    if claim.kind == "cong":
        bad = [t for t, d1, d2 in lens if abs(d1 - d2) / max(d1, d2, 1e-12) >= 8e-3]
        return not bad, "；".join(t for t, _, _ in lens)

    # 相似：三组对应边要成同一比例
    rs = [d2 / max(d1, 1e-12) for _, d1, d2 in lens]
    spread = (max(rs) - min(rs)) / max(max(rs), 1e-12)
    detail = "；".join(t for t, _, _ in lens) + f"，比值 {rs[0]:.4g}（三者差 {spread:.1%}）"
    return spread < 8e-3, detail


def verify_claim(model: "Model", x: np.ndarray, claim: Claim, p: Problem) -> tuple[bool, str]:
    """在一组解出来的坐标上检验结论。返回 (是否成立, 数值依据)。"""
    P = model.unpack(x)
    tol_len = 8e-3
    tol_deg = 0.5

    if claim.kind in ("cong", "sim"):
        return _verify_triangles(P, claim, tol_deg)

    if claim.kind == "len_equal":
        la, lb = _dist(P, claim.a), _dist(P, claim.b)
        if la is None or lb is None:
            return False, f"{claim.a} 或 {claim.b} 不在图上"
        rel = abs(la - lb) / max(la, lb, 1e-12)
        return rel < tol_len, f"{claim.a}={la:.4g}，{claim.b}={lb:.4g}（相对差 {rel:.1%}）"

    if claim.kind == "angle_equal":
        da, db = _angle_deg(P, claim.a, p), _angle_deg(P, claim.b, p)
        if da is None or db is None:
            return False, f"∠{claim.a} 或 ∠{claim.b} 的两条边不明"
        return abs(da - db) < tol_deg, f"∠{claim.a}={da:.2f}°，∠{claim.b}={db:.2f}°"

    common = set(claim.a) & set(claim.b)
    if len(common) != 1:
        # 两条线段没有公共端点时，「垂直/平行」只能按方向比，不能按夹角比
        sa, sb = _seg_pts(P, claim.a), _seg_pts(P, claim.b)
        if sa is None or sb is None:
            return False, f"{claim.a} 或 {claim.b} 不在图上"
        u, v = sa[0] - sa[1], sb[0] - sb[1]
        n = float(np.linalg.norm(u) * np.linalg.norm(v))
        if n < 1e-12:
            return False, "线段退化成一个点"
        cross = abs(float(u[0] * v[1] - u[1] * v[0])) / n
        dot = abs(float(np.dot(u, v))) / n
        if claim.kind == "parallel":
            return cross < 1e-2, f"{claim.a}、{claim.b} 夹角 {math.degrees(math.asin(min(1.0, cross))):.2f}°"
        return dot < 1e-2, f"{claim.a}、{claim.b} 夹角 {math.degrees(math.acos(min(1.0, dot))):.2f}°"

    at = common.pop()
    a = (set(claim.a) - {at}).pop()
    b = (set(claim.b) - {at}).pop()
    cos = _angle_at(P, at, a, b)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    if claim.kind == "perp":
        return abs(cos) < 1e-2, f"∠{a}{at}{b} = {ang:.2f}°"
    return abs(abs(cos) - 1.0) < 1e-2, f"∠{a}{at}{b} = {ang:.2f}°"


def parse_goal(p: Problem, groups: list[set[str]], arms: dict[str, set[str]]) -> Goal:
    pts = set(p.points)
    # 只在「求……」这一句里找目标。绝不能把整段题干拿来搜——
    # 题干里的 ∠C = 90° 是已知条件，会被误当成「求 ∠C」。
    text = (p.goal or "").strip()
    if not text:
        m = re.search(r"求\s*(.+)$", p.problem_text or "")
        text = m.group(1) if m else ""

    # 角度优先：求∠P的度数
    m = re.search(r"∠\s*([A-Z]{1,3})", text)
    if m:
        return Goal("angle", m.group(1), note="度数")

    # 长度：求 CD 的长
    m = re.search(r"([A-Z]{2})\s*的?长", text)
    if m and set(m.group(1)) <= pts:
        return Goal("length", m.group(1))

    # 面积
    if "面积" in text:
        return Goal("area", note="面积")

    # 兜底：这一句里出现过的、由两个已读点组成的线段
    for m in re.finditer(r"([A-Z])([A-Z])", text):
        name = m.group(1) + m.group(2)
        if set(name) <= pts:
            return Goal("length", name, note="按题干推断")
    return Goal("unknown")


def angle_arms(p: Problem, vertex: str, P: dict) -> tuple[str, str] | None:
    """确定 ∠vertex 的两条边。

    必须优先用真实画出来的线段。「P 在 AB 延长线上、又连了 CP」这种图里，
    ∠P 指的是 ∠CPB；若无差别地取离顶点最近的两个点，会取到 A、B 这两个
    共线点，夹角恒为 0°——实测就是这样算出了一个 0° 的假答案。
    """
    groups = collinear_groups(p)
    real = sorted(
        {b for a, b in p.segments if a == vertex and b in P}
        | {a for a, b in p.segments if b == vertex and a in P}
    )
    coll: set[str] = set()
    for g in groups:
        if vertex in g:
            coll |= g - {vertex}
    others = sorted(coll - set(real))

    if len(real) >= 2:
        best: tuple[float, str, str] | None = None
        for i in range(len(real)):
            for j in range(i + 1, len(real)):
                c = _angle_at(P, vertex, real[i], real[j])
                if best is None or c > best[0]:  # cos 越大 → 角越小，取图中常画的小角
                    best = (c, real[i], real[j])
        return (best[1], best[2]) if best else None
    if len(real) == 1 and others:
        far = min(others, key=lambda t: float(np.linalg.norm(P[t] - P[vertex])))
        return (real[0], far)
    if not real and len(coll) >= 2:
        cs = sorted(coll)
        return (cs[0], cs[1])
    return None


def _measure(model: Model, x: np.ndarray, goal: Goal, p: Problem, arms: dict[str, set[str]]) -> float | None:
    P = model.unpack(x)
    if goal.kind == "length":
        sp = _seg_pts(P, goal.name)
        return float(np.linalg.norm(sp[0] - sp[1])) if sp else None
    if goal.kind == "angle":
        name = goal.name
        if len(name) == 3:
            vertex, a, b = name[1], name[0], name[2]
        else:
            vertex = name
            pair = angle_arms(p, vertex, P)
            if pair is None:
                return None
            a, b = pair
        if not all(k in P for k in (vertex, a, b)):
            return None
        cos = max(-1.0, min(1.0, _angle_at(P, vertex, a, b)))
        return math.degrees(math.acos(cos))
    return None


# ---------------------------------------------------------------- 主入口


def seed_from_pos(p: Problem, scale: float) -> np.ndarray | None:
    """用读到的近似位置造一个初值。

    数值解本身不关心朝向：约束只约束距离和角度，镜像图形同样满足。
    所以不加约束时，解出来可能是原图的左右镜像——A 跑到右边、P 跑到左边，
    用户一对照就以为认错了。用模型给的粗略位置当第一个起点，
    解就会停在离原图最近的那个解上，重绘图方向也就对上了。

    位置精度要求很低（模型给的是 0–100 的目测值），因为后面还要数值求解。
    """
    if not p.pos or len(p.pos) < len(p.points):
        return None
    xs = [v[0] for v in p.pos.values()]
    ys = [v[1] for v in p.pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    out: list[float] = []
    for name in p.points:
        x, y = p.pos[name]
        out.append((x - min(xs)) / span * scale)
        out.append(-(y - min(ys)) / span * scale)  # 图像 y 向下，数学 y 向上

    # 和约束里的定标边对齐。初值不满足 |AB| = scale 时，求解器第一轮就要
    # 用很大的一步把它拉到尺子上，顺手就把整个形状改坏了——平行四边形实测
    # 被压成一条线、B 从 A 的左下翻到右上。先自己缩到位，后面就只剩微调。
    ref: tuple[str, str] | None = None
    ref_len = scale
    known = _lengths(p)
    for name, v in known.items():
        if len(name) == 2 and all(k in p.points for k in name):
            ref, ref_len = (name[0], name[1]), float(v)
            break
    if ref is None and p.segments:
        ref = p.segments[0]
    if ref is not None and all(k in p.points for k in ref):
        ia, ib = p.points.index(ref[0]), p.points.index(ref[1])
        d = float(np.linalg.norm(np.array(out[2 * ia:2 * ia + 2]) - np.array(out[2 * ib:2 * ib + 2])))
        if d > 1e-9:
            k = ref_len / d
            out = [v * k for v in out]

    # 再做一次和求解器一致的规范对齐：把 0 号点挪到原点、1 号点压到 x 轴。
    # 求解器为了消掉整体平移与旋转的零空间，强令 x0=y0=y1=0；初值不照做的话，
    # 它不会整体转，而是单独把 1 号点挪过去——图形就是在这被压塌的。
    # 这两步都是刚体变换，形状一点没变，只是换了个摆法。
    ang = math.atan2(out[3] - out[1], out[2] - out[0])
    ca, sa = math.cos(-ang), math.sin(-ang)
    ox, oy = out[0], out[1]
    for i in range(0, len(out), 2):
        dx, dy = out[i] - ox, out[i + 1] - oy
        out[i], out[i + 1] = ca * dx - sa * dy, sa * dx + ca * dy

    return np.asarray(out, dtype=float)


def shape_score(p: Problem, P: dict, scale: float) -> float:
    """解和原图布局的差距，越小越像。

    只比形状不比大小：两边各自按自身跨度归一，解是按题意定标的，尺寸本来
    就和原图不同。收敛到的解往往不止一个——它们都满足条件，但形状可能差得
    很远（平行四边形压成一条线也是"合法解"）。用户要拿重绘图对照原图检查
    有没有读错，所以长得像比数值漂亮更重要。
    """
    if not p.pos or len(p.points) < 3 or not all(k in P for k in p.points):
        return 0.0
    xs = [p.pos[k][0] for k in p.points]
    ys = [p.pos[k][1] for k in p.points]
    span1 = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    got = np.array([P[k] for k in p.points], dtype=float)
    span2 = max(float(np.ptp(got[:, 0])), float(np.ptp(got[:, 1])), 1e-9)

    err, n = 0.0, 0
    for i, a in enumerate(p.points):
        for b in p.points[i + 1:]:
            d1 = math.hypot(p.pos[a][0] - p.pos[b][0], p.pos[a][1] - p.pos[b][1]) / span1
            d2 = float(np.linalg.norm(P[a] - P[b])) / span2
            err += abs(d1 - d2)
            n += 1
    return err / max(n, 1)


def align_to_pos(coords: dict[str, tuple[float, float]],
                 pos: dict[str, tuple[float, float]],
                 segments: list[tuple[str, str]] | None = None) -> dict[str, tuple[float, float]]:
    """把解出的坐标摆到与读到的布局同向（只做旋转/镜像，不缩放）。

    求解时为了消掉雅可比的零空间，强制把 0 号点固定到原点、1 号点压到 x 轴上
    ——这一步会顺手把整个图形转到「AB 水平」，于是重绘图看着和原图方向对不上
    （g3 的等腰三角形就被转了 60°，A 从顶点跑到了右上角）。

    旋转角**只由主轴决定**：主轴是最长的那条线段（直径、斜边、底边），原图里
    多是横平竖直的。一开始用所有点做 Kabsch，结果某个点估偏了（g2 的 P 被估高
    了 20 个单位）就把整张图带着歪了 10°～15°，看着还是不像原图。主轴定死后
    方向必然吻合，其余点只用来决定「往哪一侧翻」。

    镜像也允许：图形满足的等式在镜像下同样成立，度量到的角度和长度一个不变
    ——它只影响画出来的样子，所以取和读到布局最接近的那一个即可。
    """
    names = [k for k in coords if k in pos]
    if len(names) < 2:
        return coords

    P = np.asarray([coords[k] for k in names], float)
    Q = np.asarray([[pos[k][0], -pos[k][1]] for k in names], float)  # 图像 y 向下 → 翻转
    P, Q = P - P.mean(axis=0), Q - Q.mean(axis=0)

    cands = [
        s for s in (segments or [])
        if s[0] in names and s[1] in names
    ]
    if not cands:
        return coords

    # 主轴按解出的真实长度挑：pos 是目测值，两条差不多长的边它经常分不清主次
    a, b = max(cands, key=lambda s: math.dist(coords[s[0]], coords[s[1]]))
    ia, ib = names.index(a), names.index(b)
    u = P[ib] - P[ia]
    v = Q[ib] - Q[ia]
    if np.linalg.norm(u) < 1e-12 or np.linalg.norm(v) < 1e-12:
        return coords

    theta = math.atan2(v[1], v[0]) - math.atan2(u[1], u[0])
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])

    # 关于主轴方向的镜像：轴不动，只把图形翻到另一侧
    ux, uy = u / np.linalg.norm(u)
    M = 2.0 * np.outer([ux, uy], [ux, uy]) - np.eye(2)

    best = R
    best_err = float(((P @ R.T - Q) ** 2).sum())
    err_m = float(((P @ (R @ M).T - Q) ** 2).sum())
    if err_m < best_err:
        best = R @ M

    out = {}
    for i, name in enumerate(names):
        w = best @ P[i]
        out[name] = (round(float(w[0]), 4), round(float(w[1]), 4))
    for name, v0 in coords.items():
        out.setdefault(name, v0)
    return out


def solve(p: Problem, tries: int = 8) -> GeoSolution:
    res = GeoSolution()

    # 先认结论：它决定这题要「一个数」还是「一句成立/不成立」，也决定
    # 哪些等式不能当条件用——待证的 CE=CF 常被模型抄进 given，照着建成
    # 约束就成了拿结论证明结论，怎么算都成立。
    claim = parse_claim(p)
    skip: set[frozenset] = set()
    if claim is not None and claim.kind == "len_equal":
        skip = {frozenset({claim.a, claim.b})}

    # 再用文字补齐图形结构（「四边形 ABCD 为正方形」这类整句声明）
    p, enriched = enrich_from_text(p, skip_equal=skip)

    if len(p.points) < 2:
        res.error = "没有读到足够的点，无法建模"
        return res

    groups = collinear_groups(p)
    arms = _vertex_arms(p, groups)
    lengths = _lengths(p)
    scale = max(lengths.values()) if lengths else 0.0
    if scale <= 0:
        scale = 1.0

    cons, used = build_constraints(p, scale, claim)
    if not cons:
        res.error = "没有从题干或图形里提取到任何可计算的几何条件"
        return res

    model = Model(p.points, cons)
    res.n_unknown = model.n()
    res.n_equations = 3 + sum(len(c.fn({k: np.zeros(2) for k in p.points}) or []) for c in cons)

    if claim is not None:
        goal = Goal("prove", claim.label(), note="证明")
    else:
        goal = parse_goal(p, groups, arms)

    res.steps.append(f"读入 {len(p.points)} 个点：{'、'.join(p.points)}")
    for n in dict.fromkeys(enriched):  # 去重：given 与题干正文会各算一遍
        res.steps.append(f"补全：{n}")
    if used:
        # 去重：题干和 given 会把同一条说两遍（「P 在 ABCD 内」实测出现两次）
        res.steps.append("条件：" + "；".join(dict.fromkeys(used)))
    if goal.kind == "unknown":
        res.error = "没看懂要求什么（题干里没找到明确的长度或角度目标）"
        res.steps.append(res.error)
        return res
    gname = {
        "length": f"求 {goal.name} 的长",
        "angle": f"求 ∠{goal.name} 的度数",
        "area": "求面积",
        "prove": f"求证 {goal.name}",
    }.get(goal.kind, "目标不明")
    res.steps.append("目标：" + gname)
    res.goal_kind = goal.kind
    res.goal_name = goal.name

    # 多初值：随机起点各解一次，看目标值是否一致
    rng = np.random.default_rng(20261110)
    values: list[float] = []
    oks: list[tuple[np.ndarray, float]] = []
    best: tuple[np.ndarray, float] | None = None

    # 第一个起点用原图的粗略布局：解有镜像对称时，这样能停在和原图朝向
    # 一致的那个解上，重绘图才不会左右颠倒。
    seeded: tuple[np.ndarray, float] | None = None
    x0_seed = seed_from_pos(p, scale)
    if x0_seed is not None:
        xs, rs = _gauss_newton(model, x0_seed)
        if rs < 1e-6:
            seeded = (xs, rs)
            oks.append(seeded)
            res.steps.append("按原图各点的大致位置作初值，重绘图与原图朝向一致")

    for t in range(tries):
        x0 = rng.uniform(-1.0, 1.0, model.n()) * scale
        if t == 0:
            x0 = np.zeros(model.n())  # 第一次给个规整起点
        x, rn = _gauss_newton(model, x0)
        if best is None or rn < best[1]:
            best = (x, rn)
        if rn < 1e-6:
            if not any(float(np.linalg.norm(x - y)) < 1e-6 for y, _ in oks):
                oks.append((x, rn))
            if goal.kind != "prove":
                v = _measure(model, x, goal, p, arms)
                if v is not None:
                    values.append(v)

    if oks:
        # 收敛到的解可能不止一个：它们都满足条件，但形状可能差很远。挑最像
        # 原图的那个——seeded 排在 oks 最前面，分数相同时自然胜出。
        best = min(oks, key=lambda it: shape_score(p, model.unpack(it[0]), scale))

    if best is None:
        res.error = "数值求解没有返回结果"
        return res
    res.residual = best[1]

    if best[1] > 1e-4:
        res.error = (
            "条件互相矛盾，找不到满足所有条件的图形"
            f"（残差 {best[1]:.3g}）。多半是某条条件读错了，请核对识别结果。"
        )
        res.steps.append(res.error)
        return res

    # 坐标先存下来。以前这里要求「目标算出来了才存」，于是条件不足的题
    # 一张图都画不出来，重绘只能退回环形布局——8 个点摆在圆周上，
    # A 在顶上、D 跑到底下，看着像彻底认错了题。其实图上点线是对的，
    # 只是答案定不下来，这两件事得分开。
    P_best = model.unpack(best[0])
    res.coords = {k: (round(float(v[0]), 4), round(float(v[1]), 4)) for k, v in P_best.items()}
    if p.pos:
        # 解出来了，但朝向可能是被规范固定转过的：转回原图的朝向再画
        res.coords = align_to_pos(res.coords, p.pos, p.segments)

    # --- 证明题：验结论，不求数 ---
    if goal.kind == "prove" and claim is not None:
        res.claim_name = claim.label()
        if claim.kind == "len_equal":
            res.claim_edges = (claim.a, claim.b)
        elif claim.kind == "angle_equal":
            res.claim_angles = (claim.a, claim.b)

        pool = oks or [best]
        results = [verify_claim(model, x, claim, p) for x, _ in pool]
        holds = [r for r in results if r[0]]
        res.steps.append(f"结论：{claim.label()}")
        for ok_i, msg in results[:4]:
            res.steps.append(("成立" if ok_i else "不成立") + "：" + msg)

        if len(holds) == len(results):
            res.ok = True
            # 图形本身唯一不唯一是另一件事：结论在每一组解上都成立，
            # 说明它被条件推出来了；但若解出来的 CE 长度各不相同，
            # 说明条件只约束住了「相等」这一件事，图形仍有自由度。
            lens = [d for d in (_dist(model.unpack(x), claim.a) for x, _ in pool) if d]
            if claim.kind == "len_equal" and len(lens) > 1:
                spread = (max(lens) - min(lens)) / max(max(lens), 1e-12)
                if spread > 2e-3:
                    res.unique = False
            if res.unique:
                res.answer = f"结论成立：{claim.label()}"
                res.steps.append(
                    f"{len(pool)} 组不同起点解出的图形都对得上，{claim.label()} 成立"
                )
            else:
                res.answer = f"结论成立：{claim.label()}（图形本身不唯一）"
                res.steps.append(
                    "图形还留着自由度（长度可变），但每一组解上结论都成立；"
                    "若要严格的演绎证明，还需要补上缺失的条件。"
                )
        else:
            bad = next(msg for ok_i, msg in results if not ok_i)
            res.ok = False
            res.error = (
                f"结论与图形条件下不符：{claim.label()} 在解出的图形上不成立（{bad}）。"
                "多半是某条已知条件读错或漏读了，请核对识别结果。"
            )
            res.steps.append(res.error)
        return res

    if not values:
        res.error = "图形能建出来，但目标量算不出来"
        res.steps.append(res.error)
        return res

    values.sort()
    vmin, vmax = values[0], values[-1]
    tol = 1e-3 if goal.kind == "length" else 1e-2
    if vmax - vmin > tol * max(1.0, abs(vmin)):
        res.unique = False
        res.error = (
            f"条件不足，答案不唯一：不同起点解出的值落在 {vmin:.4g} ~ {vmax:.4g} 之间。"
            "通常是少读了一个长度或角度。"
        )
        res.steps.append(res.error)
        return res

    v = float(np.median(values))
    res.value = round(v, 4)
    res.ok = True
    if goal.kind == "length":
        res.answer = f"{goal.name} = {res.value:g}"
        res.steps.append(
            f"坐标法解出图形（残差 {res.residual:.1e}），{goal.name} = {res.value:g}"
        )
    elif goal.kind == "angle":
        res.answer = f"∠{goal.name} = {res.value:g}°"
        res.unit = "°"
        res.steps.append(
            f"坐标法解出图形（残差 {res.residual:.1e}），∠{goal.name} = {res.value:g}°"
        )
    else:
        res.answer = f"{res.value:g}"
        res.steps.append(f"解出 {res.value:g}")

    # 记下目标角的两条边，重绘时要照这个画弧线并标注度数
    if goal.kind == "angle":
        if len(goal.name) == 3:
            res.goal_arms = (goal.name[0], goal.name[2])
        else:
            res.goal_arms = angle_arms(p, goal.name, P_best)
    res.steps.append(f"用 {len(values)} 个不同起点各解一次，目标值一致（±{tol:g}），说明图形被唯一确定")
    return res

