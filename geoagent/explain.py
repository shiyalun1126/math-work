"""把求解结果讲成学生能看懂的过程。

原来的 `steps` 是系统日志——「读入 6 个点」「补全：…」「残差 2e-14」。
那些话是说给开发者听的：它记录了系统做了什么，却没说这道题怎么做。
学生拿着它学不到任何东西，甚至看不出 CE 和 CF 是谁。

这里改成**坐标法**的完整过程：

    建系 → 写出各点坐标 → 把题干的每条条件代回坐标逐条核对 → 算目标量 → 结论

为什么是坐标法而不是「∵∴」的演绎证明：坐标法是中学就教的正规解法，
而且它的每一步都能当场验算——数字对不对，看结果就知道。
编一段漂亮的演绎证明当然更「像老师」，但那是模型写出来的，对不对没人查。
宁可给一个能验算的解法，也不给一段读起来很顺、但可能错在第三步的推理。

所有数值都来自解出来的坐标，不是另算的。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .schema import Problem
from . import reason
from .solve import (GeoSolution, _angles_given, _angle_pairs_given,
                    _inside_given, _lengths, collinear_groups, parse_claim)


@dataclass
class GeoExplain:
    text: str = ""
    ok: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "text": self.text, "note": self.note}


# ---------------------------------------------------------------- 数值格式

def _f(x: float, nd: int = 3) -> str:
    """3.0 → 3，0.7321 → 0.732，-0.0 → 0。"""
    if abs(x) < 5e-4:
        return "0"
    if abs(x - round(x)) < 5e-4:
        return f"{int(round(x))}"
    s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def _pt(xy: tuple[float, float]) -> str:
    return f"({_f(xy[0])}, {_f(xy[1])})"


def _dist(X: dict, a: str, b: str) -> float:
    (x1, y1), (x2, y2) = X[a], X[b]
    return math.hypot(x1 - x2, y1 - y2)


def _deg(X: dict, vertex: str, a: str, b: str) -> float:
    """∠a-vertex-b 的度数（0~180）。"""
    vx, vy = X[vertex]
    ux, uy = X[a][0] - vx, X[a][1] - vy
    wx, wy = X[b][0] - vx, X[b][1] - vy
    nu, nw = math.hypot(ux, uy), math.hypot(wx, wy)
    if nu < 1e-9 or nw < 1e-9:
        return 0.0
    c = max(-1.0, min(1.0, (ux * wx + uy * wy) / (nu * nw)))
    return math.degrees(math.acos(c))


# ---------------------------------------------------------------- 建系

def _pick_base(p: Problem, coords: dict, goal_seg: frozenset | None) -> tuple[str, str]:
    """挑一条边当 x 轴。

    三条规矩，按优先级：
      1. **目标那条边绝不能当基准**——把待求的 AB 摆在坐标轴上再「求出 AB」，
         是拿答案推答案，写出来学生也看不懂为什么要设 AB = 5。
      2. 长度已知的边优先：用已知量当尺子，单位才有出处。
      3. 和第一个点相连的边优先，这样「以 A 为原点」最自然。
    """
    known = _lengths(p)
    cands = [(a, b) for a, b in p.segments if a in coords and b in coords]
    if not cands:
        return "", ""

    first = p.points[0] if p.points else ""

    def rank(item: tuple[str, str]) -> tuple[int, int]:
        a, b = item
        s = 0
        if goal_seg and frozenset({a, b}) == goal_seg:
            s -= 100
        if f"{a}{b}" in known or f"{b}{a}" in known:
            s += 10
        if first in (a, b):
            s += 3
        return -s

    a, b = sorted(cands, key=rank)[0]
    # 原点取更靠前的点，读起来像「以 A 为原点」
    order = {n: i for i, n in enumerate(p.points)}
    return (a, b) if order.get(a, 99) <= order.get(b, 99) else (b, a)


def _frame(p: Problem, coords: dict, goal_seg: frozenset | None = None):
    """把解出的坐标搬到一个「看得懂」的坐标系里。

    求解器给的坐标原点和朝向是随机的（g1 的 A 在 (-2.4, 4.2)），
    直接抄给学生没法看。这里只做**平移 + 旋转**，绝不缩放——
    长度是题目给的量，动了尺度，答案就不是答案了。

    原点取第一个点，x 轴取一条边：于是「以 A 为原点，AB 为 x 轴」
    这句话永远成立，也永远和图对得上。
    """
    names = [n for n in p.points if n in coords]
    if not names:
        return {}, "", "", None
    origin, base = _pick_base(p, coords, goal_seg)
    if not origin:
        origin, base = names[0], ""

    z0 = complex(coords[origin][0], coords[origin][1])
    w = 1 + 0j
    if base:
        dz = complex(coords[base][0], coords[base][1]) - z0
        if abs(dz) > 1e-9:
            w = dz / abs(dz)

    X = {}
    for n in names:
        z = (complex(coords[n][0], coords[n][1]) - z0) / w
        X[n] = (0.0 if abs(z.real) < 5e-4 else round(z.real, 3),
                0.0 if abs(z.imag) < 5e-4 else round(z.imag, 3))

    # 朝向还有最后一个自由度：y 轴朝上还是朝下，解出来是哪个都有可能。
    # 把「大多数人在地平线以上」那一版给学生——不是数学问题，是阅读习惯问题。
    up = sum(1 for v in X.values() if v[1] > 1e-6)
    down = sum(1 for v in X.values() if v[1] < -1e-6)
    if down > up:
        X = {k: (v[0], -v[1]) for k, v in X.items()}

    known = _lengths(p)
    base_len = None
    if base:
        base_len = known.get(f"{origin}{base}") or known.get(f"{base}{origin}")
    return X, origin, base, base_len


# ---------------------------------------------------------------- 条件核对

def _norm_seg(name: str) -> str:
    return "".join(sorted(name[:2]))


def _inside_poly(X: dict, x: str, verts: list[str]) -> bool:
    """点是否在多边形里：与每条边的对顶点同侧。

    用同侧判据而不是射线法，是因为这里的多边形按题目的书写顺序给出顶点，
    顺序一定是对的；同侧判据对凸多边形既准又不用处理射线穿过顶点的退化情况。
    """
    def cross(a: str, b: str, c: str) -> float:
        ux, uy = X[b][0] - X[a][0], X[b][1] - X[a][1]
        vx, vy = X[c][0] - X[a][0], X[c][1] - X[a][1]
        return ux * vy - uy * vx

    n = len(verts)
    for i in range(n):
        a, b, c = verts[i], verts[(i + 1) % n], verts[(i + 2) % n]
        s1, s2 = cross(a, b, c), cross(a, b, x)
        if s1 * s2 < -1e-12:
            return False
    return True


def _rows(p: Problem, X: dict) -> list[tuple[str, str]]:
    """题干的每条条件，在坐标下实测出什么。

    同一件事会从好几个地方冒出来（图上画的、题面写的、推出来的），
    所以先按「这件事本身」归类去重，再决定怎么写给人看。
    """
    right: dict[str, tuple[str, str, float]] = {}      # 顶点 → (a, b, 度数)
    equal: dict[frozenset, tuple[str, str, float, float]] = {}
    para: dict[frozenset, tuple[str, str, float]] = {}
    mids: list[tuple[str, str]] = []
    lens: list[tuple[str, str]] = []
    angs: list[tuple[str, str]] = []
    cols: list[tuple[str, str]] = []
    circs: list[tuple[str, str]] = []
    insides: list[tuple[str, str]] = []

    for m in p.marks:
        if m.type == "right_angle" and m.at in X:
            ofs = [x for x in m.of if len(x) == 2 and m.at in x]
            if len(ofs) < 2:
                continue
            a, b = ofs[0].replace(m.at, ""), ofs[1].replace(m.at, "")
            if a in X and b in X and m.at not in right:
                right[m.at] = (a, b, _deg(X, m.at, a, b))

        elif m.type == "equal_length" and len(m.of) >= 2:
            a, b = m.of[0], m.of[1]
            if len(a) == 2 and len(b) == 2 and set(a) <= set(X) and set(b) <= set(X):
                key = frozenset({_norm_seg(a), _norm_seg(b)})
                if key not in equal:
                    equal[key] = (a, b, _dist(X, a[0], a[1]), _dist(X, b[0], b[1]))

        elif m.type == "parallel" and len(m.of) >= 2:
            a, b = m.of[0], m.of[1]
            if len(a) == 2 and len(b) == 2 and set(a) <= set(X) and set(b) <= set(X):
                key = frozenset({_norm_seg(a), _norm_seg(b)})
                if key not in para:
                    para[key] = (a, b, _line_angle(X, a, b))

        elif m.type == "midpoint" and m.at in X and m.of:
            seg = m.of[0] if len(m.of[0]) == 2 else ""
            if seg and set(seg) <= set(X):
                d1, d2 = _dist(X, seg[0], m.at), _dist(X, m.at, seg[1])
                row = (f"{m.at} 是 {seg} 的中点",
                       f"{seg[0]}{m.at} = {_f(d1)}，{m.at}{seg[1]} = {_f(d2)}")
                if row not in mids:
                    mids.append(row)

    for name, val in _lengths(p).items():
        if len(name) == 2 and set(name) <= set(X):
            row = (f"{name} = {_f(val)}", f"实测 {_f(_dist(X, name[0], name[1]))}")
            if row not in lens:
                lens.append(row)

    for name, val in _angles_given(p):
        if len(name) == 3 and set(name) <= set(X):
            real = _deg(X, name[1], name[0], name[2])
            row = (f"∠{name} = {_f(val)}°", f"实测 {real:.1f}°")
            if row not in angs:
                angs.append(row)

    # 「∠PBA = ∠PDA」这类不给度数、只给相等的条件。不写出来，条件表上就只剩
    # 平行那两条，看着像是它们把这道题定死的——而真正定住 P 的是这一条。
    _c = parse_claim(p)
    _goal_pair = (
        frozenset({_c.a, _c.b})
        if _c is not None and _c.kind == "angle_equal"
        else frozenset()
    )
    for n1, n2 in _angle_pairs_given(p):
        if frozenset({n1, n2}) == _goal_pair:
            continue  # 这是要证的，不是题给的条件
        if len(n1) == 3 and len(n2) == 3 and set(n1) <= set(X) and set(n2) <= set(X):
            d1 = _deg(X, n1[1], n1[0], n1[2])
            d2 = _deg(X, n2[1], n2[0], n2[2])
            row = (f"∠{n1} = ∠{n2}", f"{d1:.1f}° = {d2:.1f}°")
            if row not in angs:
                angs.append(row)

    for x, poly in _inside_given(p):
        verts = [v for v in poly if v in X]
        if len(verts) < 3 or x not in X or x in verts:
            continue
        here = _inside_poly(X, x, verts)
        row = (f"{x} 在 {poly} 内部", "在内部" if here else "**跑到形外了**")
        if row not in insides:
            insides.append(row)

    # 共线组先合并：直径上 A、B、O、P 其实是同一条直线，分成两组写会显得啰嗦
    merged: list[set[str]] = []
    for g in (set(x for x in gg if x in X) for gg in collinear_groups(p)):
        for m_ in merged:
            if len(g & m_) >= 2:
                m_ |= g
                break
        else:
            merged.append(set(g))
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if len(merged[i] & merged[j]) >= 2:
                    merged[i] |= merged[j]
                    merged.pop(j)
                    changed = True
                    break
            if changed:
                break
    for g in merged:
        xs = sorted(g)
        if len(xs) >= 3:
            cols.append((f"{'、'.join(xs)} 共线", _collinear_note(X, xs)))

    for c in p.circles:
        if c.center in X and c.through:
            pts = [t for t in c.through if t in X]
            if pts:
                rs = [_dist(X, c.center, t) for t in pts]
                circs.append((f"⊙{c.center} 过 {'、'.join(pts)}", f"半径 {_f(sum(rs) / len(rs))}"))

    # 四条边相等 + 四个直角 = 正方形。七行压成一行，否则表格比解题过程还长。
    quad = set(right)
    quad_sides: dict[str, float] = {}
    for a, b, la, lb in equal.values():
        for s, ln in ((a, la), (b, lb)):
            if set(s[:2]) <= quad:
                quad_sides.setdefault(_norm_seg(s), ln)
    square_row = None
    square_edges: set[str] = set()
    if len(quad) == 4:
        # 四个直角点之间的等长边可能不止四条（正方形的对角线也等长），
        # 所以得真找一圈首尾相接的四条边，不能光数条数
        import itertools

        for perm in itertools.permutations(sorted(quad)):
            ring = [_norm_seg("".join(pr)) for pr in zip(perm, perm[1:] + perm[:1])]
            if len(set(ring)) < 4 or not all(e in quad_sides for e in ring):
                continue
            ring_len = [quad_sides[e] for e in ring]
            ok_ang = all(abs(d - 90) < 0.5 for _a, _b, d in right.values())
            if max(ring_len) - min(ring_len) < 1e-3 and ok_ang:
                square_row = (
                    f"四边形 {'、'.join(sorted(quad))} 是正方形",
                    f"边长 {_f(ring_len[0])}，四个角都是 90°",
                )
                square_edges = set(ring)
            break

    out: list[tuple[str, str]] = []
    if square_row:
        out.append(square_row)
    else:
        for at, (a, b, d) in right.items():
            out.append((f"∠{a}{at}{b} = 90°", f"{d:.1f}°"))

    # 正方形那四条边已经并进上面一行了，剩下的等长条件（比如 AE = AC）还得单独写
    for a, b, la, lb in equal.values():
        if square_row and _norm_seg(a) in square_edges and _norm_seg(b) in square_edges:
            continue
        out.append((f"{a} = {b}", f"{_f(la)} = {_f(lb)}"))
    for a, b, d in para.values():
        out.append((f"{a} ∥ {b}", f"夹角 {_f(min(d, 180 - d), 1)}°"))
    out += mids + lens + angs + cols + circs + insides
    return out


def _line_angle(X: dict, a: str, b: str) -> float:
    """两条线段的夹角（0~180）。"""
    def ang(s: str) -> float:
        (x1, y1), (x2, y2) = X[s[0]], X[s[1]]
        return math.degrees(math.atan2(y2 - y1, x2 - x1))

    d = abs(ang(a) - ang(b)) % 360
    return d if d <= 180 else 360 - d


def _collinear_note(X: dict, g: list[str]) -> str:
    a, b, c = (X[name] for name in g[:3])
    s = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    base = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
    return f"偏离直线 {_f(abs(s) / base, 2)}（0 表示共线）"


# ---------------------------------------------------------------- 组装

def _goal_body(p: Problem, sol: GeoSolution, X: dict) -> list[str]:
    """第四步的内容：把题目问的量按公式算一遍。"""
    out: list[str] = []
    k = sol.goal_kind

    if k == "prove" and sol.claim_edges:
        a, b = sol.claim_edges
        if not (set(a) <= set(X) and set(b) <= set(X)):
            return out
        for name in (a, b):
            out += _len_lines(X, name)
        la, lb = _dist(X, a[0], a[1]), _dist(X, b[0], b[1])
        same = abs(la - lb) <= max(1e-3, 0.008 * max(la, lb))
        if same:
            out.append(f"\n{_f(la)} 和 {_f(lb)} 是同一个数，所以 **{sol.claim_name} 成立**。")
        else:
            out.append(f"\n两者不相等（差 {_f(abs(la - lb))}），**结论不成立**。")

    elif k == "prove" and sol.claim_angles:
        # 角相等和边相等是两套算法：边用两点距离，角用余弦定理。以前只认
        # claim_edges，于是「求证 ∠PAB = ∠PCB」这类题算到最后什么都不写，
        # 界面上的「结论」下面是空的。
        n1, n2 = sol.claim_angles
        if not (len(n1) == 3 and len(n2) == 3):
            return out
        if not (set(n1) <= set(X) and set(n2) <= set(X)):
            return out
        for n_ in (n1, n2):
            out += _angle_lines(X, n_[1], n_[0], n_[2], "")
        d1 = _deg(X, n1[1], n1[0], n1[2])
        d2 = _deg(X, n2[1], n2[0], n2[2])
        if abs(d1 - d2) <= 0.05:
            out.append(f"\n两个角都是 {d1:.1f}°，所以 **{sol.claim_name} 成立**。")
        else:
            out.append(
                f"\n{d1:.1f}° ≠ {d2:.1f}°，**结论不成立**（差 {abs(d1 - d2):.1f}°）。"
            )

    elif k == "length" and sol.goal_name and len(sol.goal_name) == 2:
        if not set(sol.goal_name) <= set(X):
            return out
        out += _len_lines(X, sol.goal_name)

    elif k == "angle" and sol.goal_name:
        name, arms = sol.goal_name, sol.goal_arms
        if len(name) == 3:
            vertex, a, b = name[1], name[0], name[2]
        elif arms:
            vertex, (a, b) = name, arms
        else:
            vertex = ""
        if vertex and all(t in X for t in (vertex, a, b)):
            full = f"{a}{vertex}{b}"
            # 题目写「∠P」，我们要把两条边说清楚；两个名字都给学生，免得对不上
            aka = f"（题目写作 ∠{name}）" if name != full else ""
            out += _angle_lines(X, vertex, a, b, aka)

    return out


def _len_lines(X: dict, name: str) -> list[str]:
    a, b = name[0], name[1]
    (x1, y1), (x2, y2) = X[a], X[b]

    def term(p: float, q: float) -> str:
        # 减数是负数时要加括号，否则 (3.6 - -4.8) 会被读成减减
        return f"({_f(p)} - {_f(q)})" if q >= 0 else f"({_f(p)} + {_f(-q)})"

    d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
    d = math.sqrt(d2)
    out = [f"\n**{a}{b}** 的长："]
    out.append(
        f"$${a}{b}=\\sqrt{{{term(x1, x2)}^2+{term(y1, y2)}^2}}"
        f"=\\sqrt{{{_f(d2)}}}\\approx {_f(d)}$$"
    )
    return out


def _angle_lines(X: dict, vertex: str, a: str, b: str, aka: str = "") -> list[str]:
    la, lb, ab = _dist(X, a, vertex), _dist(X, b, vertex), _dist(X, a, b)
    deg = _deg(X, vertex, a, b)
    cos = math.cos(math.radians(deg))
    out = [f"\n**∠{a}{vertex}{b}**{aka}的度数："]
    out.append(f"$${a}{vertex}={_f(la)},\\ {b}{vertex}={_f(lb)},\\ {a}{b}={_f(ab)}$$")
    out.append(
        f"由余弦定理　$\\cos\\angle {a}{vertex}{b}=\\dfrac{{{a}{vertex}^2+{b}{vertex}^2-{a}{b}^2}}"
        f"{{2\\cdot {a}{vertex}\\cdot {b}{vertex}}}={_f(cos)}$，"
        f"得 $\\angle {a}{vertex}{b}\\approx {deg:.1f}^\\circ$。"
    )
    return out


def _condition_lines(p: Problem) -> list[str]:
    """把读到的条件摊开给人看，不写成系统术语。"""
    out = [g.strip() for g in p.given if g and g.strip()]
    for m in p.marks:
        if m.type == "right_angle" and m.at:
            out.append(f"{m.at} 处标着直角")
        elif m.type == "equal_length" and len(m.of) >= 2:
            out.append(f"{m.of[0]} 和 {m.of[1]} 一样长（图上标记）")
        elif m.type == "parallel" and len(m.of) >= 2:
            out.append(f"{m.of[0]} ∥ {m.of[1]}（图上标记）")
        elif m.type == "midpoint" and m.at and m.of:
            out.append(f"{m.at} 是 {m.of[0]} 的中点（图上标记）")
    return list(dict.fromkeys(out))


def diagnose(p: Problem, sol: GeoSolution) -> str:
    """解不出来时给的话。

    以前这里直接把 `sol.steps` 倒出来：读入 5 个点、补全、残差 2e-14、不成立、
    不成立……那是给自己排查用的日志，学生看了只知道"它没做出来"，不知道
    该怎么办。失败的时候恰恰最需要说清楚：我读到了什么、卡在哪、你该改什么。
    """
    lines = ["### 这题没做出来", ""]
    if sol.error:
        lines.append(f"**卡在哪**：{sol.error}")
    else:
        lines.append("**卡在哪**：按读到的条件推不出结论。")
    lines.append("")

    conds = _condition_lines(p)
    if conds:
        lines += ["**我读到的条件**", ""]
        lines += [f"- {c}" for c in conds]
        lines.append("")

    if p.points:
        segs = "、".join(f"{a}{b}" for a, b in p.segments) or "（没读到线段）"
        lines += [
            f"**我读到的图**：{len(p.points)} 个点 {'、'.join(p.points)}，"
            f"{len(p.segments)} 条线 {segs}。",
            "",
        ]

    lines += [
        "**先核对这三件事**",
        "",
        "1. **有没有一条条件没读进来。** 最容易丢的是「∠PBA = ∠PDA」这种"
        "只给相等、不给度数的条件，和「P 在图形内部」这种不带数字的位置条件——"
        "它们在转写时最容易被当成废话跳过去，但缺了它们，图就定不住。",
        "2. **图上有没有少画线。** 辅助线常常被拍得很淡，或者和另一条线贴在一起。"
        "把原图和下面的重绘图对照着看，少一条就能看出来。",
        "3. **题目或图形本身有没有抄错。** 结论对不上时，这也是一个可能。",
        "",
        "> 改法：把文本框里的题干或结构改对，直接重算，不用重新拍照。",
    ]
    return "\n".join(lines)


def write(p: Problem, sol: GeoSolution, force_coords: bool = False) -> GeoExplain:
    """生成学生视角的解题过程。

    优先走**几何推理**（`reason.py`）：平行、角等、等边对等角，一步一条定理。
    推不出来才退回坐标法——那是兜底，不是首选：它必须先把图形量出来，
    题目没给长度时只能设某条边为 1，学生看到的就全是"凭什么设成 1"。

    `force_coords=True` 强制走坐标法，只给测试用。
    """
    if sol.ok and not force_coords:
        pr = reason.prove(p, sol)
        if pr.ok and pr.steps:
            return GeoExplain(text=reason.render(pr, sol), ok=True)

    if not sol.ok or not sol.coords:
        # 解不出来也要说话，而且要说人话。返回 ok=False 是对的——这不是解题
        # 过程，是诊断；界面据此知道不该把「结论」当成结果展示。
        return GeoExplain(
            text=diagnose(p, sol), ok=False, note="这题没做出来，下面是卡在哪"
        )

    goal_seg = (
        frozenset(sol.goal_name)
        if sol.goal_kind == "length" and sol.goal_name and len(sol.goal_name) == 2
        else None
    )

    X, origin, base, base_len = _frame(p, sol.coords, goal_seg)
    if not X:
        return GeoExplain(note="坐标缺失")

    names = list(X)
    # 证明题的结论可能是「边相等」也可能是「角相等」，同一句开场白套上去
    # 会说错话（要证 ∠PAB = ∠PCB 却说「要证两条线段相等」）。
    what = "两个角相等" if "∠" in (sol.claim_name or "") else "两条线段相等"
    opener = {
        "prove": f"要证{what}，最稳的办法是**把每个点都算出来**——用坐标法。"
                 "不用想辅助线，条件够不够、算得对不对，都能当场验算。",
        "length": "这道题用**坐标法**：把题干的条件都写成对坐标的要求，解出每个点，"
                  "再用两点间距离公式算长度。",
        "angle": "这道题用**坐标法**：把题干的条件都写成对坐标的要求，解出每个点，"
                 "再用余弦定理算角度。",
    }.get(sol.goal_kind, "这道题用**坐标法**：把题干的条件写成对坐标的要求，解出每个点，再算题目问的量。")

    # 小标题按顺序编号：有几节算几节，不写死「三」「四」（条件表可能为空）
    CN = "一二三四五六"
    n = [0]

    def sec(title: str) -> str:
        s = f"#### {CN[n[0]]}、{title}" if n[0] < len(CN) else f"#### {title}"
        n[0] += 1
        return s

    lines = [sec("建系"), ""]
    if base and base_len is not None:
        lines.append(
            f"以 **{origin} 为原点**、**{origin}{base} 所在直线为 x 轴**建立坐标系。"
            f"{origin}{base} = {_f(base_len)} 是题目给的，所以 "
            f"{origin}{_pt(X[origin])}、{base}{_pt(X[base])}。"
        )
    elif base:
        lb = _dist(X, origin, base)
        unit = f"，设 {origin}{base} = 1 作为单位长度" if abs(lb - 1) < 1e-3 else ""
        lines.append(
            f"以 **{origin} 为原点**、**{origin}{base} 所在直线为 x 轴**建立坐标系{unit}。"
            f"于是 {origin}{_pt(X[origin])}、{base}{_pt(X[base])}。"
        )
        if unit:
            # 学生看到「设 AB = 1」第一反应是「凭什么」。不解释，后面每一步
            # 都会被打个问号——而这恰恰是坐标法最容易被质疑的地方。
            lines += [
                "",
                "题目没给任何长度，给的只有角和平行、相等这类关系。这些关系在"
                "整个图形一起放大或缩小时都照样成立，所以设 "
                f"{origin}{base} = 1 不会改变答案，只是让数字好算。"
                "如果题目给了长度，就该用题目的数，不能自己设。",
            ]
    else:
        lines.append(f"以 **{origin} 为原点**建立坐标系，{origin}{_pt(X[origin])}。")

    lines += ["", sec("解出各点坐标"), ""]
    lines.append(
        "每个点有 x、y 两个坐标，也就是两个未知数。题干里的每一条条件"
        "——平行、等长、直角、点在线上——都能写成关于这些坐标的一个方程，"
        "把它们联立起来解，就得到："
    )
    lines.append("")
    lines.append("　　" + "　".join(f"{n_}{_pt(X[n_])}" for n_ in names))

    rows = _rows(p, X)
    if rows:
        lines += ["", sec("逐条核对题干条件"), ""]
        lines.append("| 题目给的条件 | 代回坐标后的实测 |")
        lines.append("|---|---|")
        for cond, real in rows:
            lines.append(f"| {cond} | {real} |")

    body = _goal_body(p, sol, X)
    if body:
        goal_what = "两个角" if "∠" in (sol.claim_name or "") else "两条线段"
        title = {"prove": f"比较题目要证的{goal_what}", "length": "算题目问的长度",
                 "angle": "算题目问的角度"}.get(sol.goal_kind, "算题目问的量")
        lines += ["", sec(title)] + body

    if sol.answer:
        lines += ["", "#### 结论", "", f"**{sol.answer}**"]

    return GeoExplain(text="\n".join(["### 解题过程", "", opener, ""] + lines), ok=True)


def outline(sol: GeoSolution) -> str:
    """一句话说明这题是怎么做的，给摘要用。"""
    if not sol.ok:
        return ""
    return {
        "prove": "几何推理：先判全等/相似或找角的关系，再落到对应边、对应角。",
        "length": "几何推理：逐条条件套定理（勾股、相似、斜边中线…）算出长度。",
        "angle": "几何推理：逐条条件套定理（平行、等腰、全等、内角和…）算出角度。",
    }.get(sol.goal_kind, "")
