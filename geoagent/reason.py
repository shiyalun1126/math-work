"""定理驱动的几何推理。

为什么要有这一层。坐标法能把答案算对，但学生看到的是一串点坐标——
看不出哪两条线平行、哪个角相等、凭什么相等，也就学不到东西。
更糟的是，坐标法必须先把图形量出来：题目没给长度时只能设某条边为 1，
于是每个坐标都成了"凭空来的数字"。用户的话是：

    「几何题尽量还是用几何思维，谁平行，哪个角相等，用推理的形式」

所以这里换成前向链推理：从题面条件和图形结构出发，套定理，一步步推。
每条结论都挂着它的定理名和前提，输出长这样：

    ∵ DE ∥ AC，CD 是截线
    ∴ ∠CDE = ∠ACD（两直线平行，内错角相等）

坐标在这里只剩一个用途：**验算**。每条规则推出的结论，都在解出的图形上
量一遍，对不上就整条丢弃——推理链漂亮但错一步，过不了这道关。

长度一律符号化。题目没给数值时，把第一条边定为 a，所有长度写成 a 的倍数
（√2a、a/2），不会出现「设 AD = 1 所以 E 在 (0.366, 1.366)」这种东西。
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field

from .schema import Problem
from .solve import (GeoSolution, _angles_given, _lengths, collinear_groups,
                    ray_targets, tangent_pairs)

MAX_FACTS = 8000        # 事实上限，防止规则互相点火烧穿
TOL_D = 0.6             # 角度验算容差（度）
TOL_L = 3e-3            # 长度验算相对容差


# ---------------------------------------------------------------- 基本对象


def _seg(a: str, b: str) -> str:
    """线段名，统一成字母序：CE 与 EC 是同一条。"""
    return "".join(sorted((a, b)))


def _ang(v: str, a: str, b: str) -> tuple[str, str, str]:
    """角名，统一成 (顶点, 小, 大)：∠ACD 与 ∠DCA 是同一个角。"""
    x, y = sorted((a, b))
    return (v, x, y)


def _ang_name(t: tuple[str, str, str]) -> str:
    v, a, b = t
    return f"{a}{v}{b}"


def _lname(line: tuple[str, str]) -> str:
    return f"{line[0]}{line[1]}"


def _norm(kind: str, args: tuple) -> tuple:
    """等号两边、平行的两条线、相等的两个角，都不分先后。"""
    if kind in ("eqseg", "eqang", "para"):
        return tuple(sorted(args, key=str))
    if kind == "prod":
        # 圆幂：PA·PB = PC·PD。两段一组，组内与组间都不分先后。
        # 段名也顺手归一次序：AC 与 CA 是同一条边，别存成两条事实。
        a, b = (tuple(sorted("".join(sorted(s)) for s in x)) for x in args)
        return (a, b) if a <= b else (b, a)
    return args


# ---------------------------------------------------------------- 三角形拆件


def _tri_sides(t: tuple[str, str, str]) -> list[str]:
    """三条边，第 k 条是 v_k v_(k+1)，正对第 k 个顶点。"""
    return [_seg(t[0], t[1]), _seg(t[1], t[2]), _seg(t[2], t[0])]


def _tri_angs(t: tuple[str, str, str]) -> list[tuple[str, str, str]]:
    """三个内角，第 k 个的顶点是 v_k。"""
    return [_ang(t[0], t[1], t[2]), _ang(t[1], t[2], t[0]), _ang(t[2], t[0], t[1])]


def _side_between(i: int, j: int) -> int:
    """第 i、j 两个内角夹的那条边（返回边序号）。"""
    if (i + 1) % 3 == j:
        return i
    if (j + 1) % 3 == i:
        return j
    return -1


def _ang_between(i: int, j: int) -> int:
    """第 i、j 两条边夹的那个角（返回角序号）。"""
    if (i + 1) % 3 == j:
        return j
    if (j + 1) % 3 == i:
        return i
    return -1


def _tri_rots(t: tuple[str, str, str]):
    a, b, c = t
    return ((a, b, c), (b, c, a), (c, a, b))


def _canon_forms(t1: tuple, t2: tuple):
    """一条关系能写成的所有等价形式。

    写的顺序就是对应关系，所以不能像线段那样简单排序。能变的只有三件事：
    整体轮换、整体翻转、两个三角形对调——都不改变「谁和谁对应」。
    """
    for a, b in ((t1, t2), (t2, t1)):
        for r1, r2 in zip(_tri_rots(tuple(a)), _tri_rots(tuple(b))):
            for rev in (False, True):
                x = tuple(r1[::-1]) if rev else tuple(r1)
                y = tuple(r2[::-1]) if rev else tuple(r2)
                yield x, y


def _canon_pair(t1: tuple, t2: tuple) -> tuple[tuple, tuple]:
    """全等关系的规范写法。"""
    return min(_canon_forms(t1, t2), key=lambda p: (p[0], p[1]))


def _canon_sim(t1: tuple, t2: tuple, k: float | None) -> tuple:
    """相似关系的规范写法。对调两个三角形时，相似比要取倒数。"""
    best: tuple | None = None
    for a, b, kk in ((t1, t2, k), (t2, t1, None if k is None else 1.0 / k)):
        for x, y in _canon_forms(a, b):
            if best is None or (x, y) < (best[0], best[1]):
                best = (x, y, kk)
    x, y, kk = best
    return x, y, (None if kk is None else round(kk, 6))



@dataclass
class Fact:
    kind: str
    args: tuple
    pre: list[str] = field(default_factory=list)
    rule: str = ""
    deps: tuple = ()

    @property
    def key(self) -> tuple:
        return (self.kind, self.args)


class KB:
    def __init__(self) -> None:
        self.f: dict[tuple, Fact] = {}

    def add(self, kind: str, args: tuple, pre: list[str], rule: str = "",
            deps: tuple = ()) -> bool:
        args = _norm(kind, args)
        k = (kind, args)
        if k in self.f or len(self.f) >= MAX_FACTS:
            return False
        self.f[k] = Fact(kind, args, pre, rule, deps)
        return True

    def of(self, kind: str) -> list[Fact]:
        return [v for k, v in self.f.items() if k[0] == kind]

    def has(self, kind: str, args: tuple) -> bool:
        return (kind, _norm(kind, args)) in self.f

    def val(self, kind: str, args: tuple):
        f = self.f.get((kind, _norm(kind, args)))
        return f.args[-1] if f else None

    def angval(self, t: tuple[str, str, str]) -> float | None:
        """按角名查度数。事实存的是 (角, 度数)，不能拿角名当整个 args。"""
        for f in self.of("angval"):
            if f.args[0] == t:
                return f.args[1]
        return None


# ---------------------------------------------------------------- 几何上下文


class Ctx:
    """图形结构 + 数值验算。

    结构（共线、连线、between）来自题面与读图结果；数值只用于验算，
    以及在 asin 这类有多解的地方挑出图形实际用的那一个。
    """

    def __init__(self, p: Problem, X: dict[str, tuple[float, float]],
                 between: list[tuple[str, str, str]] | None = None) -> None:
        self.p = p
        self.X = dict(X)
        self.segs = [(a, b) for a, b in p.segments if a in X and b in X]
        self.segset = {_seg(a, b) for a, b in self.segs}
        self.groups: list[set[str]] = []
        for g in collinear_groups(p):
            g2 = set(g) & set(X)
            if len(g2) >= 2:
                self.groups.append(g2)
        self.between = [b for b in (between or []) if all(x in X for x in b)]
        self._tri: list[tuple[str, str, str]] | None = None
        # 题目没给长度时，坐标已按基准边缩放到 1，长度写成 a 的倍数
        self.unit: float | None = 1.0

    def lf(self, v: float) -> str:
        return _len_expr(v, self.unit)

    # --- 结构 ---

    def linked(self, a: str, b: str) -> bool:
        """图上画了这条线。"""
        return a != b and _seg(a, b) in self.segset

    def reach(self, a: str, b: str) -> bool:
        """两点之间连得上：画了线，或者落在同一条直线上。

        F 在 AE 上时图上只有 AE 一条线，但 EF 同样是 △CEF 的一条边，
        不认这一点就凑不出三角形，最后一步 CE=CF 也就推不出来。
        """
        return self.linked(a, b) or self.group(a, b) is not None

    def group(self, *pts: str) -> set[str] | None:
        for g in self.groups:
            if all(x in g for x in pts):
                return g
        return None

    def mid_between(self, x: str, a: str, b: str) -> bool:
        return (x, a, b) in self.between or (x, b, a) in self.between

    def triangles(self) -> list[tuple[str, str, str]]:
        if self._tri is None:
            out = []
            for a, b, c in itertools.combinations(sorted(self.X), 3):
                if self.group(a, b, c) is not None:
                    continue          # 共线的三点不构成三角形
                if self.reach(a, b) and self.reach(b, c) and self.reach(a, c):
                    out.append((a, b, c))
            self._tri = out
        return self._tri

    # --- 数值（只用来验算 / 消歧） ---

    def dist(self, a: str, b: str) -> float:
        (x1, y1), (x2, y2) = self.X[a], self.X[b]
        return math.hypot(x1 - x2, y1 - y2)

    def deg(self, v: str, a: str, b: str) -> float:
        vx, vy = self.X[v]
        ux, uy = self.X[a][0] - vx, self.X[a][1] - vy
        wx, wy = self.X[b][0] - vx, self.X[b][1] - vy
        nu, nw = math.hypot(ux, uy), math.hypot(wx, wy)
        if nu < 1e-9 or nw < 1e-9:
            return 0.0
        c = max(-1.0, min(1.0, (ux * wx + uy * wy) / (nu * nw)))
        return math.degrees(math.acos(c))

    def d(self, t: tuple[str, str, str]) -> float:
        return self.deg(t[0], t[1], t[2])

    def side(self, pt: str, x: str, y: str) -> float:
        """点 pt 在直线 xy 的哪一侧（带符号），共线时为 0。"""
        (x1, y1), (x2, y2), (px, py) = self.X[x], self.X[y], self.X[pt]
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


# ---------------------------------------------------------------- 数值格式


_RAD = {
    2: 1.41421356, 3: 1.73205081, 5: 2.23606798, 6: 2.44948974,
    7: 2.64575131, 8: 2.82842712, 10: 3.16227766, 12: 3.46410162,
    13: 3.60555128, 15: 3.87298335, 17: 4.12310563, 18: 4.24264069,
    20: 4.47213595, 24: 4.89897949, 27: 5.19615242, 32: 5.65685425,
}


def _num(x: float, nd: int = 3) -> str:
    if abs(x) < 5e-4 * max(1.0, abs(x)):
        return "0"
    if abs(x - round(x)) < 5e-4:
        return str(int(round(x)))
    return f"{x:.{nd}f}".rstrip("0").rstrip(".")


def _len_expr(v: float, unit: float | None) -> str:
    """长度写成 a 的倍数：1.414a → √2a，0.5a → a/2。"""
    if unit is None:
        return _num(v)
    r = v / unit
    for q in (1, 2, 3):
        for n, rv in _RAD.items():
            if abs(r * q - rv) < 2e-3:
                return (f"√{n}" if q == 1 else f"√{n}/{q}") + "a"
    if abs(r - round(r)) < 2e-3:
        k = int(round(r))
        return "a" if k == 1 else f"{k}a"
    for q in (2, 3, 4):
        if abs(r * q - round(r * q)) < 2e-3:
            k = int(round(r * q))
            return f"{'a' if k == 1 else str(k) + 'a'}/{q}"
    return _num(r) + "a"


def _len_of(kb: KB, name: str) -> float | None:
    for f in kb.of("len"):
        if f.args[0] == name:
            return f.args[1]
    return None


# ---------------------------------------------------------------- 规则


def _rules(kb: KB, ctx: Ctx):
    # 顺序有讲究：先说清直角，等腰和内角和才用得上；
    # 正弦定理放最后——它是解三角形的计算手段，能靠定理走就别用它。
    yield from _r_right(kb, ctx)
    yield from _r_parallel(kb, ctx)
    yield from _r_corresponding(kb, ctx)
    yield from _r_triangle_sum(kb, ctx)
    yield from _r_split(kb, ctx)
    yield from _r_sum_prop(kb, ctx)
    yield from _r_eq_prop(kb, ctx)
    yield from _r_mid_seg(kb, ctx)
    # 全等/相似放在等量代换之后：判定要的是一堆 eqseg、eqang，得先攒够
    yield from _r_cong(kb, ctx)
    yield from _r_cong_prop(kb, ctx)
    yield from _r_sim(kb, ctx)
    yield from _r_sim_prop(kb, ctx)
    yield from _r_iso(kb, ctx)
    yield from _r_iso_angle(kb, ctx)
    yield from _r_ray(kb, ctx)
    yield from _r_pyth(kb, ctx)
    yield from _r_median(kb, ctx)
    # 圆这一组放在后面：它给出的是角，还要靠内角和、等量代换继续往下走
    yield from _r_diam_right(kb, ctx)
    yield from _r_inscribed(kb, ctx)
    yield from _r_center_angle(kb, ctx)
    yield from _r_tang_perp(kb, ctx)
    yield from _r_tang_chord(kb, ctx)
    yield from _r_tang_len(kb, ctx)
    yield from _r_perp_chord(kb, ctx)
    # 圆幂要在长度都齐了之后算：它自己产出的长度还能再被勾股、相似接着用
    yield from _r_chord_prod(kb, ctx)
    yield from _r_tangent_secant(kb, ctx)
    yield from _r_prod_solve(kb, ctx)
    yield from _r_straight(kb, ctx)
    yield from _r_sine(kb, ctx)


# --- 平行线：内错角 / 同旁内角 --------------------------------------

def _r_parallel(kb: KB, ctx: Ctx):
    for f in kb.of("para"):
        L1, L2 = f.args
        for X in set(L1):
            for Y in set(L2):
                if X == Y or not ctx.linked(X, Y):
                    continue
                for A in set(L1) - {X}:
                    for E in set(L2) - {Y}:
                        s = ctx.side(A, X, Y) * ctx.side(E, X, Y)
                        if abs(s) < 1e-9:
                            continue
                        a1, a2 = _ang(X, A, Y), _ang(Y, X, E)
                        d1, d2 = ctx.d(a1), ctx.d(a2)
                        pre = [f"{_lname(L1)} ∥ {_lname(L2)}，{X}{Y} 是截线"]
                        # 只认内错角。同旁内角互补也能走，但它给出的是一步
                        # 「180 减掉多少」，学生看不到「哪个角等于哪个角」；
                        # 内错角 + 角的分割能推出同样的结果，而且更像几何。
                        if s < 0 and abs(d1 - d2) <= TOL_D:
                            yield ("eqang", (a1, a2), pre, "两直线平行，内错角相等", (f.key,))


def _between_num(X: dict, mid: str, a: str, b: str, tol: float = 2e-2) -> bool:
    """mid 落在 a、b 之间（按已解出的位置量，只用来定方向）。"""
    if not all(k in X for k in (mid, a, b)):
        return False
    ab = math.dist(X[a], X[b])
    if ab < 1e-9:
        return False
    d = math.dist(X[a], X[mid]) + math.dist(X[mid], X[b]) - ab
    return abs(d) / ab < tol


def _between_pts(ctx: Ctx, mid: str, a: str, b: str, tol: float = 2e-2) -> bool:
    """mid 落在 a、b 之间（按已解出的位置量，只用来定方向）。"""
    d = ctx.dist(a, mid) + ctx.dist(mid, b) - ctx.dist(a, b)
    return abs(d) / max(ctx.dist(a, b), 1e-9) < tol


def _r_corresponding(kb: KB, ctx: Ctx):
    """两直线平行，同位角相等。

    内错角那条只管「截线两侧交错」的情形。可 DE∥BC 截出 △ADE 时，
    ∠ADE 与 ∠ABC 在截线 AB 的同一侧——那是同位角。少了这一条，
    「平行 → 两角相等 → 相似」这条路根本起不了步，AA 判定等于没有入口。

    同位角怎么认：取截线上在 X 外侧的点 T，则 ∠TXA（X 处）与 ∠TYC（Y 处）
    是同一个方位的角。两个角都用 T 打头，写出来学生一眼能对上。
    """
    for f in kb.of("para"):
        # 两条直线谁是 L1 不影响结论，但决定「截线上哪个点在 X 外侧」，
        # 所以两个方向都要走：截线 AC 上 E 落在 A、C 之间，只有把 DE 当 L1
        # 时 T=A 才在 E 的外侧，∠AED 与 ∠ACB 才对得上。
        for L1, L2 in (f.args, f.args[::-1]):
            for X in set(L1):
                for Y in set(L2):
                    # 截线不一定画全：D 落在 AB 上时，D 与 B 之间没有单独的线段，
                    # 但 AB 这条直线就是截线。只认画出来的线会漏掉最常见的情形。
                    if X == Y or not ctx.reach(X, Y):
                        continue
                    for T in ctx.X:
                        if T in (X, Y) or not ctx.group(T, X, Y):
                            continue
                        if not _between_pts(ctx, X, T, Y):
                            continue
                        for A in set(L1) - {X}:
                            for C in set(L2) - {Y}:
                                if A not in ctx.X or C not in ctx.X:
                                    continue
                                if ctx.side(A, X, Y) * ctx.side(C, X, Y) <= 0:
                                    continue      # 必须落在截线同一侧
                                a1, a2 = _ang(X, T, A), _ang(Y, T, C)
                                if abs(ctx.d(a1) - ctx.d(a2)) > TOL_D:
                                    continue
                                yield ("eqang", (a1, a2),
                                       [f"{_lname(L1)} ∥ {_lname(L2)}，{T}{Y} 是截线"],
                                       "两直线平行，同位角相等", (f.key,))


# --- 三角形内角和 ---------------------------------------------------

def _r_triangle_sum(kb: KB, ctx: Ctx):
    for a, b, c in ctx.triangles():
        angs = (_ang(a, b, c), _ang(b, a, c), _ang(c, a, b))
        if abs(sum(ctx.d(t) for t in angs) - 180) > TOL_D:
            continue
        yield ("angsum", tuple(angs) + (180.0,), [f"△{a}{b}{c} 的三个内角"],
               "三角形内角和为 180°", ())


# --- 角的分割：∠AOB = ∠AOC + ∠COB ---------------------------------

def _r_split(kb: KB, ctx: Ctx):
    for v in ctx.X:
        arms = sorted(x for x in ctx.X if x != v and ctx.linked(v, x))
        for a, b, c in itertools.permutations(arms, 3):
            da, db, tot = ctx.deg(v, a, c), ctx.deg(v, c, b), ctx.deg(v, a, b)
            if da < TOL_D or db < TOL_D or tot < TOL_D:
                continue
            if abs(da + db - tot) > TOL_D:
                continue
            yield ("angsum", (_ang(v, a, c), _ang(v, c, b), _ang(v, a, b)),
                   [f"{c} 落在 ∠{a}{v}{b} 的内部"], "角的和", ())


# --- 角和 → 单个角 --------------------------------------------------

def _r_sum_prop(kb: KB, ctx: Ctx):
    for f in kb.of("angsum"):
        terms, target = f.args[:-1], f.args[-1]
        tval = kb.angval(target) if isinstance(target, tuple) else float(target)
        known, miss = [], []
        for t in terms:
            v = kb.angval(t)
            (known.append((t, v)) if v is not None else miss.append(t))

        if tval is not None and len(miss) == 1:
            u = miss[0]
            val = round(tval - sum(v for _, v in known), 4)
            if not (TOL_D < val < 180 - TOL_D) or abs(ctx.d(u) - val) > TOL_D:
                continue
            pre = [f"∠{_ang_name(t)} = {_num(v)}°" for t, v in known]
            if isinstance(target, tuple):
                pre.insert(0, f"∠{_ang_name(target)} = {_num(tval)}°")
            deps = [f.key] + [("angval", (t, v)) for t, v in known]
            if isinstance(target, tuple):
                deps.append(("angval", (target, tval)))
            yield ("angval", (u, val), pre, f.rule or "角的和", tuple(deps))

        elif isinstance(target, tuple) and tval is None and not miss:
            val = round(sum(v for _, v in known), 4)
            if not (TOL_D < val < 180 - TOL_D) or abs(ctx.d(target) - val) > TOL_D:
                continue
            pre = [f"∠{_ang_name(t)} = {_num(v)}°" for t, v in known]
            yield ("angval", (target, val), pre, "角的和",
                   tuple([("angval", (t, v)) for t, v in known]))

    # 两个角相等、又正好互补 ⇒ 各 90°。等腰三角形「三线合一」的最后一步就是
    # 它：全等给出 ∠ADB = ∠ADC，B、D、C 共线给出两角和为 180°，于是各是直角。
    for f in kb.of("angsum"):
        terms, target = f.args[:-1], f.args[-1]
        if len(terms) != 2 or isinstance(target, tuple):
            continue
        if abs(float(target) - 180) > 1e-6:
            continue
        t1, t2 = terms
        if t1 == t2 or not kb.has("eqang", (t1, t2)):
            continue
        for u in (t1, t2):
            if kb.angval(u) is not None or abs(ctx.d(u) - 90) > TOL_D:
                continue
            yield ("angval", (u, 90.0),
                   [f"∠{_ang_name(t1)} + ∠{_ang_name(t2)} = 180°",
                    f"∠{_ang_name(t1)} = ∠{_ang_name(t2)}"],
                   "两个角相等且互补，各为 90°", (f.key,))


# --- 等量代换 -------------------------------------------------------

def _r_eq_prop(kb: KB, ctx: Ctx):
    eqs = kb.of("eqang")
    for f in eqs:
        a1, a2 = f.args
        for src, dst in ((a1, a2), (a2, a1)):
            v = kb.angval(src)
            if v is None or kb.has("angval", (dst, v)):
                continue
            if abs(ctx.d(dst) - v) > TOL_D:
                continue
            yield ("angval", (dst, v),
                   [f"∠{_ang_name(src)} = ∠{_ang_name(dst)}",
                    f"∠{_ang_name(src)} = {_num(v)}°"],
                   "等量代换", (f.key, ("angval", (src, v))))

    for f in eqs:
        a1, a2 = f.args
        for g in eqs:
            if g is f:
                continue
            b1, b2 = g.args
            if a2 == b1 and a1 != b2 and not kb.has("eqang", (a1, b2)):
                if abs(ctx.d(a1) - ctx.d(b2)) > TOL_D:
                    continue
                yield ("eqang", (a1, b2),
                       [f"∠{_ang_name(a1)} = ∠{_ang_name(a2)}",
                        f"∠{_ang_name(b1)} = ∠{_ang_name(b2)}"],
                       "等量代换", (f.key, g.key))

    for f in kb.of("eqseg"):
        s1, s2 = f.args
        for src, dst in ((s1, s2), (s2, s1)):
            v = _len_of(kb, src)
            if v is None or _len_of(kb, dst) is not None:
                continue
            if abs(ctx.dist(dst[0], dst[1]) - v) / max(v, 1e-9) > TOL_L:
                continue
            yield ("len", (dst, v), [f"{src} = {dst}", f"{src} = {ctx.lf(v)}"],
                   "等量代换", (f.key, ("len", (src, v))))

    # 两个角量出来一样，就是相等。「等角对等边」要先有一条 eqang 才走得动，
    # 而结论那一步往往正是「算出两个角都是 75°，所以两条边相等」。
    av = [(f.args[0], f.args[1]) for f in kb.of("angval")]
    for i in range(len(av)):
        for j in range(i + 1, len(av)):
            (t1, v1), (t2, v2) = av[i], av[j]
            if t1 == t2 or abs(v1 - v2) > 0.05 or kb.has("eqang", (t1, t2)):
                continue
            yield ("eqang", (t1, t2),
                   [f"∠{_ang_name(t1)} = {_num(v1)}°", f"∠{_ang_name(t2)} = {_num(v2)}°"],
                   "等量代换", (("angval", (t1, v1)), ("angval", (t2, v2))))

    # 等长传递：正方形给了三条等边，第四条要靠传递才出来。
    # 每条都要双向展开——(AB,AD) 与 (AB,CD) 只有从 (AD,AB) 这一侧接得上。
    pairs: list[tuple[str, str]] = []
    for f in kb.of("eqseg"):
        s1, s2 = f.args
        pairs += [(s1, s2), (s2, s1)]
    for a, b in pairs:
        for c, d in pairs:
            if b != c or a == d or kb.has("eqseg", (a, d)):
                continue
            da, dd = ctx.dist(a[0], a[1]), ctx.dist(d[0], d[1])
            if abs(da - dd) / max(da, 1e-9) > TOL_L:
                continue
            yield ("eqseg", (a, d), [f"{a} = {b}", f"{c} = {d}"],
                   "等量代换", ())


# --- 全等三角形 -----------------------------------------------------


def _r_mid_seg(kb: KB, ctx: Ctx):
    """M 是 AB 的中点 ⇒ AM = MB。

    以前中点的用途只有「斜边中线」一条，可中点在证明里最常见的用法就是
    拆出两条相等的边，去凑全等（三线合一那题全靠它）。
    """
    for f in kb.of("mid"):
        m, seg = f.args
        a, b = seg[0], seg[1]
        if m in (a, b) or not all(x in ctx.X for x in (a, b, m)):
            continue
        s1, s2 = _seg(a, m), _seg(m, b)
        if s1 == s2 or kb.has("eqseg", (s1, s2)):
            continue
        d1, d2 = ctx.dist(a, m), ctx.dist(m, b)
        if abs(d1 - d2) / max(d1, d2, 1e-9) > TOL_L:
            continue
        yield ("eqseg", (s1, s2), [f"{m} 是 {seg} 的中点"],
               "线段中点的定义", (f.key,))


def _is_right(kb: KB, t: tuple[str, str, str]) -> bool:
    if kb.has("right", (t,)):
        return True
    v = kb.angval(t)
    return v is not None and abs(v - 90) < TOL_D


def _cong_check(kb: KB, t1: tuple, t2: tuple):
    """在给定的顶点对应下，看看能不能判全等。返回 (判定, 条件文字, 依赖)。

    判定顺序按「条件越少越先说」排：HL 两条就够，AAS 要三条，
    学生写作业时也更愿意用前面的。
    """
    S1, S2 = _tri_sides(t1), _tri_sides(t2)
    A1, A2 = _tri_angs(t1), _tri_angs(t2)

    # 公共边（△ABC 与 △ADC 共用 AC）两边其实是同一条线段，不会出现在
    # eqseg 里，但它确实是一条成立的对应——斜边为公共边的 HL 全靠它。
    # 代价是可能拿「同一条边」去凑条件，所以下面要求至少两条**真**条件。
    same_s = [S1[k] == S2[k] for k in range(3)]
    es = [same_s[k] or kb.has("eqseg", (S1[k], S2[k])) for k in range(3)]
    ea = [A1[k] != A2[k] and kb.has("eqang", (A1[k], A2[k])) for k in range(3)]

    def ng(sides=(), angs=()) -> int:
        """数一数用了几条真条件（公共边不算）。"""
        return sum(1 for k in sides if not same_s[k]) + len(angs)

    def sd(k: int) -> str:
        return f"{S1[k]} = {S2[k]}" if not same_s[k] else f"{S1[k]} 是公共边"

    def ag(k: int) -> str:
        return f"∠{_ang_name(A1[k])} = ∠{_ang_name(A2[k])}"

    def dep(kind: str, k: int) -> tuple:
        return (kind, _norm(kind, (S1[k], S2[k]) if kind == "eqseg" else (A1[k], A2[k])))

    # 斜边、直角边：两个直角三角形，斜边和一条直角边对应相等。
    # 斜边是直角顶点的对边——第 k 个角的对边是第 (k+1) 条边；第 k 条边
    # 连的是 v_k、v_(k+1)，正对 v_(k+2)。写反了会把一条直角边当斜边。
    for k in range(3):
        if not (_is_right(kb, A1[k]) and _is_right(kb, A2[k])):
            continue
        hyp = (k + 1) % 3
        legs = [m for m in (k, (k + 2) % 3) if es[m]]
        if es[hyp] and legs:
            m = legs[0]
            if ng((hyp, m)) < 2:
                continue
            return ("斜边、直角边（HL）", [sd(hyp), sd(m)],
                    (dep("eqseg", hyp), dep("eqseg", m)))

    if all(es) and ng((0, 1, 2)) >= 2:
        return ("边边边（SSS）", [sd(0), sd(1), sd(2)],
                tuple(dep("eqseg", k) for k in range(3)))

    for i in range(3):
        for j in range(i + 1, 3):
            if es[i] and es[j]:
                k = _ang_between(i, j)
                if k >= 0 and ea[k] and ng((i, j), (k,)) >= 2:
                    return ("边角边（SAS）", [sd(i), sd(j), ag(k)],
                            (dep("eqseg", i), dep("eqseg", j), dep("eqang", k)))
            if ea[i] and ea[j]:
                k = _side_between(i, j)
                if k >= 0 and es[k] and ng((k,), (i, j)) >= 2:
                    return ("角边角（ASA）", [ag(i), ag(j), sd(k)],
                            (dep("eqang", i), dep("eqang", j), dep("eqseg", k)))
                other = [m for m in range(3) if m != k]
                for m in other:
                    if es[m] and ng((m,), (i, j)) >= 2:
                        return ("角角边（AAS）", [ag(i), ag(j), sd(m)],
                                (dep("eqang", i), dep("eqang", j), dep("eqseg", m)))
    return None


def _cong_holds(ctx: Ctx, t1: tuple, t2: tuple) -> bool:
    """在解出的图形上量一遍：判定说得再对，图形对不上也不能用。"""
    for k in range(3):
        s1, s2 = _seg(t1[k], t1[(k + 1) % 3]), _seg(t2[k], t2[(k + 1) % 3])
        d1, d2 = ctx.dist(s1[0], s1[1]), ctx.dist(s2[0], s2[1])
        if abs(d1 - d2) / max(d1, d2, 1e-9) > TOL_L:
            return False
    return True


def _r_cong(kb: KB, ctx: Ctx):
    """判定两个三角形全等。

    对应关系得自己找：六个字母写出来只是两个集合，谁跟谁对应要靠条件定。
    这里穷举 6 种对应，哪种能凑齐判定条件就用哪种。
    """
    tris = ctx.triangles()
    for i, t1 in enumerate(tris):
        for t2 in tris[i + 1:]:
            for perm in itertools.permutations(t2):
                if not _cong_holds(ctx, t1, perm):
                    continue
                got = _cong_check(kb, t1, perm)
                if got is None:
                    continue
                crit, pre, deps = got
                x, y = _canon_pair(t1, perm)
                yield ("cong", (x, y), pre, crit, deps)


def _r_cong_prop(kb: KB, ctx: Ctx):
    """全等三角形的对应边相等、对应角相等。

    这才是判定的用处：证完全等，才知道 AE = BF、∠BAE = ∠CBF。
    """
    for f in kb.of("cong"):
        x, y = f.args
        pre = [f"△{''.join(x)} ≌ △{''.join(y)}"]
        S1, S2 = _tri_sides(x), _tri_sides(y)
        A1, A2 = _tri_angs(x), _tri_angs(y)

        for k in range(3):
            s1, s2 = S1[k], S2[k]
            if s1 != s2 and not kb.has("eqseg", (s1, s2)):
                d1 = ctx.dist(s1[0], s1[1])
                d2 = ctx.dist(s2[0], s2[1])
                if abs(d1 - d2) / max(d1, d2, 1e-9) <= TOL_L:
                    yield ("eqseg", (s1, s2), pre, "全等三角形的对应边相等", (f.key,))

        for k in range(3):
            a1, a2 = A1[k], A2[k]
            if a1 != a2 and not kb.has("eqang", (a1, a2)):
                if abs(ctx.d(a1) - ctx.d(a2)) <= TOL_D:
                    yield ("eqang", (a1, a2), pre, "全等三角形的对应角相等", (f.key,))


# --- 相似三角形 -----------------------------------------------------


def _sim_check(kb: KB, ctx: Ctx, t1: tuple, t2: tuple):
    """判相似，并尽量给出相似比。返回 (判定, 条件文字, 依赖, 相似比)。"""
    S1, S2 = _tri_sides(t1), _tri_sides(t2)
    A1, A2 = _tri_angs(t1), _tri_angs(t2)

    # 数值兜底：三组对应边必须成同一比例，否则这条路是错的
    m1 = [ctx.dist(s[0], s[1]) for s in S1]
    m2 = [ctx.dist(s[0], s[1]) for s in S2]
    r = [m2[k] / max(m1[k], 1e-12) for k in range(3)]
    if max(r) - min(r) > 0.02 * max(r):
        return None

    ea = [A1[k] != A2[k] and kb.has("eqang", (A1[k], A2[k])) for k in range(3)]
    l1 = [_len_of(kb, s) for s in S1]
    l2 = [_len_of(kb, s) for s in S2]
    lr = [l2[k] / l1[k] if (l1[k] and l2[k]) else None for k in range(3)]

    def ag(k: int) -> str:
        return f"∠{_ang_name(A1[k])} = ∠{_ang_name(A2[k])}"

    k_ratio = next((v for v in lr if v is not None), None)
    if k_ratio is not None and abs(k_ratio - 1.0) < 1e-6:
        k_ratio = None      # 比例是 1 就是全等，那条路另有判定，别在这里混

    # 相似比只能从已知长度算，所以得把这组长度写进 ∵ 里，不然学生不知道
    # 「相似比 0.5」是从哪儿冒出来的。
    ratio_pre: list[str] = []
    if k_ratio is not None:
        i = next(i for i in range(3) if lr[i] is not None)
        ratio_pre.append(f"{S1[i]} = {ctx.lf(l1[i])}，{S2[i]} = {ctx.lf(l2[i])}")

    # 两角分别相等（AA）——初中最常用的一条
    if sum(ea) >= 2:
        ks = [k for k in range(3) if ea[k]][:2]
        return ("两角分别相等（AA）", [ag(k) for k in ks] + ratio_pre,
                tuple(("eqang", _norm("eqang", (A1[k], A2[k]))) for k in ks), k_ratio)

    # 两边成比例且夹角相等（SAS）
    for i in range(3):
        for j in range(i + 1, 3):
            if lr[i] is None or lr[j] is None:
                continue
            if abs(lr[i] - lr[j]) > 1e-3:
                continue
            k = _ang_between(i, j)
            if k >= 0 and ea[k]:
                return ("两边成比例且夹角相等（SAS）",
                        [f"{S1[i]} = {ctx.lf(l1[i])}，{S2[i]} = {ctx.lf(l2[i])}",
                         f"{S1[j]} = {ctx.lf(l1[j])}，{S2[j]} = {ctx.lf(l2[j])}", ag(k)],
                        (("eqang", _norm("eqang", (A1[k], A2[k]))),), lr[i])

    # 三边成比例（SSS）
    if all(v is not None for v in lr) and max(lr) - min(lr) < 1e-3:
        return ("三边成比例（SSS）",
                [f"{S1[i]}:{S2[i]} = {S1[(i + 1) % 3]}:{S2[(i + 1) % 3]}" for i in range(2)],
                (), lr[0])
    return None


def _r_sim(kb: KB, ctx: Ctx):
    tris = ctx.triangles()
    for i, t1 in enumerate(tris):
        for t2 in tris[i + 1:]:
            for perm in itertools.permutations(t2):
                got = _sim_check(kb, ctx, t1, perm)
                if got is None:
                    continue
                crit, pre, deps, kr = got
                x, y, kr = _canon_sim(t1, perm, kr)
                yield ("sim", (x, y, kr), pre, crit, deps)


def _r_sim_prop(kb: KB, ctx: Ctx):
    """相似三角形的对应角相等、对应边成比例。"""
    for f in kb.of("sim"):
        x, y, kr = f.args
        pre = [f"△{''.join(x)} ∽ △{''.join(y)}"
               + (f"，相似比 {_num(kr)}" if kr is not None else "")]
        S1, S2 = _tri_sides(x), _tri_sides(y)
        A1, A2 = _tri_angs(x), _tri_angs(y)

        for k in range(3):
            a1, a2 = A1[k], A2[k]
            if a1 != a2 and not kb.has("eqang", (a1, a2)):
                if abs(ctx.d(a1) - ctx.d(a2)) <= TOL_D:
                    yield ("eqang", (a1, a2), pre, "相似三角形的对应角相等", (f.key,))

        if kr is None:
            continue
        for k in range(3):
            s1, s2 = S1[k], S2[k]
            if s1 == s2:
                continue
            v1, v2 = _len_of(kb, s1), _len_of(kb, s2)
            for src, dst, mult in ((v1, s2, kr), (v2, s1, 1.0 / kr)):
                if src is None or _len_of(kb, dst) is not None:
                    continue
                want = src * mult
                if abs(ctx.dist(dst[0], dst[1]) - want) / max(want, 1e-9) > TOL_L:
                    continue
                yield ("len", (dst, round(want, 6)), pre,
                       "相似三角形对应边成比例", (f.key,))


# --- 等腰三角形的角度计算 -------------------------------------------


def _r_iso_angle(kb: KB, ctx: Ctx):
    """顶角已知求底角，或底角已知求顶角。

    「等边对等角」只给出两个角相等，还得再走一步内角和才能落到数值上。
    这一步学生天天在写（顶角 90° 的等腰三角形，底角各 45°），单独成规则。
    """
    for tri in ctx.triangles():
        a, b, c = tri
        for apex, o1, o2 in ((a, b, c), (b, a, c), (c, a, b)):
            if not kb.has("eqseg", (_seg(apex, o1), _seg(apex, o2))):
                continue
            top = kb.angval(_ang(apex, o1, o2))
            g1, g2 = _ang(o1, apex, o2), _ang(o2, apex, o1)

            if top is not None:
                val = round((180 - top) / 2, 4)
                for g in (g1, g2):        # 两个底角都要给，只给一个就断链
                    if kb.angval(g) is not None or val < TOL_D:
                        continue
                    if abs(ctx.d(g) - val) > TOL_D:
                        continue
                    yield ("angval", (g, val),
                           [f"{_seg(apex, o1)} = {_seg(apex, o2)}（△{''.join(sorted((apex, o1, o2)))} 是等腰三角形）",
                            f"顶角 ∠{_ang_name(_ang(apex, o1, o2))} = {_num(top)}°"],
                           "等腰三角形两底角相等，三角形内角和 180°",
                           (("angval", (_ang(apex, o1, o2), top)),))

            base = kb.angval(g1)
            if base is not None and top is None:
                val = round(180 - 2 * base, 4)
                if abs(ctx.d(_ang(apex, o1, o2)) - val) > TOL_D or val < TOL_D:
                    continue
                yield ("angval", (_ang(apex, o1, o2), val),
                       [f"{_seg(apex, o1)} = {_seg(apex, o2)}（△{''.join(sorted((apex, o1, o2)))} 是等腰三角形）",
                        f"底角 ∠{_ang_name(g1)} = {_num(base)}°"],
                       "等腰三角形两底角相等，三角形内角和 180°", ())


# --- 等腰：等边对等角、等角对等边 -----------------------------------

def _r_iso(kb: KB, ctx: Ctx):
    for tri in ctx.triangles():
        a, b, c = tri
        for apex, o1, o2 in ((a, b, c), (b, a, c), (c, a, b)):
            if not kb.has("eqseg", (_seg(apex, o1), _seg(apex, o2))):
                continue
            g1, g2 = _ang(o1, apex, o2), _ang(o2, apex, o1)
            if abs(ctx.d(g1) - ctx.d(g2)) > TOL_D:
                continue
            yield ("eqang", (g1, g2),
                   [f"{_seg(apex, o1)} = {_seg(apex, o2)}（△{''.join(sorted((apex, o1, o2)))} 的两腰）"],
                   "等边对等角", (("eqseg", (_seg(apex, o1), _seg(apex, o2))),))

    for f in kb.of("eqang"):
        a1, a2 = f.args
        for tri in ctx.triangles():
            if not (set(a1) <= set(tri) and set(a2) <= set(tri)):
                continue
            s1, s2 = _seg(a1[1], a1[2]), _seg(a2[1], a2[2])
            if s1 == s2 or kb.has("eqseg", (s1, s2)):
                continue
            d1, d2 = ctx.dist(s1[0], s1[1]), ctx.dist(s2[0], s2[1])
            if abs(d1 - d2) / max(d1, 1e-9) > TOL_L:
                continue
            yield ("eqseg", (s1, s2), [f"∠{_ang_name(a1)} = ∠{_ang_name(a2)}"],
                   "等角对等边", (f.key,))
            break


# --- 同一条射线上的角 -----------------------------------------------

def _r_ray(kb: KB, ctx: Ctx):
    """P 在 AB 上，则射线 V→P 与 V→A 是同一条，∠XVP = ∠XVA。

    顶点 V 可以是 A 或 B 本身——「F 在 CD 上」要用的正是射线 C→F 与 C→D，
    把端点排掉就把这条推没了。
    """
    for x, a, b in ctx.between:            # x 在 a、b 之间
        for v in ctx.X:
            if v == x or not ctx.group(v, a, b):
                continue
            for other in ctx.X:
                if other in (v, a, b, x) or not ctx.linked(v, other):
                    continue
                # 射线 v→x 与 v→? 同向：v 就是端点 a 时，另一头是 b
                far = b if v == a else a
                t1, t2 = _ang(v, other, x), _ang(v, other, far)
                if t1 == t2 or abs(ctx.d(t1) - ctx.d(t2)) > TOL_D:
                    continue
                yield ("eqang", (t1, t2),
                       [f"{x} 在 {a}{b} 上，射线 {v}{x} 与 {v}{far} 是同一条"],
                       "同一条射线上的角相等", ())


# --- 直角 -----------------------------------------------------------

def _r_right(kb: KB, ctx: Ctx):
    for f in kb.of("right"):
        yield ("angval", (f.args[0], 90.0),
               [f"∠{_ang_name(f.args[0])} = 90°"], "直角的定义", (f.key,))

    # 直角的一边上任取一点，和顶点连成的仍是直角：∠ABC = 90°、E 在 BC 上
    # ⇒ ∠ABE = 90°。不写这条，正方形题得先「同射线角相等」再「等量代换」，
    # 两步才走到同一个结论，学生看着像在绕。
    for f in kb.of("right"):
        v, p, q = f.args[0]
        for ext, other in ((q, p), (p, q)):
            for x in ctx.X:
                if x in (v, p, q) or not ctx.group(x, v, ext):
                    continue
                a = _ang(v, other, x)
                if abs(ctx.d(a) - 90) > TOL_D or kb.has("right", (a,)):
                    continue
                yield ("right", (a,),
                       [f"∠{_ang_name(f.args[0])} = 90°（已知）",
                        f"{x} 在 {_lname((v, ext))} 所在的直线上"],
                       "直角的一边上取一点，与顶点连成的角仍是直角", (f.key,))


# --- 勾股定理 -------------------------------------------------------

def _r_pyth(kb: KB, ctx: Ctx):
    for tri in ctx.triangles():
        a, b, c = tri
        for v, p, q in ((a, b, c), (b, a, c), (c, a, b)):
            ang = _ang(v, p, q)
            if not kb.has("right", (ang,)):
                continue
            hyp = _seg(p, q)
            l1, l2 = _len_of(kb, _seg(v, p)), _len_of(kb, _seg(v, q))
            lh = _len_of(kb, hyp)
            if lh is None and l1 is not None and l2 is not None:
                yield ("len", (hyp, round(math.hypot(l1, l2), 6)),
                       [f"∠{_ang_name(ang)} = 90°，{_seg(v, p)} = {ctx.lf(l1)}，{_seg(v, q)} = {ctx.lf(l2)}"],
                       "勾股定理",
                       (("len", (_seg(v, p), l1)), ("len", (_seg(v, q), l2))))
            elif l2 is None and lh is not None and l1 is not None and lh > l1:
                yield ("len", (_seg(v, q), round(math.sqrt(max(lh * lh - l1 * l1, 0.0)), 6)),
                       [f"∠{_ang_name(ang)} = 90°，{_seg(v, p)} = {ctx.lf(l1)}，{hyp} = {ctx.lf(lh)}"],
                       "勾股定理", ())

        # 逆定理：三边满足 a² + b² = c²，那么 c 所对的角是直角。
        # 「3、4、5」这种题全靠它把长度关系翻成直角——没有它，旋转构造出来的
        # 三角形算得再准，也走不到「所以是直角三角形」那一步。
        for v, p, q in ((a, b, c), (b, a, c), (c, a, b)):
            ang = _ang(v, p, q)
            if kb.has("right", (ang,)) or kb.angval(ang) is not None:
                continue
            l1, l2 = _len_of(kb, _seg(v, p)), _len_of(kb, _seg(v, q))
            lh = _len_of(kb, _seg(p, q))
            if l1 is None or l2 is None or lh is None:
                continue
            if abs(l1 * l1 + l2 * l2 - lh * lh) / max(lh * lh, 1e-9) > 2e-3:
                continue
            if abs(ctx.d(ang) - 90) > TOL_D:
                continue
            yield ("right", (ang,),
                   [f"{_seg(v, p)} = {ctx.lf(l1)}，{_seg(v, q)} = {ctx.lf(l2)}，{_seg(p, q)} = {ctx.lf(lh)}",
                    f"{ctx.lf(l1)}² + {ctx.lf(l2)}² = {ctx.lf(lh)}²"],
                   "勾股定理的逆定理", ())


# --- 直角三角形斜边上的中线 -----------------------------------------

def _r_median(kb: KB, ctx: Ctx):
    for f in kb.of("mid"):
        m, seg = f.args
        a, b = seg[0], seg[1]
        # 中点在斜边上，所以 A、M、B 共线，凑不出三角形——
        # 只能从「谁是直角顶点」反着找，不能走 triangles()。
        for c in ctx.X:
            if c in (a, b, m):
                continue
            ang = _ang(c, a, b)
            if not kb.has("right", (ang,)):
                continue
            if not (ctx.reach(c, a) and ctx.reach(c, b)):
                continue
            la, lb = _len_of(kb, _seg(m, a)), _len_of(kb, _seg(m, b))
            lh = _len_of(kb, seg)
            pre = [f"∠{_ang_name(ang)} = 90°，{m} 是斜边 {seg} 的中点"]
            deps: tuple = (f.key,)
            if lh is not None:
                # CD（直角顶点到斜边中点）和 DA、DB 都等于斜边的一半
                pre.append(f"{seg} = {ctx.lf(lh)}")
                deps += (("len", (seg, lh)),)
                half = round(lh / 2, 6)
                for t in (_seg(c, m), _seg(m, a), _seg(m, b)):
                    if _len_of(kb, t) is None:
                        yield ("len", (t, half), pre,
                               "直角三角形斜边上的中线等于斜边的一半", deps)
            elif la is not None and lb is None:
                yield ("len", (_seg(m, b), la),
                       [f"∠{_ang_name(ang)} = 90°，{m} 是斜边 {seg} 的中点"],
                       "直角三角形斜边上的中线等于斜边的一半", (f.key,))


# --- 正弦定理 -------------------------------------------------------

def _r_sine(kb: KB, ctx: Ctx):
    """已知两边和其中一边的对角，求另一边的对角。"""
    for tri in ctx.triangles():
        a, b, c = tri
        pairs = [
            (_seg(b, c), _ang(a, b, c)),   # 边 BC 对 ∠A
            (_seg(a, c), _ang(b, a, c)),
            (_seg(a, b), _ang(c, a, b)),
        ]
        for i, j in itertools.permutations(range(3), 2):
            si, thi = pairs[i]
            sj, thj = pairs[j]
            li, lj = _len_of(kb, si), _len_of(kb, sj)
            vi, vj = kb.angval(thi), kb.angval(thj)
            if li is None or lj is None or vi is None or vj is not None:
                continue
            s = lj * math.sin(math.radians(vi)) / li
            if not (-1.0 <= s <= 1.0) or s < 1e-9:
                continue
            actual = ctx.d(thj)
            base = math.degrees(math.asin(s))
            pick = min((base, 180 - base), key=lambda x: abs(x - actual))
            if abs(pick - actual) > TOL_D or pick < TOL_D:
                continue
            yield ("angval", (thj, round(pick, 4)),
                   [f"在 △{a}{b}{c} 中，{si} = {ctx.lf(li)}，{sj} = {ctx.lf(lj)}，∠{_ang_name(thi)} = {_num(vi)}°"],
                   "正弦定理",
                   (("len", (si, li)), ("len", (sj, lj)), ("angval", (thi, vi))))


# --- 圆 -------------------------------------------------------------
#
# 圆的定理和直线形不一样：条件全藏在「谁在圆上」里。所以先把圆周上的点、
# 直径、切线整理成事实（见 _seed），下面每条规则只认这些事实。
# 「同弧」「同侧」这类关系没法从题面读到，一律按解出来的位置判——判错了
# 结论过不了数值验算，整条丢弃，不会留下一条漂亮的假证明。


def _ring(kb: KB, ctx: Ctx, o: str) -> list[str]:
    """⊙o 上、且解出来有坐标的点。"""
    return sorted(f.args[0] for f in kb.of("on_circ")
                  if f.args[1] == o and f.args[0] in ctx.X)


def _r_diam_right(kb: KB, ctx: Ctx):
    """直径所对的圆周角是直角。"""
    for f in kb.of("diam"):
        a, b, o = f.args
        for c in _ring(kb, ctx, o):
            if c in (a, b):
                continue
            t = _ang(c, a, b)
            if kb.has("right", (t,)) or abs(ctx.d(t) - 90) > TOL_D:
                continue
            yield ("right", (t,),
                   [f"{a}{b} 是 ⊙{o} 的直径，{c} 在圆上"],
                   "直径所对的圆周角是直角",
                   (f.key, ("on_circ", (c, o))))


def _r_inscribed(kb: KB, ctx: Ctx):
    """同弧所对的圆周角相等；圆内接四边形对角互补。

    这两个是同一件事的两面：C、D 在弦 AB 同侧，⌒AB 是同一段弧，两角相等；
    在异侧，四边形 ACBD 内接于圆，两角是对角，互补。用「同侧还是异侧」
    一句话就分开了，不用另外认什么四边形。
    """
    seen: set[tuple] = set()
    for f in kb.of("center"):
        o = f.args[0]
        ring = _ring(kb, ctx, o)
        for a, b in itertools.combinations(ring, 2):
            rest = [x for x in ring if x not in (a, b)]
            for c, d in itertools.combinations(rest, 2):
                t1, t2 = _ang(c, a, b), _ang(d, a, b)
                if t1 == t2 or frozenset((t1, t2)) in seen:
                    continue
                seen.add(frozenset((t1, t2)))
                s = ctx.side(c, a, b) * ctx.side(d, a, b)
                if s > 1e-9:
                    if abs(ctx.d(t1) - ctx.d(t2)) > TOL_D:
                        continue
                    yield ("eqang", (t1, t2),
                           [f"{c}、{d} 都在弦 {a}{b} 的同侧，⌒{a}{b} 是同一条弧"],
                           "同弧所对的圆周角相等", (f.key,))
                elif s < -1e-9:
                    if abs(ctx.d(t1) + ctx.d(t2) - 180) > TOL_D:
                        continue
                    yield ("angsum", (t1, t2, 180.0),
                           [f"{''.join(sorted([a, c, b, d]))} 内接于 ⊙{o}，"
                            f"∠{_ang_name(t1)} 与 ∠{_ang_name(t2)} 是对角"],
                           "圆内接四边形的对角互补", (f.key,))


def _r_center_angle(kb: KB, ctx: Ctx):
    """同弧所对的圆心角是圆周角的两倍。

    圆心角取的是图形上实际画出来的那个（∠AOB 恒 ≤ 180°）。若题目要的是
    优弧上的圆心角，量出来会对不上，这里直接跳过而不是写个 360° 减出来的数
    ——∠AOB 在板书里就是那个小角，写成 250° 学生会以为画错了。
    """
    for f in kb.of("center"):
        o = f.args[0]
        ring = _ring(kb, ctx, o)
        for a, b in itertools.combinations(ring, 2):
            central = _ang(o, a, b)
            for c in ring:
                if c in (a, b):
                    continue
                ins = _ang(c, a, b)
                vi, vc = kb.angval(ins), kb.angval(central)
                if vi is not None and vc is None:
                    val = round(2 * vi, 4)
                    if not (TOL_D < val < 180 - TOL_D) or abs(ctx.d(central) - val) > TOL_D:
                        continue
                    yield ("angval", (central, val),
                           [f"∠{_ang_name(ins)} = {_num(vi)}°"],
                           "同弧所对的圆心角是圆周角的两倍",
                           (f.key, ("angval", (ins, vi))))
                elif vc is not None and vi is None:
                    val = round(vc / 2, 4)
                    if not (TOL_D < val) or abs(ctx.d(ins) - val) > TOL_D:
                        continue
                    yield ("angval", (ins, val),
                           [f"∠{_ang_name(central)} = {_num(vc)}°"],
                           "同弧所对的圆周角是圆心角的一半",
                           (f.key, ("angval", (central, vc))))


def _r_tang_perp(kb: KB, ctx: Ctx):
    """切线垂直于过切点的半径。"""
    for f in kb.of("tang"):
        tp, o, far = f.args
        t = _ang(tp, o, far)
        if kb.has("right", (t,)) or abs(ctx.d(t) - 90) > TOL_D:
            continue
        yield ("right", (t,),
               [f"{tp}{far} 是 ⊙{o} 在点 {tp} 处的切线，{o}{tp} 是半径"],
               "切线垂直于过切点的半径", (f.key,))


def _r_tang_chord(kb: KB, ctx: Ctx):
    """弦切角等于它所夹的弧对的圆周角。

    这条是圆里最管用的一条：题上给了切线、又要求一个跟切线有关的角度时，
    光靠「切线 ⊥ 半径」往往绕不出来，而弦切角一步就把角度搬到圆上。
    """
    for f in kb.of("tang"):
        tp, o, far = f.args
        ring = _ring(kb, ctx, o)
        for b in ring:
            if b == tp:
                continue
            tan_ang = _ang(tp, far, b)          # 切线与弦 tp-b 的夹角
            for c in ring:
                if c in (tp, b):
                    continue
                ins = _ang(c, tp, b)            # 弦 tp-b 所对的圆周角
                if ins == tan_ang or kb.has("eqang", (tan_ang, ins)):
                    continue
                if abs(ctx.d(tan_ang) - ctx.d(ins)) > TOL_D:
                    continue
                yield ("eqang", (tan_ang, ins),
                       [f"{tp}{far} 是切线，{tp}{b} 是过切点的弦，切点在 {tp}"],
                       "弦切角等于它所夹的弧对的圆周角", (f.key,))


def _r_tang_len(kb: KB, ctx: Ctx):
    """从圆外一点引两条切线，切线长相等。"""
    by_far: dict[str, list[Fact]] = {}
    for f in kb.of("tang"):
        by_far.setdefault(f.args[2], []).append(f)
    for far, fs in by_far.items():
        for f1, f2 in itertools.combinations(fs, 2):
            tp1, o, _ = f1.args
            tp2 = f2.args[0]
            if f1.args[1] != f2.args[1] or tp1 == tp2:
                continue
            s1, s2 = _seg(far, tp1), _seg(far, tp2)
            if kb.has("eqseg", (s1, s2)):
                continue
            d1, d2 = ctx.dist(far, tp1), ctx.dist(far, tp2)
            if abs(d1 - d2) / max(d1, d2, 1e-9) > TOL_L:
                continue
            yield ("eqseg", (s1, s2),
                   [f"{tp1}{far}、{tp2}{far} 都是过 {far} 的 ⊙{o} 的切线"],
                   "过圆外一点引圆的两条切线，切线长相等", (f1.key, f2.key))


def _r_perp_chord(kb: KB, ctx: Ctx):
    """垂径定理：直径垂直于弦，就平分这条弦。"""
    for f in kb.of("diam"):
        a, b, o = f.args
        ring = _ring(kb, ctx, o)
        for m in ctx.X:
            if m in (a, b, o) or not ctx.group(m, a, b):
                continue
            for u, v in itertools.combinations(ring, 2):
                if m in (u, v):
                    continue
                # 垂足连向弦的两端，这条弦才算「过垂足」；光有共线不够，
                # 读图很少会把「E 在 CD 上」写进题面。
                if not (ctx.reach(m, u) and ctx.reach(m, v)):
                    continue
                if abs(ctx.deg(m, u, a) - 90) > TOL_D:
                    continue
                if not _between_pts(ctx, m, u, v):
                    continue
                if kb.has("mid", (m, _seg(u, v))):
                    continue
                yield ("mid", (m, _seg(u, v)),
                       [f"{a}{b} 是直径，{a}{b} ⊥ 弦 {u}{v}，垂足是 {m}"],
                       "垂径定理（直径垂直于弦，平分这条弦）", (f.key,))


# --- 圆幂：相交弦 / 割线 / 切割线 ------------------------------------

def _on_line(ctx: Ctx, pt: str, a: str, b: str, tol: float = 8e-3) -> bool:
    """pt 落在直线 ab 上（按坐标量，容差按图幅比例）。

    这里的判据和「between」不同：割线的交点常常落在圆外，压根不在线段
    之间，但那条「直线」确实穿过它。所以只判到直线的距离。
    """
    if not all(x in ctx.X for x in (pt, a, b)):
        return False
    L = ctx.dist(a, b)
    if L < 1e-9:
        return False
    return abs(ctx.side(pt, a, b)) / (L * L) < tol


def _prod_text(args: tuple) -> str:
    (s1, s2), (s3, s4) = args
    left = f"{s1}²" if s1 == s2 else f"{s1}·{s2}"
    right = f"{s3}²" if s3 == s4 else f"{s3}·{s4}"
    return f"{left} = {right}"


def _r_chord_prod(kb: KB, ctx: Ctx):
    """相交弦定理与割线定理：同一点引出的两条线，被圆截成四段，两两乘积相等。

    这其实是同一件事（点对圆的幂），叫法随位置变：点在圆内叫相交弦，
    在圆外叫割线。判「内还是外」要半径，而半径是量出来的——所以它只
    决定写哪个定理名，等式本身照样要在图上验一遍。

    为什么值得加：这类题给的往往是「PA=2，PB=6，PC=3，求 PD」，光靠
    相似三角形也能绕出来，但绕得远；这条定理一步到位。
    """
    for f in kb.of("center"):
        o = f.args[0]
        ring = _ring(kb, ctx, o)
        if len(ring) < 4:
            continue
        r = ctx.dist(o, ring[0])
        for p in ctx.X:
            if p == o or p in ring:
                continue
            chords = [(a, b) for a, b in itertools.combinations(ring, 2)
                      if _on_line(ctx, p, a, b)]
            for (a, b), (c, d) in itertools.combinations(chords, 2):
                if len({a, b, c, d}) < 4:
                    continue
                s1, s2 = _seg(p, a), _seg(p, b)
                s3, s4 = _seg(p, c), _seg(p, d)
                if len({s1, s2, s3, s4}) < 4:
                    continue
                if kb.has("prod", ((s1, s2), (s3, s4))):
                    continue
                l1 = ctx.dist(p, a) * ctx.dist(p, b)
                l2 = ctx.dist(p, c) * ctx.dist(p, d)
                if abs(l1 - l2) / max(l1, l2, 1e-9) > 1e-2:
                    continue
                inside = ctx.dist(p, o) < r
                word = "弦" if inside else "割线"
                yield ("prod", ((s1, s2), (s3, s4)),
                       [f"{a}{b}、{c}{d} 是过点 {p} 的两条{word}"],
                       "相交弦定理" if inside else "割线定理", (f.key,))


def _r_tangent_secant(kb: KB, ctx: Ctx):
    """切割线定理：从圆外一点引一条切线和一条割线，切线长的平方等于两段之积。"""
    for f in kb.of("tang"):
        tp, o, far = f.args
        ring = _ring(kb, ctx, o)
        if len(ring) < 2:
            continue
        for a, b in itertools.combinations(ring, 2):
            if far in (a, b) or not _on_line(ctx, far, a, b):
                continue
            s = _seg(far, tp)
            s1, s2 = _seg(far, a), _seg(far, b)
            if len({s, s1, s2}) < 3:
                continue
            if kb.has("prod", ((s, s), (s1, s2))):
                continue
            l1 = ctx.dist(far, tp) ** 2
            l2 = ctx.dist(far, a) * ctx.dist(far, b)
            if abs(l1 - l2) / max(l1, l2, 1e-9) > 1e-2:
                continue
            yield ("prod", ((s, s), (s1, s2)),
                   [f"{tp}{far} 是 ⊙{o} 的切线，切点为 {tp}，{a}{b} 是过 {far} 的割线"],
                   "切割线定理", (f.key,))


def _r_prod_solve(kb: KB, ctx: Ctx):
    """四个长度知道三个，第四个由圆幂等式算出来。

    切线那条要小心：等式左边是 PT·PT，未知量出现两次，开方而不是除法。
    这就是「PG·PG = PA·PB ⇒ PG = √(PA·PB)」，写错成一次方程一定算错。
    """
    for f in kb.of("prod"):
        segs = list(f.args[0]) + list(f.args[1])
        vals = [_len_of(kb, s) for s in segs]
        for i, name in enumerate(segs):
            if vals[i] is not None:
                continue
            mine = segs[:2] if i < 2 else segs[2:]
            theirs = segs[2:] if i < 2 else segs[:2]
            if any(v is None for v in
                   ([vals[2], vals[3]] if i < 2 else [vals[0], vals[1]])):
                continue
            other, cnt = 1.0, 0
            for s in mine:
                if s == name:
                    cnt += 1
                    continue
                v = _len_of(kb, s)
                if v is None:
                    other = None
                    break
                other *= v
            if other is None or cnt == 0 or other <= 0:
                continue
            opp = 1.0
            for v in ([vals[2], vals[3]] if i < 2 else [vals[0], vals[1]]):
                opp *= v
            want = (opp / other) ** (1.0 / cnt)
            if want <= 0:
                continue
            d = ctx.dist(name[0], name[1])
            if abs(d - want) / max(want, 1e-9) > TOL_L * 5:
                continue
            pre = [_prod_text(f.args)]
            seen_s: set[str] = set()
            for s, v in zip(segs, vals):
                if v is None or s in seen_s:
                    continue
                seen_s.add(s)
                pre.append(f"{s} = {ctx.lf(v)}")
            yield ("len", (name, round(want, 6)), pre, f.rule, (f.key,))


# --- 平角 -----------------------------------------------------------

def _r_straight(kb: KB, ctx: Ctx):
    """平角：A、X、B 共线，则 ∠CXA + ∠CXB = 180°。

    顶点必须是中间那个点 X——这才是「两条射线反向」。拿端点当顶点，
    X 和另一端在同一射线上，两个角相等而不互补（那是「同一条射线」的事）。
    早期这里写成 `v in (x, a, b): continue`，把唯一正确的情形也挡掉了，
    于是「∠ABC = 65° ⇒ ∠PBC = 115°」这一步永远走不出来。
    """
    for x, a, b in ctx.between:
        for c in ctx.X:
            if c in (x, a, b) or not ctx.linked(x, c):
                continue
            t1, t2 = _ang(x, c, a), _ang(x, c, b)
            if t1 == t2 or abs(ctx.d(t1) + ctx.d(t2) - 180) > TOL_D:
                continue
            yield ("angsum", (t1, t2, 180.0),
                   [f"{a}、{x}、{b} 在同一条直线上（{x} 在 {a}{b} 上）"],
                   "平角为 180°", ())


# ---------------------------------------------------------------- 初始事实


def _seed(p: Problem, ctx: Ctx, kb: KB, unit_seg: tuple[str, str] | None) -> None:
    """把题面与图形结构翻成事实。"""
    known = _lengths(p)

    # 题目给了长度：直接用
    for name, v in known.items():
        if len(name) == 2 and all(x in ctx.X for x in name):
            kb.add("len", (_seg(name[0], name[1]), float(v)),
                   [f"{name} = {_num(v)}（已知）"], "已知")

    # 一个长度都没给：把第一条边定为 a
    if not known and unit_seg and all(x in ctx.X for x in unit_seg):
        a, b = unit_seg
        kb.add("len", (_seg(a, b), 1.0), [f"设 {a}{b} = a"], "设元")

    # 直角
    for m in p.marks:
        if m.type == "right_angle" and m.at in ctx.X:
            ofs = [x for x in m.of if len(x) == 2 and m.at in x]
            if len(ofs) < 2:
                continue
            u = ofs[0].replace(m.at, "")
            w = ofs[1].replace(m.at, "")
            if u in ctx.X and w in ctx.X:
                kb.add("right", (_ang(m.at, u, w),),
                       [f"∠{u}{m.at}{w} = 90°（已知）"], "已知")

    # 题干给的角度
    for name, v in _angles_given(p):
        if len(name) == 3 and all(x in ctx.X for x in name):
            t = _ang(name[1], name[0], name[2])   # 「∠ABC」的顶点是 B
            if abs(v - 90) < 0.5:
                kb.add("right", (t,), [f"∠{name} = 90°（已知）"], "已知")
            else:
                kb.add("angval", (t, float(v)), [f"∠{name} = {_num(v)}°（已知）"], "已知")
        elif len(name) == 1 and name in ctx.X:
            arms = sorted(x for x in ctx.X if x != name and ctx.linked(name, x))
            if len(arms) >= 2:
                # 顶点就是 name。写成 _ang(arms[0], name, arms[1]) 会把顶点搬到
                # arms[0] 上——「∠A = 25°」被记成 ∠ABC，整道题的角度全错位。
                kb.add("angval", (_ang(name, arms[0], arms[1]), float(v)),
                       [f"∠{name} = {_num(v)}°（已知）"], "已知")

    # 等长、平行、中点
    for m in p.marks:
        if m.type == "equal_length" and len(m.of) >= 2:
            a, b = m.of[0], m.of[1]
            if len(a) == 2 and len(b) == 2 and set(a) <= set(ctx.X) and set(b) <= set(ctx.X):
                kb.add("eqseg", (_seg(*a), _seg(*b)), [f"{a} = {b}（已知）"], "已知")
        elif m.type == "parallel" and len(m.of) >= 2:
            a, b = m.of[0], m.of[1]
            if len(a) == 2 and len(b) == 2 and set(a) <= set(ctx.X) and set(b) <= set(ctx.X):
                kb.add("para", ((a[0], a[1]), (b[0], b[1])), [f"{a} ∥ {b}（已知）"], "已知")
        elif m.type == "midpoint" and m.at in ctx.X and m.of:
            s = m.of[0]
            if len(s) == 2 and set(s) <= set(ctx.X):
                kb.add("mid", (m.at, _seg(*s)), [f"{m.at} 是 {s} 的中点（已知）"], "已知")

    # 共线（三点组）
    for g in ctx.groups:
        if len(g) == 3:
            xs = tuple(sorted(g))
            kb.add("coll", (xs,), [f"{'、'.join(xs)} 共线"], "已知")

    # 圆：圆心、圆周上的点、半径、直径、切线。这几条是圆的全部起点——
    # 圆的定理都长在「谁在圆上」「谁是直径」「谁是切线」上，缺一个就无从下手。
    for c in p.circles:
        o = c.center
        if not o or o not in ctx.X:
            continue
        ring = [x for x in dict.fromkeys(c.through or [])
                if x in ctx.X and x != o]
        if len(ring) < 2:
            continue
        kb.add("center", (o,), [f"{o} 是圆心"], "已知")
        for x in ring:
            kb.add("on_circ", (x, o), [f"{x} 在 ⊙{o} 上"], "已知")
        # 同圆半径相等：这一条不写，「等腰三角形 OBC」就永远凑不出来
        r0 = ring[0]
        for x in ring[1:]:
            kb.add("eqseg", (_seg(o, r0), _seg(o, x)),
                   [f"{o}{r0}、{o}{x} 都是 ⊙{o} 的半径"], "同圆半径相等")
        # 直径：圆心落在两点之间
        for x, y in itertools.combinations(ring, 2):
            if _between_pts(ctx, o, x, y):
                kb.add("diam", (x, y, o), [f"{x}{y} 经过圆心 {o}"], "直径的定义")

    for tp, o, far in tangent_pairs(p):
        if all(x in ctx.X for x in (tp, o, far)):
            kb.add("tang", (tp, o, far),
                   [f"{tp}{far} 是 ⊙{o} 在点 {tp} 处的切线"], "已知")


# ---------------------------------------------------------------- 求解


@dataclass
class Aux:
    """一条（或一组）候选辅助线。

    连线型的辅助线只是往图上加边；作垂线、取中点要造新点，新点的坐标
    由几何作图直接算出来（垂足就是投影、中点就是平均），所以它和原图上
    读到的点一样能参与验算——这正是敢把辅助线写进答案的前提。
    """

    desc: str = ""                                  # 「连接 AE」「过 P 作 PH ⊥ AB 于 H」
    segs: tuple = ()                                # 新增线段
    pts: dict = field(default_factory=dict)         # 新点 → 坐标
    rels: tuple = ()                                # 要补进题面的位置关系
    rights: tuple = ()                              # (顶点, 臂1, 臂2)：作图作出的直角
    mids: tuple = ()                                # (点, "AB")：作图作出的中点
    facts: tuple = ()                               # ((kind, args), …)：作图直接给出的关系


@dataclass
class Proof:
    ok: bool = False
    steps: list[tuple[str, list[str], str, str]] = field(default_factory=list)
    goal: str = ""
    note: str = ""
    unit: float | None = None
    unit_name: str = ""
    aux: str = ""       # 靠哪条辅助线走通的（没用到就是空）


def _trace(kb: KB, goal: tuple) -> list[Fact]:
    """回溯目标依赖，拓扑排序。"""
    seen: dict[tuple, Fact] = {}
    order: list[Fact] = []

    def walk(k: tuple) -> None:
        if k in seen or k not in kb.f:
            return
        f = kb.f[k]
        seen[k] = f
        for d in f.deps:
            walk(d)
        order.append(f)

    walk(goal)
    return order


def _find_goal(kb: KB, sol: GeoSolution) -> tuple | None:
    """把题目要求翻译成要查的事实 key。"""
    if sol.goal_kind == "prove" and sol.claim_edges:
        a, b = sol.claim_edges
        s1, s2 = _seg(a[0], a[1]), _seg(b[0], b[1])
        for cand in ((s1, s2), (s2, s1)):
            if kb.has("eqseg", cand):
                return ("eqseg", _norm("eqseg", cand))
        return None

    # 「求证 △ABE ≌ △BCF」：结论本身就是一个全等事实
    if sol.goal_kind == "prove" and sol.claim_name:
        m = re.match(r"△([A-Z]{3})\s*≌\s*△([A-Z]{3})", sol.claim_name)
        if m:
            want = _canon_pair(tuple(m.group(1)), tuple(m.group(2)))
            return ("cong", want) if kb.has("cong", want) else None
        m = re.match(r"△([A-Z]{3})\s*∽\s*△([A-Z]{3})", sol.claim_name)
        if m:
            want = _canon_pair(tuple(m.group(1)), tuple(m.group(2)))
            for f in kb.of("sim"):
                if f.args[:2] == want:
                    return f.key
            return None

        # 「求证 ∠ADB = ∠ADC」：结论是一条 eqang
        m = re.match(r"∠([A-Z]{3})\s*=\s*∠([A-Z]{3})", sol.claim_name)
        if m:
            t1 = _ang(m.group(1)[1], m.group(1)[0], m.group(1)[2])
            t2 = _ang(m.group(2)[1], m.group(2)[0], m.group(2)[2])
            if kb.has("eqang", (t1, t2)):
                return ("eqang", _norm("eqang", (t1, t2)))
            return None

        # 「求证 AD ⊥ BC」：就是证交角为 90°，落到 right 或 angval 上
        m = re.match(r"([A-Z]{2})\s*⊥\s*([A-Z]{2})", sol.claim_name)
        if m:
            a, b = m.group(1), m.group(2)
            common = set(a) & set(b)
            if len(common) == 1:
                v = common.pop()
                u = a.replace(v, "")
                w = b.replace(v, "")
                t = _ang(v, u, w)
                if kb.has("right", (t,)):
                    return ("right", (t,))
                for f in kb.of("angval"):
                    if f.args[0] == t and abs(f.args[1] - 90) < TOL_D:
                        return f.key
            return None

        m = re.match(r"([A-Z]{2})\s*∥\s*([A-Z]{2})", sol.claim_name)
        if m:
            args = _norm("para", ((m.group(1)[0], m.group(1)[1]),
                                  (m.group(2)[0], m.group(2)[1])))
            return ("para", args) if kb.has("para", args) else None

    if sol.goal_kind == "length" and sol.goal_name and len(sol.goal_name) == 2:
        name = _seg(sol.goal_name[0], sol.goal_name[1])
        for f in kb.of("len"):
            if f.args[0] == name:
                return f.key
        return None

    if sol.goal_kind == "angle" and sol.goal_name:
        name = sol.goal_name
        if len(name) == 3:
            # 「∠ACB」的顶点是中间那个字母
            t = _ang(name[1], name[0], name[2])
        elif sol.goal_arms:
            # goal_arms 是两条边（臂），顶点是 name 本身。写成
            # _ang(arms[0], name, arms[1]) 会把顶点搬到 arms[0] 上——
            # 求 ∠P 变成求 ∠PCB，永远查不到，整条推理白跑。
            t = _ang(name, sol.goal_arms[0], sol.goal_arms[1])
        else:
            return None
        for f in kb.of("angval"):
            if f.args[0] == t:
                return f.key
    return None


def _fact_holds(ctx: Ctx, kind: str, args: tuple) -> bool:
    """辅助线声称的关系，先在解出的图形上量一遍。

    作图造出来的点，坐标是算出来的（投影、平均、旋转、求交），所以它们
    和图上读到的点地位相同——但「算出来」不等于「对」：旋转角写成 90、
    实际该是 60，得到的全等就是假的。这条函数是辅助线的验算关。
    """
    try:
        if kind == "eqseg":
            a, b = args
            d1, d2 = ctx.dist(a[0], a[1]), ctx.dist(b[0], b[1])
            return abs(d1 - d2) / max(d1, d2, 1e-9) < 2e-3
        if kind == "eqang":
            t1, t2 = args
            return abs(ctx.d(t1) - ctx.d(t2)) < TOL_D
        if kind == "angval":
            return abs(ctx.d(args[0]) - args[1]) < TOL_D
        if kind == "right":
            return abs(ctx.d(args[0]) - 90) < TOL_D
        if kind == "para":
            (a, b), (c, d) = args
            ux, uy = ctx.X[b][0] - ctx.X[a][0], ctx.X[b][1] - ctx.X[a][1]
            wx, wy = ctx.X[d][0] - ctx.X[c][0], ctx.X[d][1] - ctx.X[c][1]
            nu, nw = math.hypot(ux, uy), math.hypot(wx, wy)
            if nu < 1e-9 or nw < 1e-9:
                return False
            return abs(ux * wy - uy * wx) / (nu * nw) < 5e-3
        if kind == "cong":
            return _cong_holds(ctx, args[0], args[1])
        if kind == "mid":
            m, seg = args
            a, b = seg[0], seg[1]
            d1, d2 = ctx.dist(a, m), ctx.dist(m, b)
            return abs(d1 - d2) / max(d1, d2, 1e-9) < 2e-3
    except (KeyError, IndexError, TypeError):
        return False
    return True


def _attempt(p: Problem, sol: GeoSolution, between: list | None,
             aux: Aux | None = None) -> Proof:
    """跑一遍完整推理。aux 是临时补上的辅助线。"""
    if not sol.coords:
        return Proof(note="没有图形坐标，无法验算推理")

    # 读图只给图上画了的东西；「四边形 ABCD 为正方形」这种整句声明得靠 enrich
    # 翻成直角和平行，否则推理连起步的结构都没有。证明题的结论本身要从条件里
    # 剔掉——拿结论当已知，什么都能"证"出来。
    from .solve import enrich_from_text, parse_claim

    claim = parse_claim(p)
    skip = ({frozenset({claim.a, claim.b})}
            if claim and claim.kind == "len_equal" else set())
    p, _ = enrich_from_text(p, skip_equal=skip)

    X = dict(sol.coords)
    if aux:
        import copy

        q = copy.deepcopy(p)
        for name, xy in aux.pts.items():
            if name not in q.points:
                q.points.append(name)
            X[name] = xy
        have = {tuple(sorted(s)) for s in q.segments}
        for a, b in aux.segs:
            if tuple(sorted((a, b))) not in have:
                q.segments.append((a, b))
                have.add(tuple(sorted((a, b))))
        for r in aux.rels:
            if r not in q.relations:
                q.relations.append(r)
        p = q
        # 新点改变了共线与「在…上」的关系，缓存下来的 between 已经过期
        between = None

    if between is None:
        from . import relations
        between = list(relations.derive(p, skip_equal=skip).between)
        # 圆心落在直径上、P 落在延长线上：这两类 between 只在圆和射线的
        # 语境里成立，relations 不管，但「平角」「同射线」两条规则全靠它。
        seen_b = {tuple(b) for b in between}
        for c in p.circles:
            o = c.center
            ring = [x for x in dict.fromkeys(c.through or [])
                    if x in X and x != o]
            for x, y in itertools.combinations(ring, 2):
                if (o, x, y) not in seen_b and _between_num(X, o, x, y):
                    between.append((o, x, y))
                    seen_b.add((o, x, y))
        for tp, ta, tb in ray_targets(p):
            if all(k in X for k in (tp, ta, tb)) and (tb, ta, tp) not in seen_b:
                between.append((tb, ta, tp))
                seen_b.add((tb, ta, tp))

    known = _lengths(p)
    unit_seg = None
    for a, b in p.segments:
        if a in X and b in X:
            unit_seg = (a, b)
            break

    unit = None
    unit_name = ""
    if not known and unit_seg:
        s = math.dist(X[unit_seg[0]], X[unit_seg[1]])
        if s > 1e-9:
            X = {k: (v[0] / s, v[1] / s) for k, v in X.items()}
            unit = 1.0
            unit_name = f"{unit_seg[0]}{unit_seg[1]}"

    ctx = Ctx(p, X, between)
    ctx.unit = unit            # 题目有具体长度时不符号化，直接写数值
    kb = KB()
    _seed(p, ctx, kb, unit_seg)

    # 作图作出的直角、中点：和读图读到的地位一样，都是条件
    if aux:
        # 但「画上去」不等于「成立」：公式里写错一个符号，得到的等长、
        # 平行全是假的，而且它们会被当成出发条件，让整条链看起来天衣无缝。
        # 所以作图给出的关系也要先在解出的图形上量一遍才准进事实库。
        for kind, args in aux.facts:
            if not _fact_holds(ctx, kind, args):
                return Proof(note=f"辅助线「{aux.desc}」作出的关系与图形不符，已丢弃")
        for v, a, b in aux.rights:
            kb.add("right", (_ang(v, a, b),), [aux.desc], "作图（作垂线）")
        for m, seg in aux.mids:
            kb.add("mid", (m, seg), [aux.desc], "作图（取中点）")
        for kind, args in aux.facts:
            kb.add(kind, args, [aux.desc], "作图")

    for _ in range(14):
        new = 0
        for kind, args, pre, rule, deps in _rules(kb, ctx):
            if kb.add(kind, args, pre, rule, deps):
                new += 1
        if not new:
            break

    goal = _find_goal(kb, sol)
    if goal is None:
        return Proof(note="推理链没有走到结论")

    chain = _trace(kb, goal)
    if not chain:
        return Proof(note="找不到通往结论的推理链")

    pr = Proof(ok=True, unit=unit, unit_name=unit_name)
    for f in chain:
        pr.steps.append((f.kind, list(f.pre), _concl(f, unit), f.rule))
    # 整条链一个长度都没用到，就别在开头声明「长度都用 a 表示」——
    # 求角度的题看到这句会莫名其妙。
    if not any(f.kind == "len" for f in chain):
        pr.unit_name = ""
    # 结论按题目的写法给：问「△ADE ∽ △ABC」就写这个顺序，别自作主张改成
    # 规范写法——学生对着题看，字母顺序变了会以为答的不是同一件事。
    if sol.goal_kind == "prove" and sol.claim_name:
        pr.goal = sol.claim_name
    elif (sol.goal_kind == "length" and sol.goal_name
          and len(sol.goal_name) == 2 and kb.f[goal].kind == "len"):
        # 求 PB 就写 PB，别写库里排好序的 BP
        pr.goal = f"{sol.goal_name} = {_len_expr(kb.f[goal].args[1], unit)}"
    else:
        pr.goal = _concl(kb.f[goal], unit)
    return pr


def _focus_pts(p: Problem, sol: GeoSolution) -> set[str]:
    """跟目标有关的点。辅助线先往这几个点上试。"""
    focus: set[str] = set()
    if sol.claim_edges:
        focus |= set(sol.claim_edges[0]) | set(sol.claim_edges[1])
    if sol.goal_name and len(sol.goal_name) == 2:
        focus |= set(sol.goal_name)
    # 「求 ∠APB」是三个字母：三个点都跟目标有关，别只认线段名那种两字母写法
    if sol.goal_name and len(sol.goal_name) == 3:
        focus |= set(sol.goal_name)
    # 「求证 △ABE ≌ △BCF」时，该连的线一定落在这六个字母之间
    m = re.match(r"△([A-Z]{3})\s*[≌∽]\s*△([A-Z]{3})", sol.claim_name or "")
    if m:
        focus |= set(m.group(1)) | set(m.group(2))
    return focus


def _aux_lines(p: Problem, sol: GeoSolution, limit: int = 6) -> list[tuple[str, str]]:
    """「连接两点」——最常用的一类辅助线。

    共线的两点不试：连上去不产生任何新三角形，白跑一趟。与目标有关的点先试。
    """
    if not sol.coords:
        return []
    pts = [x for x in p.points if x in sol.coords]
    have = {frozenset(s) for s in p.segments}
    groups = collinear_groups(p)
    focus = _focus_pts(p, sol)

    cands: list[tuple[int, float, str, str]] = []
    for a, b in itertools.combinations(pts, 2):
        if frozenset((a, b)) in have:
            continue
        if any(a in g and b in g for g in groups):
            continue
        score = -len(focus & {a, b})
        cands.append((score, math.dist(sol.coords[a], sol.coords[b]), a, b))
    cands.sort()
    return [(a, b) for _, _, a, b in cands[:limit]]


_FREE_NAMES = "HKMNGSTUVWXYZ"


def _free_name(taken: set[str]) -> str:
    for ch in _FREE_NAMES:
        if ch not in taken:
            return ch
    return ""


def _near_any(X: dict, xy: tuple[float, float], taken: set[str],
              scale: float, tol: float = 0.05) -> bool:
    return any(q in X and math.dist(xy, X[q]) < tol * scale for q in taken)


def _span(X: dict) -> float:
    if not X:
        return 1e-6
    xs = [v[0] for v in X.values()]
    ys = [v[1] for v in X.values()]
    return max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)


def _aux_foot(p: Problem, sol: GeoSolution, limit: int = 3) -> list[Aux]:
    """作垂线：过一点向一条边作垂线，垂足是新点。

    「作高」是几何题里出现最多、也最容易被跳过的一步——面积、勾股、
    等腰的底角，都要先有那条高。垂足的坐标就是投影，算得出来，所以
    由此得到的直角和长度照样能验算。
    """
    X = sol.coords
    scale = _span(X)
    pts = [x for x in p.points if x in X]
    focus = _focus_pts(p, sol)

    cands: list[tuple[int, float, str, str, str, tuple[float, float]]] = []
    for pt in pts:
        for a, b in p.segments:
            if pt in (a, b) or a not in X or b not in X:
                continue
            (ax, ay), (bx, by), (px, py) = X[a], X[b], X[pt]
            ux, uy = bx - ax, by - ay
            L2 = ux * ux + uy * uy
            if L2 < 1e-12:
                continue
            t = ((px - ax) * ux + (py - ay) * uy) / L2
            # 垂足落在端点附近等于没作：那条「高」本来就画在图上了
            if not (0.12 < t < 0.88):
                continue
            foot = (ax + t * ux, ay + t * uy)
            if _near_any(X, foot, set(X), scale):
                continue
            score = -len(focus & {pt, a, b})
            cands.append((score, math.dist(foot, X[pt]), pt, a, b, foot))

    cands.sort()
    out: list[Aux] = []
    used = set(p.points)
    for _, _, pt, a, b, foot in cands:
        if len(out) >= limit:
            break
        if _near_any(X, foot, used, scale):
            continue                    # 同一个垂足只作一次
        h = _free_name(used)
        if not h:
            break
        used.add(h)
        out.append(Aux(
            desc=f"过 {pt} 作 {h} ⊥ {a}{b} 于 {h}",
            pts={h: foot},
            segs=((pt, h),),
            rels=(f"{h}在{a}{b}上",),
            rights=((h, pt, a),),
        ))
    return out


def _aux_mid(p: Problem, sol: GeoSolution, limit: int = 2) -> list[Aux]:
    """取中点：给三角形的一条边取中点，连成中线。

    中线值钱在于「斜边中线等于斜边一半」和等腰三角形里的三线合一。
    只在三个点真的连成三角形时才造——否则中点连出去什么也拼不出来。
    """
    X = sol.coords
    scale = _span(X)
    segset = {tuple(sorted(s)) for s in p.segments}
    groups = collinear_groups(p)
    focus = _focus_pts(p, sol)

    cands: list[tuple[int, float, str, str, str]] = []
    for a, b in segset:
        if a not in X or b not in X:
            continue
        m = ((X[a][0] + X[b][0]) / 2, (X[a][1] + X[b][1]) / 2)
        if _near_any(X, m, set(X), scale):
            continue                    # 中点本来就是图上的点，不用再造
        for c in p.points:
            if c in (a, b) or c not in X:
                continue
            if any(a in g and b in g and c in g for g in groups):
                continue
            if (tuple(sorted((a, c))) not in segset
                    or tuple(sorted((b, c))) not in segset):
                continue
            score = -len(focus & {a, b, c})
            cands.append((score, math.dist(X[a], X[b]), a, b, c))

    cands.sort()
    out: list[Aux] = []
    used = set(p.points)
    for _, _, a, b, c in cands:
        if len(out) >= limit:
            break
        m = ((X[a][0] + X[b][0]) / 2, (X[a][1] + X[b][1]) / 2)
        if _near_any(X, m, used, scale):
            continue                    # 同一个中点只造一次
        n = _free_name(used)
        if not n:
            break
        used.add(n)
        out.append(Aux(
            desc=f"取 {a}{b} 的中点 {n}，连接 {c}{n}",
            pts={n: m},
            segs=((c, n),),
            rels=(f"{n}在{a}{b}上",),
            mids=((n, _seg(a, b)),),
        ))
    return out


def _aux_extend(p: Problem, sol: GeoSolution, limit: int = 2) -> list[Aux]:
    """延长：把一条线延长出去，与另一条线相交于新点。

    「延长 AE 交 CD 于 G」几乎每道需要构造相似/全等的题都有。交点坐标
    由两条直线求交直接得到，所以连出来的角、算出来的比例都能验算。
    """
    X = sol.coords
    scale = _span(X)
    segs = [(a, b) for a, b in p.segments if a in X and b in X]
    focus = _focus_pts(p, sol)

    cands: list[tuple[int, float, str, str, str, str, float, float]] = []
    for (a, b), (c, d) in itertools.combinations(segs, 2):
        if len({a, b, c, d}) < 4:
            continue
        (x1, y1), (x2, y2) = X[a], X[b]
        (x3, y3), (x4, y4) = X[c], X[d]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6 * scale * scale:
            continue                     # 平行，没有交点
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
        if _near_any(X, (px, py), set(X), scale):
            continue                     # 交点已经是图上的点
        ta = _param((x1, y1), (x2, y2), (px, py))
        tc = _param((x3, y3), (x4, y4), (px, py))
        if 0 <= ta <= 1 and 0 <= tc <= 1:
            continue                     # 两条边本来就在图里相交，图早画了
        if not (-1.5 <= ta <= 2.5 and -1.5 <= tc <= 2.5):
            continue                     # 跑到很远的地方，多半不是这道题要的延长
        score = -len(focus & {a, b, c, d})
        cands.append((score, math.dist((px, py), X[a]), a, b, c, d, px, py))

    cands.sort()
    out: list[Aux] = []
    used = set(p.points)
    for _, _, a, b, c, d, px, py in cands:
        if len(out) >= limit:
            break
        g = _free_name(used)
        if not g:
            break
        used.add(g)
        out.append(Aux(
            desc=f"延长 {a}{b} 与 {c}{d} 相交于 {g}",
            pts={g: (px, py)},
            segs=((a, g),),
            rels=(f"{g}在{a}{b}的延长线上",),
        ))
    return out


def _param(p0: tuple[float, float], p1: tuple[float, float],
           q: tuple[float, float]) -> float:
    """q 在直线 p0p1 上的参数：0 是 p0，1 是 p1。"""
    ux, uy = p1[0] - p0[0], p1[1] - p0[1]
    L2 = ux * ux + uy * uy
    if L2 < 1e-12:
        return 0.0
    return ((q[0] - p0[0]) * ux + (q[1] - p0[1]) * uy) / L2


def _cross(a: tuple[float, float], b: tuple[float, float],
           c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _meet(p0, u, p1, w):
    """直线 p0+t·u 与 p1+s·w 的交点，返回 (t, s, 点) 或 None。

    解 p0 + t·u = p1 + s·w。写成矩阵就是 [u, -w]·(t, s)ᵀ = p1-p0，行列式是
    -D（D 是 u、w 的叉积）。s 前面那个负号不能丢：丢了以后交点算得对，
    参数却是反的，于是「交点是否落在这条边上」每次判断都拿反了的 s 去比。
    """
    D = u[0] * w[1] - u[1] * w[0]
    if abs(D) < 1e-9:
        return None
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    t = (vx * w[1] - vy * w[0]) / D
    s = -(u[0] * vy - u[1] * vx) / D
    return t, s, (p0[0] + t * u[0], p0[1] + t * u[1])


def _rot_xy(xy, o, deg: float):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    dx, dy = xy[0] - o[0], xy[1] - o[1]
    return (o[0] + c * dx - s * dy, o[1] + s * dx + c * dy)


def _tris_of(p: Problem, X: dict) -> list[tuple[str, str, str]]:
    """图上真正画出来的三角形（三条边都在、三点不共线）。"""
    pts = [x for x in p.points if x in X]
    segset = {_seg(a, b) for a, b in p.segments if a in X and b in X}
    scale = _span({k: X[k] for k in pts}) if pts else 1.0
    out = []
    for a, b, c in itertools.combinations(pts, 3):
        if not all(_seg(x, y) in segset
                   for x, y in ((a, b), (b, c), (a, c))):
            continue
        if abs(_cross(X[a], X[b], X[c])) / (scale * scale) < 2e-3:
            continue                      # 共线，不是三角形
        out.append((a, b, c))
    return out


def _aux_parallel(p: Problem, sol: GeoSolution, limit: int = 2) -> list[Aux]:
    """作平行线：过一点作某条边的平行线，交另一条边于新点。

    平行线是构造相似最省事的办法——画出来立刻有两组同位角，AA 判定
    就有了入口，而相似比一带，长度全都跟着出来。新点是「过 P 的平行线」
    与那条边的交点，坐标是两直线求交，所以照样能验算。
    """
    X = sol.coords
    scale = _span(X)
    pts = [x for x in p.points if x in X]
    segs = [(a, b) for a, b in p.segments if a in X and b in X]
    focus = _focus_pts(p, sol)

    cands: list[tuple[int, float, str, str, str, str, tuple]] = []
    for pt in pts:
        for a, b in segs:
            if pt in (a, b):
                continue
            u = (X[b][0] - X[a][0], X[b][1] - X[a][1])
            nu = math.hypot(*u)
            if nu < 1e-9:
                continue
            for c, d in segs:
                if a in (c, d) and b in (c, d):
                    continue                  # 同一条边
                if pt in (c, d):
                    continue
                w = (X[d][0] - X[c][0], X[d][1] - X[c][1])
                got = _meet(X[pt], u, X[c], w)
                if got is None:
                    continue                  # 与目标边平行，交不出来
                t, s, q = got
                if abs(t) < 1e-6:
                    continue
                if not (0.08 < s < 0.92):
                    continue                  # 交点得真落在这条边上
                if _near_any(X, q, set(X), scale):
                    continue                  # 交点已经是图上的点
                score = -len(focus & {pt, a, b, c, d})
                cands.append((score, abs(t), pt, a, b, c + d, q))

    cands.sort()
    out: list[Aux] = []
    used = set(p.points)
    for _, _, pt, a, b, cd, q in cands:
        if len(out) >= limit:
            break
        if _near_any(X, q, used, scale):
            continue
        n = _free_name(used)
        if not n:
            break
        used.add(n)
        out.append(Aux(
            desc=f"过 {pt} 作 {pt}{n} ∥ {a}{b}，交 {cd} 于 {n}",
            pts={n: q},
            segs=((pt, n),),
            rels=(f"{n}在{cd}上",),
            facts=(("para", ((pt, n), (a, b))),),
        ))
    return out


def _aux_bisector(p: Problem, sol: GeoSolution, limit: int = 2) -> list[Aux]:
    """作角平分线：从顶点引一条平分已知角的射线，交对边于新点。

    角平分线一次给出两个相等的角，等腰三角形、内心、角平分线上的点到
    两边等距，都是从这一对等角长出来的。方向取两臂单位向量之和——落在
    角内部，长度随意，和哪条边相交就取哪个交点。
    """
    X = sol.coords
    scale = _span(X)
    segs = [(a, b) for a, b in p.segments if a in X and b in X]
    segset = {_seg(a, b) for a, b in segs}
    focus = _focus_pts(p, sol)

    angles: list[tuple[str, str, str]] = []
    for v in p.points:
        if v not in X:
            continue
        arms = sorted(x for x in p.points
                      if x != v and x in X and _seg(v, x) in segset)
        for a, b in itertools.combinations(arms, 2):
            if _seg(a, b) not in segset:
                continue                       # 得是个真角，不是一条直线上的两段
            angles.append((v, a, b))

    cands: list[tuple[int, float, str, str, str, str, tuple]] = []
    for v, a, b in angles:
        ua = (X[a][0] - X[v][0], X[a][1] - X[v][1])
        ub = (X[b][0] - X[v][0], X[b][1] - X[v][1])
        na, nb = math.hypot(*ua), math.hypot(*ub)
        if na < 1e-9 or nb < 1e-9:
            continue
        u = (ua[0] / na + ub[0] / nb, ua[1] / na + ub[1] / nb)
        if math.hypot(*u) < 1e-6:
            continue                           # 平角，没有角平分线
        for c, d in segs:
            if v in (c, d):
                continue
            w = (X[d][0] - X[c][0], X[d][1] - X[c][1])
            got = _meet(X[v], u, X[c], w)
            if got is None:
                continue
            t, s, q = got
            if t <= 1e-6:
                continue                       # 交点必须在射线上
            if not (0.08 < s < 0.92):
                continue
            if _near_any(X, q, set(X), scale):
                continue
            score = -len(focus & {v, a, b, c, d})
            cands.append((score, t, v, a, b, c + d, q))

    cands.sort()
    out: list[Aux] = []
    used = set(p.points)
    for _, _, v, a, b, cd, q in cands:
        if len(out) >= limit:
            break
        if _near_any(X, q, used, scale):
            continue
        n = _free_name(used)
        if not n:
            break
        used.add(n)
        out.append(Aux(
            desc=f"作 {v}{n} 平分 ∠{a}{v}{b}，交 {cd} 于 {n}",
            pts={n: q},
            segs=((v, n),),
            rels=(f"{n}在{cd}上",),
            facts=(("eqang", (_ang(v, a, n), _ang(v, n, b))),),
        ))
    return out


def _aux_rotate(p: Problem, sol: GeoSolution, limit: int = 2) -> list[Aux]:
    """旋转：把一个三角形绕它自己的一个顶点转 60° 或 90°。

    旋转是「把分散的条件搬到一起」的标准手段。转完立刻多出一对全等
    三角形（旋转保距），角度、线段都跟着搬过去——正三角形里「PA、PB、PC
    三条散线」的题，全靠这一转凑成一个三角形。

    新点坐标就是真的绕那点转过去算的，所以「全等」是真的全等，能验算。
    如果转过去的点碰巧落在图上已有的点（正三角形绕顶点转 60°，第三个
    顶点就是现成的），就直接认用那个点，不另起名字。
    """
    X = sol.coords
    scale = _span(X)
    tris = _tris_of(p, X)
    focus = _focus_pts(p, sol)

    cands = []
    for tri in tris:
        for k in range(3):
            o = tri[k]
            u, w = tri[(k + 1) % 3], tri[(k + 2) % 3]
            if X[u] == X[o] or X[w] == X[o]:
                continue
            # 转过去如果正好是第三个顶点，等于没转出新东西
            for deg in (90, -90, 60, -60):
                u2 = _rot_xy(X[u], X[o], deg)
                w2 = _rot_xy(X[w], X[o], deg)
                nu = _hit(X, u2, scale)
                nw = _hit(X, w2, scale)
                if nu is not None and nw is not None:
                    continue
                if nu == o or nw == o:
                    continue
                score = -len(focus & set(tri)) - (nu is not None) - (nw is not None)
                cands.append((score, tri, o, u, w, deg, nu, nw, u2, w2))

    cands.sort(key=lambda c: (c[0], -abs(c[5])))
    out: list[Aux] = []
    used = set(p.points)
    for _, tri, o, u, w, deg, nu, nw, u2, w2 in cands:
        if len(out) >= limit:
            break
        pts: dict[str, tuple[float, float]] = {}
        names: dict[str, str] = {}
        for old, hit, xy in ((u, nu, u2), (w, nw, w2)):
            if hit is not None:
                names[old] = hit
                continue
            if _near_any(X, xy, used | set(pts), scale):
                names[old] = ""
                continue
            n = _free_name(used | set(pts))
            if not n:
                names[old] = ""
                continue
            pts[n] = xy
            names[old] = n
        if not pts:
            continue
        pairs = [(old, names[old]) for old in (u, w) if names[old]]
        if len(pairs) < 2:
            continue
        u_n, w_n = names[u], names[w]
        tri2 = _canon_pair((o, u, w), (o, u_n, w_n))
        facts: list[tuple] = [("cong", tri2)]
        segs = [(o, u_n), (u_n, w_n)]
        drawn = {_seg(a, b) for a, b in p.segments}
        for old, new in pairs:
            if _seg(old, new) not in drawn:
                segs.append((old, new))
            # 旋转角记绝对值：∠POP′ 量出来总是正的，写 -90 会被验算挡掉
            facts.append(("angval", (_ang(o, old, new), float(abs(deg)))))
        used |= set(pts)
        out.append(Aux(
            desc=f"把 △{o}{u}{w} 绕 {o} 旋转 {abs(deg)}° 得 △{o}{u_n}{w_n}",
            pts=pts,
            segs=tuple(segs),
            facts=tuple(facts),
        ))
    return out


def _hit(X: dict, xy: tuple[float, float], scale: float) -> str | None:
    """xy 落在哪个已有名字上（旋转后对上现成的点）。"""
    for k, v in X.items():
        if math.dist(xy, v) < 1e-3 * scale:
            return k
    return None


def _aux_reflect(p: Problem, sol: GeoSolution, limit: int = 1) -> list[Aux]:
    """翻折：把一点关于一条直线对称过去（折痕就是这条线）。

    折叠问题的标准做法：折痕垂直平分对应点的连线。对称点的坐标就是
    「投影再翻一倍」，所以「垂直」「平分」这两条是算出来的，不是声明的。
    """
    X = sol.coords
    scale = _span(X)
    pts = [x for x in p.points if x in X]
    segs = [(a, b) for a, b in p.segments if a in X and b in X]
    focus = _focus_pts(p, sol)

    cands: list[tuple[int, float, str, str, str, tuple, tuple, str | None]] = []
    for pt in pts:
        for a, b in segs:
            if pt in (a, b):
                continue
            (ax, ay), (bx, by), (px, py) = X[a], X[b], X[pt]
            ux, uy = bx - ax, by - ay
            L2 = ux * ux + uy * uy
            if L2 < 1e-12:
                continue
            t = ((px - ax) * ux + (py - ay) * uy) / L2
            foot = (ax + t * ux, ay + t * uy)
            if abs(t) < 1e-6 or abs(t - 1) < 1e-6:
                continue                     # 点就在端点上，翻过去没意义
            if not (-0.3 < t < 1.3):
                continue                     # 折痕够不着，不是这道题的折法
            img = (2 * foot[0] - px, 2 * foot[1] - py)
            if _near_any(X, img, set(X), scale):
                continue                     # 翻过去的点本来就在图上
            h = _hit(X, foot, scale)
            score = -len(focus & {pt, a, b})
            cands.append((score, math.dist(img, X[pt]), pt, a, b, img, foot, h))

    cands.sort(key=lambda c: (c[0], c[1]))
    out: list[Aux] = []
    for _, _, pt, a, b, img, foot, h in cands:
        if len(out) >= limit:
            break
        names_used = set(p.points)
        pts: dict[str, tuple[float, float]] = {}
        if h is None:                      # 垂足不是现成的点，就得造一个
            h = _free_name(names_used)
            if not h:
                continue
            names_used.add(h)
            pts[h] = foot
        n = _free_name(names_used)
        if not n:
            continue
        pts[n] = img
        out.append(Aux(
            desc=f"把 {pt} 沿 {a}{b} 翻折到 {n}，{a}{b} 垂直平分 {pt}{n}",
            pts=pts,
            segs=((pt, n),),
            rels=(f"{h}在{a}{b}上",),
            facts=(("right", (_ang(h, pt, a),)), ("mid", (h, _seg(pt, n)))),
        ))
    return out


def _aux_plans(p: Problem, sol: GeoSolution) -> list[Aux]:
    """候选辅助线，按「值不值得试」排。

    先连线（最便宜，端点都在图上），再作垂线，再取中点，最后才是延长——
    延长造出来的点最不确定，放在最后。
    """
    out: list[Aux] = []
    if not sol.coords or len(sol.coords) < 3:
        return out
    lines = _aux_lines(p, sol)
    out += [Aux(desc=f"连接 {a}{b}", segs=((a, b),)) for a, b in lines]
    # 成对连：「连接 AC、BD」这种对角线组合，只连一条往往凑不出全等
    for (a1, b1), (a2, b2) in itertools.combinations(lines[:4], 2):
        out.append(Aux(desc=f"连接 {a1}{b1}、{a2}{b2}",
                       segs=((a1, b1), (a2, b2))))
    out += _aux_foot(p, sol)
    out += _aux_mid(p, sol)
    out += _aux_parallel(p, sol)
    out += _aux_bisector(p, sol)
    out += _aux_extend(p, sol)
    # 旋转、翻折放在最后：它们给出的结论最"强"，也最不该在简单题上抢戏
    out += _aux_rotate(p, sol)
    out += _aux_reflect(p, sol)
    return out


def prove(p: Problem, sol: GeoSolution, between: list | None = None,
          aux: bool = True) -> Proof:
    """尝试用定理推一遍。推不出来就返回 ok=False，不硬编。

    推不动时先别急着认输：几何题卡住十有八九是「少了一条辅助线」。
    把候选辅助线挨个接上再推一遍，哪条能通就报哪条——因为每个结论都
    在解出的图形上验算过，所以这条线不是猜的，是真能走通的那一条。
    """
    pr = _attempt(p, sol, between)
    if pr.ok or not aux:
        return pr

    for plan in _aux_plans(p, sol):
        pr2 = _attempt(p, sol, None, aux=plan)
        if pr2.ok and pr2.steps:
            pr2.aux = plan.desc
            return pr2
    return pr



def _concl(f: Fact, unit: float | None) -> str:
    k, a = f.kind, f.args
    if k == "eqseg":
        return f"{a[0]} = {a[1]}"
    if k == "eqang":
        return f"∠{_ang_name(a[0])} = ∠{_ang_name(a[1])}"
    if k == "cong":
        return f"△{''.join(a[0])} ≌ △{''.join(a[1])}"
    if k == "sim":
        return f"△{''.join(a[0])} ∽ △{''.join(a[1])}"
    if k == "angval":
        return f"∠{_ang_name(a[0])} = {_num(a[1])}°"
    if k == "len":
        return f"{a[0]} = {_len_expr(a[1], unit)}"
    if k == "right":
        return f"∠{_ang_name(a[0])} = 90°"
    if k == "para":
        return f"{_lname(a[0])} ∥ {_lname(a[1])}"
    if k == "coll":
        return f"{'、'.join(a[0])} 共线"
    if k == "mid":
        return f"{a[0]} 是 {a[1]} 的中点"
    if k == "diam":
        return f"{a[0]}{a[1]} 是 ⊙{a[2]} 的直径"
    if k == "tang":
        return f"{a[0]}{a[2]} 是 ⊙{a[1]} 的切线"
    if k == "on_circ":
        return f"{a[0]} 在 ⊙{a[1]} 上"
    if k == "center":
        return f"{a[0]} 是圆心"
    if k == "prod":
        return _prod_text(a)
    return k


def render(pr: Proof, sol: GeoSolution) -> str:
    """把推理链写成板书。"""
    out = ["### 解题过程", ""]
    head = ("下面用**几何推理**一步步推：先由条件找角，再由角找边，"
            "每一步都写明用了哪条定理。")
    if pr.unit_name:
        head += (f"题目没有给出具体长度，下面的长度都用 **a** 表示"
                 f"（a = {pr.unit_name}；结论与 a 取多大无关）。")
    out += [head, ""]
    if pr.aux:
        moved = ("旋转" in pr.aux) or ("翻折" in pr.aux)
        tag = "**辅助变换**" if moved else "**辅助线**"
        tail = ("把一个三角形搬到别处，是为了让散在各处的条件落到同一个三角形里，"
                "凑出可以直接判定的全等——旋转、翻折的题，多半就差这一步。"
                if moved else
                "图上本来没有这条线，添上它才凑得出可以判定、可以计算的三角形"
                "——几何题卡住时，多半就差这一步。")
        out += [f"> {tag}：{pr.aux}。{tail}", ""]

    n = 0
    sine_at = 0
    for kind, pre, concl, rule in pr.steps:
        # angsum 是「△CEF 的三个内角」这类桥梁事实，本身不是一步结论；
        # 「直角的定义」是 ∵∠ABC=90° ∴∠ABC=90°，同义反复；
        # 「设 AB=a」写在开头那句话里就够了；长度传递是枝节，结论用得到时会出现在 ∵ 里。
        if kind == "angsum" or not pre or rule in ("已知", "直角的定义", "设元"):
            continue
        # 圆心、圆周点、直径、切线是起手就成立的结构，不是推导出来的一步；
        # 写成「∵ BD 经过圆心 O ∴ BD 是直径」纯属凑行数。
        if kind in ("diam", "tang", "on_circ", "center"):
            continue
        if kind == "len" and rule == "等量代换":
            continue
        n += 1
        if rule == "正弦定理" and not sine_at:
            sine_at = n
        out.append(f"{n}. **∵** {'；'.join(pre)}")
        out.append(f"　　**∴** {concl}" + (f"（{rule}）" if rule else ""))
        out.append("")

    if pr.goal:
        out += ["#### 结论", "", f"**{pr.goal}**"]

    if sine_at:
        out += [
            "",
            f"> 第 {sine_at} 步用了正弦定理，属于高中内容。"
            "按初中要求可过 A 作 DE 的垂线，用直角三角形求解，结论相同。",
        ]
    return "\n".join(out)
