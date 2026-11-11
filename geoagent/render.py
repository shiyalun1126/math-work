"""把识别到的结构重画一遍。

这一步是整个几何链路里最容易被用户接受的校验手段：公式认错了要自己
对照字符串，图形认错了只要看一眼重绘图就知道少没少线、点错没错位。

两种画法：
  - 求解成功时，用解出来的真实坐标画，几何关系是准的
  - 解不出来时，退化成环形布局，只保证「点线连接关系」对
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

from .schema import Mark, Problem

BG = (252, 251, 248)
INK = (30, 30, 30)
POINT = (208, 74, 74)
CIRCLE = (60, 110, 170)
HILITE = (200, 40, 40)   # 目标量专用：只标「题目问的那个角/那条边」
SHADE = (255, 214, 204)  # 结论边的底衬，让红线和黑线叠在一起也分得开

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\arial.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _layout_ring(points: list[str], size: int, pad: int) -> dict[str, tuple[float, float]]:
    """没有坐标时的兜底布局：按点顺序摆在圆周上。"""
    n = max(1, len(points))
    r = (size - 2 * pad) / 2
    cx = cy = size / 2
    return {
        name: (cx + r * math.cos(-math.pi / 2 + 2 * math.pi * i / n),
               cy + r * math.sin(-math.pi / 2 + 2 * math.pi * i / n))
        for i, name in enumerate(points)
    }


def _fit(coords: dict[str, tuple[float, float]], size: int, pad: int) -> dict[str, tuple[float, float]]:
    """把坐标等比缩放并居中，同时翻转 y（数学 y 向上，图像 y 向下）。"""
    xs = [v[0] for v in coords.values()]
    ys = [v[1] for v in coords.values()]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    span = max(w, h, 1e-9)
    k = (size - 2 * pad) / span
    cx0 = (max(xs) + min(xs)) / 2
    cy0 = (max(ys) + min(ys)) / 2
    out = {}
    for name, (x, y) in coords.items():
        out[name] = (size / 2 + (x - cx0) * k, size / 2 - (y - cy0) * k)
    return out


def _right_angle_mark(d: ImageDraw.ImageDraw, at: tuple[float, float],
                      a: tuple[float, float], b: tuple[float, float], r: float = 16) -> None:
    """直角小方块：沿两条边各取一小段，再补一个角点。"""
    def unit(p, q):
        dx, dy = q[0] - p[0], q[1] - p[1]
        L = math.hypot(dx, dy) or 1
        return dx / L, dy / L

    u1 = unit(at, a)
    u2 = unit(at, b)
    p1 = (at[0] + u1[0] * r, at[1] + u1[1] * r)
    p2 = (at[0] + u2[0] * r, at[1] + u2[1] * r)
    corner = (p1[0] + p2[0] - at[0], p1[1] + p2[1] - at[1])
    d.line([p1, corner], fill=INK, width=2)
    d.line([p2, corner], fill=INK, width=2)


def _parallel_arrow(d: ImageDraw.ImageDraw, a: tuple[float, float], b: tuple[float, float]) -> None:
    """平行标记：线段中点画一个「>」。"""
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    p1 = (mx - ux * 6 + nx * 8, my - uy * 6 + ny * 8)
    p2 = (mx + ux * 6, my + uy * 6)
    p3 = (mx - ux * 6 - nx * 8, my - uy * 6 - ny * 8)
    d.line([p1, p2], fill=INK, width=2)
    d.line([p3, p2], fill=INK, width=2)


def _equal_ticks(d: ImageDraw.ImageDraw, a: tuple[float, float], b: tuple[float, float]) -> None:
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1
    nx, ny = -dy / L * 7, dx / L * 7
    d.line([(mx - nx, my - ny), (mx + nx, my + ny)], fill=INK, width=2)


def _extra_links(p: Problem, pos: dict[str, tuple[float, float]]) -> list[tuple[str, str]]:
    """补画「共线但没画出来」的那几段。

    「P 在 AB 的延长线上」这种条件，模型往往只把 A、B 连成线段，
    P 就成了一颗悬空的点——图上只有一条切线连着它，看起来像认错了。
    按共线关系把相邻的两点补连起来，图才是完整的一条直线。
    """
    try:
        from .solve import collinear_groups

        groups = collinear_groups(p)
    except Exception:  # noqa: BLE001
        return []

    have = {tuple(sorted(s)) for s in p.segments}
    out: list[tuple[str, str]] = []
    for g in groups:
        names = [x for x in g if x in pos]
        if len(names) < 3:
            continue
        a = names[0]
        ux, uy = pos[names[1]][0] - pos[a][0], pos[names[1]][1] - pos[a][1]
        L = math.hypot(ux, uy) or 1
        ux, uy = ux / L, uy / L
        names.sort(key=lambda t: (pos[t][0] - pos[a][0]) * ux + (pos[t][1] - pos[a][1]) * uy)
        for i in range(len(names) - 1):
            pair = tuple(sorted((names[i], names[i + 1])))
            if pair in have:
                continue
            have.add(pair)
            out.append(pair)
    return out


def _angle_arc(d: ImageDraw.ImageDraw, pos: dict[str, tuple[float, float]],
               vertex: str, a: str, b: str, r: float = 44) -> tuple[float, float]:
    """在顶点处画一段弧，返回弧外一点（用来放度数标签）。"""
    vx, vy = pos[vertex]
    a1 = math.degrees(math.atan2(pos[a][1] - vy, pos[a][0] - vx))
    a2 = math.degrees(math.atan2(pos[b][1] - vy, pos[b][0] - vx))
    delta = (a2 - a1 + 180.0) % 360.0 - 180.0   # 取小弧，别绕一圈
    s, e = (a1, a1 + delta) if delta >= 0 else (a1 + delta, a1)
    d.arc([vx - r, vy - r, vx + r, vy + r], start=s, end=e, fill=HILITE, width=3)
    mid = math.radians(a1 + delta / 2.0)
    return (vx + math.cos(mid) * (r + 32), vy + math.sin(mid) * (r + 32))


def _seg_dist(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """点到线段的距离。"""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _free_spot(pos: dict[str, tuple[float, float]], lines: list[tuple],
               w: float, h: float, size: int) -> tuple[float, float]:
    """在图四周找一个不压线的位置放标注。

    结论标签写死在两条边的中点外侧是不够的：这题的 CE、CF 都在图形右侧，
    标签必然被 CF 那条竖线穿过去，「求证 CE = CF」读起来像打翻了字。
    与其猜方向，不如把四周的几个候选位置都试一遍，挑离所有线最远的那个。
    """
    xs = [q[0] for q in pos.values()]
    ys = [q[1] for q in pos.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

    cands = [
        ((x0 + x1) / 2, y0 - h),          # 上
        ((x0 + x1) / 2, y1 + h),          # 下
        (x0 - w / 2 - 6, (y0 + y1) / 2),  # 左
        (x1 + w / 2 + 6, (y0 + y1) / 2),  # 右
        ((x0 + x1) / 2, y1 + h * 1.9),    # 再往下一点，给底部点标签让位
    ]

    best, best_d = cands[0], -1.0
    for cx, cy in cands:
        # 夹进画布，否则字会被裁掉
        cx = max(w / 2 + 4, min(size - w / 2 - 4, cx))
        cy = max(h / 2 + 4, min(size - h / 2 - 4, cy))
        d = min((_seg_dist((cx, cy), a, b) for a, b in lines), default=1e9)
        if d > best_d:
            best, best_d = (cx, cy), d
    return best


def _goal_mark(d: ImageDraw.ImageDraw, pos: dict[str, tuple[float, float]],
               goal: tuple, value: float | None, font: ImageFont.FreeTypeFont,
               size: int = 520) -> None:
    """把「题目问的那个量」在图上标出来。

    不标的话，图上一个直角方块、几条线，用户根本看不出系统在算哪个角——
    g2 就是这样被当成「标错了角」的。
    """
    kind, name, arms = goal

    if kind == "compare" and arms:
        # 证明题：结论是两条边相等，就把这两条边一起标红，再写上结论
        segs = [s for s in arms if len(s) == 2 and s[0] in pos and s[1] in pos]
        if not segs:
            return
        for s in segs:
            d.line([pos[s[0]], pos[s[1]]], fill=SHADE, width=9)
            d.line([pos[s[0]], pos[s[1]]], fill=HILITE, width=4)

        label = f"求证 {name}"
        tw = font.getlength(label)
        try:
            th = font.size * 1.3
        except AttributeError:
            th = 26.0
        # 用「所有点两两相连」当避让目标：这是真实线段的超集，宁可躲远一点
        names = list(pos)
        lines = [
            (pos[names[i]], pos[names[j]])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]
        tx, ty = _free_spot(pos, lines, tw, th, size)
        d.rectangle([tx - tw / 2 - 4, ty - th / 2, tx + tw / 2 + 4, ty + th / 2],
                    fill=BG)
        d.text((tx, ty), label, font=font, fill=HILITE, anchor="mm")
        return

    if kind == "angle" and arms:
        a, b = arms
        vertex = name if len(name) == 1 else name[1]
        if vertex in pos and a in pos and b in pos:
            tx, ty = _angle_arc(d, pos, vertex, a, b)
            label = f"∠{name} = {value:g}°" if value is not None else f"∠{name}"
            # 描一圈底色：标注躲不开线条和字母，描边比挪位置可靠
            d.text((tx, ty), label, font=font, fill=HILITE, anchor="mm",
                   stroke_width=4, stroke_fill=BG)
        return

    if kind == "length" and len(name) == 2:
        x, y = name[0], name[1]
        if x in pos and y in pos:
            d.line([pos[x], pos[y]], fill=HILITE, width=5)
            mx = (pos[x][0] + pos[y][0]) / 2
            my = (pos[x][1] + pos[y][1]) / 2
            # 沿垂线方向让开线段本身，否则字会压在被标红的那条线上
            dx, dy = pos[y][0] - pos[x][0], pos[y][1] - pos[x][1]
            L = math.hypot(dx, dy) or 1
            cx = sum(q[0] for q in pos.values()) / len(pos)
            cy = sum(q[1] for q in pos.values()) / len(pos)
            nx, ny = -dy / L, dx / L
            if (mx - cx) * nx + (my - cy) * ny < 0:   # 朝远离重心的一侧
                nx, ny = -nx, -ny
            label = f"{name} = {value:g}" if value is not None else name
            d.text((mx + nx * 26, my + ny * 26), label, font=font, fill=HILITE,
                   anchor="mm", stroke_width=4, stroke_fill=BG)


def pos_layout(p: Problem, pts: list[str]) -> dict[str, tuple[float, float]] | None:
    """用读图时记下的各点位置摆图。

    这是「解不出坐标」时最该走的一条路，以前却直接跳到了环形布局：
    8 个点按顺序排到圆周上，A 在正上方、D 到底下、E 跑到左边，和原图
    毫无关系——用户看一眼就说「重绘偏离题」。

    其实模型早就给了每点的近似位置（0–100 的目测值），形状够用。
    位置是图像坐标（y 向下），这里翻成数学坐标，交给 _fit 统一缩放。
    """
    pos = {k: v for k, v in (p.pos or {}).items() if k in set(pts)}
    if not pos or set(pos) < set(pts):
        return None
    return {k: (v[0], -v[1]) for k, v in pos.items()}


def render(p: Problem, coords: dict[str, tuple[float, float]] | None = None,
           size: int = 520, pad: int = 54,
           goal: tuple | None = None, value: float | None = None) -> Image.Image:
    im = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(im)

    pts = list(coords.keys()) if coords else list(p.points)
    if not pts:
        d.text((size // 2 - 60, size // 2), "没有读到图形", font=_font(22), fill=(150, 150, 150))
        return im

    # 坐标 > 读到的位置 > 环形示意，一级比一级弱
    guess = coords or pos_layout(p, pts)
    pos = _fit(guess, size, pad) if guess else _layout_ring(pts, size, pad)

    # 圆
    for c in p.circles:
        node = pos.get(c.center or "")
        if not node:
            continue
        near = [pos[x] for x in (c.through or []) if x in pos]
        if not near:
            continue
        r = sum(math.dist(node, q) for q in near) / len(near)
        d.ellipse([node[0] - r, node[1] - r, node[0] + r, node[1] + r], outline=CIRCLE, width=2)

    # 线段
    for a, b in p.segments:
        if a in pos and b in pos:
            d.line([pos[a], pos[b]], fill=INK, width=3)

    # 共线但没画出来的那几段（延长线、P 到 B 之类），否则点会悬空
    for a, b in _extra_links(p, pos):
        if a in pos and b in pos:
            d.line([pos[a], pos[b]], fill=INK, width=3)

    # 标记
    for m in p.marks:
        if m.type == "right_angle" and m.at in pos:
            ofs = [x for x in m.of if len(x) == 2 and m.at in x]
            if len(ofs) >= 2:
                e1 = ofs[0].replace(m.at, "")
                e2 = ofs[1].replace(m.at, "")
                if e1 in pos and e2 in pos:
                    _right_angle_mark(d, pos[m.at], pos[e1], pos[e2], r=19)
        elif m.type == "equal_length":
            for name in m.of:
                if len(name) == 2 and name[0] in pos and name[1] in pos:
                    _equal_ticks(d, pos[name[0]], pos[name[1]])
        elif m.type == "midpoint" and m.at in pos and m.of:
            # 中点标记的 of 是「整条线段」，等长关系实际落在 at 两侧的那两段上
            seg = m.of[0]
            if len(seg) == 2 and seg[0] in pos and seg[1] in pos:
                _equal_ticks(d, pos[m.at], pos[seg[0]])
                _equal_ticks(d, pos[m.at], pos[seg[1]])
        elif m.type == "parallel":
            for name in m.of:
                if len(name) == 2 and name[0] in pos and name[1] in pos:
                    _parallel_arrow(d, pos[name[0]], pos[name[1]])

    # 点与标签：标签朝「远离图形重心」的方向偏，避免压在线或点上
    f = _font(22)
    cx = sum(x for x, _ in pos.values()) / len(pos)
    cy = sum(y for _, y in pos.values()) / len(pos)
    for name, (x, y) in pos.items():
        d.ellipse([x - 3.5, y - 3.5, x + 3.5, y + 3.5], fill=POINT)
        dx, dy = x - cx, y - cy
        L = math.hypot(dx, dy)
        if L < 1e-6:
            tx, ty = x + 14, y - 14
        else:
            tx, ty = x + dx / L * 20, y + dy / L * 20
        d.text((tx, ty), name, font=f, fill=(20, 20, 20), anchor="mm")

    # 目标量放最后画，压在所有线条和标签之上
    if goal:
        _goal_mark(d, pos, goal, value, _font(22), size=size)

    return im


def side_by_side(original: str | None, redraw: Image.Image, size: int = 520) -> Image.Image:
    """左原图右重绘，方便直接对照。"""
    if not original:
        return redraw
    try:
        src = Image.open(original)
    except Exception:
        return redraw
    src = src.convert("RGB")
    # 裁掉纸面空白，否则原图缩下来内容只剩一小块，没法对照
    try:
        from mathagent.perceive import _ink_bbox

        box = _ink_bbox(src)
        if box:
            x0, y0, x1, y1 = box
            pad = 12
            src = src.crop(
                (max(0, x0 - pad), max(0, y0 - pad),
                 min(src.width, x1 + pad), min(src.height, y1 + pad))
            )
    except Exception:
        pass
    k = size / max(src.width, src.height)
    src = src.resize((max(1, int(src.width * k)), max(1, int(src.height * k))), Image.LANCZOS)

    canvas = Image.new("RGB", (src.width + redraw.width + 12, max(src.height, redraw.height)), (255, 255, 255))
    canvas.paste(src, (0, (canvas.height - src.height) // 2))
    canvas.paste(redraw, (src.width + 12, (canvas.height - redraw.height) // 2))
    return canvas


def describe_diff(p: Problem, coords: dict[str, tuple[float, float]] | None, basis: str = "") -> str:
    """用一句话说明这张重绘图是根据什么画的。"""
    if basis == "pos" or (not coords and p.pos):
        return (
            "按读到的各点位置重绘，点线关系就是识别结果；形状比例不作数——"
            "对照左图只看有没有少线、点有没有错位。"
        )
    if coords:
        return "按解出的真实坐标重绘，几何关系是准的；对照左图看点和线有没有缺漏。"
    return "没能解出坐标，按点的连接关系示意重绘，只看点线关系，比例不作数。"

