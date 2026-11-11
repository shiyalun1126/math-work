"""生成「题干 + 图形」版面的几何题测试图。

目的：探测现有 VLM 通道面对真实试卷版面（文字与图形同页）时，
能不能分出「这是题干、这是图」，以及能不能读出图中的几何结构。

不是生产功能，只用于能力探测。
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "geo"

KAI = r"C:\Windows\Fonts\STKAITI.TTF"
HEI = r"C:\Windows\Fonts\simhei.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def draw_angle_mark(d: ImageDraw.ImageDraw, vx: float, vy: float,
                    a1: float, a2: float, r: float, color=(30, 30, 30)) -> None:
    """直角小方块：在顶点处，两边方向 a1/a2（弧度）之间画 L 形。"""
    p1 = (vx + r * math.cos(a1), vy + r * math.sin(a1))
    p2 = (vx + r * math.cos(a2), vy + r * math.sin(a2))
    corner = (p1[0] + p2[0] - vx, p1[1] + p2[1] - vy)
    d.line([p1, corner], fill=color, width=3)
    d.line([p2, corner], fill=color, width=3)


def tick(d: ImageDraw.ImageDraw, x1: float, y1: float, x2: float, y2: float,
         n: int = 1, size: int = 9) -> None:
    """等长标记：线段中点画 n 道短横。"""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy) or 1
    nx, ny = -dy / L, dx / L  # 法向
    tx, ty = dx / L, dy / L   # 切向
    for i in range(n):
        off = (i - (n - 1) / 2) * 7
        cx, cy = mx + tx * off, my + ty * off
        d.line([(cx - nx * size, cy - ny * size), (cx + nx * size, cy + ny * size)],
               fill=(30, 30, 30), width=3)


def panel_triangle() -> Image.Image:
    """题 1：直角三角形 + 斜边中点。"""
    W, H = 1100, 620
    im = Image.new("RGB", (W, H), (252, 251, 248))
    d = ImageDraw.Draw(im)

    d.text((60, 50), "1. 如图，在△ABC中，∠C = 90°，AC = 6，", font=font(KAI, 34), fill=(25, 25, 25))
    d.text((60, 100), "BC = 8，D 是 AB 的中点，求 CD 的长。", font=font(KAI, 34), fill=(25, 25, 25))

    # 直角三角形：C 为直角顶点
    cx, cy = 330, 520
    k = 34
    A = (cx, cy - 6 * k)          # 正上方，AC=6
    B = (cx + 8 * k, cy)          # 正右方，BC=8
    C = (cx, cy)
    D = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)

    for p, q in ((A, B), (B, C), (C, A), (C, D)):
        d.line([p, q], fill=(30, 30, 30), width=4)

    # 直角标记（从 C 指向 A 是 -90°，指向 B 是 0°）
    # 半径取小一点，避免与 CD 连线交叉造成歧义
    draw_angle_mark(d, cx, cy, -math.pi / 2, 0.0, 20)

    # D 是中点的等长标记：AD 与 DB 各一道
    tick(d, *A, *D, n=1)
    tick(d, *D, *B, n=1)

    lbl = font(KAI, 30)
    d.text((A[0] - 34, A[1] - 40), "A", font=lbl, fill=(20, 20, 20))
    d.text((B[0] + 10, B[1] - 16), "B", font=lbl, fill=(20, 20, 20))
    d.text((C[0] - 36, C[1] - 8), "C", font=lbl, fill=(20, 20, 20))
    d.text((D[0] + 12, D[1] - 34), "D", font=lbl, fill=(20, 20, 20))

    # 边长标注
    d.text((A[0] - 62, (A[1] + C[1]) / 2 - 18), "6", font=font(KAI, 28), fill=(20, 20, 20))
    d.text(((B[0] + C[0]) / 2 - 10, C[1] + 8), "8", font=font(KAI, 28), fill=(20, 20, 20))

    return im


def panel_circle() -> Image.Image:
    """题 2：圆 + 直径 + 切点，考切线垂直于半径。"""
    W, H = 1100, 620
    im = Image.new("RGB", (W, H), (252, 251, 248))
    d = ImageDraw.Draw(im)

    d.text((60, 50), "2. 如图，AB 是⊙O 的直径，点 C 在圆上，", font=font(KAI, 34), fill=(25, 25, 25))
    d.text((60, 100), "过 C 作⊙O 的切线，交 AB 的延长线于 P。", font=font(KAI, 34), fill=(25, 25, 25))
    d.text((60, 150), "若∠A = 25°，求∠P 的度数。", font=font(KAI, 34), fill=(25, 25, 25))

    ox, oy, R = 430, 400, 150
    d.ellipse([ox - R, oy - R, ox + R, oy + R], outline=(30, 30, 30), width=4)

    A = (ox - R, oy)
    B = (ox + R, oy)
    d.line([A, B], fill=(30, 30, 30), width=4)

    # C 在圆上（上方偏右）
    ang = math.radians(-58)
    C = (ox + R * math.cos(ang), oy + R * math.sin(ang))
    d.line([A, C], fill=(30, 30, 30), width=4)
    d.line([B, C], fill=(30, 30, 30), width=4)

    # 切线：过 C 垂直于 OC，交 AB 延长线于 P
    tangent_ang = ang + math.pi / 2
    # 参数化求与 y=oy 的交点
    t = (oy - C[1]) / math.sin(tangent_ang)
    P = (C[0] + t * math.cos(tangent_ang), oy)
    d.line([C, P], fill=(30, 30, 30), width=4)
    d.line([C, (C[0] - t * math.cos(tangent_ang), oy - t * math.sin(tangent_ang))],
           fill=(30, 30, 30), width=2)

    # O 与 OC
    d.line([(ox, oy), C], fill=(30, 30, 30), width=3)
    d.ellipse([ox - 4, oy - 4, ox + 4, oy + 4], fill=(30, 30, 30))

    # 直角标记：OC ⟂ CP
    draw_angle_mark(d, C[0], C[1], ang + math.pi, tangent_ang, 22)

    lbl = font(KAI, 28)
    d.text((A[0] - 30, A[1] + 6), "A", font=lbl, fill=(20, 20, 20))
    d.text((B[0] + 8, B[1] - 34), "B", font=lbl, fill=(20, 20, 20))
    d.text((C[0] + 6, C[1] - 36), "C", font=lbl, fill=(20, 20, 20))
    d.text((ox - 30, oy - 34), "O", font=lbl, fill=(20, 20, 20))
    d.text((P[0] + 6, P[1] - 34), "P", font=lbl, fill=(20, 20, 20))

    return im


def panel_isosceles() -> Image.Image:
    """题 3：等腰三角形，两腰等长 + 一个底角，求顶角。"""
    W, H = 1100, 620
    im = Image.new("RGB", (W, H), (252, 251, 248))
    d = ImageDraw.Draw(im)

    d.text((60, 50), "3. 如图，在△ABC中，AB = AC，∠B = 70°，", font=font(KAI, 34), fill=(25, 25, 25))
    d.text((60, 100), "求∠A的度数。", font=font(KAI, 34), fill=(25, 25, 25))

    A = (430, 190)
    B = (300, 520)
    C = (560, 520)
    for p, q in ((A, B), (B, C), (C, A)):
        d.line([p, q], fill=(30, 30, 30), width=4)

    # 两腰各一道等长标记
    tick(d, *A, *B, n=1)
    tick(d, *A, *C, n=1)

    lbl = font(KAI, 30)
    d.text((A[0] - 16, A[1] - 44), "A", font=lbl, fill=(20, 20, 20))
    d.text((B[0] - 34, B[1] + 6), "B", font=lbl, fill=(20, 20, 20))
    d.text((C[0] + 12, C[1] + 6), "C", font=lbl, fill=(20, 20, 20))

    return im


def panel_pythagoras() -> Image.Image:
    """题 4：直角三角形给两条直角边，求斜边——验证目标解析能不能换个量。"""
    W, H = 1100, 620
    im = Image.new("RGB", (W, H), (252, 251, 248))
    d = ImageDraw.Draw(im)

    d.text((60, 50), "4. 如图，在△ABC中，∠C = 90°，AC = 3，", font=font(KAI, 34), fill=(25, 25, 25))
    d.text((60, 100), "BC = 4，求 AB 的长。", font=font(KAI, 34), fill=(25, 25, 25))

    cx, cy = 330, 520
    k = 42
    A = (cx, cy - 3 * k)
    B = (cx + 4 * k, cy)
    C = (cx, cy)
    for p, q in ((A, B), (B, C), (C, A)):
        d.line([p, q], fill=(30, 30, 30), width=4)
    draw_angle_mark(d, cx, cy, -math.pi / 2, 0.0, 22)

    lbl = font(KAI, 30)
    d.text((A[0] - 34, A[1] - 40), "A", font=lbl, fill=(20, 20, 20))
    d.text((B[0] + 10, B[1] - 16), "B", font=lbl, fill=(20, 20, 20))
    d.text((C[0] - 36, C[1] - 8), "C", font=lbl, fill=(20, 20, 20))
    d.text((A[0] - 56, (A[1] + C[1]) / 2 - 18), "3", font=font(KAI, 28), fill=(20, 20, 20))
    d.text(((B[0] + C[0]) / 2 - 10, C[1] + 8), "4", font=font(KAI, 28), fill=(20, 20, 20))

    return im


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panels = (
        ("g1", panel_triangle),
        ("g2", panel_circle),
        ("g3", panel_isosceles),
        ("g4", panel_pythagoras),
    )
    for name, fn in panels:
        img = fn()
        # 模拟拍照：轻微旋转 + 轻微噪点
        img = img.rotate(0.6, resample=Image.BICUBIC, fillcolor=(252, 251, 248))
        path = OUT / f"{name}.png"
        img.save(path)
        print(f"已生成 {path}  ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
