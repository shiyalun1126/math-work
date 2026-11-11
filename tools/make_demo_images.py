"""生成演示样例图（不需要真实拍照）。

用系统自带的 Ink Free 手写体把式子画在纸上，加一点旋转和阴影，
目的是让 demo 流程能跑通，不是冒充真实手写数据。
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures"

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\STKAITI.TTF",  # 华文楷体：手写感 + 数学符号齐全
    r"C:\Windows\Fonts\simkai.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\Inkfree.ttf",
]

SAMPLES = {
    "demo1": ["x² - 5x + 6 = 0"],
    "demo2": ["x + 1", "────── = 3", "x - 2"],
    "demo3": ["x² - 3x + 2 = 0"],
    "demo4": ["∫ x e^{x} dx"],
    "demo5": ["x + y = 3", "x - y = 1"],
    "demo6": ["2x + 1 = 5"],
    "demo7": ["3x - 5 = x + 7"],
}


def pick_font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def draw_rich(d: ImageDraw.ImageDraw, xy, text: str, font, fill) -> int:
    """支持 ^{..} 上标的简易排版，返回画完后的 x 坐标。

    楷体里没有 ˣ 这个字形，直接写 "eˣ" 会渲染成空白或方框，
    模型读不到自然也就识别不出上标——图错了不能赖模型。
    """
    x, y = xy
    size = font.size
    sup_font = pick_font(max(16, int(size * 0.62)))
    i = 0
    while i < len(text):
        if text[i] == "^" and i + 1 < len(text) and text[i + 1] == "{":
            j = text.index("}", i)
            seg = text[i + 2 : j]
            d.text((x, y - size * 0.16), seg, font=sup_font, fill=fill)
            x += d.textlength(seg, font=sup_font)
            i = j + 1
            continue
        d.text((x, y), text[i], font=font, fill=fill)
        x += d.textlength(text[i], font=font)
        i += 1
    return x


def measure(d: ImageDraw.ImageDraw, text: str, font) -> float:
    if "^{" not in text:
        return d.textlength(text, font=font)
    sup_font = pick_font(max(16, int(font.size * 0.62)))
    total, i = 0.0, 0
    while i < len(text):
        if text[i] == "^" and i + 1 < len(text) and text[i + 1] == "{":
            j = text.index("}", i)
            total += d.textlength(text[i + 2 : j], font=sup_font)
            i = j + 1
            continue
        total += d.textlength(text[i], font=font)
        i += 1
    return total


def render(lines: list[str], path: Path, angle: float = -1.4) -> None:
    font = pick_font(58)
    pad, lh = 40, 84
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    widths = [measure(probe, t, font) for t in lines]
    w = int(max(widths)) + pad * 2 + 20
    h = lh * len(lines) + pad * 2

    im = Image.new("RGB", (w, h), (253, 252, 246))
    d = ImageDraw.Draw(im)
    for i, t in enumerate(lines):
        tw = measure(d, t, font)
        draw_rich(d, ((w - tw) / 2, pad + i * lh), t, font, (38, 38, 42))
    # 轻微纸纹：几条极淡的横线
    for y in range(pad, h - pad, 26):
        d.line([(12, y), (w - 12, y)], fill=(246, 244, 236), width=1)
    im = im.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(253, 252, 246))
    im.save(path, quality=95)
    print(f"  {path.name}  {im.size[0]}x{im.size[1]}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("生成演示样例：")
    for i, (stem, lines) in enumerate(SAMPLES.items()):
        render(lines, OUT / f"{stem}.png", angle=-1.2 + 0.4 * (i % 3))
    print("完成。运行 python app.py 打开界面。")


if __name__ == "__main__":
    sys.exit(main())
