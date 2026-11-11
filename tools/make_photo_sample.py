"""把干净的式子“拍成照片”，用来压测预处理链路。

真实拍照的退化主要有四种，这里逐项模拟：
  1. 透视——手机不会正对纸张，四条边长度不等
  2. 光照——台灯在一侧，纸面有明暗渐变，还有笔影
  3. 噪声——高感光下的噪点
  4. 糊——手持抖动加对焦不准

它不能替代真实手写数据（字迹仍是印刷体），只用来验证裁边、
摆正、缩放这几步在退化条件下是否还站得住。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "photos"
# 必须用华文楷体：中易楷体（simkai）没有 ² 字形，会渲染成空方框，
# 模型读到方框只能靠猜，评估就成了自欺欺人
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\STKAITI.TTF",
    r"C:\Windows\Fonts\simkai.ttf",
]


def pick_font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def formula_page(text: str) -> Image.Image:
    """A4 竖版，式子只占中间一行，模拟真实整页照的占比。"""
    w, h = 1240, 1754
    im = Image.new("RGB", (w, h), (250, 248, 241))
    d = ImageDraw.Draw(im)
    for y in range(60, h - 40, 62):  # 横线本
        d.line([(60, y), (w - 60, y)], fill=(232, 230, 222), width=2)
    f = pick_font(64)
    tw = d.textlength(text, font=f)
    d.text(((w - tw) / 2, 640), text, font=f, fill=(28, 28, 34))
    return im


def perspective(im: Image.Image, strength: float, seed: int = 7) -> Image.Image:
    """四角各偏移一点：等效于裁掉一个不规则四边形之外的部分。"""
    w, h = im.size
    s = strength * min(w, h)
    rnd = np.random.default_rng(seed)
    dst = [
        (rnd.uniform(0, s), rnd.uniform(0, s)),
        (w - rnd.uniform(0, s), rnd.uniform(0, s * 0.6)),
        (w - rnd.uniform(0, s * 0.7), h - rnd.uniform(0, s)),
        (rnd.uniform(0, s * 0.8), h - rnd.uniform(0, s * 0.7)),
    ]
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(dst, fill=255)
    # 纸外是桌面，偏灰
    out = Image.new("RGB", (w, h), (222, 218, 210))
    out.paste(im, (0, 0), mask)
    return out


def lighting(im: Image.Image, strength: float = 0.28) -> Image.Image:
    """左上亮右下暗的线性渐变，模拟侧向台灯。"""
    w, h = im.size
    yy, xx = np.mgrid[0:h, 0:w]
    ramp = (0.55 * xx / w + 0.45 * yy / h) * strength
    factor = np.clip(1.0 - ramp, 0.45, 1.0)[..., None]
    a = np.asarray(im).astype(np.float32) * factor
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def noise(im: Image.Image, amp: float) -> Image.Image:
    a = np.asarray(im).astype(np.int16)
    a += np.random.default_rng(3).normal(0, amp, a.shape).astype(np.int16)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def make(text: str, path: Path, level: int) -> None:
    """level 0–4：越大越糊、越斜、噪点越多、压缩越狠。"""
    im = formula_page(text)
    im = perspective(im, 0.010 + 0.014 * level)
    im = lighting(im, 0.18 + 0.05 * level)
    if level >= 1:
        im = im.filter(ImageFilter.GaussianBlur(0.6 + 0.45 * level))
    im = noise(im, 3 + 3 * level)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=max(45, 76 - 7 * level))
    print(f"  {path.name}  {im.size[0]}x{im.size[1]}  level={level}  {path.stat().st_size // 1024}KB")


def main() -> int:
    print("生成手机拍摄模拟图：")
    cases = [
        ("photo_easy", "x² - 5x + 6 = 0", 0),
        ("photo_mid", "x² - 5x + 6 = 0", 2),
        ("photo_hard", "x² - 5x + 6 = 0", 4),
        ("photo_frac", "(x+1)/(x-2) = 3", 2),
    ]
    for name, text, lv in cases:
        make(text, OUT / f"{name}.jpg", lv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
