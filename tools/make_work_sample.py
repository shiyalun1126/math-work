"""造几张"学生作业照"，用来压测拍照批改。

为什么要造：真实手写数据没法随项目分发（涉及隐私，也没法保证每次一样）。
合成页至少能验证三件事——切行切得开、逐行认得出、认不出时不会冤枉人。

刻意做的退化：横线本、轻微旋转、纸面明暗、噪点、轻微模糊。
字迹仍是印刷体，所以它**不能**替代真实手写，只能验证链路对不对。

用法：
    python tools/make_work_sample.py
输出：fixtures/work/*.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "work"

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\STKAITI.TTF",
    r"C:\Windows\Fonts\simkai.ttf",
]

# 三份过程，各代表一类要能判出来的错
PAGES = {
    # 过程成立
    "work_ok": [
        "x² - 5x + 6 = 0",
        "两边同乘2得 2x² - 10x + 12 = 0",
        "(x - 2)(x - 3) = 0",
        "x = 2 或 x = 3",
    ],
    # 伪推导：过程错，答案对
    "work_fake": [
        "x² - 5x + 6 = 0",
        "x² - 6 = 5x",
        "x² - 5x = 6",
        "x = 2 或 x = 3",
    ],
    # 漏解：只写了一个根
    "work_miss": [
        "x² - 5x + 6 = 0",
        "(x - 2)(x - 3) = 0",
        "x = 2",
    ],
}


def pick_font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def page(lines: list[str], seed: int = 0) -> Image.Image:
    w, h = 1240, 1754
    rnd = np.random.default_rng(seed)
    im = Image.new("RGB", (w, h), (250, 248, 241))
    d = ImageDraw.Draw(im)

    for y in range(70, h - 40, 66):                      # 横线本
        d.line([(70, y), (w - 70, y)], fill=(231, 229, 220), width=2)

    f = pick_font(58)
    y = 300
    for line in lines:
        d.text((130 + float(rnd.uniform(-4, 4)), y), line, font=f, fill=(32, 32, 40))
        y += 128

    im = im.rotate(-1.1, resample=Image.BICUBIC, expand=False, fillcolor=(250, 248, 241))
    im = im.filter(ImageFilter.GaussianBlur(0.6))

    arr = np.asarray(im).astype(np.int16)
    arr += rnd.normal(0, 3.2, arr.shape).astype(np.int16)   # 噪点
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (stem, lines) in enumerate(PAGES.items()):
        p = OUT / f"{stem}.jpg"
        page(lines, seed=11 + i).save(p, quality=92)
        print(f"{p}  ({len(lines)} 行)")
    print(f"\n共 {len(PAGES)} 张 → {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
