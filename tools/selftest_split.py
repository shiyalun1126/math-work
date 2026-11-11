"""自测：造一张合成"手写页"，验证 split_page.py 能否切对。

用法：
    python tools/selftest_split.py

造页规则：8 行公式，其中第 3、4 行故意贴近（模拟粘连），
另外画一条贯穿整页的横线（模拟横线本），整页再旋转 1.2°。
"""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

LINES = [
    "2x+5=13",
    "x^2-5x+6=0",
    "(2x-6)/2=6",
    "x^2+5x+6",
    "sqrt(x+3)=x-1",
    "int 0..1 x^2 dx",
    "lim x->0 sinx/x",
    "y=x^3",
]
# 第 3、4 行只差 11 px，小于合并阈值，会被并成一条行带
YS = [180, 350, 520, 585, 760, 930, 1100, 1270]


def build_page(path: Path) -> None:
    img = Image.new("L", (1240, 1754), 255)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 46)
    except OSError:
        font = ImageFont.load_default()

    for text, y in zip(LINES, YS):
        d.text((150, y), text, fill=25, font=font)

    d.line([(60, 690), (1180, 690)], fill=110, width=2)   # 横线本的线，应被抹掉
    img = img.rotate(1.2, resample=Image.BICUBIC, expand=True, fillcolor=255)
    img.save(path)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="split_selftest_"))
    page = tmp / "page_01.png"
    out = tmp / "crops"
    build_page(page)

    print(f"合成测试页：{page}")
    proc = subprocess.run(
        [PY, str(ROOT / "tools" / "split_page.py"), str(page), "--out", str(out), "--expect", str(len(LINES)), "--debug"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr.strip())
        sys.exit(1)

    crops = sorted(p for p in out.glob("page_01_*.jpg") if not p.stem.endswith("_debug"))
    print(f"\n切出 {len(crops)} 张，预期 {len(LINES)} 张")

    with (out / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        print(f"  行{r['line']:>2}  y={r['y0']:>4}-{r['y1']:<4} 高{r['h']:>3}  宽{r['w']}")

    heights = [int(r["h"]) for r in rows]
    if len(crops) != len(LINES):
        print("\n失败：行数与预期不符")
        sys.exit(1)
    if max(heights) > 3 * (sum(heights) / len(heights)):
        print("\n失败：存在异常高的行带，说明粘连未切开")
        sys.exit(1)

    print("\n通过：行数一致，且无异常高行带")
    print(f"（临时目录保留在 {tmp}，可打开 {out / 'page_01_debug.jpg'} 看切分效果）")


if __name__ == "__main__":
    main()
