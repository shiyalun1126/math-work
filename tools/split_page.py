"""把整页手写照片自动切成单行公式小图（命令行）。

算法在 `mathagent/lines.py`，这里只是薄壳：界面上的拍照批改调的是同一份代码，
所以"命令行切对了、界面上切错了"这种事不会发生。

用法：
    python tools/split_page.py data/inbox/page_01.jpg --out data/crops
    python tools/split_page.py data/inbox --out data/crops --expect 8
    python tools/split_page.py data/inbox/page_01.jpg --out data/crops --debug

输出：
    <out>/<页名>_<行号>.jpg      单行小图，行号从 01 起、自上而下
    <out>/manifest.csv           页名/行号/文件名/坐标，供后续对回 shotlist
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mathagent.lines import IMG_EXT, debug_sheet, save_crops, split_page  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


FIELDS = ["page", "line", "crop", "x0", "y0", "x1", "y1", "w", "h"]


def split_one(path: Path, out_dir: Path, args, rows: list) -> int:
    page = split_page(
        path,
        deskew=not args.no_deskew,
        threshold_ratio=args.threshold_ratio,
        min_gap_ratio=args.min_gap_ratio,
    )
    paths = save_crops(page, out_dir, stem=path.stem)

    for c, p in zip(page.crops, paths):
        x0, y0, x1, y1 = c.box
        rows.append(
            {
                "page": path.name,
                "line": c.index,
                "crop": p.name,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "w": x1 - x0,
                "h": y1 - y0,
            }
        )

    if args.debug:
        debug_sheet(page).save(out_dir / f"{path.stem}_debug.jpg", quality=90)

    flag = ""
    if args.expect and len(page.crops) != args.expect:
        flag = f"  ← 预期 {args.expect} 行，实际 {len(page.crops)} 行，需人工核对"
    extra = f"  [{page.note}]" if page.note else ""
    print(f"{path.name}: {len(page.crops)} 行  纠偏 {page.skew:+.1f}°{flag}{extra}")
    return len(page.crops)


def main():
    ap = argparse.ArgumentParser(description="整页手写照片自动切行")
    ap.add_argument("src", help="图片文件或目录")
    ap.add_argument("--out", default="data/crops", help="输出目录")
    ap.add_argument("--expect", type=int, default=0, help="每页预期行数，用于报警")
    ap.add_argument("--debug", action="store_true", help="额外输出画框的调试图")
    ap.add_argument("--no-deskew", action="store_true", help="关闭纠偏")
    ap.add_argument("--threshold-ratio", type=float, default=0.03, help="行带阈值占峰值比例")
    ap.add_argument("--min-gap-ratio", type=float, default=0.012, help="行间距占页高比例，小于此值合并")
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = [src] if src.is_file() else sorted(
        p for p in src.iterdir() if p.suffix.lower() in IMG_EXT
    )
    if not pages:
        print(f"{src} 下没有图片")
        sys.exit(1)

    rows = []
    for page in pages:
        split_one(page, out_dir, args, rows)

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n共 {len(pages)} 页 → {len(rows)} 张单行图，清单：{manifest}")


if __name__ == "__main__":
    main()
