"""把切好的单行图按 id 分流到 data/self 或 data/test。

用法：
    python tools/route_crops.py --dry-run     # 只打印映射，不复制
    python tools/route_crops.py               # 实际复制

依赖：
    data/crops/manifest.csv   split_page.py 的产物
    data/pages.csv            页 ↔ id 段映射
    data/shotlist.csv         决定 id 的先后顺序
    data/labels.csv           决定每个 id 属于 train 还是 test
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def read_csv(path: Path) -> list:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印映射，不复制文件")
    ap.add_argument("--crops", default=str(DATA / "crops"))
    args = ap.parse_args()

    crops_dir = Path(args.crops)
    manifest = crops_dir / "manifest.csv"
    for p in (manifest, DATA / "pages.csv", DATA / "shotlist.csv", DATA / "labels.csv"):
        if not p.exists():
            print(f"缺少 {p}")
            sys.exit(1)

    ids = [r["id"].strip() for r in read_csv(DATA / "shotlist.csv") if r.get("id")]
    pos = {rid: i for i, rid in enumerate(ids)}
    pages = {r["page"].strip(): r for r in read_csv(DATA / "pages.csv")}
    split_of = {
        Path(r["file"]).stem: r["split"].strip() for r in read_csv(DATA / "labels.csv")
    }

    rows = read_csv(manifest)
    if not rows:
        print("manifest.csv 是空的")
        sys.exit(1)

    plan, warns = [], []
    for r in rows:
        page_raw = r["page"].strip()
        page = Path(page_raw).stem          # manifest 存的是文件名，pages.csv 存的是页名
        line = int(r["line"])
        crop = crops_dir / r["crop"].strip()
        if page not in pages:
            warns.append(f"{page_raw} 不在 pages.csv 里，跳过 {r['crop']}")
            continue
        start = pos.get(pages[page]["start_id"].strip())
        if start is None:
            warns.append(f"{page} 的 start_id 对不上 shotlist，跳过")
            continue
        idx = start + line - 1
        if idx >= len(ids):
            warns.append(f"{page} 第 {line} 行超出 id 范围")
            continue
        rid = ids[idx]
        split = split_of.get(rid, "self_test")     # 自由写那批还没标，默认进 test
        dest_dir = DATA / ("self" if split == "self_train" else "test")
        plan.append((crop, dest_dir / f"{rid}.jpg", rid, split, page, line))

    # 每页行数校验
    by_page = {}
    for r in rows:
        by_page.setdefault(Path(r["page"].strip()).stem, []).append(int(r["line"]))
    for page, lines in by_page.items():
        expect = int(pages[page]["lines"]) if page in pages else 0
        if expect and len(lines) != expect:
            warns.append(f"{page} 切出 {len(lines)} 行，pages.csv 声明 {expect} 行")

    for crop, dest, rid, split, page, line in plan:
        tag = "复制" if not args.dry_run else "计划"
        print(f"{tag}  {crop.name:<18} → {dest.relative_to(ROOT)}   ({page} 第{line}行)")
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(crop, dest)

    n_train = sum(1 for p in plan if p[3] == "self_train")
    print(f"\n共 {len(plan)} 张：self {n_train} 张，test {len(plan) - n_train} 张")
    if warns:
        print("\n警告：")
        for w in warns:
            print("  -", w)


if __name__ == "__main__":
    main()
