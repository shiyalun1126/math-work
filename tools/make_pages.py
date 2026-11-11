"""生成 pages.csv：一页照片覆盖哪些 id。

用法：
    python tools/make_pages.py --per-page 10

规则：按 shotlist.csv 的顺序，每 per-page 条为一页，id 连续。
切分后第 i 行对应本页第 i 个 id（自上而下）。
"""

import argparse
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SHOTLIST = ROOT / "data" / "shotlist.csv"
OUT = ROOT / "data" / "pages.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-page", type=int, default=10)
    args = ap.parse_args()

    with SHOTLIST.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("id") or "").strip()]

    pages = []
    for i in range(0, len(rows), args.per_page):
        chunk = rows[i : i + args.per_page]
        idx = len(pages) + 1
        types = []
        for r in chunk:
            t = (r.get("type") or "").strip()
            if t not in types:
                types.append(t)
        pages.append(
            {
                "page": f"page_{idx:02d}",
                "lines": len(chunk),
                "start_id": chunk[0]["id"].strip(),
                "end_id": chunk[-1]["id"].strip(),
                "types": "/".join(types),
                "note": "整页拍一张，切行后按序对应 id",
            }
        )

    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["page", "lines", "start_id", "end_id", "types", "note"]
        )
        writer.writeheader()
        writer.writerows(pages)

    print(f"写入 {OUT.relative_to(ROOT)}：{len(pages)} 页，共 {len(rows)} 行")
    for p in pages[:3]:
        print(f"  {p['page']}  {p['lines']} 行  {p['start_id']} → {p['end_id']}  {p['types']}")
    if len(pages) > 3:
        print(f"  ... 共 {len(pages)} 页")


if __name__ == "__main__":
    main()
