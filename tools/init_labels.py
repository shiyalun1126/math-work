"""从 shotlist.csv 生成 labels.csv。

用法：
    python tools/init_labels.py

规则：
    - split 为 self_train 的图放在 data/self/，self_test 的放在 data/test/
    - 文件名 = id + .jpg
    - latex 为空的（自由写那批）先跳过，等看图标注后再补
"""

import csv
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SHOTLIST = ROOT / "data" / "shotlist.csv"
LABELS = ROOT / "data" / "labels.csv"


def main():
    with SHOTLIST.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    out = []
    pending = []
    bad = []

    for r in rows:
        rid = (r.get("id") or "").strip()
        latex = (r.get("latex") or "").strip()
        split = (r.get("split") or "").strip()
        if not rid:
            continue
        if '"' in latex or "\n" in latex:
            bad.append(rid)
            continue
        if not latex:
            pending.append(rid)
            continue
        folder = "self" if split == "self_train" else "test"
        out.append(
            {
                "file": f"{folder}/{rid}.jpg",
                "latex": latex,
                "split": split,
                "type": (r.get("type") or "").strip(),
                "note": (r.get("note") or "").strip(),
            }
        )

    with LABELS.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "latex", "split", "type", "note"])
        writer.writeheader()
        writer.writerows(out)

    print(f"写入 {LABELS.relative_to(ROOT)}：{len(out)} 条")
    print("按 split：", dict(Counter(r["split"] for r in out)))
    print("按题型：", dict(Counter(r["type"] for r in out)))
    if pending:
        print(f"待标注（自由写）：{len(pending)} 条，看图后补进 labels.csv")
    if bad:
        print(f"latex 字段异常，已跳过：{bad}")


if __name__ == "__main__":
    main()
