"""把 shotlist.csv 转成便于阅读/打印的 shotlist.md。

用法：
    python tools/make_shotlist_md.py
"""

import csv
import sys
from collections import OrderedDict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "shotlist.csv"
DST = ROOT / "data" / "shotlist.md"

ORDER = ["方程", "方程组", "化简", "求导积分", "复合结构", "难例", "自由写"]


def main():
    with SRC.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    groups = OrderedDict((k, []) for k in ORDER)
    for r in rows:
        groups.setdefault((r.get("type") or "其他").strip(), []).append(r)

    lines = [
        "# 采集清单",
        "",
        "**一页写 10 行，整页拍一张，不用裁剪**——切行由 `tools/split_page.py` 自动完成。",
        "",
        "写法：内容照抄，字迹按你平时做题的习惯，不要刻意写工整；行距留够，别让上一行的分数线碰到下一行。",
        "",
        "页与 id 的对应见 `data/pages.csv`：page_01 自上而下就是 eq_001…eq_010。",
        "",
    ]

    for name, items in groups.items():
        if not items:
            continue
        lines.append(f"## {name}（{len(items)} 张）")
        lines.append("")
        lines.append("| id | 写法 | split | 提示 |")
        lines.append("|---|---|---|---|")
        for r in items:
            rid = (r.get("id") or "").strip()
            display = (r.get("display") or "").strip()
            split = (r.get("split") or "").strip()
            note = (r.get("note") or "").strip()
            if not display:
                display = "（随意写，不用照抄）"
            lines.append(f"| {rid} | `{display}` | {split} | {note} |")
        lines.append("")

    DST.write_text("\n".join(lines), encoding="utf-8")
    print(f"写入 {DST.relative_to(ROOT)}，共 {len(rows)} 条")


if __name__ == "__main__":
    main()
