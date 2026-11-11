"""检查 data/labels.csv 与图片文件是否一致。

用法：
    python data/check_data.py

检查项：
    1. 标注里登记的文件是否存在
    2. 目录里有图但没登记（漏标）
    3. LaTeX 字段是否为空 / 可疑
    4. 同一公式同时出现在 self_train 和 self_test（泄漏）
    5. 题型与 split 分布统计
    6. 若安装了 sympy + latex2sympy2，额外做解析检查
"""

import csv
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "labels.csv"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED = ["file", "latex", "split", "type"]


def load_rows():
    with LABELS.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("labels.csv 是空的")
            sys.exit(1)
        missing = [c for c in REQUIRED if c not in reader.fieldnames]
        if missing:
            print(f"表头缺少字段：{missing}")
            sys.exit(1)
        return [r for r in reader if r.get("file")]


def collect_images():
    found = set()
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            found.add(p.relative_to(ROOT).as_posix())
    return found


def try_parse(latex):
    """返回 None 表示未安装解析器，True/False 表示能否解析。"""
    try:
        from latex2sympy2 import latex2sympy
    except ImportError:
        return None
    try:
        latex2sympy(latex)
        return True
    except Exception:
        return False


def main():
    if not LABELS.exists():
        print(f"找不到 {LABELS}")
        sys.exit(1)

    rows = load_rows()
    images = collect_images()
    listed = {r["file"].strip() for r in rows}

    problems = []

    missing_files = sorted(f for f in listed if not (ROOT / f).exists())
    if missing_files:
        problems.append(("标注了但文件不存在", missing_files))

    unlabeled = sorted(images - listed)
    if unlabeled:
        problems.append(("有图但未登记", unlabeled))

    empty_latex = [r["file"] for r in rows if not (r.get("latex") or "").strip()]
    if empty_latex:
        problems.append(("LaTeX 为空", empty_latex))

    suspicious = []
    for r in rows:
        tex = (r.get("latex") or "").strip()
        # 中文、全角符号、单独的 a/b 写法都是常见标注错误
        if any("\u4e00" <= ch <= "\u9fff" for ch in tex):
            suspicious.append((r["file"], "含中文字符"))
        elif tex.count("{") != tex.count("}"):
            suspicious.append((r["file"], "花括号不配平"))
        elif "/" in tex and "\\frac" not in tex and "\\ " not in tex:
            suspicious.append((r["file"], "疑似用 a/b 代替 \\frac"))
    if suspicious:
        problems.append(("LaTeX 可疑", [f"{f}（{why}）" for f, why in suspicious]))

    parse_fail = []
    parsed = 0
    for r in rows:
        ok = try_parse((r.get("latex") or "").strip())
        if ok is False:
            parse_fail.append(r["file"])
        elif ok is True:
            parsed += 1
    if parse_fail:
        problems.append(("latex2sympy 解析失败", parse_fail))

    train_tex = {r["latex"].strip() for r in rows if "train" in r.get("split", "")}
    test_tex = {r["latex"].strip() for r in rows if "test" in r.get("split", "")}
    leak = sorted(train_tex & test_tex)
    if leak:
        problems.append(("训练/测试公式重复（泄漏）", leak))

    print(f"登记 {len(rows)} 条，图片文件 {len(images)} 个\n")

    split_count = Counter(r.get("split", "").strip() or "未填" for r in rows)
    type_count = Counter(r.get("type", "").strip() or "未填" for r in rows)
    print("按 split：", dict(split_count))
    print("按题型：", dict(type_count))
    if parsed:
        print(f"LaTeX 可解析：{parsed}/{len(rows)}")
    print()

    if not problems:
        print("检查通过，没有发现问题。")
        return

    for title, items in problems:
        print(f"[{title}] {len(items)} 项")
        for item in items[:20]:
            print("   ", item)
        if len(items) > 20:
            print(f"    ... 另有 {len(items) - 20} 项")
        print()


if __name__ == "__main__":
    main()
