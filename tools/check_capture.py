"""拍完照的体检脚本：缺哪张、切得对不对、真值全不全、有没有泄漏。

用法：
    python tools/check_capture.py                    # 只看缺什么
    python tools/check_capture.py --split           # 额外实切一遍，核对行数
    python tools/check_capture.py --md              # 同时写 reports/capture_check.md

它不修改任何采集文件，只报告。设计原则：宁可报"缺"，也不猜"有"。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

DATA = ROOT / "data"
INBOX = DATA / "inbox"
CROPS = DATA / "crops"
EXTS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------- 基础工具


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_image(stem: str, folder: Path) -> Path | None:
    for ext in EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def file_kind(path: Path) -> str:
    """按文件头判格式。iPhone 把 HEIC 改后缀成 .jpg 是常见的坑。"""
    try:
        head = path.read_bytes()[:16]
    except OSError:
        return "读不了"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG"):
        return "png"
    if head[4:8] == b"ftyp":
        return "HEIC（后缀改了也没用，需转成 jpg）"
    return "未知格式"


def norm_latex(s: str) -> str:
    """泄漏比对用：去掉空白和花括号，不改动数学含义的近似归一。"""
    return "".join((s or "").split()).replace("{", "").replace("}", "")


# ---------------------------------------------------------------- 各项检查


def check_single_pages() -> tuple[list[str], dict]:
    """单行公式集：20 页整页照，按 pages.csv。"""
    out: list[str] = []
    pages = read_csv(DATA / "pages.csv")
    present, missing, bad_fmt = [], [], []
    for p in pages:
        name = (p.get("page") or "").strip()
        if not name:
            continue
        img = find_image(name, INBOX)
        if img is None:
            missing.append(name)
            continue
        present.append(name)
        kind = file_kind(img)
        if kind not in ("jpeg", "png"):
            bad_fmt.append(f"{img.name}: {kind}")

    out.append(f"[单行公式集] 整页照 {len(present)}/{len(pages)} 张")
    if missing:
        out.append(f"  ✗ 缺 {len(missing)} 张：{'、'.join(missing)}")
    if bad_fmt:
        out.append(f"  ✗ 格式异常 {len(bad_fmt)} 张：{'；'.join(bad_fmt)}")
    if not missing and not bad_fmt:
        out.append("  ✓ 页层齐了")

    # 切行后的单行小图 vs labels.csv
    labels = read_csv(DATA / "labels.csv")
    have_crop, lack_crop = 0, []
    for r in labels:
        rel = (r.get("file") or "").strip()
        if not rel:
            continue
        if (DATA / rel).exists():
            have_crop += 1
        else:
            lack_crop.append(rel)
    out.append(f"[单行公式集] 已落盘单行小图 {have_crop}/{len(labels)} 张")
    if lack_crop:
        out.append(f"  · 还差 {len(lack_crop)} 张（切行后由 route_crops.py 分流）")
        out.append(f"    例：{'、'.join(lack_crop[:5])}")

    return out, {"pages_total": len(pages), "pages_have": len(present),
                 "labels_total": len(labels), "crops_have": have_crop}


def check_free_write_truth() -> list[str]:
    """自由写 50 条没有 latex 真值，看图标注后必须补进 labels.csv。"""
    shot = read_csv(DATA / "shotlist.csv")
    labels = read_csv(DATA / "labels.csv")
    free = {r["id"].strip() for r in shot
            if (r.get("type") or "").strip() == "自由写" and not (r.get("latex") or "").strip()}
    labeled = {(r.get("file") or "").split("/")[-1].replace(".jpg", "")
               for r in labels}
    missing = sorted(free - labeled)
    out = [f"[真值] 自由写 {len(free)} 条，已补 latex 标注 {len(free) - len(missing)} 条"]
    if missing:
        out.append(f"  ✗ 还差 {len(missing)} 条没标：这些照片进了测试集也没有 ground truth")
        out.append("    看图把公式写进 labels.csv，latex 列；这是拍完最容易漏的一步")
    elif free:
        out.append("  ✓ 自由写真值齐了")
    return out


def check_leakage() -> list[str]:
    """同一公式不得同时出现在 train 和 test。"""
    labels = read_csv(DATA / "labels.csv")
    buckets: dict[str, set[str]] = defaultdict(set)
    for r in labels:
        key = norm_latex(r.get("latex") or "")
        if key:
            buckets[key].add((r.get("split") or "").strip())
    bad = [k for k, v in buckets.items() if len(v) > 1]
    out = [f"[泄漏] 归一化后不同公式 {len(buckets)} 个"]
    if bad:
        out.append(f"  ✗ {len(bad)} 个公式同时进了 train 和 test：{'、'.join(bad[:5])}")
    else:
        out.append("  ✓ 未发现 train/test 重复公式")
    return out


def check_work_pages(do_split: bool) -> tuple[list[str], dict]:
    """整页解题过程集：一页一题，页名 = id。"""
    rows = read_csv(DATA / "work_shotlist.csv")
    out: list[str] = []
    present, missing = [], []
    for r in rows:
        wid = (r.get("id") or "").strip()
        if not wid:
            continue
        img = find_image(wid, INBOX)
        if img is None:
            missing.append(wid)
        else:
            present.append((wid, img))

    by_priority = Counter((r.get("priority") or "").strip() for r in rows)
    p0 = [r["id"].strip() for r in rows if (r.get("priority") or "").strip() == "P0"]
    p0_missing = [x for x in p0 if x in missing]

    out.append(f"[解题过程集] 整页照 {len(present)}/{len(rows)} 张"
               f"（P0 {by_priority.get('P0', 0)} / P1 {by_priority.get('P1', 0)} / P2 {by_priority.get('P2', 0)}）")
    if p0_missing:
        out.append(f"  ✗ P0 最小可答辩集还缺：{'、'.join(p0_missing)}")
    if missing:
        out.append(f"  · 共缺 {len(missing)} 张：{'、'.join(missing)}")
    if not missing:
        out.append("  ✓ 过程页齐了")

    # 清单自身的真值一致性
    schema_bad = []
    for r in rows:
        lines = [x for x in (r.get("lines") or "").split("|") if x.strip()]
        expect = (r.get("expect_lines") or "").strip()
        if not expect.isdigit() or int(expect) != len(lines):
            schema_bad.append(f"{r.get('id')}: expect_lines={expect} 实际 {len(lines)} 行")
        if not (r.get("truth_grade") or "").strip() or not (r.get("baseline") or "").strip():
            schema_bad.append(f"{r.get('id')}: 缺 truth_grade / baseline")
    if schema_bad:
        out.append(f"  ✗ 清单字段不自洽：{'；'.join(schema_bad[:4])}")
    else:
        out.append("  ✓ 清单真值字段与行数自洽")

    # 行数校验：优先读 manifest，--split 时实切
    line_bad = []
    if present and do_split:
        from mathagent.lines import split_page  # noqa: PLC0415
        for wid, img in present:
            row = next((r for r in rows if (r.get("id") or "").strip() == wid), None)
            if row is None:
                continue
            try:
                n = len(split_page(str(img)).crops)
            except Exception as exc:  # noqa: BLE001
                line_bad.append(f"{wid}: 切行失败 {type(exc).__name__}")
                continue
            want = int(row.get("expect_lines") or 0)
            if n != want:
                line_bad.append(f"{wid}: 切出 {n} 行，期望 {want} 行（会错位，需重拍或调行距）")
    if do_split:
        if line_bad:
            out.append(f"  ✗ 切行核对 {len(line_bad)} 页不符：{'；'.join(line_bad[:5])}")
        elif present:
            out.append("  ✓ 实切行数全部对得上")
    else:
        out.append("  · 未做切行核对（加 --split 实切一遍）")

    return out, {"work_total": len(rows), "work_have": len(present),
                 "p0_total": len(p0), "p0_missing": len(p0_missing)}


# ---------------------------------------------------------------- 汇总


def main() -> int:
    ap = argparse.ArgumentParser(description="手写数据采集体检")
    ap.add_argument("--split", action="store_true", help="实切一遍过程页，核对行数")
    ap.add_argument("--md", action="store_true", help="同时写 reports/capture_check.md")
    args = ap.parse_args()

    blocks: list[str] = []
    stats: dict = {}
    for fn in (check_single_pages,):
        lines, st = fn()
        blocks.append("\n".join(lines))
        stats.update(st)
    blocks.append("\n".join(check_free_write_truth()))
    blocks.append("\n".join(check_leakage()))
    lines, st = check_work_pages(args.split)
    blocks.append("\n".join(lines))
    stats.update(st)

    body = "\n\n".join(blocks)
    print(body)

    if args.md:
        out = ROOT / "reports" / "capture_check.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "# 手写数据采集体检\n\n"
            "由 `tools/check_capture.py` 生成，每次拍完跑一遍。\n\n"
            "```\n" + body + "\n```\n",
            encoding="utf-8",
        )
        print(f"\n已写 {out.relative_to(ROOT)}")

    # 退出码：P0 缺页或格式异常算失败，便于接 CI
    return 1 if stats.get("p0_missing", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
