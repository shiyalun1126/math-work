"""核对"仓库里到底有哪些图、哪些含真人作业、授权核没核过"。

为什么需要它：隐私问题不是某一次提交的事，而是"以后每次提交都别再带进来"。
靠人记住哪些目录要 ignore 不可靠——`data/raw/` 被 ignore 了，但拍照链路
会在 `reports/` 和 `data/` 下留派生图、切行小图、识别文本快照，这些一样能
还原出"谁写了什么、错在哪"。所以把已跟踪的图片和文本快照逐个登记，
再用这个脚本卡住"新增了没登记的资产"这件事。

三道检查：
1. 已跟踪的图片里，有没有 `data/assets.csv` 没登记的 → 报错（新增资产必须登记来源）
2. 登记表里写死的路径，仓库里还有没有 → 警告（图删了但表没改）
3. 运行时产物（错题本、当次采集快照、原图目录）是不是都被 ignore → 报错

授权这一列只能由人来填：`status=pending` 表示"已登记、发布权还没确认"。
本地跑只警告，`--strict` 才当失败——CI / 交付前用 strict，把待核实的项逼出来。

用法：
    F:/PyCharm/python.exe tools/check_assets.py
    F:/PyCharm/python.exe tools/check_assets.py --strict
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "data" / "assets.csv"

# 会被"看一眼就知道是谁"的后缀：这些必须登记
IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
# 含识别文本、判定结果的快照也当资产：没有原图，文本本身也是个人信息
TEXT_SUFFIX = {".jsonl"}

# 运行时产物：必须被 gitignore 挡住，否则下一次 git add -A 就带出去了
MUST_IGNORE = [
    "data/cases.jsonl",
    "data/cases_backup.jsonl",
    "data/history.jsonl",
    "data/photo_results.jsonl",
    "data/raw/W01.jpg",
    "data/crops/x.png",
    "data/inbox/W01.jpg",
    "data/test/eq_001.jpg",
    "data/self/eq_001.jpg",
]

FIELDS = ["path", "kind", "generator", "contains_personal", "status", "note"]
KINDS = {"synthetic", "screenshot", "photo_derived", "text_snapshot"}
PERSONAL = {"no", "yes", "unknown"}
STATUS = {"ok", "pending"}


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"git {' '.join(args)} 失败")
    return out.stdout


def tracked_files() -> list[str]:
    return [x for x in git("ls-files").splitlines() if x.strip()]


def load_table() -> list[dict]:
    if not TABLE.exists():
        raise SystemExit(f"找不到 {TABLE}")
    # utf-8-sig：这张表是给人改的，Excel 另存会加 BOM
    with TABLE.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != FIELDS:
            raise SystemExit(f"{TABLE} 表头应是 {FIELDS}，实际 {reader.fieldnames}")
        # 空行不算错，直接跳过：手工维护的表尾常留一个换行
        rows = [r for r in reader if any((v or "").strip() for v in r.values())]
    bad = [r for r in rows if not (r.get("path") or "").strip()]
    if bad:
        raise SystemExit(f"{TABLE} 有一行没写 path，补上或删掉")
    return rows


def matched(path: str, patterns: list[str]) -> bool:
    """精确路径或通配都算登记过。通配让"以后再生成 v11"不用改表。"""
    for p in patterns:
        if p == path or ("*" in p and fnmatch.fnmatch(path, p)):
            return True
    return False


def is_asset(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIX:
        return True
    return suffix in TEXT_SUFFIX and path.startswith("data/")


def check_rows(rows: list[dict]) -> list[str]:
    errs: list[str] = []
    for i, r in enumerate(rows, start=2):  # 2 = 表头下一行
        if r.get("kind") not in KINDS:
            errs.append(f"第 {i} 行 kind={r.get('kind')!r} 不在 {sorted(KINDS)}")
        if r.get("contains_personal") not in PERSONAL:
            errs.append(f"第 {i} 行 contains_personal={r.get('contains_personal')!r} "
                        f"不在 {sorted(PERSONAL)}")
        if r.get("status") not in STATUS:
            errs.append(f"第 {i} 行 status={r.get('status')!r} 不在 {sorted(STATUS)}")
        if r.get("status") == "ok" and r.get("contains_personal") == "unknown":
            errs.append(f"第 {i} 行（{r.get('path')}）标了 ok 却说 contains_personal=unknown，"
                        "先确认再填")
    return errs


def check_coverage(rows: list[dict], files: list[str]) -> list[str]:
    pats = [r["path"] for r in rows]
    missing = [f for f in files if is_asset(f) and not matched(f, pats)]
    return [f"已跟踪但没登记：{f}（在 data/assets.csv 加一行，写清来源和授权）"
            for f in missing]


def check_stale(rows: list[dict], files: list[str]) -> list[str]:
    tracked = set(files)
    skip = set(MUST_IGNORE)
    out = []
    for r in rows:
        p = r["path"]
        if p in skip:
            # 运行时产物本来就不入库：本地还没生成（比如新建的归档文件）
            # 或已经清空，都不是"表该清理"的信号。
            continue
        if "*" in p or p in tracked:
            continue
        if (ROOT / p).exists():
            out.append(f"{p} 在本地存在但没进 git（要不要入库自己定，表里可以先删）")
        else:
            out.append(f"{p} 表里有、仓库里没有（图删了就把它从表里删掉）")
    return out


def check_ignore() -> list[str]:
    errs = []
    for path in MUST_IGNORE:
        got = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT,
                             capture_output=True)
        if got.returncode != 0:
            errs.append(f"{path} 没被 ignore：运行时产物一旦进库就带出去了")
    return errs


def decide(errs: list[str], pending: list[str], strict: bool) -> int:
    """退出码只在这里定：错误必失败；pending 只在 --strict 下失败。"""
    if errs:
        return 1
    if strict and pending:
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="核对资产来源与隐私")
    ap.add_argument("--strict", action="store_true",
                    help="存在 status=pending 就算失败（交付/CI 用）")
    args = ap.parse_args()

    try:
        files = tracked_files()
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"读不到 git 索引：{exc}", file=sys.stderr)
        return 2

    rows = load_table()
    errs = check_rows(rows) + check_coverage(rows, files)
    warns = check_stale(rows, files) + check_ignore()
    pending = [r["path"] for r in rows if r["status"] == "pending"]

    for w in warns:
        print(f"警告  {w}")
    for e in errs:
        print(f"错误  {e}")
    if pending:
        print(f"\n待核实发布权 {len(pending)} 项（status=pending）：")
        for p in pending:
            print(f"  - {p}")
        print("  核实后在 data/assets.csv 把 status 改成 ok；不同意发布就把文件从 git 撤出。")

    if errs:
        print(f"\n{len(errs)} 个错误，{len(warns)} 个警告。")
        return decide(errs, pending, args.strict)
    if args.strict and pending:
        print(f"\n--strict：还有 {len(pending)} 项发布权未确认。")
        return decide(errs, pending, args.strict)
    print(f"\n已登记 {len(rows)} 条，覆盖已跟踪资产，{len(warns)} 个警告。")
    return decide(errs, pending, args.strict)


if __name__ == "__main__":
    sys.exit(main())
