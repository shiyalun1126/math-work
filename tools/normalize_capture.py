"""采集照片规范化：统一命名 + 拍摄方向校正（可审计、可重跑、可撤销）。

只做两件**不改变笔迹内容**的事：

1. 文件名统一成清单要求的大写 `W01.jpg`（用户上传的是 `w01.jpg`）。
2. 按 `data/capture_orientation.csv` 的人工确认表，把横拍页逆时针转正 90°。

明确不做：裁剪、去噪、增强对比度、抹箭头、抹阴影、调切行参数。
任何改变笔迹或掩盖缺陷的操作都属造假，不在这里发生。

原始上传**逐个备份**到 `data/inbox_raw/W##.jpg`（原样字节，未旋转），
之后每次运行都以备份为源重建，因此重复执行是幂等的，不会转两次。
每页的动作写进 `reports/capture_normalize.json`，含自动判据与人工结论的对比。

    python tools/normalize_capture.py              # dry-run，只看计划，不写文件
    python tools/normalize_capture.py --apply      # 实际生成规范视图
    python tools/normalize_capture.py --json       # 机器可读的摘要
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INBOX = ROOT / "data" / "inbox"
RAW = ROOT / "data" / "inbox_raw"
ORIENT = ROOT / "data" / "capture_orientation.csv"
REPORT = ROOT / "reports" / "capture_normalize.json"
EXTS = (".jpg", ".jpeg", ".png")
PAT = re.compile(r"^[wW](\d{2})$")


def load_orientation() -> dict[str, dict]:
    if not ORIENT.exists():
        return {}
    with ORIENT.open("r", encoding="utf-8-sig", newline="") as f:
        return {r["id"].strip().upper(): r for r in csv.DictReader(f)
                if (r.get("id") or "").strip()}


def auto_hint(path: Path) -> tuple[int, dict]:
    """只用行间空白率判断「横排还是竖排」，不判断正反。

    PIL 的投影对上下翻转不敏感，所以本判据只能区分 0°/180° 组与 90°/270° 组；
    正反由人工确认表决定。返回值 0 或 90，以及四个方向的原始得分（便于审计）。
    """
    import numpy as np
    from PIL import Image
    from mathagent.lines import binarize, clear_long_runs, row_profile

    im = Image.open(path).convert("L")
    scores: dict[str, float] = {}
    for deg in (0, 90, 180, 270):
        g = np.asarray(im.rotate(deg, expand=True), dtype=np.uint8)
        ink = clear_long_runs(binarize(g))
        prof = row_profile(ink, smooth=max(3, int(ink.shape[0] * 0.004)))
        if prof.max() <= 0:
            scores[str(deg)] = 0.0
            continue
        q = prof / prof.max()
        scores[str(deg)] = round(float((q < 0.06).mean()), 4)
    return (0 if scores["0"] >= scores["90"] else 90), scores


def discover() -> list[tuple[str, Path]]:
    """找出 inbox 里的页，返回 (W##, 当前路径)。备份存在时以备份为源，保证幂等。"""
    found: dict[str, Path] = {}
    for p in sorted(INBOX.iterdir()):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        m = PAT.match(p.stem)
        if not m:
            continue
        wid = f"W{m.group(1)}"
        backup = RAW / f"{wid}.jpg"
        found[wid] = backup if backup.exists() else p
    return sorted(found.items())


def plan(orient: dict[str, dict]) -> list[dict]:
    items = []
    for wid, src in discover():
        row = orient.get(wid, {})
        deg = int(row.get("rotate_ccw_deg") or 0)
        hint, scores = auto_hint(src)
        items.append({
            "id": wid,
            "source": str(src.relative_to(ROOT)),
            "target": f"data/inbox/{wid}.jpg",
            "backup": f"data/inbox_raw/{wid}.jpg",
            "rotate_ccw_deg": deg,
            "confirmed_by": row.get("confirmed_by", "未确认"),
            "reason": row.get("reason", ""),
            "auto_hint": hint,
            "auto_scores": scores,
            "auto_agrees": (hint == 0 and deg == 0) or (hint == 90 and deg in (90, 270)),
            "orientation_confirmed": wid in orient,
        })
    return items


def apply(items: list[dict]) -> list[dict]:
    from PIL import Image, ImageOps

    INBOX.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    for it in items:
        src = ROOT / it["source"]
        backup = ROOT / it["backup"]
        target = ROOT / it["target"]
        # 原始上传只备份一次，之后始终以它为准重建（幂等）
        if not backup.exists():
            shutil.copy2(src, backup)
            it["backup_created"] = True
        else:
            it["backup_created"] = False

        tmp = INBOX / f".normalize_{it['id']}.tmp.jpg"
        deg = it["rotate_ccw_deg"]
        if deg == 0:
            # 不用 PIL 重编码，字节级复制，笔迹完全不变
            with backup.open("rb") as fin, tmp.open("wb") as fout:
                shutil.copyfileobj(fin, fout)
        else:
            im = ImageOps.exif_transpose(Image.open(backup)).convert("RGB")
            im.rotate(deg, expand=True).save(tmp, format="JPEG", quality=95, subsampling=0)

        # 先删旧名（Windows 大小写不敏感，w01.jpg 与 W01.jpg 是同一实体），再落规范名
        for p in INBOX.iterdir():
            if p.is_file() and PAT.match(p.stem) and p.stem.upper() == it["id"] and p != tmp:
                p.unlink()
        tmp.replace(target)
        it["applied"] = True
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="采集照片规范化（命名 + 拍摄方向）")
    ap.add_argument("--apply", action="store_true", help="实际写文件；默认只预览")
    ap.add_argument("--json", action="store_true", help="打印机器可读摘要")
    args = ap.parse_args(argv)

    orient = load_orientation()
    items = plan(orient)
    if args.apply:
        items = apply(items)

    disagree = [i["id"] for i in items if not i["auto_agrees"]]
    unconfirmed = [i["id"] for i in items if not i["orientation_confirmed"]]
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "applied": bool(args.apply),
        "n_pages": len(items),
        "n_rotated": sum(1 for i in items if i["rotate_ccw_deg"] != 0),
        "auto_agrees": len(items) - len(disagree),
        "auto_disagree": disagree,
        "orientation_unconfirmed": unconfirmed,
        "note": "自动判据只分横竖不分正反；方向以 data/capture_orientation.csv 人工确认为准。"
                "原图字节保存在 data/inbox_raw/，规范视图可随时重建。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"summary": summary, "items": items},
                                 ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"[采集照片规范化] {'已执行' if args.apply else 'DRY-RUN（未写文件）'} · {len(items)} 页")
        for i in items:
            print(f"  {i['id']}  {Path(i['source']).name:<12} → {Path(i['target']).name:<10} "
                  f"旋转 {i['rotate_ccw_deg']:>3d}°  自动判据 {i['auto_hint']:>3d}° "
                  f"{'一致' if i['auto_agrees'] else '不一致 ← 以人工为准'}")
        print(f"  需转正 {summary['n_rotated']} 页；自动判据与人工一致 "
              f"{summary['auto_agrees']}/{len(items)}；不一致 {', '.join(disagree) or '无'}")
        if unconfirmed:
            print(f"  ⚠ 方向未经人工确认（按 0° 处理）：{', '.join(unconfirmed)}")
        print(f"  审计记录 → {REPORT.relative_to(ROOT)}")
        if not args.apply:
            print("  加 --apply 才实际生成；原图会先备份到 data/inbox_raw/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
