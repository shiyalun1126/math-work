"""拍照质量自检：拍完先跑这个，别等 VLM 读不出来才发现。

判据全部来自 P0 八页的实测，不是拍脑袋的阈值：

- **锐度**：行图内相邻像素平均梯度。字迹清楚的行在 3.3 以上，
  W13 那种失焦页只有 1.2，VLM 读出来是 `x^v=x`。
- **墨占比**：判为笔画的像素占行面积。真写字 0.10~0.22；
  切到纸边 / 桌面阴影时会到 0.33 上下，那种行读出来的必是幻觉。
- **压缩痕迹**：文件大小 / 像素数。微信普通发送会把 4000×3000 压到
  960×1280、30KB，笔画细节在那一步就没了，后面算法再准也救不回来。

用法：
    F:/PyCharm/python.exe tools/check_photo_quality.py            # 查 data/inbox
    F:/PyCharm/python.exe tools/check_photo_quality.py --md       # 另存报告
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mathagent.lines import load_gray, split_page  # noqa: E402

# 实测标定：清晰行 ≥3.3，失焦行 ~1.2。2.5 落在中间，宁可提示也不漏。
SHARP_MIN = 2.5
# 真写字 ≤0.22，阴影带 ≥0.33。
INK_SHADOW = 0.28
# 原图短边低于这个值，切出来的单行字高不够认。微信压缩后的典型值是 960。
SIDE_MIN = 1400
# 每像素字节数：低于 0.06 基本是被压过的。原图一般 0.2 以上。
BYTES_PER_PX_MIN = 0.06


def sharpness(gray: np.ndarray) -> float:
    a = gray.astype(np.float32)
    return float(np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean())


def ink_share(gray: np.ndarray) -> float:
    a = gray.astype(np.float32)
    thr = a.mean() - 0.5 * a.std()
    return float((a < thr).mean())


def check_one(path: Path) -> dict:
    size = path.stat().st_size
    gray = load_gray(path)
    h, w = gray.shape
    px = h * w

    rec: dict = {
        "page": path.stem,
        "w": w,
        "h": h,
        "kb": size / 1024,
        "bpp": size / px,
        "lines": [],
        "flags": [],
    }

    if min(h, w) < SIDE_MIN:
        rec["flags"].append(
            f"分辨率只有 {w}×{h}：多半是微信普通发送压过的。原图短边应 ≥{SIDE_MIN}。")
    if rec["bpp"] < BYTES_PER_PX_MIN:
        rec["flags"].append(
            f"压缩过重（{rec['bpp']:.3f} 字节/像素）：笔画边缘已被抹平，重传原图。")

    try:
        page = split_page(path)
    except Exception as exc:  # noqa: BLE001
        rec["flags"].append(f"切行失败：{exc}")
        return rec

    for crop in page.crops:
        g = np.asarray(crop.image.convert("L"))
        if g.size == 0:
            continue
        rec["lines"].append({
            "index": crop.index,
            "box": crop.box,
            "sharp": sharpness(g),
            "ink": ink_share(g),
        })

    if not rec["lines"]:
        rec["flags"].append("一行都没切出来")
        return rec

    blurry = [ln for ln in rec["lines"] if ln["sharp"] < SHARP_MIN]
    shadow = [ln for ln in rec["lines"] if ln["ink"] > INK_SHADOW]
    if blurry:
        idx = "、".join(str(ln["index"]) for ln in blurry)
        worst = min(ln["sharp"] for ln in blurry)
        rec["flags"].append(f"第 {idx} 行偏糊（最低锐度 {worst:.1f}，需 ≥{SHARP_MIN}）")
    if shadow:
        idx = "、".join(str(ln["index"]) for ln in shadow)
        rec["flags"].append(f"第 {idx} 行切到了阴影 / 纸边，不是字（墨占比 >{INK_SHADOW}）")

    return rec


def verdict(rec: dict) -> str:
    """三档：能交 / 建议重拍 / 必须重拍。"""
    if not rec["lines"]:
        return "重拍"
    hard = [f for f in rec["flags"] if "分辨率" in f or "压缩" in f]
    if hard:
        return "重拍"
    ratio = sum(1 for ln in rec["lines"]
                if ln["sharp"] < SHARP_MIN or ln["ink"] > INK_SHADOW) / len(rec["lines"])
    if ratio >= 0.4:
        return "重拍"
    if ratio > 0:
        return "建议重拍"
    return "能交"


def render(recs: list[dict]) -> str:
    out = ["# 拍照质量自检", ""]
    buckets: dict[str, list[str]] = {"重拍": [], "建议重拍": [], "能交": []}
    out.append("| 页 | 尺寸 | 大小 | 行数 | 锐度（最低） | 判定 | 问题 |")
    out.append("|---|---|---|---|---|---|---|")
    for r in recs:
        v = verdict(r)
        buckets[v].append(r["page"])
        sharps = [ln["sharp"] for ln in r["lines"]] or [0]
        issues = "；".join(r["flags"]) or "—"
        out.append(
            f"| {r['page']} | {r['w']}×{r['h']} | {r['kb']:.0f} KB | {len(r['lines'])} "
            f"| {min(sharps):.1f} | **{v}** | {issues} |")
    out += ["", "## 结论", ""]
    for k in ("重拍", "建议重拍", "能交"):
        if buckets[k]:
            out.append(f"- **{k}**（{len(buckets[k])}）：{'、'.join(buckets[k])}")
    out += ["", "判定口径：锐度 <2.5 或 墨占比 >0.28 的行算不合格行；"
                "不合格行占四成以上整页重拍；命中分辨率 / 压缩告警直接重拍。", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="拍照质量自检")
    ap.add_argument("--dir", default=str(ROOT / "data" / "inbox"))
    ap.add_argument("--md", action="store_true", help="写 reports/photo_quality.md")
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.dir).glob("*.jpg"))
    if not paths:
        print(f"{args.dir} 里没有 jpg")
        return 0

    recs = [check_one(p) for p in paths]
    md = render(recs)
    print(md)

    if args.md:
        out = ROOT / "reports" / "photo_quality.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\n报告 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
