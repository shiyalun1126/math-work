"""拍摄视图增强：把手机暗光拍出来的灰纸提成白纸，让笔画回到该有的对比度。

**这不是修饰，是补偿拍摄条件。** 界限划得很清：

做的事——
- **光照拉平**：除以大尺度背景（纸面无论多暗都拉到同一亮度），消除暗角、
  单侧阴影、顶灯造成的亮度梯度。笔画的相对位置、粗细、连笔关系一个像素都不动。
- **轻度去噪 + 锐化**：抵消 JPEG 块噪和手机降噪抹掉的边缘。强度只到刚够，
  过冲产生的光晕会被切行当成墨迹（实测 percent≥160 时墨占比从 0.15 涨到 0.48）。

不做的事——擦箭头、擦阴影、抹涂黑、补笔画、裁掉拍进来的相邻行、改任何一个字。
那些是造假，一旦做了，评测数字就不再代表真实手写的可识别性。

**基准线保留**：未增强的规范化视图存一份在 `data/inbox_plain/`，
`data/inbox/` 放增强版。谁要核对"原图长什么样"，看 plain。

始终以 plain 为源，因此重复执行幂等，不会层层叠加锐化。

    F:/PyCharm/python.exe tools/enhance_capture.py            # dry-run
    F:/PyCharm/python.exe tools/enhance_capture.py --apply    # 生成
    F:/PyCharm/python.exe tools/enhance_capture.py --apply --md
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mathagent.lines import load_gray  # noqa: E402

INBOX = ROOT / "data" / "inbox"
PLAIN = ROOT / "data" / "inbox_plain"
REPORT = ROOT / "reports" / "capture_enhance.json"

# 参数是按 P0 八页实测选的：拉平给对比度，轻锐化补边缘，不做放大。
# 放大到 2x 反而让"相邻像素梯度"这个指标掉一半（梯度被摊到更多像素上），
# 视觉上字更大，但对 VLM 没有实测收益，还多花一倍时间。
FLAT_RADIUS = 0.18      # 背景估计核半径 = 短边 × 0.18
MEDIAN = 3              # 中值半径：去 JPEG 块噪
UNSHARP_RADIUS = 1.0
UNSHARP_PERCENT = 70
AUTOCONTRAST_CUTOFF = 1


def flat_field(gray: np.ndarray, radius_ratio: float = FLAT_RADIUS) -> np.ndarray:
    """除以大尺度背景，抹平不均匀光照。"""
    r = max(9, int(min(gray.shape) * radius_ratio))
    bg = np.asarray(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(r)), dtype=np.float32)
    bg = np.maximum(bg, 1.0)
    out = gray.astype(np.float32) / bg
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    out = (out - lo) / max(hi - lo, 1e-6)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def enhance(path: Path) -> Image.Image:
    gray = load_gray(path)
    im = Image.fromarray(flat_field(gray))
    im = im.filter(ImageFilter.MedianFilter(MEDIAN))
    im = ImageOps.autocontrast(im, cutoff=AUTOCONTRAST_CUTOFF)
    im = im.filter(ImageFilter.UnsharpMask(
        radius=UNSHARP_RADIUS, percent=UNSHARP_PERCENT, threshold=2))
    return im


def quality(path: Path) -> dict:
    """与 check_photo_quality 同口径：对比度、锐度（缩到 900 长边后）、文件大小。"""
    gray = load_gray(path)
    im = Image.fromarray(gray)
    im.thumbnail((900, 900), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    return {
        "w": int(gray.shape[1]),
        "h": int(gray.shape[0]),
        "kb": round(path.stat().st_size / 1024, 1),
        "contrast": round(float(np.percentile(gray, 98) - np.percentile(gray, 2)), 1),
        "sharpness": round(float(np.abs(np.diff(a, axis=0)).mean()
                                + np.abs(np.diff(a, axis=1)).mean()), 2),
    }


def seed_plain() -> int:
    """第一次跑：把当前的 inbox 视图存成 plain 基准。之后不再覆盖。"""
    PLAIN.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(INBOX.glob("W*.jpg")):
        dst = PLAIN / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
            n += 1
    return n


def plan() -> list[dict]:
    items = []
    for src in sorted(PLAIN.glob("W*.jpg")):
        items.append({"id": src.stem, "src": src, "dst": INBOX / src.name,
                      "before": quality(src)})
    return items


def apply(items: list[dict]) -> list[dict]:
    INBOX.mkdir(parents=True, exist_ok=True)
    out = []
    for it in items:
        im = enhance(it["src"])
        im.save(it["dst"], quality=95, optimize=True)
        rec = dict(it["after"] if "after" in it else {})
        rec.update({
            "id": it["id"],
            "before": it["before"],
            "after": quality(it["dst"]),
            "gain_sharpness": round(quality(it["dst"])["sharpness"]
                                    - it["before"]["sharpness"], 2),
        })
        out.append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="拍摄视图增强")
    ap.add_argument("--apply", action="store_true", help="实际写盘")
    ap.add_argument("--md", action="store_true", help="另存 markdown 报告")
    ap.add_argument("--json", action="store_true", help="输出机器可读摘要")
    args = ap.parse_args(argv)

    seeded = seed_plain() if args.apply else 0
    items = plan()
    if not items:
        print("data/inbox 里没有 W*.jpg，先跑 normalize_capture.py")
        return 1

    if not args.apply:
        print(f"[dry-run] 待增强 {len(items)} 页（源 data/inbox_plain）")
        for it in items:
            b = it["before"]
            print(f"  {it['id']}  {b['w']}x{b['h']}  对比度 {b['contrast']:5.0f}  锐度 {b['sharpness']:5.2f}")
        print("\n加 --apply 实际生成")
        return 0

    recs = apply(items)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"plain_seeded": seeded, "pages": recs},
                                 ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[拍摄视图增强] 已生成 {len(recs)} 页 · 基准 {PLAIN.name}/")
    print(f"  {'页':5s} {'对比度':>13s} {'锐度':>13s}")
    for r in recs:
        b, a = r["before"], r["after"]
        print(f"  {r['id']:5s} {b['contrast']:5.0f}→{a['contrast']:<5.0f}"
              f"  {b['sharpness']:5.2f}→{a['sharpness']:<5.2f}  ({r['gain_sharpness']:+.2f})")

    if args.md:
        md = ["# 拍摄视图增强", "",
              f"源 `data/inbox_plain/` → 输出 `data/inbox/`，{len(recs)} 页。", "",
              "| 页 | 尺寸 | 对比度 | 锐度 | 大小 |", "|---|---|---|---|---|"]
        for r in recs:
            a = r["after"]
            md.append(f"| {r['id']} | {a['w']}×{a['h']} | {r['before']['contrast']:.0f} → "
                      f"{a['contrast']:.0f} | {r['before']['sharpness']:.2f} → "
                      f"{a['sharpness']:.2f} | {a['kb']:.0f} KB |")
        md += ["", "口径：对比度 = 灰度 P98−P2；锐度 = 缩到 900 长边后相邻像素平均梯度，"
                   "与 `check_photo_quality.py` 一致。", "",
               "只做光照拉平 + 轻度锐化，未擦除、未补笔、未裁剪。原始上传在 `data/inbox_raw/`。"]
        (ROOT / "reports" / "capture_enhance.md").write_text("\n".join(md), encoding="utf-8")
        print("  报告 → reports/capture_enhance.md")

    if args.json:
        print(json.dumps(recs, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
