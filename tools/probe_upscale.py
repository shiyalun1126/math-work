"""细长行图的放大实验：只测"放大后 VLM 能不能看见上标/小字"。

与 .cache/probe_zoom.py 的区别：那个走 recognize()，图会先被 prepare_image 按
max_side=1280 缩回去，2x 实际只拿到 1.9x，3x 更是被缩到 0.47x——测了个寂寞。
这里直接编码原图送 _one()，绕开缓存和缩放，变量只剩"像素尺寸"。

用法：
    python tools/probe_upscale.py W04:1 W12:3
"""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mathagent.perceive import build_ensemble  # noqa: E402

ZOOMS = (1, 2, 3)
MIN_HEIGHT = 96     # 目标：放大后字形高度不低于这个数


def encode(im: Image.Image, quality: int = 95) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    recs = build_ensemble(k=1)
    print("识别器:", [r.name for r in recs])
    for spec in sys.argv[1:]:
        pid, idx = spec.split(":")
        img = ROOT / "data" / "inbox" / f"{pid}.jpg"
        from mathagent.lines import split_page

        sp = split_page(img)
        crop = sp.crops[int(idx) - 1].image.convert("RGB")
        print(f"\n=== {pid} 行{idx} 原尺寸 {crop.size} ===")
        for z in ZOOMS:
            target = max(z, MIN_HEIGHT / crop.height)
            im = crop.resize(
                (int(crop.width * target), int(crop.height * target)), Image.LANCZOS
            )
            b64 = encode(im)
            out = []
            for r in recs:
                try:
                    got = r._one(b64, 0.2)
                except Exception as exc:  # noqa: BLE001
                    got = f"ERR {type(exc).__name__}: {exc}"
                out.append(f"{r.name}:{got!r}")
                time.sleep(0.5)
            print(f"  {target:4.1f}x {im.size[0]}x{im.size[1]:<5d} " + " | ".join(out))


if __name__ == "__main__":
    main()
