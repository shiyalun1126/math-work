"""单行小图的预处理对照：二值化 / 加粗 / 反锐化 / 拉伸，看哪种能让模糊的字形变清楚。

只回答一个问题：这一行读不出来，是"预处理没做到"还是"模型就这样"。
两种都救不回来时就该承认这行认不准，别继续在参数上打转。
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mathagent.perceive import build_ensemble  # noqa: E402
from mathagent.lines import split_page  # noqa: E402


def encode(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def binarize(im: Image.Image, block: int = 31, k: float = 0.10) -> Image.Image:
    g = np.asarray(im.convert("L"), dtype=np.float32)
    bg = np.asarray(
        Image.fromarray(g.astype(np.uint8)).filter(ImageFilter.GaussianBlur(block)),
        dtype=np.float32,
    )
    ink = (bg - g) > np.maximum(6.0, k * bg)
    return Image.fromarray(np.where(ink, 0, 255).astype(np.uint8))


def stretch(im: Image.Image, lo: int = 5, hi: int = 95) -> Image.Image:
    g = np.asarray(im.convert("L"))
    a, b = np.percentile(g, lo), np.percentile(g, hi)
    out = np.clip((g - a) * 255.0 / max(1e-6, b - a), 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def variants(im: Image.Image) -> dict[str, Image.Image]:
    x4 = im.resize((im.width * 4, im.height * 4), Image.LANCZOS)
    b = binarize(x4)
    return {
        "4x": x4,
        "4x+拉伸": stretch(x4),
        "4x+二值": b,
        "4x+二值+加粗": b.filter(ImageFilter.MinFilter(3)),
        "4x+反锐化": x4.filter(
            ImageFilter.UnsharpMask(radius=6, percent=180, threshold=2)
        ),
    }


def main() -> None:
    recs = build_ensemble(k=1)
    for spec in sys.argv[1:]:
        pid, idx = spec.split(":")
        sp = split_page(ROOT / "data" / "inbox" / f"{pid}.jpg")
        im = sp.crops[int(idx) - 1].image.convert("RGB")
        print(f"\n=== {pid} 行{idx} {im.size} ===")
        for name, v in variants(im).items():
            b64 = encode(v)
            got = []
            for r in recs:
                try:
                    got.append(f"{r.name}:{r._one(b64, 0.2)!r}")
                except Exception as exc:  # noqa: BLE001
                    got.append(f"{r.name}:ERR {type(exc).__name__}")
            print(f"  {name:<12} {' | '.join(got)}")


if __name__ == "__main__":
    main()
