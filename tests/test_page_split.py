"""整页模式（实拍作业照）的回归测试。

背景：`split_page` 原来是给"裁剪好的公式块"写的，用全局 Otsu 判墨迹。
实拍整页照里桌面和阴影比纸暗，会被判成墨迹，行投影糊掉——P0 八页实测只有 1 页
行数对得上（W05 整页并成 1 行、W01 把桌沿切成一行）。这里把修好之后的判据钉住。

合成图不追求逼真，只覆盖三条会被打破的判据：
深色桌面不能被当成笔画、纸边的硬边缘不能被当成一行、行内断笔不能裂成两行。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from mathagent.lines import (
    MIN_LINE_H,
    PAGE_MODE_MIN_SIDE,
    drop_thick_blobs,
    highpass_ink,
    load_gray,
    split_page,
    wants_page_mode,
)


def make_page_photo(path, lines=(3,), size=(700, 900), shadow=True):
    """白纸放在深色桌面上，纸上写若干行"字"，右侧压一道阴影。"""
    w, h = size
    im = Image.new("L", size, 45)                     # 桌面：暗、均匀
    d = ImageDraw.Draw(im)
    d.rectangle((60, 80, w - 60, h - 60), fill=205)   # 纸：亮
    for li, y in enumerate([180 + i * 150 for i in range(lines[0])]):
        for k in range(6):                            # 每行若干笔段，中间留白
            x = 130 + k * 70
            d.rectangle((x, y, x + 40, y + 34), fill=25)
    if shadow:
        arr = np.asarray(im, dtype=np.float32)
        ramp = np.linspace(1.0, 0.55, w, dtype=np.float32)  # 从左到右变暗
        arr = arr * ramp[None, :]
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").save(path)
    return path


def test_page_mode_turns_on_for_page_photo_and_off_for_block():
    assert wants_page_mode((960, 1280)) is True
    assert wants_page_mode((PAGE_MODE_MIN_SIDE, PAGE_MODE_MIN_SIDE + 1)) is True
    assert wants_page_mode((220, 130)) is False       # M²E 那类公式块
    assert wants_page_mode((309, 222)) is False


def test_highpass_ignores_dark_desk(tmp_path):
    """桌面是缓变的暗面，高通后不该留下笔画；整页墨迹占比必须回到个位数百分比。"""
    path = make_page_photo(tmp_path / "page.jpg")
    gray = load_gray(path)
    ink = highpass_ink(gray)
    assert ink.mean() < 0.10, f"背景被当成墨迹了：ink={ink.mean():.3f}"


def test_drop_thick_blobs_removes_wide_dark_block():
    """整块墨迹（桌面反光、手影糊成一片）要整块剔掉，细笔画要留住。"""
    ink = np.zeros((400, 400), dtype=bool)
    ink[40:140, 60:340] = True        # 厚块：160×100 的实心区
    ink[250, 50:350] = True           # 细笔画：一条 1 像素高的横线
    ink[300:340, 50:52] = True        # 细笔画：一条 2 像素宽的竖线
    out = drop_thick_blobs(ink)
    assert not out[40:140, 60:340].any(), "厚块应该被剔掉"
    assert out[250, 50:350].any(), "细横线不能被误伤"
    assert out[300:340, 50:52].any(), "细竖线不能被误伤"


def test_page_mode_splits_three_lines(tmp_path):
    path = make_page_photo(tmp_path / "page.jpg")
    page = split_page(path)
    assert page.skew == 0.0, "整页模式不做纠偏，避免旋转插值把笔画糊掉"
    assert len(page.crops) == 3, [c.box for c in page.crops]
    hs = [c.box[3] - c.box[1] for c in page.crops]
    assert all(h >= MIN_LINE_H for h in hs)
    assert page.note == "", page.note


def test_page_mode_can_be_forced_off(tmp_path):
    """显式关掉就回到原来的全局 Otsu 路径，老参数与老数字不受影响。"""
    path = make_page_photo(tmp_path / "page.jpg")
    legacy = split_page(path, page_mode=False)
    assert legacy.skew != 0.0 or len(legacy.crops) != 3 or legacy.note


def test_blank_photo_reports_instead_of_silently_returning_nothing(tmp_path):
    path = tmp_path / "blank.jpg"
    im = Image.new("L", (700, 900), 200)
    im.convert("RGB").save(path)
    page = split_page(path)
    assert len(page.crops) == 0
    assert "没有切出任何一行" in page.note
