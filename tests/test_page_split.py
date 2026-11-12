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
    lead_fragment_end,
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


# ---------------------------------------------------------------- 越写越小的后半段
#
# 一页作业写到后面，字会变小、笔画变轻。`_candidate_bands` 的阈值是整页的
# `percentile(p, 80)`，它默认"每行墨迹都在同一水平"，于是后半段整段够不上，
# 切行少好几行——**而且不报错**，看起来就像学生那一页只写了一半。
# 实测那张二元一次方程组：10 行只切出 6 行，最终结论 `∴{x=1, y=2}` 整块不见了，
# 批改于是永远接不上"代入求值"那一步，只能判"整份认不出来"。


def make_page_with_fading_lines(path):
    """上半页字大笔画重，下半页（隔着一段空）字小笔画轻。"""
    im = Image.new("L", (700, 1200), 205)
    d = ImageDraw.Draw(im)
    for i in range(3):
        y = 120 + i * 120
        for k in range(6):
            d.rectangle((150 + k * 70, y, 150 + k * 70 + 40, y + 36), fill=25)
    for i in range(2):
        y = 720 + i * 110
        for k in range(3):
            d.rectangle((150 + k * 55, y, 150 + k * 55 + 26, y + 22), fill=98)
    im.convert("RGB").save(path)
    return path


def test_faint_tail_lines_are_recovered(tmp_path):
    path = make_page_with_fading_lines(tmp_path / "fade.jpg")
    page = split_page(path)
    assert len(page.crops) == 5, [c.box for c in page.crops]
    # 补回来的两行必须在原图上真的那两行的位置，不能是凭空多出来的碎片
    ys = [c.box[1] for c in page.crops]
    assert ys[-1] > 700, ys


def test_ruled_paper_is_not_split_into_its_rules():
    """横线本的格线不能被"补洞"补成一行行。

    格线是贯通整幅的直线，`clear_long_runs` 抹掉的是连续的那部分；
    手写字压在线上的地方抹不干净，残留的断线在局部低阈值下正好够成一条带。
    实测这页四行的作业被补成十六条等距的"行"，后面全错。
    """
    from pathlib import Path as _P

    fx = _P(__file__).resolve().parent.parent / "fixtures" / "work" / "work_fake.jpg"
    assert len(split_page(fx).crops) == 4, "横线格被当成行了"


# ---------------------------------------------------------------- 行首的括号残片
#
# 学生写方程组时那个 `{` 横跨两行，水平投影从中间把它切成上下两截，
# 各自混进相邻两行的小图里，被识别器读成 `2`、`1`、`)`、`7`。
# 于是 `2x+y=4` 读成 `22x+y=4`、`y=2` 读成 `7y=2`——
# 后者会被判成"这一步不成立"，而学生写的是对的。
# 这不是模型看错：那一竖真的在框里，模型如实说了它看见的东西。


def test_lead_fragment_is_trimmed():
    cols = np.zeros(400)
    cols[20:33] = 30          # 残片：宽 13，行高 47
    cols[62:100] = 20         # 主体
    assert lead_fragment_end(cols, 47) == 62


def test_tightly_set_coefficient_is_not_trimmed():
    """`1x` 的 1 紧挨着 x（间隙 2 px），不是残片，动了就把式子改了。"""
    cols = np.zeros(400)
    cols[20:30] = 20
    cols[32:70] = 20
    assert lead_fragment_end(cols, 47) == 20


def test_distant_noise_does_not_drag_the_box_left():
    """第一块是纸面噪点、第二块在两百多像素外时，不许跟着跳过去。

    跳过去的代价是把左边一大片（甚至内容）裁掉：W04 的「有」、W08 的「x=」
    就是这么整行退化成"读不出算式"的。
    """
    cols = np.zeros(800)
    cols[10:13] = 20          # 噪点
    cols[300:340] = 20        # 真正的内容
    assert lead_fragment_end(cols, 80) == 10


# ---------------------------------------------------------------- 越写越小的后半段
#
# 一页作业写到后面，字会变小、笔画变轻。`_candidate_bands` 的阈值是整页的
# `percentile(p, 80)`，它默认"每行墨迹都在同一水平"，于是后半段整段够不上，
# 切行少好几行——**而且不报错**，看起来就像学生那一页只写了一半。
# 实测那张二元一次方程组：10 行只切出 6 行，最终结论 `∴{x=1, y=2}` 整块不见了，
# 批改于是永远接不上"代入求值"那一步，只能判"整份认不出来"。


def make_page_with_fading_lines(path):
    """上半页字大笔画重，下半页（隔着一段空）字小笔画轻。"""
    im = Image.new("L", (700, 1200), 205)
    d = ImageDraw.Draw(im)
    for i in range(3):
        y = 120 + i * 120
        for k in range(6):
            d.rectangle((150 + k * 70, y, 150 + k * 70 + 40, y + 36), fill=25)
    for i in range(2):
        y = 720 + i * 110
        for k in range(3):
            d.rectangle((150 + k * 55, y, 150 + k * 55 + 26, y + 22), fill=98)
    im.convert("RGB").save(path)
    return path


def test_faint_tail_lines_are_recovered(tmp_path):
    path = make_page_with_fading_lines(tmp_path / "fade.jpg")
    page = split_page(path)
    assert len(page.crops) == 5, [c.box for c in page.crops]
    # 补回来的两行必须在原图上真的那两行的位置，不能是凭空多出来的碎片
    ys = [c.box[1] for c in page.crops]
    assert ys[-1] > 700, ys


def test_ruled_paper_is_not_split_into_its_rules():
    """横线本的格线不能被"补洞"补成一行行。

    格线是贯通整幅的直线，`clear_long_runs` 抹掉的是连续的那部分；
    手写字压在线上的地方抹不干净，残留的断线在局部低阈值下正好够成一条带。
    实测这页四行的作业被补成十六条等距的"行"，后面全错。
    """
    from pathlib import Path as _P

    fx = _P(__file__).resolve().parent.parent / "fixtures" / "work" / "work_fake.jpg"
    assert len(split_page(fx).crops) == 4, "横线格被当成行了"


# ---------------------------------------------------------------- 行首的括号残片
#
# 学生写方程组时那个 `{` 横跨两行，水平投影从中间把它切成上下两截，
# 各自混进相邻两行的小图里，被识别器读成 `2`、`1`、`)`、`7`。
# 于是 `2x+y=4` 读成 `22x+y=4`、`y=2` 读成 `7y=2`——
# 后者会被判成"这一步不成立"，而学生写的是对的。
# 这不是模型看错：那一竖真的在框里，模型如实说了它看见的东西。


def test_lead_fragment_is_trimmed():
    cols = np.zeros(400)
    cols[20:33] = 30          # 残片：宽 13，行高 47
    cols[62:100] = 20         # 主体
    assert lead_fragment_end(cols, 47) == 62


def test_tightly_set_coefficient_is_not_trimmed():
    """`1x` 的 1 紧挨着 x（间隙 2 px），不是残片，动了就把式子改了。"""
    cols = np.zeros(400)
    cols[20:30] = 20
    cols[32:70] = 20
    assert lead_fragment_end(cols, 47) == 20


def test_distant_noise_does_not_drag_the_box_left():
    """第一块是纸面噪点、第二块在两百多像素外时，不许跟着跳过去。

    跳过去的代价是把左边一大片（甚至内容）裁掉：W04 的「有」、W08 的「x=」
    就是这么整行退化成"读不出算式"的。
    """
    cols = np.zeros(800)
    cols[10:13] = 20          # 噪点
    cols[300:340] = 20        # 真正的内容
    assert lead_fragment_end(cols, 80) == 10
