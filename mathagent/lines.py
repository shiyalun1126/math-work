"""整页手写照片切行：把一张作业照切成一行一张小图。

这段算法原来只活在 `tools/split_page.py` 这个命令行工具里，批改链路要用它，
但批改跑在界面进程里，不方便去调一个 CLI，所以把算法搬进包里，
工具改成薄壳。两边共用同一份代码，改一处不会只改好一边。

原理（纯 numpy + Pillow，不依赖 opencv）：
    灰度化 → Otsu 二值化 → 抹掉横线本的直线 → 纠偏 → 水平投影找行带 → 切图
    行带过高（两行粘在一起）时在最深谷底再切一刀。

**它只负责"切"，不负责"切对"。** 切错一行，后面的识别和批改都会跟着错，
所以每次切完都带回来一个 `note`，把"这页可能不对"的话说出来，
而不是默默给出一堆图让人以为切得很准。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 低于这个高度的行带是墨点、污渍、横线本的断头，不是一行字。
MIN_LINE_H = 10

# 短边达到这个像素数就按"整页照"处理：画面里除了纸，还有桌面、阴影、纸边。
# M²E 那类紧凑公式块是 69×89~309×222，不会被这条规则卷进来。
PAGE_MODE_MIN_SIDE = 400


# ---------- 基础图像处理 ----------

def load_gray(path: Path) -> np.ndarray:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)          # 手机照片常带旋转标记
    im = im.convert("L")
    # 长边超过 2400 先缩下来，投影剖面对分辨率不敏感
    if max(im.size) > 2400:
        scale = 2400 / max(im.size)
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    return np.asarray(im)


def otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    sum_all = float(np.dot(np.arange(256), hist))
    w_b = 0.0
    sum_b = 0.0
    best, thr = -1.0, 127
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best, thr = between, i
    return thr


def binarize(gray: np.ndarray) -> np.ndarray:
    thr = otsu_threshold(gray)
    ink = gray <= thr
    if ink.mean() > 0.5:                       # 深底浅字，反转
        ink = ~ink
    return ink


def clear_long_runs(ink: np.ndarray, ratio: float = 0.55) -> np.ndarray:
    """抹掉横线本上的长横线，否则投影剖面全被它污染。"""
    h, w = ink.shape
    need = int(w * ratio)
    out = ink.copy()
    for y in np.where(ink.sum(1) > need)[0]:
        row = ink[y]
        d = np.diff(row.astype(np.int8))
        starts = list(np.where(d == 1)[0] + 1)
        ends = list(np.where(d == -1)[0] + 1)
        if row[0]:
            starts.insert(0, 0)
        if row[-1]:
            ends.append(w)
        for s, e in zip(starts, ends):
            if e - s >= need:
                out[y, s:e] = False
    return out


def estimate_skew(ink: np.ndarray, lo: float = -3.0, hi: float = 3.0, step: float = 0.5) -> float:
    """行投影越尖锐说明越正，取使投影平方和最大的角度。"""
    small = Image.fromarray((ink * 255).astype(np.uint8))
    small.thumbnail((700, 700), Image.LANCZOS)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(lo, hi + 1e-6, step):
        rot = small.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=0)
        p = (np.asarray(rot) > 127).sum(1).astype(np.float64)
        score = float((p ** 2).sum())
        if score > best_score:
            best_score, best_angle = score, float(angle)
    return best_angle


def rotate_gray(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return gray
    im = Image.fromarray(gray).rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
    return np.asarray(im)


# ---------- 行切分 ----------

def row_profile(ink: np.ndarray, smooth: int) -> np.ndarray:
    p = ink.sum(1).astype(np.float64)
    k = max(3, smooth | 1)
    if k <= len(p):
        p = np.convolve(p, np.ones(k) / k, mode="same")
    return p


def find_bands(p: np.ndarray, thr_ratio: float, min_gap: int) -> list:
    thr = max(3.0, p.max() * thr_ratio)
    mask = p > thr
    bands, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(mask)))
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)
    return merged


def split_tall(bands, p, median_h, depth=0):
    """一行过高说明两行粘连，在最深谷底再切。"""
    if median_h <= 0:
        return bands
    out = []
    for y0, y1 in bands:
        h = y1 - y0
        if h <= 1.6 * median_h or depth >= 3:
            out.append((y0, y1))
            continue
        lo, hi = y0 + int(h * 0.30), y0 + int(h * 0.70)
        seg = p[lo:hi]
        if len(seg) < 4:
            out.append((y0, y1))
            continue
        cut = lo + int(np.argmin(seg))
        out.extend(split_tall([(y0, cut), (cut, y1)], p, median_h, depth + 1))
    return out


def bands_to_boxes(ink: np.ndarray, bands) -> list:
    h, w = ink.shape
    boxes = []
    for y0, y1 in bands:
        cols = ink[y0:y1].sum(0)
        nz = np.where(cols > 0)[0]
        if len(nz) == 0:
            continue
        x0, x1 = int(nz[0]), int(nz[-1]) + 1
        pad_x = max(4, int(w * 0.02))
        pad_y = max(4, int((y1 - y0) * 0.12))
        boxes.append(
            (
                max(0, x0 - pad_x),
                max(0, y0 - pad_y),
                min(w, x1 + pad_x),
                min(h, y1 + pad_y),
            )
        )
    return boxes


# ---------- 整页模式 ----------
#
# 上面那套流程（Otsu + 全局阈值 + 纠偏）默认"整张图就是纸面"，这正是裁剪过的
# 公式块的情形。实拍作业照不是这样：画面里还有桌面、手影、纸沿，桌面比纸暗，
# 会被 Otsu 判成墨迹——实测 W01/W05/W08 的"墨迹"占比 0.21~0.42（正常应 <0.07），
# 行投影被背景糊住，整页并成一行或凭空多出几个行带。
#
# 整页模式换三个判据，每个都对着一条实测证据：
#   1. 小半径高通判笔画——阴影是缓变的，高通里几乎没有响应；
#   2. 腐蚀剔"厚块"——笔画细、桌面和纸边厚，腐蚀后还活着的是背景；
#   3. 行带要过验收：峰值离群（纸边硬边缘的响应是文字的 3~4 倍）、过矮的碎片、
#      以及游离在正文块之外的孤带，都不是字。
#
# 它仍然只负责"切"，不保证"切对"；切不干净时照样在 note 里说出来。


def wants_page_mode(size: tuple[int, int]) -> bool:
    """短边够大就按整页照处理。小图（公式块）走原来的路径，参数与历史数字都不动。"""
    return min(size) >= PAGE_MODE_MIN_SIDE


def highpass_ink(gray: np.ndarray, radius_ratio: float = 0.012,
                 k: float = 3.0, floor: float = 14.0) -> np.ndarray:
    """用"比周围一圈暗多少"判笔画，而不是用全局阈值判明暗。

    背景（桌面、阴影）在大半径模糊后基本不变，做差接近 0；笔画是几十灰度级的突变。
    阈值取 k 倍 MAD 而不是标准差：阴影边缘这种离群值不会把阈值拉飞。
    """
    r = max(7, int(min(gray.shape) * radius_ratio))
    bg = np.asarray(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(r)), dtype=np.float32)
    d = bg - gray.astype(np.float32)
    mad = float(np.median(np.abs(d - np.median(d)))) + 1e-6
    return d > max(floor, k * mad)


def drop_thick_blobs(ink: np.ndarray, thick_ratio: float = 0.02) -> np.ndarray:
    """剔掉"厚"的墨迹块：腐蚀一次还活着的地方是桌面、手影、纸边，不是笔画。

    在 1/4 尺度上做形态学，代价只有全分辨率的一个零头。
    """
    h, w = ink.shape
    down = 4
    small = Image.fromarray((ink * 255).astype(np.uint8)).resize(
        (max(1, w // down), max(1, h // down)), Image.NEAREST)
    rad = max(2, int(min(h, w) * thick_ratio) // down)
    core = small.filter(ImageFilter.MinFilter(2 * rad + 1))       # 腐蚀：只留厚块核心
    thick = core.filter(ImageFilter.MaxFilter(2 * rad + 1))       # 长回原大小
    mask = np.asarray(thick.resize((w, h), Image.NEAREST)) > 127
    return ink & ~mask


def _candidate_bands(p: np.ndarray, min_h: int) -> list[tuple[int, int]]:
    """先粗切：投影超过 80 分位数的连续段。宁可多切几条，交给后面验收。"""
    thr = max(3.0, float(np.percentile(p, 80)))
    mask = p > thr
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return [b for b in runs if b[1] - b[0] >= min_h]


def _band_stats(ink: np.ndarray, p: np.ndarray, bands) -> list[dict]:
    h, w = ink.shape
    rows = []
    for y0, y1 in bands:
        cols = ink[y0:y1].sum(0)
        nz = np.where(cols > 0)[0]
        if len(nz) == 0:
            continue
        rows.append({
            "y0": y0, "y1": y1, "h": y1 - y0,
            "x0": int(nz[0]), "x1": int(nz[-1]),
            "peak": float(p[y0:y1].max()),
            "touches_right_corner": bool(int(nz[0]) > 0.35 * w and int(nz[-1]) >= w - 2),
            # 画面上下缘 8% 以内算"贴边"。实拍里纸边、桌面交界都在这条带上，
            # 而正文行——哪怕写到最后一页底——基本不会压到这里。
            "near_edge": bool(y0 <= 0.08 * h or y1 >= 0.92 * h),
            # 横向铺满整幅：纸边、桌面分界线是贯通的直线，一行字不会从头写到尾。
            "spans_wide": bool(int(nz[-1]) - int(nz[0]) >= 0.88 * w),
        })
    return rows


def _is_frame_edge(r: dict) -> bool:
    """纸边 / 桌面交界：贴边**且**横向贯通。

    两个条件缺一不可。只贴边——页脚那行字也可能贴边；只贯通——一行写得满的
    算式也能占到 88% 宽。两者同时成立才不是字。
    """
    return r["near_edge"] and r["spans_wide"]


def page_bands(ink: np.ndarray, p: np.ndarray, min_h: int) -> list[tuple[int, int]]:
    """整页照的行带：粗切 → 验收 → 取正文块 → 合并同一行的上下标。"""
    rows = _band_stats(ink, p, _candidate_bands(p, min_h))
    if not rows:
        return []

    med_h = float(np.median([r["h"] for r in rows]))
    # 横线本的格线只有十几像素高，按"页高的 1.2%"兜一个绝对下限：
    # 手写单行在 1280 px 高的照片上通常 25~90 px，格线在 10~12 px，分得开。
    abs_floor = 0.012 * ink.shape[0]
    keep = [r for r in rows
            # 贴边且贯通的才是纸边。画面内部的强响应不能一律当边缘剔——
            # W16 第三行是学生整行涂黑，峰值 472（是文字的 3.6 倍），
            # 但它在正文中间、宽度只到 86%，那是内容，得留着判 unread。
            if not _is_frame_edge(r)
            # 0.45 会让"最后一行写得比别的小"被误剔：W12 第三行 22 px，
            # 中位行高 49 px，0.45×49=22.05，差 0.05 像素就丢了整整一行。
            and r["h"] >= max(0.35 * med_h, abs_floor)
            and not r["touches_right_corner"]]      # 画面右上角的遮挡物
    if not keep:
        return keep

    # 第二轮：纸沿的硬边缘还剩一部分没贴到画面边界（W01 顶部黑边的下沿，
    # 放在 118~262 px 处，横向 0..w 全宽，峰值 195~285，正文只有 78~104）。
    # 基准不能用中位数——一半以上的带是垃圾时中位数跟着垃圾走。取 25 分位
    # 当"文字该有多亮"的估计，离群 1.8 倍以上、且横向贯通的才剔。
    base = float(np.percentile([r["peak"] for r in keep], 25))
    keep = [r for r in keep if not (r["peak"] > 1.8 * base and r["spans_wide"])]
    if not keep:
        return keep

    # 正文块：相邻行带的间距都差不多，突然出现的大空档说明那是画面另一处的杂物
    gaps = [b["y0"] - a["y1"] for a, b in zip(keep, keep[1:])]
    if gaps:
        tol = 2.5 * float(np.median(gaps))
        clusters = [[keep[0]]]
        for g, r in zip(gaps, keep[1:]):
            if g > tol:
                clusters.append([r])
            else:
                clusters[-1].append(r)
        keep = max(clusters, key=lambda c: (len(c), sum(x["h"] for x in c)))

    # 同一行的上下标、断笔会切出两条，间距小于半个行高就并回去
    med_h = float(np.median([r["h"] for r in keep]))
    merged = [dict(keep[0])]
    for r in keep[1:]:
        if r["y0"] - merged[-1]["y1"] < 0.45 * med_h:
            merged[-1]["y1"] = r["y1"]
            merged[-1]["h"] = r["y1"] - merged[-1]["y0"]
        else:
            merged.append(dict(r))
    return [(m["y0"], m["y1"]) for m in merged]


def widen_boxes(boxes: list, ink, min_ink: int = 3, ratio: float = 0.32) -> list:
    """框上方确实有墨迹（上标）时，把裁框向上放宽一点，但不许压到上一行的中缝。

    为什么要放宽：行带是按"墨迹浓度"找出来的，上标 `²` 只有十几个像素、
    笔画又轻，常常够不上行带的阈值，于是整条带被判成"过矮的碎片"剔掉。
    剔掉的是**带**，可上标本身还在纸上——只是裁框从行带顶端开始，
    正好把它留在框外。W04 第一行 `x²-5x+6=0` 被读成 `x-5x+6=0` 就是这么来的：
    两台识别器都答对了它们看见的东西，框里确实没有那个 2。

    为什么要有条件：放宽会改变宽高比，而视觉模型对"这一块是什么"的判断
    跟着构图走——同样的字，框一高就可能从"一行中文"变成"一条公式"，
    甚至两台机器一起读错（W17 放宽后，行首的 `⇒` 被两台都读成数字 `2`，
    整页从"判不了"变成"判你错了"）。没有上标的行不该为别人的问题付代价，
    所以只在上面真有墨迹时才放宽。

    只往上放：上标在基线上方，基线以下的部分（下伸笔画、逗号）本来就在带里。
    上界用上一框与本框的中缝夹住，宁可少留也不能把上一行的笔画卷进来——
    卷进来会把上一行的尾巴读成这一行的开头，比丢个上标更难查。
    """
    out = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        up = int((y1 - y0) * ratio)
        top = max(0, y0 - up)
        if i > 0:
            top = max(top, (boxes[i - 1][3] + y0) // 2)
        above = ink[top:y0, max(0, x0):x1]
        if top < y0 and int(above.sum()) >= min_ink:
            out.append((x0, top, x1, y1))
        else:
            out.append((x0, y0, x1, y1))
    return out


def _split_page_mode(gray: np.ndarray, *, min_height: int) -> tuple[np.ndarray, list, float]:
    """整页模式不做纠偏。

    实测（W01~W17 八页实拍照）estimate_skew 在高通墨迹上经常顶到 ±3° 的上限，
    把图像一转，笔画被插值糊掉，反而切得更差（W17 从 4 行变 6 行）。
    方向问题应该在采集阶段解决（tools/normalize_capture.py 的转正表），
    切行这里只按行投影找带——几度以内的倾斜对水平投影几乎没有影响。
    """
    ink = drop_thick_blobs(highpass_ink(gray))
    ink = clear_long_runs(ink)          # 横线本的直线、表格线：整行抹掉，否则每根线都是一"行"
    p = row_profile(ink, smooth=max(5, int(ink.shape[0] * 0.010)))
    bands = page_bands(ink, p, max(6, int(ink.shape[0] * 0.006)))
    boxes = [b for b in bands_to_boxes(ink, bands) if b[3] - b[1] >= min_height]
    boxes = widen_boxes(boxes, ink)
    return gray, boxes, 0.0


# ---------- 对外接口 ----------

@dataclass
class Crop:
    """一行小图。index 从 1 起，和界面上显示的"第几行"是同一个数。"""

    index: int
    image: Image.Image
    box: tuple[int, int, int, int]
    path: str = ""

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


@dataclass
class PageSplit:
    page: str
    crops: list[Crop] = field(default_factory=list)
    skew: float = 0.0
    size: tuple[int, int] = (0, 0)
    note: str = ""
    sheet: Image.Image | None = None      # 纠偏后的整页图，用来画框核对

    def __len__(self) -> int:
        return len(self.crops)


def split_page(
    path: str | Path,
    *,
    deskew: bool = True,
    threshold_ratio: float = 0.03,
    min_gap_ratio: float = 0.012,
    min_height: int = MIN_LINE_H,
    page_mode: bool | None = None,
) -> PageSplit:
    """把一张整页照切成单行。不写盘——落盘是调用方的事。

    page_mode=None 时按短边自动判断（>=400 px 视为实拍整页照，走整页模式）；
    False 走原来的全局 Otsu 路径，参数与历史评测数字保持不变；
    True 强制整页模式，用于验证或已知是实拍照的场景。
    """
    path = Path(path)
    gray = load_gray(path)
    if page_mode is None:
        page_mode = wants_page_mode(gray.shape)

    if page_mode:
        gray, boxes, angle = _split_page_mode(gray, min_height=min_height)
    else:
        ink = clear_long_runs(binarize(gray))

        angle = 0.0 if not deskew else estimate_skew(ink)
        # 灰度图跟着一起转，切图从灰度图取，保留抗锯齿和笔画粗细
        gray = rotate_gray(gray, angle)
        ink = binarize(gray)

        p = row_profile(ink, smooth=max(3, int(ink.shape[0] * 0.004)))
        min_gap = max(4, int(ink.shape[0] * min_gap_ratio))
        bands = find_bands(p, threshold_ratio, min_gap)
        bands = [b for b in bands if b[1] - b[0] >= max(8, int(ink.shape[0] * 0.008))]
        if bands:
            median_h = float(np.median([b[1] - b[0] for b in bands]))
            bands = split_tall(bands, p, median_h)
        boxes = [b for b in bands_to_boxes(ink, bands) if b[3] - b[1] >= min_height]

    sheet = Image.fromarray(gray).convert("RGB")
    crops = [
        Crop(index=i, image=sheet.crop((x0, y0, x1, y1)), box=(x0, y0, x1, y1))
        for i, (x0, y0, x1, y1) in enumerate(boxes, start=1)
    ]
    return PageSplit(
        page=path.name,
        crops=crops,
        skew=angle,
        size=sheet.size,
        note=_split_note(crops, path),
        sheet=sheet,
    )


def _split_note(crops: list[Crop], path: Path) -> str:
    """切完先自己看一眼，把可疑之处说出来。

    切错一行不会报错，只会让后面识别出莫名其妙的式子——那种失败最难查，
    所以宁可在这里多说一句。
    """
    if not crops:
        return "这一页没有切出任何一行。可能整页空白，或者光照太暗、墨迹没被判成笔画。"
    hs = [c.size[1] for c in crops]
    med = float(np.median(hs))
    tall = [c.index for c in crops if c.size[1] > med * 2.2]
    short = [c.index for c in crops if c.size[1] < med * 0.45]
    bits = []
    if tall:
        bits.append(
            f"第 {'、'.join(map(str, tall))} 行明显比别人高，可能是两行粘成了一行"
        )
    if short:
        bits.append(
            f"第 {'、'.join(map(str, short))} 行特别矮，可能是墨点或被切断的笔画"
        )
    if len(crops) == 1:
        bits.append("整页只切出一行：如果这是整页作业，多半是行间距太密被判成了一行")
    return "；".join(bits)


def save_crops(split: PageSplit, out_dir: str | Path, stem: str | None = None) -> list[Path]:
    """落盘，返回路径列表，顺序与 split.crops 一致。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = stem or Path(split.page).stem
    paths = []
    for c in split.crops:
        p = out_dir / f"{base}_{c.index:02d}.jpg"
        c.image.save(p, quality=95)
        c.path = str(p)
        paths.append(p)
    return paths


def debug_sheet(split: PageSplit) -> Image.Image:
    """把切出来的框画回整页图，用来肉眼核对切得对不对。"""
    if split.sheet is None:
        return Image.new("RGB", (10, 10), "white")
    sheet = split.sheet.copy()
    d = ImageDraw.Draw(sheet)
    for c in split.crops:
        d.rectangle(list(c.box), outline=(220, 40, 40), width=3)
    return sheet
