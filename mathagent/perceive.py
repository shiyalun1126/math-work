"""感知层：多源识别，每个来源返回若干候选。

设计要点：
- 结果按图片哈希落盘，改 prompt 不重复花钱
- 单个来源失败不影响整体，失败信息记在 note 里而不是抛异常
- 两个来源架构不同，错误模式才不相关，投票才有意义
- 送模型前先裁到墨迹范围：手机拍的整页照四周几十毫米白纸，
  不裁会把这些空白一起编码，等于白白拉低有效分辨率
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
from pathlib import Path

import requests
from PIL import Image, ImageFilter, ImageOps

from . import config
from .cache import Cache, file_digest
from .types import Candidate

MAX_SIDE = 1280  # 长边超过 1280 就缩，省 token 不影响单行公式
PAD_RATIO = 0.04  # 裁完留一点边，太贴边模型容易漏掉首尾符号

# 预处理结果的进程内缓存：同一张图会被多个识别通道同时要。
_PREP_CACHE: dict[tuple, tuple[bytes, str]] = {}
_PREP_LOCK = threading.Lock()

# 限流退避：一次 429 不代表"这条通道废了"，只代表"这一秒太多"。
# 免费档实测 3~5 秒后多半就通，直接放弃会让交叉验证退化成单源。
THROTTLE_RETRY = 2      # 整轮重试次数（每轮把候选模型都试一遍）
THROTTLE_SLEEP = 4.0    # 第 n 轮前等 THROTTLE_SLEEP × n 秒

# 单次 HTTP 请求的默认上限。它**不是**一次识别的时间上限——
# 3 轮重试 × 3 个候选模型才能把这个数字乘到十几分钟。
HTTP_TIMEOUT = 90.0

# 整页时间预算：一次识别最多占用多久。
#
# 为什么要按行数给而不是给固定值：耗时基本跟行数成正比（每行两个来源、
# 两个来源各自重试），固定预算的结果是"三行的页很宽松、十行的页必超时"。
# 12 + 5n 在 3 行时给 27s、6 行 42s、24 行（MAX_LINES）132s，
# 大致对上实测的 20s/页（6 行）。
#
# 这个预算**不是性能目标**，是止损线：正常情况远用不到，只有服务变慢或
# 限流时才会撞上——那时候唯一要紧的是别让页面一直转下去。
BUDGET_BASE = 12.0
BUDGET_PER_LINE = 5.0
BUDGET_ENV = "MATHAGENT_BUDGET"   # 设成秒数覆盖；设 0 表示不限（调试用）


class RecognitionError(RuntimeError):
    pass


class BudgetExhausted(RecognitionError):
    """整页预算用完，还没发出的请求被放弃。

    单独一个类型是为了让上层能区分"识不出来"和"没时间去识"——
    前者要说"这行没认准"，后者要说"这一页超时了，已认出的部分照常可用"。
    混成一句会让学生以为是自己写得不好。
    """


def call_recognize(rec: "Recognizer", image, k: int = 1,
                   deadline: "Deadline | None" = None) -> list[Candidate]:
    """调一个识别器，兼容还没接受 deadline 的旧实现。

    这一轮给 `recognize` 加了 deadline 关键字。自己写的识别器（`tools/` 下就有，
    测试里的假识别器也是）不会跟着改，按新签名直接调会 TypeError，
    而调用方那一层的 try/except 会把它记成"这一路认不出"——
    整条通道静默失效，比抛异常还难查。

    所以只在 TypeError 明确提到 deadline 时回退，其他 TypeError 照抛：
    `recognize` 内部真出的 bug 不该被这里吃掉。
    """
    try:
        return rec.recognize(image, k=k, deadline=deadline)
    except TypeError as exc:
        if "deadline" not in str(exc):
            raise
        return rec.recognize(image, k=k)


class Deadline:
    """一次调用共享的时间预算。

    为什么需要它：`HTTP_TIMEOUT=90` 看着不吓人，但 `_one` 会走
    `(THROTTLE_RETRY+1) × 候选模型数` 次，`slot2` 有 3 个候选模型，
    单行最坏 9 × 90 + (4+8) = 822 秒；整页 24 行、并发 2，就是两个多小时。
    这些数字在一切正常时看不到，只有服务变慢或限流时才暴露——
    而那时用户已经在盯着转圈的页面，还以为是自己的图有问题。

    预算用完之后**不抛给调用方什么都不剩**：已经拿到的候选照常返回并参与投票，
    只是不再发新请求。慢的那一路缺席只影响旁证强度，不该连已有的结果一起丢掉。

    用 monotonic 而不是 time.time：系统时间被 NTP 校正时 wall clock 会跳，
    预算可能突然变成负数或凭空多出几小时。
    """

    __slots__ = ("t_end", "budget")

    def __init__(self, budget: float | None = None):
        self.budget = budget if (budget and budget > 0) else None
        self.t_end = (time.monotonic() + self.budget) if self.budget else None

    def left(self) -> float:
        if self.t_end is None:
            return float("inf")
        return self.t_end - time.monotonic()

    def expired(self, reserve: float = 0.0) -> bool:
        """reserve：还想留多少秒做事。剩余不到这个数就算"别开始了"。"""
        return self.left() <= reserve

    def cap(self, timeout: float, floor: float = 0.5) -> float:
        """把单个请求的超时压进剩余预算。

        没有这一步，预算只是个摆设：预算剩 2 秒时照样发一个 90 秒超时的请求，
        用户仍然要等 90 秒。floor 是给小请求留的最小值，避免 timeout≈0
        被 requests 当成"立即超时"而把一个本该成功的小请求判死。
        """
        if self.t_end is None:
            return timeout
        return max(floor, min(timeout, self.left()))

    def describe(self) -> str:
        if self.budget is None:
            return "未设预算"
        return f"预算 {self.budget:.0f}s，剩余 {max(0.0, self.left()):.1f}s"


def budget_for(lines: int = 1) -> float | None:
    """按行数算这次调用的预算。返回 None 表示不限（只给调试用）。

    环境变量 `MATHAGENT_BUDGET` 优先：调 prompt 时想让它跑到底看真实耗时，
    设 0 就行；线上要更紧或更松也只改环境，不用动代码。
    """
    raw = os.environ.get(BUDGET_ENV, "").strip()
    if raw:
        try:
            val = float(raw)
        except ValueError:
            val = None
        if val is not None:
            return val if val > 0 else None
    return BUDGET_BASE + BUDGET_PER_LINE * max(1, lines)

# 模型的自言自语里才有的词。手写数学一行式子不会出现这些。
_META_PROSE = re.compile(r"图片|图中|用户|规则|假设|要求|说明|以上|注意|最终结果|重新")

# 缓存键的版本指纹。
#
# 只按「图片哈希 + 模型名 + 温度」做键是不够的：改了 QWEN_PROMPT 的措辞，
# 同一张图仍然命中旧结果，看起来像"改了没效果"，实际是被缓存挡在后面。
# 这比不缓存更糟——不缓存时你会立刻看到变化，缓存会让你以为改对了。
# 所以提示词要算进键里（内容哈希，改一个字就自动失效）。
#
# 预处理（裁墨迹 / 缩放 / JPEG 质量）决定了送进模型的像素，
# 但它没法像提示词那样自动算哈希，改动时手工把 _PREP_VERSION 加一。
_PREP_VERSION = "prep1"


def _ink_bbox(im: Image.Image, thr: int = 165) -> tuple[int, int, int, int] | None:
    """返回墨迹包围盒。找不到、或者几乎占满整幅（说明不是白底纸）时返回 None。"""
    g = im.convert("L")
    w, h = g.size
    if w < 8 or h < 8:
        return None
    # 缩小再算，长图上省时间，边界误差由后面的 padding 兜住
    scale = min(1.0, 900 / max(w, h))
    small = g.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    mask = small.point(lambda v: 255 if v < thr else 0)
    box = mask.getbbox()
    if box is None:
        return None
    x0, y0, x1, y1 = (int(v / scale) for v in box)
    if (x1 - x0) * (y1 - y0) > 0.97 * w * h:
        return None  # 整幅都是"墨"，多半是深色背景，别乱裁
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def prepare_image(
    path: str | Path, crop: bool = True, max_side: int = MAX_SIDE, min_side: int = 640
) -> tuple[bytes, str]:
    """EXIF 摆正 → 裁到墨迹 → 缩到长边 max_side → JPEG。原图不动。

    同一张图会被多个识别通道各要一次。以前每一路都重新解码、裁剪、缩放一遍，
    一张 4000px 的手机照要白白算两次——而两条通道本来就是并发的，
    让它俩各算一次等于并发没并发。这里按「路径 + 修改时间 + 参数」记住结果，
    第二次直接拿 bytes。
    """
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        stamp = 0.0
    key = (str(Path(path).resolve()), stamp, crop, max_side, min_side)
    hit = _PREP_CACHE.get(key)
    if hit is not None:
        return hit
    with _PREP_LOCK:
        # 双检：等着锁的那几路通常能在别人算完之后直接命中
        hit = _PREP_CACHE.get(key)
        if hit is not None:
            return hit
        val = _prepare_image_uncached(path, crop=crop, max_side=max_side, min_side=min_side)
        if len(_PREP_CACHE) > 24:  # 一次评测动辄几十张图，别无限涨
            _PREP_CACHE.clear()
        _PREP_CACHE[key] = val
        return val


def _prepare_image_uncached(
    path: str | Path, crop: bool = True, max_side: int = MAX_SIDE, min_side: int = 640
) -> tuple[bytes, str]:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)  # 手机照片常靠 EXIF 标记方向，不摆正会横着送进去
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")

    if crop:
        box = _ink_bbox(im)
        if box:
            x0, y0, x1, y1 = box
            pad = max(8, int(PAD_RATIO * max(x1 - x0, y1 - y0)))
            im = im.crop(
                (
                    max(0, x0 - pad),
                    max(0, y0 - pad),
                    min(im.width, x1 + pad),
                    min(im.height, y1 + pad),
                )
            )

    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    elif max(w, h) < min_side:
        # 小图裁完往往只有 300px 宽，放大一点让笔画更清楚。
        # 公式题 640 就够；几何题是「题干+图形」一整页，中文小字在 640 下
        # 基本糊成一团，模型会把「AE 与 CD 相交于 F」整句漏掉——所以由调用方
        # 传更大的 min_side。
        up = min(4.0, min_side / max(1, max(w, h)))
        im = im.resize((int(w * up), int(h * up)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), "image/jpeg"


class Recognizer:
    name = "base"

    def available(self) -> bool:
        return False

    def recognize(self, image: str | Path, k: int = 1,
                  deadline: Deadline | None = None) -> list[Candidate]:
        raise NotImplementedError


def page_note(path: str | Path) -> str:
    """整页照提示。判据是"竖幅 + 多条行带"，只提醒不拦截。

    单看行带数会误判——分式、方程组、积分都会拆成多条行带；
    单看比例也会误判——竖着拍的单个式子很长。两个条件同时成立才提示。
    """
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        if w <= 0 or h / w < 0.95:
            return ""
        n = count_lines(path)
        if n < 3:
            return ""
        return (
            f"这张图像是竖幅整页（估算 {n} 行）。识别器会把整幅当成一个式子，"
            f"结果多半不对。建议先切行：`python tools/split_page.py 图片 --out data/crops`，"
            f"再把切出的单行图传进来。"
        )
    except Exception:  # noqa: BLE001
        return ""


def _ink_mask(im: Image.Image) -> "object":
    """局部对比度求墨迹：比"背景模糊图"暗足够多的像素才算笔画。

    这样选是因为整幅阈值必然失效——纸外桌面比纸暗（222 vs 250），
    在侧光下纸的暗角又能暗到和桌面差不多。前两个版本分别试过固定阈值
    和 Otsu，都会把桌面和横线本的淡线算成墨迹，于是每行都横贯全幅，
    一行公式被数成三行。
    """
    import numpy as np

    g = im.convert("L")
    arr = np.asarray(g).astype(np.int16)
    radius = max(3, int(min(g.size) * 0.06))
    bg = np.asarray(g.filter(ImageFilter.GaussianBlur(radius))).astype(np.int16)
    return arr < bg - 28


def count_lines(path: str | Path) -> int:
    """粗估图里有几行文字，用来提醒"整页照别直接喂给识别器"。

    只做水平投影数行带，不做纠偏也不切图——目的是给提示，不是精确分割。
    """
    try:
        import numpy as np

        im = ImageOps.exif_transpose(Image.open(path)).convert("L")
        w, h = im.size
        scale = min(1.0, 700 / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        ink = _ink_mask(im)
        prof = ink.sum(1).astype(float)
        if prof.max() <= 0:
            return 0
        peak = prof.max()
        mask = prof > max(3.0, peak * 0.08)
        bands = 0
        run = 0
        min_h = max(5, int(im.height * 0.006))
        for v in mask:
            if v:
                run += 1
            else:
                if run >= min_h:
                    bands += 1
                run = 0
        if run >= min_h:
            bands += 1
        return bands
    except Exception:  # noqa: BLE001
        return 0


class SimpleTexRecognizer(Recognizer):
    """SimpleTex 专用公式识别。确定性输出，k 无效，返回 1 条。"""

    name = "simpletex"

    def __init__(self, standard: bool | None = None, cache: Cache | None = None):
        if standard is None:
            standard = config.simpletex_quality() == "standard"
        self.url = config.simpletex_url()
        self.cache = cache or Cache()
        self.standard = standard
        self.name = "simpletex" if standard else "simpletex-turbo"

    def available(self) -> bool:
        return config.has_simpletex()

    def recognize(self, image: str | Path, k: int = 1,
                  deadline: Deadline | None = None) -> list[Candidate]:
        if not self.available():
            raise RecognitionError("未配置 SIMPLETEX_TOKEN（见 .env 通道 1）")
        digest = file_digest(image)
        hit = self.cache.get(digest, self.url, _PREP_VERSION)
        if hit:
            return [Candidate(**hit)]

        data, mime = prepare_image(image)
        dl = deadline or Deadline(None)
        if dl.expired(reserve=1.0):
            raise BudgetExhausted(f"本页时间用完了（{dl.describe()}）")
        t0 = time.time()
        try:
            r = requests.post(
                self.url,
                files={"file": (Path(image).name, data, mime)},
                headers={"token": config.simpletex_token()},
                timeout=dl.cap(40.0),
            )
        except requests.RequestException as exc:
            raise RecognitionError(f"网络不通：{exc}") from exc
        cost = time.time() - t0
        if r.status_code == 401:
            raise RecognitionError("令牌无效（401），检查 .env 里的 SIMPLETEX_TOKEN")
        if r.status_code == 402:
            raise RecognitionError(
                "额度不足（402）：免费次数用完了。可在 simpletex.cn 换成轻量模型"
                "（SIMPLETEX_MODEL=turbo，赠 1000 次）或购买资源包"
            )
        if r.status_code == 413:
            raise RecognitionError("图片过大或为空（413），换一张试试")
        if r.status_code != 200:
            raise RecognitionError(f"HTTP {r.status_code}: {r.text[:120]}")
        try:
            body = r.json()
        except Exception as exc:  # noqa: BLE001
            raise RecognitionError(f"返回不是 JSON：{r.text[:120]}") from exc
        if not body.get("status"):
            raise RecognitionError(f"接口返回失败：{body.get('res')}")
        res = body.get("res") or {}
        latex = (res.get("latex") or "").strip()
        if not latex:
            raise RecognitionError("返回为空（可能是空白图）")
        if latex.upper().startswith("[EMPTY"):
            raise RecognitionError("模型判定图片为空，检查是否拍到了公式")
        cand = Candidate(latex=latex, source=self.name, conf=res.get("conf"), elapsed=cost)
        self.cache.put(cand.to_dict(), digest, self.url, _PREP_VERSION)
        return [cand]


QWEN_PROMPT = (
    "把图片中的数学式子转写成 LaTeX 代码。要求："
    "1) 只输出 LaTeX 本体，不要解释，不要 markdown 代码块，不要 $ 或 $$；"
    "2) 忠实于图中写法，看不清也不要臆造，保持原样；"
    "3) 分式用 \\frac{}{}，上下标多于一个字符时加花括号；"
    "4) 必须使用 LaTeX 命令，禁止直接照抄 Unicode 数学符号："
    "积分写 \\int，根号写 \\sqrt{}，π 写 \\pi，乘号写 \\times，"
    "不等号写 \\leq \\geq \\neq，无穷写 \\infty。"
    "输出里不允许出现 ∫ √ π × ÷ ≤ ≥ ≠ ∞ 这些字符；"
    "5) 如果图里不是数学式子，只输出 NOMATH。"
)

# 提示词指纹：改动上面的 QWEN_PROMPT 一个字，这个值就变，旧缓存自动不命中。
_PROMPT_FP = hashlib.sha256(QWEN_PROMPT.encode("utf-8")).hexdigest()[:8]


def list_vl_models() -> list[str]:
    """列出当前账号实际可用的多模态模型。

    厂商会悄悄下架或改名模型，硬编码的默认名某天就会 400。
    这里一次 GET 就能把候选拉出来，比对着报错猜省事。
    """
    if not config.has_vlm():
        return []
    try:
        r = requests.get(
            f"{config.vlm_base().rstrip('/')}/models",
            headers={"Authorization": f"Bearer {config.vlm_key()}"},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        ids = [m.get("id", "") for m in r.json().get("data", []) if isinstance(m, dict)]
        return sorted(i for i in ids if i and "vl" in i.lower())
    except Exception:  # noqa: BLE001
        return []


class VLMRecognizer(Recognizer):
    """任意 OpenAI 兼容多模态端点。

    主通道默认通义千问 Qwen3-VL，第二通道默认智谱 GLM（免费）。
    统一走 /chat/completions，换成硅基流动、OpenAI、本地 vLLM
    也只需要改 .env 里的 BASE_URL 和 MODEL。

    为什么要两路：同一家厂商的两个模型，训练数据和失效模式高度重合，
    凑在一起只能算自一致性；跨厂商才算真正的交叉验证。
    """

    name = "vlm"

    def __init__(
        self,
        model: str | None = None,
        cache: Cache | None = None,
        *,
        base: str | None = None,
        api_key: str | None = None,
        slot: int = 1,
    ):
        self.slot = slot
        if slot == 1:
            self.model = model or config.vlm_model()
            self.base = (base or config.vlm_base()).rstrip("/")
            self._key = api_key or config.vlm_key()
            self.env_hint = "VLM_API_KEY / DASHSCOPE_API_KEY"
        else:
            self.model = model or config.vlm2_model()
            self.base = (base or config.vlm2_base()).rstrip("/")
            self._key = api_key or config.vlm2_key()
            self.env_hint = "VLM2_API_KEY"
        self.cache = cache or Cache()
        # 候选里显示来源族名（qwen3-vl / glm-4.6v-flash），具体版本记在 model 里。
        # 否则 VLM_MODEL 换个后缀，投票逻辑就会把它当成"第二个来源"。
        if self.model.startswith("qwen3-vl"):
            self.name = self.model.split("-")[0] + "-vl"
        else:
            self.name = self.model
        self.used_model = self.model

    def available(self) -> bool:
        return bool(self._key)

    def _fallbacks(self) -> list[str]:
        """免费额度档在高峰经常 429，按顺序换备用模型重试。

        只有第二路才配这个：主通道是付费模型，撞限流基本是账号问题，
        换模型反而掩盖真实故障。
        """
        if self.slot != 2:
            return []
        # 优先选非推理型：thinking 模型答这种短结构化任务又慢又爱截断，
        # 排在后面兜底。glm-4v-flash 虽弱，但识别结果经 _fix_functions
        # 补全函数名后可用。
        raw = config.get("VLM2_FALLBACK", "glm-4v-flash,glm-4.1v-thinking-flash")
        out = [m.strip() for m in raw.split(",") if m.strip()]
        return [m for m in out if m != self.model]

    def _post(self, model: str, data_b64: str, temperature: float,
              timeout: float = HTTP_TIMEOUT):
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{data_b64}"},
                        },
                        {"type": "text", "text": QWEN_PROMPT},
                    ],
                }
            ],
            "temperature": temperature,
            "max_tokens": 800,
        }
        try:
            return requests.post(
                f"{self.base}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise RecognitionError(f"网络不通：{exc}") from exc

    def _one(self, data_b64: str, temperature: float,
             deadline: Deadline | None = None) -> str:
        self.used_model = self.model
        models = [self.model] + self._fallbacks()
        throttled: list[str] = []
        dl = deadline or Deadline(None)

        # 限流要退避重试，不能立刻认输。免费档的 429 是"这一秒太多"，
        # 几秒后同一模型往往就通了；原来一遇 429 就切下一个模型或直接报错，
        # 结果是整页只有一路识别活着，投票退化成"单源无旁证"→ 全行待核对。
        # 那看起来像"认不出来"，其实是没去问第二遍。
        for attempt in range(THROTTLE_RETRY + 1):
            for model in models:
                # 预算见底就不再发新请求。1 秒的余量是留给建连本身的：
                # 剩 0.2 秒还发请求，只会换来一个必然的 ConnectTimeout。
                if dl.expired(reserve=1.0):
                    raise BudgetExhausted(f"本页时间用完了（{dl.describe()}）")
                r = self._post(model, data_b64, temperature, dl.cap(HTTP_TIMEOUT))
                if r.status_code == 200:
                    self.used_model = model
                    data = r.json()
                    try:
                        content = data["choices"][0]["message"]["content"]
                    except Exception as exc:  # noqa: BLE001
                        raise RecognitionError(f"返回结构异常：{str(data)[:180]}") from exc
                    if isinstance(content, list):
                        content = "".join(
                            p.get("text", "") for p in content if isinstance(p, dict)
                        )
                    return _clean_latex(content)

                if r.status_code == 429:
                    if model not in throttled:
                        throttled.append(model)
                    continue

                if r.status_code == 401:
                    raise RecognitionError(f"key 无效（401），检查 .env 里的 {self.env_hint}")
                body = r.text[:220]
                if "model" in body.lower() and "not" in body.lower():
                    hint = list_vl_models() if self.slot == 1 else []
                    tail = f"｜本账号可用：{', '.join(hint)}" if hint else ""
                    raise RecognitionError(
                        f"模型名不存在：{model}{tail}（改 .env 里对应的 MODEL）"
                    )
                raise RecognitionError(f"HTTP {r.status_code}: {body}")

            if attempt < THROTTLE_RETRY:
                nap = THROTTLE_SLEEP * (attempt + 1)
                # 睡不起就别睡：退避的前提是"过几秒重试还有机会成"，
                # 剩余预算不够再发一次请求时，只会让人多等几秒换来同样的失败。
                if dl.expired(reserve=nap + 1.0):
                    raise BudgetExhausted(
                        f"本页时间用完了（{dl.describe()}），不再退避重试")
                time.sleep(nap)

        if dl.expired():
            raise BudgetExhausted(f"本页时间用完了（{dl.describe()}）")
        raise RecognitionError(
            f"限流（429）：{'、'.join(throttled) or '全部候选模型'} 重试 "
            f"{THROTTLE_RETRY} 次仍不可用，免费档高峰常见，换时段再试"
        )

    def recognize(self, image: str | Path, k: int = 1,
                  deadline: Deadline | None = None) -> list[Candidate]:
        if not self.available():
            raise RecognitionError(f"未配置 {self.env_hint}（见 .env）")
        digest = file_digest(image)
        data, _ = prepare_image(image)
        b64 = base64.b64encode(data).decode()

        out: list[Candidate] = []
        for i in range(max(1, k)):
            # 第二次采样是"同一模型再问一遍"，用来做自一致性检查；
            # 预算不够时跳过它不影响这一行有没有结果，只是少了自洽证据。
            if i and deadline is not None and deadline.expired(reserve=1.5):
                out.append(Candidate(
                    latex="", source=self.name,
                    note=f"采样跳过：{deadline.describe()}"))
                break
            temp = round(0.2 + 0.3 * i, 2)
            key = (digest, self.model, str(temp), _PREP_VERSION, _PROMPT_FP)
            hit = self.cache.get(*key)
            if hit:
                out.append(Candidate(**hit))
                continue
            t0 = time.time()
            try:
                latex = self._one(b64, temp, deadline)
            except Exception as exc:  # noqa: BLE001
                out.append(
                    Candidate(latex="", source=self.name, note=f"{type(exc).__name__}: {exc}")
                )
                continue
            if latex.strip().upper() == "NOMATH":
                out.append(
                    Candidate(latex="", source=self.name, note="模型判断图中不是数学式子")
                )
                continue
            if not latex.strip():
                out.append(
                    Candidate(
                        latex="",
                        source=self.name,
                        note=f"{self.used_model} 输出被截断或为空，本次采样作废",
                    )
                )
                continue
            note = ""
            if self.used_model != self.model:
                note = f"主模型限流，改用 {self.used_model}"
            cand = Candidate(
                latex=latex,
                source=self.name,
                elapsed=time.time() - t0,
                note=note,
                weak=bool(note),
            )
            self.cache.put(cand.to_dict(), *key)
            out.append(cand)
        # 只丢"什么都没有"的候选。带 note 的必须留着——那是"这次为什么没读到"
        # 的唯一记录，上层靠它区分"认不出"和"没时间认"。以前一律按 latex 过滤，
        # 于是"k=2 的第二次超时跳过"这种说明在返回时被悄悄吞掉，
        # 界面上只剩"这一行没认准"，看不出是撞了时间上限。
        # 投票侧本来就忽略空 latex 的候选，保留它们不影响结论。
        return [c for c in out if c.latex or c.note] or out


# 旧名字保留，避免外部脚本 import 报错
QwenVLRecognizer = VLMRecognizer


def _clean_latex(text: str) -> str:
    s = (text or "").strip()
    # 推理型模型（GLM-4.1V-Thinking 等）会先输出一段思考过程。
    # 不清掉的话整段会被当成 LaTeX 送进解析器，必然失败。
    if "<answer>" in s:
        s = s.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    elif "<think" in s:
        # 只有开头没有 <answer>：思考过程写到一半就被 max_tokens 截断了，
        # 剩下全是半截话。当 LaTeX 送进解析器只会污染投票，直接判废。
        return ""
    s = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", s, flags=re.S).strip()
    # GLM 系列有时把最终答案裹在 <|begin_of_box|>...<|end_of_box|> 里，
    # 不剥掉的话整串送进解析器必失败，白白浪费一路独立票。
    s = re.sub(r"<\|(?:begin|end)_of_box\|>", "", s).strip()
    if s.startswith("\\boxed{") and s.endswith("}"):
        s = s[len("\\boxed{") : -1].strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        s = s.rsplit("```", 1)[0]
    s = s.strip().strip("$").strip()
    for left, right in (("\\[", "\\]"), ("\\(", "\\)")):
        if s.startswith(left) and s.endswith(right):
            s = s[len(left) : -len(right)]
    s = s.strip()

    # 兜底模型（GLM-4.1V-Thinking 这类）即使给了 <answer>，里面也常跟着
    # 一大段"注：此处基于对图片的合理判断…"的自言自语。这种脏候选进了投票，
    # 会把一行本来读对的内容判成"两派分歧"——一条脏数据毁掉整行的判定。
    if "\n" in s:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        clean = [ln for ln in lines if not _META_PROSE.search(ln) and len(ln) <= 120]
        # 优先取"像式子"的那一行（含等号或 LaTeX 命令）。学生的一行经常是
        # "两边同时乘2得" 换行 "2x²-10x+12=0"，取第一行就把式子丢了。
        mathy = [ln for ln in clean if "=" in ln or "\\" in ln]
        picked = mathy[0] if mathy else (clean[0] if clean else "")
        if not picked:
            return ""
        s = picked
    return s.strip().strip("$").strip()


class FixtureRecognizer(Recognizer):
    """演示用：按文件名从 fixtures/demo.json 读预置候选，不联网。

    存在的意义是让别人 clone 下来就能看到交互效果，
    而不是第一步先卡在注册账号上。
    """

    name = "fixture"

    def __init__(self, fixtures: str | Path | None = None):
        p = Path(fixtures) if fixtures else config.ROOT / "fixtures" / "demo.json"
        self.data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def available(self) -> bool:
        return bool(self.data)

    def recognize(self, image: str | Path, k: int = 1,
                  deadline: Deadline | None = None) -> list[Candidate]:
        stem = Path(image).stem
        entry = self.data.get(stem)
        if not entry:
            raise RecognitionError(f"演示夹具里没有 {stem}（这是你自己拍的图，需要真实密钥）")
        return [Candidate(**c) for c in entry["candidates"]]


def build_ensemble(k: int = 2, allow_fixture: bool = False) -> list[Recognizer]:
    """按配置组装可用的识别器。真源优先，没有密钥才退到夹具。"""
    recs: list[Recognizer] = []
    if config.has_simpletex():
        recs.append(SimpleTexRecognizer())
    if config.has_vlm():
        recs.append(VLMRecognizer(slot=1))
    if config.has_vlm2():
        recs.append(VLMRecognizer(slot=2))
    if not recs and allow_fixture:
        recs.append(FixtureRecognizer())
    return recs


# ---------------------------------------------------------------- 连通性自检

def _probe_image() -> bytes:
    """造一张写着 x^2-3x+2=0 的小图，用来真跑一次识别接口。"""
    from PIL import ImageDraw

    im = Image.new("RGB", (420, 110), (252, 251, 246))
    d = ImageDraw.Draw(im)
    font = None
    for p in (r"C:\Windows\Fonts\simkai.ttf", r"C:\Windows\Fonts\msyh.ttc"):
        try:
            from PIL import ImageFont

            font = ImageFont.truetype(p, 52)
            break
        except Exception:
            continue
    d.text((18, 22), "x²-3x+2=0", fill=(25, 25, 25), font=font)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def probe_simpletex() -> tuple[bool, str]:
    if not config.has_simpletex():
        return False, "未填写 SIMPLETEX_TOKEN"
    t0 = time.time()
    try:
        r = requests.post(
            config.simpletex_url(),
            files={"file": ("probe.png", _probe_image(), "image/png")},
            headers={"token": config.simpletex_token()},
            timeout=40,
        )
    except requests.RequestException as exc:
        return False, f"网络不通：{exc}"
    dt = time.time() - t0
    if r.status_code == 401:
        return False, "令牌无效（401）"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}：{r.text[:100]}"
    try:
        body = r.json()
    except Exception:
        return False, f"返回不是 JSON：{r.text[:100]}"
    if not body.get("status"):
        return False, f"接口报错：{body.get('res')}"
    latex = ((body.get("res") or {}).get("latex") or "").strip()
    return True, f"可用，{dt:.1f}s，自检图识别为 {latex or '（空）'}"


def probe_vlm(slot: int = 1) -> tuple[bool, str]:
    if slot == 1:
        if not config.has_vlm():
            return False, "未填写 DASHSCOPE_API_KEY / VLM_API_KEY"
        base, model, key = config.vlm_base(), config.vlm_model(), config.vlm_key()
    else:
        if not config.has_vlm2():
            return False, "未填写 VLM2_API_KEY（第二路可不配，配了才有跨源投票）"
        base, model, key = config.vlm2_base(), config.vlm2_model(), config.vlm2_key()

    # 第二路要跟识别时用同一套降级链，否则会出现
    # "自检说失败、实际识别能跑" 的错位。
    models = [model]
    if slot == 2:
        raw = config.get("VLM2_FALLBACK", "glm-4.1v-thinking-flash,glm-4v-flash")
        models += [
            m.strip()
            for m in raw.split(",")
            if m.strip() and m.strip() != model
        ]

    throttled: list[str] = []
    for idx, m in enumerate(models):
        t0 = time.time()
        try:
            r = requests.post(
                f"{base.rstrip('/')}/chat/completions",
                json={
                    "model": m,
                    "messages": [{"role": "user", "content": "只回复 ok"}],
                    "max_tokens": 4,
                },
                headers={"Authorization": f"Bearer {key}"},
                timeout=40,
            )
        except requests.RequestException as exc:
            return False, f"网络不通：{exc}"
        dt = time.time() - t0

        if r.status_code == 200:
            tag = "" if idx == 0 else f"（主模型 {model} 限流，已降级）"
            return True, f"可用，{dt:.1f}s，模型 {m}{tag}"
        if r.status_code == 429 and idx + 1 < len(models):
            throttled.append(m)
            continue
        if r.status_code == 401:
            return False, "key 无效（401）"
        if r.status_code == 404:
            return False, f"端点或模型不存在：{m}"
        if r.status_code == 429:
            throttled.append(m)
            return False, f"限流（429）：{'、'.join(throttled)} 都不可用，换时段再试"
        return False, f"HTTP {r.status_code}：{r.text[:140]}"

    return False, f"全部不可用：{'、'.join(throttled)}"


def probe_explain() -> tuple[bool, str]:
    if not config.has_llm():
        return False, "未配置讲解模型的 key"
    t0 = time.time()
    try:
        r = requests.post(
            f"{config.llm_base().rstrip('/')}/chat/completions",
            json={
                "model": config.llm_model(),
                "messages": [{"role": "user", "content": "只回复 ok"}],
                "max_tokens": 4,
            },
            headers={"Authorization": f"Bearer {config.llm_key()}"},
            timeout=40,
        )
    except requests.RequestException as exc:
        return False, f"网络不通：{exc}"
    dt = time.time() - t0
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}：{r.text[:140]}"
    return True, f"可用，{dt:.1f}s，模型 {config.llm_model()}"


def probe_all() -> list[tuple[str, bool, str]]:
    out = []
    for label, fn in (
        ("SimpleTex", probe_simpletex),
        ("Qwen3-VL", probe_vlm),
        ("第二路 VLM", lambda: probe_vlm(slot=2)),
        ("讲解模型", probe_explain),
    ):
        try:
            ok, msg = fn()
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        out.append((label, ok, msg))
    return out
