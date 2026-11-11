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
import io
import json
import re
import time
from pathlib import Path

import requests
from PIL import Image, ImageFilter, ImageOps

from . import config
from .cache import Cache, file_digest
from .types import Candidate

MAX_SIDE = 1280  # 长边超过 1280 就缩，省 token 不影响单行公式
PAD_RATIO = 0.04  # 裁完留一点边，太贴边模型容易漏掉首尾符号

# 限流退避：一次 429 不代表"这条通道废了"，只代表"这一秒太多"。
# 免费档实测 3~5 秒后多半就通，直接放弃会让交叉验证退化成单源。
THROTTLE_RETRY = 2      # 整轮重试次数（每轮把候选模型都试一遍）
THROTTLE_SLEEP = 4.0    # 第 n 轮前等 THROTTLE_SLEEP × n 秒

# 模型的自言自语里才有的词。手写数学一行式子不会出现这些。
_META_PROSE = re.compile(r"图片|图中|用户|规则|假设|要求|说明|以上|注意|最终结果|重新")


class RecognitionError(RuntimeError):
    pass


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
    """EXIF 摆正 → 裁到墨迹 → 缩到长边 max_side → JPEG。原图不动。"""
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

    def recognize(self, image: str | Path, k: int = 1) -> list[Candidate]:
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

    def recognize(self, image: str | Path, k: int = 1) -> list[Candidate]:
        if not self.available():
            raise RecognitionError("未配置 SIMPLETEX_TOKEN（见 .env 通道 1）")
        digest = file_digest(image)
        hit = self.cache.get(digest, self.url)
        if hit:
            return [Candidate(**hit)]

        data, mime = prepare_image(image)
        t0 = time.time()
        try:
            r = requests.post(
                self.url,
                files={"file": (Path(image).name, data, mime)},
                headers={"token": config.simpletex_token()},
                timeout=40,
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
        self.cache.put(cand.to_dict(), digest, self.url)
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

    def _post(self, model: str, data_b64: str, temperature: float):
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
                timeout=90,
            )
        except requests.RequestException as exc:
            raise RecognitionError(f"网络不通：{exc}") from exc

    def _one(self, data_b64: str, temperature: float) -> str:
        self.used_model = self.model
        models = [self.model] + self._fallbacks()
        throttled: list[str] = []

        # 限流要退避重试，不能立刻认输。免费档的 429 是"这一秒太多"，
        # 几秒后同一模型往往就通了；原来一遇 429 就切下一个模型或直接报错，
        # 结果是整页只有一路识别活着，投票退化成"单源无旁证"→ 全行待核对。
        # 那看起来像"认不出来"，其实是没去问第二遍。
        for attempt in range(THROTTLE_RETRY + 1):
            for model in models:
                r = self._post(model, data_b64, temperature)
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
                time.sleep(THROTTLE_SLEEP * (attempt + 1))

        raise RecognitionError(
            f"限流（429）：{'、'.join(throttled) or '全部候选模型'} 重试 "
            f"{THROTTLE_RETRY} 次仍不可用，免费档高峰常见，换时段再试"
        )

    def recognize(self, image: str | Path, k: int = 1) -> list[Candidate]:
        if not self.available():
            raise RecognitionError(f"未配置 {self.env_hint}（见 .env）")
        digest = file_digest(image)
        data, _ = prepare_image(image)
        b64 = base64.b64encode(data).decode()

        out: list[Candidate] = []
        for i in range(max(1, k)):
            temp = round(0.2 + 0.3 * i, 2)
            key = (digest, self.model, str(temp))
            hit = self.cache.get(*key)
            if hit:
                out.append(Candidate(**hit))
                continue
            t0 = time.time()
            try:
                latex = self._one(b64, temp)
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
        return [c for c in out if c.latex] or out


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

    def recognize(self, image: str | Path, k: int = 1) -> list[Candidate]:
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
