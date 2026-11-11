"""读图：把「题干 + 图形」的一整页照片读成结构化 Problem。

和公式识别最大的不同：这里不要 LaTeX。模型如果被要求「转成公式」，
会把整段中文题干都塞进 LaTeX（实测如此），后面全废。
所以提示词明确要 JSON，并且要求「看不清就填 null，不要猜」。
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from pathlib import Path

import requests

from mathagent import config
from mathagent.cache import Cache, file_digest
from mathagent.perceive import RecognitionError, prepare_image

from .schema import Problem

# 提示词一改，旧缓存就必须失效。上一版是手写版本号，改完提示词忘了改号，
# 结果「改了没效果」查了半天——所以这里直接拿提示词正文算哈希，
# 只要动一个字，缓存自动失效，不依赖人记得改号。
# 整页题不能按公式那样缩到 640：题干是中文小字，缩完就糊了。
GEO_MAX_SIDE = 1400
# 反过来，太小的图也得放大。截图里截出来的题图常常只有 300px 宽，
# 放大前模型会把「AE 与 CD 相交于 F」整句吞掉——实测同一张图放大 3 倍后
# problem_text 才完整，goal 才从「题干未写完」变成「CE=CF」。
GEO_MIN_SIDE = 1100

PROMPT = (
    "这是一道几何题的照片，上面有中文题干，可能还有图形。"
    "请只输出一个 JSON 对象，不要任何解释，不要 markdown 代码块。\n"
    "JSON 结构（字段名必须原样）：\n"
    "{\n"
    '  "problem_text": "题干原文",\n'
    '  "given": ["已知条件，一条一句，如 AC=6、∠C=90°、D是AB的中点"],\n'
    '  "goal": "求什么",\n'
    '  "points": ["A", "B", "C"],\n'
    '  "segments": [["A", "B"]],\n'
    '  "circles": [{"center": "O", "through": ["A", "B"]}],\n'
    '  "marks": [{"type": "right_angle", "at": "C", "of": ["AC", "BC"]}],\n'
    '  "pos": {"M": [20, 70], "N": [75, 70], "K": [50, 15]},\n'
    '  "relations": ["用文字描述图中直接可见的关系"]\n'
    "}\n"
    "规则：\n"
    "1) points 只写单个大写字母。segments 只写图上真实画出来的线，用两个点的数组表示；"
    "延长线、辅助虚线如果画了也要写。\n"
    "2) marks 的 type 只能取 right_angle / equal_length / parallel / midpoint 这四种；"
    "at 是标记所在的点，of 是标记涉及的**两条**线段名（写全两个字母，如 AC 和 BC，不要只写 A、B）。\n"
    "3) pos 给每个点的近似位置：以图形所占区域为参照，左上角是 [0,0]、右下角是 [100,100]，"
    "y 向下增大。上面 M/N/K 只是格式示范，**必须按图中各点的实际位置重新估**，不要照抄那三个数字。\n"
    "4) 只写你在图上确实看到的。看不清、拿不准的字段填 null 或 []，绝对不要猜。\n"
    "5) 如果图里根本没有几何图形，points 填 [] 并在 relations 里写「无图形」。\n"
    "6) problem_text 要**一字不漏地照抄整段题干**，不许概括、不许省略。"
    "「……与……交于 P」「求证：……」这类句子最容易漏，而它们往往正是解题的关键。\n"
    "7) 如果某个点是两条线的交点（图上线穿过了它），就在 relations 里用两条线各自的名字写明，"
    "例如「F 在 AE 上，也在 CD 上」。不要写成「F 是 AC 与 DE 的交点」——字母必须和图上一致。\n"
    "8) 直接输出紧凑 JSON：不要换行、不要缩进、不要任何前后文字。"
)

# 智谱全系 max_tokens 硬上限 1024（实测 1200 起就报 400），
# 超出会被直接拒掉，所以按通道给不同的值。
MAX_TOKENS = {1: 1600, 2: 1024}

# 缓存 key 要包含一切会改变输出的东西：提示词正文、以及图像预处理的两个
# 尺寸。只按提示词算哈希是不够的——把最小边长从 640 提到 1100 那次，
# 提示词一个字没改，可读出来的结果完全不同，旧缓存却照样命中。
PROMPT_VERSION = "geo-" + hashlib.sha256(
    f"{PROMPT}|{GEO_MAX_SIDE}|{GEO_MIN_SIDE}".encode("utf-8")
).hexdigest()[:8]


def _strip_wrapper(text: str) -> str:
    """去掉模型输出的话术包装：think 段、代码块、前后废话。"""
    s = (text or "").strip()
    if "<answer>" in s:
        s = s.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    elif "<think" in s:
        # 有开头没结尾 = 思考写到一半被截断，剩下的全是半截话
        return ""
    s = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", s, flags=re.S)
    s = re.sub(r"<\|?[a-z_]*of_box\|?>", "", s)
    s = re.sub(r"^```(?:json)?\s*", "", s.strip())
    s = re.sub(r"\s*```$", "", s.strip())
    # 只保留最外层的 {...}，模型爱在前后加「好的，这是结果：」
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        s = s[i : j + 1]
    return s.strip()


def _close_brackets(s: str) -> str:
    """补齐被截断的 JSON：闭合未完成的字符串、数组、对象。"""
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    out = s + ('"' if in_str else "")
    for ch in reversed(stack):
        out += "}" if ch == "{" else "]"
    return out


def parse_json(text: str) -> dict:
    """把模型返回的文本解析成 dict，容忍截断和包装。"""
    s = _strip_wrapper(text)
    if not s:
        raise ValueError("模型没有返回可解析内容（可能是思考被截断）")
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        raise ValueError(f"顶层不是对象，而是 {type(obj).__name__}")
    except json.JSONDecodeError:
        pass

    # 截断很常见：末尾少几个括号，或者半句话。从后往前退，逐次补括号试。
    body = s.rstrip().rstrip(",")
    for cut in range(len(body), max(len(body) - 500, 0), -1):
        cand = body[:cut].rstrip().rstrip(",")
        if cand.count("{") <= cand.count("}"):
            continue
        try:
            obj = json.loads(_close_brackets(cand))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("JSON 截断严重，修补失败")


class GeoReader:
    """单个通道的读图器。接口和公式那边的 VLMRecognizer 对齐。"""

    def __init__(
        self,
        *,
        slot: int = 1,
        model: str | None = None,
        base: str | None = None,
        api_key: str | None = None,
        cache: Cache | None = None,
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
        self.cache = cache or Cache(subdir="geo")
        self.name = self.model
        self.used_model = self.model

    def available(self) -> bool:
        return bool(self._key)

    def _fallbacks(self) -> list[str]:
        if self.slot != 2:
            return []
        raw = config.get("VLM2_FALLBACK", "glm-4v-flash,glm-4.1v-thinking-flash")
        return [m.strip() for m in raw.split(",") if m.strip() and m.strip() != self.model]

    def _post(self, model: str, b64: str, max_tokens: int | None = None) -> str:
        mt = max_tokens or MAX_TOKENS.get(self.slot, 1200)
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            "temperature": 0.1,
            # 结构化输出比一句话长得多，800 不够用，会把 JSON 拦腰截断
            "max_tokens": mt,
        }
        try:
            r = requests.post(
                f"{self.base}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=120,
            )
        except requests.RequestException as exc:
            raise RecognitionError(f"网络不通：{exc}") from exc
        if r.status_code == 401:
            raise RecognitionError(f"key 无效（401），检查 .env 里的 {self.env_hint}")
        if r.status_code == 400 and "max_tokens" in r.text:
            # 不同厂商上限不一，撞上了就退回 1024 再来一次
            if mt > 1024:
                return self._post(model, b64, 1024)
        if r.status_code != 200:
            raise RecognitionError(f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise RecognitionError(f"返回结构异常：{str(data)[:180]}") from exc
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content or ""

    def read(self, image: str | Path) -> tuple[Problem | None, str]:
        """返回 (Problem, 错误说明)。失败时 Problem 为 None。"""
        if not self.available():
            return None, f"未配置 {self.env_hint}"

        digest = file_digest(image)
        key = (digest, self.model, PROMPT_VERSION)
        hit = self.cache.get(*key)
        if hit:
            return Problem.from_dict(hit, source=self.name, raw="<cache>"), ""

        data, _ = prepare_image(
            image, crop=True, max_side=GEO_MAX_SIDE, min_side=GEO_MIN_SIDE
        )
        b64 = base64.b64encode(data).decode()

        tried: list[str] = []
        last_err = ""
        for model in [self.model] + self._fallbacks():
            tried.append(model)
            t0 = time.time()
            try:
                text = self._post(model, b64)
                obj = parse_json(text)
            except RecognitionError as exc:
                last_err = str(exc)
                if "429" in last_err or "访问量" in last_err:
                    continue  # 限流，换下一个
                break
            except ValueError as exc:
                last_err = f"{model} 输出无法解析：{exc}"
                continue
            self.used_model = model
            prob = Problem.from_dict(obj, source=model, raw=text).normative()
            prob.source = model if model == self.model else f"{self.model}→{model}"
            self.cache.put(prob.to_dict(), *key)
            note = "" if model == self.model else f"主模型限流，改用 {model}"
            return prob, note
        return None, f"读完失败（试过 {', '.join(tried)}）：{last_err}"


def readers() -> list[GeoReader]:
    out: list[GeoReader] = []
    if config.has_vlm():
        out.append(GeoReader(slot=1))
    if config.has_vlm2():
        out.append(GeoReader(slot=2))
    return out
