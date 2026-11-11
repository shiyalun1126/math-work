"""配置：项目根目录的 `.env` 是第一优先级。

读取顺序（前者覆盖后者）：
    1. 系统环境变量（便于 CI / 临时覆盖）
    2. 项目根目录 .env
    3. config.json（旧版格式，保留兼容）

设计原则：没有密钥也要能跑（走演示夹具 / 手工输入 LaTeX），
这样别人 clone 下来不会第一步就卡在注册账号上。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
CONFIG_FILE = ROOT / "config.json"
CACHE_DIR = ROOT / ".cache"

# 阿里云百炼（Qwen 系列）OpenAI 兼容端点
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_VLM_MODEL = "qwen3-vl-plus"
DEFAULT_LLM_MODEL = "qwen-flash"

# 第二路 VLM：智谱开放平台（GLM-4.6V-Flash 永久免费、国内直连、OpenAI 兼容）。
# 跨源投票要的是"不同厂商"，同厂商换模型名凑不出独立旁证——所以这里默认接别家。
ZHIPU_BASE = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_VLM2_MODEL = "glm-4.6v-flash"

# SimpleTex 开放平台
SIMPLETEX_HOST = "https://server.simpletex.cn"
SIMPLETEX_TURBO = f"{SIMPLETEX_HOST}/api/latex_ocr_turbo"
SIMPLETEX_STD = f"{SIMPLETEX_HOST}/api/latex_ocr"

# 这两组名字等价，填哪个都认。写两个是为了照顾不同平台的叫法习惯。
_VLM_KEY_ALIASES = ("VLM_API_KEY", "DASHSCOPE_API_KEY")
_LLM_KEY_ALIASES = ("LLM_API_KEY", "DASHSCOPE_API_KEY", "VLM_API_KEY")

_FILE: dict = {}


# ---------------------------------------------------------------- 加载

def _parse_env_manually(path: Path) -> int:
    """没装 python-dotenv 时的兜底解析，支持 KEY=VALUE 与 # 注释。"""
    n = 0
    try:
        text = path.read_text(encoding="utf-8-sig")
    except Exception:
        return 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        os.environ.setdefault(key, val)
        n += 1
    return n


def load_env(override: bool = False) -> bool:
    """把 .env 读进环境变量。override=True 时覆盖已有的（用于界面改完密钥后刷新）。"""
    if not ENV_FILE.exists():
        return False
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE, override=override, encoding="utf-8")
        return True
    except Exception:
        if override:
            for key in list(os.environ):
                pass
        _parse_env_manually(ENV_FILE)
        return True


def _load_json() -> dict:
    global _FILE
    if not CONFIG_FILE.exists():
        _FILE = {}
        return _FILE
    try:
        _FILE = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        _FILE = {}
    return _FILE


def reload() -> None:
    """重新读 .env 和 config.json。界面上保存密钥后调用，不必重启进程。"""
    load_env(override=True)
    _load_json()


load_env()
_load_json()


# ---------------------------------------------------------------- 取值

def get(key: str, default: str = "") -> str:
    val = os.environ.get(key)
    if val is not None and val.strip():
        return val.strip()
    if key in _FILE and str(_FILE[key]).strip():
        return str(_FILE[key]).strip()
    return default


def _first(*keys: str) -> str:
    for k in keys:
        v = get(k)
        if v:
            return v
    return ""


def simpletex_token() -> str:
    return get("SIMPLETEX_TOKEN")


def simpletex_quality() -> str:
    """standard（默认，手写更稳，赠 500 次）/ turbo（更快更便宜，赠 1000 次）。"""
    q = get("SIMPLETEX_MODEL", "standard").strip().lower()
    return q if q in ("standard", "turbo") else "standard"


def simpletex_url() -> str:
    """允许用 SIMPLETEX_BASE 换镜像或指向本地假服务器（离线自测用）。"""
    host = get("SIMPLETEX_BASE", SIMPLETEX_HOST).rstrip("/")
    path = "/api/latex_ocr" if simpletex_quality() == "standard" else "/api/latex_ocr_turbo"
    return host + path


def vlm_key() -> str:
    return _first(*_VLM_KEY_ALIASES)


def vlm_base() -> str:
    return get("VLM_BASE_URL", DASHSCOPE_BASE)


def vlm_model() -> str:
    return get("VLM_MODEL", DEFAULT_VLM_MODEL)


# ---------------------------------------------------------------- 第二路 VLM
# 与 VLM_* 完全并列的一组配置。没有它也能跑，只是退回单源模式。

def vlm2_key() -> str:
    return _first("VLM2_API_KEY", "ZHIPU_API_KEY")


def vlm2_base() -> str:
    return get("VLM2_BASE_URL", ZHIPU_BASE)


def vlm2_model() -> str:
    return get("VLM2_MODEL", DEFAULT_VLM2_MODEL)


def has_vlm2() -> bool:
    return bool(vlm2_key())


def llm_key() -> str:
    return _first(*_LLM_KEY_ALIASES)


def llm_base() -> str:
    return get("LLM_BASE_URL") or vlm_base()


def llm_model() -> str:
    return get("LLM_MODEL", DEFAULT_LLM_MODEL)


def explain_enabled() -> bool:
    """讲解层默认开，但要真有 key 才生效。ENABLE_EXPLAIN=0 可关掉。"""
    flag = get("ENABLE_EXPLAIN", "1").lower()
    if flag in ("0", "false", "no", "off"):
        return False
    return bool(llm_key())


# 旧名字，pipeline / perceive 里还在用
def dashscope_key() -> str:
    return vlm_key()


def has_simpletex() -> bool:
    return bool(simpletex_token())


def has_vlm() -> bool:
    return bool(vlm_key())


def has_dashscope() -> bool:
    return has_vlm()


def has_llm() -> bool:
    return bool(llm_key())


# ---------------------------------------------------------------- 展示

def mask(secret: str, keep: int = 4) -> str:
    """密钥脱敏，只留头尾几位，截图/投屏时不怕泄露。"""
    s = (secret or "").strip()
    if not s:
        return ""
    if len(s) <= keep * 2:
        return s[:2] + "*" * max(2, len(s) - 2)
    return f"{s[:keep]}…{s[-keep:]}"


def status() -> list[dict]:
    """给界面用的通道状态表。"""
    env_ok = ENV_FILE.exists()
    return [
        {
            "name": "SimpleTex 公式识别",
            "role": "专用模型 · 主通道",
            "key": "SIMPLETEX_TOKEN",
            "value": simpletex_token(),
            "ready": has_simpletex(),
            "detail": "标准模型（赠 500 次）"
            if simpletex_quality() == "standard"
            else "轻量模型（赠 1000 次）",
        },
        {
            "name": "Qwen3-VL 多模态",
            "role": "通用大模型 · 第一通道",
            "key": "DASHSCOPE_API_KEY",
            "value": vlm_key(),
            "ready": has_vlm(),
            "detail": f"{vlm_model()} @ {vlm_base()}",
        },
        {
            "name": "第二路多模态",
            "role": "跨源投票 · 换厂商才有意义",
            "key": "VLM2_API_KEY",
            "value": vlm2_key(),
            "ready": has_vlm2(),
            "detail": f"{vlm2_model()} @ {vlm2_base()}",
        },
        {
            "name": "讲解大模型",
            "role": "只讲思路，不做计算",
            "key": "LLM_API_KEY",
            "value": llm_key(),
            "ready": explain_enabled(),
            "detail": llm_model(),
        },
        {"env_file": str(ENV_FILE), "exists": env_ok},
    ]


def describe() -> str:
    """一行说明当前可用能力，UI 顶部直接显示。"""
    parts = []
    parts.append("SimpleTex ✓" if has_simpletex() else "SimpleTex ✗")
    parts.append("Qwen3-VL ✓" if has_vlm() else "Qwen3-VL ✗")
    parts.append("第二路 ✓" if has_vlm2() else "第二路 ✗")
    parts.append("讲解 ✓" if explain_enabled() else "讲解 ✗")
    if has_simpletex() or has_vlm() or has_vlm2():
        n = sum(1 for f in (has_simpletex(), has_vlm(), has_vlm2()) if f)
        src = f"{n} 路独立来源" if n >= 2 else "单源（置信度上限 80%）"
        return f"云端识别已就绪 · {src} · " + " / ".join(parts)
    return "未配置密钥：当前为演示模式（内置样例 + 可手工输入 LaTeX）。填好 .env 后点「测试连接」"


CONFIG_TEMPLATE = """{
  "SIMPLETEX_TOKEN": "在这里填 SimpleTex 用户授权令牌（simpletex.cn 用户中心获取）",
  "DASHSCOPE_API_KEY": "在这里填阿里云百炼 API Key（可选，用于第二通道）"
}
"""


def ensure_config_template() -> Path:
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    return CONFIG_FILE


def set_env_value(key: str, value: str) -> None:
    """改写 .env 里的某一项（保留注释与其他行）。界面上填密钥时用。"""
    lines = []
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    hit = False
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("#") or "=" not in s:
            continue
        if s.split("=", 1)[0].strip().lstrip("export ").strip() == key:
            lines[i] = f"{key}={value}"
            hit = True
    if not hit:
        lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[key] = value
