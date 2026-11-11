"""探测：现有 VLM 通道面对「题干 + 图形」版面时能做什么、不能做什么。

两个探针：
  A. 沿用现有识别提示词（要求转 LaTeX）—— 看点整页题会得到什么
  B. 换成结构化抽取提示词 —— 看能否拆出题干 / 条件 / 图形关系

只读探测，不写缓存以外的任何东西。
"""

from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mathagent import config  # noqa: E402

# 探针 A：项目现在实际用的提示词（转 LaTeX）
PROMPT_LATEX = (
    "把图片中的数学式子转写成 LaTeX 代码。要求："
    "1) 只输出 LaTeX 本体，不要解释，不要 markdown 代码块，不要 $ 或 $$；"
    "2) 忠实于图中写法，看不清也不要臆造，保持原样；"
    "3) 分式用 \\frac{}{}，上下标多于一个字符时加花括号；"
    "4) 如果图里不是数学式子，只输出 NOMATH。"
)

# 探针 B：结构化抽取
PROMPT_STRUCT = (
    "这是一道几何题的照片。请只输出一个 JSON，不要任何解释文字，不要 markdown 代码块。\n"
    "JSON 结构：\n"
    "{\n"
    '  "problem_text": "题干原文（含已知与所求）",\n'
    '  "given": ["已知条件，每条一句话，如 AC=6"],\n'
    '  "goal": "要求什么",\n'
    '  "points": ["A", "B"],\n'
    '  "segments": [["A", "B"]],\n'
    '  "circles": [{"center": "O", "through": ["A", "B"]}],\n'
    '  "marks": [{"type": "right_angle", "at": "C", "of": ["AC", "BC"]}],\n'
    '  "relations": ["AD = DB（D 是 AB 中点）"]\n'
    "}\n"
    "要求：只报图中确实画出来的点和线；"
    "marks 的 type 只能取 right_angle / equal_length / parallel / midpoint；"
    "看不清的字段填 null，不要猜。"
)


def encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def ask(base: str, model: str, key: str, img_b64: str, prompt: str) -> str:
    r = requests.post(
        f"{base.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1200,
        },
        headers={"Authorization": f"Bearer {key}"},
        timeout=120,
    )
    if r.status_code != 200:
        return f"<HTTP {r.status_code}> {r.text[:160]}"
    d = r.json()
    c = d["choices"][0]["message"]["content"]
    if isinstance(c, list):
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c


def strip_fence(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def chan() -> list[tuple[str, str, str, str]]:
    out = []
    if config.has_vlm():
        out.append(("通道1", config.vlm_base(), config.vlm_model(), config.vlm_key()))
    if config.has_vlm2():
        out.append(("通道2", config.vlm2_base(), config.vlm2_model(), config.vlm2_key()))
    return out


def main() -> None:
    imgs = [ROOT / "fixtures" / "geo" / "g1.png", ROOT / "fixtures" / "geo" / "g2.png"]
    for path in imgs:
        if not path.exists():
            print(f"缺图：{path}")
            continue
        b64 = encode(path)
        print("=" * 70)
        print(f"图片：{path.name}")
        for tag, base, model, key in chan():
            print(f"\n--- {tag} / {model} ---")
            a = ask(base, model, key, b64, PROMPT_LATEX)
            print(f"[探针A 转LaTeX] {a[:120]!r}")

            b = ask(base, model, key, b64, PROMPT_STRUCT)
            b2 = strip_fence(b)
            print(f"[探针B 结构化] 原始长度 {len(b)}")
            try:
                obj = json.loads(b2)
                print("  JSON 解析：成功")
                for k in ("problem_text", "goal"):
                    v = obj.get(k)
                    print(f"  {k:<13}: {str(v)[:80]}")
                for k in ("given", "points", "segments", "marks", "relations"):
                    v = obj.get(k)
                    print(f"  {k:<13}: {v}")
            except Exception as exc:  # noqa: BLE001
                print(f"  JSON 解析：失败 ({type(exc).__name__})")
                print(f"  原文前 200 字：{b2[:200]!r}")


if __name__ == "__main__":
    main()
