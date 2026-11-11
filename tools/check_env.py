"""密钥与链路自检，命令行版。

用法：
    python tools/check_env.py                 # 只看配置状态
    python tools/check_env.py --probe         # 真调一次接口验证 key
    python tools/check_env.py --image a.jpg   # 直接跑一张图，看识别和求解结果
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from mathagent import config, pipeline  # noqa: E402
from mathagent.perceive import probe_all  # noqa: E402


def _w(s: str) -> int:
    """显示宽度：中文按 2 算，否则表格会歪。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(s))


def _pad(s: str, width: int) -> str:
    return str(s) + " " * max(0, width - _w(s))


def show_config() -> None:
    print(f"配置文件：{config.ENV_FILE}")
    print(f"           {'存在' if config.ENV_FILE.exists() else '不存在（从 .env.example 复制一份）'}")
    print()
    print(_pad("通道", 22) + _pad("状态", 10) + _pad("当前值", 22) + "备注")
    print("-" * 82)
    for s in config.status():
        if "env_file" in s:
            continue
        val = config.mask(s["value"]) if s["value"] else "（未填写）"
        mark = "就绪" if s["ready"] else "未配置"
        print(_pad(s["name"], 22) + _pad(mark, 10) + _pad(val, 22) + s.get("detail", ""))
    print()
    print(config.describe())


def do_probe() -> int:
    print("\n连通性自检（会真实调用接口）")
    print("-" * 82)
    bad = 0
    for label, ok, msg in probe_all():
        print(_pad("[OK]" if ok else "[FAIL]", 8) + _pad(label, 16) + msg)
        bad += 0 if ok else 1
    print()
    if bad:
        print("有通道不通。优先检查：key 是否粘全、是否开通了对应模型、网络是否可达。")
    return bad


def run_image(path: str) -> int:
    p = Path(path)
    if not p.exists():
        print(f"找不到图片：{p}")
        return 2
    print(f"\n识别：{p.name}")
    print("-" * 78)
    r = pipeline.run(image=p, k=2)
    for c in r.recognition:
        print(f"  [{c.source}] {c.latex or '（空）'} {c.note}")
    print(f"\n判定：{r.verification.verdict}  一致度 {r.verification.confidence:.0%}")
    for x in r.verification.reasons:
        print(f"  · {x}")
    print(f"\n选中：{r.verification.chosen}")
    if r.solved.solutions:
        print(f"答案：{'；'.join(r.solved.solutions)}")
    for i, st in enumerate(r.solved.steps, 1):
        print(f"  {i}. {st.title}：{st.latex}")
    if r.solved.dropped:
        print("  已剔除：" + "；".join(r.solved.dropped))
    if r.explain and r.explain.ok:
        print(f"\n讲评（{r.explain.model}）：{r.explain.text}")
    elif r.explain and r.explain.note:
        print(f"\n讲评未生成：{r.explain.note}")
    print(f"\n耗时：{ {k: round(v, 2) for k, v in r.timings.items()} }")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="密钥与链路自检")
    ap.add_argument("--probe", action="store_true", help="真实调用接口验证 key")
    ap.add_argument("--models", action="store_true", help="列出账号可用的多模态模型")
    ap.add_argument("--image", help="跑一张图，输出识别与求解结果")
    args = ap.parse_args()

    show_config()
    code = 0
    if args.models:
        from mathagent.perceive import list_vl_models

        names = list_vl_models()
        print("可用的多模态模型")
        print("-" * 78)
        if names:
            for n in names:
                mark = " ← 当前使用" if n == config.vlm_model() else ""
                print(f"  {n}{mark}")
        else:
            print("  取不到（key 无效或网络不通）")
        print()
    if args.probe:
        code |= do_probe()
    if args.image:
        code |= run_image(args.image)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
