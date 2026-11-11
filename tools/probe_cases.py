"""能力探测：造一批不在内置样例里的题目，走真实识别通道，看边界到底在哪。

用法：
    F:/PyCharm/python.exe tools/probe_cases.py

图片生成到 fixtures/probe/，结果只打印不落盘（pipeline 自己会存 runs/）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from make_demo_images import render  # noqa: E402  复用画图逻辑

from mathagent import pipeline  # noqa: E402

OUT = ROOT / "fixtures" / "probe"

# 故意选内置七例完全没有覆盖的题型
CASES = {
    "p1": ["sin x = 1/2"],                    # 三角方程
    "p2": ["x³ - 6x² + 11x - 6 = 0"],         # 三次方程
    "p3": ["√(x + 3) = x - 3"],               # 根式方程
    "p4": ["ln x = 1"],                       # 对数方程
    "p5": ["∫ x sin x dx"],                   # 分部积分
    "p6": ["2ˣ = 8"],                         # 指数字形（ˣ 可能没字形）
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("生成探测图：")
    for i, (stem, lines) in enumerate(CASES.items()):
        render(lines, OUT / f"{stem}.png", angle=-1.0 + 0.35 * (i % 3))

    print("\n真实通道识别（不进夹具、不吃缓存时才会调 API）：\n")
    header = f"{'题':<6}{'识别到的 LaTeX':<34}{'结论':<28}{'置信':<8}{'耗时'}"
    print(header)
    print("-" * 100)

    for stem, lines in CASES.items():
        img = OUT / f"{stem}.png"
        t0 = time.time()
        try:
            r = pipeline.run(image=img, k=2)
        except Exception as e:  # 单题炸掉不该中断整批
            print(f"{stem:<6}{'<异常>':<34}{type(e).__name__ + ': ' + str(e)[:20]:<28}")
            continue
        dt = time.time() - t0

        latex = r.verification.chosen or "<空>"
        if len(latex) > 32:
            latex = latex[:30] + ".."

        sv = r.solved
        if sv.solutions:
            concl = "解 " + ", ".join(str(s) for s in sv.solutions)
        elif sv.ok:
            concl = "已化简"
        else:
            concl = "未解出：" + (sv.error or r.parsed.error or "")[:12]
        if len(concl) > 26:
            concl = concl[:24] + ".."

        conf = f"{r.verification.confidence:.0%}" if r.verification else "-"
        print(f"{stem:<6}{latex:<34}{concl:<28}{conf:<8}{dt:.1f}s")

    print("\n图在 fixtures/probe/，原始结果在 runs/")


if __name__ == "__main__":
    sys.exit(main())
