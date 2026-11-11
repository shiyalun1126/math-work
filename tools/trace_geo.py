"""把一张几何题图完整跑一遍，把中间结果全部打印出来，方便定位是哪一步错了。

用法：python tools/trace_geo.py runs/_g5.png
和界面走的是同一条链路（geoagent.pipeline.run），只是不画图、不吞异常。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoagent import pipeline, render  # noqa: E402


def main(path: str) -> None:
    res = pipeline.run(path)
    print("=" * 70)
    print("verdict:", res.verdict, "confidence:", round(res.confidence, 3))
    if res.problem:
        print("--- 读到的结构 ---")
        print(json.dumps(res.problem.to_dict(), ensure_ascii=False, indent=2))
    print("--- 校验 ---")
    for c in res.checks:
        print(f"[{c.level}] {c.title} — {c.detail}")
    print("--- 求解 ---")
    s = res.solution
    print("ok:", s.ok, "unique:", s.unique, "value:", s.value, "residual:", s.residual)
    print("answer:", s.answer)
    print("error:", s.error)
    print("coords:", s.coords)
    print("goal:", s.goal_kind, s.goal_name, s.goal_arms)
    print("steps:")
    for i, st in enumerate(s.steps, 1):
        print(f"  {i}. {st}")
    print("notes:", res.notes)
    if res.problem:
        coords = s.coords if s.coords else None
        img = render.render(res.problem, coords, size=460, goal=pipeline.goal_of(res),
                            value=s.value)
        out = Path("runs") / ("trace_" + Path(path).stem + ".png")
        out.parent.mkdir(exist_ok=True)
        img.save(out)
        print("重绘图:", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/_g5.png")
