"""命令行跑一遍全部演示样例，不需要打开界面。

跑法：
  F:/PyCharm/python.exe demo.py            有密钥就走真实识别，没密钥用内置夹具
  F:/PyCharm/python.exe demo.py --offline  强制用夹具，结果可复现、不花钱
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mathagent import config, pipeline  # noqa: E402
from mathagent.solve import summary  # noqa: E402

FIX = config.ROOT / "fixtures" / "demo.json"


def main() -> int:
    offline = "--offline" in sys.argv
    demos = json.loads(FIX.read_text(encoding="utf-8"))
    print(config.describe() + ("（离线夹具）" if offline else ""))
    print("=" * 96)
    rows = []
    for stem, meta in demos.items():
        img = config.ROOT / "fixtures" / f"{stem}.png"
        r = pipeline.run(image=img, k=2 if not offline else 1, force_fixture=offline, save=False)
        v, sr = r.verification, r.solved
        rows.append((stem, meta["label"], v.verdict, v.confidence, summary(sr)))
        print(f"\n[{stem}] {meta['label']}  ->  {v.verdict.upper()}  conf={v.confidence:.0%}")
        for c in r.recognition:
            conf = f"{c.conf:.2f}" if c.conf is not None else "—"
            print(f"    · {c.source:10} conf={conf:5}  {c.latex or c.note}")
        for why in v.reasons:
            print(f"    ! {why}")
        print(f"    解析: kind={r.parsed.kind} ok={r.parsed.ok} {r.parsed.error}")
        print(f"    求解: {summary(sr)}")
        for st in sr.steps:
            print(f"        - {st.title}: {st.latex}")

    print("\n" + "=" * 96)
    print(f"{'样例':8} {'判定':8} {'一致度':6} 结论")
    for stem, label, verdict, conf, res in rows:
        print(f"{stem:8} {verdict:8} {conf:5.0%}  {res}")
    review = sum(1 for r in rows if r[2] == "review")
    print(f"\n共 {len(rows)} 例，其中 {review} 例触发人工确认（{review / len(rows):.0%}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
