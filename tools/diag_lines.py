"""逐行核对：把每一行的**所有候选**摊开，看判"待核对"的根子在哪。

用法：
    python tools/diag_lines.py W04 W12

只走缓存能命中的路径也照样跑：识别结果按图哈希落盘，同一张行图第二次是免费的，
所以这个脚本可以反复跑而不会重复花钱。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mathagent import photocheck  # noqa: E402
from mathagent.lines import split_page, save_crops  # noqa: E402
from mathagent.perceive import build_ensemble  # noqa: E402
from mathagent.types import Candidate  # noqa: E402


def run(page_id: str, k: int = 1) -> None:
    img = ROOT / "data" / "inbox" / f"{page_id}.jpg"
    page = split_page(img)
    out_dir = ROOT / "data" / "work" / f"diag_{page_id}"
    paths = save_crops(
        type(page)(page=page.page, crops=page.crops, size=page.size),
        out_dir,
        stem=f"diag_{page_id}",
    )
    recs = build_ensemble(k=k)
    print(f"\n{'='*72}\n{page_id}  切出 {len(page.crops)} 行  识别器={[r.name for r in recs]}")
    lines = []
    for crop, path in zip(page.crops, paths):
        print(f"\n--- 行{crop.index} box={crop.box} {crop.image.size[0]}x{crop.image.size[1]}px")
        cands: list[Candidate] = []
        for rec in recs:
            try:
                got = rec.recognize(path, k=k)
            except Exception as exc:  # noqa: BLE001
                print(f"    {rec.name:<14} ERR {type(exc).__name__}: {exc}")
                continue
            for c in got:
                print(f"    {rec.name:<14} {c.latex!r}  {('[' + c.note + ']') if c.note else ''}")
                cands.append(c)
        latex, conf, others, verdict, notes = photocheck._vote(cands, len(recs) > 1)
        print(f"    投票 ⇒ {latex!r} conf={conf:.2f} verdict={verdict}")
        for n in notes:
            print(f"       · {n}")
        texts = [photocheck.readable(c.latex) for c in cands if (c.latex or "").strip()]
        if verdict != "ok" and latex and photocheck._prose_ok(latex, cands):
            verdict = "prose"
        lines.append(
            photocheck.LineRead(
                index=crop.index,
                crop=path,
                latex=latex,
                text=photocheck.readable(latex),
                verdict=verdict,
                confidence=conf,
                sources=sorted({c.source for c in cands if (c.latex or "").strip()}),
                reasons=notes,
                alternatives=others,
            )
        )
    photocheck._tiebreak_by_math(lines)
    print("\n  ── 裁决后 ──")
    for lr in lines:
        print(f"    行{lr.index} {lr.verdict:<7} conf={lr.confidence:.2f} sure={lr.sure} {lr.text!r}")
        for n in lr.reasons:
            print(f"         · {n}")


if __name__ == "__main__":
    ids = sys.argv[1:] or ["W04", "W12"]
    for pid in ids:
        run(pid)
