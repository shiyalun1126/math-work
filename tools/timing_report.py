"""把 runs/ 里的耗时摊开看：P50 / P95 / max，以及冷热分开算。

为什么需要这个工具：文档里写「10 秒端到端」，但这句话从来没有对着数据核过。
不看分布的话，"慢"只是一句感觉；看了才知道慢在哪一段、有多少比例真的超过，
以及**上界到底是多少**。

数据来源是每次运行的 `runs/*.json`（`pipeline._save` 落的 timings）。
它是本机历史，不是压测：样本随时间累积，早期记录可能在修 bug 之前，
所以报告里连同时间范围一起给，避免拿旧数字当现状。

用法：
    F:/PyCharm/python.exe tools/timing_report.py
    F:/PyCharm/python.exe tools/timing_report.py --since 2026-11-12   # 只看某天之后
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

# 低于这个数的 perceive 基本是缓存命中：真实网络识别不会在 50 毫秒内返回
HOT_THRESHOLD = 0.05

# 单看某一阶段会漏掉"总时间"这件事，所以按顺序列出来
STAGES = ["perceive", "verify", "parse", "solve", "explain"]


def quantile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def load(since: str = "") -> list[dict]:
    out: list[dict] = []
    for path in RUNS.glob("*.json"):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        t = d.get("timings") or {}
        if "perceive" not in t:
            continue
        ts = datetime.fromtimestamp(path.stat().st_mtime)
        if since and ts.strftime("%Y-%m-%d") < since:
            continue
        sources = {r.get("source", "?") for r in (d.get("recognition") or [])}
        out.append({
            "ts": ts,
            "timings": t,
            "sources": sources,
            "total": sum(float(t.get(k) or 0) for k in STAGES),
        })
    out.sort(key=lambda r: r["ts"])
    return out


def row(name: str, values: list[float]) -> str:
    if not values:
        return f"{name:<12} {0:>5}        —       —       —"
    return (f"{name:<12} {len(values):>5} "
            f"{quantile(values, 0.5):>8.2f} {quantile(values, 0.95):>7.2f} "
            f"{max(values):>7.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="运行耗时分布")
    ap.add_argument("--since", default="", help="只统计这一天之后的记录，如 2026-11-12")
    args = ap.parse_args()

    rows = load(args.since)
    if not rows:
        print("runs/ 里没有带 timings 的记录。先跑一次识别（或用界面批改一张）再来。")
        return 2

    print(f"样本 {len(rows)} 次运行，"
          f"{rows[0]['ts']:%Y-%m-%d %H:%M} ~ {rows[-1]['ts']:%Y-%m-%d %H:%M}")
    print()
    print(f"{'阶段':<10} {'n':>5} {'P50':>8} {'P95':>7} {'max':>7}")
    for stage in STAGES:
        vals = [float(r["timings"][stage]) for r in rows if stage in r["timings"]]
        print(row(stage, vals))
    print(row("合计", [r["total"] for r in rows]))

    cold = [r for r in rows if float(r["timings"].get("perceive") or 0) >= HOT_THRESHOLD]
    hot = [r for r in rows if float(r["timings"].get("perceive") or 0) < HOT_THRESHOLD]
    print()
    print(f"按识别是否命中缓存分（阈值 {HOT_THRESHOLD}s）：")
    print(f"  冷（真打网络） n={len(cold)}  " +
          (f"perceive P50={quantile([float(r['timings']['perceive']) for r in cold], .5):.2f} "
           f"P95={quantile([float(r['timings']['perceive']) for r in cold], .95):.2f}" if cold else "—"))
    print(f"  热（缓存命中） n={len(hot)}  " +
          (f"perceive P50={quantile([float(r['timings']['perceive']) for r in hot], .5):.2f}" if hot else "—"))

    over = sum(1 for r in rows if r["total"] > 10)
    print()
    print(f"合计 > 10 秒的有 {over} 次（{over / len(rows):.1%}）。")
    print("注意：这是本机历史样本，不是压测；早期记录可能早于当前代码。")

    # 求解那一段出现过远超超时值的记录（守卫上线之前）
    slow = [r for r in rows if float(r["timings"].get("solve") or 0) > 30]
    if slow:
        print(f"\n有 {len(slow)} 次 solve 超过 30 秒，最早 "
              f"{slow[0]['ts']:%Y-%m-%d}：那时还没有守卫（R01），"
              "现在的上界由 SOLVE_TIMEOUT 决定。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
