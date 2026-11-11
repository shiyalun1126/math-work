"""评测通用统计工具。

刻意不依赖 numpy / scipy：这门课的评测只需要比例、置信区间和混淆矩阵，
引一个科学计算栈进 requirements 是纯负债。

所有随机过程都带固定种子，报告里的数字必须能复现——否则消融实验没法比。
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Callable, Iterable

DEFAULT_SEED = 20261110
N_BOOT = 1000


def ratio(num: int, den: int) -> float | None:
    """分母为 0 时返回 None，而不是 0。

    「这道题没有伪推导样本」和「伪推导一个都没抓到」是两件完全不同的事，
    塞成 0 会让报告凭空多出一个难看的数字。
    """
    return num / den if den else None


def bootstrap_ci(values: list[int | float], n_boot: int = N_BOOT,
                 seed: int | None = None) -> tuple[float, float] | None:
    """0/1 序列的比例 bootstrap 95% 置信区间（百分位法）。

    seed 留空才读模块级 DEFAULT_SEED：默认参数在 import 时就绑定了值，
    写成 `seed=DEFAULT_SEED` 的话，`run_eval --seed` 改了也白改。
    """
    if not values:
        return None
    rng = random.Random(DEFAULT_SEED if seed is None else seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return lo, hi


def pct(x: float | None, ci: tuple[float, float] | None = None) -> str:
    """写成 `62.5% [51.3, 73.1]`，和技术文档 §6.3 的约定一致。"""
    if x is None:
        return "—"
    if ci is None:
        return f"{x * 100:.1f}%"
    return f"{x * 100:.1f}% [{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]"


def metric(bits: list[int], n_boot: int = N_BOOT,
           seed: int | None = None) -> dict:
    """一组 0/1 观测 → 比例、区间、样本量。"""
    return {
        "n": len(bits),
        "hit": sum(bits),
        "value": ratio(sum(bits), len(bits)),
        "ci": bootstrap_ci(bits, n_boot, seed) if bits else None,
    }


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else None)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def confusion(pairs: list[tuple[str, str]], labels: Iterable[str] | None = None) -> dict:
    """(真值, 预测) 列表 → 混淆矩阵。labels 固定行/列顺序，便于两次运行对比。"""
    seen = labels or sorted({a for a, _ in pairs} | {b for _, b in pairs})
    seen = list(seen)
    table = {a: Counter() for a in seen}
    for a, b in pairs:
        if a in table:
            table[a][b] += 1
    return {"labels": seen,
            "rows": {a: [table[a][b] for b in seen] for a in seen}}


def by_group(records: list[dict], hit_key: str, group_key: str = "group",
             weight_key: str | None = None) -> dict[str, dict]:
    """分层统计。整体准确率会掩盖难例，报告必须能拆开看。"""
    buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in records:
        hit = int(r[hit_key])
        w = int(r.get(weight_key, 1)) if weight_key else 1
        buckets[r.get(group_key, "未分组")].append((hit, w))
    out = {}
    for g, items in buckets.items():
        n = sum(w for _, w in items)
        h = sum(hit * w for hit, w in items)
        out[g] = {"n": n, "hit": h, "value": ratio(h, n)}
    return dict(sorted(out.items()))


def table(headers: list[str], rows: list[list[str]]) -> str:
    """markdown 表格。单元里的竖线会破坏表格，先转义。"""
    def cell(x) -> str:
        return str(x).replace("|", "\\|").replace("\n", " ")
    head = "| " + " | ".join(cell(h) for h in headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def select(records: list[dict], pred: Callable[[dict], bool]) -> list[dict]:
    return [r for r in records if pred(r)]
