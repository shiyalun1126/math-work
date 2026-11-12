"""M²E 真实版式下的切行评测（外部验证集）。

`tools/selftest_split.py` 用的是合成图，切得再准也说明不了真实试卷什么样。
M²E 是真实手机拍摄的多行公式块，`tex` 用 `\n` 给出逐行标注，于是
「切出来的行数 == 标注行数」可以当一个不依赖 VLM、不依赖人工的硬指标。

**参数纪律（脚本层面强制）**：
  - 调参只能在 `train` 上做（`--tune` 且 `--split train`）；
  - 报数字只能在 `val/test` 上做，参数从 `--params` 文件读进来；
  - 禁止在 test 上边调边报，否则 100% 也只是一个拟合出来的数。

**这个指标的边界**：只比对「行数」，不比对「行框位置」。M²E 没有逐行 bbox，
所以 exact=100% 也不能证明每行内容切对了；报告里必须连同这句一起引用。

**可复核性**：报告 JSON 带全量逐块 `records`（`id/n_true/n_pred/delta/...`）、
`seed`、`n_boot` 与输入指纹（标注文件 sha256 + 标注内容 sha256 + 参数 sha256）。
任何人拿到同一份标注和参数，都能按 records 重算 exact / MAE / CI，不需要重跑切行。

用法：

  # 1. 先用默认参数报一个 baseline（不调参，最诚实）
  F:/PyCharm/python.exe eval/run_line_eval.py --split test --limit 500
  # 2. 想调参：只在 train 上扫，结果写 reports/line_params.json
  F:/PyCharm/python.exe eval/run_line_eval.py --split train --tune --tune-limit 300
  # 3. 用调好的参数，在没碰过的 test 上报最终数字
  F:/PyCharm/python.exe eval/run_line_eval.py --split test --params reports/line_params.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import metrics  # noqa: E402
from mathagent.lines import debug_sheet, split_page  # noqa: E402

DEFAULT_BLOCKS = ROOT / "data" / "public" / "m2e" / "blocks.jsonl"
DEFAULT_IMAGES = ROOT / "data" / "public" / "m2e_raw" / "images"
REPORTS = ROOT / "reports"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 与 mathagent.lines.split_page 的默认值保持一致，改这里要写进报告。
# page_mode 写 null 就是"按短边自动判断"：M²E 里短边 ≥400 px 的块会走整页模式，
# 那几张图上下面三个阈值参数根本不生效——这一点必须写在参数块里，否则读者会
# 以为自己看到的是"这组阈值跑出来的数"。
DEFAULT_PARAMS = {
    "deskew": True,
    "threshold_ratio": 0.03,
    "min_gap_ratio": 0.012,
    "min_height": 10,
    "page_mode": None,
}

# 小网格：M²E 是紧凑公式块小图，整页作业照的默认参数（尤其纠偏）未必适用，
# 所以把 deskew 也纳入；刻意不做成大搜索，免得把 test 泄漏进调参。
GRID = [
    {"deskew": ds, "threshold_ratio": tr, "min_gap_ratio": mg, "min_height": mh}
    for ds in (True, False)
    for tr in (0.01, 0.03)
    for mg in (0.004, 0.012)
    for mh in (6, 10)
]

CAVEATS = [
    "外部验证集：M²E (Wente47 / ModelScope)，CC BY-NC 4.0，非本项目自采数据，指标须与 self_test 分表报告。",
    "M²E 是多行公式块，不是学生整页解题过程；没有对错/错因标注，不能当批改评测集。",
    "只比对行数，不比对行框位置：exact 高不等于每行内容都切对，需配合 --debug 目检。",
    "默认纳入含 \\t 的竖式对齐块（M²E 中占多数，全跳过会让样本失去代表性），按 align 分层报；"
    "--exclude-align 可排除。",
    "M²E 图多为 70~300 px 的紧凑小图，与整页作业照不是同一分布；"
    "这里的数字不能外推成产品对整页作业的切行准确率。",
    "`page_mode: null` 表示按短边自动判断：短边 ≥400 px 的块走整页模式，"
    "此时 `threshold_ratio` / `min_gap_ratio` / `deskew` 对它不生效。",
]


# --------------------------------------------------------------------------- 数据


def load_blocks(path: Path, split: str, limit: int = 0) -> tuple[list[dict], int]:
    """返回（本次要评测的块，该 split 的总块数）。

    两者分开是因为 `--limit` 很常用：报告里写"该 split 共 2000 块、这次评测前 500 块"
    才是实话，把 `len(blocks)` 当总数写会让下一个人以为数据集只有 500 块。
    """
    if not path.exists():
        raise SystemExit(
            f"找不到 {path}\n"
            "先跑：F:/PyCharm/python.exe tools/import_public.py m2e --src "
            "data/public/m2e_raw --limit 2000"
        )
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        if rec.get("split") == split:
            rows.append(rec)
    total = len(rows)
    return (rows[:limit] if limit else rows), total


def index_images(directory: Path) -> tuple[dict[str, Path], int]:
    """zip 里常带子目录，评测按 basename 找图；重名记数但不静默覆盖。"""
    out: dict[str, Path] = {}
    collisions = 0
    if not directory.is_dir():
        return out, 0
    for p in directory.rglob("*"):
        if p.suffix.lower() not in IMG_EXT:
            continue
        if p.name in out:
            collisions += 1
            continue
        out[p.name] = p
    return out, collisions


def group_of(n: int) -> str:
    if n <= 2:
        return "1-2 行"
    if n == 3:
        return "3 行"
    if n == 4:
        return "4 行"
    return "5 行及以上"


# --------------------------------------------------------------------------- 指纹


def _digest(obj) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def input_fingerprint(blocks_path: Path, blocks: list[dict], params: dict,
                      split: str) -> dict:
    """把「跑的是哪份标注、哪组参数」压成可比对的摘要。

    逐块 records 落盘之后，比对两份报告时顺序应该是：先看指纹一不一样（输入变没变），
    再看某几块 delta 变了没（代码变没变）。没有指纹就只能重跑一遍再猜。
    """
    raw = blocks_path.read_bytes() if blocks_path.exists() else b""
    labels = [[str(b.get("id", "")), int(b.get("n_lines") or 0),
               int(bool(b.get("has_align")))] for b in blocks]
    try:
        shown = blocks_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = str(blocks_path)
    return {
        "blocks_file": shown,
        "blocks_sha256": hashlib.sha256(raw).hexdigest(),
        "labels_sha256": _digest(labels),
        "params_sha256": _digest(params),
        "n_loaded": len(blocks),
        "split": split,
    }


# 偏差分桶：一眼能看出是整体欠切（负数堆在一起）还是零星过切。
DELTA_BUCKETS = [
    (None, -3, "≤ -3"),
    (-2, -2, "-2"),
    (-1, -1, "-1"),
    (0, 0, "0（行数一致）"),
    (1, 1, "+1"),
    (2, 2, "+2"),
    (3, None, "≥ +3"),
]


def delta_histogram(records: list[dict]) -> list[list[str]]:
    rows = []
    for lo, hi, label in DELTA_BUCKETS:
        n = sum(1 for r in records
                if (lo is None or r["delta"] >= lo) and (hi is None or r["delta"] <= hi))
        rows.append([label, str(n)])
    return rows


def worst_records(records: list[dict], n: int = 15) -> list[dict]:
    """按 |delta| 挑最该先看目检的块；同分时按 id 稳定排序，两次跑出来的名单一致。"""
    return sorted(records, key=lambda r: (-abs(r["delta"]), str(r["id"])))[:n]


# --------------------------------------------------------------------------- 评测


def eval_blocks(blocks: list[dict], images: dict[str, Path], params: dict,
                split_fn=split_page, skip_align: bool = False,
                debug_dir: Path | None = None, debug_n: int = 0) -> dict:
    records: list[dict] = []
    missing: list[str] = []
    skipped_align = 0
    errors: list[str] = []
    debug_written = 0

    for b in blocks:
        if skip_align and b.get("has_align"):
            skipped_align += 1
            continue
        name = Path(str(b.get("image", ""))).name
        path = images.get(name)
        if path is None:
            missing.append(str(b.get("id", name)))
            continue
        try:
            result = split_fn(path, **params)
            n_pred = len(result)
            note = str(getattr(result, "note", "") or "")
        except Exception as exc:  # noqa: BLE001 - 单张坏图不该让整批评测崩掉
            errors.append(f"{b.get('id', name)}: {exc}")
            continue
        if debug_dir is not None and debug_written < debug_n and getattr(result, "sheet", None):
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_sheet(result).save(debug_dir / f"{b.get('id', 'x')}.jpg", quality=90)
            debug_written += 1
        n_true = int(b.get("n_lines", 0))
        if n_true <= 0:
            errors.append(f"{b.get('id', name)}: 标注行数为 0")
            continue
        delta = n_pred - n_true
        size = list(getattr(result, "size", ()) or ())
        records.append({
            "id": b.get("id", name), "n_true": n_true, "n_pred": n_pred,
            "delta": delta, "exact": int(delta == 0),
            "within1": int(abs(delta) <= 1), "group": group_of(n_true),
            "align_group": "竖式对齐" if b.get("has_align") else "普通",
            "size": size, "note": note,
        })
    return {"records": records, "missing": missing, "skipped_align": skipped_align,
            "errors": errors, "debug_written": debug_written}


def score(records: list[dict], seed: int | None, n_boot: int) -> dict:
    if not records:
        return {"n": 0, "note": "没有可评测的块，不产出任何指标"}
    exact = metrics.metric([r["exact"] for r in records], n_boot, seed)
    within1 = metrics.metric([r["within1"] for r in records], n_boot, seed)
    deltas = [r["delta"] for r in records]
    return {
        "n": len(records),
        "exact": exact,
        "within1": within1,
        "mae": sum(abs(d) for d in deltas) / len(deltas),
        "bias": sum(deltas) / len(deltas),
        "over": metrics.ratio(sum(d > 0 for d in deltas), len(deltas)),
        "under": metrics.ratio(sum(d < 0 for d in deltas), len(deltas)),
        "by_group": metrics.by_group(records, "exact"),
        "by_align": metrics.by_group(records, "exact", "align_group"),
    }


def tune(blocks: list[dict], images: dict[str, Path], params_base: dict,
         split_fn=split_page, tune_limit: int = 300, skip_align: bool = False) -> dict:
    """在 train 上扫小网格，按 exact 选参数。返回的表要进报告，方便复现。"""
    subset = blocks[:tune_limit] if tune_limit else blocks
    table_rows = []
    for g in GRID:
        params = {**params_base, **g}
        got = eval_blocks(subset, images, params, split_fn, skip_align)
        s = score(got["records"], None, 100)  # 调参不需要置信区间，省时间
        table_rows.append({
            "params": g, "n": s.get("n", 0),
            "exact": s.get("exact", {}).get("value"),
            "mae": s.get("mae"),
        })
    usable = [r for r in table_rows if r["n"] and r["exact"] is not None]
    if not usable:
        raise SystemExit("调参没有任何可评测的块：检查图片目录与 --images 是否对得上。")
    best = max(usable, key=lambda r: (r["exact"], -(r["mae"] or 99)))
    return {
        "split": "train", "n_candidates": len(GRID), "n_tune": len(subset),
        "best": {**params_base, **best["params"]}, "table": table_rows,
        "note": "参数在 train 上选出；最终指标必须在 val/test 上用这组参数重跑。",
    }


# --------------------------------------------------------------------------- 报告


def render_md(report: dict) -> str:
    s = report["metrics"]
    p = report["params"]
    lines = [
        "# M²E 真实版式切行评测（外部验证集）",
        "",
        f"生成时间：{report['generated']} · split=`{report['split']}` · "
        f"参数来源：`{report['params_source']}`",
        "",
        "> ⚠️ 这不是自采数据，数字不能和 `self_test` 混在一张表里报。",
    ]
    if report.get("tune"):
        lines += [
            "",
            "> ⚠️ 本报告是 **train 上调参跑出来的数**，只用于选参数，不能当最终指标。"
            "请用 `--params reports/line_params.json --split test` 重跑后引用那个报告。",
        ]
    lines += [
        "",
        "## 口径与边界",
        "",
    ]
    lines += [f"- {c}" for c in CAVEATS]
    lines += [
        "",
        "## 样本",
        "",
        f"- 该 split 共 `{report['n_blocks']}` 块；本次载入 "
        f"`{report.get('n_loaded', report['n_blocks'])}` 块"
        + ("（`--limit` 截断，数字只代表这一截）"
           if report.get("n_loaded", report["n_blocks"]) < report["n_blocks"] else "")
        + f"；实际评测 `{s.get('n', 0)}` 块",
        f"- 按 --exclude-align 跳过 `{report['skipped_align']}` 块；"
        f"找不到图片 `{len(report['missing'])}`；切图报错 `{len(report['errors'])}`",
    ]
    if report["missing"]:
        lines.append(f"- 缺图样例：{', '.join(report['missing'][:5])}")
    if report["errors"]:
        lines.append(f"- 报错样例：{report['errors'][0]}")
    lines += ["", "## 指标", ""]
    if not s.get("n"):
        lines.append("**没有可评测的样本，未产出指标。** 先按文档 §3~§5 完成下载与导入。")
    else:
        lines.append(metrics.table(
            ["指标", "数值", "95% CI"],
            [
                ["行数完全一致 exact",
                 metrics.pct(s["exact"]["value"], s["exact"]["ci"]),
                 f"{s['exact']['hit']}/{s['exact']['n']}"],
                ["误差 ≤1 行", metrics.pct(s["within1"]["value"], s["within1"]["ci"]),
                 f"{s['within1']['hit']}/{s['within1']['n']}"],
                ["平均绝对行数误差 MAE", f"{s['mae']:.3f}", "—"],
                ["平均偏差 bias（正=多切）", f"{s['bias']:+.3f}", "—"],
                ["过切率", metrics.pct(s["over"]), "—"],
                ["欠切率", metrics.pct(s["under"]), "—"],
            ]))
        lines += ["", "### 按标注行数分层（exact）", ""]
        rows = [[g, str(d["n"]), metrics.pct(d["value"])]
                for g, d in sorted(s["by_group"].items())]
        lines.append(metrics.table(["真值行数", "块数", "行数完全一致"], rows))
        lines += ["", "### 按是否竖式对齐分层（exact）", ""]
        rows = [[g, str(d["n"]), metrics.pct(d["value"])]
                for g, d in sorted(s["by_align"].items())]
        lines.append(metrics.table(["块类型", "块数", "行数完全一致"], rows))
        if (s["exact"]["value"] or 0) < 0.5:
            lines += [
                "",
                "> ⚠️ **结果解读（别把这个数当产品失败，也别当成功）**："
                "`split_page` 的默认参数是为整页作业照调的，而 M²E 是 70~300 px 的紧凑公式块——"
                "行距极小、分数上下标跨行、竖式两列、还带横线本线条；对小图做纠偏旋转会把相邻行糊成一行。"
                "这个数字说明**切行模块需要小图专用预处理**（放大后再切 / 关闭纠偏 / 更小的行间合并阈值 / "
                "投影谷底递归切单行带），不能外推成产品对整页作业的准确率。"
                "产品主场景的指标仍以自采 P0 八页为准。",
            ]
        recs0 = report.get("records") or []
        collapsed = [r for r in recs0 if r["n_pred"] == 1 and r["n_true"] >= 3]
        if len(collapsed) >= max(5, int(0.05 * len(recs0))):
            lines += [
                "",
                f"> ⚠️ **最集中的单一失败**：`{len(collapsed)}` 块真值 ≥3 行却只切出 1 行"
                f"（占 {metrics.pct(len(collapsed) / len(recs0))}）。"
                "机制（读代码可复现，不是猜的）：这些小图行距只有 2~3 px，"
                "`min_gap = max(4, ...)` 把相邻行合并成一条带；"
                "条带过滤后只剩这一条，`split_tall` 再拿它自己的高度当参照"
                "（`h <= 1.6 * median_h` 就不切），**单带场景永远不会触发切分**。"
                "这是算法缺陷，不是阈值没调好——调参救不了它，要改的是「一条带怎么估行高」。",
            ]
    recs = report.get("records") or []
    if recs:
        fp = report.get("inputs") or {}
        lines += [
            "", "## 逐块记录",
            "",
            f"- 全量 `{len(recs)}` 条已落盘到同名 JSON 的 `records`，字段："
            "`id` / `n_true` / `n_pred` / `delta` / `exact` / `within1` / "
            "`group` / `align_group` / `size` / `note`。任一指标都能按这堆记录重算。",
            f"- seed=`{report.get('seed')}`、bootstrap 重采样 `{report.get('n_boot')}` 次；"
            "同一份输入重跑，records 与区间都应逐位相同。",
            f"- 输入指纹：标注 `sha256:{str(fp.get('blocks_sha256', ''))[:16]}`"
            f"（本次载入 {fp.get('n_loaded', '?')} 块）、"
            f"标注内容 `sha256:{str(fp.get('labels_sha256', ''))[:16]}`、"
            f"参数 `sha256:{str(fp.get('params_sha256', ''))[:16]}`。"
            "两份报告数字不一致时，先比这三个指纹：参数变了、标注换了，都不算回归。",
            "",
            "### 偏差分布（预测行数 − 标注行数）",
            "",
            metrics.table(["偏差", "块数"], delta_histogram(recs)),
            "",
            "### 偏差最大的 15 块（优先目检）",
            "",
        ]
        rows = []
        for r in worst_records(recs):
            note = str(r.get("note") or "")
            rows.append([r["id"], str(r["n_true"]), str(r["n_pred"]),
                         f"{r['delta']:+d}", r.get("align_group", ""),
                         note[:28] + ("…" if len(note) > 28 else "")])
        lines.append(metrics.table(
            ["块 ID", "真值", "预测", "偏差", "类型", "切图备注"], rows))
        lines += [
            "",
            "> 复核单块：`--debug N` 只导前 N 张，按 ID 定位要自己在 "
            "`data/public/m2e/blocks.jsonl` 里找到同名记录，再对该图单独跑一次 "
            "`split_page`；JSON 里每条都带 `size`，可以先按短边判断它是走了整页模式"
            "还是块模式。",
        ]
    lines += ["", "## 切行参数（必须随数字一起报）", "",
              "```json", json.dumps(p, ensure_ascii=False, indent=2), "```", ""]
    if report.get("tune"):
        t = report["tune"]
        lines += ["## 调参记录（只在 train 上做）", "",
                  f"候选 {t['n_candidates']} 组 / 调参样本 {t['n_tune']} 块；"
                  f"最优 exact = {metrics.pct(max((r['exact'] or 0) for r in t['table']))}。",
                  "完整表在同名 JSON 的 `tune.table` 里。", ""]
    lines += [
        "## 复现",
        "",
        "```bash",
        report["reproduce"],
        "```",
        "",
        "## 怎么读这个数",
        "",
        "- `exact` 低 + `bias` 为负 → 行间太密被合成一行，`split_tall` 没切开；"
        "优先看 `5 行及以上` 那一层。",
        "- `exact` 低 + `bias` 为正 → 墨点/分数/上下标被当独立行；"
        "优先看 `1-2 行` 那一层。",
        "- 想确认切的位置对不对，重跑时加 `--debug 20`，看 `reports/line_debug/` 里画框的图。",
        "- 这个数只给「拍照批改」的**切行模块**做外部效度证据，"
        "**不替代** `docs/手写数据采集清单.md` 里 P0 八页的伪推导 / 拍摄退化真值。",
        "",
        "出处：M²E / LAST, ACM MM 2023 · https://www.modelscope.cn/datasets/Wente47/M2E",
        "",
    ]
    return "\n".join(lines)


def dump_report(report: dict, path: Path) -> None:
    """写报告 JSON：头部缩进好看，records 一行一条。

    2000 条逐块记录如果也按 indent=2 展开，文件会膨胀到 600 KB 且 diff 全是噪声；
    一行一条既省体积，又能让 `git diff` 精确指到"哪一块的 n_pred 变了"——
    这正是落 records 的初衷。
    """
    head = {k: v for k, v in report.items() if k != "records"}
    text = json.dumps(head, ensure_ascii=False, indent=2).rstrip()
    recs = report.get("records") or []
    cut = text.rfind("}")
    body = text[:cut].rstrip()
    if recs:
        lines = ",\n".join("    " + json.dumps(r, ensure_ascii=False, sort_keys=True)
                           for r in recs)
        text = f'{body},\n  "records": [\n{lines}\n  ]\n}}\n'
    else:
        text = f'{body},\n  "records": []\n}}\n'
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- CLI


def main() -> int:
    ap = argparse.ArgumentParser(description="M²E 真实版式 → 切行召回率（外部验证集）")
    ap.add_argument("--blocks", default=str(DEFAULT_BLOCKS))
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="最多评测多少块（0=全部）")
    ap.add_argument("--params", help="参数 JSON 文件（一般是 reports/line_params.json）")
    ap.add_argument("--tune", action="store_true", help="在 train 上扫参数网格")
    ap.add_argument("--tune-limit", type=int, default=300, help="调参用多少块")
    ap.add_argument("--exclude-align", action="store_true",
                    help="排除含 \\t 的竖式对齐块（默认纳入并分层报告）")
    ap.add_argument("--debug", type=int, default=0, help="导出前 N 张画框图")
    ap.add_argument("--n-boot", type=int, default=metrics.N_BOOT)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="写 _smoke_ 前缀，不污染正式报告")
    ap.add_argument("--tag", default="", help="报告名后缀，如 baseline / tuned，便于多次对比")
    args = ap.parse_args()

    if args.tune and args.split != "train":
        raise SystemExit(
            "参数只允许在 train 上调：先 `--split train --tune`，"
            "再 `--split test --params reports/line_params.json`。"
            "在 test 上边调边报是自欺，脚本不允许。"
        )

    params = dict(DEFAULT_PARAMS)
    source = "默认参数（split_page 默认值，未调参）"
    if args.params:
        pf = Path(args.params)
        payload = json.loads(pf.read_text(encoding="utf-8"))
        params.update(payload.get("best", payload))
        source = str(pf)

    blocks, n_total = load_blocks(Path(args.blocks), args.split, args.limit)
    images, collisions = index_images(Path(args.images))
    if not images:
        print(f"警告：{args.images} 里没有图片。先跑 tools/fetch_m2e.py unzip。"
              "本次只产出样本统计，不伪造指标。")

    skip_align = args.exclude_align
    tuned = None
    if args.tune:
        tuned = tune(blocks, images, params, tune_limit=args.tune_limit,
                     skip_align=skip_align)
        params = tuned["best"]
        source = "train 网格调参（见 tune.table）"
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "line_params.json").write_text(
            json.dumps(tuned, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"参数已写入 {REPORTS / 'line_params.json'}；"
              f"train 上最优 exact = {metrics.pct(max((r['exact'] or 0) for r in tuned['table']))}")
        print("下一步在未碰过的 test 上重跑："
              "eval/run_line_eval.py --split test --params reports/line_params.json")

    debug_dir = REPORTS / "line_debug" if args.debug else None
    got = eval_blocks(blocks, images, params, skip_align=skip_align,
                      debug_dir=debug_dir, debug_n=args.debug)
    # 显式落一个种子：默认值在 metrics 里，报告中写"None"等于没写，复现时对不上。
    seed = args.seed if args.seed is not None else metrics.DEFAULT_SEED
    s = score(got["records"], seed, args.n_boot)
    fp = input_fingerprint(Path(args.blocks), blocks, params, args.split)
    fp["limit"] = args.limit

    from datetime import datetime
    report = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "M²E (Wente47/M2E, ModelScope)",
        "license": "CC BY-NC 4.0",
        "split": args.split, "params": params, "params_source": source,
        "n_blocks": n_total, "n_loaded": len(blocks),
        "skipped_align": got["skipped_align"],
        "missing": got["missing"], "errors": got["errors"],
        "image_name_collisions": collisions, "caveats": CAVEATS,
        "seed": seed, "n_boot": args.n_boot, "inputs": fp,
        "metrics": s, "records": got["records"], "tune": tuned,
        "reproduce": (
            f"F:/PyCharm/python.exe eval/run_line_eval.py --split {args.split}"
            + (f" --params {args.params}" if args.params else "")
            + (f" --limit {args.limit}" if args.limit else "")
            + (f" --seed {args.seed}" if args.seed is not None else "")
            + (f" --n-boot {args.n_boot}" if args.n_boot != metrics.N_BOOT else "")
        ),
    }
    tag = f"_{args.tag}" if args.tag else ""
    prefix = ("_smoke_" if args.smoke else "") + "eval_lines" + tag
    REPORTS.mkdir(parents=True, exist_ok=True)
    dump_report(report, REPORTS / f"{prefix}.json")
    (REPORTS / f"{prefix}.md").write_text(render_md(report), encoding="utf-8")
    print(f"评测 {s.get('n', 0)} 块 → {REPORTS / (prefix + '.md')}")
    if s.get("n"):
        print(f"行数完全一致 {metrics.pct(s['exact']['value'], s['exact']['ci'])} · "
              f"MAE {s['mae']:.3f} · bias {s['bias']:+.3f}")
    else:
        # 没有可评测的块 ≠ 评测通过。退出码 2 让脚本/CI 没法把空跑当成功。
        print("状态：未执行（没有可评测的块，报告已写明未产出指标，不当作成功评测）")
        return 2
    if got["missing"]:
        print(f"有 {len(got['missing'])} 块找不到图片，已从指标中剔除并在报告披露。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
