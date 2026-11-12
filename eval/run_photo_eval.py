"""真实手写过程页 → 拍照批改 → 对照人工真值 → 出报告。

和 eval/run_eval.py 的分工：那边吃构造文本/假识别器，验证代码逻辑；
这边吃 data/inbox/W*.jpg，回答"真实手写上到底行不行"。真值在
data/work_shotlist.csv，按数学规范标，不是从引擎输出反推的。

三个阶段分开跑，照片没到位也能验证后半段：

    python -m eval.run_photo_eval --collect    # 调 VLM 采集，写 data/photo_results.jsonl
    python -m eval.run_photo_eval              # 读结果，出 reports/eval_photo_real.{json,md}
    python -m eval.run_photo_eval --smoke      # 内置假结果跑通报告，离线、不打网络
    python -m eval.run_photo_eval --results reports/golden_photo_results.jsonl

--collect 会逐个调两个 VLM、按页花钱；重跑前想清楚。
缺照片只报缺哪些，不算失败。
**没有可用结果快照时退出码是 2（未执行），不是 0**：空跑被当成"评测通过"是这份
报告以前最容易误导人的地方。原图不入库，所以 `--results` 允许对一份入库的脱敏
快照复算指标；但复算只证明"当时的判定是什么"，不等于重跑了一遍识别。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import metrics as M  # noqa: E402

DATA = ROOT / "data"
INBOX = DATA / "inbox"
SHOTLIST = DATA / "work_shotlist.csv"
RESULTS = DATA / "photo_results.jsonl"
OUT_DIR = ROOT / "reports"
EXTS = (".jpg", ".jpeg", ".png")

GRADES = ["pass", "fake", "partial", "wrong", "unknown", "gap", "unread"]
# 判"有错"的档位。unread 是"判不了"，不是"判错"。
BAD = ("wrong", "fake", "partial")


def truth_rows() -> list[dict]:
    with SHOTLIST.open("r", encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f)
                if (r.get("id") or "").strip() and (r.get("truth_grade") or "").strip() != "na"]


def find_image(wid: str) -> Path | None:
    for ext in EXTS:
        p = INBOX / f"{wid}{ext}"
        if p.exists():
            return p
    return None


# ------------------------------------------------------------------ 采集


def collect(rows: list[dict], only: set[str] | None = None) -> int:
    from mathagent.photocheck import check  # noqa: PLC0415  只在采集时才依赖 VLM

    todo = [r for r in rows if only is None or r["id"].strip() in only]
    out: list[dict] = []
    for i, r in enumerate(todo, 1):
        wid = r["id"].strip()
        img = find_image(wid)
        if img is None:
            print(f"[{i}/{len(todo)}] {wid} 缺照片，跳过")
            continue
        t0 = time.time()
        try:
            pc = check(str(img), problem=(r.get("problem") or "").strip())
            w = pc.work
            rec = {
                "id": wid,
                "group": r.get("group", ""),
                "priority": r.get("priority", ""),
                "image": str(img.relative_to(ROOT)),
                "truth_grade": r.get("truth_grade", ""),
                "truth_first": int(r.get("truth_first") or -1),
                "truth_type": r.get("truth_type", ""),
                "baseline": r.get("baseline", ""),
                "pred_grade": w.grade,
                "pred_first": w.first_error,
                "pred_type": w.error_type or "",
                "n_lines": len(pc.lines),
                "n_unsure": sum(1 for x in pc.lines if not x.sure),
                "n_unread": pc.n_unread,
                "text": pc.text,
                "lines": [
                    {"index": x.index, "latex": x.latex, "verdict": x.verdict,
                     "sure": bool(x.sure), "confidence": x.confidence}
                    for x in pc.lines
                ],
                # 撤回机制的核心证据：认不准的行里还有没有硬断言
                "false_certainty": any(
                    s.verdict in ("wrong", "partial") and bool(s.counter)
                    for s in w.steps
                ) and w.grade == "unread",
                "notes": pc.notes,
                "elapsed": round(time.time() - t0, 2),
            }
            print(f"[{i}/{len(todo)}] {wid} 真 {rec['truth_grade']:>8s} "
                  f"测 {rec['pred_grade']:>8s} · {rec['n_unsure']} 行没认准 "
                  f"· {rec['elapsed']}s")
        except Exception as exc:  # noqa: BLE001
            rec = {
                "id": wid, "group": r.get("group", ""), "priority": r.get("priority", ""),
                "image": str(img.relative_to(ROOT)),
                "truth_grade": r.get("truth_grade", ""),
                "truth_first": int(r.get("truth_first") or -1),
                "truth_type": r.get("truth_type", ""), "baseline": r.get("baseline", ""),
                "pred_grade": "error", "pred_first": -1, "pred_type": "",
                "n_lines": 0, "n_unsure": 0, "n_unread": 0, "text": "",
                "lines": [], "false_certainty": False,
                "notes": [f"{type(exc).__name__}: {exc}"],
                "elapsed": round(time.time() - t0, 2),
            }
            print(f"[{i}/{len(todo)}] {wid} 崩了：{type(exc).__name__}: {exc}")
        out.append(rec)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", encoding="utf-8", newline="\n") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"写入 {RESULTS.relative_to(ROOT)}：{len(out)} 页")
    return 0


def _shown_path(p: Path) -> str:
    """报告里写仓库内相对路径：绝对路径换个机器就读不懂，也会暴露本地目录结构。"""
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return str(p)


def load_results(path: Path | None = None) -> list[dict]:
    """读结果快照。默认 `data/photo_results.jsonl`，也可以显式指定。

    `--results` 存在的理由：原图不在仓库里（隐私），但结果快照可以脱敏后入库。
    评审要复算 `reports/eval_photo_real.md` 时，应该能对它指定的那份快照跑，
    而不是被迫读 `data/` 下某个"当次运行留下的"文件——那个文件明天就被覆盖了。
    支持 `.jsonl`（每行一条）与 `.json`（本脚本自己写出的 `{"records": [...]}`）。
    """
    p = path or RESULTS
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
        recs = payload.get("records") if isinstance(payload, dict) else payload
        return [r for r in (recs or []) if isinstance(r, dict)]
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ------------------------------------------------------------------ 指标


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if r.get("pred_grade") != "error"]
    for r in ok:
        r["grade_hit"] = int(r["truth_grade"] == r["pred_grade"])
        r["false_accused"] = int(r["truth_grade"] in ("pass", "gap")
                                 and r["pred_grade"] in BAD)
        r["fake_hit"] = int(r["truth_grade"] == "fake" and r["pred_grade"] == "fake")
        r["withdraw_hit"] = int(r["truth_grade"] == "unread"
                                and r["pred_grade"] == "unread")
        r["baseline_diff"] = int(r["pred_grade"] != r["baseline"])

    clean = [r for r in ok if r["truth_grade"] in ("pass", "gap")]
    fakes = [r for r in ok if r["truth_grade"] == "fake"]
    unread = [r for r in ok if r["truth_grade"] == "unread"]

    mism = [
        {"id": r["id"], "group": r["group"], "truth": r["truth_grade"],
         "pred": r["pred_grade"], "baseline": r["baseline"],
         "n_unsure": r["n_unsure"], "text": r.get("text", "")[:160]}
        for r in ok if not r["grade_hit"]
    ]
    return {
        "n": len(ok),
        "n_error": len(records) - len(ok),
        "conclusion": M.metric([r["grade_hit"] for r in ok]),
        "false_accusation": M.metric([r["false_accused"] for r in clean]),
        "fake_recall": M.metric([r["fake_hit"] for r in fakes]),
        "withdraw": M.metric([r["withdraw_hit"] for r in unread]),
        "no_false_certainty": M.metric(
            [int(not r["false_certainty"]) for r in unread]
        ),
        "baseline_diff": M.metric([r["baseline_diff"] for r in ok]),
        "confusion": M.confusion(
            [(r["truth_grade"], r["pred_grade"]) for r in ok], GRADES
        ),
        "by_group": M.by_group(ok, "grade_hit"),
        "by_priority": M.by_group(ok, "grade_hit", "priority"),
        "mismatches": mism,
    }


def render(s: dict, records: list[dict], smoke: bool, source: str = "",
           source_how: str = "") -> str:
    add: list[str] = []
    add.append("# 真实手写过程页评测报告")
    add.append("")
    if smoke:
        add.append("> **冒烟运行**：结果是内置的假数据，只验证报告管线能否跑通，"
                   "**不是实测数字，不得引用**。")
        add.append("")
    if source:
        how = f"（{source_how}）" if source_how else ""
        add.append(f"> 结果快照：`{source}`{how}。"
                   "原图不在仓库里，这份报告只保证**指标可复算**——"
                   "它能证明当时这批页的判定是什么，不能替代重跑识别。")
        add.append("")
    add.append(f"样本 {s['n']} 页" + (f"，{s['n_error']} 页采集中崩了" if s["n_error"] else "")
               + f"。真值来自 `data/work_shotlist.csv`，按数学规范标注。")
    add.append("")
    add.append("## 1. 主指标")
    add.append("")
    rows = [
        ["结论准确率 ↑", M.pct(s["conclusion"]["value"], s["conclusion"]["ci"]),
         f"{s['conclusion']['hit']}/{s['conclusion']['n']}", "总判定与人工真值一致"],
        ["冤枉率 ↓", M.pct(s["false_accusation"]["value"], s["false_accusation"]["ci"]),
         f"{s['false_accusation']['hit']}/{s['false_accusation']['n']}",
         "本来对的过程被判有错；这项比准确率重要"],
        ["伪推导召回 ↑", M.pct(s["fake_recall"]["value"], s["fake_recall"]["ci"]),
         f"{s['fake_recall']['hit']}/{s['fake_recall']['n']}",
         "过程错、答案对的那类，作业帮抓不到"],
        ["撤回正确率 ↑", M.pct(s["withdraw"]["value"], s["withdraw"]["ci"]),
         f"{s['withdraw']['hit']}/{s['withdraw']['n']}",
         "有认不准的行时是否转成 unread 而不是继续判错"],
        ["无假确定 ↑", M.pct(s["no_false_certainty"]["value"]),
         f"{s['no_false_certainty']['hit']}/{s['no_false_certainty']['n']}",
         "unread 页里有没有还带反例的硬断言"],
        ["与文字基线不同 ↑", M.pct(s["baseline_diff"]["value"]),
         f"{s['baseline_diff']['hit']}/{s['baseline_diff']['n']}",
         "识别引入的差异；不是错误，用来定位差异来源"],
    ]
    add.append(M.table(["指标", "数值 [95% CI]", "命中/样本", "口径"], rows))
    add.append("")
    add.append("比例用 bootstrap 百分位法给区间（技术文档 §6.3）。页数通常只有十几，"
               "区间会极宽，点估计只能看趋势。**不要只报一个数。**")
    add.append("")
    add.append("## 2. 分层")
    add.append("")
    add.append(M.table(
        ["组", "结论命中", "准确率"],
        [[g, f"{v['hit']}/{v['n']}", M.pct(v["value"])]
         for g, v in s["by_group"].items()],
    ))
    add.append("")
    add.append("## 3. 结论不一致")
    add.append("")
    if s["mismatches"]:
        add.append(M.table(
            ["ID", "组", "真值", "预测", "文字基线", "没认准行数", "识别出的过程"],
            [[m["id"], m["group"], m["truth"], m["pred"], m["baseline"],
              m["n_unsure"], m["text"].replace("\n", " / ")] for m in s["mismatches"]],
        ))
    else:
        add.append("无。")
    add.append("")
    add.append("不一致要看两件事：**是识别读错了字，还是判据本身有问题**。"
               "把 `text`（系统读出来的过程）和 `work_shotlist.csv` 的 `lines` 并排看就知道。"
               "属于前者的记进识别误差，属于后者的才是批改引擎缺陷。")
    add.append("")
    add.append("## 4. 混淆矩阵")
    add.append("")
    labels = s["confusion"]["labels"]
    add.append(M.table(["真值 \\ 预测"] + labels,
                       [[a] + s["confusion"]["rows"][a] for a in labels]))
    add.append("")
    add.append("最该盯的两个格子：`pass` 行落到 `fake`/`wrong` 列（冤枉），"
               "以及 `fake` 行落到 `pass` 列（漏抓了最值钱的那类错误）。")
    add.append("")
    add.append("## 5. 边界")
    add.append("")
    add.append("- 真值清单与批改引擎出自同一人，**有共同盲区**；这批页只能说明"
               "'这些手写页上没出现剩余缺陷'，不能外推到所有手写分布。")
    add.append("- 每页只有一道题、按清单照抄，**不是自然书写的作业照**。"
               "要更接近真实，应另采一批不看清单自由作答的页。")
    add.append("- `W20`（一页两题）真值为 `na`，不进指标；它只用来记录已知局限。")
    add.append("- 采集会调两个外部 VLM，结果受模型版本与网络影响；"
               "`data/photo_results.jsonl` 是当次快照，重跑前请另存。")
    add.append("")
    return "\n".join(add) + "\n"


def smoke_records() -> list[dict]:
    """离线假结果，只为验证报告渲染。数值一律不可引用。"""
    spec = [
        ("W01", "正确过程", "P0", "pass", "pass", 0),
        ("W05", "伪推导", "P0", "fake", "fake", 0),
        ("W08", "明显错", "P0", "wrong", "wrong", 0),
        ("W12", "漏解", "P0", "partial", "partial", 0),
        ("W16", "拍摄退化", "P0", "unread", "unread", 1),
        ("W17", "拍摄退化", "P0", "unread", "unread", 1),
    ]
    return [
        {"id": i, "group": g, "priority": p, "image": f"(smoke){i}.jpg",
         "truth_grade": t, "truth_first": -1, "truth_type": "", "baseline": b,
         "pred_grade": b, "pred_first": -1, "pred_type": "", "n_lines": 3,
         "n_unsure": u, "n_unread": u, "text": "(冒烟数据)", "lines": [],
         "false_certainty": False, "notes": [], "elapsed": 0.0}
        for i, g, p, t, b, u in spec
    ]


# ------------------------------------------------------------------ 主流程


def main() -> int:
    ap = argparse.ArgumentParser(description="真实手写过程页评测")
    ap.add_argument("--collect", action="store_true", help="调 VLM 采集结果（花钱）")
    ap.add_argument("--only", default="", help="只采这几页，逗号分隔，如 W01,W05")
    ap.add_argument("--smoke", action="store_true", help="离线假结果跑通报告")
    ap.add_argument("--results", default="",
                    help="显式指定结果快照（.jsonl 或 .json）；默认 data/photo_results.jsonl")
    args = ap.parse_args()

    rows = truth_rows()
    if args.collect:
        only = {x.strip() for x in args.only.split(",") if x.strip()} or None
        return collect(rows, only)

    source = ""
    if args.smoke:
        records = smoke_records()
    else:
        snap = Path(args.results) if args.results else None
        if snap is not None and not snap.exists():
            print(f"状态：未执行（--results 指向的 {snap} 不存在）")
            return 2
        records = load_results(snap)
        source = _shown_path(snap or RESULTS)
        missing = [r["id"].strip() for r in rows if find_image(r["id"].strip()) is None]
        if not records:
            # 空跑不是成功：以前这里 return 0，日志看着像"评测通过"，其实一个字都没测。
            print("状态：未执行（没有可用的结果快照，本轮不产出任何指标）")
            print("先跑：")
            print("  F:/PyCharm/python.exe -m eval.run_photo_eval --collect")
            print(f"当前缺照片 {len(missing)}/{len(rows)} 页：{'、'.join(missing[:12])}"
                  + (" …" if len(missing) > 12 else ""))
            print("或者用 --results 指向一份已有的快照（.jsonl / .json）。")
            return 2
        if missing:
            print(f"注意：还有 {len(missing)} 页没拍，报告只覆盖已采的 {len(records)} 页。")

    s = summarize(records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 冒烟的假结果绝不写进正式报告路径：评审误打开会当成实测数字引用。
    stem = "_smoke_photo_real" if args.smoke else "eval_photo_real"
    payload = {"smoke": bool(args.smoke), "status": "ok",
               "results_source": source if not args.smoke else "(内置冒烟数据)",
               "summary": s, "records": records}
    (OUT_DIR / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = render(s, records, args.smoke,
                "" if args.smoke else source,
                "" if args.smoke else ("--results 显式指定" if snap else "默认快照"))
    (OUT_DIR / f"{stem}.md").write_text(md, encoding="utf-8")

    print(f"结论准确率 {M.pct(s['conclusion']['value'], s['conclusion']['ci'])} "
          f"({s['conclusion']['hit']}/{s['conclusion']['n']})")
    print(f"冤枉率     {M.pct(s['false_accusation']['value'])}")
    print(f"伪推导召回 {M.pct(s['fake_recall']['value'])}")
    print(f"撤回正确率 {M.pct(s['withdraw']['value'])}")
    print(f"不一致     {len(s['mismatches'])} 条")
    print(f"报告       {OUT_DIR / f'{stem}.md'}"
          + ("（冒烟，不可引用）" if args.smoke else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
