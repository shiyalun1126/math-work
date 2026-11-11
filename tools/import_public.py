"""把公开手写数学数据集导成本项目能吃的格式。

只做离线转换，不联网、不改原始包。下载命令见 docs/公开数据集调研.md §5。
输出统一放在 data/public/<dataset>/，与自采数据（data/labels.csv、data/inbox/）分开，
免得把"公开数据"和"自己拍的"混成一个评测集。

三个子命令：

  probe         检查下载到的目录里有什么，不写文件
  mathwriting   InkML 笔迹 → PNG + labels.csv（数据集 A：单行公式）
  m2e           多行 jsonl → blocks.jsonl / lines.csv（数据集 B：切行压力测试）
  scratchmath   parquet → 图片 + metadata.jsonl（错因诊断，只做参考不进主指标）

典型用法：

  F:/PyCharm/python.exe tools/import_public.py probe mathwriting --src D:/dl/mw
  F:/PyCharm/python.exe tools/import_public.py mathwriting --src D:/dl/mw --limit 200
  F:/PyCharm/python.exe tools/import_public.py m2e --src D:/dl/M2E --limit 500
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
PUBLIC = DATA / "public"
INKML_NS = "{http://www.w3.org/2003/InkML}"

# 数据集各自的许可与出处，写进 manifest，免得半年后忘了这堆图是哪来的
SOURCES = {
    "mathwriting": {
        "name": "MathWriting (Google LLC)",
        "url": "https://github.com/google-research/google-research/tree/master/mathwriting",
        "license": "CC BY-NC-SA 4.0（其中的 LaTeX 表达式另受 Wikipedia CC BY-SA 4.0 约束）",
        "note": "在线笔迹 InkML，非纸笔拍照；真值用 normalizedLabel 字段",
    },
    "m2e": {
        "name": "M²E 多行手写数学公式 (Wente47 / ModelScope)",
        "url": "https://www.modelscope.cn/datasets/Wente47/M2E",
        "license": "CC BY-NC 4.0",
        "note": "真实试卷手机拍摄；tex 字段用 \\n 分行，含 \\t 对齐列",
    },
    "scratchmath": {
        "name": "ScratchMath (AIED 2026)",
        "url": "https://huggingface.co/datasets/songdj/ScratchMath",
        "license": "CC BY-NC-SA 4.0",
        "note": "真实学生草稿图 + 7 类错因；没有逐行 LaTeX 转写",
    },
}


# --------------------------------------------------------------------------- 通用


def _maybe_extract(src: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """允许直接把 .tgz / .zip 交给脚本，省得手动解包。"""
    if src.is_dir():
        return src, None
    tmp = tempfile.TemporaryDirectory(prefix="import_public_")
    out = Path(tmp.name)
    if src.suffix in (".tgz", ".gz") or src.name.endswith(".tar.gz"):
        with tarfile.open(src, "r:*") as tf:
            tf.extractall(out, filter="data")
    elif src.suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(src) as zf:
            zf.extractall(out)
    else:
        tmp.cleanup()
        raise SystemExit(f"不认识的文件：{src}（要目录 / .tgz / .zip）")
    subs = [p for p in out.iterdir() if p.is_dir()]
    return (subs[0] if len(subs) == 1 else out), tmp


def _find(src: Path, filename: str) -> Path | None:
    """包里可能多套一层目录，往上/往下找一层。"""
    for cand in (src / filename, *list(src.glob(f"*/{filename}"))):
        if cand.exists():
            return cand
    return None


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _manifest(out: Path, key: str, extra: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(SOURCES[key])
    payload.update(extra or {})
    (out / "SOURCE.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _cas_ok(latex: str) -> bool:
    """这条真值能不能被本项目的 CAS 解析——决定它能不能当求解类评测的真值。"""
    try:
        from mathagent.latex import parse

        return bool(parse(latex).ok)
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- InkML


def read_inkml(path: Path, fields: str = "normalizedLabel,label"
               ) -> tuple[str, str, list[list[tuple[float, float]]]]:
    """返回 (真值 LaTeX, sampleId, 笔画坐标)。

    MathWriting 用 normalizedLabel；CROHME 用 truth（且常带 $...$ 外壳），
    所以字段名做成参数，两边共用一份渲染代码。
    """
    root = ET.parse(path).getroot()
    ann: dict[str, str] = {}
    for a in root.findall(f"{INKML_NS}annotation"):
        # ElementTree 的 elem["x"] 是按整数取子节点，属性要走 .get()
        if a.get("type"):
            ann[a.get("type")] = (a.text or "").strip()
    label = ""
    for key in fields.split(","):
        label = (ann.get(key.strip()) or "").strip().strip("$").strip()
        if label:
            break
    strokes: list[list[tuple[float, float]]] = []
    for tr in root.iter(f"{INKML_NS}trace"):
        pts: list[tuple[float, float]] = []
        for chunk in (tr.text or "").split(","):
            nums = chunk.split()
            if len(nums) >= 2:
                try:
                    pts.append((float(nums[0]), float(nums[1])))
                except ValueError:
                    continue
        if pts:
            strokes.append(pts)
    return label, ann.get("sampleId", path.stem), strokes


def render_ink(strokes: list[list[tuple[float, float]]], height: int = 96,
               pen: float = 1.5):
    """把在线笔迹画成位图。坐标没归一化，必须按每条自己的外接框缩放。"""
    from PIL import Image, ImageDraw

    pts = [p for s in strokes for p in s]
    if not pts:
        raise ValueError("空笔迹")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    span_y = max(y1 - y0, 1.0)
    span_x = max(x1 - x0, 1.0)
    scale = (height - 16) / span_y
    pad = 8
    w = min(int(span_x * scale + 2 * pad), 4000)
    img = Image.new("L", (max(w, 16), height), 255)
    d = ImageDraw.Draw(img)
    width = max(2, int(round(2 * pen * scale)))

    def xy(p):
        return (int(round((p[0] - x0) * scale + pad)),
                int(round((p[1] - y0) * scale + pad)))

    for s in strokes:
        seq = [xy(p) for p in s]
        if len(seq) == 1:
            x, y = seq[0]
            d.ellipse((x - width // 2, y - width // 2,
                       x + width // 2, y + width // 2), fill=0)
        else:
            d.line(seq, fill=0, width=width, joint="curve")
            r = width / 2
            for x, y in (seq[0], seq[-1]):
                d.ellipse((x - r, y - r, x + r, y + r), fill=0)
    return img.convert("RGB")


# --------------------------------------------------------------------------- 子命令


def cmd_probe(args) -> int:
    src, tmp = _maybe_extract(Path(args.src))
    key = args.dataset
    print(f"目录 {src}")
    if key == "mathwriting":
        for split in ("train", "valid", "test", "symbols", "synthetic"):
            d = _find(src, split)
            n = len(list(d.glob("*.inkml"))) if d and d.is_dir() else 0
            print(f"  {split:10s} {n} 条 InkML")
        sample = next(iter((_find(src, "train") or src).glob("*.inkml")), None)
        if sample:
            label, sid, strokes = read_inkml(sample)
            print(f"  样例 {sid}: {label[:80]!r} | {len(strokes)} 笔")
    elif key == "m2e":
        for split in ("train", "val", "test"):
            f = _find(src, f"{split}.jsonl")
            if not f:
                print(f"  {split}.jsonl 缺失")
                continue
            lines = f.read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0]) if lines else {}
            print(f"  {split}.jsonl {len(lines)} 行 | 字段 {list(first)}")
            if first:
                print(f"    样例 tex: {str(first.get('tex',''))[:110]!r}")
        z = _find(src, "images.zip")
        print(f"  images.zip {'有' if z else '无'}（图片要单独解压）")
    elif key == "scratchmath":
        for p in sorted(src.rglob("*.parquet")):
            print(f"  {p.relative_to(src)} | {p.stat().st_size / 1e6:.1f} MB")
    if tmp:
        tmp.cleanup()
    return 0


def cmd_mathwriting(args) -> int:
    src, tmp = _maybe_extract(Path(args.src))
    out = Path(args.out) if args.out else PUBLIC / "mathwriting"
    splits = args.splits.split(",")
    rows: list[list] = []
    stats: dict[str, int] = {}
    for split in splits:
        d = _find(src, split)
        if not d or not d.is_dir():
            print(f"跳过 {split}：目录不存在")
            continue
        files = sorted(d.glob("*.inkml"))[: args.limit] if args.limit else sorted(d.glob("*.inkml"))
        img_dir = out / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        n_ok = 0
        for f in files:
            label, sid, strokes = read_inkml(f, args.label_field)
            if not label or not strokes:
                continue
            try:
                render_ink(strokes, height=args.height, pen=args.pen).save(
                    img_dir / f"{sid}.png"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  渲染失败 {sid}: {exc}")
                continue
            rel = f"images/{split}/{sid}.png"
            cas = "1" if args.check_parse and _cas_ok(label) else ""
            rows.append([
                rel, label,
                "self_train" if split == "train" else "self_test",
                f"公开-MathWriting-{split}",
                f"sampleId={sid}" + (f"; cas_ok={cas}" if cas else ""),
            ])
            n_ok += 1
        stats[split] = n_ok
        print(f"  {split}: 导出 {n_ok} 张 → {img_dir}")
    _write_csv(out / "labels.csv", ["file", "latex", "split", "type", "note"], rows)
    _manifest(out, "mathwriting", {"exported": stats, "check_parse": args.check_parse})
    print(f"共 {len(rows)} 条 → {out / 'labels.csv'}")
    if args.check_parse:
        ok = sum(1 for r in rows if "cas_ok=1" in r[4])
        print(f"其中 {ok}/{len(rows)} 条真值能被本项目 CAS 解析"
              "（其余只能做识别评测，不能做求解评测）")
    if tmp:
        tmp.cleanup()
    return 0


def _despace(tex: str) -> str:
    """M²E 把 '15.4' 写成 '1 5 . 4'；拼回数字串，但别动 \\times 这类命令。

    只合并两侧都是数字/小数点的空白，反复几轮才能把 '1 5 . 4' 全拼上。
    这是有损的启发式，所以默认关掉，导出的 blocks.jsonl 里永远留着 tex_raw。
    """
    for _ in range(4):
        new = re.sub(r"(?<=[\d.])\s+(?=[\d.])", "", tex)
        if new == tex:
            break
        tex = new
    return tex


def cmd_m2e(args) -> int:
    src, tmp = _maybe_extract(Path(args.src))
    out = Path(args.out) if args.out else PUBLIC / "m2e"
    blocks: list[dict] = []
    line_rows: list[list] = []
    for split, fname in (("train", "train.jsonl"), ("val", "val.jsonl"),
                         ("test", "test.jsonl")):
        f = _find(src, fname)
        if not f:
            print(f"跳过 {fname}：不存在")
            continue
        raw = f.read_text(encoding="utf-8").splitlines()
        if args.limit:
            raw = raw[: args.limit]
        for i, ln in enumerate(raw):
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            tex = str(rec.get("tex", ""))
            parts = [p.strip() for p in re.split(r"\\n|\n", tex) if p.strip()]
            if args.despace:
                parts = [_despace(p) for p in parts]
            has_align = any("\\t" in p for p in parts)
            bid = f"{split}_{Path(str(rec.get('name', i))).stem}"
            blocks.append({
                "id": bid, "split": split, "image": rec.get("name", ""),
                "tex_raw": tex[:600], "lines": parts, "n_lines": len(parts),
                "has_align": has_align,
            })
            for j, p in enumerate(parts, 1):
                line_rows.append([bid, j, p])
        print(f"  {fname}: 累计 {len(blocks)} 个多行块")
    _write_jsonl(out / "blocks.jsonl", blocks)
    _write_csv(out / "lines.csv", ["block_id", "line_no", "latex"], line_rows)
    dist: dict[str, int] = {}
    for b in blocks:
        k = str(min(b["n_lines"], 10))
        dist[k] = dist.get(k, 0) + 1
    _manifest(out, "m2e", {
        "blocks": len(blocks), "lines": len(line_rows),
        "n_lines_distribution": dist, "despace": args.despace,
        "caveat": "M²E 是公式块，不是学生解题过程；含竖式对齐(\\t)，无错因标注",
    })
    print(f"共 {len(blocks)} 块 / {len(line_rows)} 行 → {out}")
    print("行数分布（10+ 并入 10）:", dict(sorted(dist.items(), key=lambda kv: int(kv[0]))))
    if tmp:
        tmp.cleanup()
    return 0


def cmd_scratchmath(args) -> int:
    try:
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError:
        raise SystemExit(
            "缺 pyarrow：F:/PyCharm/python.exe -m pip install pyarrow pandas\n"
            "（只导 ScratchMath 需要，另外两个子命令不依赖它）"
        )
    import base64

    src = Path(args.src)
    out = Path(args.out) if args.out else PUBLIC / "scratchmath"
    files = sorted(src.rglob("*.parquet")) if src.is_dir() else [src]
    meta: list[dict] = []
    for f in files:
        subset = f.parent.name if f.parent.name in ("primary", "middle") else "unknown"
        table = pq.read_table(f)
        data = table.to_pylist()
        if args.limit:
            data = data[: args.limit]
        img_dir = out / "images" / subset
        img_dir.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(data):
            qid = str(row.get("question_id") or f"{subset}_{i}")
            blob = row.get("student_scratchwork")
            name = ""
            if isinstance(blob, dict):
                raw = blob.get("bytes")
                if raw:
                    suffix = Path(str(blob.get("path") or "x.jpg")).suffix or ".jpg"
                    name = f"{qid}{suffix}"
                    (img_dir / name).write_bytes(raw)
            elif isinstance(blob, (bytes, bytearray)):
                name = f"{qid}.jpg"
                (img_dir / name).write_bytes(bytes(blob))
            elif isinstance(blob, str) and blob.startswith("data:"):
                name = f"{qid}.jpg"
                (img_dir / name).write_bytes(base64.b64decode(blob.split(",", 1)[1]))
            meta.append({
                "id": qid, "subset": subset, "image": f"images/{subset}/{name}" if name else "",
                "question": row.get("question"), "answer": row.get("answer"),
                "solution": row.get("solution"),
                "student_answer": row.get("student_answer"),
                "error_category": row.get("error_category"),
                "error_explanation": row.get("error_explanation"),
            })
    _write_jsonl(out / "metadata.jsonl", meta)
    _manifest(out, "scratchmath", {
        "samples": len(meta),
        "caveat": "无逐行 LaTeX 转写、多为小学应用题，不能进本项目的逐步批改主指标",
    })
    print(f"共 {len(meta)} 条 → {out / 'metadata.jsonl'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="公开手写数学数据集 → 本项目格式")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, name: str):
        p.add_argument("--src", required=True, help="下载解压后的目录，或 .tgz/.zip/.parquet")
        p.add_argument("--out", help=f"输出目录，默认 data/public/{name}")
        p.add_argument("--limit", type=int, default=0, help="每个分片最多导几条（0=全部）")

    mw = sub.add_parser("mathwriting", help="InkML → PNG + labels.csv")
    common(mw, "mathwriting")
    mw.add_argument("--splits", default="train,valid,test")
    mw.add_argument("--height", type=int, default=96, help="渲染图高（像素）")
    mw.add_argument("--pen", type=float, default=1.5, help="笔刷半径")
    mw.add_argument("--label-field", default="normalizedLabel,label",
                    help="真值字段名，CROHME 传 truth")
    mw.add_argument("--check-parse", action="store_true", help="顺带统计真值能否被本项目 CAS 解析")
    mw.set_defaults(func=cmd_mathwriting)

    m2 = sub.add_parser("m2e", help="多行 jsonl → blocks.jsonl / lines.csv")
    common(m2, "m2e")
    m2.add_argument("--despace", action="store_true", help="把 '1 5 . 4' 拼回 '15.4'")
    m2.set_defaults(func=cmd_m2e)

    sm = sub.add_parser("scratchmath", help="parquet → 图片 + metadata.jsonl")
    common(sm, "scratchmath")
    sm.set_defaults(func=cmd_scratchmath)

    pp = sub.add_parser("probe", help="只检查下载内容，不导出")
    pp.add_argument("dataset", choices=["mathwriting", "m2e", "scratchmath"])
    pp.add_argument("--src", required=True)
    pp.set_defaults(func=cmd_probe)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
