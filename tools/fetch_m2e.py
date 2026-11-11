"""M²E（魔搭 Wente47/M2E）下载与离线验收。

职责边界：**下载交给官方 modelscope CLI**（断点续传、LFS 都靠它），
本脚本只做三件本地事——拼命令、验文件、解图片。不自己实现 HTTP 下载，
因为魔搭的文件走 LFS，匿名 resolve 会返回「当前操作需要登录」，自己写容易
下到一个 130 字节的指针文件还以为成功了。

Token 有两个来源，优先级：环境变量 MODELSCOPE_SDK_TOKEN > 项目根 .env 的同名键。
`.env` 已被 .gitignore 忽略（项目本来就用它放 API 密钥）。写进 .env 是**代跑场景**的推荐做法：
每个终端命令都是独立 shell，export 一次不会传到下一次调用；写文件才跨调用有效。
脚本绝不打印令牌明文、绝不写进日志/报告，download 命令里也不带明文（先 login，让官方 CLI 自己缓存）。

用法：

  # 0. 开工体检（不回显令牌），令牌写进项目根 .env 的
  #    MODELSCOPE_SDK_TOKEN=ms-... 即可，.env 已被 gitignore
  F:/PyCharm/python.exe tools/fetch_m2e.py check
  # 1. 看要跑什么（默认 dry-run，不联网、不下载）
  F:/PyCharm/python.exe tools/fetch_m2e.py labels
  # 2. 真下载标注文件（约 18 MB，先跑这一步验证账号通不通）
  F:/PyCharm/python.exe tools/fetch_m2e.py labels --exec
  # 3. 真下载图包（543 MB）
  F:/PyCharm/python.exe tools/fetch_m2e.py images --exec
  # 4. 验收 + 解压
  F:/PyCharm/python.exe tools/fetch_m2e.py verify
  F:/PyCharm/python.exe tools/fetch_m2e.py unzip
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = ROOT / "data" / "public" / "m2e_raw"
REPO = "Wente47/M2E"
LABEL_FILES = ("README.md", "M2E.json", "train.jsonl", "val.jsonl", "test.jsonl")
IMAGE_ZIP = "images.zip"
ZIP_MAGIC = b"PK\x03\x04"
LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 官方 repo tree 核实过的量级；低于下限说明下到指针文件或被截断
MIN_BYTES = {"train.jsonl": 10_000_000, "val.jsonl": 1_000_000,
             "test.jsonl": 1_000_000, IMAGE_ZIP: 500_000_000}


# --------------------------------------------------------------------------- 命令


def find_cli(explicit: str | None = None) -> str | None:
    """Windows 上 SDK 常装在解释器的 Scripts/ 里，未必进 PATH。"""
    if explicit:
        return explicit
    found = shutil.which("modelscope")
    if found:
        return found
    for rel in ("Scripts/modelscope.exe", "Scripts/modelscope", "bin/modelscope"):
        cand = Path(sys.executable).parent / rel
        if cand.exists():
            return str(cand)
    return None


def download_commands(cli: str, root: Path, stage: str) -> list[list[str]]:
    base = [cli, "download", "--dataset", REPO, "--local_dir", str(root)]
    out = []
    if stage in ("labels", "all"):
        out.append([*base, *LABEL_FILES])
    if stage in ("images", "all"):
        out.append([*base, IMAGE_ZIP])
    return out


def login_command(cli: str) -> list[str]:
    # 占位符：真实 token 只在 --exec 时从环境变量注入，不会出现在打印内容里
    return [cli, "login", "--token", "<MODELSCOPE_SDK_TOKEN>"]


def show_cmd(cmd: list[str]) -> str:
    """打印出来要能直接复制执行，路径带空格就补引号。"""
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


def run_quiet(cmd: list[str], token: str, tail: int = 15) -> int:
    """捕获 CLI 输出、脱敏后再打印。

    官方 CLI 出错时有可能把 argv 一起回显，而 argv 里可能带令牌；
    直接继承 stdout 等于把令牌写进对话记录。宁可放弃实时进度条。
    """
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True,
                          errors="replace")
    text = (proc.stdout or "") + (proc.stderr or "")
    if token:
        text = text.replace(token, "<TOKEN>")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines:
        print("\n".join(lines[-tail:]))
    return proc.returncode


def run_download(cli: str, root: Path, stage: str, token: str) -> int:
    root.mkdir(parents=True, exist_ok=True)
    inline_token = False
    if token:
        print("登录官方 CLI（输出脱敏）…", flush=True)
        rc = run_quiet([cli, "login", "--token", token], token)
        if rc:
            # 旧版 SDK 没有 login 子命令，退回 download --token。
            # 明文只进 argv，不打印、不写文件；失败多半是 token 本身不对。
            inline_token = True
            print("login 子命令不可用或失败，退回 `download --token`；"
                  "若随后报鉴权失败，请检查 token 是否复制完整、是否过期。")
    else:
        print("未提供 MODELSCOPE_SDK_TOKEN，按匿名方式尝试；M²E 要求登录，"
              "若报「当前操作需要登录」请按文档 §1 取令牌。")
    for cmd in download_commands(cli, root, stage):
        actual = [*cmd, "--token", token] if inline_token else cmd
        shown = [*cmd, "--token", "<TOKEN>"] if inline_token else cmd
        print(">", show_cmd(shown), flush=True)
        print("  下载中，结束后统一输出脱敏日志…", flush=True)
        rc = run_quiet(actual, token)
        if rc:
            print(f"下载失败（退出码 {rc}），已停止。可重跑本命令续传。")
            return rc
    return 0


# --------------------------------------------------------------------------- 验收


def read_blocks(root: Path) -> dict[str, list[dict]]:
    """把三个 jsonl 读进来，顺便验每行是不是合法 JSON、有没有 tex/name。"""
    out: dict[str, list[dict]] = {}
    for split, fname in (("train", "train.jsonl"), ("val", "val.jsonl"),
                         ("test", "test.jsonl")):
        p = root / fname
        rows, bad = [], 0
        if p.exists():
            for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                    if "tex" not in rec or "name" not in rec:
                        bad += 1
                        continue
                    rows.append(rec)
                except json.JSONDecodeError:
                    bad += 1
        out[split] = rows
        out[f"{split}_bad"] = bad  # type: ignore[assignment]
    return out


def verify(root: Path, require_images: bool = False) -> dict:
    """下完之后先验一遍再说。返回结构化结果，测试可直接断言。"""
    out: dict = {"root": str(root), "labels": {}, "splits": {},
                 "zip": {}, "problems": [], "ok": False}
    for f in LABEL_FILES:
        p = root / f
        if not p.exists():
            out["problems"].append(f"缺文件 {f}")
            continue
        size = p.stat().st_size
        out["labels"][f] = size
        if f in MIN_BYTES and size < MIN_BYTES[f]:
            out["problems"].append(
                f"{f} 只有 {size:,} B（预期 ≥{MIN_BYTES[f]:,}），疑似 LFS 指针或下载不完整"
            )

    blocks = read_blocks(root)
    for split in ("train", "val", "test"):
        rows = blocks[split]
        dist: dict[str, int] = {}
        for r in rows:
            n = len([x for x in str(r.get("tex", "")).replace("\\n", "\n").split("\n")
                     if x.strip()])
            k = str(min(n, 10))
            dist[k] = dist.get(k, 0) + 1
        out["splits"][split] = {
            "rows": len(rows), "bad": blocks[f"{split}_bad"],
            "n_lines_distribution": dict(sorted(dist.items(), key=lambda kv: int(kv[0]))),
        }
        if blocks[f"{split}_bad"]:
            out["problems"].append(f"{split}.jsonl 有 {blocks[f'{split}_bad']} 行坏记录")

    z = root / IMAGE_ZIP
    if not z.exists():
        out["zip"] = {"present": False}
        if require_images:
            out["problems"].append(f"缺 {IMAGE_ZIP}（543 MB 图包）")
    else:
        head = z.open("rb").read(200)
        info: dict = {"present": True, "bytes": z.stat().st_size}
        if head.startswith(LFS_MAGIC):
            info["lfs_pointer"] = True
            out["problems"].append(
                f"{IMAGE_ZIP} 是 Git LFS 指针文件，不是真图包；请装 git-lfs 或用 CLI 重下"
            )
        elif not head.startswith(ZIP_MAGIC):
            info["not_zip"] = True
            out["problems"].append(f"{IMAGE_ZIP} 开头不是 PK，不是有效 ZIP")
        else:
            try:
                with zipfile.ZipFile(z) as zf:
                    imgs = [i for i in zf.namelist()
                            if Path(i).suffix.lower() in IMG_EXT]
                    info["images"] = len(imgs)
                    info["sample"] = imgs[:3]
            except zipfile.BadZipFile as exc:
                info["bad_zip"] = True
                out["problems"].append(f"{IMAGE_ZIP} 打不开：{exc}")
            if z.stat().st_size < MIN_BYTES[IMAGE_ZIP]:
                out["problems"].append(
                    f"{IMAGE_ZIP} 只有 {z.stat().st_size:,} B，可能被截断"
                )
        out["zip"] = info

    out["ok"] = not out["problems"]
    return out


def reference_names(root: Path) -> set[str]:
    blocks = read_blocks(root)
    names: set[str] = set()
    for split in ("train", "val", "test"):
        for r in blocks[split]:
            names.add(Path(str(r.get("name", ""))).name)
    return names


def block_image_names(path: Path) -> set[str]:
    """从 blocks.jsonl 取出评测真正会用到的图片名。

    图包里有近 10 万张图，而一次评测通常只跑 500~2000 块；
    全解压既慢又占盘，且没有任何额外信息量。
    """
    names: set[str] = set()
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        n = Path(str(json.loads(ln).get("image", ""))).name
        if n:
            names.add(n)
    return names


def unzip_images(root: Path, limit: int = 0,
                 wanted: set[str] | None = None) -> dict:
    """把 zip 里的图片摊平到 images/，并核对与标注 name 的对齐率。

    zip 里常带一层子目录；评测按 basename 找图，所以这里也按 basename 落盘。
    不同子目录重名会记进 collisions，宁可让报告披露也不悄悄覆盖。
    wanted 给定时只解压这批图，并用「目标覆盖率」而不是全量对齐率来判断好坏。
    """
    z = root / IMAGE_ZIP
    if not z.exists():
        raise SystemExit(f"找不到 {z}，先跑 images --exec")
    with z.open("rb") as f:
        if not f.read(4).startswith(ZIP_MAGIC[:2]):
            raise SystemExit(f"{z} 不是真 ZIP（可能是 LFS 指针），先跑 verify 看问题")
    dest = root / "images"
    dest.mkdir(parents=True, exist_ok=True)
    written, collisions, skipped = 0, 0, 0
    with zipfile.ZipFile(z) as zf:
        for item in zf.infolist():
            if item.is_dir() or Path(item.filename).suffix.lower() not in IMG_EXT:
                continue
            base = Path(item.filename).name
            if wanted is not None and base not in wanted:
                skipped += 1
                continue
            if limit and written >= limit:
                break
            target = dest / base
            if target.exists():
                collisions += 1
                continue
            with zf.open(item) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            written += 1
    ref = reference_names(root)
    have = {p.name for p in dest.iterdir() if p.suffix.lower() in IMG_EXT}
    matched = len(have & ref)
    return {
        "dest": str(dest), "written": written, "collisions": collisions,
        "skipped_not_needed": skipped,
        "total_in_dir": len(have), "reference_names": len(ref),
        "matched": matched,
        "match_rate": matched / len(ref) if ref else None,
        "wanted": len(wanted) if wanted is not None else None,
        "target_coverage": (len(have & wanted) / len(wanted)) if wanted else None,
    }


def next_steps(root: Path, blocks: Path | None = None) -> str:
    py = sys.executable
    eval_cmd = f"  3) {py} eval/run_line_eval.py --split test --images {root / 'images'}"
    if blocks:
        eval_cmd += f" --blocks {blocks}"
    return "\n".join([
        "下一步（按顺序）：",
        f"  1) {py} tools/import_public.py probe m2e --src {root}",
        f"  2) {py} tools/import_public.py m2e --src {root} --limit 2000",
        eval_cmd,
        "  4) 看 reports/eval_lines.md（外部验证集，指标与自采数据分表报）",
        "  若改过 --root，第 3 步还要带 --blocks <导入输出目录>/blocks.jsonl",
    ])


ENV_FILE = ROOT / ".env"


def load_token() -> str:
    """读令牌，只返回给调用方，不打印。环境变量优先于 .env。"""
    tok = os.environ.get("MODELSCOPE_SDK_TOKEN", "").strip()
    if tok:
        return tok
    if not ENV_FILE.exists():
        return ""
    try:
        from dotenv import dotenv_values  # 项目已有 python-dotenv
        return str(dotenv_values(ENV_FILE).get("MODELSCOPE_SDK_TOKEN") or "").strip()
    except ImportError:
        pass
    # 没装 dotenv 也能用：只挑这一行，不解析、不回显整个文件
    for line in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("MODELSCOPE_SDK_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def status(root: Path) -> dict:
    """开工前体检：只报「有没有」，绝不回显令牌内容或 .env 里其它密钥。"""
    cli = find_cli()
    token = load_token()
    src = ("环境变量" if os.environ.get("MODELSCOPE_SDK_TOKEN", "").strip()
           else ("项目 .env" if token else "未提供"))
    try:
        free_gb = round(shutil.disk_usage(str(Path(ROOT).anchor)).free / 1e9, 1)
    except OSError:
        free_gb = None
    out = {
        "sdk_installed": cli is not None,
        "cli_path": cli or "未找到（装完 SDK 后应在解释器目录 Scripts/ 下）",
        "token_ready": bool(token),
        "token_source": src,
        "free_disk_gb": free_gb,
        "download_dir": str(root),
        "download_dir_exists": root.exists(),
    }
    missing = []
    if not cli:
        missing.append("装 SDK：F:/PyCharm/python.exe -m pip install -i "
                       "https://pypi.tuna.tsinghua.edu.cn/simple modelscope")
    if not token:
        missing.append("把 MODELSCOPE_SDK_TOKEN=ms-... 追加到项目根 .env（已 gitignore）")
    if free_gb is not None and free_gb < 2:
        missing.append(f"磁盘只剩 {free_gb} GB，建议留 2 GB 以上")
    out["todo"] = missing
    out["ready"] = not missing
    return out


# --------------------------------------------------------------------------- CLI


def main() -> int:
    ap = argparse.ArgumentParser(description="M²E 下载（官方 CLI）与离线验收")
    ap.add_argument("stage", choices=["check", "labels", "images", "all", "verify", "unzip"],
                    help="check=开工体检（不回显令牌）；labels=标注小文件；"
                         "images=543MB 图包；verify=只验收；unzip=只解压")
    ap.add_argument("--exec", action="store_true", help="真正执行下载（默认只打印命令）")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="下载目录")
    ap.add_argument("--cli", help="modelscope CLI 路径（默认自动找）")
    ap.add_argument("--limit", type=int, default=0, help="解压图片数上限（0=全部）")
    ap.add_argument("--only-blocks", help="unzip：只解压该 blocks.jsonl 用到的图片（推荐）")
    ap.add_argument("--require-images", action="store_true", help="verify 时把缺图包也算问题")
    ap.add_argument("--json", action="store_true", help="验收结果输出 JSON")
    args = ap.parse_args()
    root = Path(args.root)

    if args.stage == "check":
        st = status(root)
        print(json.dumps(st, ensure_ascii=False, indent=2))
        if st["todo"]:
            print("还差：")
            for t in st["todo"]:
                print(f"  - {t}")
        else:
            print("就绪：SDK 与令牌都在，可以直接跑 labels --exec。")
        return 0

    if args.stage == "verify":
        result = verify(root, args.require_images)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"目录 {root}")
            for f, size in result["labels"].items():
                print(f"  {f:16s} {size:>12,} B")
            for s in ("train", "val", "test"):
                d = result["splits"][s]
                print(f"  {s:16s} {d['rows']:>7,} 条 | 坏行 {d['bad']} | 行数分布 {d['n_lines_distribution']}")
            print(f"  images.zip        {result['zip']}")
            if result["problems"]:
                print("问题：")
                for p in result["problems"]:
                    print(f"  - {p}")
            else:
                print("验收通过。")
                print(next_steps(root))
        return 0 if result["ok"] else 1

    if args.stage == "unzip":
        wanted = (block_image_names(Path(args.only_blocks))
                  if args.only_blocks else None)
        result = unzip_images(root, args.limit, wanted)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        cover = result["target_coverage"]
        if cover is not None:
            if cover < 0.9:
                print(f"警告：目标图只覆盖 {cover:.1%}，检查 zip 内命名规则与 blocks.jsonl 的 image 是否一致。")
        elif result["match_rate"] is not None and result["match_rate"] < 0.9:
            print("警告：图片名与标注 name 对齐率偏低，检查 zip 里是不是还有别的命名规则。")
        print(next_steps(root, Path(args.only_blocks) if args.only_blocks else None))
        return 0

    cli = find_cli(args.cli)
    if not cli and args.exec:
        print("找不到 modelscope 命令。先装 SDK：")
        print(f"  {sys.executable} -m pip install modelscope")
        print("装完仍找不到就传 --cli <modelscope完整路径>（Windows 常在解释器目录 Scripts/ 下）。")
        return 2
    show_cli = cli or "modelscope"
    if not cli:
        print("本机还没装 modelscope SDK，下面是命令预览；装完才能执行：")
        print(f"  {sys.executable} -m pip install modelscope\n")
    token = load_token()
    print(f"CLI: {show_cli}")
    if token:
        print(">", show_cmd(login_command(show_cli)))
    for cmd in download_commands(show_cli, root, args.stage):
        print(">", show_cmd(cmd))
    if not args.exec:
        print("\n以上只是命令预览（dry-run）。加 --exec 才真正下载。")
        print("先跑 labels 验证登录，通了再下 543 MB 的图包。")
        return 0

    rc = run_download(cli, root, args.stage, token)
    if rc:
        return rc
    result = verify(root, require_images=(args.stage in ("images", "all")))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
