"""M²E 下载校验 + 切行评测的离线回归：不联网、不需要 token、不需要真数据集。

真下载要有魔搭账号、543 MB 图包和磁盘，不能进 CI；这里把「命令对不对、
下到的东西真不真、指标算得对不对、参数纪律守没守住」拆成纯本地断言。
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fetch_m2e as fm  # noqa: E402
from eval import run_line_eval as rle  # noqa: E402


# --------------------------------------------------------------------------- 夹具


def write_labels(root: Path, names_per_split: dict[str, list[str]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("M2E", encoding="utf-8")
    (root / "M2E.json").write_text("{}", encoding="utf-8")
    for split in ("train", "val", "test"):
        rows = [json.dumps({"name": n, "tex": "a \\n b \\n c"})
                for n in names_per_split.get(split, [])]
        (root / f"{split}.jsonl").write_text("\n".join(rows), encoding="utf-8")


def synthetic_three_line_image(path: Path) -> None:
    """三行短笔画：不能画长横线，否则 clear_long_runs 会当成横线本抹掉。"""
    im = Image.new("L", (220, 130), 255)
    d = ImageDraw.Draw(im)
    for y in (12, 52, 92):
        for k in range(7):
            x = 12 + k * 28
            d.rectangle((x, y, x + 16, y + 16), fill=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").save(path)


class FakeSplit:
    def __init__(self, n: int, note: str = ""):
        self._n = n
        self.note = note
        self.sheet = None

    def __len__(self):
        return self._n


# --------------------------------------------------------------------------- 下载校验


def test_verify_detects_lfs_pointer(tmp_path, monkeypatch):
    """最容易踩的坑：git 没装 lfs，下到 130 字节的指针还以为成功了。"""
    root = tmp_path / "m2e_raw"
    write_labels(root, {"train": ["1.jpg"], "test": ["2.jpg"]})
    # 测试用的是迷你假数据，关掉体积下限（空 split 文件 0 B 会误报），只验结构
    monkeypatch.setattr(fm, "MIN_BYTES", {k: 0 for k in fm.MIN_BYTES})
    (root / "images.zip").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 543249796\n"
    )
    got = fm.verify(root, require_images=True)
    assert got["zip"]["lfs_pointer"] is True
    assert any("LFS 指针" in p for p in got["problems"])
    assert got["ok"] is False
    assert got["splits"]["train"]["rows"] == 1
    assert got["splits"]["train"]["n_lines_distribution"] == {"3": 1}


def test_verify_counts_images_in_real_zip(tmp_path, monkeypatch):
    root = tmp_path / "m2e_raw"
    write_labels(root, {"test": ["1.jpg"]})
    # 测试用的是迷你假数据，关掉体积下限（空 split 文件 0 B 会误报），只验结构
    monkeypatch.setattr(fm, "MIN_BYTES", {k: 0 for k in fm.MIN_BYTES})
    with zipfile.ZipFile(root / "images.zip", "w") as zf:
        zf.writestr("images/a/1.jpg", b"fake")
        zf.writestr("images/b/2.jpg", b"fake")
        zf.writestr("README.txt", b"not an image")
    got = fm.verify(root, require_images=True)
    assert got["zip"]["images"] == 2
    assert got["ok"] is True


def test_unzip_flattens_and_reports_name_alignment(tmp_path):
    root = tmp_path / "m2e_raw"
    write_labels(root, {"test": ["1.jpg", "2.jpg", "3.jpg"]})
    with zipfile.ZipFile(root / "images.zip", "w") as zf:
        zf.writestr("a/1.jpg", b"one")
        zf.writestr("b/2.jpg", b"two")
        zf.writestr("c/other.jpg", b"three")
    got = fm.unzip_images(root)
    assert got["written"] == 3
    assert got["matched"] == 2
    assert got["match_rate"] == pytest.approx(2 / 3)
    assert (root / "images" / "1.jpg").read_bytes() == b"one"


def test_unzip_only_blocks_skips_unneeded_images(tmp_path):
    root = tmp_path / "m2e_raw"
    write_labels(root, {"test": ["1.jpg", "2.jpg", "3.jpg"]})
    with zipfile.ZipFile(root / "images.zip", "w") as zf:
        for n in ("1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"):
            zf.writestr(f"images/a/{n}", b"x")
    bf = tmp_path / "blocks.jsonl"
    bf.write_text("\n".join(json.dumps({"image": n}) for n in ("1.jpg", "3.jpg")),
                  encoding="utf-8")
    got = fm.unzip_images(root, wanted=fm.block_image_names(bf))
    assert got["written"] == 2
    assert got["skipped_not_needed"] == 3
    assert got["target_coverage"] == 1.0
    assert sorted(p.name for p in (root / "images").iterdir()) == ["1.jpg", "3.jpg"]


def test_run_quiet_never_prints_token(monkeypatch, capsys):
    class FakeProc:
        returncode = 0
        stdout = "usage: modelscope login --token ms-SUPERSECRET\n"
        stderr = ""
    monkeypatch.setattr(fm.subprocess, "run", lambda *a, **k: FakeProc())
    assert fm.run_quiet(["x"], "ms-SUPERSECRET") == 0
    out = capsys.readouterr().out
    assert "ms-SUPERSECRET" not in out
    assert "<TOKEN>" in out


def test_download_commands_never_carry_the_token(tmp_path):
    cmds = fm.download_commands("modelscope", tmp_path, "all")
    assert len(cmds) == 2
    assert "images.zip" in cmds[1]
    assert "train.jsonl" in cmds[0]
    assert "SECRET_TOKEN" not in " ".join(" ".join(c) for c in cmds)
    assert fm.login_command("modelscope")[-1] == "<MODELSCOPE_SDK_TOKEN>"
    assert '"' in fm.show_cmd(["modelscope", "download", "--local_dir", "D:/my data/M2E"])


def test_load_token_prefers_env_then_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("MODELSCOPE_SDK_TOKEN=from_file\nOTHER_KEY=not_a_token\n",
                   encoding="utf-8")
    monkeypatch.setattr(fm, "ENV_FILE", env)
    monkeypatch.setenv("MODELSCOPE_SDK_TOKEN", "from_env")
    assert fm.load_token() == "from_env"
    monkeypatch.delenv("MODELSCOPE_SDK_TOKEN")
    assert fm.load_token() == "from_file"


def test_status_never_exposes_any_secret(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("MODELSCOPE_SDK_TOKEN=SUPERSECRET\nOTHER_KEY=OTHERSECRET\n",
                   encoding="utf-8")
    monkeypatch.setattr(fm, "ENV_FILE", env)
    monkeypatch.delenv("MODELSCOPE_SDK_TOKEN", raising=False)
    st = fm.status(tmp_path)
    blob = json.dumps(st, ensure_ascii=False)
    assert st["token_ready"] is True
    assert st["token_source"] == "项目 .env"
    assert "SUPERSECRET" not in blob and "OTHERSECRET" not in blob
    assert "ms-" not in blob


# --------------------------------------------------------------------------- 切行评测


def test_eval_blocks_counts_metrics_and_discloses_missing(tmp_path):
    blocks = [
        {"id": "b1", "image": "1.jpg", "n_lines": 2, "has_align": False},
        {"id": "b2", "image": "2.jpg", "n_lines": 3, "has_align": False},
        {"id": "b3", "image": "3.jpg", "n_lines": 4, "has_align": True},   # 竖式，跳过
        {"id": "b4", "image": "4.jpg", "n_lines": 4, "has_align": False},
        {"id": "b5", "image": "5.jpg", "n_lines": 3, "has_align": False},   # 缺图
    ]
    images = {f"{i}.jpg": tmp_path / f"{i}.jpg" for i in (1, 2, 3, 4)}
    pred = {"1.jpg": 2, "2.jpg": 4, "4.jpg": 4}
    split_fn = lambda path, **kw: FakeSplit(pred[Path(path).name])
    got = rle.eval_blocks(blocks, images, rle.DEFAULT_PARAMS,
                          split_fn=split_fn, skip_align=True)
    s = rle.score(got["records"], seed=1, n_boot=50)
    assert got["skipped_align"] == 1
    assert got["missing"] == ["b5"]
    assert s["n"] == 3
    assert s["exact"]["hit"] == 2 and s["exact"]["value"] == pytest.approx(2 / 3)
    assert s["within1"]["hit"] == 3
    assert s["mae"] == pytest.approx(1 / 3)
    assert s["over"] == pytest.approx(1 / 3) and s["under"] == 0
    assert "3 行" in s["by_group"]


def test_align_blocks_included_and_layered_by_default(tmp_path):
    """M²E 里竖式块占多数，全跳过会让样本失去代表性。"""
    blocks = [
        {"id": "a", "image": "1.jpg", "n_lines": 3, "has_align": True},
        {"id": "b", "image": "2.jpg", "n_lines": 3, "has_align": False},
    ]
    images = {f"{i}.jpg": tmp_path / f"{i}.jpg" for i in (1, 2)}
    split_fn = lambda path, **kw: FakeSplit(3)
    got = rle.eval_blocks(blocks, images, rle.DEFAULT_PARAMS, split_fn=split_fn)
    s = rle.score(got["records"], seed=1, n_boot=20)
    assert got["skipped_align"] == 0
    assert s["n"] == 2
    assert set(s["by_align"]) == {"竖式对齐", "普通"}


def test_real_split_page_splits_synthetic_block(tmp_path):
    """不注入假切分器，确认评测入口和 mathagent.lines 真的接得上。"""
    path = tmp_path / "block.png"
    synthetic_three_line_image(path)
    blocks = [{"id": "x", "image": "block.png", "n_lines": 3, "has_align": False}]
    got = rle.eval_blocks(blocks, {"block.png": path}, rle.DEFAULT_PARAMS)
    assert got["records"][0]["n_pred"] == 3
    assert got["records"][0]["exact"] == 1


def test_report_states_limits_and_params(tmp_path):
    report = {
        "generated": "2026-11-10 22:00:00", "split": "test",
        "params_source": "默认参数（split_page 默认值，未调参）",
        "params": rle.DEFAULT_PARAMS, "n_blocks": 10,
        "skipped_align": 1, "missing": [], "errors": [],
        "metrics": {"n": 0, "note": "没有可评测的块"}, "tune": None,
        "reproduce": "python eval/run_line_eval.py --split test",
    }
    md = rle.render_md(report)
    assert "CC BY-NC 4.0" in md
    assert "不替代" in md
    assert "只比对行数" in md
    assert "未产出指标" in md
    assert '"threshold_ratio"' in md
    assert "train 上调参" not in md

    report["tune"] = {"n_candidates": 18, "n_tune": 300,
                      "table": [{"exact": 0.7}], "best": rle.DEFAULT_PARAMS}
    assert "不能当最终指标" in rle.render_md(report)


def test_tune_on_test_is_rejected():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_line_eval.py"),
         "--split", "test", "--tune"],
        capture_output=True, text=True, cwd=str(ROOT), check=False,
    )
    assert proc.returncode != 0
    assert "参数只允许在 train 上调" in proc.stderr


def test_smoke_report_does_not_overwrite_official(tmp_path, monkeypatch):
    blocks = [{"id": "t1", "split": "test", "image": "1.jpg",
               "n_lines": 3, "has_align": False}]
    bf = tmp_path / "blocks.jsonl"
    bf.write_text("\n".join(json.dumps(b) for b in blocks), encoding="utf-8")
    img_dir = tmp_path / "images"
    synthetic_three_line_image(img_dir / "1.jpg")
    monkeypatch.setattr(rle, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(sys, "argv", [
        "run_line_eval.py", "--blocks", str(bf), "--images", str(img_dir),
        "--split", "test", "--smoke", "--n-boot", "20",
    ])
    assert rle.main() == 0
    assert (tmp_path / "reports" / "_smoke_eval_lines.md").exists()
    assert (tmp_path / "reports" / "_smoke_eval_lines.json").exists()
    assert not (tmp_path / "reports" / "eval_lines.md").exists()
    payload = json.loads((tmp_path / "reports" / "_smoke_eval_lines.json").read_text(
        encoding="utf-8"))
    assert payload["metrics"]["n"] == 1
    assert payload["metrics"]["exact"]["hit"] == 1
    assert payload["license"] == "CC BY-NC 4.0"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
