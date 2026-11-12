"""资产来源与隐私护栏。

这组测试盯的不是某张图，而是"以后每次提交都别再带进来"这件事：
新增图片必须登记、运行时产物必须被 ignore。R03 的验收标准就是这两条，
所以它们得能被机器查，而不是靠人记得。
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import check_assets as ca  # noqa: E402


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="没有 git")


def row(path: str, **kw) -> dict:
    base = {"path": path, "kind": "synthetic", "generator": "tools/x.py",
            "contains_personal": "no", "status": "ok", "note": ""}
    base.update(kw)
    return base


# --------------------------------------------------------------- 登记表本身

def test_real_table_is_well_formed():
    rows = ca.load_table()
    assert rows, "data/assets.csv 不该是空的"
    assert ca.check_rows(rows) == []


def test_real_table_uses_documented_header():
    with (ROOT / "data" / "assets.csv").open(encoding="utf-8-sig", newline="") as fh:
        assert next(csv.reader(fh)) == ca.FIELDS


def test_rejects_bad_status_and_kind():
    errs = ca.check_rows([row("a.png", kind="whatever")])
    assert any("kind" in e for e in errs)
    errs = ca.check_rows([row("a.png", status="也许")])
    assert any("status" in e for e in errs)


def test_ok_cannot_be_unknown_personal():
    """标了 ok 又说不知道含不含个人信息，等于没核过。"""
    errs = ca.check_rows([row("a.png", contains_personal="unknown", status="ok")])
    assert any("contains_personal" in e for e in errs)


def test_pending_may_be_unknown():
    assert ca.check_rows([row("a.png", contains_personal="unknown", status="pending")]) == []


# ------------------------------------------------------------------ 覆盖率

def test_unregistered_image_is_caught():
    rows = [row("docs/ok.png")]
    errs = ca.check_coverage(rows, ["docs/ok.png", "reports/new_sheet.jpg"])
    assert len(errs) == 1 and "reports/new_sheet.jpg" in errs[0]


def test_glob_covers_future_files():
    """以后生成 photo_results_v11 不该再改表。"""
    rows = [row("data/photo_results_v*.jsonl", kind="text_snapshot",
                contains_personal="unknown", status="pending")]
    assert ca.check_coverage(rows, ["data/photo_results_v11.jsonl"]) == []


def test_non_asset_files_need_no_entry():
    """代码和文档不是资产，不该被追着登记。"""
    rows = [row("a.png")]
    assert ca.check_coverage(rows, ["a.png", "mathagent/solve.py", "README.md"]) == []


def test_stale_entry_is_warned_not_failed():
    warns = ca.check_stale([row("reports/gone.png")], ["a.png"])
    assert warns and "reports/gone.png" in warns[0]


# ------------------------------------------------------- 真实仓库的两条硬约束

@needs_git
def test_every_tracked_asset_is_registered():
    """仓库里已有的图，必须都能在表里找到来源。新增图没登记 → 这里红。"""
    files = ca.tracked_files()
    assert ca.check_coverage(ca.load_table(), files) == []


@needs_git
def test_runtime_outputs_are_ignored():
    """错题本、当次快照、原图目录：任何一项漏了，下一次 add -A 就带出去了。"""
    assert ca.check_ignore() == []


def test_decide_only_fails_strict_on_pending():
    assert ca.decide([], ["x"], strict=False) == 0
    assert ca.decide([], ["x"], strict=True) == 1
    assert ca.decide([], [], strict=True) == 0
    assert ca.decide(["boom"], [], strict=False) == 1
