"""公开数据集导入的离线回归测试：不联网、不依赖 pyarrow。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import import_public as ip  # noqa: E402


INKML = """<ink xmlns="http://www.w3.org/2003/InkML">
  <annotation type="label">2x+5=13</annotation>
  <annotation type="normalizedLabel">2x+5=13</annotation>
  <annotation type="sampleId">abc123</annotation>
  <annotation type="splitTagOriginal">train</annotation>
  <traceFormat><channel name="X" type="decimal"/><channel name="Y" type="decimal"/>
    <channel name="T" type="decimal"/></traceFormat>
  <trace id="0">10 20 0,20 25 10,30 20 20</trace>
  <trace id="1">10 40 30,20 40 40,30 40 50</trace>
</ink>"""


def test_read_inkml_uses_normalized_label_and_keeps_strokes(tmp_path):
    p = tmp_path / "a.inkml"
    p.write_text(INKML, encoding="utf-8")
    label, sid, strokes = ip.read_inkml(p)
    assert label == "2x+5=13"
    assert sid == "abc123"
    assert len(strokes) == 2
    assert strokes[0][0] == (10.0, 20.0)


def test_render_ink_is_not_blank(tmp_path):
    p = tmp_path / "a.inkml"
    p.write_text(INKML, encoding="utf-8")
    _, _, strokes = ip.read_inkml(p)
    img = ip.render_ink(strokes, height=64)
    assert img.size[1] == 64
    assert img.convert("L").getextrema()[0] < 128, "应该有黑色笔画，不能是白板"


def test_despace_joins_digit_runs_but_keeps_commands():
    assert ip._despace("1 5 . 4 \\times 4 = 6 1 . 6") == "15.4 \\times 4 = 61.6"
    assert ip._despace("9 . 9 \\times 3 . 8") == "9.9 \\times 3.8"


def test_m2e_import_splits_lines_and_flags_alignment(tmp_path, monkeypatch):
    src = tmp_path / "m2e"
    src.mkdir()
    rec = {"name": "0.jpg", "tex": "1 5 . 4 \\times 4 \\t 1 5 . 4 \\n \\times 4 \\n 6 1 . 6"}
    (src / "train.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    out = tmp_path / "out"

    from types import SimpleNamespace

    args = SimpleNamespace(src=str(src), out=str(out), limit=0, despace=True)
    ip.cmd_m2e(args)
    blocks = [json.loads(x) for x in
              (out / "blocks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(blocks) == 1
    b = blocks[0]
    assert b["n_lines"] == 3
    assert b["has_align"] is True
    assert b["lines"][0].startswith("15.4")
    assert b["tex_raw"] == rec["tex"], "有损的归一化不能覆盖原始标注"
    assert (out / "SOURCE.json").exists()


def test_manifest_records_license():
    assert "CC BY-NC" in ip.SOURCES["m2e"]["license"]
    assert "CC BY-NC-SA" in ip.SOURCES["mathwriting"]["license"]
    assert "CC BY-NC-SA" in ip.SOURCES["scratchmath"]["license"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
