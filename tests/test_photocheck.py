"""拍照批改的测试。

不打网络：识别器换成本地假实现，喂什么返回什么。这样测的是**链路的判断**，
不是模型今天认得好不好——后者会随模型版本变，测了也守不住。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mathagent import photocheck
from mathagent.photocheck import (
    LineRead,
    PhotoCheck,
    _apply_uncertainty,
    _canon,
    _judge,
    _vote,
    readable,
)
from mathagent.types import Candidate, WorkCheck
from mathagent.workcheck import grade

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "work" / "work_fake.jpg"


# ---------------------------------------------------------------- 文字还原


def test_readable_keeps_math_structure():
    """`\\frac{1}{2}` 不能变成 `\\frac12`——那是另一个东西。"""
    out = readable(r"\frac{1}{2}x + 3 = 0")
    assert r"\frac{1}{2}" in out


def test_readable_unwraps_text():
    assert readable(r"\text{两边同乘2} 2x=4") == "两边同乘2 2x=4"


def test_readable_strips_delimiters():
    assert readable(r"\[x = 1\]") == "x = 1"
    assert readable(r"$x = 1$") == "x = 1"


def test_readable_keeps_chinese_or():
    """`或` 是学生分两个根的标准写法，不能丢。"""
    assert "或" in readable(r"x = 2 \text{ 或 } x = 3")


def test_readable_strips_leading_derivation_marks():
    """行首的 ⇒ / → 是"所以"的标记，不是式子的一部分。

    实拍页（W01/W17）每行都以 ⇒ 开头，三个识别来源分别读成 `\\Rightarrow`、
    `\\rightarrow`、`arrow`，投票把它们判成三派，一行正确的推导就这么作废了。
    """
    assert readable(r"\rightarrow (x-2)(x-3)=0") == "(x-2)(x-3)=0"
    assert readable(r"\Rightarrow x = 1") == "x = 1"
    assert readable("arrow (x-2)(x-3)=0") == "(x-2)(x-3)=0"
    assert readable(r"\therefore x=3") == "x=3"
    # 连着两个箭头（学生手快写重了）也要剥干净
    assert readable(r"\Rightarrow \Rightarrow x=1") == "x=1"


def test_readable_keeps_arrow_inside_line():
    """句子中间的箭头不动——那里可能有别的含义，剥了就是改内容。"""
    assert r"\rightarrow" in readable(r"A \rightarrow B")


def test_canon_same_line_different_arrow_readings():
    """同一行被不同来源读成 ⇒ / arrow / 不读，必须归一成同一个键。"""
    a = _canon(r"\Rightarrow x=2 \text{ 或 } x=3")
    b = _canon(r"arrow x=2 或 x=3")
    c = _canon(r"x = 2 \text{ 或 } x = 3")
    assert a == b == c == "x=2或x=3"


# ---------------------------------------------------------------- 投票


def _c(latex, source):
    return Candidate(latex=latex, source=source)


def test_canon_treats_superscript_forms_as_same():
    assert _canon("x^{2} - 5x") == _canon("x²-5x")


def test_vote_two_vendors_agree():
    latex, conf, others, verdict, _ = _vote(
        [_c("x^2-5x+6=0", "qwen3-vl"), _c(r"x^{2} - 5x + 6 = 0", "glm-4v")]
    )
    assert verdict == "ok"
    assert conf == pytest.approx(1.0)
    assert latex


def test_vote_disagree_goes_to_review():
    latex, conf, others, verdict, notes = _vote(
        [_c("x^2-5x+6=0", "qwen3-vl"), _c("x^2+5x+6=0", "glm-4v")]
    )
    assert verdict == "review"
    assert conf <= 0.5
    assert others, "另一派要留作备选，界面要靠它显示「别的可能」"


def test_vote_single_source_needs_corroboration():
    """一个来源读出内容，可能是编的，不能用来定对错。"""
    _, conf, _, verdict, _ = _vote([_c("x^2-5x+6=0", "qwen3-vl")], True)
    assert verdict == "review"
    assert conf <= 0.8


def test_vote_single_source_ok_when_only_one_recognizer():
    """环境里就一个识别器时，"要旁证"等于"永远判不了"，只能放宽。"""
    _, conf, _, verdict, _ = _vote([_c("x^2-5x+6=0", "qwen3-vl")], False)
    assert verdict == "ok"
    assert conf <= 0.8


def test_vote_all_empty_is_failed():
    assert _vote([_c("", "qwen3-vl")])[3] == "failed"


# ---------------------------------------------------------------- 不确定性传播


def _lines(sure: list[bool]) -> list[LineRead]:
    return [
        LineRead(index=i, latex="x", text="x", verdict="ok" if s else "review",
                 confidence=1.0 if s else 0.5)
        for i, s in enumerate(sure, start=1)
    ]


def test_uncertain_line_withdraws_its_own_verdict():
    work: WorkCheck = grade("x^2-5x+6=0\nx^2-6=5x\nx=2 或 x=3")
    assert work.grade == "fake", "先确认这份过程本来是判错的"

    _apply_uncertainty(work, _lines([True, False, True]))

    step = next(s for s in work.steps if s.idx == 2)
    assert step.verdict == "suspect"
    assert step.counter == "", "反例建立在没认准的字上，必须撤掉"
    assert "没认准" in step.reason


def test_uncertain_baseline_withdraws_the_next_verdict():
    """上一步没认准，拿它当基准比出来的"不成立"也不作数。"""
    work: WorkCheck = grade("x^2-5x+6=0\nx^2-6=5x\nx=2 或 x=3")

    _apply_uncertainty(work, _lines([True, False, True]))

    # 第 3 行（x=2 或 x=3）拿第 2 行当基准，基准没认准，结论撤回
    step3 = next(s for s in work.steps if s.idx == 3)
    assert step3.verdict == "suspect"


def test_uncertain_baseline_withdraws_praise_too():
    """依据没认准，判"成立"的那一步同样不能算数——错夸人也是装作确定。"""
    work: WorkCheck = grade("x^2-5x+6=0\n(x-2)(x-3)=0\nx=2 或 x=3")
    assert work.grade == "pass", "先确认这份过程本来全对"
    assert next(s for s in work.steps if s.idx == 3).verdict == "ok"

    _apply_uncertainty(work, _lines([True, False, True]))

    # 第 2 行没认准：它自己判不了，第 3 行拿它当依据，也跟着确认不了
    step2 = next(s for s in work.steps if s.idx == 2)
    step3 = next(s for s in work.steps if s.idx == 3)
    assert step2.verdict == "unknown"
    assert step3.verdict == "unknown"
    assert work.grade == "unread", "有步骤的「成立」确认不了，就不能说过程成立"


def test_all_verdicts_withdrawn_becomes_unread_not_wrong():
    work: WorkCheck = grade("x^2-5x+6=0\nx^2-6=5x\nx=2 或 x=3")
    _apply_uncertainty(work, _lines([True, False, True]))
    assert work.grade == "unread"
    assert work.error_type == "", "认不准不能写成错因，否则错题本会被污染"
    assert "不是你写错了" in work.advice


def test_sure_lines_change_nothing():
    work: WorkCheck = grade("x^2-5x+6=0\nx^2-6=5x\nx=2 或 x=3")
    before = [(s.idx, s.verdict) for s in work.steps]
    notes = _apply_uncertainty(work, _lines([True, True, True]))
    assert notes == []
    assert [(s.idx, s.verdict) for s in work.steps] == before
    assert work.grade == "fake"


def test_hole_between_steps_withdraws_the_comparison():
    """中间一行没读出来，隔着窟窿比出来的结论不作数。"""
    work: WorkCheck = grade("x^2-5x+6=0\n?\nx^2-5x=6\nx=2 或 x=3")
    _apply_uncertainty(work, _lines([True, False, True, True]))

    # 第 3 步是隔着第 2 行的窟窿跟题面比的，结论必须撤回
    step3 = next(s for s in work.steps if s.idx == 3)
    assert step3.verdict == "suspect"


# ---------------------------------------------------------------- 端到端（假识别器）


class FakeRec:
    """按小图文件名里的行号返回预置结果。"""

    def __init__(self, name: str, script: dict[int, str]):
        self.name = name
        self.script = script

    def available(self) -> bool:
        return True

    def recognize(self, image, k: int = 1):
        idx = int(Path(image).stem.rsplit("_", 1)[-1])
        return [Candidate(latex=self.script.get(idx, ""), source=self.name)]


def _run(tmp_path, monkeypatch, a: dict[int, str], b: dict[int, str]) -> PhotoCheck:
    monkeypatch.setattr(photocheck, "WORK_DIR", tmp_path)
    monkeypatch.setattr(
        photocheck, "build_ensemble",
        lambda **kw: [FakeRec("qwen3-vl", a), FakeRec("glm-4v", b)],
    )
    return photocheck.check(str(FIXTURE))


def test_photo_end_to_end_fake_derivation(tmp_path, monkeypatch):
    """那张"过程错、答案对"的样例照，走完整条链路仍然判 fake。"""
    script = {
        1: "x^{2} - 5x + 6 = 0",
        2: "x^{2} - 6 = 5x",
        3: "x^{2} - 5x = 6",
        4: r"x = 2 \text{ 或 } x = 3",
    }
    pc = _run(tmp_path, monkeypatch, script, script)
    assert pc.work.grade == "fake"
    assert pc.work.error_type == "伪推导"
    assert len(pc.lines) == 4
    assert "5x" in pc.text


def test_photo_gallery_matches_lines(tmp_path, monkeypatch):
    script = {1: "x^{2} - 5x + 6 = 0", 2: "x^{2} - 6 = 5x",
              3: "x^{2} - 5x = 6", 4: r"x = 2 \text{ 或 } x = 3"}
    pc = _run(tmp_path, monkeypatch, script, script)
    gallery = pc.gallery()
    assert len(gallery) == len(pc.lines)
    assert all(Path(p).exists() for p, _ in gallery)


def test_photo_missing_line_leaves_placeholder_and_no_accusation(tmp_path, monkeypatch):
    """认不出的一行留 `?`，不让它自己变成"你错了"，也不让行错位。"""
    a = {1: "x^{2} - 5x + 6 = 0", 3: "x^{2} - 5x = 6", 4: r"x = 2"}
    b = dict(a)
    pc = _run(tmp_path, monkeypatch, a, b)
    assert "?" in pc.text.split("\n"), "认不出要留占位，不能让后面的行错位"

    # 那一行自己必须是"读不出"，不是"不成立"
    hole = next(s for s in pc.work.steps if s.idx == 2)
    assert hole.verdict == "unknown", "没认出来的行不能变成错误结论"
    # 隔着窟窿比出来的第 3 步，结论必须撤回
    after = next(s for s in pc.work.steps if s.idx == 3)
    assert after.verdict == "suspect"
    assert any("没认" in n for n in pc.notes)


def test_photo_all_lines_unsure_gives_no_verdict(tmp_path, monkeypatch):
    """所有行都只有一家认出来 → 整份不判。"""
    a = {1: "x^{2} - 5x + 6 = 0", 2: "x^{2} - 6 = 5x",
         3: "x^{2} - 5x = 6", 4: r"x = 2"}
    b: dict[int, str] = {}
    pc = _run(tmp_path, monkeypatch, a, b)
    assert pc.work.grade == "unread"
    assert not any(s.verdict == "wrong" for s in pc.work.steps)


def test_photo_single_source_line_is_not_judged(tmp_path, monkeypatch):
    """只有一家认出某行时，那一行不能拿来判错。"""
    a = {1: "x^{2} - 5x + 6 = 0", 2: "x^{2} - 6 = 5x",
         3: "x^{2} - 5x = 6", 4: r"x = 2"}
    b = {1: "x^{2} - 5x + 6 = 0", 3: "x^{2} - 5x = 6", 4: r"x = 2"}  # 第 2 行只有 a 读出来
    pc = _run(tmp_path, monkeypatch, a, b)
    line2 = next(x for x in pc.lines if x.index == 2)
    assert line2.verdict == "review"
    # 拿没认准的第 2 行当依据的判定，两个方向都必须撤回
    step2 = next(s for s in pc.work.steps if s.idx == 2)
    assert step2.verdict == "suspect"
    step3 = next(s for s in pc.work.steps if s.idx == 3)
    assert step3.verdict == "unknown"


# ---------------------------------------------------------------- 行首残影与说明行
#
# 下面这批用例全部来自 P0 八页里真实读错的行。每一条都对应一次"字读对了、
# 清洗没跟到底"或"两台机器措辞不同、式子其实一样"，不是假想的边界。


def test_readable_drops_broken_arrow_digit():
    """`⇒` 被读成 `=7`：行首不可能有等号，那个数字是箭头下半截的影子。"""
    assert readable(r"=7\ x=2\text{ 或 }x=3") == "x=2 或 x=3"
    assert readable("=7 x=2 或 x=3") == "x=2 或 x=3"


def test_readable_keeps_digit_glued_to_variable():
    """换行接着写的 `= 2x+1` 里，2 是系数，与 x 相连，不能当影子剥掉。"""
    assert readable("= 2x+1") == "= 2x+1"
    assert readable("=2x+1") == "=2x+1"


def test_readable_drops_leading_closing_bracket():
    """学生用大括号把几行括在一起，切行时右半边落进了下一行开头。"""
    assert readable(")x^{2}-6=5x") == "x^{2}-6=5x"


def test_canon_dangling_equals_means_zero():
    """一行写到最后一个字符时模型常漏掉它：`...=` 与 `...=0` 是同一件事。"""
    assert _canon("2x^2-10x+12=") == _canon("2x^2-10x+12=0")
    assert _canon("x=") == _canon("x=0")


def test_vote_wording_difference_is_corroborated():
    """措辞不同、式子一样时，旁证要按式子数，不能只数赢的那一组。"""
    _, conf, _, verdict, notes = _vote([
        _c("所以：(x-2)(x-3)=0", "qwen3-vl"),
        _c("所求点：(x-2)(x-3)=0", "glm-4.6v-flash"),
    ])
    assert verdict == "ok"
    assert conf >= photocheck.UNREAD_CONF
    assert any("措辞" in n for n in notes)


def test_prose_line_is_not_counted_as_a_hole():
    """说明行认成"有边"还是"两边"不影响任何结论，不该拖累整页。"""
    cands = [
        _c("有边同时乘2得：", "qwen3-vl"),
        Candidate(latex=r"\frac{AB}{BC}=\frac{AC}{BD}", source="glm-4.6v-flash", weak=True),
    ]
    assert photocheck._prose_ok("有边同时乘2得：", cands)
    lr = LineRead(index=2, latex="有边同时乘2得：", text="有边同时乘2得：",
                  verdict="prose", confidence=0.5)
    assert lr.sure


def test_prose_rule_does_not_swallow_a_real_equation():
    """主模型读出了式子，就不能拿"说明行"放过。"""
    cands = [_c("2x-6=12", "qwen3-vl"),
             Candidate(latex="两边同时乘2得", source="glm-4.6v-flash", weak=True)]
    assert not photocheck._prose_ok("2x-6=12", cands)


def _lr(index, latex, verdict="ok", alts=()):
    return LineRead(index=index, latex=latex, text=readable(latex),
                    verdict=verdict, confidence=1.0 if verdict == "ok" else 0.5,
                    alternatives=list(alts))


def test_tiebreak_looks_down_when_no_line_above():
    """第一行没有上一行可比，但下面那行能对上——证据在下面，就该用它。"""
    lines = [
        _lr(1, "x-5x+6=0", "review", alts=["x^2-5x+6=0"]),
        _lr(2, "有边同时乘2得：", "prose"),
        _lr(3, "2x^{2}-10x+12=0"),
    ]
    photocheck._tiebreak_by_math(lines)
    assert lines[0].verdict == "ok"
    assert lines[0].latex == "x^2-5x+6=0"


def test_tiebreak_accepts_two_readings_that_are_the_same_equation():
    """两种写法都与上一行同解，对批改等价，判 ok 而不是判"裁不了"。"""
    lines = [
        _lr(1, "x^{2}-5x+6=0"),
        _lr(2, "-2(x-2)(x-3)=0", "review", alts=["(x-2)(x-3)=0"]),
    ]
    photocheck._tiebreak_by_math(lines)
    assert lines[1].verdict == "ok"


def test_tiebreak_leaves_genuine_errors_alone():
    """两派都接不上上一行，说明这一步本来就错，不能硬裁。"""
    lines = [
        _lr(1, "x^{2}-5x+6=0"),
        _lr(2, "(x-2)(x-5)=0", "review", alts=["(x-2)(x-5)=0"]),
    ]
    photocheck._tiebreak_by_math(lines)
    assert lines[1].verdict == "review"


def test_leading_digit_accusation_is_softened():
    """`⇒` 被读成 2：去掉行首系数后与上一行同解，就不判这一行错。"""
    work = grade(["x^{2}-5x+6=0", "x^{2}-6=5x", "2x^{2}-5x=6", "x=2 或 x=3"])
    photocheck._soften_digit_accusations(work)
    step3 = next(s for s in work.steps if s.idx == 3)
    assert step3.verdict != "wrong"
    assert not step3.counter, "反例建立在可疑的数字上，必须一起撤掉"


def test_leading_digit_does_not_excuse_a_real_coefficient_error():
    """真的写错系数时，去掉系数也接不上上一行，照常判错。"""
    work = grade(["x^{2}=4", "2x^{2}=9"])
    photocheck._soften_digit_accusations(work)
    assert any(s.verdict == "wrong" for s in work.steps)


# ---------------------------------------------------------------- 行尾编号与残片
#
# 学生给每一步编号是常态（"③-①得：5y=10 ④"），编号不是数学内容。
# 留着它的代价比想象中大：两个来源对同一行各读各的、本该由「按数学一致性裁决」
# 按"哪个接得上邻行"裁掉——而它要先把候选解析成表达式。
# 行尾多个 ②，`x+3y=7②` 就解析不了，裁决静默失效，整行退回"待核对"。


@pytest.mark.parametrize("raw, want", [
    ("x+3y=7\\quad\\text{②}", "x+3y=7"),
    ("x+3y=7\\quad(2)", "x+3y=7"),
    ("y = 2 \\textcircled{5}", "y = 2"),
    ("2. x + 6y = 14 \\tag{3}", "2. x + 6y = 14"),
    ("(3)-(1)\\text{得}:\\ 5y=10\\ (4)", "(3)-(1)得: 5y=10"),
    ("`7y=2`", "7y=2"),
])
def test_tail_marks_are_stripped(raw, want):
    assert readable(raw) == want


def test_bracketed_number_that_is_content_is_kept():
    """`x=2(3)` 里的 (3) 是乘 3，剥掉就把答案从 6 改成 2 了。"""
    assert readable("x=2(3)") == "x=2(3)"
    assert readable("(x-2)(x-3)=0") == "(x-2)(x-3)=0"


def test_lone_leading_one_is_dropped():
    """行首孤立的系数 1 与式子等价；它多半是上一行 `}` 的右半截被读成了数字。"""
    assert readable("1y=2") == "y=2"
    assert readable("1x+3y=7") == "x+3y=7"
    # 但不能碰这些：1 不是孤立的系数
    assert readable("12x=4") == "12x=4"
    assert readable("1.5x=3") == "1.5x=3"
    assert readable("1 = 2") == "1 = 2"


# ---------------------------------------------------------------- 行尾编号与残片
#
# 学生给每一步编号是常态（"③-①得：5y=10 ④"），编号不是数学内容。
# 留着它的代价比想象中大：两个来源对同一行各读各的、本该由「按数学一致性裁决」
# 按"哪个接得上邻行"裁掉——而它要先把候选解析成表达式。
# 行尾多个 ②，`x+3y=7②` 就解析不了，裁决静默失效，整行退回"待核对"。


@pytest.mark.parametrize("raw, want", [
    (r"x+3y=7\quad\text{②}", "x+3y=7"),
    (r"x+3y=7\quad(2)", "x+3y=7"),
    (r"y = 2 \textcircled{5}", "y = 2"),
    (r"2. x + 6y = 14 \tag{3}", "2. x + 6y = 14"),
    (r"(3)-(1)\text{得}:\ 5y=10\ (4)", "(3)-(1)得: 5y=10"),
    ("`7y=2`", "7y=2"),
])
def test_tail_marks_are_stripped(raw, want):
    assert readable(raw) == want


def test_bracketed_number_that_is_content_is_kept():
    """`x=2(3)` 里的 (3) 是乘 3，剥掉就把答案从 6 改成 2 了。"""
    assert readable("x=2(3)") == "x=2(3)"
    assert readable("(x-2)(x-3)=0") == "(x-2)(x-3)=0"


def test_lone_leading_one_is_dropped():
    """行首孤立的系数 1 与式子等价；它多半是上一行 `}` 的右半截被读成了数字。"""
    assert readable("1y=2") == "y=2"
    assert readable("1x+3y=7") == "x+3y=7"
    # 但不能碰这些：1 不是孤立的系数
    assert readable("12x=4") == "12x=4"
    assert readable("1.5x=3") == "1.5x=3"
    assert readable("1 = 2") == "1 = 2"


# ---------------------------------------------------------------- 手动补识别
#
# 识别不可能百分之百。这里的做法是：切行必须切全（缺行是静默失败，见 test_page_split），
# 但认不出的行**允许认不出**，由用户在预填文本里改成自己写的式子，再判一次。
#
# 关键的一条不是"能改"，而是**没改的行不许被顺手确认**：
# `y=2` 被读成 `7y=2` 时，用户只改了别的行、这一行原样留着，要是当成他写的，
# 接下来判出来的"不成立"全是冤案。
#
# 拆成三条判据，缺一条这套机制就会说谎：
#   1. 改了 → 按用户给的式子判，并在结论里点名"这行是手输的"；
#   2. 没改也没勾 → 保持系统原来的把握（窟窿还是窟窿）；
#   3. 没改但勾了核对 → 按用户确认的读法判（原文本来就对，只是来源没谈拢）。


def _page(lines, problem=""):
    """按 (行号, 判定, 把握, 文本) 造一份识别结果，再照真实链路判一次。"""
    from mathagent.workcheck import grade

    reads = [
        LineRead(index=i, latex=t, text=t, verdict=v, confidence=c, sources=["qwen"])
        for i, v, c, t in lines
    ]
    return PhotoCheck(
        work=grade([r.text for r in reads], problem=problem), lines=reads, page="p.jpg"
    )


SYSTEM_PAGE = [
    (1, "review", 0.5, "2x+y=4"),
    (2, "ok", 1.0, "x+3y=7"),
    (3, "prose", 0.5, "②两边同时乘2得："),
    (4, "ok", 0.8, "2x+6y=14"),
    (5, "review", 0.5, "5y=10"),
    (6, "ok", 1.0, "y = 2"),
    (7, "prose", 0.5, "将①代入②得："),
    (8, "ok", 0.8, "x=1"),
    (9, "review", 0.5, "7 y = 2"),
]


def test_edit_text_keeps_the_original_line_numbers():
    """行号是回来对齐用的：改完只剩八行，系统还得知道哪行是原来的第 10 行。"""
    pc = _page(SYSTEM_PAGE)
    text = photocheck.edit_text(pc)
    assert text.splitlines()[0] == "1. 2x+y=4"
    assert text.splitlines()[-1] == "9. 7 y = 2"


def test_manual_fix_turns_a_hole_into_a_judged_line():
    pc = _page(SYSTEM_PAGE)
    fixed = photocheck.edit_text(pc).replace("9. 7 y = 2", "9. y=2")
    out = photocheck.regrade(pc, fixed, confirm={1, 5})
    assert out.work.grade == "pass", out.work.verdict
    last = [s for s in out.work.steps if s.idx == 9][0]
    assert last.verdict == "ok", last.row
    joined = " ".join(out.notes)
    assert "第 9 行是你手动输入的" in joined


def test_untouched_unsure_line_stays_a_hole():
    """只改一行，别的没认准的行不许跟着变成"已确认"。"""
    pc = _page(SYSTEM_PAGE)
    fixed = photocheck.edit_text(pc).replace("9. 7 y = 2", "9. y=2")
    out = photocheck.regrade(pc, fixed)
    assert out.work.grade == "unread", out.work.headline
    assert [lr.index for lr in out.lines if not lr.sure] == [1, 5]


def test_confirmed_line_is_trusted_without_editing():
    """原文本来就对、只是两个来源没谈拢：勾一下就该按它算。"""
    pc = _page(SYSTEM_PAGE)
    fixed = photocheck.edit_text(pc).replace("9. 7 y = 2", "9. y=2")
    out = photocheck.regrade(pc, fixed, confirm={1, 5})
    assert out.work.grade == "pass"
    assert "第 1、5 行你核对过、没改动" in " ".join(out.notes)


def test_manual_line_is_never_dressed_up_as_recognized():
    """手输的行不能标成系统认出来的——那是把用户的判断算成自己的功劳。"""
    pc = _page(SYSTEM_PAGE)
    fixed = photocheck.edit_text(pc).replace("9. 7 y = 2", "9. y=2")
    out = photocheck.regrade(pc, fixed, confirm={1, 5})
    touched = [lr for lr in out.lines if photocheck.MANUAL_SOURCE in lr.sources]
    assert [lr.index for lr in touched] == [9]
    assert touched[0].text == "y=2"


def test_manual_fix_of_the_problem_line_moves_the_criterion():
    """改的正好是题面那一行时，判定必须按改过的题算。

    这是手改最容易踩的一脚：题面被认成 `x²-5x+7=0`，学生写 `x=2`、`x=3`。
    用户把题面改成 `x²-5x+6=0`——那学生全对。可如果判定还捧着旧题面，
    结论会是「过程成立，但解不全，少了两个复根」：改对了反而判得更离谱。
    """
    pc = _judge(
        [
            LineRead(index=1, text="x^2-5x+7=0", verdict="ok", confidence=0.9, sources=["qwen"]),
            LineRead(index=2, text="x=2 或 x=3", verdict="ok", confidence=0.9, sources=["qwen"]),
        ],
        problem="",
        page="p.jpg",
    )
    assert pc.problem_line == 1
    assert pc.work.grade == "wrong"

    fixed = photocheck.edit_text(pc).replace("x^2-5x+7=0", "x^2-5x+6=0")
    out = photocheck.regrade(pc, fixed)
    assert out.work.problem == "x^2-5x+6=0", out.work.problem
    assert out.work.grade == "pass", (out.work.grade, out.work.headline)
    # 题面那一行是用户给的，结论里必须点名它同时充当了判定基准
    assert "同时被当作题面" in " ".join(out.notes)


def test_problem_line_keeps_tracking_across_two_rounds():
    """连改两次也要跟得上：第二次不能又退回第一次的题面。"""
    pc = _judge(
        [
            LineRead(index=1, text="x^2-5x+7=0", verdict="ok", confidence=0.9, sources=["qwen"]),
            LineRead(index=2, text="x=2 或 x=3", verdict="ok", confidence=0.9, sources=["qwen"]),
        ],
        problem="",
        page="p.jpg",
    )
    one = photocheck.regrade(pc, photocheck.edit_text(pc).replace("x^2-5x+7=0", "x^2-5x+6=0"))
    two = photocheck.regrade(one, photocheck.edit_text(one).replace("x^2-5x+6=0", "x^2-5x+8=0"))
    assert two.work.problem == "x^2-5x+8=0", two.work.problem
    assert two.work.grade != "pass"


def test_deleting_the_problem_line_keeps_the_old_problem_and_says_so():
    """题面那一行被删掉：没有新题面可依，退回原题面，但必须说明。"""
    pc = _judge(
        [
            LineRead(index=1, text="x^2-5x+7=0", verdict="ok", confidence=0.9, sources=["qwen"]),
            LineRead(index=2, text="x=2 或 x=3", verdict="ok", confidence=0.9, sources=["qwen"]),
        ],
        problem="",
        page="p.jpg",
    )
    out = photocheck.regrade(pc, "2. x=2 或 x=3")
    assert out.work.problem == "x^2-5x+7=0"
    assert "原来是题面" in " ".join(out.notes)


def test_explicit_problem_still_wins_over_the_misread_line():
    """用户在题面输入框里填了题，就不该再被某一行抢过去。"""
    pc = _judge(
        [
            LineRead(index=1, text="x^2-5x+7=0", verdict="ok", confidence=0.9, sources=["qwen"]),
            LineRead(index=2, text="x=2 或 x=3", verdict="ok", confidence=0.9, sources=["qwen"]),
        ],
        problem="x^2-5x+6=0",
        page="p.jpg",
    )
    out = photocheck.regrade(pc, photocheck.edit_text(pc))
    assert out.work.problem == "x^2-5x+6=0"
    # 页面上那一行仍是认错的题面抄写，不该被当成新题面顶掉用户填的题
    assert any(lr.text == "x^2-5x+7=0" for lr in out.lines)


def test_unread_problem_line_withdraws_the_whole_verdict():
    """题面没认准时不许下结论。

    标准解和每一处"和题对不上"都是相对题面算的。题面悬空，"你这一步错了"
    就只是一串看着确定的猜测；"伪推导"这种指控更是完全靠标准解撑着。
    """
    pc = _judge(
        [
            LineRead(index=1, text="x^2-5x+7=0", verdict="review", confidence=0.5, sources=["qwen"]),
            LineRead(index=2, text="x=2 或 x=3", verdict="ok", confidence=0.9, sources=["qwen"]),
        ],
        problem="",
        page="p.jpg",
    )
    assert pc.problem_line == 1
    assert pc.work.grade == "unread", pc.work.headline
    assert "题面" in pc.work.headline
    assert pc.work.expected == [] and pc.work.error_type == ""
    assert not any(s.verdict in ("wrong", "partial") for s in pc.work.steps)
    assert "题面" in " ".join(pc.notes)


def test_confident_problem_line_keeps_normal_grading():
    """题面认准了，该判的照判——这条护栏不能变成"一律不判"。"""
    pc = _judge(
        [
            LineRead(index=1, text="x^2-5x+6=0", verdict="ok", confidence=0.95, sources=["qwen"]),
            LineRead(index=2, text="x=2 或 x=3", verdict="ok", confidence=0.95, sources=["qwen"]),
        ],
        problem="",
        page="p.jpg",
    )
    assert pc.work.grade == "pass", pc.work.headline


def test_placeholder_left_alone_is_still_unread():
    """留着 `?` 就是承认没读出来，不能当成用户给了内容。"""
    pc = _page([(1, "ok", 1.0, "x+1=3"), (2, "failed", 0.0, "")])
    out = photocheck.regrade(pc, "1. x+1=3\n2. ?")
    assert out.work.grade == "unread"


def test_text_without_line_numbers_falls_back_to_position():
    """用户把行号删干净了，就按顺序对应——总比全部当成新行更接近他的本意。"""
    pc = _page([(1, "ok", 1.0, "x+1=3"), (2, "review", 0.5, "x=2")])
    out = photocheck.regrade(pc, "x+1=3\nx=2", confirm={2})
    assert out.work.grade != "unread"
    assert out.lines[1].sure


def test_deleted_line_is_reported():
    pc = _page(SYSTEM_PAGE)
    text = "\n".join(
        ln for ln in photocheck.edit_text(pc).splitlines() if not ln.startswith("9. ")
    )
    out = photocheck.regrade(pc, text, confirm={1, 5})
    assert any("第 9 行没出现在你给的过程里" in n for n in out.notes)


def test_regrade_does_not_mutate_the_original_result():
    """重判要能反复点：改一次再改一次，不能把上一轮的结果叠进去。"""
    pc = _page(SYSTEM_PAGE)
    before = [(lr.index, lr.text, lr.verdict) for lr in pc.lines]
    photocheck.regrade(pc, photocheck.edit_text(pc).replace("9. 7 y = 2", "9. y=2"))
    after = [(lr.index, lr.text, lr.verdict) for lr in pc.lines]
    assert before == after
