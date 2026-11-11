"""入口判断：这张图该走公式链路，还是该转去几何链路。

守的是一次真实的误投：用户把「四边形 ABCD 为正方形…求证 CE=CF」的照片
传到了公式页签，识别本身读得好好的，最后却蹦出一句「等号数量异常：2 个」。
错的不是识别，是入口——而界面上那句话让用户以为系统坏了。
"""

from __future__ import annotations

from mathagent import pipeline
from mathagent.latex import parse
from mathagent.route import detect, plain

GEO_PAGE = (
    r"\text{第一题：}\\\text{如图，四边形 ABCD 为正方形，DE \parallel AC，AE = AC，"
    r"AE \text{ 与 } CD \text{ 相交于 } F。}\\\text{求证：} CE = CF\text{。}"
)


def test_geometry_page_is_routed_not_parsed():
    """整页几何题要判成 geometry，并且不装模作样地去解析。"""
    r = detect(GEO_PAGE)
    assert r.kind == "geometry", r
    assert "求证" in r.hints and "四边形" in r.hints

    pe = parse(GEO_PAGE)
    assert not pe.ok
    assert pe.kind == "geometry"
    assert "几何题" in pe.error


def test_angle_formula_is_not_mistaken_for_geometry():
    """含 ∠ 的纯算式不能被误伤——这是最容易误判的一类。"""
    assert detect(r"\angle A + \angle B = 180^\circ").kind == "formula"
    assert detect(r"x^2-5x+6=0").kind == "formula"


def test_word_problem_is_prose():
    """应用题是文字，但不算几何题；两种「不是算式」要分开说。"""
    r = detect("小明买了3个苹果共花15元，每个苹果多少元？")
    assert r.kind == "prose"
    assert not r.is_geometry


def test_pipeline_marks_route_and_skips_solving():
    """识别出的东西不是算式时，结论只有一句：换链路，别硬算。"""
    res = pipeline.run(latex=GEO_PAGE, confirmed=True, save=False)
    assert res.route == "geometry"
    assert res.verification.verdict == "routed"
    assert not res.solved.ok
    assert "几何题" in res.verification.reasons[0]


def test_plain_reads_the_recognized_text():
    """界面上要贴人话，不能把 \\text{} 和反斜杠原样甩出来。"""
    txt = plain(GEO_PAGE)
    assert "\\text" not in txt
    assert "四边形" in txt and "∥" in txt
    assert txt.count("\n") >= 2      # 原来的 \\ 换行要还原成换行


def test_route_survives_empty_choice():
    """投票全废（chosen 为空）时，仍要能从最长的候选里认出这是几何题。

    实测就是这样：整页题被判「无法解析」，chosen 是空串，
    只看 chosen 的话永远发现不了投错门。
    """
    res = pipeline.run(latex=GEO_PAGE, confirmed=True, save=False)
    # confirmed 路径下 chosen 非空；这里直接验证兜底函数本身
    from mathagent.pipeline import _longest
    from mathagent.types import Candidate

    cands = [Candidate(latex="", source="a"), Candidate(latex=GEO_PAGE, source="b")]
    assert _longest(cands) == GEO_PAGE
    assert detect(_longest(cands)).kind == "geometry"
    assert res.route == "geometry"
