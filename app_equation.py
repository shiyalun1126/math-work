"""方程类：一个题型，两条路，做透它。

跑法：F:/PyCharm/python.exe app_equation.py

只做方程：一元一次、一元二次、二元一次方程组（分式方程 / 无理方程算，
但自动验根）。别的题型不接——不是没做，是刻意不做，理由写在
`mathagent/equations.py` 开头。界面上也把这条边界写在明面上。

两条路：
  ① 搜答案：上传题目（照片或题干文字）→ 抽出方程 → 求解 → 答案 + 验算；
  ② 批改过程：过程（照片或逐行文字）→ 逐行判定 → 反例 + 结论。
两条路共用同一套题型判定：出范围就一起出范围，不会出现"搜答案敢答、
批改不敢判"这种前后不一致。
"""

from __future__ import annotations

import os

import gradio as gr

from mathagent import equations, photocheck, workcheck
from mathagent.types import WorkCheck

LATEX_DELIMS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]

# ---------------------------------------------------------------- 样式

_COVER = {
    "in": ("#E1F5EE", "#1D9E75", "#04342C", "覆盖范围内"),
    "edge": ("#FAEEDA", "#EF9F27", "#412402", "能答，但要交代边界"),
    "out": ("#FCEBEB", "#E24B4A", "#501313", "不在覆盖范围内"),
}

_GRADE = {
    "pass": ("#E1F5EE", "#1D9E75", "#04342C", "过程成立"),
    "partial": ("#FAEEDA", "#EF9F27", "#412402", "过程成立，但解不全"),
    "fake": ("#FBEAF0", "#D4537E", "#4B1528", "过程不成立，答案却是对的"),
    "wrong": ("#FCEBEB", "#E24B4A", "#501313", "过程有错"),
    "unread": ("#EEF1F6", "#7A8CA6", "#22303F", "这几行没认准，先不判"),
    "unknown": ("#F6F5F1", "#B4B2A9", "#2C2C2A", "只能做过程自检"),
    "gap": ("#F6F5F1", "#B4B2A9", "#2C2C2A", "没有可批改的式子"),
}

_MARK = {"ok": "✓ 通过", "wrong": "✗ 不成立", "partial": "△ 漏解",
         "suspect": "? 待核对", "note": "—", "unknown": "? 跳过"}

CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto;}
#head h1 {font-size: 22px; margin: 0 0 2px 0;}
#head p {color: #5F5E5A; font-size: 13px; margin: 0 0 6px 0;}
.katex-display {text-align: left !important; margin: 2px 0 8px 0 !important;}
.katex-display > .katex {text-align: left !important;}
#steps p {margin: 2px 0 6px 0; line-height: 1.5;}
#steps h3 {margin: 10px 0 4px 0; font-size: 15px;}
#answer .katex-display > .katex {font-size: 1.45em;}
"""

IDLE = """
<div style="background:#F6F5F1;border:1px solid #E4E2DB;border-radius:12px;padding:14px 16px;">
  <div style="font-size:15px;font-weight:600;color:#4A4843;">这件事我只做方程</div>
  <div style="font-size:13px;color:#6B6A65;margin-top:4px;line-height:1.7;">
    一元一次方程 · 一元二次方程 · 二元一次方程组 · 含参数方程（按参数分情形）。
    分式方程和无理方程能解，但会自动验根、把增根剔掉。
    三次以上、三角指数对数、不等式、几何——不接。
  </div>
</div>"""

HEAD = """
<div id="head">
  <h1>方程 · 一个题型做透</h1>
  <p>题目进 → 答案出；过程进 → 判定出。答案每一步都由 CAS 验算，
     认不准的行不会算你错。</p>
</div>"""


# ---------------------------------------------------------------- 搜答案

def _cover_badge(ans) -> str:
    t = ans.topic
    bg, border, fg, label = _COVER.get(t.coverage, _COVER["out"])
    detail = t.label
    if t.vars:
        detail += f" · 未知数 {'、'.join(t.vars)}"
    extra = ans.note or t.note or ans.error or ""
    prov = "" if ans.provided else (
        '<div style="margin-top:8px;padding:7px 10px;border-radius:8px;'
        'background:#fff8e6;border:1px solid #f0d9a8;font-size:12.5px;color:#5a4a20;">'
        '这个方程是<b>模型从题面文字转写</b>的，不是原题直接给的形式。'
        '下面的解是按它算的——先看一眼方程对不对。</div>'
    )
    return f"""
<div style="background:{bg};border:1px solid {border};border-radius:12px;padding:14px 16px;">
  <div style="font-size:15px;font-weight:600;color:{fg};">{label} · {detail}</div>
  <div style="font-size:13px;color:{fg};opacity:.9;margin-top:4px;">{extra}</div>
  {prov}
</div>"""


def _answer_md(ans) -> str:
    if ans.error and not ans.ok:
        return f"### 不给结论\n\n{ans.error}"
    if not ans.ok:
        return ("### 没算出来\n\n"
                f"{ans.error or 'CAS 没给出解'}。\n\n"
                "> 这种情况不硬给答案：解不出多半是式子里有读错的符号，"
                "把方程改对再算一次更快。")
    out = ["### 答案", ""]
    if ans.branches:
        # 含参：一支一行。压成 "分类讨论：①…②…" 一行，最容易漏看的正是被压掉的那一支。
        out.append("**按参数分情形**")
        out.append("")
        out += [f"$${b}$$" for b in ans.branches]
        out += ["", "> 每一支都取了一组实际参数值代回验过；"
                    "取不到值的情形（根本不存在）不会出现在这里。"]
    else:
        out.append(f"$${ans.answer_latex}$$")
        if ans.checks:
            out += ["", "**验算一下**", ""]
            out += [f"- {c}" for c in ans.checks]
    if ans.dropped:
        out += ["", "**这些解要划掉**", ""] + [f"- {d}" for d in ans.dropped]
    return "\n".join(out)


def _steps_md(ans) -> str:
    if not ans.steps:
        return ""
    out = ["### 怎么做", ""]
    for i, st in enumerate(ans.steps, 1):
        out.append(f"**{i}. {st['title']}**\n\n$${st['latex']}$$")
    body = "\n\n".join(out)
    if ans.explain:
        body += ("\n\n### 讲评\n\n" + ans.explain
                 + "\n\n> 讲评只讲思路；答案和每一步的数都是程序真算出来的，不是模型猜的。")
    return body


def _meta_md(meta: dict | None, ans) -> str:
    """给用户看的说明。这里只放**对他有用**的话。

    耗时、内部分步计时这类东西属于运维日志，摆在解题结果下面只会让人分心——
    学生关心的是"这答案靠不靠得住"，不关心 solve 花了 0.00 秒。
    完整数据仍在下面的 JSON 里，调试时照样查得到。
    """
    bits = []
    if meta:
        src = "、".join(meta.get("sources") or []) or "无"
        conf = meta.get("confidence", 0) or 0
        if conf >= 0.9:
            bits.append(f"照片识别可信度 {conf:.0%}（{src}）")
        else:
            bits.append(
                f"照片识别可信度只有 {conf:.0%}（{src}）——"
                "建议对一下上面的方程有没有读错"
            )
        if meta.get("read"):
            bits.append(f"读到的是：`{meta['read'][:80]}`")
    if ans.cached:
        bits.append("这道题之前算过，直接给的结论")
    return "\n\n".join(f"> {b}" for b in bits)


def on_search(image, text, allow_setup, use_explain, progress=gr.Progress()):
    t = (text or "").strip()
    if not t and not image:
        raise gr.Error("先上传题目照片，或者把题干 / 方程贴进左边的框")
    meta = None
    try:
        if t:
            progress(0.3, desc="抽方程")
            ans = equations.answer(t, allow_setup=allow_setup, explain=use_explain)
        else:
            progress(0.2, desc="读题")
            ans, meta = equations.answer_image(image, allow_setup=allow_setup,
                                               explain=use_explain)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
    progress(1.0, desc="完成")
    return (
        _cover_badge(ans),
        ans.latex or "",
        _answer_md(ans),
        _steps_md(ans),
        _meta_md(meta, ans),
        ans.to_dict(),
    )


def on_resolve(latex: str, use_explain: bool):
    if not (latex or "").strip():
        raise gr.Error("方程是空的")
    ans = equations.answer(latex, allow_setup=False, explain=use_explain,
                           source="manual")
    return (
        _cover_badge(ans),
        ans.latex or latex,
        _answer_md(ans),
        _steps_md(ans),
        _meta_md(None, ans),
        ans.to_dict(),
    )


# ---------------------------------------------------------------- 批改过程

def _grade_badge(check: WorkCheck, scope_note: str = "") -> str:
    bg, border, fg, title = _GRADE.get(check.grade, _GRADE["unknown"])
    # 错后面的步骤即使"局部成立"也不算通过：它们是建立在错的前提上的，
    # 把这一步数进"通过 1 步"，学生会以为答案那行被判对了。
    passed = [
        s for s in check.steps
        if s.verdict == "ok" and (check.first_error <= 0 or s.idx < check.first_error)
    ]
    sub = f"查了 {check.n_checked} 步，通过 {len(passed)} 步"
    local = [
        s for s in check.steps
        if s.verdict == "ok" and check.first_error > 0 and s.idx > check.first_error
    ]
    if local:
        sub += f"（第 {'、'.join(str(s.idx) for s in local)} 步只是局部成立）"
    if check.first_error > 0:
        sub += f" · 第 {check.first_error} 步起不成立"
    if check.error_type:
        sub += f" · 错因：{check.error_type}"
    warn = ""
    if scope_note:
        warn = ('<div style="margin-top:8px;padding:7px 10px;border-radius:8px;'
                'background:#fff8e6;border:1px solid #f0d9a8;font-size:12.5px;'
                f'color:#5a4a20;">{scope_note}</div>')
    return f"""
<div style="background:{bg};border:1px solid {border};border-radius:12px;padding:14px 16px;">
  <div style="font-size:15px;font-weight:600;color:{fg};">{title}</div>
  <div style="font-size:13px;color:{fg};opacity:.9;margin-top:4px;">{check.headline}</div>
  <div style="font-size:12.5px;color:{fg};opacity:.75;margin-top:6px;">{sub}</div>
  {warn}
</div>"""


def _cell(text: str) -> str:
    return (text or "").replace("|", r"\|").replace("\n", " ")


def _table_md(check: WorkCheck) -> str:
    if not check.steps:
        return "_（没有读到任何一行）_"
    out = ["| # | 你写的 | 判定 | 依据 / 反例 |", "|---|---|---|---|"]
    for s in check.steps:
        if s.kind == "prose":
            body = s.raw
        elif s.kind == "assign" and s.raw:
            body = f"${s.raw}$"
        else:
            body = f"${s.latex}$"
        mark = s.row[0] if s.row else _MARK.get(s.verdict, s.verdict)
        out.append(f"| {s.idx} | {_cell(body)} | {mark} | {_cell(s.counter or s.reason)} |")
    return "\n".join(out)


def _advice_md(check: WorkCheck) -> str:
    if not check.advice and not check.notes:
        return ""
    out = ["### 错在哪，怎么改", "", check.advice or ""]
    if check.expected:
        out.append(f"**标准答案**：{'、'.join(check.expected)}")
    if check.got:
        out.append(f"**你写的**：{'、'.join(check.got)}")
    if check.notes:
        out += ["", "\n".join(f"> {n}" for n in check.notes)]
    return "\n\n".join(x for x in out if x)


def _scope_note(problem: str) -> str:
    """题面出了方程范围，逐行结论就只是自检，不能当"你错了"。"""
    if not (problem or "").strip():
        return ""
    eq = equations.extract(problem)
    if eq.topic.coverage == "out":
        return (f"题面判为「{eq.topic.label}」，不在方程范围内——"
                "下面逐行只做过程内部一致性检查，别把它当最终判定。")
    return ""


def on_grade_text(problem, work):
    if not (work or "").strip() and not (problem or "").strip():
        raise gr.Error("至少给一道题，或者贴入一段过程")
    try:
        check = workcheck.grade(work or "", problem=problem or "")
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
    return (
        _grade_badge(check, _scope_note(problem)),
        _table_md(check),
        _advice_md(check),
        check.to_dict(),
    )


def _reads_md(pc) -> str:
    if not pc.lines:
        return "### 逐行识别\n\n_（这一页没有切出任何一行）_"
    out = ["### 逐行识别", "", pc.read_table_md()]
    bad = [x for x in pc.lines if not x.sure]
    if bad:
        out += ["", f"> 第 {'、'.join(str(x.index) for x in bad)} 行没认准。"
                    "它们在批改表里是「? 待核对」，不产生对错结论。"]
    return "\n".join(out)


def on_grade_photo(problem, image, progress=gr.Progress()):
    if not image:
        raise gr.Error("先上传一张过程照片")
    progress(0.15, desc="切行")
    try:
        pc = photocheck.check(image, problem=problem or "")
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
    progress(1.0, desc="完成")
    return (
        _grade_badge(pc.work, _scope_note(problem or pc.work.problem)),
        _table_md(pc.work),
        _advice_md(pc.work),
        pc.text,
        pc.lines_html(),
        _reads_md(pc),
        pc.to_dict(),
    )


# ---------------------------------------------------------------- 界面

SEARCH_EXAMPLES = [
    ["解方程：3x+6=15"],
    ["x^2-5x+6=0"],
    ["解方程组：x+y=3, x-y=1"],
    ["mx^2+2x+1=0"],
]

GRADE_EXAMPLES = [
    ["x^2-5x+6=0", "x^2-5x+6=0\nx^2-6=5x\nx^2-5x=6\nx=2 或 x=3"],
    ["x^2-5x+6=0", "x^2-5x+6=0\n两边同乘2得 2x^2-10x+12=0\n所以 (x-2)(x-3)=0\nx=2"],
    ["2x-6=12", "2x-6=12\n2x=12\nx=6"],
    ["x+y=5, x-y=1", "x+y=5\nx-y=1\n2x=6\nx=3\ny=2\n所以 x=3，y=2"],
    ["x+y=5, x-y=1", "x+y=5\nx-y=3\n2x=8\nx=3\ny=2"],
]


with gr.Blocks(title="方程 · 一个题型做透") as demo:
    gr.HTML(HEAD)

    with gr.Tabs():
        with gr.Tab("① 搜答案 · 题目 → 答案"):
            with gr.Row():
                with gr.Column(scale=4):
                    image = gr.Image(label="题目照片（可选）", type="filepath",
                                     height=220, sources=["upload", "clipboard"])
                    text = gr.Textbox(
                        label="题干 / 方程（可直接粘贴题目文字）",
                        lines=4,
                        placeholder="解方程：x^2-5x+6=0",
                    )
                    with gr.Row():
                        allow_setup = gr.Checkbox(
                            value=True, label="应用题允许先列方程（调模型）",
                            info="通篇叙述、没有等号时，让模型把题面翻成方程；求解仍由 CAS 做",
                        )
                        use_explain = gr.Checkbox(value=True, label="生成讲评")
                    search_btn = gr.Button("搜答案", variant="primary")
                    gr.Examples(examples=SEARCH_EXAMPLES, inputs=[text],
                                label="三个例子（不花钱）")
                with gr.Column(scale=6):
                    s_badge = gr.HTML(IDLE)
                    s_eq = gr.Textbox(label="抽出来的方程（可改，改完点「按这个方程重算」）",
                                      lines=2)
                    s_resolve = gr.Button("按这个方程重算", size="sm")
                    s_answer = gr.Markdown(elem_id="answer", latex_delimiters=LATEX_DELIMS)
                    s_steps = gr.Markdown(elem_id="steps", latex_delimiters=LATEX_DELIMS)
                    s_meta = gr.Markdown()
            with gr.Accordion("完整结果 JSON", open=False):
                s_json = gr.JSON(height=220)

        with gr.Tab("② 批改 · 过程 → 判定"):
            with gr.Row():
                with gr.Column(scale=4):
                    problem = gr.Textbox(label="题面（可不填，不填就拿过程第一行当题面）",
                                         lines=2, placeholder="x^2-5x+6=0")
                    with gr.Tabs():
                        with gr.Tab("手打文字"):
                            work = gr.Textbox(label="解题过程（一行一步）", lines=12,
                                              placeholder="x^2-5x+6=0\n(x-2)(x-3)=0\nx=2 或 x=3")
                            grade_btn = gr.Button("批改这份过程", variant="primary")
                            gr.Examples(examples=GRADE_EXAMPLES, inputs=[problem, work],
                                        label="点一下试试（不花钱）")
                        with gr.Tab("拍照"):
                            photo = gr.Image(type="filepath", label="拍一张过程（整页）",
                                             height=280)
                            photo_btn = gr.Button("拍照批改", variant="primary")
                with gr.Column(scale=6):
                    g_badge = gr.HTML(
                        "<div style='background:#F6F5F1;border:1px solid #E4E2DB;"
                        "border-radius:12px;padding:14px 16px;font-size:13px;"
                        "color:#6B6A65;'>等待批改。判据是局部有效性：这一步能不能"
                        "从上一步推出来。和你算得不一样，不等于你错了。</div>")
                    g_table = gr.Markdown()
                    g_advice = gr.Markdown(latex_delimiters=LATEX_DELIMS, elem_id="steps")
                    g_gallery = gr.HTML(label="切出来的每一行")
                    g_reads = gr.Markdown()
            with gr.Accordion("完整结果 JSON", open=False):
                g_json = gr.JSON(height=220)

    s_outputs = [s_badge, s_eq, s_answer, s_steps, s_meta, s_json]
    search_btn.click(on_search, inputs=[image, text, allow_setup, use_explain],
                     outputs=s_outputs)
    s_resolve.click(on_resolve, inputs=[s_eq, use_explain], outputs=s_outputs)

    g_outputs = [g_badge, g_table, g_advice, g_json]
    grade_btn.click(on_grade_text, inputs=[problem, work], outputs=g_outputs)
    photo_btn.click(
        on_grade_photo, inputs=[problem, photo],
        outputs=[g_badge, g_table, g_advice, work, g_gallery, g_reads, g_json],
    )


if __name__ == "__main__":
    demo.queue().launch(
        theme=gr.themes.Default(primary_hue="teal", neutral_hue="gray"),
        css=CSS,
        server_name="127.0.0.1",
        server_port=int(os.environ.get("EQ_PORT", "7861")),
        show_error=True,
    )
