"""学生端页面：一个题型，两件事，看得懂就行。

跑法：F:/PyCharm/python.exe app_student.py

和 `app_equation.py` 的区别只在**外面那一层**：同一套 `mathagent`
（求解、验算、逐行判定），换成学生看得下去的样子。所以那边继续当调试台，
这边给学生用。

设计上刻意克制：
- 一列到底，760px 阅读宽度，不做仪表盘那种满屏卡片。
- 只用一种强调色（墨绿）；红只留给"这一步错了"，黄只留给"没认准"。
- 分区靠一条细线，不靠圆角灰盒子堆叠——盒子一多，重点就被框没了。
- 标题用宋体，正文用黑体：像教辅，不像后台。
"""

from __future__ import annotations

import os
import re

import gradio as gr

from mathagent import equations, guard, photocheck, workcheck
from mathagent.types import WorkCheck

LATEX_DELIMS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]

CSS = """
:root{
  --paper:#FAF8F3; --ink:#1C1B18; --ink2:#55524B; --ink3:#8A857B;
  --rule:#E3DED2; --accent:#2E6B57; --bad:#A8452E; --warn:#9A6B1F;
}
/* 整页就是一张纸：不给内容区再套一层白底，否则看起来像浮在页面上的卡片。
   宽度必须同时给 width，只写 max-width 时容器会被内容撑成 ~420px（实测）。 */
html, body, gradio-app, gradio-app > .app{background:var(--paper) !important;}
.gradio-container{
  width:100% !important; max-width:860px !important;
  margin:0 auto !important; padding:0 26px 80px !important;
  background:var(--paper) !important;
}
.gradio-container > .main{padding:0 !important; max-width:100% !important;}
footer, .built-with, .show-api{display:none !important;}

body, .gradio-container, .prose, textarea, input, button{
  font-family:"PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif;
  font-variant-numeric:tabular-nums;
  color:var(--ink);
}

/* ---------- 页头 ---------- */
#masthead{padding:46px 0 22px;border-bottom:1px solid var(--rule);margin-bottom:26px;}
#masthead .kicker{
  font-size:11.5px;letter-spacing:.18em;color:var(--ink3);
  text-transform:uppercase;margin-bottom:14px;
}
#masthead h1{
  font-family:"Songti SC","Noto Serif SC",Georgia,serif;
  font-size:34px;font-weight:600;letter-spacing:.02em;margin:0 0 10px;line-height:1.25;
}
#masthead p{font-size:14.5px;color:var(--ink2);margin:0;line-height:1.8;}
#masthead .scope{font-size:12.5px;color:var(--ink3);margin-top:12px;letter-spacing:.02em;}

/* ---------- 模式切换 ---------- */
#switch{margin:0 0 18px !important;gap:8px !important;}
#switch button{
  border-radius:8px !important;font-size:14px !important;height:40px !important;
  box-shadow:none !important;border:1px solid transparent !important;
  background:transparent !important;color:var(--ink3) !important;font-weight:500 !important;
}
#switch button.primary{
  background:#FFFFFF !important;border-color:var(--ink) !important;
  color:var(--ink) !important;font-weight:600 !important;
}
#switch button.secondary:hover{color:var(--ink2) !important;}

/* ---------- 输入 ---------- */
.gradio-container textarea, .gradio-container input[type=text]{
  background:#fff !important;border:1px solid var(--rule) !important;
  border-radius:8px !important;box-shadow:none !important;font-size:14.5px !important;
  line-height:1.7 !important;color:var(--ink) !important;
}
.gradio-container textarea:focus, .gradio-container input[type=text]:focus{
  border-color:var(--accent) !important;box-shadow:0 0 0 3px rgba(46,107,87,.10) !important;
}
.gradio-container label span{font-size:12.5px !important;color:var(--ink2) !important;}

/* 主动作按钮 */
.gradio-container button.primary{
  background:var(--accent) !important;border:1px solid var(--accent) !important;
  color:#fff !important;border-radius:8px !important;font-size:14.5px !important;
  font-weight:500 !important;height:44px !important;box-shadow:none !important;
}
.gradio-container button.primary:hover{background:#245746 !important;}

/* ---------- 结果区 ---------- */
.out{background:transparent !important;border:0 !important;padding:0 !important;}
.out .prose{font-size:15.5px;line-height:1.9;color:var(--ink);}
.out .prose h3{
  font-family:"Songti SC","Noto Serif SC",Georgia,serif;
  font-size:15px;font-weight:600;letter-spacing:.06em;color:var(--ink2);
  margin:30px 0 10px;padding-top:16px;border-top:1px solid var(--rule);
}
.out .prose h3:first-child{border-top:0;padding-top:0;margin-top:6px;}
.out .prose p{margin:6px 0;}
.out .prose blockquote{
  margin:12px 0;padding:0 0 0 14px;border-left:2px solid var(--rule);
  color:var(--ink2);font-size:13px;background:transparent;
}
.out .prose ul{margin:6px 0 6px 2px;padding-left:18px;}
.out .prose li{margin:3px 0;font-size:14px;color:var(--ink2);}
.katex-display{text-align:left !important;margin:6px 0 12px !important;}
.katex-display>.katex{text-align:left !important;}

#answer .katex-display>.katex{font-size:1.85em;}
#answer .prose{font-size:16px;}

#steps .prose strong{font-weight:600;color:var(--ink2);font-size:13.5px;letter-spacing:.02em;}

/* ---------- 结论条 ---------- */
#verdict{border-left:3px solid var(--ink3);padding:2px 0 2px 16px;margin:6px 0 4px;}
#verdict .t{font-size:17px;font-weight:600;line-height:1.4;}
#verdict .s{font-size:13px;color:var(--ink2);margin-top:6px;line-height:1.7;}

/* ---------- 逐行表 ---------- */
/* 固定列宽。自动布局下"依据"把"你写的"挤到 140px，
   `(x+1)(x-6)=0` 会断成 `(x+1)(x-` + `6)=0`，看着像学生写错了。 */
.out .prose table{
  width:100%;border-collapse:collapse;margin:8px 0 4px;font-size:14px;
  table-layout:fixed;
}
.out .prose th:nth-child(1), .out .prose td:nth-child(1){width:34px;}
.out .prose th:nth-child(2), .out .prose td:nth-child(2){width:31%;}
.out .prose th:nth-child(3), .out .prose td:nth-child(3){width:15%;white-space:nowrap;}
@media (max-width:640px){
  .out .prose table{table-layout:auto;font-size:13px;}
  .out .prose th:nth-child(2), .out .prose td:nth-child(2){width:auto;}
}
.out .prose th{
  text-align:left;font-weight:500;font-size:12px;letter-spacing:.04em;
  color:var(--ink3);border-bottom:1px solid var(--ink3);padding:8px 10px 6px;
  white-space:nowrap;
}
.out .prose td{
  border-bottom:1px solid var(--rule);padding:11px 10px;vertical-align:top;
  line-height:1.6;color:var(--ink);
}
.out .prose td:first-child{color:var(--ink3);width:28px;}
.out .prose tr:last-child td{border-bottom:0;}

/* 自组件被自动套的 form 层：灰底、无边框，直接抹掉 */
.gradio-container .form, .gradio-container form{
  background:transparent !important;border:0 !important;
  box-shadow:none !important;padding:0 !important;
}

/* 输入框：Gradio 的"块"和里面的 input 各画一层边框，成了框套框。
   只留"块"这一层白底细框，里面的输入本体不画边。 */
.gradio-container .block:has(textarea),
.gradio-container .block:has(input[type="text"]){
  background:#FFFFFF !important;border:1px solid var(--rule) !important;
  border-radius:8px !important;box-shadow:none !important;
}
.gradio-container .block:has(textarea) textarea,
.gradio-container .block:has(input[type="text"]) input[type="text"]{
  border:0 !important;background:transparent !important;box-shadow:none !important;
}

/* 勾选框：Gradio 给它套了一层带边框的白底，放在纸上像多了一个空输入框 */
.gradio-container .bare{border:0 !important;background:transparent !important;
  box-shadow:none !important;padding-left:0 !important;margin-top:2px !important;}
.gradio-container .bare label{font-size:13.5px !important;color:var(--ink2) !important;}

/* 上传区：默认是浅灰白，放在纸底上像贴了一块膏药 */
.gradio-container .image-container, .gradio-container .upload-container,
.gradio-container .wrap.default{background:#FFFFFF !important;}

/* 行图 */
#lineimgs img{width:100%;height:auto;display:block;border:1px solid var(--rule);
  border-radius:6px;background:#fff;}
"""

HEAD = """
<div id="masthead">
  <div class="kicker">初中数学 · 一个题型</div>
  <h1>方程</h1>
  <p>把题目拍下来，或者把解题过程拍下来。<br>
     答案怎么来的、第几步站不住，都会告诉你。</p>
  <div class="scope">一元一次 · 一元二次 · 二元一次方程组 · 含参数</div>
</div>"""

_TONE = {
    # grade: (左边线颜色, 标题, 说明)
    "pass": ("#2E6B57", "这份过程成立"),
    "partial": ("#9A6B1F", "过程成立，但解不全"),
    "fake": ("#A8452E", "过程不成立，答案却是对的"),
    "wrong": ("#A8452E", "这份过程有错"),
    "unread": ("#8A857B", "有几行没认准，先不下结论"),
    "gap": ("#8A857B", "没有可批改的式子"),
    "unknown": ("#8A857B", "只能做过程自检"),
}


# ---------------------------------------------------------------- 渲染

def _wrap(md: str, elem: str) -> str:
    return f"<div id='{elem}'>{md}</div>"


def _verdict_html(check: WorkCheck, scope_note: str = "") -> str:
    """结论条：一条竖线 + 三行字。

    不做成彩色卡片。卡片一上色，整页最响的就是那块背景，
    可学生要读的是"第几步错了"，不是"你错了"这个事实本身。
    """
    color, title = _TONE.get(check.grade, _TONE["unknown"])
    passed = [
        s for s in check.steps
        if s.verdict == "ok" and (check.first_error <= 0 or s.idx < check.first_error)
    ]
    local = [
        s for s in check.steps
        if s.verdict == "ok" and check.first_error > 0 and s.idx > check.first_error
    ]
    bits = [f"查了 {check.n_checked} 步，通过 {len(passed)} 步"]
    if check.error_type:
        bits.append(f"错因：{check.error_type}")
    if local:
        bits.append(f"第 {'、'.join(str(s.idx) for s in local)} 步只是局部成立")
    sub = " · ".join(bits)
    note = ""
    if scope_note:
        note = (f"<div class='s' style='margin-top:10px;color:#9A6B1F;'>{scope_note}</div>")
    return (
        f"<div id='verdict' style='border-left-color:{color};'>"
        f"<div class='t' style='color:{color};'>{title}</div>"
        f"<div class='s'>{check.headline}</div>"
        f"<div class='s' style='color:#8A857B;'>{sub}</div>{note}</div>"
    )


def _cover_md(ans) -> str:
    t = ans.topic
    if t.coverage == "in":
        return ""
    word = "能答，但要说清边界" if t.coverage == "edge" else "不在这个题型的范围里"
    vt = equations.vars_text(t)
    detail = t.label + (f" · {vt}" if vt else "")
    extra = ans.note or t.note or ans.error or ""
    return f"> **{word}** · {detail}\n>\n> {extra}"


def _answer_md(ans) -> str:
    if ans.error and not ans.ok:
        return f"### 答案\n\n**不给结论**：{ans.error}"
    if not ans.ok:
        return ("### 答案\n\n"
                f"**没算出来**：{ans.error or '解不出来'}\n\n"
                "> 这种情况不硬给答案。多半是式子里有符号读错了，改对再算一次更快。")
    out: list[str] = []
    if ans.branches:
        out.append("### 答案\n\n**要按参数分情况**\n")
        out += [f"$${b}$$" for b in ans.branches]
        out.append("> 每一支都取了一组实际参数值代回去验过。")
    else:
        out.append(f"### 答案\n\n$${ans.answer_latex}$$")
        if ans.checks:
            out.append("**代回去验一下**\n")
            out += [f"- {c}" for c in ans.checks]
    if ans.dropped:
        out.append("**这些解要划掉**\n")
        out += [f"- {d}" for d in ans.dropped]
    return "\n".join(out)


def _derive_md(ans) -> str:
    """配方法推导。只有「二次项系数带字母」的方程才有这一段。

    单独一章而不是塞进「怎么算的」：求解是从题到答案，推导是从公式到公式，
    两件事混在一起，学生会以为每道题都得先配一遍方。
    """
    dv = getattr(ans, "derivation", None) or []
    if not dv:
        return ""
    out = ["### 求根公式怎么推出来的\n"]
    out.append("> 配方法。中间几步都是两边同除、同加，和上一步同解；"
               "开方带回 ±，两支都代回原方程验过。\n")
    for i, st in enumerate(dv, 1):
        out.append(f"**{i}　{st['title']}**\n")
        out.append(f"$${st['latex']}$$")
    return "\n".join(out)


def _steps_md(ans) -> str:
    head = _derive_md(ans)
    if not ans.steps:
        return head
    out = ["### 怎么算的\n"]
    for i, st in enumerate(ans.steps, 1):
        out.append(f"**{i}　{st['title']}**\n")
        out.append(f"$${st['latex']}$$")
    body = "\n".join(out)
    if ans.explain:
        body += "\n\n### 为什么这么算\n\n" + ans.explain
    return f"{head}\n\n{body}" if head else body


def _meta_md(meta: dict | None, ans) -> str:
    bits = []
    if meta:
        conf = meta.get("confidence", 0) or 0
        if conf >= 0.9:
            bits.append(f"照片认得很清楚（{conf:.0%}）")
        else:
            bits.append(f"照片只有 {conf:.0%} 的把握，对一下上面的方程有没有读错")
    if bool(getattr(ans, "cached", False)):
        bits.append("这道题之前算过，直接给的结论")
    return "\n".join(f"> {b}" for b in bits)


def _assign_cell(raw: str) -> str:
    """一行里写了两个答案时，逗号要放在数学环境**外面**。

    `$x=-1\\quad x=6$` 整个交给 KaTeX，空格会被吞掉，显示成 `x=-1x=6`——
    看上去像学生把两个答案连写成了一个式子。拆成两个 $..$ 才看得清是"两个答案"。

    分隔符只认真正用来分答案的那几个（`\\quad`、逗号、分号、"或"），
    **不认普通空格**：`y = 2` 里的空格是式子自己的排版，按空格拆会显示成
    「y，=，2」——一行正确的手写被拆成三个符号，比不拆更让人怀疑系统。
    """
    parts = [
        p.strip()
        for p in re.split(r"\\qquad|\\quad|\\[,;! ]|[，,;；]|或", raw)
        if p.strip()
    ]
    if len(parts) <= 1:
        return f"${raw.strip()}$"
    return "，".join(f"${p}$" for p in parts)


def _table_md(check: WorkCheck) -> str:
    if not check.steps:
        return "_没有读到任何一行。_"
    rows = ["| 　 | 你写的 | 判定 | 依据 |", "|---|---|---|---|"]
    for s in check.steps:
        if s.kind == "prose":
            body = s.raw
        elif s.kind == "assign" and s.raw:
            body = _assign_cell(s.raw)
        else:
            body = f"${s.latex}$"
        mark = s.row[0] if s.row else s.verdict
        tail = (s.counter or s.reason or "").replace("|", "\\|").replace("\n", " ")
        if tail.strip() == mark.strip():
            tail = ""
        rows.append("| {} | {} | {} | {} |".format(
            s.idx, body.replace("|", "\\|"), mark, tail))
    return "\n".join(rows)


def _advice_md(check: WorkCheck) -> str:
    out = []
    if check.advice:
        out.append("### 怎么改\n\n" + check.advice)
    tail = []
    if check.expected:
        tail.append(f"**标准答案**　{'、'.join(check.expected)}")
    if check.got:
        tail.append(f"**你写的**　{'、'.join(check.got)}")
    if tail:
        out.append("\n\n".join(tail))
    if check.notes:
        out.append("\n".join(f"> {n}" for n in check.notes))
    return "\n\n".join(x for x in out if x)


def _reads_md(pc) -> str:
    if not pc.lines:
        return "_这一页没有切出任何一行。_"
    out = ["### 我看到了什么\n", pc.read_table_md()]
    manual = [x.index for x in pc.lines if photocheck.MANUAL_SOURCE in x.sources]
    bad = [x for x in pc.lines if not x.sure]
    if manual:
        out.append(f"> 第 {'、'.join(str(i) for i in manual)} 行是你手动输入并确认过的，"
                   "表里按你写的式子判。")
    if bad:
        out.append(f"> 第 {'、'.join(str(x.index) for x in bad)} 行没认准，"
                   "它们在批改表里是「待核对」，不算你错。"
                   "可以在下面「有行没认准」框里直接改成你写的样子，再判一次。")
    return "\n".join(out)


def _scope_note(problem: str) -> str:
    if not (problem or "").strip():
        return ""
    eq = equations.extract(problem)
    if eq.topic.coverage == "out":
        return (f"题面是「{eq.topic.label}」，不在方程范围内。"
                "下面只检查过程内部前后对不对得上，别当成最终判定。")
    return ""


# ---------------------------------------------------------------- 动作

def run_answer(text, image, explain, progress=gr.Progress()):
    t = (text or "").strip()
    if not t and not image:
        raise gr.Error("先拍一张题目，或者把题目打进来")
    meta = None
    try:
        if t:
            progress(0.4, desc="抽方程")
            ans = equations.answer(t, allow_setup=True, explain=explain)
        else:
            progress(0.25, desc="读题")
            ans, meta = equations.answer_image(image, allow_setup=True, explain=explain)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
    progress(1.0, desc="好了")
    return _cover_md(ans), _answer_md(ans), _steps_md(ans), _meta_md(meta, ans)


def _doubt_choices(pc) -> list[str]:
    """没认准的行，做成勾选项：`6. y = z`。行号在前，回来才认得出是哪一行。"""
    return [
        f"{x.index}. {x.text or '（没读出来）'}" for x in pc.lines if not x.sure
    ]


def _confirm_ids(labels) -> set[int]:
    """把勾选的标签还原成原行号。"""
    out: set[int] = set()
    for s in labels or []:
        m = re.match(r"\s*(\d{1,2})\s*[.、]", str(s))
        if m:
            out.add(int(m.group(1)))
    return out


def run_check(problem, text, image, progress=gr.Progress()):
    w = (text or "").strip()
    if not w and not image:
        raise gr.Error("先拍一张过程，或者把过程一行一行打进来")
    progress(0.2, desc="看过程")
    try:
        if w:
            check = workcheck.grade(w, problem=problem or "")
            progress(1.0, desc="好了")
            return (
                _verdict_html(check, _scope_note(problem)),
                _table_md(check),
                _advice_md(check),
                "",
                "",
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(choices=[], value=[], visible=False),
                None,
            )
        pc = photocheck.check(image, problem=problem or "")
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
    progress(1.0, desc="好了")
    doubt = _doubt_choices(pc)
    has_photo = bool(pc.lines)
    return (
        _verdict_html(pc.work, _scope_note(problem or pc.work.problem)),
        _table_md(pc.work),
        _advice_md(pc.work),
        _reads_md(pc),
        pc.lines_html(),
        gr.update(value=photocheck.edit_text(pc), visible=has_photo),
        gr.update(visible=has_photo),
        gr.update(choices=doubt, value=[], visible=bool(doubt)),
        pc,
    )


def run_regrade(fix_text, doubted, problem, state, progress=gr.Progress()):
    """按用户改过 / 核对过的过程重判。不重新识别，所以几秒就出结果。"""
    if state is None:
        raise gr.Error("先拍一张过程照片批改一次，再来改")
    if not (fix_text or "").strip():
        raise gr.Error("过程是空的。想删掉某一行，删那一行就行，别整段清空")
    progress(0.3, desc="按你给的过程重算")
    try:
        pc = photocheck.regrade(
            state, fix_text, confirm=_confirm_ids(doubted), problem=problem or ""
        )
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
    progress(1.0, desc="好了")
    return (
        _verdict_html(pc.work, _scope_note(problem or pc.work.problem)),
        _table_md(pc.work),
        _advice_md(pc.work),
        _reads_md(pc),
        pc.lines_html(),
    )


def switch(mode: str):
    """两个模式互相切，返回 (按钮样式、两块可见性)。"""
    answer = mode == "answer"
    return (
        gr.update(variant="primary" if answer else "secondary"),
        gr.update(variant="secondary" if answer else "primary"),
        gr.update(visible=answer),
        gr.update(visible=not answer),
    )


# ---------------------------------------------------------------- 界面

with gr.Blocks(title="方程") as demo:
    gr.HTML(HEAD)
    mode = gr.State("answer")

    with gr.Row(elem_id="switch"):
        b_answer = gr.Button("对答案", variant="primary")
        b_check = gr.Button("批改过程", variant="secondary")

    # ------------------------------------------------ 对答案
    with gr.Column(visible=True) as pane_answer:
        # 照片放第一位：学生手上就是一张卷子，打字是退路，不是主路。
        a_image = gr.Image(type="filepath", label="题目照片", height=190)
        a_type = gr.Checkbox(label="没有照片，我直接打字", value=False,
                             elem_classes=["bare"])
        a_text = gr.Textbox(
            label="题目文字", lines=3, visible=False,
            placeholder="3x+6=15　　／　　x^2-5x+6=0　　／　　解方程组 x+y=3, x-y=1",
        )
        a_explain = gr.Checkbox(label="顺便讲讲思路", value=True, elem_classes=["bare"])
        a_go = gr.Button("看答案", variant="primary")

        with gr.Column(elem_classes=["out"]):
            a_cover = gr.Markdown(latex_delimiters=LATEX_DELIMS)
            a_ans = gr.Markdown(
                "拍完或打完，点上面的按钮。答案会出现在这里，下面接着是每一步怎么来的。",
                latex_delimiters=LATEX_DELIMS, elem_id="answer")
            a_steps = gr.Markdown(latex_delimiters=LATEX_DELIMS, elem_id="steps")
            a_meta = gr.Markdown()

    # ------------------------------------------------ 批改
    with gr.Column(visible=False) as pane_check:
        c_problem = gr.Textbox(
            label="题目（可以不填）", lines=1,
            placeholder="填了能看出「第二个方程抄错了」这类问题；不填就只能查前后对不对得上",
        )
        c_image = gr.Image(type="filepath", label="过程照片（整页）", height=220)
        c_type = gr.Checkbox(label="没有照片，我直接打过程", value=False,
                             elem_classes=["bare"])
        c_text = gr.Textbox(
            label="过程（一行一步）", lines=8, visible=False,
            placeholder="x^2-5x+6=0\n(x-2)(x-3)=0\nx=2 或 x=3",
        )
        c_go = gr.Button("批改", variant="primary")

        with gr.Column(elem_classes=["out"]):
            c_verdict = gr.HTML(
                "<div style=\"color:#8A857B;font-size:14px;line-height:1.8;padding:4px 0;\">"
                "把过程拍下来或打进来，点「批改」。会逐行告诉你：哪一步还站得住，哪一步开始不成立。"
                "</div>")
            c_rows = gr.Markdown(latex_delimiters=LATEX_DELIMS)
            c_advice = gr.Markdown(latex_delimiters=LATEX_DELIMS)
            with gr.Column(visible=False, elem_id="fixbox") as c_fixbox:
                gr.Markdown(
                    "### 有行没认准？直接改成你写的\n"
                    "下面每行前面的数字是照片上的行号。把认错的行改成你写的式子"
                    "（`^` 表示乘方，比如 `x^2`），点下面的按钮重判——"
                    "**不会重新识别**，改哪行算哪行。\n\n"
                    "认不出的行留着一个 `?`，把它替换掉就行。"
                    "整行删掉＝这行不算；漏切的行，在它该在的位置补一行（**前面不写数字**）。"
                )
                c_fix = gr.Textbox(
                    label="逐行过程（可改）", lines=10, visible=False,
                    placeholder="1. 2x+y=4\n2. x+3y=7",
                )
                c_doubt = gr.CheckboxGroup(
                    label="上面这几行系统没认准。核对后没错的，勾上（按这个读法算）",
                    choices=[], visible=False, elem_classes=["bare"],
                )
                c_fix_go = gr.Button("按我改的重新判", variant="secondary")
            c_reads = gr.Markdown(latex_delimiters=LATEX_DELIMS)
            with gr.Column(elem_id="lineimgs"):
                c_lines = gr.HTML()
            c_state = gr.State(None)

    b_answer.click(lambda: "answer", None, mode).then(
        switch, mode, [b_answer, b_check, pane_answer, pane_check])
    b_check.click(lambda: "check", None, mode).then(
        switch, mode, [b_answer, b_check, pane_answer, pane_check])

    a_type.change(lambda v: gr.update(visible=bool(v)), a_type, a_text)
    c_type.change(lambda v: gr.update(visible=bool(v)), c_type, c_text)

    a_go.click(run_answer, [a_text, a_image, a_explain],
               [a_cover, a_ans, a_steps, a_meta])
    a_text.submit(run_answer, [a_text, a_image, a_explain],
                  [a_cover, a_ans, a_steps, a_meta])

    c_go.click(run_check, [c_problem, c_text, c_image],
               [c_verdict, c_rows, c_advice, c_reads, c_lines,
                c_fix, c_fixbox, c_doubt, c_state])
    c_fix_go.click(run_regrade, [c_fix, c_doubt, c_problem, c_state],
                   [c_verdict, c_rows, c_advice, c_reads, c_lines])


if __name__ == "__main__":
    # 先把超时守卫的子进程拉起来，省掉第一页的 1 秒等待（后台线程，不挡启动）。
    guard.warmup()
    demo.queue().launch(
        css=CSS,
        theme=gr.themes.Base(),
        server_name="127.0.0.1",
        server_port=int(os.environ.get("STU_PORT", "7862")),
        # 不把 traceback 推到前端：里面有本机绝对路径和内部调用栈，
        # 分享给别人时等于把自己的目录结构贴出去。
        # gr.Error 是显式抛的，消息照样会显示，不受这个开关影响。
        show_error=False,
    )
