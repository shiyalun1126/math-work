"""MVP 界面：上传/样例 -> 识别 -> （歧义则追问）-> 求解 -> 讲评。

跑法：F:/PyCharm/python.exe app.py
密钥写在项目根目录 .env 里，也可以在界面「密钥与连接」里直接填。
"""

from __future__ import annotations

import json
import os

import gradio as gr

from mathagent import cases, config, photocheck, pipeline, workcheck
from mathagent.latex import parse as parse_latex
from mathagent.perceive import probe_all
from mathagent.route import plain as route_plain
from mathagent.solve import solve as solve_latex
from mathagent.types import PipelineResult, WorkCheck

from geoagent import pipeline as geo_pipeline
from geoagent import render as geo_render
from geoagent.schema import Problem as GeoProblem

FIXTURES = config.ROOT / "fixtures" / "demo.json"
DEMOS: dict = json.loads(FIXTURES.read_text(encoding="utf-8")) if FIXTURES.exists() else {}
SAMPLES = [(v.get("label", k), k) for k, v in DEMOS.items()]
DEFAULT_IMG = os.environ.get("MVP_DEFAULT_IMG") or str(config.ROOT / "fixtures" / "demo1.png")

LATEX_DELIMS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]

_BADGE = {
    "ok": ("#E1F5EE", "#1D9E75", "#04342C", "识别一致，可放心求解"),
    "review": ("#FAEEDA", "#EF9F27", "#412402", "存在歧义，请确认后继续"),
    "failed": ("#FCEBEB", "#E24B4A", "#501313", "识别失败"),
    "routed": ("#E6F1FB", "#378ADD", "#0B2B45", "这不是算式"),
}


def badge_html(r: PipelineResult) -> str:
    v = r.verification
    bg, border, fg, title = _BADGE.get(v.verdict, _BADGE["review"])
    if v.verdict == "routed":
        # 「识别失败」和「投错了门」是两件事：这张图的识别其实是对的，
        # 错的是拿它去解析算式。说错一个字，用户就会去重拍照片。
        title = {
            "geometry": "这是一道几何题，公式链路不适用",
            "prose": "这是一段题目文字，不是算式",
        }.get(r.route, title)
    src = sorted({c.source for c in r.recognition if c.latex})
    reasons = "".join(f"<li>{x}</li>" for x in v.reasons)
    if v.verdict == "routed":
        # 这里报「一致度 0%」会误导：候选判废是因为「不是算式」，
        # 不是因为两路读得不一样。识别质量这件事在这条路上根本没测。
        sub = f"来源 {'、'.join(src) if src else '无'} · 耗时 {sum(r.timings.values()):.2f}s"
    else:
        sub = (
            f"一致度 {v.confidence:.0%} · 来源 {'、'.join(src) if src else '无'} · "
            f"耗时 {sum(r.timings.values()):.2f}s"
        )
    tag = ""
    if v.verdict == "ok" and not v.independent:
        title = "识别完成（单源核对过，无第二来源印证）"
        tag = (
            '<span style="display:inline-block;margin-left:6px;padding:1px 7px;'
            'border-radius:9px;background:#fdf0d5;color:#8a5a00;font-size:11.5px;'
            'vertical-align:middle;">单源 · 无旁证</span>'
        )
    note = ""
    if r.image_note:
        note = (
            '<div style="margin-top:10px;padding:8px 10px;border-radius:8px;'
            'background:#fff8e6;border:1px solid #f0d9a8;font-size:12.5px;color:#5a4a20;">'
            f"⚠️ {r.image_note}</div>"
        )
    return f"""
<div style="background:{bg};border:1px solid {border};border-radius:12px;padding:14px 16px;">
  <div style="font-size:15px;font-weight:600;color:{fg};">{title}{tag}</div>
  <div style="font-size:13px;color:{fg};opacity:.85;margin-top:4px;">{sub}</div>
  <ul style="margin:8px 0 0 18px;padding:0;font-size:13px;color:{fg};opacity:.9;">{reasons}</ul>
  {note}
</div>"""


def formula_md(latex: str) -> str:
    if not latex.strip():
        return "_（暂无）_"
    return f"$${latex}$$"


def route_md(r: PipelineResult, auto: bool = False) -> str:
    """投错门时讲清楚：哪错了、为什么不是识别错、接下来去哪。"""
    head = {
        "geometry": (
            "### 这是一道几何题，不是算式\n\n"
            "这张图读出来的是一段题干文字，不是一个式子（读到的内容照抄在下面）。"
            "**公式链路只解已经列好的式子**，它没有读图、建坐标、解图形的能力，所以这里不给答案。"
        ),
        "prose": (
            "### 这不是算式\n\n"
            "识别到的是一段叙述文字。公式链路只解已经列好的式子，"
            "读不懂「谁比谁多几个」这类叙述。"
        ),
    }.get(r.route, "### 入口不对")
    out = [head]
    read = route_plain(r.parsed.latex or "")
    if read:
        quoted = "\n".join(f"> {line}" for line in read.splitlines()[:6] if line.strip())
        out.append("**读到的内容**（识别本身是对的）：\n\n" + quoted)
    if r.route_hints:
        out.append("判定依据：" + "、".join(f"`{h}`" for h in r.route_hints[:6]))
    if r.route == "geometry" and auto:
        out.append(
            "已经**就地转给几何链路**：同一张图重新读了一次，按「读图 → 重画 → 图文校验 → "
            "坐标求解」走，重绘图就在下面。完整结果也同步到了「几何题（题干 + 图形）」页签，"
            "那里可以手改结构重算。"
        )
    elif r.route == "geometry":
        out.append(
            "请到「几何题（题干 + 图形）」页签**上传这张图**再求解——"
            "只给文字的话，图里的点、线、标记读不到，条件会缺一半。"
        )
    else:
        out.append("这种题目前需要先把式子列出来（或换一种题型），再交给这个工具。")
    return "\n\n".join(out)


def steps_md(r: PipelineResult, auto: bool = False) -> str:
    pe, sr = r.parsed, r.solved
    if r.route:
        return route_md(r, auto=auto)
    if not pe.ok:
        return f"### 解析失败\n\n`{pe.error}`\n\n识别出的式子无法转成语法树，因此不给出答案。"
    out = ["### 解题步骤"]
    if r.verification.verdict == "review":
        out.append("> 下面按**当前候选**计算。若识别不对，点上方其他候选或直接改式子，结果会立刻重算。")
    if not sr.steps:
        out.append(f"_{sr.error or '没有可展示的步骤'}_")
    for i, st in enumerate(sr.steps, 1):
        out.append(f"**{i}. {st.title}**\n\n$${st.latex}$$")
    if sr.solutions:
        sol = "；".join(sr.solutions)
        out.append(f"### 结果\n\n$${sol}$$")
    if sr.dropped:
        out.append("> **未通过验算、已剔除的步骤**：" + "；".join(sr.dropped))
    if r.explain and r.explain.ok:
        out.append(
            "### 讲评\n\n"
            f"{r.explain.text}\n\n"
            f"> 由 `{r.explain.model}` 生成。它只讲思路、不参与计算，"
            f"公式与答案全部来自 SymPy 验算结果。"
        )
    elif r.explain and r.explain.note:
        out.append(f"> 讲评未生成：{r.explain.note}")
    return "\n\n".join(out)


def cand_update(r: PipelineResult):
    bad = {(c.latex or "").strip() for c in r.verification.rejected}
    chosen = (r.verification.chosen or "").strip()
    seen, items = set(), []
    for c in r.recognition:
        tex = (c.latex or "").strip()
        if not tex or tex in seen:
            continue
        seen.add(tex)
        tag = c.source + (f" · {c.conf:.2f}" if c.conf is not None else "")
        if tex in bad:
            tag += " · 无法解析"
        items.append((tag, tex))
    items.sort(key=lambda x: 0 if x[1] == chosen else 1)
    choices = [(f"{t}｜{tex[:44]}", tex) for t, tex in items]
    values = [v for _, v in choices]

    # Gradio 的 Radio 有个硬脾气：value 不在 choices 里就直接抛 Error，
    # 而且一抛就把整页所有输出组件都刷成"错误"。上游只要有任何一步
    # 把 chosen 换成规整版/改写版，界面就整体崩掉——所以这里必须兜住：
    # 认不出来的当前式，就自己补一个选项进去，绝不把不匹配的值往下传。
    if chosen and chosen not in values:
        choices.insert(0, (f"当前式｜{chosen[:44]}", chosen))
        values.insert(0, chosen)

    return gr.update(
        choices=choices,
        value=chosen if chosen in values else (values[0] if values else None),
        visible=len(choices) > 1,
        label=f"候选项（{len(choices)} 条，点一下即采用）",
    )


def render(r: PipelineResult, auto: bool = False):
    return (
        badge_html(r),
        r.verification.chosen,
        formula_md(r.verification.chosen),
        cand_update(r),
        steps_md(r, auto=auto),
        r.to_dict(),
    )


def on_image(img):
    if not img:
        raise gr.Error("请先上传图片或点一个样例")
    try:
        r = pipeline.run(image=img, k=2)
    except ValueError as exc:  # 没有密钥这类可预期的问题，直接把话说明白
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc

    if r.route != "geometry":
        return (*render(r), gr.update(value=None, visible=False), *GEO_IDLE_OUTS)

    # 几何题投到公式页签是常事（两个页签就在并排）。图已经上传了、
    # 钱也已经花了，再让用户自己去另一个页签重传一遍毫无道理——
    # 同一张图就地转过去跑，结果直接画在下面。
    try:
        geo = _geo_outputs(geo_pipeline.run(img))
        pic = gr.update(value=geo[0], visible=True)
    except Exception as exc:  # noqa: BLE001
        geo = (
            None,
            f"**自动转几何链路失败**：`{type(exc).__name__}` {exc}\n\n"
            "可以到「几何题」页签重试。",
            "", {}, "",
        )
        pic = gr.update(value=None, visible=False)
    return (*render(r, auto=True), pic, *geo)


def on_latex(latex: str):
    if not (latex or "").strip():
        raise gr.Error("输入为空")
    try:
        r = pipeline.resolve(latex)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
    # 手改式子重算只是同一个问题的另一种读法，不去动几何页签里已有的结果
    return (*render(r), gr.update(), *[gr.update() for _ in range(5)])


# ---------------------------------------------------------------- 密钥面板

def env_status_md() -> str:
    rows = [
        "| 通道 | 状态 | 当前值 |",
        "|---|---|---|",
    ]
    for s in config.status():
        if "env_file" in s:
            continue
        name = f"**{s['name']}**<br><small>{s['role']}</small>"
        mark = "✅ 就绪" if s["ready"] else "⬜ 未配置"
        val = config.mask(s["value"]) if s["value"] else "—"
        cell = f"`{val}`"
        if s.get("detail") and s["ready"]:
            cell += f"<br><small>{s['detail']}</small>"
        rows.append(f"| {name} | {mark} | {cell} |")
    path = config.ENV_FILE
    rows.append("")
    rows.append(f"配置文件：`{path}`（{'已存在' if path.exists() else '不存在'}）")
    if not (config.has_simpletex() or config.has_vlm()):
        rows.append("")
        rows.append(
            "> 还没有任何可用通道，当前只能看内置样例。"
            "下面的输入框粘贴密钥后点「保存到 .env」，或直接用编辑器改那个文件。"
        )
    return "\n".join(rows)


def on_save(st_token: str, vlm_token: str, vlm2_token: str = ""):
    changed = []
    if (st_token or "").strip():
        config.set_env_value("SIMPLETEX_TOKEN", st_token.strip())
        changed.append("SIMPLETEX_TOKEN")
    if (vlm_token or "").strip():
        config.set_env_value("DASHSCOPE_API_KEY", vlm_token.strip())
        changed.append("DASHSCOPE_API_KEY")
    if (vlm2_token or "").strip():
        config.set_env_value("VLM2_API_KEY", vlm2_token.strip())
        changed.append("VLM2_API_KEY")
    config.reload()
    msg = (
        f"已写入 `.env`：{'、'.join(changed)}，并已重新加载。下一步点「测试连接」确认能通。"
        if changed
        else "没填新内容，仅重新加载了 .env。"
    )
    return (
        env_status_md(),
        msg,
        config.describe(),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
    )


def on_probe_wrapper():
    config.reload()
    results = probe_all()
    lines = ["**连通性自检**（实打实调一次接口，会消耗极少额度）", ""]
    for label, ok, msg in results:
        lines.append(f"- {'✅' if ok else '❌'} **{label}**：{msg}")
    if not any(ok for _, ok, _ in results):
        lines.append("")
        lines.append("> 都不通。先检查 key 是否粘全、账号是否开通了对应模型。")
    return "\n".join(lines), config.describe()


# ---------------------------------------------------------------- 批改过程

_GRADE_STYLE = {
    "pass": ("#E1F5EE", "#1D9E75", "#04342C", "过程成立"),
    "partial": ("#FAEEDA", "#EF9F27", "#412402", "过程成立，但解不全"),
    "fake": ("#FBEAF0", "#D4537E", "#4B1528", "过程不成立，答案却是对的"),
    "wrong": ("#FCEBEB", "#E24B4A", "#501313", "过程有错"),
    "unread": ("#EEF1F6", "#7A8CA6", "#22303F", "这几行没认准，先不判"),
    "unknown": ("#F6F5F1", "#B4B2A9", "#2C2C2A", "只能做过程自检"),
    "gap": ("#F6F5F1", "#B4B2A9", "#2C2C2A", "没有可批改的式子"),
}

_MARK = {
    "ok": "✓ 通过",
    "wrong": "✗ 不成立",
    "partial": "△ 漏解",
    "suspect": "? 待核对",
    "note": "—",
    "unknown": "? 跳过",
}


def grade_badge(check: WorkCheck) -> str:
    bg, border, fg, title = _GRADE_STYLE.get(check.grade, _GRADE_STYLE["unknown"])
    sub = f"查了 {check.n_checked} 步，通过 {check.n_ok} 步"
    if check.first_error > 0:
        sub += f" · 第 {check.first_error} 步起不成立"
    if check.error_type:
        sub += f" · 错因：{check.error_type}"
    return f"""
<div style="background:{bg};border:1px solid {border};border-radius:12px;padding:14px 16px;">
  <div style="font-size:15px;font-weight:600;color:{fg};">{title}</div>
  <div style="font-size:13px;color:{fg};opacity:.9;margin-top:4px;">{check.headline}</div>
  <div style="font-size:12.5px;color:{fg};opacity:.75;margin-top:6px;">{sub}</div>
</div>"""


def _cell(text: str) -> str:
    """表格单元格里的竖线和换行会把 markdown 表格撑破。"""
    return (text or "").replace("|", r"\|").replace("\n", " ")


def step_table_md(check: WorkCheck) -> str:
    if not check.steps:
        return "_（没有读到任何一行）_"
    out = ["| # | 你写的 | 判定 | 依据 / 反例 |", "|---|---|---|---|"]
    for s in check.steps:
        if s.kind == "prose":
            body = s.raw
        elif s.kind == "assign" and s.raw:
            # 答案行按原样显示：`x=2 或 x=3` 只剩 `x=2`，
            # 学生看到的第一反应会是"我明明写了两个根"，那是在替系统背锅
            body = f"${s.raw}$"
        else:
            body = f"${s.latex}$"
        why = s.counter or s.reason or ""
        mark = s.row[0] if s.row else _MARK.get(s.verdict, s.verdict)
        out.append(f"| {s.idx} | {_cell(body)} | {mark} | {_cell(why)} |")
    return "\n".join(out)


def advice_md(check: WorkCheck) -> str:
    if not check.advice and not check.notes:
        return ""
    out = ["### 错在哪，怎么改", "", check.advice or ""]
    if check.expected:
        out.append(f"**标准答案**：{'、'.join(check.expected)}")
    if check.got:
        out.append(f"**你写的**：{'、'.join(check.got)}")
    if check.notes:
        out.append("")
        out.append("\n".join(f"> {n}" for n in check.notes))
    return "\n\n".join(x for x in out if x)


def solution_md(problem: str) -> str:
    """标准解法：CAS 独立算一遍，不参考学生怎么写。"""
    if not (problem or "").strip():
        return ""
    pe = parse_latex(problem)
    if not pe.ok:
        return f"> 题面读不出算式（`{pe.error}`），给不出标准解法。"
    sr = solve_latex(pe)
    if not sr.ok:
        return f"> 标准解法也没算出来：{sr.error}。\n\n这题超出当前能力范围，不给答案——硬给一个错的比不给更糟。"
    out = ["### 正确做法", "", "_这一段由 CAS 独立算出，与学生的过程无关。_", ""]
    for i, st in enumerate(sr.steps, 1):
        out.append(f"**{i}. {st.title}**\n\n$${st.latex}$$")
    if sr.solutions:
        out.append(f"**答案**：{'；'.join(sr.solutions)}")
    return "\n\n".join(out)


def _run_grade(problem: str, work: str) -> WorkCheck:
    if not (work or "").strip() and not (problem or "").strip():
        raise gr.Error("至少给一道题，或者贴入一段过程")
    try:
        return workcheck.grade(work or "", problem=problem or "")
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc


def on_grade(problem, work):
    check = _run_grade(problem, work)
    return (
        grade_badge(check),
        step_table_md(check),
        advice_md(check),
        solution_md(check.problem) if check.problem_ok else "",
        check.to_dict(),
    )


def on_save_case(problem, work):
    check = _run_grade(problem, work)
    if check.grade in ("pass", "gap"):
        return (
            cases.summary_md(),
            cases.rows(),
            f"判定是「{check.grade}」，没有错可记，不入错题本。",
        )
    if check.grade == "unread":
        return (
            cases.summary_md(),
            cases.rows(),
            "判定是「没认准」，这不能算作错，不入错题本。把认不准的那几行改成正确写法再判一次。",
        )
    # 之前错过同一道题：这是复发，比新错更值得看
    repeats = cases.same_type(check.problem)
    rec = cases.add(check)
    tail = f"；这道题以前错过 {len(repeats)} 次" if repeats else ""
    return (
        cases.summary_md(),
        cases.rows(),
        f"已存进错题本：`{rec['id']}` · 错因「{rec['error_type'] or '未归类'}」{tail}",
    )


def on_photo(problem, image, progress=gr.Progress()):
    """拍一张解题过程 → 切行 → 逐行识别 → 批改。"""
    if not image:
        raise gr.Error("先上传一张作业照片，或者点右边的内置样例")
    progress(0.15, desc="切行")
    try:
        pc = photocheck.check(image, problem=problem or "")
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
    progress(1.0, desc="完成")
    return (
        grade_badge(pc.work),
        step_table_md(pc.work),
        advice_md(pc.work),
        solution_md(pc.work.problem) if pc.work.problem_ok else "",
        pc.to_dict(),
        pc.text,                       # 回填到文本框：改一个字就能重判，不用重拍
        pc.lines_html(),
        reads_md(pc),
    )


def reads_md(pc) -> str:
    """识别层的账本：每行认成什么、几个来源、还有什么别的可能。

    批改结论如果是错的，根子几乎总在这张表里——所以它必须和结论一起出现，
    而不是藏在"高级设置"里。
    """
    if not pc.lines:
        return "### 逐行识别\n\n_（这一页没有切出任何一行）_"
    out = ["### 逐行识别", "", pc.read_table_md()]
    bad = [x for x in pc.lines if not x.sure]
    if bad:
        out.append("")
        out.append(
            f"> 第 {'、'.join(str(x.index) for x in bad)} 行没认准。"
            "它们在批改表里是「? 待核对」，不产生对错结论；"
            "改对之后按「批改这份过程」可以按文字重判。"
        )
    return "\n".join(out)


WORK_EXAMPLES = [
    [
        "x^2-5x+6=0",
        "x^2-5x+6=0\nx^2-6=5x\nx^2-5x=6\nx=2 或 x=3",
    ],
    [
        "x^2-5x+6=0",
        "x^2-5x+6=0\n两边同乘2得 2x^2-10x+12=0\n所以 (x-2)(x-3)=0\nx=2",
    ],
    [
        "2x-6=12",
        "2x-6=12\n2x=12\nx=6",
    ],
]

WORK_HINT = (
    "<div style='font-size:12.5px;color:#6B6A65;line-height:1.75'>"
    "一行一步，跟作业本上一样写。<b>可以夹人话</b>（「两边同乘2」「由①得」），"
    "系统会把说明行和式子分开看。<br>"
    "判据是<b>局部有效性</b>：这一步能不能从上一步推出来。所以一题多解不会被判错——"
    "和你算得不一样，不等于你错了。"
    "</div>"
)

PHOTO_HINT = (
    "<div style='font-size:12.5px;color:#6B6A65;line-height:1.75'>"
    "<b>怎么拍</b>：整页拍进去，别裁太狠；正对纸面，光别从一侧打；一行一步，别挤成两行。<br>"
    "拍歪了会自动纠偏，横线本的线会先被抹掉再切行。<br>"
    "<b>认不准的行会标出来，不会算你错。</b>"
    "一行认错，后面的反例就全建立在错的字上——所以认不准的时候系统只说"
    "「待核对」，不说「你不成立」。"
    "</div>"
)

WORK_DIR = config.ROOT / "fixtures" / "work"
WORK_SAMPLES = [(f"样例 · {p.stem}", str(p)) for p in sorted(WORK_DIR.glob("*.jpg"))]


CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto;}
#title h1 {font-size: 22px; margin-bottom: 2px;}
#sub {color: #5F5E5A; font-size: 13px; margin-top: 0;}
.katex-display {text-align: left !important; margin: 2px 0 8px 0 !important;}
.katex-display > .katex {text-align: left !important;}
#steps p {margin: 2px 0 6px 0; line-height: 1.5;}
#steps h3 {margin: 10px 0 4px 0; font-size: 15px;}
#envpanel table {width: 100%; font-size: 12.5px;}
#envpanel td:first-child, #envpanel th:first-child {white-space: nowrap;}
#envpanel td {vertical-align: top;}
"""

CONFIGURED = config.has_simpletex() or config.has_vlm()

IDLE_BADGE = """
<div style="background:#F6F5F1;border:1px solid #E4E2DB;border-radius:12px;padding:14px 16px;">
  <div style="font-size:15px;font-weight:600;color:#4A4843;">等待识别</div>
  <div style="font-size:13px;color:#6B6A65;margin-top:4px;">
    上传一张手写照片，或点左侧「内置样例」，然后按「识别并求解」。
  </div>
</div>"""

IDLE_STEPS = (
    "### 解题步骤\n\n"
    "识别完成后，这里依次列出**已经验算过**的每一步：\n\n"
    "- 方程会代回验证，增根会被剔除\n"
    "- 积分会反向求导确认能还原\n"
    "- 讲评只讲思路，不替系统算数"
)

EMPTY_INIT = (
    IDLE_BADGE,
    "",
    "",
    {"choices": [], "value": None, "visible": False, "label": "候选"},
    IDLE_STEPS,
    {},
)

if CONFIGURED:
    # 配好密钥后不在启动时自动请求，避免一开界面就花钱；只放静态占位。
    # 需要截图或现场演示预热时，显式设 MVP_AUTORUN=1 再启动。
    INIT = EMPTY_INIT
    if os.getenv("MVP_AUTORUN"):
        try:
            INIT = render(pipeline.run(image=DEFAULT_IMG, k=2, save=False))
        except Exception:  # noqa: BLE001
            INIT = EMPTY_INIT
else:
    try:
        INIT = render(pipeline.run(image=DEFAULT_IMG, k=2, save=False))
    except Exception:  # noqa: BLE001
        INIT = EMPTY_INIT

GEO_DIR = config.ROOT / "fixtures" / "geo"
GEO_SAMPLES = [(f"几何 {p.stem.upper()}", str(p)) for p in sorted(GEO_DIR.glob("*.png"))]

GEO_IDLE = (
    "### 等待读图\n\n"
    "上传一张**题干 + 图形**都在一页里的几何题照片。\n\n"
    "系统会先读出题干和图形结构，**把理解结果重画给你看**，再给答案。\n"
    "图读错了，你会一眼看出来——这是它和拍题软件最大的不同。"
)

TAB_FORMULA, TAB_GEO = "tab-formula", "tab-geo"
TAB_WORK, TAB_BOOK = "tab-work", "tab-book"
GEO_IDLE_OUTS = (None, GEO_IDLE, "", {}, "")


def _geo_outputs(res) -> tuple:
    """把结果整理成界面要的 5 个输出。"""
    coords, basis = geo_pipeline.figure_coords(res)
    pic = (
        geo_render.render(
            res.problem, coords, size=460,
            goal=geo_pipeline.goal_of(res), value=res.solution.value,
        )
        if res.problem
        else None
    )
    sol = res.solution
    # 学生视角的解题过程优先。它只在解出图形后才有，解不出来时退回系统步骤——
    # 那种时候「为什么没解出来」本身就是最该给的信息。
    steps = res.explain.text
    if not steps:
        if sol.steps:
            steps = "### 求解过程\n\n" + "\n".join(
                f"{i}. {s}" for i, s in enumerate(sol.steps, 1)
            )
        elif sol.error:
            steps = f"### 算不出来\n\n{sol.error}"
    if res.problem is not None:
        # 重绘只是「我读到了什么」的可视化，不是原图的复制品。不写清楚，
        # 用户会拿它当尺子量比例，然后问「为什么这个角不是 90 度」。
        steps += "\n\n> " + geo_render.describe_diff(res.problem, coords, basis)
    if res.problem is not None and not res.problem.points:
        # 反过来的误投：纯算式题传到了几何页签。图里当然没有点，
        # 不点破的话用户只会看到「没有读到足够的点」然后重拍一张。
        steps += (
            "\n\n> 图里没读到任何点。如果这是一道**纯算式题**（没有几何图形），"
            "请用「公式题」页签。"
        )
    if res.explain.ok and sol.steps:
        # 这些是排查用的，不是解题内容：压成一行放最后，别再占半个屏幕
        bits = []
        if res.problem is not None:
            bits.append(f"读入 {len(res.problem.points)} 个点")
        filled = sum(1 for s in sol.steps if s.startswith("补全"))
        if filled:
            bits.append(f"补全 {filled} 条只写在题干里的条件")
        if sol.n_unknown:
            bits.append(f"{sol.n_unknown} 个待定坐标")
        if sol.residual == sol.residual:
            bits.append(f"残差 {sol.residual:.0e}")
        if bits:
            steps += "\n\n> **系统核对**：" + "；".join(bits) + "。细节见「完整结果 JSON」。"
    if res.notes:
        steps += "\n\n" + "\n".join(f"- {n}" for n in res.notes)
    code = ""
    if res.problem is not None:
        code = json.dumps(res.problem.to_dict(), ensure_ascii=False, indent=2)
    return pic, geo_pipeline.summary_md(res), steps, res.to_dict(), code


def on_geo(img):
    if not img:
        return None, GEO_IDLE, "", {}, ""
    try:
        return _geo_outputs(geo_pipeline.run(img))
    except Exception as exc:  # noqa: BLE001
        return None, f"**出错**：`{type(exc).__name__}` {exc}", "", {}, ""


def on_geo_recount(code: str):
    """用户手改了识别结构后重算——不重新调 API，只重跑校验和求解。"""
    try:
        data = json.loads(code or "{}")
    except json.JSONDecodeError as exc:
        return None, f"**JSON 格式不对**：{exc}", "", {}, code
    if not isinstance(data, dict) or not data.get("points"):
        return None, "**结构不完整**：至少要有 `points` 字段。", "", {}, code
    try:
        p = GeoProblem.from_dict(data, source="手工修正").normative()
        return _geo_outputs(geo_pipeline.resolve(p))
    except Exception as exc:  # noqa: BLE001
        return None, f"**重算失败**：`{type(exc).__name__}` {exc}", "", {}, code


with gr.Blocks(title="手写数学识别 Agent · MVP") as demo:
    gr.Markdown(
        "# 手写数学 · 搜题 / 批改 / 错题本\n"
        "拍照求解 → 逐行批改学生自己写的过程 → 错因归档。"
        "**算的部分全部交给 CAS，认不准或判不了时把话说在前面，不硬猜。**",
        elem_id="title",
    )
    status_line = gr.Markdown(config.describe(), elem_id="sub")

    # 密钥面板放在页签外，公式题和几何题共用同一套通道
    with gr.Accordion("密钥与连接（.env）", open=not CONFIGURED):
        env_html = gr.Markdown(env_status_md(), elem_id="envpanel")
        st_box = gr.Textbox(
            label="SIMPLETEX_TOKEN",
            type="password",
            placeholder="粘贴后点「保存到 .env」，无需改代码",
        )
        vlm_box = gr.Textbox(
            label="DASHSCOPE_API_KEY / VLM_API_KEY",
            type="password",
            placeholder="阿里云百炼 API-KEY",
        )
        vlm2_box = gr.Textbox(
            label="VLM2_API_KEY（智谱，第二路 · 可不填）",
            type="password",
            placeholder="open.bigmodel.cn 的 API Key；填了才有跨源投票",
        )
        with gr.Row():
            save_btn = gr.Button("保存到 .env", size="sm")
            probe_btn = gr.Button("测试连接", size="sm")
        save_msg = gr.Markdown()
        probe_md = gr.Markdown()

    with gr.Tabs() as tabs:
        with gr.Tab("搜题 · 求解", id=TAB_FORMULA):
            with gr.Row():
                with gr.Column(scale=4):
                    image = gr.Image(
                        label="题目图片（手机拍的整页或单行都行）",
                        type="filepath",
                        height=240,
                        value=(DEFAULT_IMG if (not CONFIGURED or os.getenv("MVP_AUTORUN")) else None),
                        sources=["upload", "clipboard"],
                    )
                    sample = gr.Radio(
                        choices=SAMPLES,
                        value=(SAMPLES[0][1] if SAMPLES else None) if not CONFIGURED else None,
                        label="内置样例（不花钱，用来先看交互）",
                        visible=bool(SAMPLES),
                    )
                    with gr.Row():
                        run_btn = gr.Button("识别并求解", variant="primary", scale=2)
                        load_btn = gr.Button("载入样例", scale=1)

                with gr.Column(scale=6):
                    badge = gr.HTML(value=INIT[0])
                    latex_box = gr.Textbox(
                        label="识别结果（可直接修改，改完点“按此式求解”）",
                        lines=2,
                        value=INIT[1],
                    )
                    solve_btn = gr.Button("按此式求解", variant="secondary")
                    preview = gr.Markdown(value=INIT[2], latex_delimiters=LATEX_DELIMS)
                    cands = gr.Radio(
                        choices=INIT[3]["choices"],
                        value=INIT[3]["value"],
                        visible=bool(INIT[3]["visible"]),
                        label=INIT[3]["label"],
                    )
                    steps = gr.Markdown(value=INIT[4], latex_delimiters=LATEX_DELIMS, elem_id="steps")
                    # 只在「其实是几何题」时才露出来：把几何链路的重绘图直接
                    # 画在公式页签下，用户不用再找一个页签重传一遍
                    route_pic = gr.Image(
                        label="同一张图自动转几何链路后的重绘图",
                        height=320,
                        interactive=False,
                        visible=False,
                    )

            with gr.Accordion("完整结果 JSON（给评测脚本用）", open=False):
                payload = gr.JSON(value=INIT[5], height=260)

        with gr.Tab("几何题（题干 + 图形）", id=TAB_GEO):
            with gr.Row():
                with gr.Column(scale=4):
                    geo_img = gr.Image(
                        label="几何题照片（题干和图形在同一页）",
                        type="filepath",
                        height=260,
                        sources=["upload", "clipboard"],
                    )
                    geo_sample = gr.Radio(
                        choices=GEO_SAMPLES,
                        value=None,
                        label="内置样例（不花钱）",
                        visible=bool(GEO_SAMPLES),
                    )
                    geo_run = gr.Button("读图并求解", variant="primary")
                    gr.Markdown(
                        "<div style='font-size:12.5px;color:#6B6A65;line-height:1.7'>"
                        "读图顺序是：先把题干和图形读成结构 → 把结构<b>重画给你看</b> "
                        "→ 图文互相印证 → 解出答案。<br>"
                        "重绘图和原图对不上，说明读错了，直接在下方改结构重算。"
                        "</div>"
                    )

                with gr.Column(scale=6):
                    geo_md = gr.Markdown(GEO_IDLE)
                    geo_pic = gr.Image(
                        label="重绘图（对照原图看结构有没有缺漏）",
                        height=300,
                        interactive=False,
                    )
                    geo_steps = gr.Markdown(latex_delimiters=LATEX_DELIMS)

            with gr.Accordion("识别结构（可改，改完点重算）", open=False):
                geo_code = gr.Code(
                    language="json",
                    lines=16,
                    label="points / segments / marks 都能直接改",
                )
                geo_recount = gr.Button("按此结构重算", variant="secondary")

            with gr.Accordion("完整结果 JSON", open=False):
                geo_json = gr.JSON(height=240)

        with gr.Tab("批改过程", id=TAB_WORK):
            with gr.Row():
                with gr.Column(scale=4):
                    problem_box = gr.Textbox(
                        label="题面（可不填，不填就拿过程第一行当题面）",
                        lines=2,
                        placeholder="x^2-5x+6=0",
                    )
                    with gr.Tabs():
                        with gr.Tab("手打文字"):
                            work_box = gr.Textbox(
                                label="学生的解题过程（一行一步）",
                                lines=14,
                                placeholder="x^2-5x+6=0\n(x-2)(x-3)=0\nx=2 或 x=3",
                            )
                            grade_btn = gr.Button("批改这份过程", variant="primary")
                            gr.Examples(
                                examples=WORK_EXAMPLES,
                                inputs=[problem_box, work_box],
                                label="三个例子（点一下填进去）",
                            )
                        with gr.Tab("拍照"):
                            photo_in = gr.Image(
                                type="filepath",
                                label="拍一张解题过程（整页）",
                                height=300,
                            )
                            photo_btn = gr.Button("拍照批改", variant="primary")
                            gr.Examples(
                                examples=[[p] for _, p in WORK_SAMPLES],
                                inputs=[photo_in],
                                label="三张内置样例（点一下填进去）",
                            )
                            gr.HTML(PHOTO_HINT)
                    gr.HTML(WORK_HINT)

                with gr.Column(scale=6):
                    work_badge = gr.HTML(
                        "<div style='background:#F6F5F1;border:1px solid #E4E2DB;"
                        "border-radius:12px;padding:14px 16px;font-size:13px;"
                        "color:#6B6A65;'>等待批改。左边贴入过程或拍一张，右边会逐行告诉你哪一步站不住。</div>"
                    )
                    work_steps = gr.Markdown(latex_delimiters=LATEX_DELIMS, elem_id="steps")
                    photo_gallery = gr.HTML(label="切出来的每一行（对着它核对识别结果）")
                    photo_reads = gr.Markdown()
                    work_advice = gr.Markdown(latex_delimiters=LATEX_DELIMS, elem_id="steps")
                    work_sol = gr.Markdown(latex_delimiters=LATEX_DELIMS, elem_id="steps")
                    with gr.Row():
                        case_save_btn = gr.Button("存进错题本", size="sm")
                        case_msg = gr.Markdown()

            with gr.Accordion("完整结果 JSON（给评测脚本用）", open=False):
                work_json = gr.JSON(height=240)

        with gr.Tab("错题本", id=TAB_BOOK):
            book_md = gr.Markdown(cases.summary_md())
            book_df = gr.Dataframe(
                headers=cases.headers(),
                value=cases.rows(),
                interactive=False,
                label="按「该先补的」排序，不是按时间",
            )
            with gr.Row():
                book_refresh = gr.Button("刷新", size="sm")
                gr.Markdown(
                    "<div style='font-size:12.5px;color:#6B6A65;padding-top:8px'>"
                    "这里存的不是题目，是**你在哪一步出的错**。同一类错因连着出现三条，"
                    "说明那不是粗心，是那个动作还没学会。</div>"
                )


    book_refresh.click(
        lambda: (cases.summary_md(), cases.rows()),
        outputs=[book_md, book_df],
    )

    work_outputs = [work_badge, work_steps, work_advice, work_sol, work_json]
    grade_btn.click(on_grade, inputs=[problem_box, work_box], outputs=work_outputs)
    photo_btn.click(
        on_photo,
        inputs=[problem_box, photo_in],
        outputs=work_outputs + [work_box, photo_gallery, photo_reads],
    )
    case_save_btn.click(
        on_save_case,
        inputs=[problem_box, work_box],
        outputs=[book_md, book_df, case_msg],
    )

    # 公式页签的按钮除了回填自己那 6 个输出，还可能把同一张图转给几何链路：
    # 多出来的 1 张重绘图 + 5 个几何输出就挂在这里
    outputs = [
        badge, latex_box, preview, cands, steps, payload,
        route_pic, geo_pic, geo_md, geo_steps, geo_json, geo_code,
    ]

    run_btn.click(on_image, inputs=image, outputs=outputs)
    load_btn.click(
        lambda s: str(config.ROOT / "fixtures" / f"{s}.png") if s else None,
        inputs=sample,
        outputs=image,
    ).then(on_image, inputs=image, outputs=outputs)
    solve_btn.click(on_latex, inputs=latex_box, outputs=outputs)
    cands.change(on_latex, inputs=cands, outputs=outputs)

    geo_outputs = [geo_pic, geo_md, geo_steps, geo_json, geo_code]
    geo_run.click(on_geo, inputs=geo_img, outputs=geo_outputs)
    geo_sample.change(
        lambda s: s, inputs=geo_sample, outputs=geo_img
    ).then(on_geo, inputs=geo_img, outputs=geo_outputs)
    geo_recount.click(on_geo_recount, inputs=geo_code, outputs=geo_outputs)

    save_btn.click(
        on_save,
        inputs=[st_box, vlm_box, vlm2_box],
        outputs=[env_html, save_msg, status_line, st_box, vlm_box, vlm2_box],
    )
    probe_btn.click(on_probe_wrapper, outputs=[probe_md, status_line])


if __name__ == "__main__":
    demo.queue().launch(
        theme=gr.themes.Default(primary_hue="teal", neutral_hue="gray"),
        css=CSS,
        server_name="127.0.0.1",
        server_port=int(os.environ.get("MVP_PORT", "7860")),
        show_error=True,
    )
