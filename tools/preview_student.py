"""把学生端的结果页渲染成静态 HTML，用来截图核对排版。

跑法：F:/PyCharm/python.exe tools/preview_student.py
产出：.cache/shots/preview_answer.html / preview_check.html

为什么要有这个东西：Gradio 的界面要点击之后才有结果，而我想在设计阶段
就用离屏浏览器看一眼"答案区 + 逐行表"到底长什么样。这里复用 app_student
的渲染函数和同一份 CSS，只把 Gradio 的壳换成一个静态页面——
样式走的是同一份字符串，不会出现"预览好看、真页面不是这样"。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app_student as S  # noqa: E402
from mathagent import equations, workcheck  # noqa: E402

SHELL = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet"
 href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer
 src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[
   {{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}});"></script>
<style>
  html,body{{margin:0;background:#FAF8F3;}}
  body{{padding:46px 26px 80px;}}
  .page{{max-width:808px;margin:0 auto;}}
  .mock{{margin:0 0 18px;}}
  .mock .box{{background:#fff;border:1px solid #E3DED2;border-radius:8px;
    padding:14px 16px;font-size:14.5px;color:#55524B;line-height:1.7;}}
  .btn{{display:block;width:100%;height:44px;border-radius:8px;border:0;
    background:#2E6B57;color:#fff;font-size:14.5px;margin:14px 0 26px;}}
  {css}
</style></head>
<body><div class="page">{body}</div></body></html>"""


def md2html(md: str) -> str:
    """只覆盖这里用到的几种行：### 标题、**粗体**、- 列表、> 引用、空行分段。

    不引入 markdown 库：预览只要能把层次看出来，不需要完整语法。
    """
    out, buf, mode = [], [], None

    def flush():
        nonlocal buf, mode
        if not buf:
            mode = None
            return
        text = "\n".join(buf).strip()
        if not text:
            buf, mode = [], None
            return
        if mode == "ul":
            items = "".join(f"<li>{ln[2:].strip()}</li>" for ln in buf if ln.strip())
            out.append(f"<ul>{items}</ul>")
        elif mode == "bq":
            body = " ".join(ln.lstrip("> ").strip() for ln in buf)
            out.append(f"<blockquote>{body}</blockquote>")
        elif mode == "table":
            rows = [ln.strip().strip("|").split("|") for ln in buf if ln.strip()]
            head = "".join(f"<th>{c.strip()}</th>" for c in rows[0])
            body = "".join(
                "<tr>" + "".join(f"<td>{c.strip()}</td>" for c in r) + "</tr>"
                for r in rows[2:]
            )
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        else:
            out.append(f"<p>{text}</p>")
        buf, mode = [], None

    for line in md.split("\n"):
        s = line.strip()
        if not s:
            flush()
            continue
        if s.startswith("### "):
            flush()
            out.append(f"<h3>{s[4:]}</h3>")
            continue
        if s.startswith("|"):
            if mode != "table":
                flush()
                mode = "table"
            buf.append(s)
            continue
        if s.startswith("> "):
            if mode != "bq":
                flush()
                mode = "bq"
            buf.append(s)
            continue
        if s.startswith("- "):
            if mode != "ul":
                flush()
                mode = "ul"
            buf.append(s)
            continue
        if mode in ("table", "bq", "ul"):
            flush()
        buf.append(s)
        mode = "p"
    flush()
    html = "\n".join(out)
    html = html.replace("**", "")
    return html


def build(kind: str) -> str:
    if kind == "answer":
        ans = equations.answer("x^2-5x+6=0", explain=False)
        body = (
            '<div class="mock"><div class="box">x^2-5x+6=0</div></div>'
            '<button class="btn">看答案</button>'
            f'<div class="out"><div id="answer" class="prose">'
            f'{md2html(S._answer_md(ans))}</div></div>'
            f'<div class="out"><div id="steps" class="prose">'
            f'{md2html(S._steps_md(ans))}</div></div>'
        )
        return SHELL.format(title="对答案", css=S.CSS, body=S.HEAD + body)

    check = workcheck.grade(
        "x^2-5x+6=0\n(x+1)(x-6)=0\nx=-1 x=6", problem="x^2-5x+6=0")
    body = (
        '<div class="mock"><div class="box">x^2-5x+6=0</div></div>'
        '<button class="btn">批改</button>'
        + S._verdict_html(check)
        + f'<div class="out"><div class="prose">{md2html(S._table_md(check))}</div></div>'
        + f'<div class="out"><div class="prose">{md2html(S._advice_md(check))}</div></div>'
    )
    return SHELL.format(title="批改过程", css=S.CSS, body=S.HEAD + body)


def main() -> None:
    out = ROOT / ".cache" / "shots"
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("answer", "check"):
        p = out / f"preview_{kind}.html"
        p.write_text(build(kind), encoding="utf-8")
        print("wrote", p)


if __name__ == "__main__":
    main()
