"""拍照批改：整页照片 → 切行 → 逐行识别 → 拼成过程 → 批改。

这条链路最危险的地方不在识别率，而在**把认错当成学生错**。

一行认错，`workcheck` 会照常给出反例、照常判"这一步不成立"，而且看起来
毫不含糊——因为它是拿认错的那行当真值算的。用户看到的是"系统说我这一步错了"，
实际上是他自己写的没错、系统看错了。批改系统里最不能犯的就是这个错：
判错的门槛本来就高，一旦有一次是冤枉的，之后所有"你错了"都不作数了。

所以这里的规矩是：

  1. 每一行都单独投票（多来源），拿不到一致就说"没认准"，不硬猜。
  2. 没认准的行**不产生错误结论**：它下面那一步即使算出来不成立，也降级成
     "待核对"，反例作废——反例建立在认错的字上，就不再是证据。
  3. 没认准的行多于三分之一，整份不给总判定（`unread`）。adjacency
     （相邻两式比较）本来就靠"上下行连着"，中间空一格，后面的判断都不作数。
  4. 认不出的行在过程文本里留一个 `?` 占位，不删。删掉会让后面的行错位，
     把"A 和 C 相邻"当成"A 和 B 相邻"，比留个问号糟得多。
"""

from __future__ import annotations

import base64
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import sympy

from . import config
from .lines import PageSplit, save_crops, split_page
from .perceive import build_ensemble
from .types import Candidate, WorkCheck
from .workcheck import grade

# 一致度低于这个值就算"没认准"。0.75 的来历：两个独立来源一致是 1.0，
# 单来源自一致被 verify 封顶在 0.8，所以 0.75 恰好把"只有一家说"划进可疑区。
UNREAD_CONF = 0.75

# 没认准的行占比超过这个数，整份不判。三分之一是个经验值：
# 一行认不出还能靠上下文推，每三行认不出一行，上下文本身就没了。
MAX_UNREAD_SHARE = 0.34

# 认不出的行留这个占位。不能用中文占位符——`workcheck` 会把带中文的行
# 当"说明行"（不判定、直接跳过），看起来像正常识别过；用 `?` 才会被判成
# "读不出算式"，在表里明确标出来。
PLACEHOLDER = "?"

# 单页最多认多少行。不是技术限制，是钱和时间的限制：一行两次调用，
# 四十行就是一整分钟的等待和一笔不小的额度，这种时候该提醒用户分页拍。
MAX_LINES = 24

# 人工改过的行，`sources` 里记这一条。它不是"识别来源"，是为了让界面和批改层
# 都能一眼看出哪几行是用户自己给的——那几行的对错不由系统负责。
MANUAL_SOURCE = "手输"
# 用户核对过、但一个字没改的行（原文本来就对，只是系统没把握）。
# 与 `手输` 分开记，是因为这两件事的保证不一样：一个是"我告诉系统写的是什么"，
# 一个是"系统看对了，我确认过"。
CONFIRM_SOURCE = "核对"

WORK_DIR = config.ROOT / "data" / "work"

# 行号前缀：`3. y=2`。只给人对照用，`workcheck._ENUM` 在解析前会剥掉；
# 万一人把它删了也没关系，`_split_edits` 会退回"按位置对应"。
_EDIT_NUM = re.compile(r"^\s*(\d{1,2})\s*[.、)）:：]\s*(.*)$")


@dataclass
class LineRead:
    """一行小图，以及系统从它身上读到了什么、有多大把握。"""

    index: int                  # 行号，从 1 起，与 crops 一致
    crop: str = ""              # 小图路径，界面上要显示
    latex: str = ""             # 投票选中的那条候选
    text: str = ""              # 还原成能读的文字（\text{...} 剥成中文）
    verdict: str = "failed"     # ok / review / failed / blank
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)

    @property
    def sure(self) -> bool:
        """这一行能不能拿来做判定依据。

        `prose`（纯说明行）算"看清了"：它本来就不参与对错判定，
        措辞差一个字不改变任何结论，把它算成窟窿反而是虚报风险。
        """
        if self.verdict == "prose":
            return bool(self.text)
        return self.verdict == "ok" and self.confidence >= UNREAD_CONF and bool(self.text)

    @property
    def why_not(self) -> str:
        if self.verdict == "blank":
            return "这一小块几乎空白，没有字"
        if self.verdict == "failed":
            return "没读出任何式子"
        if self.confidence < UNREAD_CONF:
            return f"把握只有 {self.confidence:.0%}"
        return ""

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "crop": self.crop,
            "latex": self.latex,
            "text": self.text,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "sources": self.sources,
            "reasons": self.reasons,
            "alternatives": self.alternatives,
        }


@dataclass
class PhotoCheck:
    """一次拍照批改的完整结果。"""

    work: WorkCheck
    lines: list[LineRead] = field(default_factory=list)
    text: str = ""              # 拼好的过程文本，可以直接放进输入框再编辑
    page: str = ""
    split_note: str = ""
    notes: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    # 题面取自第几行（1 基；0 表示题面不是从某一行来的）。
    # 记下来是为了手改时能跟着改：用户把题面那一行的 `7` 改成 `6`，
    # 判定必须按改过的题算，不能还拿旧的。
    problem_line: int = 0

    @property
    def n_unread(self) -> int:
        return sum(1 for x in self.lines if not x.sure)

    def gallery(self) -> list[tuple[str, str]]:
        """逐行小图 + 说明，给界面上的画廊用。"""
        out = []
        for x in self.lines:
            mark = "·" if x.verdict == "prose" else ("✓" if x.sure else "?")
            body = x.text or "没认出"
            out.append((str(x.crop), f"{x.index} {mark} {body}"[:60]))
        return out

    def lines_html(self) -> str:
        """逐行小图铺成一列，按原始比例显示。

        不能用缩略图方格：行图是 574×102、672×83 这种又扁又长的横条，
        塞进方格后按高度放大十倍、横向只留中间 15%——学生看到的就是两个笔画的
        局部大特写，还以为识别把字认歪了。切得很准，是**显示**把它裁坏了。

        base64 直接嵌进 HTML：图只有几 KB，省掉文件服务和缓存失效的麻烦。
        """
        if not self.lines:
            return (
                "<div style='color:#6B6A65;font-size:13px;padding:10px 0;'>"
                "还没有切出任何一行。</div>"
            )
        blocks = []
        for path, caption in self.gallery():
            try:
                raw = Path(path).read_bytes()
            except OSError:
                continue
            mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
            b64 = base64.b64encode(raw).decode("ascii")
            parts = caption.split(" ")
            idx = parts[0]
            mark = parts[1] if len(parts) > 1 else ""
            body = " ".join(parts[2:]) if len(parts) > 2 else ""
            color = "#2F7D63" if mark == "✓" else (
                "#9A6A00" if mark == "?" else "#6B6A65"
            )
            blocks.append(
                "<div style='margin:0 0 12px;'>"
                f"<div style='font-size:12px;color:{color};margin:0 0 4px;'>"
                f"第 {idx} 行 {mark} {body}</div>"
                f"<img src='data:{mime};base64,{b64}' "
                "style='display:block;width:100%;height:auto;border:1px solid #E4E2DB;"
                "border-radius:8px;background:#fff;'>"
                "</div>"
            )
        return (
            "<div style='max-height:420px;overflow-y:auto;padding-right:6px;'>"
            + "".join(blocks)
            + "</div>"
        )

    def read_table_md(self) -> str:
        """识别层自己的账本：每行认成什么、几个来源、还有别的可能。

        这张表是给人核对识别结果的。批改结论错了，根子多半在这张表里。
        """
        if not self.lines:
            return "_（没有切出任何一行）_"
        out = [
            "| 行 | 认成 | 把握 | 来源 | 别的可能 |",
            "|---|---|---|---|---|",
        ]
        for x in self.lines:
            body = x.latex or "—"
            if x.verdict == "prose":
                body = f"{body}（说明行，不参与判定）"
            conf = f"{x.confidence:.0%}" if x.verdict not in ("blank", "prose") else "—"
            src = "、".join(x.sources) or "—"
            # 说明行的"别的可能"多半是备用小模型对着一行中文硬凑的公式，
            # 列出来只会让人以为系统在那行看见了公式。
            alt = "—" if x.verdict == "prose" else ("；".join(x.alternatives[:2]) or "—")
            out.append(
                f"| {x.index} | `{_cell(body)}` | {conf} | {_cell(src)} | {_cell(alt)} |"
            )
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "text": self.text,
            "split_note": self.split_note,
            "notes": self.notes,
            "n_unread": self.n_unread,
            "lines": [x.to_dict() for x in self.lines],
            "work": self.work.to_dict(),
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
        }


def _cell(text: str) -> str:
    return (text or "").replace("|", r"\|").replace("\n", " ")


# ---------------------------------------------------------------- 识别结果转文字


_PAIRS = (("\\[", "\\]"), ("\\(", "\\)"), ("$$", "$$"))

# 学生写推导时习惯每行开头加 ⇒ / → 表示"所以"。这不该参与比对：
# 两个识别来源一个读成 \Rightarrow、一个读成 arrow、一个干脆跳过，
# _canon 就把同一行判成三派，投票打平 → 整行作废。可这一行本来是对的。
# 行首推导符（以及 W04 那种"解："引导词）不承载任何数学信息，一律剥掉。
_DERIVE_PREFIX = re.compile(
    r"^(?:"
    r"\\(?:Rightarrow|Longrightarrow|longrightarrow|rightarrow|to|implies"
    r"|therefore|because)\b"
    r"|[⇒→⟹⇔∴∵]|=>|⇒|->|arrow(?![A-Za-z])|=+\s*\)"
    r")"
    r"[\s,，:：]*"
)


def _strip_derive_marks(s: str) -> str:
    """只剥行首连续出现的推导符，句子中间的箭头不动（那里可能有别的含义）。"""
    # 行首的等号绝不可能是算式的一部分。学生把"⇒"写成"-)"，模型有时读成
    # "=7"——后面那个数字是箭头下半截的影子，不是系数。只在这个数字与后面的
    # 内容之间有空隙时才当影子剥掉：`=7 x=2` 剥成 `x=2`，而 `= 2x+1`
    # （换行接着写的式子）里 2 与 x 是连着的，必须留下。
    s = re.sub(r"^=+\s*\d+\s+(?=\S)", "", s).lstrip()
    for _ in range(4):
        new = _DERIVE_PREFIX.sub("", s, count=1).lstrip()
        if new == s:
            break
        s = new
    return s


# 行尾的步骤编号：①…⑳、`\textcircled{5}`、`\tag{3}`、`(4)`。
#
# 学生给每一步编号是常态（"③-①得：5y=10 ④"），但编号不是数学内容。
# 留着它的代价比想象中大：这一页两派识别对这一行各读各的，
# 本该由 `_tiebreak_by_math` 按"哪个接得上邻行"裁掉——
# 而它要先把候选解析成表达式。行尾多个 ②，`x+3y=7②` 就解析不了，
# 裁决静默失效，整行退回"待核对"。识别明明有两个来源，却一条都对不上账。
_TAIL_MARK = re.compile(
    r"(?:"
    r"\\tag\*?\{[^{}]*\}"
    r"|\\textcircled\{[^{}]*\}"
    r"|[\u2460-\u2473]"                       # ①..⑳
    # `(4)` 前面必须有空白才算编号：`5y=10 (4)` 是编号，
    # 而 `x=2(3)` 里的 (3) 是乘 3，剥掉就把答案从 6 改成 2 了。
    r"|(?<=\s)[（(]\s*\d{1,2}\s*[)）]"
    r")\s*$"
)


def _strip_tail_marks(s: str) -> str:
    """剥行尾编号，允许连着剥几个（"… ③ ④"这种写两遍的）。"""
    for _ in range(3):
        # 编号前面常有一个排版用的强制空格 `\ (4)`，剥掉编号后剩个光秃秃的
        # 反斜杠——它自己就是一个不合法的 LaTeX 命令，照样让整行解析失败。
        new = _TAIL_MARK.sub("", s).rstrip(" \\")
        if new == s:
            break
        s = new
    return s


def readable(latex: str) -> str:
    """把一条识别结果还原成"能再喂回解析器"的文字。

    不能直接用 `route.plain()`：它会把所有花括号删掉，`\\frac{1}{2}` 变成
    `\\frac12`，那是另一个东西了。这里只剥两类东西——`\\text{...}`（学生写的
    中文说明）和 `\\left/\\right`（排版用的括号尺寸），数学结构一个不动。
    """
    s = (latex or "").strip()
    if not s:
        return ""
    # 有些来源会用 markdown 包一层代码格式：`` `7y=2` ``。
    # 反引号不是数学符号，留着会让整行解析失败。
    s = s.strip("`").strip()
    # 整行被包在 \[...\]、$...$ 里：外层是排版标记，不是内容
    for left, right in _PAIRS:
        if s.startswith(left) and s.endswith(right) and len(s) > len(left) + len(right):
            s = s[len(left) : -len(right)].strip()
    if len(s) > 2 and s.startswith("$") and s.endswith("$") and not s.startswith("$$"):
        s = s[1:-1].strip()
    # \text{...} 会套娃，一层层剥到不动为止
    for _ in range(6):
        new = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
        if new == s:
            break
        s = new
    # `\left` / `\right` 是括号尺寸标记，要剥；但不能吃掉 `\rightarrow` 的前半截——
    # 不带边界的 `\right` 会把它削成 `arrow`，识别结果里那些莫名的 arrow 就是这么来的。
    s = re.sub(r"\\(?:left|right)(?![A-Za-z])\s*", "", s)
    s = re.sub(r"\\[,;:!]", " ", s)
    s = re.sub(r"\\\s", " ", s)          # `\ ` 是排版用的强制空格，没有数学含义
    s = re.sub(r"\\(?:quad|qquad)", " ", s)
    # 编号放在最后剥。模型写的是 `\text{①}`、`\quad ①`、`\quad(2)`，
    # 前面几步没走完时末尾还是个 `}` 或反斜杠，正则够不着，编号就整个留在式子里，
    # 解析和"按数学一致性裁决"跟着一起失效。这一步必须看到剥干净后的样子。
    s = _strip_tail_marks(s)
    s = re.sub(r"[ \t]{2,}", " ", s).strip()
    # 行首孤立的右括号：学生爱用一个大括号把连着几步括在一起，切行时
    # 右半边落进了下一行开头（W05 第二行 `)x²-6=5x`）。它不是算式的一部分，
    # 留着会让整行解析失败，退化成"读不出算式"。
    s = re.sub(r"^[\]\)）】}]+\s*", "", s)
    # 行首孤立的系数 1：`1y=2` 和 `y=2` 是同一个式子，学生不会写 `1y`。
    # 这个 1 多半是上一行那个 `}` 的右半截被读成了数字——那一竖在框里，
    # 模型答的是它看见的东西。只吃"1 后面紧跟字母"，`12x` 这种不动。
    s = re.sub(r"^1(?=[a-zA-Z])", "", s)
    return _strip_derive_marks(s)


# ---------------------------------------------------------------- 逐行识别


def _is_blank(image) -> bool:
    """几乎是纯色的图（笔影、纸边、桌面）不该送去识别，白花钱还会带回噪音。"""
    try:
        g = image.convert("L")
        w, h = g.size
        if w < 8 or h < 6:
            return True
        stat = __import__("numpy").asarray(g)
        return float(stat.std()) < 4.0
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- 逐行投票

_PUNCT = str.maketrans({
    "（": "(", "）": ")", "＝": "=", "＋": "+", "－": "-", "×": "*", "÷": "/",
    "，": ",", "；": ";", "：": ":", "²": "^2", "³": "^3", "·": "*",
})


def _canon(latex: str) -> str:
    """把一条候选压成"比对键"。只用来判断两个来源说的是不是同一件事。

    不能拿 `verify.cluster()` 代替：它把候选解析成**一个** SymPy 表达式来判等价，
    而学生的一行经常是"两边同乘2得 2x²-10x+12=0"或"x=2 或 x=3"——整体解析必失败，
    于是两条正确的识别一起被判废，看起来像"没认出来"，实际是判据用错了。

    这里只做与数学无关的归一：空白、全角、`²` 与 `^2`。数学结构一个不动，
    所以 `\\frac{1}{2}x` 和 `\\frac{x}{2}` 会被当成两条不同的话——宁严不宽，
    这两种写法在投票阶段判成"不一致"，界面上会显示"待核对"，不会误判成学生错。
    """
    s = readable(latex).translate(_PUNCT)
    # 上标只包一个字符时花括号可有可无：`x^{v}` 与 `x^v` 是同一条。
    # 原来只对数字做归一，于是 W13 的 `x^{v}=x`（qwen）和 `x^v=x`（glm）
    # 被判成两派，明明两台机器读的是一模一样的东西。
    s = re.sub(r"\^\{([0-9A-Za-z])\}", r"^\1", s)
    s = re.sub(r"\\(?:times|cdot)", "*", s)
    s = re.sub(r"\\div", "/", s)
    s = re.sub(r"\s+", "", s)
    s = s.strip("。.、,")
    # 行尾悬空的等号：`2x^2-10x+12=` 与 `2x^2-10x+12=0` 是同一个意思——
    # 手写一行写到最后一个字符时模型常漏掉它，两台机器一台漏一台没漏，
    # 就被判成"读出了不同的内容"。按"等式右边默认是 0"补齐，只影响比对键。
    if s.endswith("="):
        s += "0"
    return s


_CJK = re.compile(r"[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]+")

# 一条式子必然带一个关系符或结构记号；只带中文的说明行一个都没有。
_MATH_STRUCT = re.compile(r"[=<>≤≥≠≈]|\\frac|\\sqrt|\\sum|\\int|\^|_")


def _is_prose(text: str) -> bool:
    """这一行是"两边同时乘2得"这种说明，还是真式子。

    判据是**有没有数学结构**，不是有没有中文——学生常在式子后面跟一句
    "所以"、"所求为"，那些行照样是式子。
    """
    if not text or not _CJK.search(text):
        return False
    return not _MATH_STRUCT.search(text)


def _math_key(latex: str) -> str:
    """只留数学部分的比对键：中文说明、全角标点、行首多余符号一律不计。

    实拍里同一行两个来源经常只在中文上分家——
    "两边同时乘2得"被读成"有边同时乘2得"、"所以"被读成"所求为"，
    式子一模一样。这种分歧不该把整行判成"待核对"：学生没写错，
    错的是一台机器看歪了一笔，另一台看对了，两边其实都认出了同一行式子。

    数学部分仍然逐字比对，一个符号都不放宽——这是"宁严不宽"的底线。
    """
    s = _CJK.sub("", _canon(latex))
    # 中文被拿走后会露出前后括号/冒号，剥掉行首的孤立标点：
    # `：(x-2)(x-3)=0` 与 `(x-2)(x-3)=0` 是同一行。
    s = s.lstrip(":：,，;；、()（）[]【】")
    # 行首孤立的右括号：W05 二、三行被学生用大括号连成一组，
    # 切行时括号的右半边落进了下一行开头（`)x^{2}-6=5x`）。
    s = s.rstrip(":：,，;；、")
    return s.strip()


def _vendor(source: str) -> str:
    s = (source or "unknown").lower()
    for v in ("simpletex", "qwen", "gpt", "claude", "glm", "gemini", "fixture", "manual"):
        if s.startswith(v):
            return v
    return s


def _weak(c: Candidate) -> bool:
    """这条候选是不是备用小模型读的。

    除了 `Candidate.weak`，还要认出"改用 XXX"这句备注：`weak` 字段是后加的，
    早先落盘的缓存里没有它，只认字段会让同一行在"缓存命中"和"重新识别"时
    得到不同结论——这种随缓存漂移的行为比读错一行更难查。
    """
    return bool(getattr(c, "weak", False)) or "改用" in (c.note or "")


def _vote(
    cands: list[Candidate], require_corroboration: bool = True
) -> tuple[str, float, list[str], str, list[str]]:
    """按"文字一致"投票，返回 (选中的 latex, 一致度, 备选, 结论, 说明)。

    证据是两个来源是否读出了同一句话，而不是这一行能不能解析成表达式。

    `require_corroboration` 是这条链路里最重要的一条规矩的来源：**判错要有旁证**。
    只有一个来源读出内容时，它可能是在编——对着空白或涂改块，视觉模型经常
    硬凑出一个像样的公式。这种"独一份"的识别可以用来展示，不可以用来定对错。
    所以只要环境里有两个以上的识别器，单来源一律降为"待核对"。
    """
    groups: dict[str, list[Candidate]] = {}
    for c in cands:
        if not (c.latex or "").strip():
            continue
        key = _canon(c.latex)
        if not key:
            continue
        groups.setdefault(key, []).append(c)
    if not groups:
        return "", 0.0, [], "failed", []

    # 选赢家：先比票数，再比"有没有主模型参与"。备用小模型读出来的东西
    # 常常是对着空白或中文硬凑的公式，平票时让它赢会把整行拖成"待核对"。
    best_key, members = max(
        groups.items(),
        key=lambda kv: (
            len(kv[1]),
            any(not _weak(m) for m in kv[1]),
            len(kv[0]),
        ),
    )
    total = sum(len(g) for g in groups.values())
    conf = len(members) / total

    # 同一家的两个采样一致，是自一致性，不是交叉验证（和 verify 同一条规矩）
    vendors = {_vendor(m.source) for m in members}
    independent = len(vendors) >= 2
    if not independent:
        conf = min(conf, 0.8)

    # 旁证要按"式子"数，不能只数赢的那一组。
    # W04 第四行 qwen 读"所以：(x-2)(x-3)=0"、glm 读"所求点：(x-2)(x-3)=0"，
    # 式子一模一样，只是中文措辞不同，于是分成两组、赢的那组只剩一个来源，
    # 被判成"没有旁证"。可旁证明明是有的——两台机器都认出了同一条式子。
    # 分组用的是整行比对键（含中文），所以措辞一分家，票就散了；
    # 这里按数学部分重新数一遍来源。
    win_math = _math_key(members[0].latex)
    peers = [c for g in groups.values() for c in g if _math_key(c.latex) == win_math]
    peer_vendors = {_vendor(c.source) for c in peers}
    corroborated = independent or (len(peer_vendors) >= 2 and len(peers) > len(members))

    verdict = "ok"
    notes: list[str] = []
    if len(groups) > 1:
        runner = max(len(g) for k, g in groups.items() if k != best_key)
        if runner >= len(members):        # 两派打平，裁不了
            # 先看是不是"假分歧"：中文说明的措辞不同，式子一模一样。
            # 这不是学生写得不清楚，是两台机器各看歪了同一笔中文——
            # 把它们判成"待核对"，等于让措辞差异冒充数学分歧。
            math_keys = {_math_key(g[0].latex) for g in groups.values()}
            math_keys.discard("")
            if len(math_keys) == 1 and len({_canon(g[0].latex) for g in groups.values()}) > 1:
                notes.append("来源之间只有中文说明的措辞不同，式子部分一致")
                conf = max(conf, 0.8)
            elif len(math_keys) == 1:
                notes.append("来源之间的差异只在排版记号，内容一致")
                conf = max(conf, 0.8)
            else:
                verdict = "review"
                conf = min(conf, 0.5)
                notes.append("两个来源读出了不同的内容，裁不了")
    if require_corroboration and not corroborated:
        verdict = "review"
        conf = min(conf, 0.5)
        notes.append(f"只有 {'、'.join(sorted(vendors))} 一个来源读出这一行，没有旁证")

    others = [
        readable(g[0].latex)
        for k, g in sorted(groups.items(), key=lambda kv: -len(kv[1]))
        if k != best_key and g[0].latex
    ][:3]
    return members[0].latex, conf, others, verdict, notes


def _same_equation(a: str, b: str) -> bool:
    """两行是不是"同一个方程"（解集相同，允许两边同乘一个常数）。

    用途只有一个：两个识别源对同一行读出两种写法时，判断哪一种接得上上一行。
    这比"再拉一个模型投票"硬——投票只能说明两台机器看法一致，同解能说明
    这一步在数学上成立。
    """
    from .latex import parse, systems_equivalent
    from .verify import to_expr

    pa, pb = parse(a), parse(b)
    if not (pa.ok and pb.ok):
        return False
    # 方程组单独比：把各条相加再比会抵消约束，两个不同的组会被判成同一行。
    if pa.kind == "system" or pb.kind == "system":
        return bool(
            pa.kind == pb.kind == "system" and systems_equivalent(pa.obj, pb.obj)
        )
    ea, eb = to_expr(pa), to_expr(pb)
    if ea is None or eb is None:
        return False
    try:
        if ea == eb:
            return True
        if ea == 0 or eb == 0:
            return False
        # 两边同乘一个非零常数是同一个方程（2x²-10x+12=0 与 x²-5x+6=0 同解），
        # 系数比是常数就成立；比出来带 x 的，是两个不同的方程。
        ratio = sympy.simplify(ea / eb)
        return bool(ratio.is_number) and ratio.is_zero is False
    except Exception:  # noqa: BLE001
        return False


def _neighbor(lines: list[LineRead], i: int) -> str:
    """给第 i 行找一条"能拿来对账"的邻行：先看上一行，没有就看下一行。

    上一行优先是因为推导是往下走的；第一行没有上一行时才往上看——
    W04 第一行 `x²-5x+6=0` 的第二台机器读成了 `x-5x+6=0`（少个上标），
    而第三行 `2x²-10x+12=0` 正好是它乘 2 的结果。没有这条"往下看"，
    第一行就永远裁不了，整页跟着判不了——可证据就在下面那行摆着。

    往上看只跳过说明行以外的东西没意义：说明行（prose）不是式子，跳过它
    继续往上找是安全的，因为它不承载数学内容。

    返回的是 `text`（清洗过的）而不是 `latex`：同解判断要先解析，
    而原始 latex 里还挂着 `\\text{②}`、`\\quad(2)` 这类编号，一解析就失败，
    这条裁决会静默失效——两个来源明明都读出了能对账的式子，却裁不动。
    """
    for j in range(i - 1, -1, -1):
        if lines[j].verdict == "prose":
            continue                      # 说明行不承载数学内容，跳过它不算跨窟窿
        return lines[j].text if lines[j].verdict == "ok" else ""
    for j in range(i + 1, len(lines)):
        if lines[j].verdict == "prose":
            continue
        return lines[j].text if lines[j].verdict == "ok" else ""
    return ""


def _truncated(text: str) -> bool:
    """这一行读到关系符就断了：`x=`。缺了右边，一定不是完整的读法。"""
    s = (text or "").rstrip().rstrip("$")
    return s.endswith(("=", "<", ">", "\\le", "\\ge", "\\ne"))


def _tiebreak_by_math(lines: list[LineRead]) -> None:
    """两派读法不同时，用"哪一派接得上邻行"来裁，而不是让两个模型继续投票。

    平票只说明两台机器看法不同，不说明哪台对；但这一行该写什么，邻行
    其实已经约束住了——`(x-2)(x-3)=0` 展开正好是上一行的 `x²-5x+6=0`，
    `(x-2)(x-5)=0` 展开不是。这不是猜，是验证。

    三种结局，各有各的道理：
    - 恰好一派接得上 → 裁给它；
    - 两派都接得上 → 它们是同解的不同写法（`(x-2)(x-3)=0` 与
      `-2(x-2)(x-3)=0`），对批改完全等价，判 ok 并说明；挑短的那个，
      因为"多解释一个字"总比"多解释三个字"更可能没看错；
    - 两派都接不上 → 学生这一步本身就在错，或者邻行自己也读错了，
      这时候硬裁会把错误固化成"已确认"。
    """
    for i, lr in enumerate(lines):
        if lr.verdict != "review" or not lr.alternatives:
            continue
        # 候选一律用 `text`（清洗过的），和 `_neighbor` 保持同一套标准。
        # 一边用原始 latex、一边用清洗后的文本，同解判断必然对不上——
        # 这种"半截 bug"最麻烦：代码在跑，日志也正常，就是不生效。
        cands = [c for c in dict.fromkeys([lr.text, *lr.alternatives]) if c]
        anchor = _neighbor(lines, i)
        fits = [c for c in cands if anchor and _same_equation(anchor, c)]

        if not fits:
            # 接不上邻行时还有一条路：这一行本来就缺了半截。
            # 这一页 `x=1` 被一个来源读成 `x=`——关系符后面什么都没有，
            # 不可能是完整读法；另一个来源读全了。没有理由留着残缺的那个。
            if len(cands) < 2:
                continue
            whole = [c for c in cands if not _truncated(c)]
            if len(whole) != 1:
                continue
            cut = [c for c in cands if _truncated(c)]
            lr.text = lr.latex = whole[0]
            lr.verdict = "ok"
            lr.confidence = max(lr.confidence, 0.8)
            lr.reasons = [
                f"两个识别源一处读成 `{cut[0]}`、一处读成 `{whole[0]}`；"
                f"`{cut[0]}` 停在关系符上没有右边，取读完整的那一个"
            ]
            continue

        if len(fits) == 1:
            rejected = [c for c in cands if c != fits[0]]
            lr.text = lr.latex = fits[0]
            lr.verdict = "ok"
            lr.confidence = max(lr.confidence, 0.8)
            lr.reasons = [
                "两个识别源读法不同，按数学一致性裁决："
                f"只有 {fits[0]} 与相邻的 {anchor} 同解"
                + (f"（另一种 {rejected[0]} 不成立）" if rejected else "")
            ]
        else:
            best = min(fits, key=lambda c: (len(_canon(c)), cands.index(c)))
            lr.text = lr.latex = best
            lr.verdict = "ok"
            lr.confidence = max(lr.confidence, 0.8)
            lr.reasons = [
                "两个识别源写法不同，但都与相邻的 "
                f"{anchor} 同解，对批改等价，取 {best}"
            ]


def _prose_ok(latex: str, cands: list[Candidate]) -> bool:
    """这一行能不能按"说明行"放过：选中的读法是纯文字，且主模型也没读出式子。

    备用小模型偶尔会对着中文硬凑一条公式，拿它的幻觉去否掉主模型的
    "这是一句说明"，正好把最可靠的那一路废掉，所以只看主模型的意见。
    """
    if not _is_prose(readable(latex)):
        return False
    strong = [readable(c.latex) for c in cands if (c.latex or "").strip() and not _weak(c)]
    return bool(strong) and all(_is_prose(t) for t in strong)


def _read_one(index: int, crop_path: str, recs, k: int, require_corroboration: bool = True) -> LineRead:
    cands: list[Candidate] = []
    for rec in recs:
        try:
            cands.extend(rec.recognize(crop_path, k=k))
        except Exception as exc:  # noqa: BLE001
            cands.append(
                Candidate(latex="", source=rec.name, note=f"{type(exc).__name__}: {exc}")
            )
    latex, conf, others, verdict, notes = _vote(cands, require_corroboration)
    notes = notes + [c.note for c in cands if c.note][:2]
    sources = sorted({c.source for c in cands if (c.latex or "").strip()})

    # 说明行不该拖累整页：它不参与对错判定，认成"有边"还是"两边"不影响结论。
    if verdict != "ok" and latex and _prose_ok(latex, cands):
        verdict = "prose"
        strong = [readable(c.latex) for c in cands if (c.latex or "").strip() and not _weak(c)]
        notes = [
            "这一行只有文字说明、没有式子，不参与对错判定；"
            "不同来源措辞不一致（" + " / ".join(dict.fromkeys(strong)) + "）不影响结论"
        ]
    return LineRead(
        index=index,
        crop=crop_path,
        latex=latex,
        text=readable(latex),
        verdict=verdict,
        confidence=conf,
        sources=sources,
        reasons=notes,
        alternatives=others,
    )


def read_page(
    image: str | Path,
    *,
    k: int = 1,
    # 并发 2 而不是 4：两个识别源同时被 4 路并发打，免费档的 glm 会 429，
    # 一旦限流那一路就整页缺席，投票退化成"只有 qwen 一家"→ 全部待核对。
    # 实测并发 2 时 glm 能稳定返回；慢一倍换真交叉验证，值。
    workers: int = 2,
    allow_fixture: bool = True,
) -> tuple[PageSplit, list[LineRead], dict[str, float]]:
    """切行 + 逐行识别。不批改，只回答"这一页上写了什么"。

    单独拆出来是因为它也能当独立工具用：核对识别结果时不想连带跑一遍化简。
    """
    t0 = time.time()
    page = split_page(image)
    crops = page.crops
    if len(crops) > MAX_LINES:
        crops = crops[:MAX_LINES]
        page.note = (page.note + "；" if page.note else "") + (
            f"这一页切出 {len(page.crops)} 行，只认前 {MAX_LINES} 行。"
            "整页作业建议分几张拍，一次拍太多既慢又容易串行。"
        )

    out_dir = WORK_DIR / Path(image).stem
    # 上一轮的小图先清掉：同一个文件名反复拍，残留的旧图会让人对着错误的小图核对
    if out_dir.exists():
        for old in out_dir.glob("*.jpg"):
            try:
                old.unlink()
            except OSError:
                pass
    paths = save_crops(PageSplit(page=page.page, crops=crops, size=page.size), out_dir,
                       stem=Path(image).stem)
    timings = {"split": time.time() - t0}

    recs = build_ensemble(k=k, allow_fixture=allow_fixture)
    if not recs:
        raise ValueError(
            "还没有可用的识别通道，拍照批改至少要一个能读图的模型。"
            "请在 .env 里填 DASHSCOPE_API_KEY 或 SIMPLETEX_TOKEN。"
        )

    t0 = time.time()
    results: list[LineRead] = []
    todo: list[tuple[int, str]] = []
    for crop, path in zip(crops, paths):
        if _is_blank(crop.image):
            results.append(
                LineRead(index=crop.index, crop=path, verdict="blank",
                         reasons=["这一小块几乎空白"])
            )
        else:
            todo.append((crop.index, path))

    # 逐行并发：单行两个来源、六行就是一打请求，串着等要等半分钟
    require = len(recs) > 1        # 只有一个识别器时"要旁证"等于"永远判不了"
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(todo) or 1))) as pool:
        futures = [
            pool.submit(_read_one, idx, path, recs, k, require) for idx, path in todo
        ]
        results.extend(f.result() for f in futures)
    results.sort(key=lambda r: r.index)
    _tiebreak_by_math(results)
    timings["recognize"] = time.time() - t0
    return page, results, timings


# ---------------------------------------------------------------- 批改


def _by_step(work: WorkCheck, lines: list[LineRead]) -> dict[int, LineRead]:
    return {i: lr for i, lr in enumerate(lines, start=1) if i <= len(work.steps)}


def _base_sound(work: WorkCheck, by_step: dict[int, LineRead], i: int) -> bool:
    """第 i 步的判定基准可靠吗。

    基准是往上找最近一条等式。一路找过去，只要跨过一条"读不出"或"没认准"的行，
    这个基准就不可靠了——那种时候系统是隔着窟窿在比，比出来的反例依然是准确的，
    但它对准的是"哪一行"，而不是"学生哪一步错了"。
    """
    j = i - 1
    while j >= 0:
        s = work.steps[j]
        if s.kind == "prose":
            j -= 1
            continue
        if s.kind == "unparsed":
            return False
        lr = by_step.get(s.idx)
        return True if lr is None else lr.sure
    return True


def _apply_uncertainty(work: WorkCheck, lines: list[LineRead]) -> list[str]:
    """把"没认准"的信息灌进判定里。这是整个模块的要害，改之前先想清楚。

    三条规则，全部围绕一件事：**断言必须有可靠的依据**，方向不限——
    冤枉人和错夸人是同一种装作确定。
      · 这一行自己没认准 → 判"不成立"不作数（反例作废），判"成立"也不作数；
      · 判定基准跨过了没认出/没认准的行 → 隔着窟窿比出来的结论两个方向都不作数；
      · 真错/真成立的结论一条都不剩 → 不能再给相应的总判定（unread）。
    """
    notes: list[str] = []
    unsure = {lr.index for lr in lines if not lr.sure}
    if not unsure:
        return notes

    by_step = _by_step(work, lines)
    demoted: list[int] = []
    weakened = 0

    def demote(step, why: str) -> None:
        step.verdict = "suspect"
        step.counter = ""          # 反例建立在没认准的字上，就不再是证据
        # 原来的 reason 是"这一步和上一步不同解"——那是按认错的字算出来的，
        # 留着会和"待核对"自相矛盾，必须一起换掉。
        step.reason = why
        step.row = ["? 待核对", why]
        demoted.append(step.idx)

    for i, step in enumerate(work.steps):
        lr = by_step.get(step.idx)
        own_sure = lr is None or lr.sure
        base_ok = _base_sound(work, by_step, i)
        if step.verdict in ("wrong", "partial"):
            # 一、这一行自己没认准
            if not own_sure:
                demote(step, f"这一行没认准（{lr.why_not or '把握不足'}），先核对写的是什么，再判对错")
                continue
            # 二、判定基准不可靠：隔着窟窿比出来的"不成立"不作数
            if not base_ok:
                demote(step, "往上找最近一条能对照的式子时跨过了没认出来的行，这个结论不作数")
        elif step.verdict == "ok" and step.kind not in ("prose",):
            # "成立"同样是断言。依据没认准就夸这步对，和冤枉人是同一种
            # 装作确定，只是方向相反——都要收回来。
            if not own_sure or not base_ok:
                weakened += 1
                step.verdict = "unknown"
                step.reason = (
                    "这一行"
                    + ("自己" if not own_sure else "的对照基准")
                    + "没认准，对错都确认不了，等核对完再看"
                )
                step.row = ["? 跟着待核对", step.reason]

    real = [s.idx for s in work.steps if s.verdict in ("wrong", "partial")]
    holes = sorted(unsure)
    notes.append(
        f"第 {'、'.join(map(str, holes))} 行没认准，已在表里标「? 待核对」；"
        "这些行不产生对错结论。"
    )
    if demoted:
        notes.append(
            f"原本判「不成立」的第 {'、'.join(map(str, demoted))} 行，因为识别不可靠已撤回。"
            "冤枉一次，后面所有判定都不作数了——所以这里宁可不说。"
        )
    elif real:
        # 真错的行还在，但窟窿可能正是第一处错：错误从第几行开始，说不准
        notes.append(
            f"注意：第 {'、'.join(map(str, holes))} 行没认出来，"
            "真正开始错的那一行可能比表里标的更早。"
        )

    # 结论重算：真错的一步都不剩，就不能再给"有错"的总判定；
    # "成立"被削弱过，就不能再给"过程成立"的总判定——两个方向都不许装作确定。
    if not real and work.grade in ("wrong", "fake", "partial"):
        work.grade = "unread"
        work.headline = f"{len(holes)} 行没认准，判不了对错"
        work.error_type = ""
        work.advice = (
            "这不是你写错了，是这几行没认出来。把下面「有行没认准」里那几行改成你写的式子，"
            "点「按我改的重新判」——不用重新拍照，改哪行算哪行。"
        )
    elif weakened and work.grade == "pass":
        work.grade = "unread"
        work.headline = f"{len(holes)} 行没认准，{weakened} 步的「成立」确认不了"
        work.advice = (
            "有几步是拿没认准的行当依据判成「成立」的，这种「对」不能算数。"
            "先核对那几行，或者改成文字重判一次。"
        )
    elif real:
        work.first_error = real[0]
    return notes


# 行首的系数：`2x^2-5x=6` 里的 2。学生写系数很常见，所以不能一律当垃圾，
# 只有在"去掉它就和上一行同解"时才拿来怀疑自己看错了。
_LEAD_COEFF = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*(?=[A-Za-z\\])")


def _soften_digit_accusations(work: WorkCheck) -> list[str]:
    """行首差一个数字就不判错——那一笔很可能是行首的推导符，不是系数。

    W17 第三行学生写的是 `x²-5x=6`，两台识别器都读成 `2x²-5x=6`：行首那个
    "⇒"被读成了数字 2。批改引擎照实算，得出"这一步和上一步不同解"，
    反例、结论一应俱全——可学生没写错，是开头多了一个不是他写的 2。

    判据不能是"看着像"：只有当**去掉这个系数之后与上一行同解**时才撤回。
    真的写错系数（`2x²` 写成 `3x²`）时，去掉系数多半也接不上上一行，
    照常判错。宁可漏抓，不可冤枉——这条链路上两个方向的代价差得远。
    """
    notes: list[str] = []
    for i, step in enumerate(work.steps):
        if step.verdict != "wrong" or step.kind != "equation" or not step.latex:
            continue
        prev = next(
            (work.steps[j] for j in range(i - 1, -1, -1)
             if work.steps[j].kind in ("equation", "assign") and work.steps[j].latex),
            None,
        )
        if prev is None:
            continue
        stripped = _LEAD_COEFF.sub("", step.latex, count=1)
        if stripped == step.latex:
            continue
        if not _same_equation(prev.latex, stripped):
            continue
        step.verdict = "suspect"
        step.counter = ""
        step.reason = (
            f"行首多出一个数字：去掉它之后这一行与上一行同解。"
            f"多半是把行首的推导符看成了系数，这一步先不判错"
        )
        step.row = ["? 待核对", step.reason]
        notes.append(
            f"第 {step.idx} 行按去掉行首系数的读法能和上一行对上，"
            "没有据此判错——一个数字之差不足以断定学生写错。"
        )

    real = [s.idx for s in work.steps if s.verdict in ("wrong", "partial")]
    if notes and not real and work.grade in ("wrong", "fake", "partial"):
        work.grade = "unread"
        work.headline = "唯一一处不成立是靠一个可疑的数字判的，撤回"
        work.error_type = ""
        work.advice = (
            "这一处对不上，差的只是行首一个数字——很可能是把推导符看成了系数。"
            "照原样重拍也行，或者直接在下面「有行没认准」里改成正确写法，"
            "再点一次「按我改的重新判」。"
        )
    return notes


def _too_many_holes(work: WorkCheck, lines: list[LineRead]) -> bool:
    judgeable = [s for s in work.steps if s.kind not in ("prose",) and s.verdict != "note"]
    if not judgeable:
        return False
    holes = sum(1 for lr in lines if not lr.sure)
    return holes / max(1, len(judgeable)) > MAX_UNREAD_SHARE


def _problem_line(lines: list[LineRead], problem: str) -> int:
    """题面是从哪一行来的（1 基行号，找不到就是 0）。

    只做一件事：把"题面"这句话和行文本对上号。手改时靠它把题面跟着换——
    否则用户把 `x²-5x+7=0` 改成 `x²-5x+6=0`，判定还在拿旧题算，
    明明改对了反而得出「解不全，少了两个复根」，比不改更糟。
    """
    from .latex import parse

    target = (problem or "").strip()
    if not target:
        return 0
    pt = parse(target)
    key = pt.sympy_str if pt.ok else readable(target)
    for lr in lines:
        body = (lr.text or "").strip()
        if not body:
            continue
        if readable(body) == readable(target):
            return lr.index
        q = parse(body)
        if (q.sympy_str if q.ok else readable(body)) == key:
            return lr.index
    return 0


def _withdraw_on_bad_problem(
    work: WorkCheck, kept: list[LineRead], pidx: int
) -> list[str]:
    """题面那一行没认准时，把整份结论撤掉。

    批改里每一个"错"最后都落在"和题面对不上"：标准解是从题面算出来的，
    反例也是相对题面算的。题面本身没认准，后面推得再严密，也只是一串
    看起来很确定的猜测。

    最该撤的是"伪推导"：它的意思是"答案对、过程错"，这个判断**完全**靠
    标准解支撑。题面错了，"你过程是凑的"就是一顶扣错了的帽子——
    比不判糟得多。
    """
    if not pidx:
        return []
    bl = next((lr for lr in kept if lr.index == pidx), None)
    if bl is None or bl.sure:
        return []

    work.grade = "unread"
    work.error_type = ""
    work.expected = []
    work.first_error = -1
    # 标准解已经撤了，"见下方正确做法"这句话就指着一片空白，得一起删掉
    work.notes = [n for n in work.notes if "标准解法见下方" not in n]
    work.headline = f"第 {pidx} 行是题面，这一行没认准，整份判不了"
    for s in work.steps:
        if s.verdict in ("wrong", "partial"):
            s.verdict = "suspect"
            s.counter = ""
            s.reason = "题面没认准，这一步是相对一道不确定的题判的，不作数"
            s.row = ["? 待核对", s.reason]
    work.advice = (
        "标准解、以及每一处「和题对不上」的判断，都是从题面算出来的。"
        "题面这一行没认准，这些结论就都悬空了。"
        "把题面改成正确的式子（或者填到上面的输入框里），再点一次「按我改的重新判」，"
        "不用重拍。"
    )
    return [
        f"第 {pidx} 行是题面，这一行没认准：基于题面的结论（标准解、伪推导、"
        "漏解）全部撤回，逐行结果仅供参考。"
    ]


def _judge(
    kept: list[LineRead],
    *,
    problem: str,
    page: str,
    problem_image: str | None = None,
    split_note: str = "",
    timings: dict[str, float] | None = None,
) -> PhotoCheck:
    """拿已经读好的行去批改。

    识别和判定拆开，是因为它们成败完全不同：识别要钱要时间、还可能重来一次
    就换个结果；判定是纯本地计算，改一行字重算一次没有任何代价。
    界面上的"手动改成我写的式子再判一次"就是复用这里——不重新拍照、不重新识别。
    """
    timings = {} if timings is None else timings
    t0 = time.time()
    texts = [lr.text if lr.text else PLACEHOLDER for lr in kept]
    work = grade(texts, problem=problem or "", problem_image=problem_image or "")
    timings["grade"] = time.time() - t0
    # 题面对应哪一行。显式给的题面也可能就是某一行的文字（手改后就是这样），
    # 所以这里不区分"系统认的"还是"用户填的"，能对上就记下来。
    pidx = _problem_line(kept, work.problem)

    notes = _apply_uncertainty(work, kept)
    # 判错的门槛：先看这条"错"是不是建立在一个可能是推导符的数字上
    notes += _soften_digit_accusations(work)
    if _too_many_holes(work, kept):
        work.grade = "unread"
        work.error_type = ""
        work.headline = (
            f"这一页 {len(kept)} 行里有 {sum(1 for lr in kept if not lr.sure)} 行没认准，"
            "整份判不了"
        )
        work.advice = (
            "相邻两式比较要求上下行是连着的。中间空一格，后面的判断全部作废，"
            "所以整份不给结论——这比给一个可能是错的结论有用。"
            "没认准的那几行，可以在下面「有行没认准」里改成你写的式子，"
            "再点「按我改的重新判」，不用重拍。"
        )
        notes.append("没认准的行太多了，整份不给总判定；逐行结果仍列在表里，仅供参考。")

    # 题面没认准：上面每一步都是按"和题面比"算的，题面悬空就得全撤。
    # 放在最后是因为它的结论最具体，该由它来写总判定那句话。
    notes += _withdraw_on_bad_problem(work, kept, pidx)

    manual = sorted(
        lr.index for lr in kept
        if MANUAL_SOURCE in lr.sources or CONFIRM_SOURCE in lr.sources
    )
    if manual:
        # 这句话必须留在结论里。人工改过的行系统没看见原件，它按用户给的式子
        # 算得再准，也只能保证"你给的这个过程内部自洽"。
        changed = sorted(lr.index for lr in kept if MANUAL_SOURCE in lr.sources)
        checked = sorted(lr.index for lr in kept if CONFIRM_SOURCE in lr.sources)
        bits = []
        if changed:
            bits.append(f"第 {'、'.join(map(str, changed))} 行是你手动输入的")
        if checked:
            bits.append(f"第 {'、'.join(map(str, checked))} 行你核对过、没改动")
        if pidx and pidx in (set(changed) | set(checked)):
            # 这一行不只是"某一步"，它是判定的基准。不说清楚，用户会以为
            # 改的是无关的一行，看到后面成片判错也不知道为什么。
            bits.append(f"第 {pidx} 行同时被当作题面，后面每一行的对错都是相对这道题算的")
        notes.append("；".join(bits) + "。系统没核对原件，这几行按你给的式子算，"
                     "其余各行仍按相邻两式是否等价来判。")

    if split_note:
        notes.append(f"切行：{split_note}")

    # 说明并进 WorkCheck，界面上「错在哪」那一段会把它们一起显示出来；
    # 单独留在 PhotoCheck 里的话，用户看不到"为什么判不了"。
    for n in notes:
        if n not in work.notes:
            work.notes.append(n)

    return PhotoCheck(
        work=work,
        lines=kept,
        text="\n".join(texts),
        page=page,
        split_note=split_note,
        notes=notes,
        timings=timings,
        problem_line=pidx,
    )


def check(
    image: str | Path,
    problem: str = "",
    *,
    k: int = 1,
    workers: int = 2,     # 见 read_page：并发过高会把第二路识别源打到限流
    allow_fixture: bool = True,
) -> PhotoCheck:
    """拍一张解题过程，批改它。

    problem 为空时，第一行能读出来的等式会被当成题面（作业本上第一行通常就是原题）。
    """
    image = str(image)
    page, lines, timings = read_page(
        image, k=k, workers=workers, allow_fixture=allow_fixture
    )
    # 空白小块不进过程文本，也不进步骤表——它本来就不是"一行"
    kept = [lr for lr in lines if lr.verdict != "blank"]
    return _judge(
        kept,
        problem=problem,
        page=image,
        problem_image=image,
        split_note=page.note,
        timings=timings,
    )


# ---------------------------------------------------------------- 人工修正


def edit_text(pc: PhotoCheck) -> str:
    """把识别结果摊成"一行一行、带原行号"的文本，给用户改。

    行号的作用是**对齐**：用户改完只剩 6 行，系统还得知道哪一行对应原来的第 6 行，
    否则原来那行的"没认准"标记就丢了，或者张冠李戴到别的行上。
    """
    return "\n".join(
        f"{lr.index}. {lr.text or PLACEHOLDER}" for lr in pc.lines
    )


def _split_edits(text: str) -> list[tuple[int | None, str]]:
    """把用户改过的文本拆成 [(原行号或 None, 内容)]。

    两种输入都要吃得下：带行号的（界面预填的样子）和不带行号的（用户全删了自己重打）。
    """
    rows = [ln.strip() for ln in (text or "").replace("\r\n", "\n").split("\n")]
    rows = [r for r in rows if r]
    out: list[tuple[int | None, str]] = []
    for r in rows:
        m = _EDIT_NUM.match(r)
        if m:
            out.append((int(m.group(1)), m.group(2).strip()))
        else:
            out.append((None, r))
    if all(i is None for i, _ in out):
        # 用户把行号删干净了。按位置对应——这比"全都当成新行"更接近他的本意，
        # 因为预填的文本本来就是按原行顺序排的。
        return [(k + 1, body) for k, (_, body) in enumerate(out)]
    return out


def _faithful(body: str, orig: str) -> bool:
    """用户这一行到底改没改。

    比的是"清洗后能否再解析成同一个式子"，不是逐字相同：去掉一个空格、
    把 `2 x` 写成 `2x`，系统本来就不区分，不该因此把这行标成"手输"。
    """
    from .latex import parse

    if body.strip() == (orig or "").strip():
        return True
    a, b = readable(body), readable(orig or "")
    if a == b:
        return True
    pa, pb = parse(a), parse(b)
    return pa.ok and pb.ok and pa.sympy_str == pb.sympy_str


def regrade(
    pc: PhotoCheck,
    text: str,
    *,
    confirm: Iterable[int] = (),
    problem: str = "",
    page: str = "",
) -> PhotoCheck:
    """按用户改过（或核对过）的过程重新判一遍，不重新识别。

    三种行，三种待遇。区别只在一件事上：**这个式子是系统看清的，还是人给的**。
      · 改过的行（`手输`）→ 标 `ok`、置信度 1.0，结论里点名，系统不为它背书；
      · 没改、但用户在勾选框里核对过的行（`核对`）→ 同样当确定，
        因为这行原文本来就对，只是两个来源没谈拢；
      · 没改也没核对的行 → 保留原来的把握，原来"没认准"的仍是窟窿。
    这一条不能松成"提交了就全算数"：`y=2` 被读成 `7y=2` 时，原样保留就等于
    把一个字都没改的错读当真值，接下来判出来的"不成立"全是冤案。

    其余：行被删掉就当不存在；新加的行（没有行号）按顺序接在后面。
    """
    edits = _split_edits(text)
    sure_ids = {int(i) for i in (confirm or ())}
    by_idx = {lr.index: lr for lr in pc.lines}
    taken: set[int] = set()
    new_lines: list[LineRead] = []
    confirmed: list[int] = []
    given: dict[int, str] = {}      # 原行号 → 用户这次给的内容（非空才记）

    for idx, body in edits:
        orig = by_idx.get(idx) if idx is not None else None
        if orig is not None:
            taken.add(idx)
        if body and body != PLACEHOLDER and idx is not None:
            given[idx] = body
        if not body or body == PLACEHOLDER:
            # 留着问号 = 承认这行没读出来，不当成用户给了内容
            new_lines.append(
                LineRead(
                    index=len(new_lines) + 1,
                    crop=orig.crop if orig else "",
                    verdict="failed",
                    reasons=["这一行留了问号，没有内容可判"],
                )
            )
            continue
        if orig is not None and _faithful(body, orig.text):
            # 一个字没改。用户勾了核对，才算他确认过；否则保持系统原来的把握。
            if idx in sure_ids:
                confirmed.append(idx)
                new_lines.append(
                    LineRead(
                        index=len(new_lines) + 1,
                        crop=orig.crop,
                        latex=body,
                        text=body,
                        verdict="ok",
                        confidence=1.0,
                        sources=[CONFIRM_SOURCE],
                        reasons=["这一行你核对过，按这个读法判"],
                    )
                )
                continue
            new_lines.append(
                LineRead(
                    index=len(new_lines) + 1,
                    crop=orig.crop,
                    latex=orig.latex,
                    text=orig.text,
                    verdict=orig.verdict,
                    confidence=orig.confidence,
                    sources=orig.sources,
                    reasons=orig.reasons,
                    alternatives=orig.alternatives,
                )
            )
            continue
        new_lines.append(
            LineRead(
                index=len(new_lines) + 1,
                crop=orig.crop if orig else "",
                latex=body,
                text=body,
                verdict="ok",
                confidence=1.0,
                sources=[MANUAL_SOURCE],
                reasons=["这一行按你输入的式子判"],
            )
        )

    # 用户删掉的行：不是他的内容，不进判定。但如果是空白块（本来就不是一行），
    # 删了也无所谓，不必提醒。
    # 题面跟着改。题面本来是系统自己从某一行认出来的，用户改的就是那一行时，
    # 判定必须按改过的题算——沿用 `pc.work.problem` 会拿旧题判新过程，
    # 改对了反而判出「少了两个复根」这种结论。
    prob = (problem or "").strip()
    base_note = ""
    if not prob:
        if pc.problem_line:
            body = given.get(pc.problem_line)
            if body:
                prob = body
            else:
                prob = pc.work.problem
                base_note = (
                    f"第 {pc.problem_line} 行原来是题面，这次你把它删了（或留了问号），"
                    "仍按原题面判。如果那道题也认错了，把题面填到上面的输入框里再判一次。"
                )
        else:
            # 题面不是从某一行来的（用户自己填的），照旧用。
            prob = pc.work.problem

    dropped = [i for i in by_idx if i not in taken]
    out = _judge(
        new_lines,
        problem=prob,
        page=page or pc.page,
        split_note=pc.split_note,
    )
    if base_note:
        out.notes.append(base_note)
        out.work.notes.append(base_note)
    if dropped:
        out.notes.append(
            f"第 {'、'.join(map(str, sorted(dropped)))} 行没出现在你给的过程里，"
            "已按删除处理——如果那是漏写的，补回去再判一次。"
        )
        out.work.notes.append(out.notes[-1])
    return out
