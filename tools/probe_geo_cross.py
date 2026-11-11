"""探测：题干条件与图形标记能否自动交叉校验。

几何题的优势在于「同一件事会说两遍」：
文字写「∠C = 90°」，图上还画个直角小方块；文字写「D 是 AB 中点」，
图上还画等长标记。两处对不上，就说明有一处读错了——这是公式题
没有的纠错信号。

本脚本只做规则层比对，验证这个信号是否存在、是否稳定。
"""

from __future__ import annotations

import re

# 复用上一步探测拿到的真实抽取结果（qwen3-vl-plus）
CASES = {
    "g1": {
        "given": ["∠C = 90°", "AC = 6", "BC = 8", "D 是 AB 的中点"],
        "points": ["A", "B", "C", "D"],
        "segments": [["A", "C"], ["C", "B"], ["A", "B"], ["C", "D"]],
        "marks": [
            {"type": "right_angle", "at": "C", "of": ["AC", "BC"]},
            {"type": "equal_length", "at": "D", "of": ["AD", "DB"]},
        ],
    },
    "g2": {
        "given": ["AB是⊙O的直径", "点C在圆上", "PC是⊙O在点C处的切线", "∠A = 25°"],
        "points": ["A", "B", "C", "O", "P"],
        "segments": [["A", "B"], ["A", "C"], ["B", "C"], ["O", "C"], ["C", "P"]],
        "marks": [{"type": "right_angle", "at": "C", "of": ["OC", "CP"]}],
    },
}


def text_conditions(given: list[str]) -> dict:
    """从题干文字里抽出可核对的几何事实。"""
    out = {"right_angle": [], "midpoint": [], "diameter": [], "tangent": [], "lengths": []}

    for s in given:
        s2 = s.replace(" ", "")
        # 直角：∠C = 90°
        for m in re.finditer(r"∠([A-Z])\s*[=＝]\s*90", s2):
            out["right_angle"].append(m.group(1))
        # 中点：D 是 AB 的中点
        m = re.search(r"([A-Z])\s*(?:是|为)\s*([A-Z])([A-Z])\s*的?中点", s2)
        if m:
            out["midpoint"].append((m.group(1), m.group(2), m.group(3)))
        # 直径
        if "直径" in s2:
            m = re.search(r"([A-Z])([A-Z])(?:是|为)", s2)
            if m:
                out["diameter"].append((m.group(1), m.group(2)))
        # 切线
        if "切线" in s2:
            out["tangent"].append(s)
        # 定长：AC = 6
        for m in re.finditer(r"([A-Z])([A-Z])\s*[=＝]\s*(\d+(?:\.\d+)?)", s2):
            out["lengths"].append((m.group(1) + m.group(2), float(m.group(3))))
    return out


def figure_facts(fig: dict) -> dict:
    """从图形结构里抽出同类事实。"""
    out = {"right_angle": [], "midpoint": [], "tangent": []}
    for mk in fig.get("marks") or []:
        t = (mk.get("type") or "").lower()
        if t == "right_angle":
            out["right_angle"].append(mk.get("at"))
        elif t == "midpoint":
            out["midpoint"].append(mk.get("at"))
        elif t == "equal_length":
            of = mk.get("of") or []
            if len(of) == 2 and len(of[0]) == 2 and len(of[1]) == 2:
                # AD 与 DB 等长且共用 D ⇒ D 是 AB 中点
                a, b = of
                if a[1] == b[0]:
                    out["midpoint"].append(a[1])
    # 切线 ⇒ 圆心与切点连线垂直于切线：图上表现为 OC ⟂ CP
    segs = {tuple(sorted(s)) for s in fig.get("segments") or []}
    for (p, q) in segs:
        if any(mk.get("type") == "right_angle" for mk in (fig.get("marks") or [])):
            out["tangent"].append(f"{p}{q} 处存在直角（可能对应切线）")
            break
    return out


def check(name: str, case: dict) -> None:
    tc = text_conditions(case["given"])
    ff = figure_facts(case)

    print(f"\n{'=' * 64}\n{name}")
    print(f"  文字条件：直角={tc['right_angle']} 中点={tc['midpoint']} "
          f"直径={tc['diameter']} 定长={tc['lengths']}")
    print(f"  图形标记：直角={ff['right_angle']} 中点={ff['midpoint']}")

    verdicts = []

    # 直角：文字与图形是否互相印证
    if tc["right_angle"] and ff["right_angle"]:
        both = set(tc["right_angle"]) & set(ff["right_angle"])
        if both:
            verdicts.append(("印证", f"∠{sorted(both)[0]}=90° 文字与直角标记一致"))
        else:
            verdicts.append(("冲突", f"文字称 ∠{tc['right_angle'][0]}=90°，"
                                     f"但图上直角标在 {ff['right_angle']} —— 必有一处读错"))
    elif tc["right_angle"] and not ff["right_angle"]:
        verdicts.append(("缺证据", "文字给了直角，图上没找到直角标记（可能没画，也可能漏读）"))

    # 中点
    if tc["midpoint"] and ff["midpoint"]:
        same = {m[0] for m in tc["midpoint"]} & set(ff["midpoint"])
        if same:
            verdicts.append(("印证", f"{sorted(same)[0]} 是中点，文字与等长标记一致"))
        else:
            verdicts.append(("冲突", "文字与图形的中点位置不一致"))
    elif tc["midpoint"] and not ff["midpoint"]:
        verdicts.append(("缺证据", "文字称中点，图上没找到等长标记"))

    # 图形补充：文字没提、但图上有的信息
    if ff["right_angle"] and not tc["right_angle"]:
        verdicts.append(
            ("图形补充", f"{ff['right_angle'][0]} 处有直角标记，文字未直接给出"
                        "——切线垂直于半径，这正是解题关键")
        )

    for tag, msg in verdicts:
        mark = {"印证": "  [一致]", "冲突": "  [冲突]", "缺证据": "  [存疑]", "图形补充": "  [补充]"}[tag]
        print(f"{mark} {msg}")

    # 结构完整性
    pts = set(case["points"])
    bad = [s for s in case["segments"] if not set(s) <= pts]
    print(f"  [结构] 线段涉及的点都在点集内：{not bad}"
          + (f"，越界：{bad}" if bad else ""))


def main() -> None:
    for name, case in CASES.items():
        check(name, case)
    print(f"\n{'=' * 64}")
    print("结论：文字与图形的『双份陈述』可以被规则自动比对，")
    print("      冲突即代表识别出错，这比单看一路识别结果多一层保险。")


if __name__ == "__main__":
    main()
