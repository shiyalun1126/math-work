"""把评测结果渲染成能直接贴进结课报告的 markdown。"""

from __future__ import annotations

from eval import metrics as M


def _row_for_mismatch(r: dict) -> list[str]:
    work = r["work"].replace("\n", " → ")
    if len(work) > 58:
        work = work[:55] + "…"
    return [r["id"], r["group"], f"`{r['truth_grade']}`", f"`{r['pred_grade']}`",
            str(r["truth_first"]), str(r["pred_first"]), work, r["note"]]


def render(p: dict) -> str:
    w, ph = p["work"], p["photo"]
    s, ps = w.get("summary", {}), ph.get("summary", {})
    out: list[str] = []
    add = out.append

    add("# 批改链路评测报告")
    add("")
    add(f"生成于 {p['generated_at']} · bootstrap {p['n_boot']} 次 · 随机种子 {p['seed']}  ")
    add(f"真值文件 `eval/work_cases.jsonl` (sha {p['cases_sha']['work']}) / "
        f"`eval/photo_cases.jsonl` (sha {p['cases_sha']['photo']})  ")
    add("复现：`F:/PyCharm/python.exe -m eval.run_eval`")
    add("")
    add("> 真值按数学规范人工标注，不是从引擎输出反推的。所有不一致原样列出，"
        "需要逐条裁定「标注写错」还是「引擎有 bug」。裁定前不得修改真值去迎合实现。")
    add("")

    if not s:
        add("_本次未跑文本批改评测。_")
        return "\n".join(out)

    c = s["conclusion"]
    add("## 1. 主指标")
    add("")
    rows = [
        ["结论准确率", M.pct(c["value"], c["ci"]), f"{c['hit']}/{c['n']}",
         "总判定 grade 与人工真值一致"],
        ["冤枉率 ↓", M.pct(s["false_accusation"]["value"], s["false_accusation"]["ci"]),
         f"{s['false_accusation']['hit']}/{s['false_accusation']['n']}",
         "本来全对的过程被判成 wrong / fake / partial"],
        ["漏抓率 ↓", M.pct(s["let_through"]["value"], s["let_through"]["ci"]),
         f"{s['let_through']['hit']}/{s['let_through']['n']}",
         "有错的过程被判成 pass"],
        ["伪推导召回率 ↑", M.pct(s["fake_recall"]["value"], s["fake_recall"]["ci"]),
         f"{s['fake_recall']['hit']}/{s['fake_recall']['n']}",
         "过程错、答案对的样本里抓住多少"],
        ["伪推导精确率 ↑", M.pct(s["fake_prf"]["precision"]),
         f"{s['fake_prf']['tp']}/{s['fake_prf']['tp'] + s['fake_prf']['fp']}",
         "报成 fake 的样本里真有多少是 fake"],
        ["首错定位准确率 ↑", M.pct(s["first_error"]["value"]),
         f"{s['first_error']['hit']}/{s['first_error']['n']}",
         "只在有错的样本上统计，定位到第一个不成立的行号"],
        ["错因归因准确率 ↑", M.pct(s["error_type"]["value"]),
         f"{s['error_type']['hit']}/{s['error_type']['n']}",
         "严格口径：标签字面完全一致，只在标了错因的样本上统计"],
        ["错因族级一致率 ↑", M.pct(s["error_family"]["value"]),
         f"{s['error_family']['hit']}/{s['error_family']['n']}",
         "把「答案错」与「代回不成立」归入同族后再比，衡量归因方向"],
    ]
    add(M.table(["指标", "数值 [95% CI]", "命中/样本", "口径"], rows))
    add("")
    add("比例指标用 bootstrap 百分位法给区间（技术文档 §6.3）。样本量小，"
        "点估计只能看趋势，不能当结论。↓ 越低越好，↑ 越高越好。")
    add("")
    add("> **不要拿这组 100% 当能力证明。** 样本是按已知场景人工构造的，"
        "引擎与评测集出自同一人，存在共同盲区；而且这是修完两个 bug 之后重跑的结果。"
        "它的含义只是「这批构造样本里已无剩余缺陷」。真实手写作业的分布完全没有覆盖，"
        "见 §7。真正有信息量的是下一节的修复前后对照。")
    add("")
    add("## 1.1 首轮 → 修复后")
    add("")
    add("首轮跑出 38/40，暴露了两个真 bug 和一条口径分歧。修完重跑才是上表的数字。")
    add("")
    add(M.table(
        ["指标", "首轮", "修复后", "变化来自"],
        [
            ["结论准确率", "95.0% (38/40)", "100% (40/40)",
             "修 F02 多解切分、C04 赋值误判；C03 口径裁定"],
            ["冤枉率", "6.2% (1/16)", "0% (0/17)",
             "「解得 x=2 或 x=3」曾只读出 x=3，被误判漏解"],
            ["伪推导召回", "100% (8/8)", "100% (8/8)", "两轮都抓住，未受影响"],
            ["错因严格一致率", "68.2%", "76.2%", "分母因 C03 口径裁定变化"],
            ["错因族级一致率", "—", "100%", "口径分歧集中在标签命名，不是归因方向"],
        ],
    ))
    add("")
    add("两个 bug 都是**评测才发现的**，135 项单元测试全过也没抓到："
        "单元测试验的是「写出来的分支对不对」，评测验的是「没想到的输入会怎样」。"
        "这也是为什么功能冻结后必须补这一层。")
    add("")

    add("## 2. 分层：整体准确率会掩盖难例")
    add("")
    g = s["by_group"]
    add(M.table(
        ["样本组", "命中/样本", "结论准确率"],
        [[k, f"{v['hit']}/{v['n']}", M.pct(v["value"])] for k, v in g.items()],
    ))
    add("")

    add("## 3. 混淆矩阵（行=真值，列=预测）")
    add("")
    cm = s["confusion"]
    labels = cm["labels"]
    add(M.table(
        ["真值 \\ 预测", *labels],
        [[a, *[str(x) if x else "·" for x in cm["rows"][a]]] for a in labels],
    ))
    add("")
    add("最该盯的两个格子：`pass` 行落到 `fake` / `wrong` 列（冤枉），"
        "以及 `fake` 行落到 `pass` 列（漏抓了最值钱的那类错误）。"
        "本表只统计文本批改的 40 条；拍照链路的 `unread` 单独在 §6 评。")
    add("")

    add("## 4. 结论不一致清单")
    add("")
    mm = s["mismatches"]
    if not mm:
        add("修复后为 0 条。首轮有 2 条（F02 冤枉、C03 口径），处置见 §1.1 与 §8。")
    else:
        add(M.table(
            ["ID", "组", "真值", "预测", "真首错", "测首错", "过程", "备注"],
            [_row_for_mismatch(r) for r in mm],
        ))
    add("")
    add("裁定原则：属于引擎缺陷的修引擎，不改真值；属于标注口径分歧的，"
        "改真值并在 §8 写清理由。**没有为了让数字好看而修订任何一条判定结果。**")
    add("")

    add("## 5. 定位与归因的偏差")
    add("")
    fm, tm = s["first_mismatches"], s["type_mismatches"]
    add("**首错定位不一致**：" + (
        "、".join(f"{x['id']}（真 {x['truth']} / 测 {x['pred']}）" for x in fm) if fm else "无"
    ))
    add("")
    add("**错因归因不一致**：" + (
        "、".join(f"{x['id']}（真 {x['truth']} / 测 {x['pred']}）" for x in tm) if tm else "无"
    ))
    add("")
    if tm:
        add("这几条全部是同一类分歧：人工标「答案错」，引擎报「代回不成立」。"
            "后者其实更精确——它给出的是可验证的事实（把答案代回原式，残差不为 0），"
            "而「答案错」只是同义反复。**因此没有修改真值去抹平这 5 条**，"
            "严格一致率就留在 76.2%。根因是错因体系缺一份书面定义："
            "`cases.py` 的 `SEVERITY` 里两个标签都列了，却没说何时用哪个，"
            "标注员和实现各选了一个。已在 `docs/批改链路.md` §6 补定义；"
            "是否统一标签属于产品决策，留给二期。")
        add("")

    if ps:
        add("## 6. 拍照链路：不确定性传播")
        add("")
        pr = ps["withdraw_prf"]
        add(M.table(
            ["指标", "数值", "说明"],
            [
                ["撤回精确率", M.pct(pr["precision"]), "系统撤回的断言里，确实该撤回的比例"],
                ["撤回召回率", M.pct(pr["recall"]), "该撤回的断言里，系统撤回了多少"],
                ["撤回 F1", M.pct(pr["f1"]), ""],
                ["无假确定", M.pct(ps["no_false_certainty"]["value"]),
                 "没认准的行上不残留 ✓ / ✗ 断言"],
                ["反例作废", M.pct(ps["counter_cleared"]["value"]),
                 "断言撤回时反例一起清空，不留伪证据"],
                ["总判定守门", M.pct(ps["grade_guard"]["value"]),
                 "总判定与真值一致：该降 unread 的降了，该保留 fake 的没被抹掉"],
            ],
        ))
        add("")
        add(M.table(
            ["ID", "期望撤回", "实际撤回", "总判定", "备注"],
            [[r["id"], ",".join(map(str, r["expect_withdrawn"])) or "—",
              ",".join(map(str, r["withdrawn"])) or "—",
              f"{r['pred_grade']}" + ("" if r["grade_hit"] else f" (期望 {r['expect_grade']})"),
              r["note"]] for r in ph["records"]],
        ))
        add("")
        bad = ps["mismatches"]
        if bad:
            add(f"**{len(bad)} 条未通过**：" + "、".join(r["id"] for r in bad) + "。")
            add("")
            add("U05 是唯一一处真缺口：题面那一行没认准时，系统只撤回了直接依赖它的第 2 步，"
                "第 3 步仍判「不成立」，总判定也留着 `fake`。但题面认不准意味着"
                "**标准解的来源本身就失效了**，整份结论都该作废，而不是只撤回一步。"
                "当前 `_base_sound` 是逐行往回找最近基准的局部规则，管不到「题面级」不确定性。"
                "这条真值不改、数字不修饰，留作待修项。")
            add("")

    add("## 7. 口径与边界")
    add("")
    add("- 评测对象是**文本版解题过程**，不经过图像识别；识别链路的指标需要真实手写照片，"
        "本轮 `data/test/` 为空，尚未纳入。")
    add("- 样本由人工构造并标注，覆盖正常/一题多解、伪推导、漏解、明显错、无法判定、版式叙述六组；"
        "**不是随机抽样**，因此置信区间只反映这批样本内的重采样波动，不外推到真实作业分布。")
    add("- 冤枉率只统计真值 `pass` 的样本；把 `gap` / `unknown` 判成别的，不算冤枉但算结论错。")
    add("- 首错定位与错因归因只在相应子样本上统计，分母不是全量 `n`。")
    add("- 拍照撤回用构造的 `LineRead` 模拟「哪一行没认准」，不调 VLM，因此可离线重复。")
    add("- **题面级不确定性未传播**（U05）：题面没认准时标准解来源失效，"
        "当前只撤回直接断言，未把整份结论降为 `unread`。已列入待修，不计入通过率修饰。")
    add("- 应用题、含参讨论、几何证明、一页多题、分式与方程组版式均未覆盖；"
        "这些在 `docs/批改链路.md` §7 里已声明为范围外。")
    add("")
    add("## 8. 标注修订记录")
    add("")
    add(M.table(
        ["ID", "字段", "原值", "新值", "理由"],
        [
            ["C03", "grade / error_type / group",
             "partial / 未写答案 / 漏解", "pass / 空 / 版式与叙述",
             "最后一行 `(x-2)(x-3)=0` 已给出完整解集，系统明确披露按它读解，信息没有丢失，不构成漏解。"
             "同时改了引擎文案：不再说「答案也对」，改为说明答案是推断出来的"],
            ["D02/D03/D04/D06/D08", "error_type", "答案错", "**保留原值**",
             "引擎报「代回不成立」更精确，但这是标签命名分歧而非判定错误；"
             "改真值会抹掉这个发现，因此保留，改用族级一致率补充观察"],
            ["F02 / C04", "真值", "pass / partial", "**未改**",
             "这两条暴露的是引擎 bug（多解切分、赋值误判），修的是 `workcheck.py`，不是真值"],
        ],
    ))
    add("")
    add("修订纪律：只改口径，不改判定结果；每条都留理由。"
        "真值文件带 sha256 前缀记录在报告抬头，可与历史运行对照。")
    add("")
    return "\n".join(out) + "\n"
