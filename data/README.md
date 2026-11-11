# 数据说明

分两类，用途完全不同。

- **公开数据**（`raw/`）：用来训练和跑基线，够多，不用你操心。
- **自采数据**（`self/`、`test/`）：用来证明"在真实场景能用"，**这部分只能你来**，是报告里唯一别人抄不走的证据。

---

## 一、公开数据源（按优先级）

| 数据集 | 内容 | 获取方式 | 许可 | 用途 |
|---|---|---|---|---|
| **HME100K** | 99,109 张真实拍摄手写公式，74,502 训练 / 24,607 测试，245 类符号，jpg + txt 标注；另带 easy/medium/hard 难度划分 | https://ai.100tal.com/openData/formulaRecognition （需注册申请，728.8 MB） | 学术研究用途 | **首选训练集**，中文 K-12 场景最贴合 |
| **MathWriting** | 23 万真人手写 + 40 万合成，InkML 笔迹格式，带归一化 LaTeX | 完整 2.9 GB：https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz ；抽样 1.5 MB：`mathwriting-2024-excerpt.tgz`（同目录） | CC BY-NC-SA 4.0 | 在线笔迹路线的主力；也可栅格化成图片 |
| **MathWriting 图片版** | 已渲染成 JPEG，带 train/val/test split | https://huggingface.co/datasets/deepcopy/MathWriting-human | CC BY-NC-SA 4.0 | 省一步渲染，直接当离线图像用 |
| **UniMER-1M / UniMER-Test** | 印刷+手写混合，按公式类型分 split，配套 UniMERNet 基线 | https://huggingface.co/datasets/wanderkid/UniMER_Dataset | 见仓库说明 | 印刷体兜底 + 官方 benchmark 对照 |
| **CROHME** | 在线手写竞赛数据，8,835 训练 / 986 评测 | 官网 http://crohme.liris.cnrs.fr/ 常不可访问，镜像：https://aistudio.baidu.com/datasetdetail/174727 | 学术研究用途 | 与文献可比的老基准，别当主力 |
| **IM2LATEX-100K** | 纯合成印刷公式 | HuggingFace / Kaggle 搜 `im2latex` | 见来源页 | 只用于预训练或增强，别用于评测 |

几何图形（二期再看）：

| 数据集 | 内容 |
|---|---|
| FormalGeo7K | 形式化几何题 + 求解器，https://github.com/FormalGeo/FormalGeo |
| Geometry3K / PGPS9K | 几何题图文对（Inter-GPS 项目） |
| GeoQA / UniGeo | 几何题与程序化标注 |

**许可提醒**：MathWriting 是 CC BY-NC-SA 4.0，非商用可以，报告里必须注明来源和许可；HME100K 走学术申请通道，别二次分发。

---

## 二、需要你提供的（这是重点）

### 1. 自采测试集：150–200 张（最重要）

为什么必须自采：公开测试集是别人采的，分布你控制不了。你要在报告里写"在独立采集的真实样本上达到 X%"，这句话只有自己的数据能支撑。

**覆盖配比建议**（一期题型）：

| 类型 | 张数 | 说明 |
|---|---|---|
| 一元一次 / 二次方程 | 40 | 含移项、分式方程 |
| 方程组 | 25 | 两元、三元各半 |
| 代数化简 / 因式分解 | 30 | 含根号、指数 |
| 求导 / 积分 | 35 | 只到简单幂函数、多项式 |
| 含分式、根号、上下标的复合式 | 30 | 结构难点 |
| 特意写潦草的难例 | 40 | 见下条 |

### 2. 难例集：30–50 张

不要小看这个，这是你校验模块唯一能证明价值的地方。**特意写这些**：

- `x` 和 `×`、`1` 和 `l`、`2` 和 `z`、`5` 和 `S`、`\partial` 和 `d`
- 上下标位置模糊（`x^2` 的 2 到底在上还是平）
- 连笔、快速书写、笔画粘连
- 分数线的长短不一（`\frac{a+b}{c}` 写成横线穿过半个式子）
- 涂改、划掉重写

### 3. 一道题的题型清单

一期到底覆盖哪几类，你定，我按这个配训练数据和测试集。默认按上面的表。

### 4.（可选）平板笔迹

如果有 iPad / 安卓平板 + 手写笔，采 100–200 条 stroke 数据（JSON：每笔是点序列 + 时间戳）。在线笔迹路线精度高一个档次，但只在有设备时值得做。

---

## 三、怎么采、怎么标

### 拍摄规范（v2：整页拍，自动切行）

详细论证见 `docs/采集与切分方案.md`，这里只列要点。

- **排版**：A4 白纸（不要横线本），页边距 2 cm，每行占 2.5 cm 竖向空间，**一页写 10 行**
- **硬约束**：上下两行的墨迹不能接触——分式的横线、根号下沿、下标、连笔尾巴，都不能碰到下一行
- **不写题号**：会被切进单行图，变成公式的一部分
- **拍摄**：手机竖直俯拍，纸占画面 70% 以上，整张纸的四边都在画面内（这是确认没拍到纸外的标记），倾斜 ≤ 5°
- **环境**：自然光或柔光，避免单侧强光（半页发黑会让二值化失效）
- **不要用扫描 App**：会锐化 + 透视校正 + 加白边，与真实场景不一致
- **保留原图**：不压缩、不美颜、不裁剪，直接拷 jpg

倾斜要看的是**整张纸**有没有歪，不是每行公式单独端正。

### 交付格式

```
data/inbox/page_01.jpg     ← 一页一张，覆盖 eq_001…eq_010
data/inbox/page_02.jpg
...
```

页与 id 的对应关系在 `data/pages.csv`（20 页覆盖 200 条）。拍摄时**不用管 train/test 划分**，切行后按 id 自动分流。

切行：

```bash
F:/PyCharm/python.exe tools/split_page.py data/inbox --out data/crops --expect 10 --debug
```

### 文件命名

```
self/s_001.jpg   s_002.jpg ...     自采，可用于训练/调试
test/t_001.jpg   t_002.jpg ...     自采，只测不训，任何人不得拿来调参
```

### 标注格式

统一写在 `labels.csv`，UTF-8，逗号分隔，LaTeX 字段用双引号包裹（防止内部逗号冲突）：

```
file,latex,split,type,note
self/s_001.jpg,"\frac{2x-6}{2}=6",self_train,方程,
test/t_001.jpg,"x^{2}-5x+6=0",self_test,方程,
```

LaTeX 书写约定（很重要，直接影响能否解析）：

- 分数用 `\frac{分子}{分母}`，不要写 `a/b`
- 上下标用 `^{}` `_{}`，多字符一定加花括号
- 根号 `\sqrt{x}`、`\sqrt[3]{x}`
- 积分 `\int_{0}^{1} x^{2} \, dx`，微分 `dx` 前留空格
- 乘号：数字间用 `\times`，字母间省略
- 不要用 `\left` `\right`（简化解析），除非确实需要
- 不要写中文、不要写单位

### 最小启动量

- **50 张** 就能把链路跑通（先给我这个）
- **200 张** 够做正式评估
- **500 张** 够做 LoRA 微调
- 超过 1000 张收益递减，别花那个时间

---

## 四、检查工具

标完跑一下：

```bash
python data/check_data.py
```

会报：缺文件、多余文件、LaTeX 解析失败、类别分布、split 泄漏（同一张图同时出现在 train 和 test）。

---

## 五、交付流程（照这个来）

1. 打开 `data/shotlist.md`，按表里的 **写法** 写，**一页 10 行**（内容照抄，字迹按平时习惯）
2. 整页拍一张，命名 `page_01.jpg`、`page_02.jpg` …（对应关系见 `data/pages.csv`）
3. 全部丢进 `data/inbox/`
4. 告诉我，我跑切行 + 检查 + 补标签

一页 10 行，200 条样本 = 20 张照片。想分给同学的话，按页拆：一人 2–3 页。

---

## 六、目录约定

```
data/
├── shotlist.csv     采集清单（id / 题型 / split / 写法 / LaTeX / 提示）
├── shotlist.md      同一份内容，便于阅读和打印
├── pages.csv        页 ↔ id 段映射（20 页覆盖 200 条）
├── inbox/           整页原图放这里，page_01.jpg …
├── crops/           切行产物 + manifest.csv
├── raw/             公开数据集，按原始结构存放，不修改
├── self/            自采，可训练
├── test/            自采，只测不训
├── labels.csv       标注（由 shotlist 生成，自由写那批待人工补）
└── check_data.py
tools/
├── split_page.py        整页切行主程序
├── selftest_split.py    切分自测
├── make_pages.py        生成 pages.csv
├── init_labels.py       生成 labels.csv
└── make_shotlist_md.py  生成 shotlist.md
docs/
└── 采集与切分方案.md    采集规范与算法说明
```

