# 采集清单

**一页写 10 行，整页拍一张，不用裁剪**——切行由 `tools/split_page.py` 自动完成。

写法：内容照抄，字迹按你平时做题的习惯，不要刻意写工整；行距留够，别让上一行的分数线碰到下一行。

页与 id 的对应见 `data/pages.csv`：page_01 自上而下就是 eq_001…eq_010。

## 方程（30 张）

| id | 写法 | split | 提示 |
|---|---|---|---|
| eq_001 | `2x+5=13` | self_train |  |
| eq_002 | `3x-7=2x+4` | self_train |  |
| eq_003 | `5(x-2)=3x+4` | self_train |  |
| eq_004 | `(2x-6)/2=6` | self_train |  |
| eq_005 | `x/3+1=5` | self_test |  |
| eq_006 | `4-2x=x+10` | self_train |  |
| eq_007 | `2(x+3)-5=3x-1` | self_train |  |
| eq_008 | `(x+1)/2-(x-1)/3=1` | self_train |  |
| eq_009 | `x²-5x+6=0` | self_train |  |
| eq_010 | `x²-4=0` | self_test |  |
| eq_011 | `2x²-7x+3=0` | self_train |  |
| eq_012 | `x²+2x-8=0` | self_train |  |
| eq_013 | `3x²=12x` | self_train |  |
| eq_014 | `x²-6x+9=0` | self_train |  |
| eq_015 | `(x-1)²=4` | self_test |  |
| eq_016 | `x²+4x+1=0` | self_train |  |
| eq_017 | `5x²-3x-2=0` | self_train |  |
| eq_018 | `x²+x-1=0` | self_train |  |
| eq_019 | `(x+2)/x=3` | self_train |  |
| eq_020 | `(x+1)/(x-2)=3` | self_test | 分式方程要验增根 |
| eq_021 | `1/(x-1)+2=3/(x-1)` | self_train |  |
| eq_022 | `√(x+3)=x-1` | self_train | 需验根 |
| eq_023 | `√(2x-1)=3` | self_train |  |
| eq_024 | `√(x²-4)=x-2` | self_train | 需验根 |
| eq_025 | `2^x=8` | self_test |  |
| eq_026 | `3^(x+1)=27` | self_train |  |
| eq_027 | `log₂x=3` | self_train |  |
| eq_028 | `lg(x-1)=2` | self_train |  |
| eq_029 | `|x-2|=5` | self_train |  |
| eq_030 | `x+1/x=3` | self_test |  |

## 方程组（10 张）

| id | 写法 | split | 提示 |
|---|---|---|---|
| sys_001 | `{2x+y=7, x-y=2}` | self_train | 两行用大括号 |
| sys_002 | `{3x+2y=12, x-2y=-4}` | self_train |  |
| sys_003 | `{x+y=5, x-y=1}` | self_train |  |
| sys_004 | `{2x+3y=13, 3x-2y=0}` | self_train |  |
| sys_005 | `{x+2y=8, 3x-y=3}` | self_test |  |
| sys_006 | `{4x-3y=5, 2x+5y=9}` | self_train |  |
| sys_007 | `{x/2+y/3=2, x-y=1}` | self_train | 分式系数 |
| sys_008 | `{x²+y=5, x+y=3}` | self_train | 非线性 |
| sys_009 | `{x-y=2, xy=15}` | self_train | 非线性 |
| sys_010 | `{2x+y=1, x²-y=0}` | self_test | 非线性 |

## 化简（25 张）

| id | 写法 | split | 提示 |
|---|---|---|---|
| simp_001 | `(x²-1)/(x+1)` | self_train |  |
| simp_002 | `(a+b)²-(a-b)²` | self_train |  |
| simp_003 | `x²-9` | self_train | 因式分解 |
| simp_004 | `2x·3x²` | self_train |  |
| simp_005 | `√8+√18` | self_test |  |
| simp_006 | `(2√3)²` | self_train |  |
| simp_007 | `1/(1+√2)` | self_train | 分母有理化 |
| simp_008 | `(x+1)/(x²-1)` | self_train |  |
| simp_009 | `a³-a` | self_train |  |
| simp_010 | `(3a²b)·(2ab²)` | self_test |  |
| simp_011 | `√12-√3` | self_train |  |
| simp_012 | `(√5+1)(√5-1)` | self_train |  |
| simp_013 | `(a-b)/(a²-b²)` | self_train |  |
| simp_014 | `x²+5x+6` | self_train | 因式分解 |
| simp_015 | `4x²-12x+9` | self_test | 完全平方 |
| simp_016 | `a²-2ab+b²` | self_train |  |
| simp_017 | `(x³-8)/(x-2)` | self_train |  |
| simp_018 | `2/(x-1)-2/(x+1)` | self_train |  |
| simp_019 | `(a+1/a)²` | self_train |  |
| simp_020 | `√18-√2` | self_test |  |
| simp_021 | `(x²-4x+4)/(x-2)` | self_train |  |
| simp_022 | `3√2·2√3` | self_train |  |
| simp_023 | `log₂8+log₂4` | self_train |  |
| simp_024 | `sin²x+cos²x` | self_test |  |
| simp_025 | `2x/(x²-1)` | self_train | 部分分式 |

## 求导积分（25 张）

| id | 写法 | split | 提示 |
|---|---|---|---|
| calc_001 | `d/dx(x³)` | self_train |  |
| calc_002 | `d/dx(x²+2x)` | self_train |  |
| calc_003 | `d/dx(sin x)` | self_train |  |
| calc_004 | `y=x³ 求 y'` | self_train |  |
| calc_005 | `∫x²dx` | self_test |  |
| calc_006 | `∫₀¹x²dx` | self_train |  |
| calc_007 | `∫(2x+1)dx` | self_train |  |
| calc_008 | `∫sin x dx` | self_train |  |
| calc_009 | `∫₁²(1/x)dx` | self_test |  |
| calc_010 | `∫e^x dx` | self_train |  |
| calc_011 | `d/dx(e^x)` | self_train |  |
| calc_012 | `d/dx(ln x)` | self_train |  |
| calc_013 | `d/dx(x·sin x)` | self_train |  |
| calc_014 | `y=x²e^x` | self_test |  |
| calc_015 | `∫x·e^x dx` | self_train | 分部积分 |
| calc_016 | `lim x→0 sinx/x` | self_train |  |
| calc_017 | `lim x→∞ (1+1/x)^x` | self_train |  |
| calc_018 | `lim x→2 (x²-4)/(x-2)` | self_test |  |
| calc_019 | `∑(i=1 到 n) i` | self_train |  |
| calc_020 | `∑(k=0 到 3) 2^k` | self_train |  |
| calc_021 | `∫₀^π sin x dx` | self_train |  |
| calc_022 | `∫₁^e (1/x) dx` | self_test |  |
| calc_023 | `d/dx(√x)` | self_train |  |
| calc_024 | `∫√x dx` | self_train |  |
| calc_025 | `y=1/x² 求 y'` | self_test |  |

## 复合结构（30 张）

| id | 写法 | split | 提示 |
|---|---|---|---|
| comp_001 | `√((x+1)/(x-1))` | self_train | 根号套分式 |
| comp_002 | `(√x+1)/(√x-1)` | self_train |  |
| comp_003 | `x^(1/2)+x^(-1/2)` | self_train | 上标是分数 |
| comp_004 | `(x²+1)/(x²-1)` | self_train |  |
| comp_005 | `√(x²+2x+1)` | self_test |  |
| comp_006 | `1/(√x+1)` | self_train |  |
| comp_007 | `2^(n+1)/2^n` | self_train |  |
| comp_008 | `log_a(x²)` | self_train | 底数下标+真数上标 |
| comp_009 | `cos2x=2cos²x-1` | self_test |  |
| comp_010 | `√(1-x²)` | self_train |  |
| comp_011 | `(x-1)/(x²+x)` | self_train |  |
| comp_012 | `e^(-x²)` | self_train | 上标带负号和平方 |
| comp_013 | `n!/(k!(n-k)!)` | self_train |  |
| comp_014 | `(a^(1/3))·(a^(2/3))` | self_train |  |
| comp_015 | `1/(1+e^(-x))` | self_test |  |
| comp_016 | `(x+y)/(x-y)` | self_train |  |
| comp_017 | `√((1+√5)/2)` | self_train | 根号套根号 |
| comp_018 | `3/(x²-1)` | self_train |  |
| comp_019 | `(x²+1)²` | self_train |  |
| comp_020 | `√(a+b)·√(a-b)` | self_test |  |
| comp_021 | `(1/x+1/y)/(1/x-1/y)` | self_train | 繁分数 |
| comp_022 | `log(x²+1)` | self_train |  |
| comp_023 | `2/(1+√(1-x))` | self_train |  |
| comp_024 | `√(x+√x)` | self_train |  |
| comp_025 | `(a-b)²/(a²-b²)` | self_test |  |
| comp_026 | `((x+1)/x)²` | self_train | 括号套分式 |
| comp_027 | `x/(1+x²)` | self_train |  |
| comp_028 | `√(2x²+3x-1)` | self_train |  |
| comp_029 | `a³+b³=(a+b)(a²-ab+b²)` | self_train |  |
| comp_030 | `1/(a+1/(b+1/c))` | self_test | 三层繁分数 |

## 难例（30 张）

| id | 写法 | split | 提示 |
|---|---|---|---|
| hard_001 | `2×3=6` | self_test | 乘号写得像 x |
| hard_002 | `x×y=6` | self_test | x 与 × 混在一起 |
| hard_003 | `3×x+1` | self_test | 两个符号都要区分 |
| hard_004 | `5×2x` | self_test | 乘法与字母连写 |
| hard_005 | `21-1=20` | self_test | 1 写成竖线 |
| hard_006 | `l+1=3` | self_test | 字母 l 与数字 1 |
| hard_007 | `1/7` | self_test | 1 和 7 都竖着写 |
| hard_008 | `2z+3=7` | self_test | z 写得像 2 |
| hard_009 | `z²=4` | self_test | z 与 2 混淆 |
| hard_010 | `3z-2=1` | self_test |  |
| hard_011 | `5x-3=2` | self_test | 5 写得像 S |
| hard_012 | `S=5t` | self_test | S 与 5 |
| hard_013 | `O(0` | self_test | O(0,0) |
| hard_014 | `x²+y²=1` | self_test | 上标写得小且贴字母 |
| hard_015 | `e^x=2` | self_test | 上标位置模糊 |
| hard_016 | `a^(n+1)` | self_test | 上标带括号 |
| hard_017 | `x₁+x₂=3` | self_test | 下标 |
| hard_018 | `a_n·a_(n-1)` | self_test | 下标边界不清 |
| hard_019 | `S_n=n²` | self_test | 上下标同时出现 |
| hard_020 | `(a+b)/(a-b)` | self_test | 分数线画得过长 |
| hard_021 | `(x+1)/(x-1)=2` | self_test | 分母线过短 |
| hard_022 | `1/(2x)` | self_test | 2x 是否在分式线内 |
| hard_023 | `a+(b+c)` | self_test | 括号大小层次 |
| hard_024 | `√(x+1)+√(x-1)` | self_test | 两个根号是否连笔 |
| hard_025 | `(x+(y-z))²` | self_test | 嵌套括号 |
| hard_026 | `∫₀¹x dx` | self_test | 上下限写得潦草 |
| hard_027 | `lim x→0 (sinx)/x` | self_test | lim 下方字挤在一起 |
| hard_028 | `sinx/cosx` | self_test | sinx 连写成一团 |
| hard_029 | `(1/x)/(1/y)` | self_test | 分数套分数 |
| hard_030 | `2x+3=7` | self_test | 写错后划掉重写在旁边 |

## 自由写（50 张）

| id | 写法 | split | 提示 |
|---|---|---|---|
| free_001 | `（随意写，不用照抄）` | self_test |  |
| free_002 | `（随意写，不用照抄）` | self_test |  |
| free_003 | `（随意写，不用照抄）` | self_test |  |
| free_004 | `（随意写，不用照抄）` | self_test |  |
| free_005 | `（随意写，不用照抄）` | self_test |  |
| free_006 | `（随意写，不用照抄）` | self_test |  |
| free_007 | `（随意写，不用照抄）` | self_test |  |
| free_008 | `（随意写，不用照抄）` | self_test |  |
| free_009 | `（随意写，不用照抄）` | self_test |  |
| free_010 | `（随意写，不用照抄）` | self_test |  |
| free_011 | `（随意写，不用照抄）` | self_test |  |
| free_012 | `（随意写，不用照抄）` | self_test |  |
| free_013 | `（随意写，不用照抄）` | self_test |  |
| free_014 | `（随意写，不用照抄）` | self_test |  |
| free_015 | `（随意写，不用照抄）` | self_test |  |
| free_016 | `（随意写，不用照抄）` | self_test |  |
| free_017 | `（随意写，不用照抄）` | self_test |  |
| free_018 | `（随意写，不用照抄）` | self_test |  |
| free_019 | `（随意写，不用照抄）` | self_test |  |
| free_020 | `（随意写，不用照抄）` | self_test |  |
| free_021 | `（随意写，不用照抄）` | self_test |  |
| free_022 | `（随意写，不用照抄）` | self_test |  |
| free_023 | `（随意写，不用照抄）` | self_test |  |
| free_024 | `（随意写，不用照抄）` | self_test |  |
| free_025 | `（随意写，不用照抄）` | self_test |  |
| free_026 | `（随意写，不用照抄）` | self_test |  |
| free_027 | `（随意写，不用照抄）` | self_test |  |
| free_028 | `（随意写，不用照抄）` | self_test |  |
| free_029 | `（随意写，不用照抄）` | self_test |  |
| free_030 | `（随意写，不用照抄）` | self_test |  |
| free_031 | `（随意写，不用照抄）` | self_test |  |
| free_032 | `（随意写，不用照抄）` | self_test |  |
| free_033 | `（随意写，不用照抄）` | self_test |  |
| free_034 | `（随意写，不用照抄）` | self_test |  |
| free_035 | `（随意写，不用照抄）` | self_test |  |
| free_036 | `（随意写，不用照抄）` | self_test |  |
| free_037 | `（随意写，不用照抄）` | self_test |  |
| free_038 | `（随意写，不用照抄）` | self_test |  |
| free_039 | `（随意写，不用照抄）` | self_test |  |
| free_040 | `（随意写，不用照抄）` | self_test |  |
| free_041 | `（随意写，不用照抄）` | self_test |  |
| free_042 | `（随意写，不用照抄）` | self_test |  |
| free_043 | `（随意写，不用照抄）` | self_test |  |
| free_044 | `（随意写，不用照抄）` | self_test |  |
| free_045 | `（随意写，不用照抄）` | self_test |  |
| free_046 | `（随意写，不用照抄）` | self_test |  |
| free_047 | `（随意写，不用照抄）` | self_test |  |
| free_048 | `（随意写，不用照抄）` | self_test |  |
| free_049 | `（随意写，不用照抄）` | self_test |  |
| free_050 | `（随意写，不用照抄）` | self_test |  |
