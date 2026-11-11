# 手写数据采集体检

由 `tools/check_capture.py` 生成，每次拍完跑一遍。

```
[单行公式集] 整页照 0/20 张
  ✗ 缺 20 张：page_01、page_02、page_03、page_04、page_05、page_06、page_07、page_08、page_09、page_10、page_11、page_12、page_13、page_14、page_15、page_16、page_17、page_18、page_19、page_20
[单行公式集] 已落盘单行小图 0/150 张
  · 还差 150 张（切行后由 route_crops.py 分流）
    例：self/eq_001.jpg、self/eq_002.jpg、self/eq_003.jpg、self/eq_004.jpg、test/eq_005.jpg

[真值] 自由写 50 条，已补 latex 标注 0 条
  ✗ 还差 50 条没标：这些照片进了测试集也没有 ground truth
    看图把公式写进 labels.csv，latex 列；这是拍完最容易漏的一步

[泄漏] 归一化后不同公式 150 个
  ✓ 未发现 train/test 重复公式

[解题过程集] 整页照 8/20 张（P0 8 / P1 11 / P2 1）
  · 共缺 12 张：W02、W03、W06、W07、W09、W10、W11、W14、W15、W18、W19、W20
  ✓ 清单真值字段与行数自洽
  ✓ 实切行数全部对得上
```
