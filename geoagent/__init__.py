"""几何题链路：读图 → 图文交叉校验 → 坐标法求解 → 重绘比对。

和 mathagent 的分工：mathagent 处理「一个式子」的转写与符号求解，
geoagent 处理「题干 + 图形」的结构化与几何求解。两条链路不共用中间表示。
"""

from __future__ import annotations

__all__ = ["schema", "read", "crosscheck", "solve", "render", "pipeline"]
