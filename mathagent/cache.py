"""磁盘缓存。

为什么必须有：改一次 prompt / 评测逻辑就把 1500 次 API 调用重跑一遍，
两小时能把免费额度烧完，而且两次结果因模型随机性不可比。
缓存之后变的是分析代码，不变的是已花的钱。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .config import CACHE_DIR


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


class Cache:
    def __init__(self, enabled: bool = True, subdir: str = "api"):
        self.enabled = enabled
        self.dir = CACHE_DIR / subdir

    def get(self, *parts: str) -> Any | None:
        if not self.enabled:
            return None
        p = self.dir / f"{_key(*parts)}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def put(self, value: Any, *parts: str) -> None:
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / f"{_key(*parts)}.json"
        try:
            p.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def remember(self, fn: Callable[[], Any], *parts: str) -> Any:
        hit = self.get(*parts)
        if hit is not None:
            return hit
        val = fn()
        self.put(val, *parts)
        return val
