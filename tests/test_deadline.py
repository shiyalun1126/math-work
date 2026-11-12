"""识别链的时间预算：撞上限时怎么退，以及退得不丢结果。

这一组盯的是 R02 里的两个具体失效：
1. 单个请求 90 秒超时 × 3 轮重试 × 3 个候选模型 = 单行 13 分钟，没有上限；
2. 退避 4 + 8 秒是无条件的，即使预算已经不够再发一次请求。

测法全是假响应，不打网络、不花钱。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mathagent import perceive  # noqa: E402
from mathagent.perceive import (  # noqa: E402
    BudgetExhausted,
    Deadline,
    VLMRecognizer,
    budget_for,
)
from mathagent.types import Candidate  # noqa: E402


class FakeResp:
    def __init__(self, code: int = 200, body: dict | None = None, text: str = ""):
        self.status_code = code
        self._body = body or {}
        self.text = text or ("429 太多请求" if code == 429 else "")

    def json(self):
        return self._body


def ok_resp(latex: str) -> FakeResp:
    return FakeResp(200, {"choices": [{"message": {"content": latex}}]})


@pytest.fixture()
def vlm(tmp_path):
    """不读 .env 的识别器：base/key 都显式给，缓存落 tmp。"""
    from mathagent.cache import Cache
    r = VLMRecognizer(model="qwen3-vl-plus", base="http://127.0.0.1:9/v1",
                      api_key="test", slot=1, cache=Cache(tmp_path))
    # 关掉缓存：同一张白图的 digest 固定，上一次测试存进去的结果会让
    # "预算耗尽"的用例直接命中缓存——测出来的就不是预算逻辑了。
    r.cache.get = lambda *a, **k: None
    r.cache.put = lambda *a, **k: None
    return r


def img(tmp_path) -> Path:
    from PIL import Image
    p = tmp_path / "line.jpg"
    Image.new("RGB", (60, 30), "white").save(p, "JPEG")
    return p


# ------------------------------------------------------------- Deadline 本身

def test_no_budget_means_unlimited():
    dl = Deadline(None)
    assert dl.left() == float("inf")
    assert dl.expired() is False
    assert dl.cap(90) == 90


def test_budget_counts_down_and_expires():
    dl = Deadline(0.2)
    assert dl.expired() is False
    time.sleep(0.25)
    assert dl.expired() is True
    assert dl.expired(reserve=1.0) is True


def test_cap_never_exceeds_budget():
    dl = Deadline(3.0)
    assert dl.cap(90) <= 3.0
    # 剩余再多也不能把请求的超时放大到比默认值更久
    assert dl.cap(2.0) <= 2.0


def test_cap_has_floor():
    """剩余 0.01 秒时不能返回 0.01——requests 会当场 ConnectTimeout，
    一个本来能成功的小请求被预算判死，属于自己制造的失败。"""
    dl = Deadline(0.02)
    assert dl.cap(90, floor=0.5) >= 0.5


def test_budget_for_scales_with_lines(monkeypatch):
    monkeypatch.delenv(perceive.BUDGET_ENV, raising=False)
    small = budget_for(2)
    big = budget_for(10)
    assert big > small > 0
    assert budget_for(1) == perceive.BUDGET_BASE + perceive.BUDGET_PER_LINE


def test_budget_env_overrides_and_zero_disables(monkeypatch):
    monkeypatch.setenv(perceive.BUDGET_ENV, "7")
    assert budget_for(10) == 7.0
    monkeypatch.setenv(perceive.BUDGET_ENV, "0")
    assert budget_for(10) is None      # 调试用：不限
    monkeypatch.setenv(perceive.BUDGET_ENV, "乱写")
    monkeypatch.delenv(perceive.BUDGET_ENV)
    assert budget_for(3) == perceive.BUDGET_BASE + perceive.BUDGET_PER_LINE * 3


# --------------------------------------------------- 撞上限以后真的会停

def test_all_429_stops_at_budget_instead_of_minutes(vlm, tmp_path, monkeypatch):
    """全 429 的最坏情况：以前是 3 轮 × 90s 超时 + 12s 退避。

    现在预算 1.5s，应该在一秒内认输，而且**不睡那个 4 秒的退避**。
    """
    calls = []

    def fake_post(url, **kw):
        calls.append(kw.get("timeout"))
        return FakeResp(429)

    slept = []
    monkeypatch.setattr(perceive.requests, "post", fake_post)
    monkeypatch.setattr(perceive.time, "sleep", lambda s: slept.append(s))

    t0 = time.monotonic()
    with pytest.raises(BudgetExhausted):
        vlm._one("abc", 0.2, Deadline(1.5))
    cost = time.monotonic() - t0

    assert cost < 1.0, f"撞上限后还耗了 {cost:.2f}s"
    assert slept == [], f"预算不够还睡退避：{slept}"
    assert calls and all(t <= 1.5 for t in calls), f"请求超时没压进预算：{calls}"


def test_post_timeout_is_capped_by_deadline(vlm, monkeypatch):
    seen = {}

    def fake_post(url, **kw):
        seen["timeout"] = kw.get("timeout")
        return ok_resp("x=1")

    monkeypatch.setattr(perceive.requests, "post", fake_post)
    vlm._one("abc", 0.2, Deadline(2.0))
    assert seen["timeout"] <= 2.0


def test_success_returns_immediately(vlm, monkeypatch):
    monkeypatch.setattr(perceive.requests, "post",
                        lambda *a, **k: ok_resp("x = 1"))
    assert vlm._one("abc", 0.2, Deadline(30)) == "x = 1"


# --------------------------------------------------- 超时不丢已经拿到的结果

def test_second_sample_skipped_but_first_kept(vlm, tmp_path, monkeypatch):
    """k=2：第一次成功，第二次没预算。

    第二次是"同一模型再问一遍"做自一致性，跳过它不该把第一次的结果扔掉——
    丢了等于"因为想多问一句，反而一句都不答"。
    """
    state = {"n": 0}

    def fake_post(url, **kw):
        state["n"] += 1
        time.sleep(0.8)                 # 模拟一次真实耗时，把预算吃掉
        return ok_resp("x = 1")

    monkeypatch.setattr(perceive.requests, "post", fake_post)
    # 1.5s 预算：够第一次（要 >1s 才肯开始），不够第二次
    got = vlm.recognize(img(tmp_path), k=2, deadline=Deadline(1.5))

    assert got[0].latex == "x = 1"      # 第一次的结果还在
    assert any("采样跳过" in (c.note or "") for c in got[1:]), \
        f"第二次应该说明为什么跳过：{got}"
    assert state["n"] == 1


def test_budget_already_gone_returns_note_not_crash(vlm, tmp_path, monkeypatch):
    monkeypatch.setattr(perceive.requests, "post",
                        lambda *a, **k: ok_resp("x=1"))
    got = vlm.recognize(img(tmp_path), k=1, deadline=Deadline(0.01))
    assert len(got) == 1
    assert got[0].latex == ""
    assert "BudgetExhausted" in (got[0].note or "")


# ------------------------------------------------- 逐行读取层要把预算传下去

class _Recorder:
    """只记录被传进来的 deadline，不做识别。"""
    name = "rec"

    def __init__(self):
        self.seen: list[object] = []

    def available(self) -> bool:
        return True

    def recognize(self, image, k=1, deadline=None):
        self.seen.append(deadline)
        return [Candidate(latex="x=1", source="rec")]


def test_read_one_passes_deadline_to_every_source(tmp_path):
    from mathagent.photocheck import _read_one

    a, b = _Recorder(), _Recorder()
    dl = Deadline(5)
    _read_one(1, str(tmp_path / "c.jpg"), [a, b], 1, False, dl)
    assert a.seen == [dl] and b.seen == [dl]


def test_read_one_surfaces_budget_note(tmp_path):
    from mathagent.photocheck import _read_one

    class Exhausted:
        name = "slow"

        def available(self):
            return True

        def recognize(self, image, k=1, deadline=None):
            raise BudgetExhausted("本页时间用完了（预算 3s，剩余 0.0s）")

    lr = _read_one(1, str(tmp_path / "c.jpg"), [Exhausted()], 1, False, Deadline(3))
    assert lr.verdict == "failed"
    assert any("BudgetExhausted" in (x or "") for x in lr.reasons)


# --------------------------------------------- 旧签名识别器不能被静默弄死

def test_call_recognize_tolerates_old_signature(tmp_path):
    """自己写的识别器还停在 recognize(image, k) 时，不该整条通道失效。"""
    from mathagent.perceive import call_recognize

    class Old:
        name = "old"

        def recognize(self, image, k=1):          # 没有 deadline
            return [Candidate(latex="x=1", source="old")]

    got = call_recognize(Old(), "x.jpg", 1, Deadline(5))
    assert got[0].latex == "x=1"


def test_call_recognize_does_not_swallow_other_type_errors():
    from mathagent.perceive import call_recognize

    class Broken:
        name = "broken"

        def recognize(self, image, k=1, deadline=None):
            raise TypeError("又是 NoneType 不可下标")   # 与 deadline 无关

    with pytest.raises(TypeError, match="NoneType"):
        call_recognize(Broken(), "x.jpg", 1, Deadline(5))
