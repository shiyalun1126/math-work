"""超时守卫：到点要真的把计算停掉，不能只停止等待。

验收报告 R01 的原话是「20 次超时后 20 个后台任务存活」。线程版就是这样：
``join(timeout)`` 只让**调用方**不等了，那段 sympy 还在算，还握着 GIL。
换成子进程之后，超时是 kill——进程表里不留东西。
"""

import threading
import time

import sympy

from mathagent import guard


def test_normal_ops_round_trip():
    x = sympy.Symbol("x")
    assert guard.run("simplify", x + x, timeout=30) == 2 * x
    assert guard.run("simplify_zero", x - x, timeout=30) is True
    assert guard.run("solve", x**2 - 5 * x + 6, x, timeout=30) == [2, 3]


def test_result_is_not_a_mangled_object():
    """送过去再送回来，符号还是那个符号（pickle 之后 Symbol 按名字相等）。"""
    x, y = sympy.Symbol("x"), sympy.Symbol("y")
    got = guard.run("simplify", (x + y) ** 2, timeout=30)
    assert sympy.expand(got) == x**2 + 2 * x * y + y**2


def test_error_returns_default_instead_of_raising():
    # 送一个 picklable 但 simplify 不了的对象
    assert guard.run("simplify", object(), timeout=30, default="D") == "D"


def test_unknown_op_is_a_failure_not_a_crash():
    assert guard.run("no-such-op", timeout=30, default="D") == "D"


def test_timeout_kills_the_calculation_instead_of_leaving_it_running():
    before = threading.active_count()
    t0 = time.time()
    for _ in range(3):
        assert guard.run("spin", 5, timeout=0.2, default="T") == "T"
    # 三轮各 5 秒的活，总耗时不该累加到 15 秒——说明是真的停了
    assert time.time() - t0 < 3
    # 干活的那一端已经被杀掉，不是"还在后台跑"
    assert guard._worker.live() == 0
    # 也不该每次超时都甩下一条自旋的线程
    assert threading.active_count() <= before + 2


def test_worker_restarts_after_timeout():
    assert guard.run("spin", 5, timeout=0.2, default="T") == "T"
    assert guard.run("ping", timeout=30) is True
    assert guard._worker.live() == 1


def test_thread_fallback_still_answers(monkeypatch):
    """子进程起不来的环境（沙箱禁 exec）退回线程，功能不能没有。"""
    monkeypatch.setattr(guard, "_forced", True)
    x = sympy.Symbol("x")
    assert guard.run("simplify", x * 2 + x * 3, timeout=30) == 5 * x
    assert guard.try_run("simplify", object(), timeout=30)[0] == guard.ERR
