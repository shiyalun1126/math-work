"""带超时的符号运算：算不完就杀掉，不留残骸。

原来的写法是 daemon 线程 + ``join(timeout)``。它只解决了「等多久」，没解决
「算不完怎么办」：``join`` 到点只是**不相等了**，那段 sympy 还在跑，而且握着
GIL。一页里连着几次超时，进程里就躺着几条还在算的线程——界面越用越慢，
连「超时」这个结论本身都不准了：主线程要拿 GIL 才能把结果返回给用户，
拿不到就还是卡着。

线程杀不掉，这是 CPython 的硬限制（没有安全的异步中断，``PyThreadState_SetAsyncExc``
只在两次字节码之间生效，而 sympy 里很多循环都在 C 层）。所以只能换进程。

一个常驻 worker 子进程，一次跑一件事：

* 正常：父进程送 ``(op, args)``，子进程算完送回 ``(status, value)``；
* 超时：``kill`` 掉子进程，下次用时重启。代价是重新 import sympy 的那一秒，
  换来的是「超时之后进程里干干净净」。
* 子进程跑挂了（OOM、段错误）：这一次算作失败，同样重启，**不重试**——
  同一道题再跑一遍大概率还是崩，重试只会把卡死时间翻倍。

子进程起不来的环境（沙箱禁 fork/exec、嵌入式解释器）退回线程执行：会退化成
「只能停止等待」，但至少功能还在。``MATHAGENT_GUARD=thread`` 可以强制走这条路。
"""

from __future__ import annotations

import atexit
import os
import pickle
import struct
import subprocess
import sys
import threading

import sympy

DEFAULT_TIMEOUT = 2.5
_STARTUP = 30.0  # 子进程启动 + import sympy 的上限

OK = "ok"
ERR = "err"        # 运算抛了异常
TIMEOUT = "timeout"  # 到点没算完，已杀掉
LOST = "lost"      # 子进程中途死了（崩溃 / 被杀）
DEAD = "dead"      # 起不来


def _dispatch(op: str, args: tuple):
    """真正干活的地方。子进程和线程兜底都走这一个入口，行为不会分叉。"""
    if op == "ping":
        return True
    if op == "spin":  # 自检用：模拟一段算不完的计算
        import time

        time.sleep(float(args[0]))
        return True
    if op == "simplify":
        return sympy.simplify(args[0])
    if op == "simplify_zero":
        return sympy.simplify(args[0]) == 0
    if op == "solve":
        return sympy.solve(args[0], args[1])
    if op == "scaled":
        from .latex import _scaled

        return bool(_scaled(args[0], args[1]))
    if op == "equivalent":
        from .verify import equivalent

        return bool(equivalent(args[0], args[1]))
    if op == "equivalent_strict":
        # 只认"两边相减化简为零"，不认"差一个常数倍"。认题面用这个：
        # `2x=6` 与题面 `x=3` 差两倍，但它是学生的**推导**，不是抄题。
        from .verify import equivalent

        return bool(equivalent(args[0], args[1], scaled=False))
    if op == "solve_core":
        from .solve import _solve_core

        return _solve_core(args[0])
    raise KeyError(f"未知的运算：{op}")


def _readn(stream, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("worker 提前退出")
        buf += chunk
    return buf


def _warn_once(msg: str) -> None:
    global _warned
    if _warned:
        return
    _warned = True
    print(f"[guard] {msg}；超时改用线程实现（只能停止等待，无法中断计算）",
          file=sys.stderr)


_warned = False


class _Worker:
    """常驻子进程。一次只跑一件事，所以进出都上锁。"""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._disabled = False
        self.spawned: list[subprocess.Popen] = []  # 自检用：留个账

    # ---------------------------------------------------------------- 生命周期

    def _spawn(self) -> subprocess.Popen:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (root, env.get("PYTHONPATH", "")) if p
        )
        env.setdefault("PYTHONIOENCODING", "utf-8")
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            [sys.executable, "-m", "mathagent._worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=root,
            env=env,
            **kwargs,
        )
        self.spawned.append(proc)
        return proc

    def _alive(self) -> subprocess.Popen | None:
        p = self._proc
        return p if p is not None and p.poll() is None else None

    def _restart(self) -> subprocess.Popen | None:
        self.kill()
        try:
            self._proc = self._spawn()
        except Exception as exc:  # noqa: BLE001
            self._disabled = True
            _warn_once(f"起不了子进程（{type(exc).__name__}: {exc}）")
            return None
        return self._proc

    def kill(self) -> None:
        p, self._proc = self._proc, None
        if p is None:
            return
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass
        try:
            p.wait(timeout=3)
        except Exception:  # noqa: BLE001
            pass
        for stream in (p.stdin, p.stdout):
            try:
                if stream is not None:
                    stream.close()
            except Exception:  # noqa: BLE001
                pass

    # -------------------------------------------------------------------- 调用

    def call(self, op: str, args: tuple, timeout: float) -> tuple[str, object]:
        payload = pickle.dumps((op, tuple(args)), protocol=4)
        frame = b"\x01" + struct.pack("!I", len(payload)) + payload
        with self._lock:
            for attempt in (0, 1):
                proc = self._alive() or self._restart()
                if proc is None:
                    return DEAD, None
                try:
                    assert proc.stdin is not None
                    proc.stdin.write(frame)
                    proc.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    self.kill()
                    continue
                status, value = self._read(proc, timeout)
                if status == LOST and attempt == 0:
                    continue  # 进程死在半路，换一个新的再送一次
                return status, value
            return LOST, None

    def _read(self, proc: subprocess.Popen, timeout: float) -> tuple[str, object]:
        box: dict = {}

        def reader() -> None:
            try:
                head = _readn(proc.stdout, 1)  # type: ignore[arg-type]
                n = struct.unpack("!I", _readn(proc.stdout, 4))[0]  # type: ignore[arg-type]
                box["r"] = (head, _readn(proc.stdout, n))  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                box["e"] = exc

        t = threading.Thread(target=reader, daemon=True, name="guard-reader")
        t.start()
        t.join(timeout)
        if t.is_alive():
            # 读线程还挂在 read 上。kill 会让管道断开，read 随即返回 EOF，
            # 线程自己就退了——它阻塞在系统调用里，不占 GIL，也不烧 CPU。
            self.kill()
            return TIMEOUT, None
        if "e" in box:
            self.kill()
            return LOST, None
        head, body = box["r"]
        try:
            value = pickle.loads(body)
        except Exception:  # noqa: BLE001
            self.kill()
            return LOST, None
        return (OK if head == b"\x01" else ERR), value

    def live(self) -> int:
        """还活着的 worker 个数。正常最多 1 个，超时之后应当是 0。"""
        return sum(1 for p in self.spawned if p.poll() is None)


_worker = _Worker()
_forced = os.environ.get("MATHAGENT_GUARD", "").strip().lower() in ("thread", "inproc", "off")


def _inproc(op: str, args: tuple, timeout: float, default):
    box: dict = {}

    def work() -> None:
        try:
            box["v"] = _dispatch(op, args)
        except Exception:  # noqa: BLE001
            box["v"] = default

    t = threading.Thread(target=work, daemon=True, name=f"guard-{op}")
    t.start()
    t.join(timeout)
    return box.get("v", default)


def try_run(op: str, *args, timeout: float = DEFAULT_TIMEOUT) -> tuple[str, object]:
    """跑一次运算，把「成败」也交回去。

    需要区分「算超时」和「算出来报错」的调用点（求解层就是这样：超时要说
    「这题我算不了」，报错要说清楚是什么错）用这个；只要一个值的用 ``run``。
    """
    if _forced or _worker._disabled:
        return _inproc_status(op, args, timeout)
    status, value = _worker.call(op, args, timeout)
    if status == DEAD:  # 子进程根本起不来，退回线程
        return _inproc_status(op, args, timeout)
    return status, value


def _inproc_status(op: str, args: tuple, timeout: float) -> tuple[str, object]:
    box: dict = {}

    def work() -> None:
        try:
            box["v"] = (OK, _dispatch(op, args))
        except Exception as exc:  # noqa: BLE001
            box["v"] = (ERR, f"{type(exc).__name__}: {exc}")

    t = threading.Thread(target=work, daemon=True, name=f"guard-{op}")
    t.start()
    t.join(timeout)
    got = box.get("v")
    if got is None:
        return TIMEOUT, None  # 只能停止等待，算不完的那段仍在后台
    return got


def run(op: str, *args, timeout: float = DEFAULT_TIMEOUT, default=None):
    """跑一次运算，失败/超时/崩溃都给 ``default``。"""
    status, value = try_run(op, *args, timeout=timeout)
    return value if status == OK else default


def warmup() -> None:
    """提前把子进程拉起来，省掉第一页的那一秒。后台线程，不挡启动。"""
    if _forced:
        return
    threading.Thread(
        target=lambda: _worker.call("ping", (), _STARTUP), daemon=True, name="guard-warmup"
    ).start()


@atexit.register
def _shutdown() -> None:
    _worker.kill()
