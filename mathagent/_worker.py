"""超时守卫的子进程端：读一件、算一件、写回结果，然后等下一件。

协议是二进制长度前缀：

    父 → 子   b"\\x01" + uint32 长度 + pickle((op, args))
    子 → 父   b"\\x01" + uint32 长度 + pickle(结果)              成功
              b"\\x00" + uint32 长度 + pickle((类型名, 消息))      异常

stdout 是协议通道。任何一句误打印都会把字节流搅乱，所以启动时先把 fd 1
复制出来独占，再把 fd 1 指到 stderr——库里的 print、警告、进度条全部改道，
协议不受影响。stdin 也一样处理：先复制，再把 fd 0 指到空设备。

这个进程不参与业务逻辑，算完就等下一件；父进程超时会直接 kill 它，
所以这里不需要处理任何「取消」信号。
"""

from __future__ import annotations

import os
import pickle
import struct
import sys


def main() -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.setmode(0, os.O_BINARY)
        msvcrt.setmode(1, os.O_BINARY)

    out = os.fdopen(os.dup(1), "wb", 0)
    os.dup2(2, 1)  # 之后谁 print 都进 stderr
    sys.stdout = sys.stderr

    src = os.fdopen(os.dup(0), "rb", 0)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)

    from .guard import _dispatch, _readn

    while True:
        if not src.read(1):
            return  # 父进程关了管道，正常收工
        n = struct.unpack("!I", _readn(src, 4))[0]
        op, args = pickle.loads(_readn(src, n))
        try:
            payload = pickle.dumps(_dispatch(op, args), protocol=4)
            frame = b"\x01" + struct.pack("!I", len(payload)) + payload
        except BaseException as exc:  # noqa: BLE001
            # 算不出来和「结果送不回去」都从这里出去，父进程拿到的都是 ERR
            payload = pickle.dumps((type(exc).__name__, str(exc)), protocol=4)
            frame = b"\x00" + struct.pack("!I", len(payload)) + payload
        out.write(frame)
        out.flush()


if __name__ == "__main__":
    main()
