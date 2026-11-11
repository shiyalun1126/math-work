"""离线自测：起一个假服务器模拟 SimpleTex 与 OpenAI 兼容端点，验证整条链路。

为什么要有这个：没有密钥时，识别、投票、降级这些分支全都跑不到。
用本地假服务器按官方文档的返回格式应答，可以在不花钱、不联网的情况下
验证"请求格式对不对、返回解析对不对、两个来源不一致时会不会追问"。

跑法：F:/PyCharm/python.exe tests/test_online_mock.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

PORT = 8791

# 假服务器要返回什么，测试用例里会改
STATE = {"simpletex": {"latex": "x^2-5x+6=0", "conf": 0.97}, "vlm": "x^{2}-5x+6=0"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静音
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""

        if self.path.startswith("/api/latex_ocr"):
            if not self.headers.get("token"):
                return self._json({"status": False, "res": "token missing"}, 401)
            if b"file" not in raw:
                return self._json({"status": False, "res": "image_missing"}, 413)
            st = STATE["simpletex"]
            if st is None:
                return self._json({"status": False, "res": "image_proc_error"}, 500)
            return self._json(
                {"status": True, "res": {"latex": st["latex"], "conf": st["conf"]}, "request_id": "t1"}
            )

        if self.path.startswith("/v1/chat/completions"):
            if not (self.headers.get("Authorization") or "").startswith("Bearer "):
                return self._json({"error": {"message": "no key"}}, 401)
            body = json.loads(raw or b"{}")
            model = body.get("model", "")
            msgs = body.get("messages") or []
            text = json.dumps(msgs, ensure_ascii=False)

            # 讲解请求：系统提示里带"不能说公式"，此时返回一段不带公式的话
            is_explain = "讲评" in text or "数学老师" in text
            if is_explain:
                return self._json(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "这道题的关键是先化成标准形式，再用判别式判断根的个数。"
                                    "移项时注意负号，最容易在这一步丢分。"
                                }
                            }
                        ]
                    }
                )
            # 识别请求：按当前用例返回
            return self._json({"choices": [{"message": {"content": STATE["vlm"]}}]})

        self._json({"error": "not found"}, 404)


def serve():
    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def setup_env():
    """注意：不能在这里调 config.reload()。

    reload 会用 .env 里的空值覆盖 os.environ，把下面这些假密钥冲掉，
    于是 build_ensemble 找不到通道、悄悄退回内置夹具——
    整条测试会"通过"，但测的其实是夹具，是个很隐蔽的假绿。
    """
    base = f"http://127.0.0.1:{PORT}"
    os.environ["SIMPLETEX_BASE"] = base
    os.environ["SIMPLETEX_TOKEN"] = "mock-uat"
    os.environ["SIMPLETEX_MODEL"] = "standard"
    os.environ["VLM_BASE_URL"] = f"{base}/v1"
    os.environ["DASHSCOPE_API_KEY"] = "mock-key"
    os.environ["VLM_MODEL"] = "qwen3-vl-flash"
    os.environ["LLM_MODEL"] = "qwen-flash"
    os.environ["ENABLE_EXPLAIN"] = "1"

    from mathagent import config

    assert config.has_simpletex() and config.has_vlm(), "假密钥没生效，测试会被误导"
    print(f"通道：{config.describe()}")
    print(f"SimpleTex 指向：{config.simpletex_url()}")

    import shutil

    shutil.rmtree(config.CACHE_DIR, ignore_errors=True)


def main() -> int:
    serve()
    setup_env()
    from mathagent import config, pipeline  # noqa: E402

    img = str(ROOT / "fixtures" / "demo1.png")
    ok_all = True

    def check(name, cond, detail=""):
        nonlocal ok_all
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ← ' + detail}")
        ok_all = ok_all and cond

    print("用例 1：两源一致 → 应判 ok，并给出两根")
    r = pipeline.run(image=img, k=2, save=False)
    print(f"        候选：{[(c.source, c.latex) for c in r.recognition]}")
    print(f"        判定：{r.verification.verdict} 答案：{r.solved.solutions}")
    srcs = {c.source for c in r.recognition if c.latex}
    check("两个来源都出了结果", srcs == {"simpletex", "qwen3-vl"}, str(srcs))
    check("判 ok", r.verification.verdict == "ok", r.verification.verdict)
    check("解为 2 和 3", r.solved.solutions == ["x = 2", "x = 3"], str(r.solved.solutions))
    check("讲解已生成", bool(r.explain and r.explain.ok), getattr(r.explain, "note", ""))

    print("\n用例 2：Qwen 把 x 认成 × → 应判 review 并列出候选")
    STATE["vlm"] = r"\times^2-5x+6=0"
    shutil_clear = True
    import shutil

    shutil.rmtree(config.CACHE_DIR, ignore_errors=True)
    r2 = pipeline.run(image=img, k=2, save=False)
    print(f"        候选：{[(c.source, c.latex) for c in r2.recognition]}")
    print(f"        判定：{r2.verification.verdict} 理由：{r2.verification.reasons}")
    check("判 review", r2.verification.verdict == "review", r2.verification.verdict)
    check("有被排除的候选", len(r2.verification.rejected) >= 1)
    check("仍选出可解析的那条", r2.verification.chosen == "x^2-5x+6=0", r2.verification.chosen)

    print("\n用例 3：SimpleTex 报 500 → 单源失败不拖垮整体")
    STATE["simpletex"] = None
    STATE["vlm"] = "x^{2}-5x+6=0"
    shutil.rmtree(config.CACHE_DIR, ignore_errors=True)
    r3 = pipeline.run(image=img, k=2, save=False)
    notes = [c.note for c in r3.recognition if c.note]
    print(f"        备注：{notes}")
    print(f"        判定：{r3.verification.verdict} 答案：{r3.solved.solutions}")
    check("失败被记进 note 而非抛异常", any("simpletex" in str(n) or "HTTP" in str(n) for n in notes))
    check("仍能出结果", r3.solved.solutions == ["x = 2", "x = 3"], str(r3.solved.solutions))

    print("\n用例 4：讲解里出现算错的东西 → 应被丢弃")
    from mathagent.explain import check

    ok, why = check("先移项，得到 2+2=5，所以答案是别的东西。", ["x^2-5x+6=0"])
    print(f"        {why}")
    check("错误等式被拦下", not ok)

    ok2, _ = check("这道题先化成标准形式，再用判别式判断根的个数，注意移项要变号。", [])
    check("正常讲解放行", ok2)

    print("\n" + ("全部通过" if ok_all else "有用例未通过"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
