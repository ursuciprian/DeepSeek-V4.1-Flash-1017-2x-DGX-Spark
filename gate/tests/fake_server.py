"""A stdlib fake of the OpenAI endpoints the DS41 scripts use (TF / vLLM shaped), deterministic, for the smoke test.

  python3 tests/fake_server.py [port] [--perturb]

Chat: SSE or JSON, reasoning_content when thinking is on, tool_calls for tool_choice=required / tool-asking prompts
(fidelity: finds the planted path), answers for the gate's fixed questions, otherwise seeded filler words.
Completions: return_token_ids replies and vLLM-style prompt_logprobs. /metrics: vLLM spec-decode counters.
--perturb makes a 'candidate' build: seed-22 replies diverge, every 40th prompt position flips top-1.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VOCAB = [f"{a}{b}" for a in ("lan", "mor", "tes", "vik", "dor", "sal", "pim", "qua", "ren", "zul", "ob", "fe", "gri",
                             "hu", "jas", "kel") for b in ("a", "o", "en", "ir", "us", "et", "al", "um", "ix", "or")]


def h(*xs) -> int:
    return int(hashlib.sha256(repr(xs).encode()).hexdigest()[:12], 16)


def words(n, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(VOCAB) for _ in range(n))


class Fake(BaseHTTPRequestHandler):
    perturb = False
    stats = {"drafts": 0, "acc": [0, 0, 0]}
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/metrics":
            s = Fake.stats
            lines = [f"vllm:spec_decode_num_drafts_total {s['drafts']}"] + [
                f'vllm:spec_decode_num_accepted_tokens_per_pos{{position="{i}"}} {v}' for i, v in enumerate(s["acc"])]
            b = ("\n".join(lines) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            self._json({"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/chat/completions":
            return self.chat(body)
        if self.path == "/v1/completions":
            return self.completions(body)
        self._json({"error": "not found"}, 404)

    # -- chat ----------------------------------------------------------------------------------------------------
    def reply(self, b):
        msgs, tools = b["messages"], b.get("tools") or []
        kw = b.get("chat_template_kwargs") or {}
        thinking = bool(kw.get("thinking") or kw.get("enable_thinking"))
        last = msgs[-1]
        text = last.get("content") or ""
        mx = b.get("max_tokens", 512)
        seed = b.get("seed", 0)
        reasoning = ("Let me think this through carefully. " + words(20, seed)) if thinking else ""
        call = None
        if last["role"] == "tool":
            content = f"The result is {last['content']}."
        elif tools and (b.get("tool_choice") == "required" or "tool" in text.lower() or "bash" in text.lower()):
            name = tools[0]["function"]["name"]
            if name == "bash":
                tid = re.search(r"invoice_id (\S+)", text).group(1)
                path = next(m["content"].split(":")[0] for m in msgs if m["role"] == "tool" and tid in m["content"])
                args = {"command": f"tail -n 50 {path}"}
            else:
                args = {"expr": "2+2"}
            call, content = {"name": name, "arguments": json.dumps(args)}, ""
        elif "17 * 23" in text:
            content = "391"
        elif m := re.search(r"storage code of item (\d+)", text):
            content = re.search(rf"Item {m.group(1)}: storage code (\w+)", msgs[0]["content"]).group(1)
        else:
            n = mx if "chapter" in text.lower() else min(mx, 600)
            content = words(n, h(text, seed, len(msgs)))
        n = len((reasoning + " " + content).split()) + (8 if call else 0)
        usage = {"prompt_tokens": len(json.dumps(msgs)) // 4, "completion_tokens": n, "total_tokens": n}
        with Fake.lock:
            Fake.stats["drafts"] += n // 3
            for i, frac in enumerate((0.8, 0.6, 0.4)):
                Fake.stats["acc"][i] += int(n // 3 * frac)
        return reasoning, content, call, usage

    def chat(self, b):
        reasoning, content, call, usage = self.reply(b)
        finish = "tool_calls" if call else "stop"
        tf = {"stats": {"drafting": {"windows": 10, "rows": 30, "dspark_rounds": 10, "dspark_drafts": 30,
                                     "dspark_kept": 20, "lookup_rounds": 0, "lookup_drafts": 0, "lookup_kept": 0,
                                     "dspark_tokens_per_round": 3.0, "tokens_per_round": 3.0}}}
        if not b.get("stream"):
            msg = {"role": "assistant", "content": content, "reasoning_content": reasoning or None}
            if call:
                msg["tool_calls"] = [{"id": "call_0", "type": "function", "function": call}]
            return self._json({"choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage,
                               "tensorfold": tf})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(delta, fin=None, **extra):
            ch = {"choices": [{"index": 0, "delta": delta, "finish_reason": fin}], **extra}
            self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode())

        send({"role": "assistant"})
        for field, txt in (("reasoning_content", reasoning), ("content", content)):
            ws = txt.split(" ")
            for i in range(0, len(ws), 8):
                send({field: (" " if i else "") + " ".join(ws[i:i + 8])})
        if call:
            send({"tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                                  "function": {"name": call["name"], "arguments": ""}}]})
            send({"tool_calls": [{"index": 0, "function": {"arguments": call["arguments"]}}]})
        send({}, finish)
        self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    # -- completions ---------------------------------------------------------------------------------------------
    def completions(self, b):
        prompt, k = b["prompt"], b.get("prompt_logprobs")
        ids = [h(w) % 50000 for w in prompt.split()]
        seed = b.get("seed", 0) if b.get("temperature", 1) > 0 else -1
        out = [h(prompt, seed, i) % 50000 for i in range(b.get("max_tokens", 16))]
        if self.perturb and seed == 22 and len(out) > 5:
            out[5:] = [x + 1 for x in out[5:]]
        ch = {"index": 0, "text": " ".join(map(str, out)), "token_ids": out, "prompt_token_ids": ids,
              "finish_reason": "length"}
        if k:
            pl = [None]
            for i in range(1, len(ids)):
                rng = random.Random(h(tuple(ids[:i])))
                lps = sorted((-abs(rng.gauss(0, 3)) - j * 0.5 for j in range(k)), reverse=True)
                toks = [ids[i]] + [rng.randrange(50000) for _ in range(k - 1)]
                if self.perturb:
                    lps = [x + 0.01 * rng.random() for x in lps]
                    if i % 40 == 0:
                        toks[0], toks[1] = toks[1], toks[0]
                pl.append({str(t): {"logprob": lp, "rank": j + 1} for j, (t, lp) in enumerate(zip(toks, lps))})
            ch["prompt_logprobs"] = pl
        self._json({"choices": [ch], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out)}})


def serve(port=0, perturb=False):
    cls = type("FakeP" if perturb else "FakeR", (Fake,), {"perturb": perturb})
    ThreadingHTTPServer.request_queue_size = 64          # default backlog 5 resets C8 connections on macOS
    srv = ThreadingHTTPServer(("127.0.0.1", port), cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8899
    srv, url = serve(port, "--perturb" in sys.argv)
    print("fake OpenAI server on", url)
    threading.Event().wait()
