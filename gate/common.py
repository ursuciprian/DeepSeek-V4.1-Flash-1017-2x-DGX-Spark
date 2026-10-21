"""Shared OpenAI-API client for the DS41 gate, bench and exactness scripts (stdlib only).

Works against TensorFold's DeepSeek-V4.1 server and vLLM alike: thinking goes in chat_template_kwargs as both
``thinking`` and ``enable_thinking`` (TF reads either; vLLM's deepseek_v41 tokenizer reads ``thinking``), effort as
``chat_template_kwargs.reasoning_effort`` (int 1-100). TF samples by keyed Gumbel-max on (seed, position, token) and an
omitted seed is seed_for(prompt): repeats at T>0 are only independent with distinct ``seed`` values.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request

BASE = os.environ.get("BASE_URL", "http://localhost:8888")
MODEL = os.environ.get("MODEL", "deepseek-v4.1-flash-1017")
# Always sent explicitly: TF's server default is GLM's top_k=20 / top_p=0.95. On TF, top_p=1.0 + top_k=0 at T>0
# takes the slow full-vocab host sampler; 0 < top_p < 1 takes the fast nucleus path (min_p is ignored by TF).
SERVE = {"temperature": 1.0, "top_p": 1.0, "top_k": 0}       # 1017 card default
BASECARD = {"temperature": 1.0, "top_p": 0.95, "top_k": 0}   # base card default
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0}


def think(on: bool, effort: int | None = None) -> dict:
    kw = {"thinking": on, "enable_thinking": on}
    if on and effort is not None:
        kw["reasoning_effort"] = effort
    return {"chat_template_kwargs": kw}


def post(base: str, path: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(base.rstrip("/") + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(base: str, model: str, messages: list, *, max_tokens: int = 512, sampler: dict = SERVE,
         thinking: bool = False, effort: int | None = None, tools: list | None = None, timeout: float = 3600,
         **extra) -> dict:
    """One streamed chat completion. Returns content, reasoning, tool_calls, usage, finish and stream timing
    (t0 submit, t_first / t_last token-carrying chunk, monotonic seconds)."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, **sampler, **think(thinking, effort), **extra}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    t_first = t_last = None
    content, reasoning, calls, usage, finish, chunks = [], [], {}, {}, None, 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ch = json.loads(payload)
            if ch.get("usage"):
                usage = ch["usage"]
            for c in ch.get("choices") or []:
                d = c.get("delta") or {}
                rc = d.get("reasoning_content")
                rc = d.get("reasoning") if rc is None else rc      # TF may send both: count once
                got = False
                if d.get("content"):
                    content.append(d["content"]); got = True
                if rc:
                    reasoning.append(rc); got = True
                for tc in d.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": tc.get("id") or f"call_{len(calls)}",
                                                                  "type": "function",
                                                                  "function": {"name": "", "arguments": ""}})
                    fn = tc.get("function") or {}
                    slot["function"]["name"] += fn.get("name") or ""
                    slot["function"]["arguments"] += fn.get("arguments") or ""
                    got = True
                if got:
                    now = time.monotonic()
                    t_first = t_first or now
                    t_last = now
                    chunks += 1
                finish = c.get("finish_reason") or finish
    return {"content": "".join(content), "reasoning": "".join(reasoning),
            "tool_calls": [calls[i] for i in sorted(calls)], "usage": usage, "finish": finish,
            "t0": t0, "t_first": t_first, "t_last": t_last, "t_end": time.monotonic(), "chunks": chunks}


def decode_tps(r: dict) -> float | None:
    """Completion tokens after the first, over first-to-last token time (TTFT excluded)."""
    n = (r["usage"] or {}).get("completion_tokens") or r["chunks"]
    if not r["t_first"] or r["t_last"] <= r["t_first"] or n < 2:
        return None
    return (n - 1) / (r["t_last"] - r["t_first"])


# -- output health --------------------------------------------------------------------------------------------------
CJK = re.compile("[぀-ヿ㐀-䶿一-鿿가-힯＀-￯]")
BAD = re.compile("[�\x00-\x08\x0b\x0c\x0e-\x1f]")


def corrupted(text: str) -> int:
    """Corrupted characters in an English-only reply: U+FFFD, C0 controls, CJK / fullwidth splices."""
    return len(BAD.findall(text)) + len(CJK.findall(text))


def loop_reason(text: str) -> str:
    """Non-empty when ``text`` looks degenerate: a short alnum fragment repeated 8+ times in a row, a line repeated
    4+ times in a row, or most 12-word shingles of the last 3000 words repeated (a long-period loop)."""
    m = re.search(r"(\w.{0,19}?)\1{7,}", text, re.S)
    if m:
        return f"fragment {m.group(1)!r} x8+"
    prev, run = None, 0
    for line in text.splitlines():
        s = line.strip()
        if len(s) >= 10 and re.search(r"\w", s) and s == prev:
            run += 1
            if run >= 3:
                return f"line repeated 4+ times: {s[:60]!r}"
        else:
            prev, run = s, 0
    words = text.split()[-3000:]
    if len(words) >= 600:
        sh = [" ".join(words[i:i + 12]) for i in range(len(words) - 11)]
        rep = 1 - len(set(sh)) / len(sh)
        if rep > 0.5:
            return f"{rep:.0%} of 12-word shingles repeated"
    return ""


if __name__ == "__main__":
    assert corrupted("hello") == 0 and corrupted("he�l中lo") == 2
    assert loop_reason("Let me go. " * 20)
    assert not loop_reason("def f():\n        return 1\n# ----------------\n")
    assert loop_reason("\n".join(["OK, emit the answer now."] * 6))
    import random
    rng = random.Random(1)
    assert not loop_reason(" ".join(rng.choice("alpha beta gamma delta eps zeta eta theta iota kappa lambda mu nu xi "
                                               "omicron pi rho sigma tau upsilon phi chi psi omega".split())
                                    for _ in range(5000)))
    print("common selftest OK")
