#!/usr/bin/env python3
"""DS41 quality gate: one entry point, OpenAI API only (TF or vLLM), pass/fail summary JSON, exit 1 on any FAIL.

  BASE_URL=http://localhost:8888 MODEL=deepseek-v4.1-flash-1017 \\
    python3 gate.py --label base --out results/gate-base.json
  python3 gate.py --label engram-off --engram off --compare-to results/gate-base.json --out results/gate-eoff.json

Probes (thresholds fixed here, before data; do not loosen after a run):
  reasoning       thinking on: non-empty reasoning_content + correct answer; thinking off: empty reasoning, answer
  tool_turn       tools declared: assistant call -> tool result -> answer uses the result
  tool_only_turn  1017: thinking on, NO tools declared, tool-result-only user turn: reply has reasoning, uses the
                  result, and the prior assistant reasoning is kept in the prompt (prompt_tokens grows >= 50 when it
                  is sent vs empty; the pre-10-16 encoding drops it: delta 0)
  tc45            tool_choice=required on a trivial question: a calculator call; 3 tries, >= 1 pass (Qwen gate rule)
  hardmode        tool-eval-bench --hardmode: Quality >= 88 and its TC-45 line PASS
  fidelity        identifier fidelity (fidelity_probe.py) 8K/16K/32K/64K/128K + 256K rung (248K target):
                  20/20 exact up to 64K, >= 19/20 at 128K and 248K
  prefix          4 concurrent 4-turn chats on one ~8K shared prefix: every code retrieved exactly
  long_session    multi-turn chapters until >= 28K decoded tokens (cap 32K): no loop in any reply
  long_decode     one effort-100 request, max_tokens 32768: no loop in reasoning + content
  corruption      two 1024-token essays (thinking off / effort 75) at T=1.0 + every text above: 0 corrupted chars
                  (U+FFFD, C0 controls, CJK/fullwidth splices) and no degenerate essay
Engram-off control: --engram off labels the run; --compare-to prints per-probe score deltas (no pass/fail effect).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import shutil
import subprocess
import sys
import time

import common
import fidelity_probe

TEXTS: list[tuple[str, str]] = []          # (probe, text) for the corruption sweep


def keep(name, *texts):
    TEXTS.extend((name, t) for t in texts if t)


def chat(a, msgs, **kw):
    kw.setdefault("sampler", a.sampler)
    return common.chat(a.base, a.model, msgs, timeout=a.timeout, **kw)


CALC = [{"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression.",
         "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}}]
WEATHER = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def p_reasoning(a):
    q = [{"role": "user", "content": "What is 17 * 23? Reply with the number."}]
    on = chat(a, q, thinking=True, effort=75, max_tokens=4096)
    off = chat(a, q, thinking=False, max_tokens=256)
    keep("reasoning", on["content"], on["reasoning"], off["content"])
    ok_on = bool(on["reasoning"].strip()) and "391" in on["content"]
    ok_off = not off["reasoning"].strip() and "391" in off["content"]
    return {"pass": ok_on and ok_off, "on_reasoning_chars": len(on["reasoning"]), "on_content": on["content"][-200:],
            "off_reasoning_chars": len(off["reasoning"]), "off_content": off["content"][-200:]}


def p_tool_turn(a):
    msgs = [{"role": "user", "content": "What is the weather in Oslo right now? Use the tool."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_w1", "type": "function", "function": {
                "name": "get_weather", "arguments": json.dumps({"city": "Oslo"})}}]},
            {"role": "tool", "tool_call_id": "call_w1", "content": "Oslo: -3 C, heavy snow, wind 40 km/h"}]
    r = chat(a, msgs, thinking=True, tools=WEATHER, max_tokens=4096)
    keep("tool_turn", r["content"], r["reasoning"])
    ok = not r["tool_calls"] and ("-3" in r["content"] or "snow" in r["content"].lower())
    return {"pass": ok, "content": r["content"][-300:], "tool_calls": r["tool_calls"]}


REASONING_R = ("The user wants 1234 * 5678. I should not do this by hand because they asked for the calculator. "
               "Plan: call calculator with expr 1234*5678, then report the product and double-check it by "
               "estimating 1200 * 5700 = 6,840,000, so the exact answer should be a little above that, around seven "
               "million. After the tool returns I will state the exact product, mention the estimate as a sanity "
               "check, and keep the answer to one short sentence because the user prefers brevity. ") * 2


def p_tool_only_turn(a):
    def conv(reasoning):
        return [{"role": "user", "content": "Use the calculator to compute 1234 * 5678, then tell me the result."},
                {"role": "assistant", "content": "", "reasoning_content": reasoning,
                 "tool_calls": [{"id": "call_c1", "type": "function", "function": {
                     "name": "calculator", "arguments": json.dumps({"expr": "1234*5678"})}}]},
                {"role": "tool", "tool_call_id": "call_c1", "content": "7006652"}]
    full = chat(a, conv(REASONING_R), thinking=True, max_tokens=4096)          # no tools declared on purpose
    empty = chat(a, conv(""), thinking=True, max_tokens=8)
    keep("tool_only_turn", full["content"], full["reasoning"])
    delta = (full["usage"].get("prompt_tokens") or 0) - (empty["usage"].get("prompt_tokens") or 0)
    ans = re.sub(r"[,\s]", "", full["content"])
    ok = bool(full["reasoning"].strip()) and "7006652" in ans and delta >= 50
    return {"pass": ok, "reasoning_chars": len(full["reasoning"]), "content": full["content"][-200:],
            "prompt_token_delta_kept_reasoning": delta}


def p_tc45(a):
    tries = []
    for seed in (1, 2, 3):                    # distinct seeds: TF repeats with one seed are identical
        r = chat(a, [{"role": "user", "content": "What is 2 + 2?"}], thinking=True, tools=CALC,
                 tool_choice="required", max_tokens=4096, seed=seed)
        keep("tc45", r["content"], r["reasoning"])
        tries.append(bool(r["tool_calls"]) and r["tool_calls"][0]["function"]["name"] == "calculator")
    return {"pass": any(tries), "tries": tries}


def parse_hardmode(log: str) -> dict:
    q = re.search(r"Quality:\s*(\d+)/100", log)
    tc = re.search(r"TC-45\b.*?(PASS|FAIL)", log)
    return {"quality": int(q.group(1)) if q else None, "tc45": tc.group(1) if tc else None}


def p_hardmode(a):
    exe = shutil.which("tool-eval-bench")
    if not exe:
        return {"pass": False, "error": "tool-eval-bench not on PATH (pip install it, or --skip hardmode)"}
    cmd = [exe, "run", "--hardmode", "--temperature", str(a.sampler["temperature"]), "--backend", a.hardmode_backend,
           "--timeout", "600", "--max-turns", "32", "--base-url", a.base, "--model", a.model]
    log = subprocess.run(cmd, capture_output=True, text=True).stdout
    r = parse_hardmode(log)
    return {"pass": (r["quality"] or 0) >= 88 and r["tc45"] == "PASS", **r, "cmd": " ".join(cmd),
            "tail": log[-1500:]}


def p_fidelity(a):
    out, cpt, ok = [], a.chars_per_token, True
    for depth in [int(x) for x in a.depths.split(",")]:
        r = fidelity_probe.run_depth(a.base, a.model, depth, 5, a.fid_trials, 7, cpt, a.sampler["temperature"],
                                     8192, "on", 1, a.timeout)
        n = sum(r["counts"].values())
        need = n if depth <= 64000 else n - n // 20               # 20/20, or 19/20 at 128K and the 256K rung
        good = r["counts"]["exact"] >= need
        ok &= good
        if r["actual_prompt_tokens"]:                              # recalibrate chars/token for the next rung
            cpt *= depth / r["actual_prompt_tokens"]               # more tokens than asked: fewer chars a token
        out.append({"depth": depth, "actual": r["actual_prompt_tokens"], "counts": r["counts"], "need": need,
                    "pass": good})
        keep("fidelity", *[x["cmd"] for x in r["results"]])
    return {"pass": ok, "depths": out}


def p_prefix(a):
    rng = random.Random(41)
    codes = {f"{i:03d}": "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8)) for i in range(240)}
    filler = ("Inventory note: shelf audits run weekly, counts are reconciled against the ledger, and damaged items "
              "are logged with a photo before disposal. ")
    lines = [f"Item {k}: storage code {v}. {filler}" for k, v in codes.items()]
    prefix = "You answer questions about the warehouse register below.\n\n" + "\n".join(lines)
    asks = [[f"{rng.randrange(240):03d}" for _ in range(4)] for _ in range(4)]

    def convo(items):
        msgs, res = [{"role": "system", "content": prefix}], []
        for it in items:
            msgs.append({"role": "user", "content": f"What is the storage code of item {it}? Reply with the code only."})
            r = chat(a, msgs, thinking=False, max_tokens=64)
            msgs.append({"role": "assistant", "content": r["content"]})
            keep("prefix", r["content"])
            res.append({"item": it, "want": codes[it], "got": r["content"].strip()[:40],
                        "ok": codes[it] in r["content"], "prompt_tokens": r["usage"].get("prompt_tokens")})
        return res
    with cf.ThreadPoolExecutor(4) as ex:
        res = [x for conv in ex.map(convo, asks) for x in conv]
    return {"pass": all(x["ok"] for x in res), "correct": sum(x["ok"] for x in res), "total": len(res),
            "results": res}


def p_long_session(a):
    msgs = [{"role": "system", "content": "You are writing a long technical manual for operators of a small hydro "
                                          "power station. Write in plain prose with headings."},
            {"role": "user", "content": "Write chapter 1 (about 3000 words). Each later chapter covers a new topic."}]
    total, turns, loops, corrupt = 0, [], [], 0
    while total < a.session_tokens and len(turns) < 40:
        r = chat(a, msgs, thinking=False, max_tokens=min(4096, a.session_cap - total))
        n = r["usage"].get("completion_tokens") or len(r["content"].split())
        total += n
        why = common.loop_reason(r["content"])
        corrupt += common.corrupted(r["content"])
        turns.append({"turn": len(turns) + 1, "tokens": n, "finish": r["finish"], "loop": why,
                      "prompt_tokens": r["usage"].get("prompt_tokens"), "tps": common.decode_tps(r)})
        if why:
            loops.append(why)
        if not r["content"].strip():
            loops.append(f"empty reply at turn {len(turns)}")
            break
        msgs += [{"role": "assistant", "content": r["content"]},
                 {"role": "user", "content": f"Continue with chapter {len(turns) + 1}, a new topic, about 3000 words."}]
    return {"pass": not loops and total >= a.session_tokens, "decoded": total, "loops": loops,
            "corrupted": corrupt, "turns": turns}


def p_long_decode(a):
    q = ("Find every 4-digit number n such that n equals the sum of the fourth powers of its digits, and prove there "
         "are no others. Then do the same for 5-digit numbers and fifth powers.")
    r = chat(a, [{"role": "user", "content": q}], thinking=True, effort=100, max_tokens=a.long_decode_tokens)
    why = common.loop_reason(r["reasoning"]) or common.loop_reason(r["content"])
    keep("long_decode", r["content"], r["reasoning"])
    return {"pass": not why, "loop": why, "finish": r["finish"],
            "completion_tokens": r["usage"].get("completion_tokens"),
            "warn": "hit max_tokens without a loop" if r["finish"] == "length" and not why else ""}


def p_corruption(a):
    q = [{"role": "user", "content": "Write a continuous 900-word essay on how coastal lighthouses were built and "
                                     "maintained in the nineteenth century. Ordinary prose paragraphs, no lists."}]
    essays = [chat(a, q, thinking=False, max_tokens=1024, sampler=common.SERVE),
              chat(a, q, thinking=True, effort=75, max_tokens=4096, sampler=common.SERVE)]
    degenerate = [common.loop_reason(e["content"]) or ("short" if len(e["content"].strip()) < 200 else "")
                  for e in essays]
    for e in essays:
        keep("corruption", e["content"], e["reasoning"])
    per = {}
    for name, t in TEXTS:
        per[name] = per.get(name, 0) + common.corrupted(t)
    total = sum(per.values())
    return {"pass": total == 0 and not any(degenerate), "corrupted_total": total, "per_probe": per,
            "degenerate": degenerate}


PROBES = {"reasoning": p_reasoning, "tool_turn": p_tool_turn, "tool_only_turn": p_tool_only_turn, "tc45": p_tc45,
          "hardmode": p_hardmode, "fidelity": p_fidelity, "prefix": p_prefix, "long_session": p_long_session,
          "long_decode": p_long_decode, "corruption": p_corruption}       # corruption last: it sweeps TEXTS


def score(p: dict) -> float | None:
    for k in ("quality", "correct", "decoded", "corrupted_total"):
        if isinstance(p.get(k), (int, float)):
            return p[k]
    if "depths" in p:
        return sum(d["counts"]["exact"] for d in p["depths"])
    return None if p.get("pass") is None else float(bool(p["pass"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=common.BASE)
    ap.add_argument("--model", default=common.MODEL)
    ap.add_argument("--label", default="run")
    ap.add_argument("--engram", choices=["on", "off"], default="on", help="label only: how the server was booted")
    ap.add_argument("--compare-to", help="an earlier gate JSON (e.g. the Engram-on run) for score deltas")
    ap.add_argument("--only", default="", help="comma list of probes to run")
    ap.add_argument("--skip", default="", help="comma list of probes to skip (recorded as skipped)")
    ap.add_argument("--greedy", action="store_true", help="T=0 instead of the 1017 serving sampler T=1.0 top_p=1.0")
    ap.add_argument("--depths", default="8000,16000,32000,64000,128000,248000")
    ap.add_argument("--fid-trials", type=int, default=4, help="trials per planted target (5 targets)")
    ap.add_argument("--chars-per-token", type=float, default=4.6)
    ap.add_argument("--session-tokens", type=int, default=28000)
    ap.add_argument("--session-cap", type=int, default=32000)
    ap.add_argument("--long-decode-tokens", type=int, default=32768)
    ap.add_argument("--hardmode-backend", default="vllm")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", default="gate.json")
    a = ap.parse_args(argv)
    a.sampler = common.GREEDY if a.greedy else common.SERVE
    only = [x for x in a.only.split(",") if x]
    skip = {x for x in a.skip.split(",") if x}
    for n in only + list(skip):
        if n not in PROBES:
            ap.error(f"unknown probe {n!r}; known: {', '.join(PROBES)}")
    TEXTS.clear()
    res = {"label": a.label, "engram": a.engram, "base": a.base, "model": a.model, "sampler": a.sampler,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "probes": {}}
    for name, fn in PROBES.items():
        if (only and name not in only) or name in skip:
            res["probes"][name] = {"pass": None, "skipped": True}
            continue
        t = time.monotonic()
        try:
            p = fn(a)
        except Exception as ex:                       # a probe that crashes is a FAIL, never a silent skip
            p = {"pass": False, "error": f"{type(ex).__name__}: {str(ex)[:300]}"}
        p["seconds"] = round(time.monotonic() - t, 1)
        res["probes"][name] = p
        print(f"{name:15s} {'PASS' if p['pass'] else 'FAIL'}  {p['seconds']:7.1f}s  "
              f"{p.get('error') or p.get('warn') or ''}", flush=True)
    res["pass"] = all(p["pass"] is not False for p in res["probes"].values())
    if a.compare_to:
        old = json.load(open(a.compare_to))
        res["compare"] = {"to": a.compare_to, "to_label": old.get("label"), "to_engram": old.get("engram"),
                          "deltas": {n: {"this": score(p), "other": score(old["probes"].get(n, {})) if
                                         n in old["probes"] else None}
                                     for n, p in res["probes"].items() if not p.get("skipped")}}
        for n, d in res["compare"]["deltas"].items():
            print(f"  vs {old.get('label')}: {n:15s} {d['other']} -> {d['this']}")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"GATE {'PASS' if res['pass'] else 'FAIL'} -> {a.out}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
