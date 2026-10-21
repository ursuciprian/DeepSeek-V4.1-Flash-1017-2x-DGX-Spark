#!/usr/bin/env python3
"""DS41 speed matrix over the OpenAI API: C1/C4/C8 x {code, prose, thinking-75} x {greedy T=0, serve T=1.0 top_p=1.0,
base T=1.0 top_p=0.95} decode (top_k=0 always sent; every request carries a distinct seed), plus cold prefill 4K/32K/128K. Raw per-request records go into the JSON.

  BASE_URL=http://localhost:8888 MODEL=deepseek-v4.1-flash-1017 \\
    python3 bench.py --label base --repeats 3 --out results/bench-base.json
  python3 bench.py --only-prefill --sizes 4096,32768,131072 --out results/prefill.json

Decode per stream = (completion_tokens - 1) / (last token - first token), TTFT excluded; aggregate = all completion
tokens of one round / (last token of any stream - first token of any stream). Cell numbers are medians over repeats.
Prefill = prompt_tokens / TTFT with max_tokens 1, a fresh nonce at the head of every prompt (no prefix-cache hits).
Workloads: code and prose with thinking off; thinking-75 = the thinking prompts with thinking on, effort 75
(reasoning tokens count as decode).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import sys
import time
import uuid

import common

HERE = os.path.dirname(os.path.abspath(__file__))
WORKLOADS = {"code": ("code", False, None), "prose": ("prose", False, None), "thinking75": ("thinking", True, 75)}
SAMPLERS = {"greedy": common.GREEDY, "serve": common.SERVE, "base": common.BASECARD}
FILLER = ("The survey team logged each culvert, gauge and footbridge along the valley road, noting the date, the "
          "water level and any repairs the parish council had asked for in the spring. ")


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def one(a, prompt, sampler, thinking, effort, max_tokens, seed=0):
    extra = {"seed": seed, **({"draft": False} if a.no_draft else {})}
    r = common.chat(a.base, a.model, [{"role": "user", "content": prompt}], max_tokens=max_tokens, sampler=sampler,
                    thinking=thinking, effort=effort, timeout=a.timeout, **extra)
    return {"seed": seed, "tokens": (r["usage"] or {}).get("completion_tokens") or r["chunks"],
            "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"), "finish": r["finish"],
            "ttft": round(r["t_first"] - r["t0"], 4) if r["t_first"] else None, "decode_tps": common.decode_tps(r),
            "t_first": r["t_first"], "t_last": r["t_last"]}


def cell(a, prompts, wl, sname, c):
    key, thinking, effort = WORKLOADS[wl]
    pool, aggs, reqs = prompts[key], [], []
    max_tokens = a.think_tokens if thinking else a.max_tokens
    for rep in range(a.repeats):
        batch = [(pool[(rep * c + i) % len(pool)], 1000 * rep + i) for i in range(c)]   # distinct seeds
        with cf.ThreadPoolExecutor(c) as ex:
            rs = list(ex.map(lambda ps: one(a, ps[0], SAMPLERS[sname], thinking, effort, max_tokens, ps[1]), batch))
        firsts = [r["t_first"] for r in rs if r["t_first"]]
        lasts = [r["t_last"] for r in rs if r["t_last"]]
        win = (max(lasts) - min(firsts)) if firsts and lasts else 0
        aggs.append(sum(r["tokens"] for r in rs) / win if win > 0 else None)
        for r in rs:
            r["repeat"] = rep
        reqs += rs
    return {"workload": wl, "sampler": sname, "c": c, "per_stream_median": med([r["decode_tps"] for r in reqs]),
            "aggregate_median": med(aggs), "aggregates": [round(x, 2) if x else None for x in aggs],
            "ttft_median": med([r["ttft"] for r in reqs]), "requests": reqs}


def prefill(a):
    """Prefill tok/s per size and concurrency: C1 = prompt_tokens / TTFT; Cn = n distinct prompts at once, aggregate
    = their prompt tokens / the slowest TTFT (TF serializes prompts > 32K, so Cn there is mostly queueing)."""
    import concurrent.futures as cf

    out, cpt = [], a.chars_per_token

    def one(size, c):
        body = f"[nonce {uuid.uuid4()}]\n" + FILLER * max(1, int(size * c / len(FILLER)))
        r = common.chat(a.base, a.model, [{"role": "user", "content": body + "\nReply with OK."}],
                        max_tokens=1, sampler=common.GREEDY, thinking=False, timeout=a.timeout)
        return (r["usage"] or {}).get("prompt_tokens"), (r["t_first"] or r["t_end"]) - r["t0"]

    for conc in [int(x) for x in str(a.prefill_conc).split(",")]:
        for size in [int(x) for x in a.sizes.split(",")]:
            rows = []
            for _ in range(a.repeats):
                with cf.ThreadPoolExecutor(conc) as ex:
                    got = list(ex.map(lambda _: one(size, cpt), range(conc)))
                pts = [pt for pt, _ in got if pt]
                ttft = max(t for _, t in got)
                tot = sum(pts) if len(pts) == conc else None
                rows.append({"prompt_tokens": pts[0] if pts else None, "ttft": round(ttft, 3),
                             "ttfts": [round(t, 3) for _, t in got], "tps": round(tot / ttft, 1) if tot else None})
                if pts:
                    cpt *= size / (sum(pts) / len(pts))  # converge on the requested size
            out.append({"size": size, "conc": conc, "tps_median": med([x["tps"] for x in rows]),
                        "ttft_median": med([x["ttft"] for x in rows]), "rows": rows})
            print(f"prefill C{conc} {size:>7}  {out[-1]['tps_median']} tok/s  ttft {out[-1]['ttft_median']} s", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=common.BASE)
    ap.add_argument("--model", default=common.MODEL)
    ap.add_argument("--label", default="run")
    ap.add_argument("--conc", default="1,4,8")
    ap.add_argument("--workloads", default="code,prose,thinking75")
    ap.add_argument("--samplers", default="greedy,serve,base",
                    help="greedy T=0 | serve T=1.0 top_p=1.0 (1017 card) | base T=1.0 top_p=0.95 (base card)")
    ap.add_argument("--no-draft", action="store_true", help='send "draft": false (TF: spec-off control cells)')
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--think-tokens", type=int, default=1024)
    ap.add_argument("--sizes", default="4096,32768,131072")
    ap.add_argument("--prefill-conc", default="1", help="prefill concurrency levels, e.g. 1,4 (aggregate tok/s)")
    ap.add_argument("--chars-per-token", type=float, default=4.4)
    ap.add_argument("--no-prefill", action="store_true")
    ap.add_argument("--only-prefill", action="store_true")
    ap.add_argument("--prompts", default=os.path.join(HERE, "data", "bench_prompts.json"))
    ap.add_argument("--warmup", type=int, default=1, help="untimed requests before the matrix")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", default="bench.json")
    a = ap.parse_args(argv)
    prompts = json.load(open(a.prompts))
    res = {"label": a.label, "base": a.base, "model": a.model, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "args": {k: v for k, v in vars(a).items()}, "cells": [], "prefill": []}
    for _ in range(a.warmup):
        one(a, prompts["code"][0], common.GREEDY, False, None, 32)
    if not a.only_prefill:
        print(f"{'workload':11s} {'sampler':7s} {'c':>2s} {'stream':>8s} {'agg':>8s} {'ttft':>7s}")
        for wl in a.workloads.split(","):
            for sname in a.samplers.split(","):
                for c in [int(x) for x in a.conc.split(",")]:
                    r = cell(a, prompts, wl, sname, c)
                    res["cells"].append(r)
                    print(f"{wl:11s} {sname:7s} {c:>2d} {r['per_stream_median']!s:>8s} {r['aggregate_median']!s:>8s}"
                          f" {r['ttft_median']!s:>7s}", flush=True)
    if not a.no_prefill:
        res["prefill"] = prefill(a)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
