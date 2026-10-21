#!/usr/bin/env python3
"""DS41 exactness harness: candidate build vs reference build, on fixed prompts.

Two transports, because TensorFold's DS41 HTTP path has no logprobs:

1. tokens (works on TF and vLLM today): /v1/completions on fixed prompts, greedy and fixed-seed T>0, recording the
   reply token ids (``return_token_ids``; text as fallback). TF samples by keyed Gumbel-max on (seed, position,
   token), so the same prompt + seed must give the same reply on two exact builds.
     exactness.py record-tokens --base URL --model M --out ref-tokens.json
     exactness.py compare ref-tokens.json cand-tokens.json          # first divergence, match rate; exit 1 on any
2. logits, teacher-forced (vLLM over HTTP; TF in-engine): the dump format is TF's capture format
   ({"prompts": [{"id", "ids": [...], "positions": [null, {"top": [[token_id, logprob], ...]}, ...]}]}, positions[i]
   is the distribution that predicted ids[i]) so one file feeds both this compare and TF's own gate:
     exactness.py record-logits --base URL --model M --top-k 20 --out ref-logits.json      # vLLM prompt_logprobs
     python -m tensorfold.families.deepseek_v41.cuda.gate --model DIR --rank R --master ADDR \\
         --capture ref-logits.json --out tf-report.json                                    # TF, both ranks
     exactness.py compare ref-logits.json cand-logits.json [--min-top1 0.995 --max-mean-kl 1e-3 | --exact]
     exactness.py compare ref-logits.json tf-report.json            # summarises TF gate's top1 vs that capture

KL over truncated top-k (documented approximation): KL(P||Q) ~= sum_{t in topk(P)} p_t (log p_t - log q_t)
+ p_tail log(p_tail / q_tail), where a token missing from Q's top-k gets log q = min(Q's top-k logprob) (an upper
bound on q, so that term is under-estimated), p_tail = 1 - sum p over P's top-k and q_tail = 1 - sum of the q used
(floored at 1e-9). It is a lower-bound-ish estimate; with k = 20 the tail mass is usually < 1e-3.

3. DSpark acceptance (the numerics canary): per-position keep rates and tokens per round.
     exactness.py accept --base URL --model M --out acc.json [--ref acc-ref.json --max-drop 0.02]
   reads, whatever exists: each TF response's ``tensorfold.stats.drafting`` {windows, rows, dspark_rounds,
   dspark_drafts, dspark_kept, lookup_rounds, lookup_drafts, lookup_kept, dspark_tokens_per_round,
   tokens_per_round}; vLLM /metrics ``spec_decode_num_accepted_tokens_per_pos`` / ``spec_decode_num_drafts`` deltas;
   ``--request-log`` TF jsonl (tokens_per_round, rounds, decode_tps); ``--draft-log DIR`` TF_DSV41_DRAFT_LOG pickles
   ("d" events carry k drafts verified, "k" events the drafts accepted: per-position keep rate). Per-position data
   from TF HTTP alone does not exist (m2bench's depth.Stats has it): use --draft-log for that.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import urllib.request

import common

HERE = os.path.dirname(os.path.abspath(__file__))


def prompts(path):
    return json.load(open(path or os.path.join(HERE, "data", "exact_prompts.json")))


# -- record ---------------------------------------------------------------------------------------------------------
def token_ids(resp: dict, choice: dict, key: str) -> list[int] | None:
    for src in (choice, resp):
        for k in key.split("|"):
            v = src.get(k)
            if isinstance(v, list) and all(isinstance(x, int) for x in v):
                return v
    return None


def record_tokens(a):
    """Every (prompt, sampler) reply; ``--conc N`` sends them N at a time from one barrier (the batched == serial check:
    record at --conc 1 and --conc 8, then ``compare``; the output order is the same either way)."""

    from concurrent.futures import ThreadPoolExecutor

    out = {"kind": "tokens", "base": a.base, "model": a.model, "conc": a.conc, "prompts": []}
    samplers = [("greedy", {**common.GREEDY}, None)] + [
        (f"T1-seed{s}", {"temperature": 1.0, "top_p": a.top_p, "top_k": 0}, s) for s in a.seeds]
    ps = prompts(a.prompts)
    jobs = [(p, name, smp, seed) for p in ps for name, smp, seed in samplers]

    def one(job):
        p, name, smp, seed = job
        body = {"model": a.model, "prompt": p["text"], "max_tokens": a.max_tokens, "return_token_ids": True,
                **smp, **({"seed": seed} if seed is not None else {})}
        r = common.post(a.base, "/v1/completions", body, a.timeout)
        ch = r["choices"][0]
        return {"sampler": name, "seed": seed, "text": ch.get("text", ""),
                "ids": token_ids(r, ch, "token_ids|output_token_ids")}

    got = []
    with ThreadPoolExecutor(max(1, a.conc)) as ex:
        for i in range(0, len(jobs), max(1, a.conc)):         # one barrier per group of ``conc``
            got += list(ex.map(one, jobs[i:i + max(1, a.conc)]))
    k = len(samplers)
    for j, p in enumerate(ps):
        out["prompts"].append({"id": p["id"], "runs": got[j * k:(j + 1) * k]})
        print(f"{p['id']:10s} {k} runs", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


def record_logits(a):
    out = {"kind": "logits", "base": a.base, "model": a.model, "top_k": a.top_k, "prompts": []}
    for p in prompts(a.prompts):
        body = {"model": a.model, "prompt": p["text"], "max_tokens": 1, "temperature": 0, "prompt_logprobs": a.top_k,
                "return_token_ids": True}
        r = common.post(a.base, "/v1/completions", body, a.timeout)
        ch = r["choices"][0]
        ids = token_ids(r, ch, "prompt_token_ids")
        if ids is None:
            ids = common.post(a.base, "/tokenize", {"model": a.model, "prompt": p["text"]}, a.timeout)["tokens"]
        pl = ch.get("prompt_logprobs") or r.get("prompt_logprobs")
        if pl is None:
            sys.exit("server returned no prompt_logprobs: logits mode needs vLLM (TF: use its cuda/gate.py capture)")
        pos = [None if d is None else {"top": sorted(([int(t), float(v["logprob"] if isinstance(v, dict) else v)]
                                                      for t, v in d.items()), key=lambda x: -x[1])[:a.top_k]}
               for d in pl]
        out["prompts"].append({"id": p["id"], "ids": ids, "positions": pos})
        print(f"{p['id']:10s} {len(ids)} positions", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


# -- compare --------------------------------------------------------------------------------------------------------
def kl_topk(p: list, q: list) -> float:
    qd = {t: lp for t, lp in q}
    floor = min(lp for _, lp in q)
    kl = ps = qs = 0.0
    for t, lp in p:
        lq = qd.get(t, floor)
        pp = math.exp(lp)
        kl += pp * (lp - lq)
        ps += pp
        qs += math.exp(lq)
    pt, qt = max(1 - ps, 0.0), max(1 - qs, 1e-9)
    return kl + (pt * math.log(pt / qt) if pt > 0 else 0.0)


def first_div(x, y):
    for i, (u, v) in enumerate(zip(x, y)):
        if u != v:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


def compare_tokens(ref, cand, a):
    rows, bad = [], 0
    cmap = {p["id"]: {r["sampler"]: r for r in p["runs"]} for p in cand["prompts"]}
    for p in ref["prompts"]:
        for r in p["runs"]:
            c = cmap.get(p["id"], {}).get(r["sampler"])
            if c is None:
                continue
            x, y = (r["ids"], c["ids"]) if r["ids"] is not None and c["ids"] is not None else (r["text"], c["text"])
            d = first_div(x, y)
            n = max(len(x), len(y), 1)
            match = sum(u == v for u, v in zip(x, y)) / n
            bad += d is not None
            rows.append({"prompt": p["id"], "sampler": r["sampler"], "unit": "ids" if r["ids"] is not None else "chars",
                         "first_divergence": d, "match": round(match, 4), "len": [len(x), len(y)]})
            print(f"{p['id']:10s} {r['sampler']:10s} first_div {d!s:>5s} match {match:.3f}")
    ok = bad == 0 or a.allow_divergence
    return {"kind": "tokens", "pass": ok, "diverged": bad, "total": len(rows), "rows": rows}


def compare_logits(ref, cand, a):
    cmap = {p["id"]: p for p in cand["prompts"]}
    per_pos: dict[int, list] = {}
    tops, kls, maxd, rows = [], [], 0.0, []
    for p in ref["prompts"]:
        c = cmap.get(p["id"])
        if c is None:
            continue
        if c["ids"] != p["ids"]:
            rows.append({"prompt": p["id"], "error": "token ids differ (different tokenizer or prompt)"})
            continue
        t1 = []
        for i, (u, v) in enumerate(zip(p["positions"], c["positions"])):
            if not u or not v or not u.get("top") or not v.get("top"):
                continue
            hit = u["top"][0][0] == v["top"][0][0]
            kl = kl_topk(u["top"], v["top"])
            vd = dict(map(tuple, v["top"]))
            maxd = max([maxd] + [abs(lp - vd[t]) for t, lp in u["top"] if t in vd])
            tops.append(hit); kls.append(kl); t1.append(hit)
            per_pos.setdefault(i, []).append(kl)
        rows.append({"prompt": p["id"], "positions": len(t1), "top1": round(sum(t1) / max(len(t1), 1), 4)})
    if not tops:
        return {"kind": "logits", "pass": False, "error": "no comparable positions", "rows": rows}
    ks = sorted(kls)
    res = {"kind": "logits", "positions": len(tops), "top1": sum(tops) / len(tops), "mean_kl": sum(kls) / len(kls),
           "p99_kl": ks[int(0.99 * (len(ks) - 1))], "max_kl": ks[-1], "max_abs_logprob_diff": maxd, "rows": rows,
           "kl_by_position_bucket": {f"{b * 64}-{b * 64 + 63}": sum(x) / len(x) for b, x in
                                     _buckets(per_pos, 64).items()}}
    if a.exact:
        res["pass"] = res["top1"] == 1.0 and maxd <= 1e-5
    else:
        res["pass"] = res["top1"] >= a.min_top1 and res["mean_kl"] <= a.max_mean_kl
    print(f"top1 {res['top1']:.4f}  mean KL {res['mean_kl']:.2e}  p99 {res['p99_kl']:.2e}  max {res['max_kl']:.2e}  "
          f"max|dlogprob| {maxd:.2e}  over {len(tops)} positions")
    return res


def _buckets(per_pos, w):
    out: dict[int, list] = {}
    for i, xs in per_pos.items():
        out.setdefault(i // w, []).extend(xs)
    return dict(sorted(out.items()))


def compare(a):
    ref, cand = json.load(open(a.ref)), json.load(open(a.cand))
    if "top1_agreement" in cand:                 # TF cuda/gate.py report against this capture
        res = {"kind": "tf-gate-report", "top1": cand["top1_agreement"],
               "top1_first_copy": cand.get("top1_first_copy"), "pass": cand["top1_agreement"] >= a.min_top1,
               "tf_gate_pass": cand.get("pass")}
        print(f"TF gate report: top1 {res['top1']:.4f} (first copy {res['top1_first_copy']})")
    elif ref.get("kind") == "tokens":
        res = compare_tokens(ref, cand, a)
    else:
        res = compare_logits(ref, cand, a)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print("EXACTNESS", "PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 1


# -- acceptance -----------------------------------------------------------------------------------------------------
def metrics(base):
    try:
        text = urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or "spec_decode" not in line:
            continue
        name, val = line.rsplit(" ", 1)
        pos = re.search(r'position="(\d+)"', name)
        key = name.split("{")[0] + (f"[{pos.group(1)}]" if pos else "")
        try:
            out[key] = out.get(key, 0.0) + float(val)
        except ValueError:
            pass
    return out


DRAFT_KEYS = ("windows", "rows", "dspark_rounds", "dspark_drafts", "dspark_kept", "lookup_rounds", "lookup_drafts",
              "lookup_kept")


def draft_log_positions(folder):
    import pickle                                  # our own server's logs (trusted); numpy needed for the arrays
    events = []
    for f in sorted(glob.glob(os.path.join(folder, "draftlog-*.pkl"))):
        with open(f, "rb") as fh:
            events += pickle.load(fh)["events"]
    pend, offered, kept = {}, {}, {}
    for kind, ev in events:
        if kind == "d":
            pend[ev["slot"]] = ev["k"]
        elif kind == "k" and ev["slot"] in pend:
            k, acc = pend.pop(ev["slot"]), ev["accepted"]
            for i in range(k):
                offered[i] = offered.get(i, 0) + 1
                kept[i] = kept.get(i, 0) + (i < acc)
    return {str(i): round(kept[i] / offered[i], 4) for i in sorted(offered)}, sum(offered.values())


def accept(a):
    res = {"base": a.base, "model": a.model, "sampler": a.sampler}
    if a.base:
        smp = common.GREEDY if a.sampler == "greedy" else {**common.SERVE}
        m0, tot = metrics(a.base), {k: 0 for k in DRAFT_KEYS}
        tokens = rounds = 0.0
        pr = json.load(open(os.path.join(HERE, "data", "bench_prompts.json")))
        for wl in ("code", "prose"):
            for i, q in enumerate(pr[wl][:a.n]):
                body = {"model": a.model, "messages": [{"role": "user", "content": q}], "max_tokens": a.max_tokens,
                        "seed": 1000 + i, **smp, **common.think(False)}
                r = common.post(a.base, "/v1/chat/completions", body, a.timeout)
                d = ((r.get("tensorfold") or {}).get("stats") or {}).get("drafting") or {}
                for k in DRAFT_KEYS:
                    tot[k] += d.get(k) or 0
                n = (r.get("usage") or {}).get("completion_tokens") or 0
                if d.get("tokens_per_round"):
                    tokens, rounds = tokens + n, rounds + n / d["tokens_per_round"]
        m1 = metrics(a.base)
        delta = {k: m1[k] - m0.get(k, 0) for k in m1}
        drafts = next((v for k, v in delta.items() if k.endswith("spec_decode_num_drafts_total")), 0)
        res["tf_drafting"] = tot if any(tot.values()) else None
        if res["tf_drafting"]:
            res["dspark_keep_per_draft"] = tot["dspark_kept"] / max(tot["dspark_drafts"], 1)
            res["dspark_tokens_per_round"] = (tot["dspark_kept"] + tot["dspark_rounds"]) / max(tot["dspark_rounds"], 1)
            res["lookup_keep_per_draft"] = tot["lookup_kept"] / max(tot["lookup_drafts"], 1)
            res["tokens_per_round"] = tokens / rounds if rounds else None
        if drafts:
            res["vllm_per_position"] = {k.split("[")[1].rstrip("]"): round(v / drafts, 4) for k, v in delta.items()
                                        if "accepted_tokens_per_pos" in k and "[" in k}
    if a.request_log:
        recs = [json.loads(x) for x in open(a.request_log) if x.strip()]
        recs = [r for r in recs if r.get("tokens_per_round") and r.get("decode_tokens")]
        tok = sum(r["decode_tokens"] for r in recs)
        res["request_log"] = {"requests": len(recs), "tokens_per_round": tok / max(sum(
            r["decode_tokens"] / r["tokens_per_round"] for r in recs), 1e-9) if recs else None}
    if a.draft_log:
        res["draft_log_per_position"], res["draft_log_offers"] = draft_log_positions(a.draft_log)
    print(json.dumps({k: v for k, v in res.items() if k != "tf_drafting"}, indent=1))
    ok = True
    if a.ref:
        ref = json.load(open(a.ref))
        drops = {}
        for key in ("draft_log_per_position", "vllm_per_position"):
            for i, v in (res.get(key) or {}).items():
                r0 = (ref.get(key) or {}).get(i)
                if r0 is not None and r0 - v > a.max_drop:
                    drops[f"{key}[{i}]"] = round(r0 - v, 4)
        for key in ("dspark_keep_per_draft", "tokens_per_round"):
            if res.get(key) is not None and ref.get(key) and (ref[key] - res[key]) / ref[key] > a.max_drop:
                drops[key] = round(ref[key] - res[key], 4)
        res["drops_vs_ref"] = drops
        ok = not drops
        print("ACCEPTANCE", "PASS" if ok else f"FAIL {drops}")
    json.dump(res, open(a.out, "w"), indent=1)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record-tokens", "record-logits", "accept"):
        s = sub.add_parser(name)
        s.add_argument("--base", default=common.BASE)
        s.add_argument("--model", default=common.MODEL)
        s.add_argument("--timeout", type=float, default=1800)
        s.add_argument("--out", required=True)
    rt, rl, ac = sub.choices["record-tokens"], sub.choices["record-logits"], sub.choices["accept"]
    for s in (rt, rl):
        s.add_argument("--prompts", default="")
    rt.add_argument("--max-tokens", type=int, default=256)
    rt.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[11, 22, 33])
    rt.add_argument("--top-p", type=float, default=1.0)
    rt.add_argument("--conc", type=int, default=1, help="requests in flight together (batched == serial check)")
    rl.add_argument("--top-k", type=int, default=20)
    ac.add_argument("--sampler", choices=["greedy", "serve"], default="greedy")
    ac.add_argument("--n", type=int, default=4, help="prompts per workload (code, prose)")
    ac.add_argument("--max-tokens", type=int, default=384)
    ac.add_argument("--request-log", default="")
    ac.add_argument("--draft-log", default="")
    ac.add_argument("--ref", default="")
    ac.add_argument("--max-drop", type=float, default=0.02, help="absolute per position, relative for aggregates")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("cand")
    c.add_argument("--min-top1", type=float, default=0.995)
    c.add_argument("--max-mean-kl", type=float, default=1e-3)
    c.add_argument("--exact", action="store_true", help="top1 == 1 and every shared logprob within 1e-5")
    c.add_argument("--allow-divergence", action="store_true", help="tokens mode: report only")
    c.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.cmd == "accept" and not a.base and not (a.request_log or a.draft_log):
        ap.error("accept needs --base, --request-log or --draft-log")
    return {"record-tokens": record_tokens, "record-logits": record_logits, "compare": compare,
            "accept": accept}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
