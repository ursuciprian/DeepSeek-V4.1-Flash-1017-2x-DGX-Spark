#!/usr/bin/env python3
"""Ported from qwen3.8-flash-next/scripts/fidelity_probe.py (same author); DS41: ask() goes through
common.chat (thinking kwargs, serving sampler), defaults T=1.0 and depths 8K-128K plus the 256K rung.

Identifier-fidelity probe: synthetic long agent transcript with near-duplicate paths planted
at spread depths, then a final turn that forces the model to reproduce one exact path inside a
bash tool call. Scores exact / near-miss (typo, distance 1-3) / wrong / no_call, plus whether the
produced path is hallucinated (not present anywhere in the transcript).

  fidelity_probe.py --base http://host:8000 --model m --depths 8000,32000 --out fid.json
  fidelity_probe.py --selftest
"""
import argparse, json, random, re, sys

import common
from concurrent.futures import ThreadPoolExecutor

CLIENT_BASES = ["acme-billing", "acme-billling", "acme-biling", "umbra-systems", "umbra-systms",
                 "nordic-freight", "nordic-frieght", "delta-logix", "delta-logic",
                 "vertex-cargo", "vertexx-cargo", "orryn-data", "orryn-datta"]
SERVICES = ["svc-ingest_v2", "svc-ingest-v2", "svc-ingest_v3", "svc-export_v1", "svc-export-v1",
            "svc-sync_v4", "svc-relay_v2", "svc-relay-v2", "svc-reconcile_v1"]
FILES = ["ingest.log", "export.log", "sync.log", "error.log", "access.log"]
TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Run a shell command.",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                          "required": ["command"]}}}]


def hexid(rng, n=6):
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def rand_path(rng):
    d = f"/srv/clients/{rng.choice(CLIENT_BASES)}-{hexid(rng)}/{rng.choice(SERVICES)}"
    return f"{d}/logs/{rng.choice(FILES)}"


def call(idx, cmd):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"call_{idx}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]}


def result(idx, text):
    return {"role": "tool", "tool_call_id": f"call_{idx}", "content": text}


def filler_block(rng, idx, all_paths):
    kind = rng.choice(["ls", "find", "status"])
    paths = [rand_path(rng) for _ in range(rng.randint(2, 4))]
    all_paths.update(paths)
    if kind == "ls":
        cmd = f"ls -la {paths[0].rsplit('/', 1)[0]}"
        body = "\n".join(f"-rw-r--r-- 1 svc svc {rng.randint(100,99999):>6} Sep {rng.randint(1,30):02d} {p.rsplit('/',1)[1]}" for p in paths)
    elif kind == "find":
        cmd = "find /srv/clients -name '*.log' -mmin -60"
        body = "\n".join(paths)
    else:
        cmd = "git status -s"
        body = "\n".join(f" M configs/{rng.choice(CLIENT_BASES)}-{hexid(rng)}.yaml" for _ in paths)
    return [call(idx, cmd), result(idx, body)]


def target_block(rng, idx, tid, all_paths):
    path = rand_path(rng)
    all_paths.add(path)
    cmd = f"grep -rl 'invoice_id' {path.rsplit('/', 1)[0]}"
    body = f"{path}: invoice_id={tid} processed {2026}-{rng.randint(1,9):02d}-{rng.randint(1,28):02d}"
    return [call(idx, cmd), result(idx, body)], {"id": tid, "path": path}


def build_transcript(depth, k, seed, chars_per_token):
    rng = random.Random(seed * 1000 + depth)
    system = {"role": "system", "content": "You are an autonomous coding agent with a bash tool. Use the bash tool for every shell action; never fabricate output."}
    target_chars = depth * chars_per_token
    fracs = [0.10 + i * (0.80 / (k - 1)) for i in range(k)] if k > 1 else [0.5]
    msgs = [system]
    all_paths, targets = set(), []
    chars, idx, next_t = len(system["content"]), 0, 0
    while chars < target_chars or next_t < k:
        blk = filler_block(rng, idx, all_paths)
        msgs += blk
        chars += sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls", ""))) for m in blk)
        idx += 1
        if next_t < k and chars >= fracs[next_t] * target_chars:
            tid = f"INV-{1000 + next_t}-{hexid(rng, 4).upper()}"
            blk, tgt = target_block(rng, idx, tid, all_paths)
            msgs += blk
            chars += sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls", ""))) for m in blk)
            idx += 1
            targets.append(tgt)
            next_t += 1
        if idx > 200000:
            break  # ponytail: hard stop, avoids runaway loop if chars_per_token estimate is off
    return msgs, targets, all_paths


def lev(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb]


PATH_RE = re.compile(r"/[\w./-]+")


def score_command(target_path, other_paths, command):
    cands = PATH_RE.findall(command or "")
    if not cands:
        return "wrong", None, False, ""
    best = min(cands, key=lambda c: lev(c, target_path))
    d = lev(best, target_path)
    hallucinated = best != target_path and best not in other_paths
    cat = "exact" if d == 0 else "near" if d <= 3 else "wrong"
    if cat == "wrong" and target_path.startswith(best.rstrip("/") + "/"):
        cat = "explore"  # listed the right directory first; a real agent gets another turn
    return cat, d, hallucinated, best


def score_trial(target_path, other_paths, message):
    tcs = (message or {}).get("tool_calls") or []
    bash_calls = [t for t in tcs if t.get("function", {}).get("name") == "bash"]
    if not bash_calls:
        return "no_call", None, False, ""
    try:
        cmd = json.loads(bash_calls[0]["function"]["arguments"]).get("command", "")
    except Exception:
        cmd = bash_calls[0]["function"].get("arguments", "")
    return score_command(target_path, other_paths, cmd)


def ask(base, model, messages, temperature, max_tokens, thinking, timeout, seed=0):
    r = common.chat(base, model, messages, max_tokens=max_tokens, tools=TOOLS, tool_choice="auto", seed=seed,
                    sampler={"temperature": temperature, "top_p": 1.0, "top_k": 0}, thinking=thinking, timeout=timeout)
    message = {"content": r["content"] or None, "tool_calls": r["tool_calls"] or None}
    ttft = (r["t_first"] - r["t0"]) if r["t_first"] else 0.0
    return message, r["usage"], ttft, (r["t_last"] or r["t0"]) - r["t0"]


def run_depth(base, model, depth, k, trials, seed, chars_per_token, temperature, max_tokens,
              thinking, concurrency, timeout):
    msgs, targets, all_paths = build_transcript(depth, k, seed, chars_per_token)
    ask_q = ("The log for invoice_id {tid} needs a check. Call the bash tool with exactly one command: "
             "`tail -n 50` on that client's {fname} (the exact path that appeared earlier), nothing else.")
    jobs = [(t, trial) for t in targets for trial in range(trials)]

    def one(job):
        t, trial = job
        conv = msgs + [{"role": "user", "content": ask_q.format(tid=t["id"], fname=t["path"].rsplit("/", 1)[1])}]
        try:
            message, usage, ttft, dt = ask(base, model, conv, temperature, max_tokens, thinking == "on", timeout, seed=trial)
        except Exception as ex:
            return {"target": t["id"], "path": t["path"], "cat": "no_call", "dist": None,
                    "hallucinated": False, "cmd": f"ERROR {type(ex).__name__}: {str(ex)[:150]}",
                    "prompt_tokens": None, "ttft": None, "dt": None}
        cat, dist, hall, cmd = score_trial(t["path"], all_paths, message)
        if cat == "explore":
            listing = "\n".join(sorted(p.rsplit("/", 1)[1] for p in all_paths if p.startswith(cmd.rstrip("/") + "/")))
            conv2 = conv + [dict(role="assistant", content=message.get("content"), tool_calls=message["tool_calls"]),
                            {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": listing or "(empty)"},
                            {"role": "user", "content": "Now call bash once with `tail -n 50` on that exact full path."}]
            try:
                message, usage, ttft2, dt2 = ask(base, model, conv2, temperature, max_tokens, thinking == "on", timeout, seed=trial)
                cat, dist, hall, cmd = score_trial(t["path"], all_paths, message)
                ttft, dt = ttft, dt + dt2
            except Exception as ex:
                cmd = f"FOLLOWUP ERROR {type(ex).__name__}: {str(ex)[:120]}"
        return {"target": t["id"], "path": t["path"], "cat": cat, "dist": dist, "hallucinated": hall,
                "cmd": cmd, "prompt_tokens": usage.get("prompt_tokens"), "ttft": ttft, "dt": dt}

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(one, jobs))

    counts = {"exact": 0, "near": 0, "wrong": 0, "explore": 0, "no_call": 0}
    for r in results:
        counts[r["cat"]] += 1
    denom = counts["exact"] + counts["near"]
    typo_rate = counts["near"] / denom if denom else 0.0
    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    dts = [r["dt"] for r in results if r["dt"] is not None]
    toks = [r["prompt_tokens"] for r in results if r["prompt_tokens"]]
    actual = max(toks) if toks else 0
    return {"depth": depth, "actual_prompt_tokens": actual, "counts": counts, "typo_rate": typo_rate,
            "mean_ttft": sum(ttfts) / len(ttfts) if ttfts else None,
            "mean_latency": sum(dts) / len(dts) if dts else None, "results": results}


def selftest():
    a, b = "/srv/clients/acme-billing-7f3a9c/svc-ingest_v2/logs/ingest.log", "/srv/clients/acme-billling-7f3a9e/svc-ingest_v2/logs/ingest.log"
    assert lev(a, a) == 0
    assert lev(a, "abc") > 3
    others = {"/srv/clients/other-111111/svc-x/logs/y.log"}
    assert score_command(a, others, f"tail -n 50 {a}")[0] == "exact"
    cat, d, hall, best = score_command(a, others, f"tail -n 50 {b}")
    assert cat == "near" and 1 <= d <= 3 and hall, (cat, d, hall)  # typo path, not verbatim anywhere -> hallucinated
    cat2 = score_command(a, others, "tail -n 50 /srv/clients/other-111111/svc-x/logs/y.log")[0]
    assert cat2 == "wrong"
    cat3, d3, hall3, _ = score_command(a, others, "tail -n 50 /totally/made/up/path/nowhere.log")
    assert cat3 == "wrong" and hall3
    assert score_command(a, others, "ls -la " + a.rsplit("/", 1)[0])[0] == "explore"
    assert score_trial(a, others, {"tool_calls": []})[0] == "no_call"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--model")
    ap.add_argument("--depths", default="8000,16000,32000,64000,128000,248000")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--thinking", choices=["on", "off"], default="on")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--chars-per-token", type=float, default=4.6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="fidelity.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.base or not a.model:
        ap.error("--base and --model required unless --selftest")
    summary = []
    for depth in [int(x) for x in a.depths.split(",")]:
        r = run_depth(a.base, a.model, depth, a.k, a.trials, a.seed, a.chars_per_token, a.temperature,
                       a.max_tokens, a.thinking, a.concurrency, a.timeout)
        c = r["counts"]
        print(f"depth {depth:>7} | actual_tokens {r['actual_prompt_tokens']:>7} | "
              f"exact {c['exact']:>3} near {c['near']:>3} wrong {c['wrong']:>3} explore {c['explore']:>3} no_call {c['no_call']:>3} | "
              f"typo_rate {r['typo_rate']:.2f} | ttft {r['mean_ttft'] or 0:6.2f}s | lat {r['mean_latency'] or 0:6.1f}s", flush=True)
        summary.append(r)
    json.dump({"args": vars(a), "depths": summary}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
