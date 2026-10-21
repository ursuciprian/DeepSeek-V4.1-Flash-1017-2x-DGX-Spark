# gate

Quality gate, speed matrix and exactness checks for DeepSeek V4.1 Flash 1017 on two DGX Sparks. OpenAI API only,
so the same scripts score TensorFold and vLLM. Python 3.11+ standard library (numpy only to read TF draft-log
pickles). Server defaults: `BASE_URL` (default `http://localhost:8888`), `MODEL`
(default `deepseek-v4.1-flash-1017`); every script also takes `--base` / `--model`.

## Gate

```bash
export BASE_URL=http://localhost:8888 MODEL=deepseek-v4.1-flash-1017
python3 gate.py --label base --out results/gate-base.json                 # everything, exit 1 on any FAIL
python3 gate.py --label quick --only reasoning,tool_turn,tool_only_turn,tc45,prefix,corruption --out results/q.json
python3 gate.py --label engram-off --engram off --compare-to results/gate-base.json --out results/gate-eoff.json
```

| probe | checks | pass bar |
|---|---|---|
| reasoning | thinking on (effort 75) and off on 17*23 | on: reasoning_content non-empty + 391; off: no reasoning + 391 |
| tool_turn | tools declared, assistant call, tool result | answer uses the result, no new call |
| tool_only_turn | 1017 encoding fix: thinking on, no tools declared, tool-result-only user turn | reasoning non-empty, 7006652 in answer, prompt_tokens grows >= 50 when the prior reasoning is sent (pre-fix engines drop it: delta 0) |
| tc45 | tool_choice=required on "2 + 2" | calculator call in >= 1 of 3 seeds |
| hardmode | `tool-eval-bench run --hardmode` (must be on PATH) | Quality >= 88 and its TC-45 PASS |
| fidelity | identifier fidelity (ported fidelity_probe.py), 5 targets x 4 seeds per depth, 8K 16K 32K 64K 128K 248K (the 256K-context rung) | 20/20 exact to 64K, >= 19/20 at 128K and 248K |
| prefix | 4 concurrent 4-turn chats on one ~8K-token shared prefix | 16/16 codes exact |
| long_session | multi-turn chapters until >= 28K decoded tokens (cap 32K) | no loop in any reply, target reached |
| long_decode | effort-100 puzzle, max_tokens 32768 | no loop in reasoning or content (length cap without a loop = warning) |
| corruption | two 1024-token essays at T=1.0 (thinking off / effort 75) + every text from the probes above | 0 corrupted chars (U+FFFD, C0 controls, CJK / fullwidth splices), no degenerate essay |

Sampler: 1017 card T=1.0, top_p=1.0, top_k=0 (always sent, so the server default does not apply). `--greedy` runs
T=0. Thresholds live in `gate.py` and were fixed before any DS41 run; do not loosen them after seeing data.
On the current image that sampler crashes both ranks (see the top-level README's Status section). Until the fix
lands, run `gate.py --greedy`, `bench.py --samplers greedy,base` and `exactness.py record-tokens --top-p 0.95`.
`--engram off` only labels the run (boot the server with Engram disabled yourself); `--compare-to` adds per-probe
score deltas against the earlier JSON and does not change pass/fail. The fidelity rung recalibrates chars/token
from each depth's real prompt_tokens; the 248K rung leaves room for 8K of generation in a 262144 window.

## Bench matrix

```bash
python3 bench.py --label base --repeats 3 --out results/bench-base.json        # C1/C4/C8 x code/prose/thinking75 x greedy/serve/base + prefill
python3 bench.py --label specoff --no-draft --no-prefill --out results/bench-specoff.json   # TF: "draft": false
python3 bench.py --only-prefill --sizes 4096,32768,131072 --repeats 3 --out results/prefill.json
```

Samplers: greedy T=0; serve T=1.0 top_p=1.0 (1017 card; on TF this is the slow full-vocab host path); base
T=1.0 top_p=0.95 (base card, TF's fast nucleus path). Every request carries a distinct seed (TF's keyed sampler
repeats a reply exactly for the same prompt and seed). Decode per stream = (completion_tokens - 1) / (last - first
token time), TTFT excluded; aggregate = round tokens / round decode window; medians over repeats. thinking75 =
thinking on, effort 75, reasoning tokens count. Prefill = prompt_tokens / TTFT, max_tokens 1, fresh nonce per
prompt. Raw per-request records are in the JSON.

## Exactness

```bash
# token sequences (works on TF today): greedy + seeds 11,22,33 at T=1.0
python3 exactness.py record-tokens --base $REF --out ref-tokens.json
python3 exactness.py record-tokens --base $CAND --out cand-tokens.json
python3 exactness.py compare ref-tokens.json cand-tokens.json        # first divergence, match; exit 1 on any divergence

# teacher-forced logits: vLLM prompt_logprobs -> TF capture format (also TF cuda/gate.py's --capture input)
python3 exactness.py record-logits --base $VLLM --top-k 20 --out ref-logits.json
python3 exactness.py compare ref-logits.json cand-logits.json            # top1 >= 0.995, mean KL <= 1e-3
python3 exactness.py compare ref-logits.json cand-logits.json --exact    # bit-exact claims: top1 1.0, |dlogprob| <= 1e-5
python3 -m tensorfold.families.deepseek_v41.cuda.gate --model DIR --rank R --master ADDR \
    --capture ref-logits.json --out tf-report.json                        # in-engine, both ranks
python3 exactness.py compare ref-logits.json tf-report.json              # summarises TF's top1 vs that capture

# DSpark acceptance, the numerics canary
python3 exactness.py accept --base $CAND --out acc-cand.json --ref acc-ref.json --max-drop 0.02
python3 exactness.py accept --base "" --draft-log /state/draftlog --request-log /state/requests.jsonl --out acc.json
```

KL over truncated top-k is an approximation, documented in `exactness.py`: a token missing from the candidate's
top-k gets the candidate's lowest top-k logprob, plus one tail bucket. Acceptance sources: each TF response's
`tensorfold.stats.drafting` (dspark_kept / dspark_drafts, tokens_per_round), vLLM `/metrics`
`spec_decode_num_accepted_tokens_per_pos` / `spec_decode_num_drafts`, TF request log jsonl (tokens_per_round), and
`TF_DSV41_DRAFT_LOG` pickles for per-position keep rates (TF HTTP has no per-position data).

## Check

```bash
python3 tests/test_smoke.py      # gate + bench + exactness against tests/fake_server.py, < 1 s
python3 common.py && python3 fidelity_probe.py --selftest
```

## Provenance

`fidelity_probe.py`, the loop detector, the corruption checks and the hardmode / TC-45 rule (3 runs, >= 1 PASS)
come from the gate of my Qwen3.8 recipe repositories (Apache-2.0). The capture format is the one TensorFold's
in-engine gate (`cuda/gate.py`) reads; I read it for compatibility and copied no code.

## TF vs TF logits (reference build = a TensorFold build)

TensorFold's HTTP path has no logprobs, and its FP8 KV differs from DeepSeek's FP4 reference, so the logits
reference is a TF build. On both ranks of the reference image run
`python -m tensorfold.families.deepseek_v41.cuda.gate --model /model --rank R --master H --capture <any oracle> --out ref-report.json --dump ref.json`
(in the engine image), then on the candidate `--capture ref.json --kit-replies ref.json --out cand-report.json`
(top-1 agreement, logprob error, identical greedy replies), and `python3 exactness.py compare ref.json cand-dump.json`
for top-1 / KL over the top-5.
