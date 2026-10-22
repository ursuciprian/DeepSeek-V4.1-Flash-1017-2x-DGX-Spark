# Benchmarks

Every number here comes from my two DGX Sparks. The newest results are first; the older sections follow
unchanged. Chart data for the README: [docs/data/bench.csv](docs/data/bench.csv).

## 2026-10-22: v1.0.1 release recipe

v1.0.1 is the v1.0.0 image with notice changes only (same engine commit `ee6ff92`). The release recipe differs from
the 2026-10-21 boots in three settings: a KV pool of 1,048,576 tokens ([#8](../../issues/8)), the engine's default
session store (on), and `TF_DSV41_GRAPH_ROWS_MAX=32`, which I set from the A/B below. Two full gate runs passed on
these settings (one with the 1M pool alone, one with all three changes): every lane, exactness 32 of 32 at C1 and
C8, lowest MemAvailable 6.7 GiB.

Graph rows A/B, same image and recipe, boots in the order 64, 32, 32, 64. Card sampler (T=1.0, top_p 1.0), 512-token
replies, aggregate tok/s, median of 3 per boot:

| Workload | C | 64 (default), boot 1 / 2 | 32, boot 1 / 2 |
|---|--:|--:|--:|
| code | 1 | 88.2 / 89.2 | 88.0 / 89.4 |
| code | 4 | 135.6 / 127.8 | 128.7 / 127.2 |
| code | 8 | 155.8 / 159.1 | 163.1 / 164.4 |
| prose | 1 | 44.3 / 44.4 | 44.2 / 44.3 |
| prose | 4 | 85.0 / 85.2 | 85.5 / 84.7 |
| prose | 8 | 112.8 / 113.1 | 112.0 / 111.1 |
| thinking-75 | 1 | 74.7 / 75.0 | 75.0 / 74.4 |
| thinking-75 | 4 | 117.4 / 115.7 | 116.8 / 115.0 |
| thinking-75 | 8 | 133.8 / 133.3 | 134.4 / 132.9 |

32 rows gives code at C8 about 4% more; the other cells move within run-to-run noise. The decode tables in the
next section ran on the 2026-10-21 settings; on the release recipe code at C8 lands at 163-164 instead of 167.5.

### Prefill on real text

The prefill tables in the older sections use one paragraph repeated after a fresh nonce. I measured the release
recipe again with real text: a fixed shuffle of the docs, changelogs, Python package docs and license texts on a
Spark, cut with the model's tokenizer to the target length, a different slice for every prompt, a fresh nonce at the
start, `max_tokens` 1, one request at a time (C1), median of 3.

| Prompt tokens | Real text tok/s | TTFT | Repeated paragraph tok/s (2026-10-21) |
|--:|--:|--:|--:|
| 4,116 | 1,131 | 3.6 s | 1,262 |
| 32,790 | 1,665 | 19.7 s | 2,424 |
| 131,093 | 1,831 | 71.6 s | 2,391 |

Real text is 10% slower at 4K, 31% at 32K and 23% at 128K. I have not isolated the cause yet. The three prompts per
size gave 1,068 / 1,131 / 1,147, 1,641 / 1,665 / 1,672 and 1,763 / 1,831 / 1,835 tok/s. A first run with prompts cut
by character count (3,000 to 165,000 tokens) gave the same picture. Lowest MemAvailable 7.2 GiB.

Time to first token on a short prompt (one sentence, `max_tokens` 1, median of 5): 0.338 s on `/v1/chat/completions`
and 0.225 s on `/v1/completions`.

## 2026-10-21: v1.0.0 (cand4)

What changed against 2026-10-19, all in the recipe now:

- Depth policy. `TF_DSV41_DEPTH=ratio` picks each round's draft length to maximize expected tokens per unit of
  round time. It prices a round from a table of measured wall times for a whole single-request round by verify rows
  1 to 8 (`TF_DSV41_ROUND_MS=26.26,31.08,35.93,40.52,43.32,46.42,51.29,54.28`, in ms). Draft cap 5 (3 before),
  joint depth mode 2 for rounds with several requests (`TF_DSV41_DEPTH_JOINT=2`, 0 before) ([#2](../../issues/2)).
- Two scheduling knobs, set on every measured boot: `TF_DSV41_ADMIT_HOLD_MS=10` and
  `TF_DSV41_PREFILL_FIRST_ROWS=16384`.
- Cheaper verify rows. `TF_DSV41_MHC_FIN_CTAS=8` spreads the mHC finish over up to 8 CTAs, and
  `TF_DSV41_RG_SEL_ROWS=2` with `TF_DSV41_RG_NARROW=4` (2 before) changes how the router kernel handles small row
  counts. The kernels produce the same bits ([#3](../../issues/3)).
- The Engram gate runs in a C worker (`TF_DSV41_ENGRAM_CGATE=1`) that reads rows with io_uring. Docker's default
  seccomp profile refuses io_uring, so the recipe now uses `recipes/seccomp-io-uring.json`: Docker's default
  profile plus the three io_uring syscalls. The decode and prefill tables below were measured with seccomp off
  for the container. A later boot under the narrow profile had io_uring active, passed the full gate, and its
  greedy probe (set b 90.0 / 49.4 / 135.2, set a 83.1 / 46.9 / 99.2 tok/s) is 0.2 to 1.0 tok/s (under 1%)
  below the slowest of four other boots with these settings on three of six cells, and within their range on the
  other three.
- Metrics endpoint. `GET /metrics` on the serving port returns Prometheus text: requests, errors, prompt,
  completion and cached tokens, decode rounds, drafted and accepted draft tokens, decode and prefill seconds.
- Not in this release: the drafter head trim ([#4](../../issues/4)). The boot log reports the full draft head.

Setup: engine image `ds41-tf:cand4`, built from engine commit `ee6ff92`, published as
`ghcr.io/ursuciprian/ds41-tensorfold:v1.0.0`. Same weights, hardware, prompts and script as before. Booted with
sparkrun. The booted recipe differs from the file in [recipes/](recipes/) in host paths, the Engram mount point,
the request log name, the rendezvous address, an extra `SYS_PTRACE` capability for debugging, the server
default sampler (top_p 1.0 on the boot, 0.95 in the file), and seccomp unconfined instead of the narrow profile. Every benchmark and gate request sends its own sampler,
so the server default did not apply.

### Decode

Tokens per second, DSpark drafts on. "Stream" is the median per-stream rate, "aggregate" is all streams together.
Median of 3.

| Workload | Sampler | C1 | C2 stream | C2 aggregate | C4 stream | C4 aggregate | C8 stream | C8 aggregate |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| code | T=1.0, top_p 1.0 | 90.5 | 58.6 | 111.6 | 33.5 | 129.7 | 21.7 | 167.5 |
| code | T=1.0, top_p 0.95 | 89.3 | 59.0 | 114.9 | 35.2 | 134.8 | 21.9 | 169.8 |
| prose | T=1.0, top_p 1.0 | 44.6 | 33.5 | 65.5 | 23.8 | 86.5 | 15.4 | 113.9 |
| prose | T=1.0, top_p 0.95 | 45.7 | 35.1 | 66.7 | 24.2 | 90.7 | 15.4 | 115.1 |
| thinking-75 | T=1.0, top_p 1.0 | 74.9 | 51.6 | 97.8 | 39.2 | 116.7 | 24.1 | 136.0 |
| thinking-75 | T=1.0, top_p 0.95 | 73.7 | 55.7 | 101.2 | 41.0 | 114.5 | 24.3 | 142.8 |

T=1.0 with top_p 1.0 is DeepSeek's card sampler; top_p 0.95 is the recipe's server default and the 2026-10-18
baseline's sampler. Median time to first token with the card sampler was 0.30 s (code), 0.25 s (prose) and 0.43 s
(thinking-75) at C1, and 0.58, 0.45 and 0.70 s at C8.

Method as before: 8 short prompts per workload, one untimed warmup request, a distinct seed per request, 3 repeats.
The decode cap is 512 tokens for every workload, thinking-75 included (reasoning tokens count); the 2026-10-19
rows ran with the same cap. A C1 repeat runs one prompt, so the three C1 repeats are three different prompts and
their spread comes mostly from the prompts: code under the card sampler gave 78.2, 90.5 and 99.1 tok/s.

Against 2026-10-19, card sampler, same prompts and caps: C1 code 75.3 to 90.5, prose 39.3 to 44.6, thinking-75
63.3 to 74.9; C8 aggregate code 144.2 to 167.5, prose 89.8 to 113.9, thinking-75 116.8 to 136.0.

### Prefill

| Prompt | C1 tok/s | C1 TTFT | C4 aggregate tok/s | C4 TTFT |
|--:|--:|--:|--:|--:|
| 4K | 1,262 | 3.1 s | 1,407 | 11.7 s |
| 32K | 2,424 | 13.5 s | 2,397 | 54.3 s |
| 128K | 2,391 | 54.8 s | 2,383 | 220.0 s |

Same method as before (fresh nonce per prompt, `max_tokens` 1, median of 3). The 4K C1 cell varies from boot to
boot and from prompt to prompt: the three 4K prompts in this run gave 1,211, 1,262 and 1,488 tok/s, and on
2026-10-19 two boots of one configuration gave medians of 1,205 and 1,425 tok/s. I do not read the change from
1,405 (2026-10-19) to 1,262 as a regression. At C1, 32K and 128K are within 0.2% of 2026-10-19; at C4, 32K went
from 2,214 to 2,397 tok/s and 4K from 1,401 to 1,407. MemAvailable stayed at or above 8.1 GiB
on both ranks from boot to the end of these runs.

### Gate

The full release gate ran on the same image and settings on 2026-10-21, with one difference: that boot used the
previous round table (`TF_DSV41_ROUND_MS=26.76,31.95,37.59,41.84,45.66,49.37,54.14,57.53`). I fitted the shipped
table from probes on that boot, and the decode and prefill tables above ran with it. Sampler T=1.0, top_p 1.0,
top_k 0 on every gate request. A second full gate ran later the same day on the exact recipe in this repo (the
shipped round table and the narrow seccomp profile): every lane passed again, exactness 32 of 32 at C1 and at C8,
lowest MemAvailable 7.5 GiB. All lanes passed on the first gate:

| Lane | Result |
|---|---|
| reasoning | thinking on and off both answer 391 to 17*23; reasoning only when on |
| tool_turn | answered from the tool result, no new call |
| tool_only_turn | reasoning present, 7,006,652 in the answer, earlier reasoning kept in the prompt |
| tc45 | calculator call in 3 of 3 tries (bar 1 of 3) |
| hardmode | agentic tool-use suite at T=1.0: score 91 (bar 88) |
| fidelity | 20 of 20 identifiers exact at each of six depths, prompts of 15,940 to 247,292 tokens (bar 20 of 20 to 64K, 19 of 20 above) |
| prefix | 16 of 16 codes exact, 4 concurrent 4-turn chats on a shared prefix, prompts of 9,596 to 9,672 tokens |
| long_session | 29,026 tokens decoded over 9 turns, no loop (bar 28K) |
| long_decode | reached the 32,768-token cap without a loop (the gate logs this as a warning) |
| corruption | 0 corrupted characters in two 1024-token essays and the other probes' text |
| exactness | token IDs identical to the 2026-10-19 recipe in 32 of 32 sequences at C1 and 32 of 32 at C8 |

The exactness lane records 8 prompts, each greedy and at T=1.0 with seeds 11, 22 and 33, 256 tokens each, and
compares them with the same recording from the 2026-10-19 recipe. That recipe drafts with the older depth policy
(cost policy, cap 3), so the new depth policy left the tokens unchanged on these 32 sequences. MemAvailable stayed
at or above 7.4 GiB on both ranks during the gate.

The fast prefill comparison against the exact kernels (top-1 agreement 0.9982, mean KL 7.8e-4) is from 2026-10-19
and was not re-run on this image.

### Greedy decode on the gate boot

Same boot as the gate (previous round table), T=0, same prompts and method as the decode table. Median of 3.

| Workload | C1 | C2 stream | C2 aggregate | C4 stream | C4 aggregate | C8 stream | C8 aggregate |
|---|--:|--:|--:|--:|--:|--:|--:|
| code | 94.5 | 62.7 | 123.9 | 36.6 | 142.8 | 22.5 | 174.0 |
| prose | 49.4 | 36.8 | 71.7 | 25.5 | 94.0 | 16.2 | 119.4 |
| thinking-75 | 84.3 | 58.1 | 102.6 | 39.6 | 125.4 | 26.1 | 152.1 |

On the same boot, a single-request greedy probe (up to 384 tokens, 3 repeats, identical text on every repeat) gave a
median of 86.4 (code), 47.7 (prose) and 101.8 tok/s (structured output) on one prompt set, and 92.2, 50.1 and 138.0
tok/s on a second set.

## 2026-10-19: fast prefill and the GPU sampler

What changed against the baseline, all in the recipe now:

- `TF_DSV41_PREFILL_KERNELS=fast` with a tuned grouped-GEMM table baked in the image (`TF_DSV41_GM_TUNE`), and
  2048-row prefill chunks (`TF_DSV41_PREFILL_CHUNK` and `TF_DSV41_PREFILL_ROWS`, 1536 before).
- `TF_DSV41_GPU_SAMPLE=1`: sampling runs on the GPU for every sampler, so DeepSeek's card sampler (T=1.0,
  top_p 1.0) works. On the baseline image it crashed both ranks.

Same weights, hardware, prompts and script as the baseline; the engine is a newer local build of the same source.
Booted with sparkrun.

### Decode

Tokens per second, T=1.0 top_p 1.0 (the card sampler), DSpark drafts on. Median of 3.

| Workload | C1 | C2 stream | C2 aggregate | C4 stream | C4 aggregate | C8 stream | C8 aggregate |
|---|--:|--:|--:|--:|--:|--:|--:|
| code | 75.3 | 47.8 | 92.0 | 29.3 | 113.9 | 18.8 | 144.2 |
| prose | 39.3 | 28.0 | 55.3 | 19.3 | 70.0 | 12.3 | 89.8 |
| thinking-75 | 63.3 | 43.2 | 81.9 | 34.1 | 101.5 | 21.5 | 116.8 |

Median time to first token at C1 was 0.27 s (code), 0.25 s (prose) and 0.36 s (thinking-75); at C8 0.87, 0.69
and 1.04 s.

### Prefill

| Prompt | C1 tok/s | C1 TTFT | C4 aggregate tok/s | C4 TTFT |
|--:|--:|--:|--:|--:|
| 4K | 1,405 | 2.8 s | 1,401 | 11.7 s |
| 32K | 2,421 | 13.5 s | 2,214 | 59.2 s |
| 128K | 2,394 | 54.7 s | 2,395 | 218.8 s |

Same method as the baseline (fresh nonce per prompt, `max_tokens` 1, median of 3). With 2048-row chunks, both
ranks kept 7.7-8.2 GiB MemAvailable under the 128K prefill.

### Exactness and gate

- Fast prefill against the exact kernels, logits over 8 prompts x 2048 teacher-forced positions (16,376 compared):
  top-1 agreement 0.9982 (bar 0.995), mean KL 7.8e-4 (bar 1e-3).
- Gate lanes passed on this build: prefix, long session, long decode, corruption.
- Not run yet on this build: hardmode and fidelity up to 248K. They decide the image release.

## 2026-10-18 baseline

### Setup

- Recipe: [recipes/deepseek-v4.1-flash-1017-2x-dgx-spark.yaml](recipes/deepseek-v4.1-flash-1017-2x-dgx-spark.yaml),
  booted with `sparkrun run`. The measured boot had the same engine and kernel settings. The recipe file here
  differs from it in host paths, the Engram mount point inside the container (`/engram`), the request log file
  name, the rendezvous address (passed with `-e DS41_MASTER` now, set in the recipe then) and the server default
  sampler (top_p 1.0 then, 0.95 now). Every request in the matrix sent its own sampler, so the server default did
  not apply. This exact file has passed a sparkrun dry run; I have not booted it yet.
- Engine image: my local build of the TensorFold-based engine. The published `ghcr.io/ursuciprian/ds41-tensorfold`
  image will be built from the same source once the gate passes.
- Weights: `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` @ `afc7aa0a`, Engram tables from
  `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`, packed per rank.
- Hardware: 2x DGX Spark (GB10), CX-7 back to back, both RoCE rails, kernel `6.17.0-1032-nvidia`, headless.
- Benchmark script: [gate/bench.py](gate/bench.py), OpenAI API only.

### Decode

Tokens per second. "Stream" is the median per-stream rate, "aggregate" is all streams together.

| Workload | Sampler | C1 | C4 stream | C4 aggregate | C8 stream | C8 aggregate |
|---|---|--:|--:|--:|--:|--:|
| code | T=0 | 74.4 | 31.8 | 120 | 18.7 | 144 |
| code | T=1.0, top_p 0.95 | 74.7 | 30.8 | 118 | 18.3 | 139 |
| prose | T=0 | 44.6 | 21.6 | 80 | 12.5 | 94 |
| prose | T=1.0, top_p 0.95 | 41.8 | 20.2 | 76 | 12.2 | 92 |
| thinking-75 | T=0 | 69.9 | 37.2 | 92 | 22.1 | 115 |
| thinking-75 | T=1.0, top_p 0.95 | 61.3 | 37.4 | 90 | 23.5 | 106 |

Method:

- 8 prompts per workload ([gate/data/bench_prompts.json](gate/data/bench_prompts.json)), short (a few hundred
  tokens). code and prose run with thinking off; thinking-75 runs with thinking on at effort 75, and its reasoning
  tokens count.
- Decode cap 512 tokens for code and prose, 1024 for thinking-75 (reasoning included).
- Per-stream rate = (completion_tokens - 1) / (time of last token - time of first token), so time to first token
  is excluded. Aggregate = tokens of the round / decode window of the round.
- Each request has its own seed. One untimed warmup request, then 3 repeats; the table shows the median.
- DSpark drafting is on in every cell. Its drafts are exact: replies match what the model decodes without them.
- The cells at T=1.0 with top_p 1.0 (DeepSeek's card sampler) are missing because that sampler crashes this image
  (see the README's Status section).

### Prefill

| Prompt | C1 tok/s |
|--:|--:|
| 4K | 614 |
| 32K | 906 |
| 128K | 917 |

Method: one request at a time, `max_tokens` 1, a fresh nonce at the start of each prompt so nothing comes from a
prefix cache, rate = prompt_tokens / time to first token, median of 3. Exact prefill kernels, as in the recipe.
Under the 128K prefill both ranks kept 8.4-8.7 GiB MemAvailable (MemFree 1.2-2.0 GiB).

### Quality smoke

All four checks passed:

| Check | Result |
|---|---|
| multi-turn | recalled an earlier turn and answered the arithmetic follow-up |
| tool call, then a tool-result turn | called the declared weather tool, then answered from its result |
| needle at 35,259 prompt tokens | retrieved the planted code exactly (39 s wall) |
| 2048-token decode, T=0 and T=1.0 | no repeated lines, distinct 8-grams 0.999 and 1.0, stopped at the length cap |

This smoke ran on my first build of the same engine with the same weights and settings, during bring-up. The
full gate ([gate/](gate/)) is the bar for publishing the image, and it has not been run green on this image yet.

### A vLLM EXL3 build on the same weights

During bring-up I also served the same EXL3 weights and Engram packs with a vLLM EXL3 build, to pick the engine.
Both engines passed the same quality smoke. These numbers come from different probes than the matrix above, so
each row compares the two engines under one method only.

| Probe | vLLM EXL3 build | TensorFold-based engine, same day |
|---|--:|--:|
| C1 decode, code, T=0, 512 tokens, median of 3 | 36.1 | 67.2 |
| C1 decode, prose, T=0, 512 tokens, median of 3 | 25.6 | 42.6 |
| C1 decode, thinking-75, one pass, no warmup | 45.6 | 79.0 |
| C1 prefill, prompts of about 520 tokens | 604-616 | 413-451 |

The vLLM build leads C1 prefill on short prompts. The TensorFold-based engine leads decode, which is why the
recipe uses it. The same-day TensorFold numbers differ from the matrix above because they came from other probes
and from the first bring-up build, a few hours earlier.
