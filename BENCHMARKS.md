# Benchmarks

Every number here comes from my two DGX Sparks. The newest results are first; the older sections follow
unchanged. Chart data for the README: [docs/data/bench.csv](docs/data/bench.csv).

## 2026-10-28: v1.2.0 release recipe

What changed against v1.1.0: the engine only, commit `21090db` (from `bb619cf`), image
`ghcr.io/ursuciprian/ds41-tensorfold:v1.2.0`. Drafter D1, the weights and the v1.1.0 recipe settings are the same;
the recipe adds `TF_DSV41_PREFIX_SHARE: '1024'`, the new default written out.

- Shared-prefix snapshots (`TF_DSV41_PREFIX_SHARE`, default 1024 when the tokenizer's role tokens are known, `0`
  turns it off). When a prompt's system part (the tokens before its first user, assistant or reminder token) is at
  least that many tokens and equals a stored entry's or a prompt in flight's, the engine also saves a snapshot at the
  end of that part, on the 16-token grid. Later prompts over the same system prompt and tools resume there. The
  snapshot at the replay point is kept as before. In my agentic set, 18 to 20 of the 24 prompts in a bucket share a
  system part of about 1.4K or 5.9K tokens with an earlier prompt.
- Admission reclaim (`TF_DSV41_ADMIT_RECLAIM`, default 1, [#11](../../issues/11)). Before a memory refusal (HTTP
  503), both ranks drop the grammar compiler caches, run garbage collection, return freed heap to the system and
  release the pinned host cache, then the floor is checked again. Refusal messages include both ranks' usable memory,
  the price and the rows it was taken at.

Shared servers: neither these snapshots nor the session store (every release) is keyed by client, so time to first
token can show one client that another sent the same long prefix. The recipe is meant for one user or one trusted
team. On a shared multi-tenant server set `TF_DSV41_SESSIONS=0`, which turns off the store and both kinds of reuse;
`TF_DSV41_PREFIX_SHARE=0` alone turns off only the system-part snapshots. The Engram row cache and the compiled
grammar cache stay shared either way. Per-tenant keys: [#13](../../issues/13).

Setup: alternating boots on 2026-10-28 and 2026-10-29, A = v1.1.0 (`bb619cf`), B = v1.2.0 (`21090db`), in the order A
B B A, model `checkpoint-d1` on both. B boots had no prefix-reuse setting, so they ran the default under test. Both
arms used the v1.1.0 release environment. Every arm ran the decode bench, prefill on real text and the agentic
buckets; the first B boot also ran exactness, the 27 greedy C1 replies, the full gate and two more fidelity runs at
16,072 and 241,157 tokens. Before the boots, the image's tests passed (118), and a mutant of the snapshot code (no
snapshot cone at the mark) failed its test as required.

### Agentic C8

My agentic prompt set: multi-turn coding-agent, tool and infrastructure sessions. Three buckets by prompt length, 24
prompts each, sent 8 at a time, each prompt once per boot, up to 1,024 new tokens, 75% greedy and 25% at the card
sampler (T=1.0, top_p 1.0). Pair 1 = boots A1 and B1, pair 2 = B2 and A2. Wall time from the first request to the
last reply. The buckets ran a8k, a16k, a32k in that order on every boot, after the decode bench and the prefill runs.

| Bucket | Prompt tokens | Pair | Wall A → B | Change | Median TTFT A → B | Prompt tok/s A → B | Cached tokens on B |
|---|--:|--:|--:|--:|--:|--:|--:|
| a8k | 7,958 to 8,579 | 1 | 208.8 → 178.2 s | -14.7% | 44.1 → 31.6 s | 944 → 1,107 | 49,808 |
| a8k | | 2 | 184.2 → 173.3 s | -5.9% | 36.1 → 25.6 s | 1,071 → 1,138 | 51,216 |
| a16k | 15,510 to 17,065 | 1 | 333.0 → 249.5 s | -25.1% | 75.1 → 50.1 s | 1,170 → 1,562 | 118,704 |
| a16k | | 2 | 328.7 → 248.6 s | -24.4% | 67.1 → 48.5 s | 1,185 → 1,567 | 118,704 |
| a32k | 28,820 to 33,108 | 1 | 593.2 → 519.2 s | -12.5% | 164.4 → 119.8 s | 1,231 → 1,406 | 100,848 |
| a32k | | 2 | 588.0 → 517.4 s | -12.0% | 139.8 → 106.8 s | 1,242 → 1,411 | 100,848 |

Replies text-identical between A and B on all 24 requests of every bucket in both pairs (144 of 144). Cached tokens
on A: 0 in every bucket. Admission waits 65 / 60 on the A boots and 57 / 59 on the B boots, 0 refusals, 0 reclaims.
One run per bucket per boot, so the spread between the pairs (a8k -14.7% and -5.9%) is the only noise estimate I have
here.

Bucket order matters for B. The first 8 requests of a8k run together before any snapshot of their system part exists,
and a8k resumed from a snapshot on 10 and 11 of its 24 requests (B1, B2). a16k and a32k came after it on the same
boot, with a8k's system-part snapshots stored, and resumed on 24 of 24, the first requests included. So the a16k and
a32k gains include snapshots from an earlier bucket; on a fresh boot those buckets would start like a8k and gain
less. Median TTFT over 24 requests also moves more between runs than wall time does.

### Decode

Aggregate tok/s, 512 tokens (reasoning included for thinking-75), 8 short prompts per workload, median of 2 runs per
boot, pooled per arm. Serve = card sampler (T=1.0, top_p 1.0), greedy = T=0. Noise per cell is the larger of the
A-to-A and B-to-B spread, and at least 1%.

| Workload | Sampler | C | v1.1.0 | v1.2.0 | Change | Noise |
|---|---|--:|--:|--:|--:|--:|
| code | serve | 1 | 95.51 | 95.51 | +0.00% | 1.00% |
| code | serve | 4 | 136.52 | 137.01 | +0.36% | 1.00% |
| code | serve | 8 | 169.20 | 170.41 | +0.71% | 1.52% |
| code | greedy | 1 | 92.95 | 93.01 | +0.06% | 2.53% |
| prose | serve | 1 | 50.43 | 50.68 | +0.50% | 1.00% |
| prose | serve | 4 | 98.00 | 98.41 | +0.42% | 1.95% |
| prose | serve | 8 | 124.36 | 126.81 | +1.97% | 2.74% |
| prose | greedy | 1 | 53.95 | 54.12 | +0.30% | 1.00% |
| thinking-75 | serve | 1 | 75.38 | 76.47 | +1.45% | 1.82% |
| thinking-75 | serve | 4 | 118.93 | 120.04 | +0.93% | 1.56% |
| thinking-75 | serve | 8 | 140.19 | 141.87 | +1.19% | 1.30% |
| thinking-75 | greedy | 1 | 80.03 | 80.26 | +0.29% | 1.00% |

Every cell is within noise. The bench prompts are short and have no shared system part, so the snapshot code does not
run on them.

### Prefill on real text and short-prompt TTFT

The first A/B ran 3 prompts per cell per boot. Each B boot against both A boots:

| Cell | A1 / A2 | B1 | B2 | Noise |
|---|--:|--:|--:|--:|
| 4K prefill, tok/s | 1,115.6 / 1,146.7 | 1,210.9 (+7.05%) | 1,089.3 (-3.70%) | 2.75% |
| 32K prefill, tok/s | 1,617.4 / 2,051.5 | 2,050.1 | 1,673.1 | 23.66% |
| 128K prefill, tok/s | 1,808.8 / 2,148.2 | 1,858.6 | 1,861.0 | 17.15% |
| TTFT chat, s | 0.307 / 0.307 | 0.306 | 0.309 | 1.00% |
| TTFT completions, s | 0.199 / 0.196 | 0.197 | 0.197 | 1.52% |

The 4K cell was better beyond noise on B1 and worse beyond noise on B2. Neither change runs code on this path (no
shared system part, no refusal), and per-request 4K values ranged from 950 to 1,120 tok/s inside one A boot and from
984 to 1,196 inside one B boot, so 3 samples a boot could not settle it. I did not count this run either way. I wrote
a rule for a retest before its first boot: four fresh boots (A B B A), 10 samples per cell per boot; keep v1.2.0 only
if no cell is worse beyond noise when pooled (B1 + B2 against A1 + A2, 20 samples each, noise = the A1-to-A2 spread,
at least 1%) and 4K is not worse beyond noise on either B boot alone. The retest replaces the first run's prefill
verdict, whatever it showed:

| Cell | A, pooled (A1 / A2) | B, pooled (B1 / B2) | Change | Noise | Verdict |
|---|--:|--:|--:|--:|---|
| 4K prefill, tok/s | 1,151.5 (1,261.7 / 1,145.8) | 1,212.7 (1,149.6 / 1,252.4) | +5.32% | 9.63% | noise |
| 32K prefill, tok/s | 1,720.0 (1,725.7 / 1,714.9) | 1,731.8 (1,730.9 / 1,890.1) | +0.68% | 1.00% | noise |
| 128K prefill, tok/s | 1,842.9 (1,836.5 / 1,850.5) | 1,849.3 (1,855.4 / 1,839.5) | +0.35% | 1.00% | noise |
| TTFT chat, s | 0.307 (0.308 / 0.304) | 0.309 (0.317 / 0.308) | +0.65% | 1.14% | noise |
| TTFT completions, s | 0.199 (0.199 / 0.199) | 0.195 (0.194 / 0.195) | -2.26% | 1.00% | better |

4K on each B boot alone: B1 -4.49%, B2 +4.05% against the A boots, noise 9.63%, both within noise. The retest met the
rule. Two single-boot cells sit outside the rule's criteria, and I list them here: chat TTFT on B1 was 3.51% higher
(0.317 s, noise 1.14%), and 32K on B2 was 9.87% higher, where 5 of B2's 10 samples ran at about 2,055 tok/s. The 32K
and 128K cells switch between two levels from boot to boot on both images; the logs tie this to the free memory left
for 2,048-row prefill rounds (not tested directly).

### Gate

Full release gate on the first v1.2.0 boot, card sampler (T=1.0, top_p 1.0, top_k 0) on every request. All lanes
passed: reasoning, tool_turn, tool_only_turn, tc45 (3 of 3), hardmode 91 (bar 88), fidelity 20 of 20 at each of six
depths (15,940 to 247,292 tokens), prefix 16 of 16, long_session, long_decode, corruption 0. Exactness: token IDs
identical to the 2026-10-19 recipe in 32 of 32 sequences at C1 and 32 of 32 at C8. 27 greedy C1 replies
text-identical to the v1.0.1 image. Two more fidelity runs on the same boot after the gate, at 16,072 and 241,157
tokens: 20 of 20 at both depths in both runs. No admission refusal and no reclaim on that boot. Lowest MemAvailable
on rank 0 5.9 GiB to the end of the gate and 5.8 GiB during the two extra runs; rank 1 7.3 GiB.

The reclaim on a diagnostic boot (v1.2.0's reclaim on v1.1.0, the hard floor raised from 4 to 5 GiB so that a cold
248K prompt could not fit after the hardmode suite): 54 reclaims, 51 refusals. Rank 0 freed 0.00 / 0.10 / 0.24 GiB
(min / median / max) per reclaim, rank 1 0.00 / 0.00 / 0.11. The reclaim makes a refusal less likely; the limit in
[#11](../../issues/11) stays.

## 2026-10-25: v1.1.0 release recipe

What changed against v1.0.1, all in the recipe now:

- Engine commit `bb619cf` (from `ee6ff92`), image `ghcr.io/ursuciprian/ds41-tensorfold:v1.1.0`. Every change below
  gives the same tokens: exactness 32 of 32 at C1 and C8 against the 2026-10-19 recipe on each candidate image.
  - Less work outside verify in each round: the draft sampling key is drawn earlier, a few small draft kernels are
    merged, and the Engram warm-up runs later.
  - `TF_DSV41_FIRST_SERIAL=1`: a request's first round verifies its pending token alone, which lowers the time to
    first token on short prompts by 10 to 25 ms.
  - `TF_DSV41_PF_MOE_LEAN=1` plans the prefill MoE without host syncs, and `TF_DSV41_PF_SHARED_SIDE=64` runs the
    shared expert on a side stream in prefill runs of up to 64 rows.
  - `TF_DSV41_PF_ATTN_STREAMS=128` runs the attention projections on parallel streams in prefill runs of up to 128
    rows.
  - `TF_DSV41_INGEST_BATCH=1` lets the drafter take in a round's committed tokens in one batch, and
    `TF_DSV41_DRAFT_GRAPH_SLOTS=8` captures draft graphs for 5 to 8 concurrent requests (with the default 4,
    those draft passes ran without a graph). In an alternating A/B on this image (four boots), the two together
    gave code C8 +3.1%, prose C4 +1.8% and C8 +3.6%, thinking-75 C8 +1.7%, and 3.9% less time to first token on a
    short chat prompt; the other cells were within noise.
- Drafter D1 ([#6](../../issues/6)), below.

Screened on this image and not kept, one boot each against the base boots before and after it: the drafter head
trim (`TF_DSV41_DRAFT_HEAD=trim:32768`: thinking-75 C1 +2.2% but C4 -1.1%), 64 graph rows with all graphs captured
at boot (code lower at C1, C4 and C8), a round table measured again on this image (no cell changed beyond noise),
temperature calibration of the drafter's confidence (six cells lower, code C8 -3.5%), and a small-row dense prefill
kernel (short-prompt time to first token -2%, but code C4 and 4K prefill lower).

Setup: the same weights, hardware, prompts and scripts as before. Booted with sparkrun under the narrow seccomp
profile. The booted recipe differs from the file in [recipes/](recipes/) in host paths, the Engram mount point, the
request log name, the rendezvous address, an extra `SYS_PTRACE` capability, the server default sampler (top_p 1.0
on the boot, 0.95 in the file), and in how the six new knobs went in (`-e` flags at boot). I compared the running
container's environment with the file: the engine settings match. Every benchmark and gate request sends its own
sampler.

### Drafter D1

D1 is the checkpoint's DSpark drafter, refit on the served model's own replies:

- Data: 2.0M tokens of replies from this engine on my pair (prose and long documents, code and structured output,
  chat, tool calls and agentic turns, and some reasoning), 75% greedy and 25% at the card sampler. The benchmark and
  gate prompts were kept out of the training prompts by 13-gram match, and 248 sessions were held out.
- Training: low-rank updates to the drafter's dense matrices (37 matrices) and its norms and heads, with the routed
  experts frozen. The loss is a cross-entropy over the candidates the engine can pick a draft from, on the first
  three draft positions, counted only where the earlier positions were accepted.
- Merge: the update at half strength, merged into the drafter's 6-bit EXL3 layers. Merged, the draft pass costs
  the same as the original drafter's (3.5 to 4.0 ms for both across the four boots below). Applied at serve time
  without the merge, the update made the draft pass 4.5 ms and lost 6% on code at C1. Merged at full strength it
  lost 1.0% and 2.4% on code at C1 in two A/Bs, so I halved it.
- Output: one overlay file of 165 tensors (477 MB) and an index that points the `mtp.*` tensors at it. The main
  model's tensors are untouched. [docs/setup.md](docs/setup.md#drafter-d1) has the install step.

Alternating A/B on the v1.1.0 image, A = the checkpoint's own drafter, B = D1, boots in the order A B B A. Aggregate
tok/s, 512 tokens, median of 2 runs per boot, pooled per arm. Serve = card sampler (T=1.0, top_p 1.0), greedy =
T=0. The noise floor per cell is the larger of the A-to-A and B-to-B spread, and at least 1%.

| Workload | Sampler | C | Checkpoint drafter | D1 | Change | Noise |
|---|---|--:|--:|--:|--:|--:|
| code | serve | 1 | 93.87 | 94.88 | +1.1% | 1.0% |
| code | serve | 4 | 133.60 | 135.31 | +1.3% | 1.0% |
| code | serve | 8 | 168.43 | 169.04 | +0.4% | 1.0% |
| code | greedy | 1 | 91.97 | 92.08 | +0.1% | 1.0% |
| prose | serve | 1 | 44.77 | 49.98 | +11.6% | 1.0% |
| prose | serve | 4 | 88.66 | 96.68 | +9.1% | 2.2% |
| prose | serve | 8 | 117.61 | 125.39 | +6.6% | 4.4% |
| prose | greedy | 1 | 48.03 | 53.52 | +11.5% | 1.0% |
| thinking-75 | serve | 1 | 73.41 | 75.23 | +2.5% | 2.1% |
| thinking-75 | serve | 4 | 117.32 | 119.50 | +1.9% | 1.0% |
| thinking-75 | serve | 8 | 134.92 | 139.80 | +3.6% | 1.0% |
| thinking-75 | greedy | 1 | 80.12 | 79.32 | -1.0% | 1.0% |

Thinking-75 at greedy C1 is at the noise floor here, and it was also lower in the two earlier A/Bs of D1 on the
previous candidate image (-1.2% and -0.8%). I count it as a possible loss of about 1% on that cell. Prefill on
real text and the short-prompt time to first token did not change between the arms.

### Against v1.0.1

The v1.0.1 tables ran three runs per cell; these ran two. At C1 each run is one prompt, so I compare on the runs
both versions share: the same first two prompts and the same seeds. v1.0.1: the two 32-row boots of the graph-rows A/B
below (2026-10-22), runs 1 and 2 of 3. v1.1.0: the two D1 boots above (2026-10-25). Aggregate tok/s, card sampler,
per boot, then the median over the four runs. The versions ran on different days, not alternated.

| Workload | C | v1.0.1, boot 1 / 2 | v1.1.0, boot 1 / 2 | v1.0.1 | v1.1.0 | Change |
|---|--:|--:|--:|--:|--:|--:|
| code | 1 | 92.7 / 93.6 | 94.9 / 94.8 | 93.4 | 94.9 | +1.5% |
| code | 4 | 128.6 / 126.1 | 135.7 / 135.2 | 126.8 | 135.3 | +6.8% |
| code | 8 | 163.6 / 164.4 | 169.1 / 169.0 | 163.8 | 169.0 | +3.2% |
| prose | 1 | 43.8 / 43.9 | 50.0 / 50.0 | 43.9 | 50.0 | +13.9% |
| prose | 4 | 85.5 / 85.1 | 96.7 / 96.7 | 85.5 | 96.7 | +13.0% |
| prose | 8 | 113.3 / 112.8 | 125.6 / 125.1 | 113.3 | 125.4 | +10.7% |
| thinking-75 | 1 | 73.1 / 72.9 | 75.2 / 75.3 | 72.8 | 75.2 | +3.3% |
| thinking-75 | 4 | 114.7 / 114.1 | 119.7 / 118.8 | 114.1 | 119.5 | +4.7% |
| thinking-75 | 8 | 133.1 / 130.9 | 139.9 / 139.8 | 132.3 | 139.8 | +5.7% |

Code at C4 varied from 127 to 136 tok/s across the four v1.0.1 boots of the graph-rows A/B, so part of its +6.8%
can be boot noise. The per-stream median at C8 was 21.8 tok/s on code, 16.6 on prose and 23.9 on thinking-75.
Median time to first token at C1 went from 0.29 to 0.24 s (code), 0.27 to 0.22 s (prose) and 0.47 to 0.41 s
(thinking-75). At C4 thinking-75 went the other way, 0.54 to 0.66 s; code (0.58 to 0.54 s) and prose (0.44 to
0.37 s) went down.

Three runs per cell, the same method as the v1.0.1 tables: one boot of the exact release recipe (2026-10-25, every
served env key checked against the recipe file), the same prompts and seeds as the v1.0.1 runs on all 117 requests.
v1.0.1: mean of the two 32-row boots, median of 3 runs each. Aggregate tok/s, card sampler. One boot carries about
2% boot noise per cell.

| Workload | C | v1.0.1 | v1.1.0 | Change | v1.1.0 runs |
|---|--:|--:|--:|--:|--:|
| code | 1 | 88.7 | 92.8 | +4.6% | 92.8 / 95.8 / 81.6 |
| code | 4 | 127.9 | 136.0 | +6.3% | 136.0 / 135.1 / 136.3 |
| code | 8 | 163.8 | 168.7 | +3.0% | 166.8 / 169.5 / 168.7 |
| prose | 1 | 44.2 | 49.4 | +11.6% | 50.7 / 49.4 / 47.9 |
| prose | 4 | 85.1 | 96.9 | +13.8% | 97.6 / 96.9 / 95.0 |
| prose | 8 | 111.5 | 124.6 | +11.7% | 124.6 / 127.2 / 122.6 |
| thinking-75 | 1 | 74.7 | 75.4 | +1.0% | 75.4 / 74.9 / 83.5 |
| thinking-75 | 4 | 115.9 | 119.6 | +3.2% | 117.2 / 119.6 / 127.5 |
| thinking-75 | 8 | 133.6 | 141.6 | +5.9% | 141.6 / 136.7 / 147.6 |

The third code prompt at C1 is the slow one (81.6 tok/s here, 77.3 on v1.0.1), which is why the median of three
(92.8) sits below the two-run figure above (94.9).

### Prefill on real text

Same prompts and script as the v1.0.1 real-text table below (`max_tokens` 1, fresh nonce, one request at a time,
three prompts per size). Median per boot, then over all six prompts:

| Prompt tokens | Boot 1 / 2, tok/s | Six prompts, tok/s | TTFT | v1.0.1, tok/s |
|--:|--:|--:|--:|--:|
| about 4,120 | 1,214 / 1,114 | 1,122 | 3.7 s | 1,131 |
| about 32,790 | 1,713 / 1,690 | 1,701 | 19.3 s | 1,665 |
| about 131,090 | 1,842 / 1,842 | 1,842 | 71.2 s | 1,831 |

4K is noisy: its six prompts gave 1,045 to 1,256 tok/s. 32K and 128K are 2.2% and 0.6% above v1.0.1, one boot
there against two here. Time to first token on a short prompt (one sentence, `max_tokens` 1, median of 10 over two
boots): 0.316 s on `/v1/chat/completions` (0.338 on v1.0.1) and 0.201 s on `/v1/completions` (0.225).

### Gate

Full release gate on the first D1 boot, card sampler (T=1.0, top_p 1.0, top_k 0) on every request. All lanes passed:

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

Also on that boot: 27 greedy C1 replies (9 categories, 3 each) text-identical to the same set on the v1.0.1 image,
and two more fidelity runs at 241,157 tokens, both 20 of 20. Lowest MemAvailable 6.0 GiB, during the gate.

The fidelity probe at 241K to 247K tokens has an intermittent miss. Across the 17 runs at that depth since
2026-10-24 (gates and retests on this release's candidates), 2 had items with no answer: 16 of 20 on the first gate
of the v1.1.0 engine (checkpoint drafter, without the two knobs above) and 17 of 20 on a retest with the two knobs
(checkpoint drafter). No run had a wrong answer. The cause is admission, not the model: in each of the two runs the
server log has exactly as many `admission refused` lines as there were items without an answer. A cold prefill of
241K to 247K tokens needs about 0.55 GiB above the 4 GiB memory floor, and rank 0 had 4.48 to 4.52 GiB usable after
heavier probes on the same boot, so the request got HTTP 503 and the probe counted it as no answer. A refused request
does not fill the prefix cache, so the next one is cold too, which is why the misses came 3 or 4 in a row.
On a fresh boot of the v1.1.0 recipe with D1, 10 runs of the 8K and 248K rungs (400 requests, 50 of them cold) were
all exact, with 0 refusals and identical replies across all 10 runs. The floor and the gate are unchanged; the next
step is to find what holds the extra memory after the heavy probes.

## Earlier releases at a glance

| | 2026-10-18 baseline | 2026-10-19 | 2026-10-21, v1.0.0 |
|---|--:|--:|--:|
| Decode C1, code | 74.7 tok/s | 75.3 tok/s | 90.5 tok/s |
| Decode C1, prose | 41.8 tok/s | 39.3 tok/s | 44.6 tok/s |
| Decode C1, thinking-75 | 61.3 tok/s | 63.3 tok/s | 74.9 tok/s |
| Decode aggregate C8, code | 139 tok/s | 144 tok/s | 167 tok/s |
| Decode aggregate C8, thinking-75 | 106 tok/s | 117 tok/s | 136 tok/s |
| Prefill C1, 4K / 32K / 128K, repeated text | 614 / 906 / 917 tok/s | 1,405 / 2,421 / 2,394 tok/s | 1,262 / 2,424 / 2,391 tok/s |
| Prefill C1, 4K / 32K / 128K, real text | | | 1,131 / 1,665 / 1,831 tok/s (v1.0.1, 2026-10-22) |
| Sampler in the decode rows | T=1.0, top_p 0.95 | T=1.0, top_p 1.0 (card) | T=1.0, top_p 1.0 (card) |
| DeepSeek's card sampler | crashes both ranks | works (GPU sampler) | works (GPU sampler) |
| Release gate | quality smoke only | 4 lanes pass | full gate, all lanes pass |

Median of 3 runs per cell. The baseline's decode rows use top_p 0.95, because the card sampler crashed that image.
At top_p 0.95, 2026-10-21 gives 89.3 tok/s at C1 and 169.8 tok/s aggregate at C8 on code. Prefill at 4K C1 varies
from boot to boot: on 2026-10-19 two boots of one configuration gave 1,205 and 1,425 tok/s, and the three 4K prompts
of the 2026-10-21 run gave 1,211, 1,262 and 1,488 tok/s.

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
