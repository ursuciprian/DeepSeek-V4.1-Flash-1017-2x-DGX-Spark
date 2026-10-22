<h1 align="center">DeepSeek-V4.1-Flash-1017 on two DGX Sparks</h1>

<p align="center">A sparkrun recipe for DeepSeek-V4.1-Flash-1017 (EXL3 2.9 bpw, Engram from local NVMe) at TP=2 over CX-7.<br>TensorFold-based engine, exact DSpark drafts, OpenAI-compatible endpoint.</p>

<table align="center">
  <tr>
    <td align="center"><h2>89 tok/s</h2><b>decode C1</b><br><sub>code, T=1.0, top_p 1.0</sub></td>
    <td align="center"><h2>164 tok/s</h2><b>decode aggregate C8</b><br><sub>code, T=1.0, top_p 1.0</sub></td>
    <td align="center"><h2>2,424 tok/s</h2><b>prefill C1, 32K</b><br><sub>fast prefill kernels</sub></td>
    <td align="center"><h2>13.5 s</h2><b>TTFT, 32K prompt</b><br><sub>C1, cold prefix</sub></td>
  </tr>
</table>

<p align="center"><sub>262K context, 8 sequences in parallel, FP8 KV, DSpark drafts, median of 3 runs on my pair, v1.0.1 release recipe (decode 2026-10-22, prefill 2026-10-21)</sub></p>

<details>
<summary><sub>Measurement details</sub></summary>

| Metric | Value | Workload | Sampling | Date | Statistic |
|---|--:|---|---|---|---|
| Decode C1, code | 88.0 / 89.4 tok/s | 3 of 8 short code prompts, one per repeat, 512 tokens, thinking off | T=1.0, top_p 1.0 | 2026-10-22 | median of 3, two boots |
| Decode aggregate C8, code | 163.1 / 164.4 tok/s | same prompts, 8 at a time | T=1.0, top_p 1.0 | 2026-10-22 | median of 3, two boots |
| Prefill C1, 32K | 2,424 tok/s | one prompt, fresh nonce, `max_tokens` 1 | n/a | 2026-10-21 | median of 3 |
| TTFT, 32K prompt | 13.5 s | same | n/a | 2026-10-21 | median of 3 |

Image `ds41-tf:cand4` (engine commit `ee6ff92`), published as `ghcr.io/ursuciprian/ds41-tensorfold:v1.0.0`, with the settings in
[recipes/](recipes/). A C1 repeat runs one prompt, so the three C1 code repeats are three prompts: 78.2, 90.5 and
99.1 tok/s.

Method, prompts and the full matrix: [BENCHMARKS.md](BENCHMARKS.md). Chart data: [docs/data/bench.csv](docs/data/bench.csv).

</details>

This is a work in progress. I am working on speed (the drafter, scheduling under concurrency, prefill at 4K), and
the numbers above are what the recipe in this repo does today. Nothing here counts a speedup I have not measured
end to end.

## Quick start

Requirements:

- 2x DGX Spark with the ConnectX-7 ports cabled back to back, both RoCE rails up.
- [sparkrun](https://sparkrun.dev) 0.3.8 or newer, with both nodes reachable over SSH.
- About 315 GiB free on each node's NVMe, and about 285 GiB more on the node that packs the Engram tables
  while it packs (see [Hardware and storage](#hardware-and-storage)).
- The checkpoint and the packed Engram shards in place under `/srv/ds41` on both nodes:
  [docs/setup.md](docs/setup.md) walks through the download, the packing and the copy to the worker.

```sh
sparkrun registry add https://github.com/ursuciprian/DeepSeek-V4.1-Flash-1017-2x-DGX-Spark
sparkrun run deepseek-v4.1-flash-1017-2x-dgx-spark --hosts <head>,<worker> \
  -e DS41_MASTER=<head CX-7 IPv4> --no-follow
```

`DS41_MASTER` is the head's address on the CX-7 link. Without it the ranks meet on the head's default-route
address, which on a Spark is often Wi-Fi. sparkrun pulls the image
`ghcr.io/ursuciprian/ds41-tensorfold:v1.0.0` (about 11 GB to download, 25 GB unpacked) on the first
run.

`sparkrun run` returns before the engine is ready. A boot takes about 5 minutes (283-300 s on three boots on my
pair; the first boot of a new image also compiles kernels and measures a cost table). Wait for health, then
send a request:

```sh
H=http://<head>:8888
until [ "$(curl -s -o /dev/null -w '%{http_code}' $H/health)" = 200 ]; do sleep 10; done
curl -s $H/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash-1017", "max_tokens": 2048,
  "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}]}'
```

Thinking is on by default at effort `high` (75 on the engine's 1-100 scale, the thinking-75 rows in the
benchmarks). Turn it off per request with `"chat_template_kwargs": {"thinking": false}`, or set
`"reasoning_effort"` to `low`, `medium`, `high`, `max` or a number from 1 to 100.
`max_tokens` covers the reasoning and the answer together.

The server binds 0.0.0.0 with no API key. Keep it on a trusted network or put an authenticating proxy in front.
Stop it with `sparkrun stop --all`.

<details>
<summary>Paths, interfaces and other settings in the recipe</summary>

- Host paths. sparkrun fills `{model}` in the serve command from the recipe's `model:` field and mounts that
  folder at the same path. It does not fill placeholders in `executor_config.volumes`, so the recipe uses fixed
  paths: `/srv/ds41/checkpoint`, `/srv/ds41/engram-shards`, `/srv/ds41/tf-state` and the seccomp profile
  `/srv/ds41/seccomp-io-uring.json`. Put the files there, make `/srv/ds41` a symlink to where they are, or edit
  those lines in your copy of the recipe.
- Interfaces. `NCCL_SOCKET_IFNAME=enp1s0f1np1` and `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1` are the names a DGX
  Spark uses by default. If `ip -br link` and `ibv_devices` show other names on your nodes, change them, or delete
  the two lines and let sparkrun detect them. The RoCE v2 GID index is read from sysfs on each rank at start,
  because it can change between boots.
- Sampler. Requests that send no sampler get T 1.0, top_p 0.95, top_k 0. Change these server defaults with
  `-o temperature=...`, `-o top_p=...` and `-o top_k=...`. DeepSeek's card sampler (T 1.0, top_p 1.0) works with
  the GPU sampler the recipe turns on (`TF_DSV41_GPU_SAMPLE=1`); the benchmark rows below use it.
- Context 262,144 tokens, 8 sequences in parallel (aggregate decode still rises from C4 to C8 on my pair), FP8 KV
  cache, a shared KV pool of 1,048,576 tokens: 4 sessions at full context, or 8 whose prompt plus `max_tokens`
  stays under about 131,000 tokens each ([#8](../../issues/8)).
- seccomp. The Engram gate runs in a C worker that reads rows with io_uring (`TF_DSV41_ENGRAM_CGATE=1`). Docker's
  default seccomp profile refuses io_uring, so the recipe points the container at
  [recipes/seccomp-io-uring.json](recipes/seccomp-io-uring.json): Docker's default profile plus `io_uring_setup`,
  `io_uring_enter` and `io_uring_register`, nothing else. Copy it to `/srv/ds41/seccomp-io-uring.json` on both
  nodes. Without the `security_opt` line the worker is meant to fall back to a pool of pread threads; I have not
  booted that configuration.
- Depth policy. `TF_DSV41_ROUND_MS` is a table of round times measured on my pair with this image. If your pair
  differs, the line can be removed and the engine prices rounds from the cost table it calibrates at boot. I have
  not measured that combination.
- Metrics. `GET /metrics` on port 8888 returns Prometheus text: requests, errors, prompt, completion and cached
  tokens, decode rounds, drafted and accepted draft tokens, decode and prefill seconds.

</details>

## At a glance

| | 2026-10-18 baseline | 2026-10-19 | 2026-10-21, v1.0.0 |
|---|--:|--:|--:|
| Decode C1, code | 74.7 tok/s | 75.3 tok/s | 90.5 tok/s |
| Decode C1, prose | 41.8 tok/s | 39.3 tok/s | 44.6 tok/s |
| Decode C1, thinking-75 | 61.3 tok/s | 63.3 tok/s | 74.9 tok/s |
| Decode aggregate C8, code | 139 tok/s | 144 tok/s | 167 tok/s |
| Decode aggregate C8, thinking-75 | 106 tok/s | 117 tok/s | 136 tok/s |
| Prefill C1, 4K / 32K / 128K | 614 / 906 / 917 tok/s | 1,405 / 2,421 / 2,394 tok/s | 1,262 / 2,424 / 2,391 tok/s |
| Sampler in the decode rows | T=1.0, top_p 0.95 | T=1.0, top_p 1.0 (card) | T=1.0, top_p 1.0 (card) |
| DeepSeek's card sampler | crashes both ranks | works (GPU sampler) | works (GPU sampler) |
| Release gate | quality smoke only | 4 lanes pass | full gate, all lanes pass |

The baseline's decode rows use top_p 0.95, because the card sampler crashed that image. At top_p 0.95,
2026-10-21 gives 89.3 tok/s at C1 and 169.8 tok/s aggregate at C8 on code ([BENCHMARKS.md](BENCHMARKS.md)).
Prefill at 4K C1 varies from boot to boot: on 2026-10-19 two boots of one configuration gave 1,205 and 1,425 tok/s,
and the three 4K prompts of the 2026-10-21 run gave 1,211, 1,262 and 1,488 tok/s. I do not read 1,405 to 1,262 as
a regression.

## Performance

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/img/decode-dark.svg"><img src="docs/img/decode-light.svg" alt="Aggregate decode tok/s vs concurrent requests 1, 2, 4, 8 for code, prose and thinking-75, T=1.0 top_p 1.0, 2026-10-21. End values labelled." width="420"></picture>
<picture><source media="(prefers-color-scheme: dark)" srcset="docs/img/prefill-dark.svg"><img src="docs/img/prefill-light.svg" alt="Prefill tok/s vs prompt length 4K, 32K, 128K: exact kernels C1 (2026-10-18), fast kernels C1 and C4 (2026-10-21). End values labelled." width="420"></picture>

<sub>Left: aggregate decode tok/s by concurrency, v1.0.0 recipe on 2026-10-21, T=1.0 top_p 1.0, 8 short prompts per
workload, 512 tokens (reasoning included for thinking-75), median of 3. Right: prefill tok/s by prompt length, one
request (C1) or four at once (C4, aggregate), fresh nonce per prompt so nothing comes from a prefix cache, median of 3.
Exact kernels are the 2026-10-18 baseline; fast kernels are the current recipe.</sub>

Decode goes from 90.5 tok/s at C1 to 167 tok/s at C8 on code. Prose drafts accept fewer tokens per round, so it
stays lower at every concurrency. The fast prefill kernels are 2.1x to 2.7x faster than the exact ones and stay
near 2,400 tok/s out to 128K. MemAvailable stayed at or above 8.1 GiB on both ranks from boot to the end of these runs.

## Quality gate

The full release gate passed on the v1.0.0 image on 2026-10-21, at DeepSeek's card sampler (T=1.0, top_p 1.0).

| Check | Result |
|---|---|
| Reasoning on and off, tool turn, tool-result-only turn, forced tool call | pass (forced call 3 of 3) |
| Hardmode agentic tool-use suite, T=1.0 | 91 (bar 88) |
| Identifier fidelity, six depths, prompts of 15,940 to 247,292 tokens | 20 of 20 exact at every depth |
| Shared prefix, 4 concurrent chats | 16 of 16 exact |
| Long session | 29,026 tokens over 9 turns, no loop |
| Long decode, 32,768-token cap | no loop |
| Corruption | 0 corrupted characters |
| Exactness against the 2026-10-19 recipe, 32 sequences at C1 and at C8 | token-identical, 32 of 32 each |
| Fast prefill vs exact kernels, logits over 16,376 teacher-forced positions (2026-10-19) | top-1 agreement 0.9982 (bar 0.995), mean KL 7.8e-4 (bar 1e-3) |

The first gate boot used the previous round table (`TF_DSV41_ROUND_MS`). A second full gate ran on the exact
recipe in this repo (the shipped round table and the narrow seccomp profile) and passed every lane again, with
exactness 32 of 32 at C1 and at C8. Details per lane: [BENCHMARKS.md](BENCHMARKS.md#gate). The gate is in
[gate/](gate/) and talks to the server over the OpenAI API only.

## History

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/img/history-dark.svg"><img src="docs/img/history-light.svg" alt="Decode C1 code, decode aggregate C8 code and prefill C1 32K on 2026-10-18 (baseline), 2026-10-19 and 2026-10-21 (v1.0.0). Values labelled." width="640"></picture>

<sub>2026-10-18: baseline with exact prefill kernels, decode at T=1.0 top_p 0.95. 2026-10-19: fast prefill kernels
(tuned grouped GEMM, 2048-row chunks) and the GPU sampler, decode at T=1.0 top_p 1.0. 2026-10-21, v1.0.0: ratio
depth policy with a measured round table, cheaper verify rows, decode at T=1.0 top_p 1.0. Median of 3 each.</sub>

## Hardware and storage

- 2x DGX Spark: GB10, 128 GB unified memory each. The engine allocates about 100.6 GiB per rank.
- ConnectX-7 back to back, both RoCE functions in use (`rocep1s0f1`, `roceP2p1s0f1`).
- Kernel `6.17.0-1032-nvidia` on both nodes. In earlier work on this pair, `7.0.0-1019-nvidia` broke NCCL memory
  registration once about 85 GB per node was GPU-resident.
- Local NVMe on each node, because the engine reads Engram rows from disk with O_DIRECT during decode:

| What | Size | Where |
|---|--:|---|
| EXL3 2.9 bpw checkpoint | 197 GiB | both nodes |
| Packed Engram shards (2 files per rank) | 94.4 GiB | both nodes, each its own rank's files |
| Engram source shards 47 and 48 | 190 GiB | the node that packs; delete after packing |
| Rank 1's packed shards, staged for the copy | 94.4 GiB | the node that packs; delete after the copy |
| Engine image | 23 GiB | both nodes |

- Free memory is tight on unified memory. Run the nodes headless and keep at least 5 GiB MemAvailable;
  [docs/setup.md](docs/setup.md#memory) has the details.

## Status

- Recipe: the v1.0.0 settings (2026-10-19 settings plus the ratio depth policy and round table, cheaper verify
  rows and the C Engram gate) are in [recipes/](recipes/). I booted them with sparkrun on my pair; the file here
  differs from the booted one in host paths, container mount points, the request log name, how the rendezvous
  address is passed, an extra `SYS_PTRACE` capability for debugging on my boots, the server default top_p
  (1.0 on my boots, 0.95 here), and, for the decode and prefill tables, seccomp unconfined instead of the narrow
  profile (the second gate ran with the profile).
- Gate: the full release gate (10 lanes plus exactness, hardmode and fidelity to 248K included) passed on the
  v1.0.0 image on 2026-10-21 ([#5](../../issues/5)).
- Image: `ghcr.io/ursuciprian/ds41-tensorfold:v1.0.0` (also tagged `cand4-ee6ff92`), digest
  `sha256:d8a3f5bc255c58a403ce9ca2dae314d71dd8bae8998c807070821148d48a13d6`.
- In this release: depth policy [#2](../../issues/2) and cheaper verify rows [#3](../../issues/3). Not in it: the
  drafter head trim [#4](../../issues/4).
- In progress, tracked as issues; each change has to pass the gate before it lands in the recipe:
  - [#6](../../issues/6) drafter refit for prose, [#7](../../issues/7) attention requant to 4 bpw
  - [#8](../../issues/8) larger KV pool (1,048,576 tokens fits with 7.2 GiB to spare; its gate run is next), [#9](../../issues/9) prefill at 4K

## Repository

| Path | What |
|---|---|
| [recipes/](recipes/) | the sparkrun recipe |
| [docs/setup.md](docs/setup.md) | download, Engram packing, copy to the worker, headless setup, memory notes |
| [tools/pack_engram.py](tools/pack_engram.py) | packs the Engram tables into per-rank shards, and checks them |
| [gate/](gate/) | quality gate, speed matrix and exactness checks (OpenAI API only) |
| [BENCHMARKS.md](BENCHMARKS.md) | every number, with method and provenance |
| [scripts/make_charts.py](scripts/make_charts.py) | renders the charts from [docs/data/bench.csv](docs/data/bench.csv) |

## Acknowledgments

- The engine is my fork of [TensorFold](https://github.com/ashhart/TensorFold) with Jay Leaton's
  DeepSeek-V4.1-Flash family from
  [jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark).
  [tools/pack_engram.py](tools/pack_engram.py) is adapted from his Engram packer, and the image ships his prefill
  tile table unchanged.
- [recipes/seccomp-io-uring.json](recipes/seccomp-io-uring.json) is Docker's default seccomp profile from
  [moby/profiles](https://github.com/moby/profiles) with one rule added.
- The charts use the [Inter](https://github.com/rsms/inter) typeface (SIL OFL 1.1).

## License

Apache-2.0 for this repository ([LICENSE](LICENSE)). The Inter font in docs/fonts is SIL OFL 1.1
([docs/fonts/LICENSE.txt](docs/fonts/LICENSE.txt)). The engine image and the model weights have their own licenses;
see [NOTICE](NOTICE). The image is built on NVIDIA's NGC PyTorch container: pulling or running it means accepting
the NVIDIA Software License Agreement and the Product-Specific Terms for NVIDIA AI Products for the NVIDIA parts
(links in NOTICE).
