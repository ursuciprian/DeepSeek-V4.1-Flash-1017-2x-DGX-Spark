#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Adapted from scripts/pack_engram.py of jayleaton/deepseek-v41-tensorfold-spark (commit 2dcce03),
# Copyright 2026 Jay Leaton, Apache License 2.0. Modified by Ciprian Ursu, 2026 (see NOTICE for the changes).
"""Pack the DeepSeek-V4.1-Flash-1017 Engram tables into the per-rank shards the engine reads at TP=2.

Source: shards 47 and 48 of deepseek-ai/DeepSeek-V4.1-Flash-1017, which hold, per Engram layer L,
`layers.L.engram.embed.weight` (F8_E4M3 [rows, 256]) and `layers.L.engram.embed.scale` (F8_E8M0 [rows, 8]).

Output, one file per layer for this rank: `engram-l{L}-r{rank}of2.bin`. A 4,096-byte header starts with six
little-endian u64 (magic b"DSV41EN1", layer, lo, hi, total rows, 264), then rows lo .. hi - 1, each 256 value
bytes followed by its 8 scale bytes. Rank r owns the complete hash heads [12 r, 12 r + 12) of each layer: one
contiguous row range, where every head's row count is a distinct prime above engram_vocab_size - 1. That row
layout follows the Engram hashing in DeepSeek's reference inference code for the model (MIT).

  pack_engram.py --src SRC --config CONFIG --rank R --out DIR      # write this rank's two shards
  pack_engram.py --src SRC --config CONFIG --rank R --check DIR    # verify header, size and sampled rows
  pack_engram.py --selftest

Needs numpy. Reads and writes go through the page cache in 256 MiB chunks, which are dropped right after use, so
the unified memory of a Spark is not filled with 140 GB of cache.
"""

import argparse, json, os, random, struct, sys, tempfile

import numpy as np

MAGIC = 0x31344E4531565344          # b"DSV41EN1" little-endian
HEADER = 4096
VALUES, SCALES = 256, 8
ROW = VALUES + SCALES
CHUNK_ROWS = 1 << 20                # 256 MiB of values a chunk


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def head_rows(cfg):
    """{layer: [row count of each hash head]} in (order, head) order, as the model's hashing defines them."""
    seen, out = set(), {}
    for layer in cfg["engram_layer_ids"]:
        sizes = []
        for _ in range(cfg["engram_max_ngram_size"] - 1):
            cur = cfg["engram_vocab_size"] - 1
            for _ in range(cfg["engram_n_heads"]):
                cur += 1
                while not is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                sizes.append(cur)
        out[layer] = sizes
    return out


def rank_range(sizes, rank, world=2):
    part = (len(sizes) + world - 1) // world
    h0, h1 = rank * part, min(rank * part + part, len(sizes))
    return sum(sizes[:h0]), sum(sizes[:h1]), sum(sizes)


def load_config(path):
    cfg = json.load(open(path))
    return cfg.get("text_config", cfg)


def tensors(src):
    """{name: (file, absolute data offset, shape)} over every *.safetensors in src."""
    out = {}
    for name in sorted(os.listdir(src)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(src, name)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
        for key, t in head.items():
            if key != "__metadata__":
                out[key] = (path, 8 + n + t["data_offsets"][0], t["shape"])
    return out


def sources(src, layer, total):
    t = tensors(src)
    w, s = t.get(f"layers.{layer}.engram.embed.weight"), t.get(f"layers.{layer}.engram.embed.scale")
    if w is None or s is None:
        sys.exit(f"{src}: no layers.{layer}.engram.embed.weight / .scale (need shards 47 and 48)")
    if w[2] != [total, VALUES] or s[2] != [total, SCALES]:
        sys.exit(f"layer {layer}: source shapes {w[2]} / {s[2]}, the config implies [{total}, 256] / [{total}, 8]")
    return w, s


def drop(fd, off, length):
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, off, length, os.POSIX_FADV_DONTNEED)


def read_rows(f, base, width, lo, n):
    f.seek(base + lo * width)
    buf = np.empty((n, width), np.uint8)
    if f.readinto(memoryview(buf).cast("B")) != n * width:
        sys.exit(f"{f.name}: short read")
    drop(f.fileno(), base + lo * width, n * width)
    return buf


def pack(src, cfg, rank, out_dir, world=2, chunk=CHUNK_ROWS):
    os.makedirs(out_dir, exist_ok=True)
    for layer, sizes in head_rows(cfg).items():
        lo, hi, total = rank_range(sizes, rank, world)
        (wpath, woff, _), (spath, soff, _) = sources(src, layer, total)
        dst = os.path.join(out_dir, f"engram-l{layer}-r{rank}of{world}.bin")
        tmp = dst + ".part"
        head = bytearray(HEADER)
        struct.pack_into("<6Q", head, 0, MAGIC, layer, lo, hi, total, ROW)
        with open(wpath, "rb") as wf, open(spath, "rb") as sf, open(tmp, "wb") as of:
            of.write(head)
            for r in range(lo, hi, chunk):
                n = min(chunk, hi - r)
                of.write(np.concatenate([read_rows(wf, woff, VALUES, r, n), read_rows(sf, soff, SCALES, r, n)], 1))
                of.flush()
                getattr(os, "fdatasync", os.fsync)(of.fileno())
                drop(of.fileno(), 0, 0)
                print(f"\r{os.path.basename(dst)}: {r + n - lo:,} / {hi - lo:,} rows", end="", file=sys.stderr)
        os.replace(tmp, dst)
        print(f"\r{dst}: {hi - lo:,} rows [{lo:,}, {hi:,}) of {total:,}", file=sys.stderr)


def check(src, cfg, rank, pack_dir, world=2, samples=256):
    bad = 0
    for layer, sizes in head_rows(cfg).items():
        lo, hi, total = rank_range(sizes, rank, world)
        (wpath, woff, _), (spath, soff, _) = sources(src, layer, total)
        path = os.path.join(pack_dir, f"engram-l{layer}-r{rank}of{world}.bin")
        with open(path, "rb") as pf, open(wpath, "rb") as wf, open(spath, "rb") as sf:
            got = struct.unpack("<6Q", pf.read(48))
            want = (MAGIC, layer, lo, hi, total, ROW)
            size = os.fstat(pf.fileno()).st_size
            problems = []
            if got != want:
                problems.append(f"header {got[1:]} != {want[1:]}")
            if size != HEADER + (hi - lo) * ROW:
                problems.append(f"size {size} != {HEADER + (hi - lo) * ROW}")
            rng = random.Random(layer * 2 + rank)
            rows = [lo, hi - 1] + [rng.randrange(lo, hi) for _ in range(samples)]
            for r in rows if not problems else []:
                pf.seek(HEADER + (r - lo) * ROW)
                rec = pf.read(ROW)
                if rec != read_rows(wf, woff, VALUES, r, 1).tobytes() + read_rows(sf, soff, SCALES, r, 1).tobytes():
                    problems.append(f"row {r} differs")
                    break
        print(f"{path}: " + ("; ".join(problems) if problems else f"ok (header, size, {len(rows)} rows)"))
        bad += bool(problems)
    return bad


def selftest():
    # The 1017 config: the computed row ranges must equal the 1017 tables' row counts and the headers of my packs.
    dsv41 = {"engram_layer_ids": [1, 14], "engram_max_ngram_size": 4, "engram_vocab_size": 16000000, "engram_n_heads": 8}
    h = head_rows(dsv41)
    assert [sum(h[1]), sum(h[14])] == [384006168, 384016682]
    assert rank_range(h[1], 0) == (0, 192001740, 384006168)
    assert rank_range(h[14], 0) == (0, 192007016, 384016682)
    assert rank_range(h[1], 1)[:2] == (192001740, 384006168)

    # A tiny model: pack both ranks from fake safetensors, check them, and rebuild each full table from the parts.
    cfg = {"engram_layer_ids": [1, 14], "engram_max_ngram_size": 3, "engram_vocab_size": 31, "engram_n_heads": 2}
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "src"), os.path.join(d, "out")
        os.makedirs(src)
        full = {}
        for i, (layer, sizes) in enumerate(head_rows(cfg).items()):
            total = sum(sizes)
            w = rng.integers(0, 256, (total, VALUES), dtype=np.uint8)
            s = rng.integers(0, 256, (total, SCALES), dtype=np.uint8)
            full[layer] = np.concatenate([w, s], 1)
            meta = {f"layers.{layer}.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [total, VALUES],
                                                            "data_offsets": [0, w.nbytes]},
                    f"layers.{layer}.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [total, SCALES],
                                                           "data_offsets": [w.nbytes, w.nbytes + s.nbytes]}}
            raw = json.dumps(meta).encode()
            with open(os.path.join(src, f"model-0004{7 + i}-of-00048.safetensors"), "wb") as f:
                f.write(struct.pack("<Q", len(raw)) + raw + w.tobytes() + s.tobytes())
        for rank in (0, 1):
            pack(src, cfg, rank, out, chunk=7)
            assert check(src, cfg, rank, out, samples=16) == 0
        for layer, table in full.items():
            parts = [open(os.path.join(out, f"engram-l{layer}-r{r}of2.bin"), "rb").read()[HEADER:] for r in (0, 1)]
            assert b"".join(parts) == table.tobytes()
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="folder with model-00047/00048-of-00048.safetensors of the 1017 release")
    ap.add_argument("--config", help="config.json of the checkpoint (or of the 1017 release)")
    ap.add_argument("--rank", type=int, choices=(0, 1))
    ap.add_argument("--out", help="write this rank's shards here")
    ap.add_argument("--check", metavar="DIR", help="verify this rank's shards in DIR against the source")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.src and a.config and a.rank is not None and (a.out or a.check)):
        ap.error("--src, --config, --rank and one of --out / --check are required")
    cfg = load_config(a.config)
    if a.out:
        pack(a.src, cfg, a.rank, a.out)
    if a.check:
        sys.exit(1 if check(a.src, cfg, a.rank, a.check) else 0)


if __name__ == "__main__":
    main()
