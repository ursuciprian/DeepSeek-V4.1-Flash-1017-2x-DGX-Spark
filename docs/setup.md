# Setup

The steps I follow to get two DGX Sparks from a clean install to a running server: download the pinned weights,
pack the Engram tables per rank, copy everything to the worker, free display memory, and set up memory guards.

In the commands, "head" is the node that serves HTTP (rank 0) and "worker" is the other one (rank 1).
`<worker-cx7>` is the worker's IPv4 address on the CX-7 link, and `<head-cx7>` the head's.

## Layout

The recipe expects this on both nodes:

```
/srv/ds41/
  checkpoint/              EXL3 2.9 bpw checkpoint, the same on both nodes        197 GiB
  engram-shards/      this node's packed Engram shards                      94.4 GiB
    engram-l1-r<rank>of2.bin
    engram-l14-r<rank>of2.bin
  tf-state/                request log, written by the server
  seccomp-io-uring.json    copy of recipes/seccomp-io-uring.json (seccomp profile)
```

sparkrun does not fill placeholders in a recipe's volume list, so these paths are fixed in the recipe. If your
disk is mounted somewhere else, make `/srv/ds41` a symlink to a folder on it, or change the four paths in your
copy of the recipe (`model:`, the two `volumes` lines and `security_opt`).

```sh
sudo mkdir -p /srv/ds41 && sudo chown "$USER" /srv/ds41
mkdir -p /srv/ds41/tf-state
cp recipes/seccomp-io-uring.json /srv/ds41/    # from a clone of this repo
```

Run this on both nodes. sparkrun starts the container as your SSH user, so `tf-state` must belong to that user.

## Download

I pin both sources to a commit, so a later upload cannot change what you serve. The commits after `afc7aa0a` on the
checkpoint repository only change its model card.

| What | Hugging Face repository | Revision |
|---|---|---|
| checkpoint | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` | `afc7aa0aab98ee735c0f54e0a122019af1ac08f1` |
| Engram shards 47 and 48 | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `1bae0e1088993b9b82de08f59de5be5a34e42170` |

Both are MIT-licensed. On the head:

```sh
uvx --from huggingface_hub hf download LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw \
  --revision afc7aa0aab98ee735c0f54e0a122019af1ac08f1 --local-dir /srv/ds41/checkpoint

uvx --from huggingface_hub hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir /srv/ds41/engram-src-1017
```

The two Engram shards are about 190 GiB together. They hold the n-gram embedding tables of layers 1 and 14 (about
384 million rows each); the engine needs only those tables from them, and the rest of the model comes from the
checkpoint. Check them before packing:

```sh
cd /srv/ds41/engram-src-1017 && sha256sum -c - <<'EOF'
76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5  model-00047-of-00048.safetensors
9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd  model-00048-of-00048.safetensors
EOF
```

The checkpoint needs no chat template file: the engine builds prompts with its own port of DeepSeek's
`encoding/encoding.py`, which ships with the checkpoint.

## Pack the Engram tables per rank

At TP=2 each rank reads only its own half of each Engram table: rank 0 owns hash heads 0-11 of every layer, rank 1
owns heads 12-23. [tools/pack_engram.py](../tools/pack_engram.py) cuts each table into these halves and writes one
file per layer and rank, in the layout the engine reads with O_DIRECT. It needs numpy; `uv run` provides it
without touching the system Python.

Pack both ranks on the head, rank 1 into a separate folder that goes to the worker. While rank 1's files wait
for the copy, the head holds about 285 GiB more than the layout above (the source shards plus rank 1's files).

```sh
git clone https://github.com/ursuciprian/DeepSeek-V4.1-Flash-1017-2x-DGX-Spark
cd DeepSeek-V4.1-Flash-1017-2x-DGX-Spark
P="uv run --no-project --with numpy python tools/pack_engram.py --src /srv/ds41/engram-src-1017 \
  --config /srv/ds41/checkpoint/config.json"
$P --rank 0 --out /srv/ds41/engram-shards
$P --rank 1 --out /srv/ds41/engram-r1
$P --rank 0 --check /srv/ds41/engram-shards
$P --rank 1 --check /srv/ds41/engram-r1
```

Each run writes two files of about 47 GiB. `--check` compares the header, the file size, the first and last rows
and 256 random rows of each file against the source, and prints `ok` per file. The tool drops the page cache
behind itself as it goes, but run it while no model is loaded.

When both checks print `ok`, the source shards can go, unless you want to run `--check` again later:

```sh
rm -r /srv/ds41/engram-src-1017
```

## Copy to the worker over the fabric

Copy over the CX-7 link, not the management network. On my pair one rail moved about 440 MB/s, so the copy
takes 10-15 minutes.

```sh
ssh <worker-cx7> 'sudo mkdir -p /srv/ds41 && sudo chown "$USER" /srv/ds41 && mkdir -p /srv/ds41/tf-state'
rsync -a --partial --info=progress2 /srv/ds41/checkpoint <worker-cx7>:/srv/ds41/
rsync -a --partial --info=progress2 /srv/ds41/engram-r1/ <worker-cx7>:/srv/ds41/engram-shards/
rsync -a /srv/ds41/seccomp-io-uring.json <worker-cx7>:/srv/ds41/
```

rsync checks each file after transfer. Then remove the head's copy of rank 1's files:

```sh
rm -r /srv/ds41/engram-r1
```

Both nodes must end up with the same checkpoint at the same path. Two ranks with different weights can serve
wrong text instead of failing.

## Run headless

On unified memory, everything the desktop takes comes out of the model's budget. I run both nodes without a
display manager and with the NVIDIA DRM framebuffer off. On another model on this pair, that gave 1.9-2.7 GiB more
MemAvailable.

On both nodes:

```sh
sudo systemctl set-default multi-user.target
echo 'options nvidia-drm modeset=1 fbdev=0' | sudo tee /etc/modprobe.d/zz-nvidia-drm-override.conf
sudo update-initramfs -u
sudo reboot
```

The `zz-` prefix makes the file load after the driver's own `modprobe.d` entry. After the reboot, check:

```sh
systemctl get-default                                # multi-user.target
cat /sys/module/nvidia_drm/parameters/modeset        # Y
cat /sys/module/nvidia_drm/parameters/fbdev          # N
```

## Memory

The engine allocates about 100.6 GiB per rank, and the GPU and the host share the same 128 GB. Under load my
nodes keep 8-10 GiB MemAvailable, while MemFree drops to about 1 GiB. That is normal: the rest is reclaimable
cache.

- Keep at least 5 GiB MemAvailable on both nodes. The engine holds back a new prompt whose prefill would take a
  rank below the recipe's 5 GiB floor and refuses it under the 4 GiB hard floor, but it cannot plan for memory
  that other programs take. Do not start other GPU or memory-heavy jobs next to it: a build, a second model, a
  large download into the page cache.
- Run earlyoom on both nodes as a last resort. The package default (10% of memory and 10% of swap) does not
  match this load, so I set it like this:

  ```sh
  sudo apt install earlyoom
  sudo tee /etc/default/earlyoom <<'EOF'
  EARLYOOM_ARGS="-m 2 -s 80 --prefer '(python3|python)' --avoid '(systemd|sshd|dockerd|containerd)'"
  EOF
  sudo systemctl restart earlyoom
  systemctl is-active earlyoom && journalctl -u earlyoom -n 3 --no-pager
  ```

  With `-m 2` it acts at about 2.4 GiB MemAvailable, below the engine's floors, so it fires only when something
  else has gone wrong. When a node thrashes while MemAvailable still looks fine, it does not fire.
- Never run Nsight Compute kernel replay (`ncu --set full` and similar) against a loaded model. On GB10, replay
  saves device memory into host RAM. On my pair it pushed a loaded head into a memory thrash, and the node needed
  a power cycle. Profile with Nsight Systems, or profile kernels without the model loaded.
- Before a boot, check that nothing else holds memory: `free -g` and `nvidia-smi` on both nodes.

## Launch

From the head, or any machine that reaches both nodes over SSH:

```sh
sparkrun run recipes/deepseek-v4.1-flash-1017-2x-dgx-spark.yaml --hosts <head>,<worker> \
  -e DS41_MASTER=<head-cx7> --no-follow
```

The README's [Quick start](../README.md#quick-start) covers the health check and the first request.
