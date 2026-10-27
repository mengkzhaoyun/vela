# Vela

**Vela** is a high-throughput, **Multi-GPU Real-World Video Super-Resolution (VSR) Streaming Engine** designed for restoring low-resolution (480p / 576p / 720p), heavily compressed legacy live-action videos to crisp **1080p / 4K**.

Built as a sister project to [`jasna`](https://github.com/Mengkzhaoyun/jasna) and [`lada`](https://github.com/Mengkzhaoyun/lada), Vela eliminates disk I/O bottlenecks by combining **Multi-GPU GOP Work-Stealing Scheduling** with a **zero-intermediate-file hardware streaming pipeline** (`FFmpeg NVDEC -> Spandrel PLKSR / ClearRealityV1 -> GPU Antialiased Downscale + CAS -> NVENC`).

---

## Key Features

1. **Native Multi-GPU Work-Stealing Parallelism (`--gpus all`)**:
   - Automatically detects all available NVIDIA GPUs (including heterogeneous setups like **RTX 5090 + RTX 4090**).
   - Splits long-form videos (e.g., 140-minute 480p movies) into keyframe-aligned GOP segments and dispatches them through a lock-free work-stealing queue so faster GPUs automatically process more segments.
   - Crash-safe segment checkpointing (`.vela_work/`): interrupted jobs resume instantly from the last completed GOP segment.
2. **Real-World Live-Action Super-Resolution Models (`PLKSRArch` / `Spandrel`)**:
   - **`realplksr` (`4xNomosWebPhoto_RealPLKSR.pth`)** _(Default)_: Removes H.264 macroblocking and mosquito noise while reconstructing natural skin pores, teeth edges, lip highlights, and eye catchlights without plastic over-smoothing.
   - **`clearreality` (`4x-ClearRealityV1.pth`)**: High-acuity edge and fabric texture enhancement for close-up shots.
3. **Zero-Disk-Frame GPU Streaming Pipeline**:
   - Streams decoded frames directly into pinned memory -> GPU VRAM -> `4x` Neural SR -> GPU Antialiased Downscale (`1920×1080`) -> GPU **Contrast Adaptive Sharpening (CAS)** -> `hevc_nvenc` / `h264_nvenc` hardware encoding -> lossless Concat & Audio/Chapter Remux.

---

## Quick Start

### 1. CLI Usage

```bash
# Upscale a single 480p video to 1080p across all available GPUs
vela --input MKZ-033.mkv --output MKZ-033_1080p.mp4 --gpus all --model realplksr

# Batch scan a directory (/data) and upscale all <1080p videos using GPUs 0 and 1
vela --scan-dir /data --gpus 0,1 --model realplksr --codec hevc --target-bitrate 5M

# Inspect detected GPUs, VRAM, and model weights
vela info
```

### 2. Docker Deployment (RTX 4090 / RTX 5090 Multi-GPU)

```bash
docker pull registry.cn-qingdao.aliyuncs.com/wod/cuda:13.0.3-vela-v0.1.0 && \
docker rm -f vela || true && \
docker run --name vela \
  -d --rm \
  --gpus all \
  -v /data/jasna/model_weights:/app/vela/model_weights \
  -v /data/jasna/ais:/data \
  registry.cn-qingdao.aliyuncs.com/wod/cuda:13.0.3-vela-v0.1.0
```

See [`.drone/README.md`](.drone/README.md) for full production deployment, offline model download, and CI/CD instructions.
