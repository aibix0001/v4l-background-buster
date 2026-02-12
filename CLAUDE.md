# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**v4l-background-buster** (working name: `rvm-vcam`) is a standalone C++ application for real-time AI-powered background removal. It captures video from a USB camera via V4L2, runs Robust Video Matting (RVM) inference using TensorRT FP16, composites the result on GPU, and outputs to a v4l2loopback virtual camera device. Any application (OBS, Discord, Zoom, Teams) can consume the output as a normal camera.

**Current performance**: ~20-25ms/frame (~40-50 FPS) at 1080p on RTX 2060 Super with `--perf-level 3`. v0.4 adds cumulative performance optimizations while preserving v0.3 edge quality (guided filter, despill, ds_ratio=0.5).

## Build & Run Commands

```bash
# Full install (deps + build + model prep)
./install.sh

# Build only
mkdir -p build && cd build && cmake .. && make -j$(nproc)

# Setup v4l2loopback (required before running)
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="AI-Camera" exclusive_caps=1

# Run (auto-resolves model paths from capture resolution)
./build/rvm-vcam

# Run with options
./build/rvm-vcam -i /dev/video0 -o /dev/video10 --benchmark
./build/rvm-vcam --benchmark -s 0.7    # with alpha smoothing
./build/rvm-vcam --benchmark --perf-level 3  # max optimizations
./build/rvm-vcam --no-refine --despill 0  # disable post-processing
./build/rvm-vcam -W 1280 -H 720        # 720p mode

# Verify camera devices
v4l2-ctl --list-devices
```

## Development Rules

- **NVIDIA RTX only** — no Intel, no AMD, no cross-platform concerns.
- **Commit often** with meaningful commit messages.
- **PLANS/ directory** is gitignored — used for planning/brainstorming artefacts only.

## Tech Stack

- **Language**: C++17 with CUDA kernels (`.cu` files)
- **Build system**: CMake (defaults to Release build)
- **GPU inference**: TensorRT 10.15.1 (CUDA 13.1) with FP16
- **JPEG decode**: nvJPEG GPU-hybrid (CPU Huffman + GPU IDCT)
- **Video I/O**: V4L2 (capture) + v4l2loopback (output)
- **Dependencies**: CUDA 12.0 (nvcc), TensorRT 10.x, nvJPEG, libv4l-dev, v4l2loopback-dkms
- **TensorRT headers**: installed at `/usr/include/x86_64-linux-gnu/` (multiarch path — CMake `find_package` handles this)

## Architecture

The pipeline uses a **double-buffered design with a capture thread** for CPU-GPU overlap:

```
Capture thread (CPU):
  V4L2 DQBUF → memcpy to pinned staging → requeue mmap
  → nvjpegJpegStreamParse → nvjpegDecodeJpegHost (CPU Huffman)
  → signal slot ready

Main thread (GPU, single CUDA stream):
  wait for slot → nvjpegDecodeJpegTransferToDevice + Device (GPU IDCT)
  → launchRgbToFp32 (preprocess) → record slotDoneEvent, release slot
  → TensorRT RVM inference (with recurrent state ping-pong)
  → [periodic] recurrent state reset (zero r1-r4 to prevent drift)
  → [optional] adaptive alpha EMA (temporal smoothing)
  → [default] guided filter (alpha refinement using RGB guide)
  → [default] despill (suppress bg color fringe at edges)
  → launchCompositeToYuyv (alpha blend + RGB→YUYV)
  → cudaMemcpyAsync D2H → cudaStreamSynchronize → v4l2loopback write
```

Two `FrameSlot` structs alternate: while the GPU processes frame N, the CPU captures and Huffman-decodes frame N+1.

### Source Layout (`src/`)

| File | Responsibility |
|------|---------------|
| `main.cpp` | CLI argument parsing, signal handling, main loop |
| `pipeline.h/cpp` | Orchestrates pipeline: capture thread, double-buffered slots, GPU processing |
| `v4l2_capture.h/cpp` | V4L2 MMAP capture with poll timeout, stale frame draining, EIO retry |
| `v4l2_output.h/cpp` | Frame writer to v4l2loopback with colorspace metadata |
| `trt_engine.h/cpp` | TensorRT engine build/load/infer (ONNX→plan caching) |
| `cuda_kernels.h/cu` | GPU kernels: YUYV→RGB, RGB→FP32, adaptive alpha EMA, guided filter (shmem variants), despill, fused despill+composite+YUYV |

### Key Design Decisions

- **Double-buffered inputs with capture thread**: Two FrameSlots (h_staging, d_input, d_src, per-slot nvJPEG state) synchronized via condition variables and cudaEvents. Overlaps CPU Huffman decode with GPU inference.
- **nvJPEG GPU-hybrid decoder**: Decoupled 3-phase API (parse → CPU Huffman → GPU IDCT). Per-slot state objects enable concurrent decode while GPU processes previous frame.
- **All GPU memory is pre-allocated once at init** and reused per frame. No per-frame allocations.
- **Recurrent state ping-pong**: RVM uses 4 recurrent states (r1-r4) that feed back each frame. Two sets of buffers swapped via pointer swap (zero-copy).
- **Single CUDA stream**: All GPU work is serialized on one stream. Overlap is CPU-vs-GPU, not GPU-vs-GPU.
- **Multi-resolution model support**: Model paths auto-resolved from capture resolution (e.g. `models/rvm_1080p.onnx`, `models/rvm_720p.plan`). install.sh builds models for multiple resolutions.
- **Adaptive alpha temporal smoothing**: Optional EMA filter (`-s/--smooth`) with adaptive per-pixel strength — heavy smoothing at uncertain edges (alpha ~0.5), minimal at confident pixels (alpha ~0 or ~1), reduced on fast motion.
- **Fast guided filter**: Alpha matte refinement using high-res RGB luminance as guide (He & Sun 2015). Snaps soft matte edges to real image edges. Operates at 1/4 resolution for coefficient computation (eps=0.005 for crisp edges). On by default (`--no-refine` to disable).
- **Despill**: CUDA kernel suppresses background color contamination in RVM's foreground output at edges. Uses smooth parabolic falloff `4*a*(1-a)` instead of hard alpha cutoff — covers nearly-opaque hairline pixels that a hard threshold misses (`--despill`, default 0.8).
- **Recurrent state reset**: Periodic zeroing of RVM recurrent states prevents progressive color drift (`--reset-interval`, default 100 frames). Drift becomes visible within ~90 frames, so 100 is the tuned sweet spot.
- **V4L2 hardening**: poll() timeout before DQBUF, format verification after S_FMT, frame rate negotiation, dequeueLatestFrame() drains stale frames, DQBUF retry on EIO, consecutive skip counter.
- **TensorRT plan files are GPU-specific and TRT-version-specific** — must be regenerated after driver or TensorRT updates.
- **BT.601 limited-range**: YUYV→RGB uses limited-range coefficients; output sets V4L2 colorspace metadata.
- **Performance levels (`--perf-level 0-3`)**: Cumulative optimizations — Level 0: v0.3 baseline; Level 1: fused despill+composite+YUYV kernel, fused guided filter products; Level 2: + shared memory box filters, async output thread with double-buffered host output; Level 3: + CUDA graph replay for guided filter (~18 kernels captured into a graph per slot).
- **Always-on optimizations**: CMAKE_CUDA_ARCHITECTURES=75 (native SM 7.5 SASS), TRT builder optimization level 5, write-combined output buffer, pre-computed float background color passed to kernels.

### RVM TensorRT Inputs/Outputs

```
Inputs:  src [1,3,H,W], r1i..r4i (recurrent)
Outputs: fgr [1,3,H,W], pha [1,1,H,W], r1o..r4o (recurrent)
```

Recurrent state spatial dims at downsample_ratio=0.5 (1080p, internal 540x960):
- r1: [1,16,270,480], r2: [1,20,135,240], r3: [1,40,68,120], r4: [1,64,34,60]

### Known TensorRT Issue

`downsample_ratio` feeds into a Resize node and TRT may error about shape tensor types. Workaround: `scripts/patch_onnx.py` bakes `downsample_ratio=0.5` as a constant, then `onnxsim` folds all shape nodes to produce a static-shape model. This is handled by `install.sh`.
