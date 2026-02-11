# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**v4l-background-buster** (working name: `rvm-vcam`) is a standalone C++ application for real-time AI-powered background removal. It captures video from a USB camera via V4L2, runs Robust Video Matting (RVM) inference using TensorRT FP16, composites the result on GPU, and outputs to a v4l2loopback virtual camera device. Any application (OBS, Discord, Zoom, Teams) can consume the output as a normal camera.

**Target**: ~5-7ms end-to-end latency at 1080p60 on RTX 2060 Super.

## Build & Run Commands

```bash
# Build
mkdir -p build && cd build && cmake .. && make -j$(nproc)

# Setup v4l2loopback (required before running)
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="AI-Camera" exclusive_caps=1

# Run
./rvm-vcam -i /dev/video0 -o /dev/video10

# Verify camera devices
v4l2-ctl --list-devices

# Build TensorRT engine from ONNX (first run, takes 2-5 minutes)
# The binary handles this automatically via loadOrBuild(), or use:
trtexec --onnx=models/rvm_mobilenetv3_fp32.onnx --saveEngine=models/rvm.plan --fp16
```

## Development Rules

- **NVIDIA RTX only** — no Intel, no AMD, no cross-platform concerns.
- **Commit often** with meaningful commit messages.
- **PLANS/ directory** is gitignored — used for planning/brainstorming artefacts only.

## Tech Stack

- **Language**: C++17 with CUDA kernels (`.cu` files)
- **Build system**: CMake
- **GPU inference**: TensorRT 10.15.1 (CUDA 13.1) with FP16
- **Video I/O**: V4L2 (capture) + v4l2loopback (output)
- **Dependencies**: CUDA 12.0 (nvcc), TensorRT 10.x, libv4l-dev, v4l2loopback-dkms
- **TensorRT headers**: installed at `/usr/include/x86_64-linux-gnu/` (multiarch path — CMake `find_package` handles this)

## Architecture

The pipeline is a synchronous per-frame loop on a single CUDA stream:

```
V4L2 capture (MMAP) → cudaMemcpyAsync H2D → preprocess kernel (BGRA→RGB fp16 BCHW)
→ TensorRT RVM inference (with recurrent state ping-pong)
→ composite kernel (alpha blend) → color convert kernel (BGRA→YUYV)
→ cudaMemcpyAsync D2H → v4l2loopback write
```

### Source Layout (planned in `src/`)

| File | Responsibility |
|------|---------------|
| `main.cpp` | CLI argument parsing, signal handling, main loop |
| `pipeline.h/cpp` | Orchestrates the full frame pipeline |
| `v4l2_capture.h/cpp` | V4L2 MMAP capture from physical camera |
| `v4l2_output.h/cpp` | Frame writer to v4l2loopback device |
| `trt_engine.h/cpp` | TensorRT engine build/load/infer (ONNX→plan caching) |
| `cuda_kernels.h/cu` | Three GPU kernels: preprocess, composite, color convert |

### Key Design Decisions

- **All GPU memory is pre-allocated once at init** and reused per frame (~56 MB GPU, ~12.4 MB pinned host). No per-frame allocations.
- **Recurrent state ping-pong**: RVM uses 4 recurrent states (r1-r4) that feed back each frame. Two sets of buffers are maintained and swapped via pointer swap (zero-copy). Initial states are `[1,ch,1,1]` zeros that broadcast on first frame.
- **Single CUDA stream**: All kernels and transfers are serialized on one stream, with a single `cudaStreamSynchronize` before the v4l2loopback write.
- **TensorRT plan files are GPU-specific and TRT-version-specific** — must be regenerated after driver or TensorRT updates.
- **Camera may output YUYV/MJPEG/NV12** instead of BGRA. The preprocess kernel must handle the actual capture format.

### RVM TensorRT Inputs/Outputs

```
Inputs:  src [1,3,1080,1920], r1i..r4i (recurrent), downsample_ratio [1] (=0.25)
Outputs: fgr [1,3,1080,1920], pha [1,1,1080,1920], r1o..r4o (recurrent)
```

Recurrent state spatial dims at downsample_ratio=0.25 (internal 270x480):
- r1: [1,16,135,240], r2: [1,20,68,120], r3: [1,40,34,60], r4: [1,64,17,30]

### Known TensorRT Issue

`downsample_ratio` feeds into a Resize node and TRT may error about shape tensor types. Workaround: use `onnx-graphsurgeon` to bake `downsample_ratio=0.25` as a constant, or use `trtexec`.

## Implementation Phases

The detailed plan is in `PLANS/PLAN-PATH-B-TENSORRT-V4L2LOOPBACK.md`. Implementation follows 7 phases:

1. V4L2 capture + v4l2loopback passthrough (verify end-to-end)
2. TensorRT engine build/load from ONNX
3. CUDA kernels (preprocess, composite, color convert)
4. Full pipeline integration
5. GPU memory pre-allocation
6. CLI interface
7. Systemd service (optional)
