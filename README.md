# v4l-background-buster

Real-time AI background removal for Linux virtual cameras.

A standalone C++ application that captures video from a USB camera via V4L2, runs [Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting) inference with TensorRT FP16, and outputs to a v4l2loopback virtual camera. Any application — OBS, Discord, Zoom, Teams — can use the output as a normal camera device. No plugins required.

## Status

**~57 FPS at 1080p** (~17ms/frame) on an RTX 2060 Super. Double-buffered pipeline with nvJPEG GPU-hybrid JPEG decoder overlaps CPU capture with GPU inference.

## How It Works

```
Capture thread (CPU):
  USB Camera → V4L2 DQBUF → pinned staging → nvJPEG CPU Huffman decode

Main thread (GPU, single CUDA stream):
  nvJPEG GPU IDCT → preprocess (RGB→FP32 BCHW)
  → TensorRT RVM inference (FP16, recurrent states)
  → [optional] alpha EMA smoothing
  → composite (foreground × alpha + background)
  → color convert (RGB→YUYV BT.601)
  → v4l2loopback write → any app sees a normal camera
```

Two frame slots alternate: while the GPU processes frame N, the CPU captures and decodes frame N+1.

## Requirements

- **GPU**: NVIDIA RTX (tested on RTX 2060 Super)
- **OS**: Linux (tested on Ubuntu 24.04)
- **Driver**: NVIDIA 590+
- **CUDA**: 12.x (nvcc) with nvJPEG
- **TensorRT**: 10.x
- **v4l2loopback**: 0.12+

## Quick Start

The installer handles everything — dependencies, model download, patching, TensorRT engine builds:

```bash
./install.sh
```

Then run:

```bash
# Load v4l2loopback (required after each reboot)
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="AI-Camera" exclusive_caps=1

# Run (auto-detects resolution and model)
./build/rvm-vcam
```

### Manual Installation

<details>
<summary>Click to expand manual steps</summary>

#### System dependencies

```bash
sudo apt install build-essential cmake libv4l-dev v4l-utils \
  v4l2loopback-dkms v4l2loopback-utils

# TensorRT (requires NVIDIA apt repo — see https://developer.nvidia.com/tensorrt)
sudo apt install libnvinfer-dev libnvonnxparsers-dev
```

#### Model preparation

The RVM ONNX model needs patching to work with TensorRT (the `downsample_ratio` dynamic input breaks TRT's Resize parser):

```bash
# Download
mkdir -p models
wget -O models/rvm_mobilenetv3_fp32.onnx \
  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx

# Python environment
uv venv .venv && source .venv/bin/activate
uv pip install onnx==1.16.2 onnx-graphsurgeon==0.5.2 onnxsim onnxruntime

# Patch and simplify (1080p example)
python scripts/patch_onnx.py models/rvm_mobilenetv3_fp32.onnx models/rvm_patched.onnx
onnxsim models/rvm_patched.onnx models/rvm_1080p.onnx \
  --overwrite-input-shape src:1,3,1080,1920 \
    r1i:1,16,135,240 r2i:1,20,68,120 r3i:1,40,34,60 r4i:1,64,17,30

# Build TensorRT engine (takes 2-5 minutes)
trtexec --onnx=models/rvm_1080p.onnx --saveEngine=models/rvm_1080p.plan --fp16
```

Note: `.plan` files are GPU-specific and TensorRT-version-specific. Regenerate after driver or TensorRT updates.

#### Build

```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc)
```

</details>

## Usage

```bash
# Default (green screen background, auto-detect resolution)
./build/rvm-vcam

# With benchmark timing
./build/rvm-vcam --benchmark

# With alpha smoothing (reduces matte flickering)
./build/rvm-vcam -s 0.7

# Custom background color (black)
./build/rvm-vcam -b color -c 0,0,0

# 720p mode
./build/rvm-vcam -W 1280 -H 720

# Explicit model paths
./build/rvm-vcam -m models/rvm_1080p.onnx -e models/rvm_1080p.plan
```

### CLI Options

```
  -i, --input DEVICE        Camera device (default: /dev/video0)
  -o, --output DEVICE       v4l2loopback device (default: /dev/video10)
  -W, --width N             Frame width (default: 1920)
  -H, --height N            Frame height (default: 1080)
  -m, --model PATH          ONNX model path (default: auto from resolution)
  -e, --engine PATH         TensorRT plan cache (default: auto from resolution)
  -d, --downsample RATIO    RVM downsample ratio (default: 0.25)
  -b, --background MODE     green|color (default: green)
  -c, --color R,G,B         Background color for 'color' mode (default: 0,177,64)
  -s, --smooth FACTOR       Alpha temporal smoothing (0.0-1.0, default: 1.0=off)
      --no-fp16             Disable FP16 (use FP32 throughout)
      --benchmark           Print per-frame GPU and wall timing every 100 frames
  -h, --help                Show help
```

### Verify output

```bash
# List camera devices — "AI-Camera" should appear
v4l2-ctl --list-devices

# Preview output with ffplay
ffplay /dev/video10
```

## Performance

Benchmarked on RTX 2060 Super with USB MJPEG capture card at 1080p:

| Version | Frame time | FPS | JPEG decode |
|---------|-----------|-----|-------------|
| v0.1 | ~35ms | ~28 | ~15ms (libjpeg-turbo, CPU) |
| v0.2 | ~17ms | ~57 | ~3-5ms (nvJPEG GPU-hybrid) |

TensorRT inference alone: ~3.3ms. The double-buffered pipeline hides most of the CPU Huffman decode time behind GPU processing.

## License

[Apache License 2.0](LICENSE)
