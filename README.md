# v4l-background-buster

Real-time AI background removal for Linux virtual cameras.

A standalone C++ application that captures video from a USB camera via V4L2, runs [Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting) inference with TensorRT FP16, and outputs to a v4l2loopback virtual camera. Any application — OBS, Discord, Zoom, Teams — can use the output as a normal camera device. No plugins required.

## Status

**Working prototype.** End-to-end pipeline runs at ~28 FPS (35ms/frame) at 1080p on an RTX 2060 Super. TensorRT inference alone takes ~3.3ms; the main bottleneck is CPU-side MJPEG decode.

## How It Works

```
USB Camera (/dev/video0)
    → V4L2 MMAP capture (MJPEG or YUYV)
    → turbojpeg decode (MJPEG) or memcpy (YUYV)
    → CUDA upload (async, pinned memory)
    → Preprocess kernel (RGB uint8 → FP32 BCHW [0,1])
    → TensorRT RVM inference (FP16 internal, FP32 I/O, recurrent states)
    → Composite kernel (foreground × alpha + background × (1-alpha))
    → Color convert kernel (RGB → YUYV, BT.601)
    → CUDA download (async)
    → v4l2loopback write (/dev/video10)
    → Any app sees a normal camera
```

## Requirements

- **GPU**: NVIDIA RTX (tested on RTX 2060 Super)
- **OS**: Linux (tested on Ubuntu 24.04)
- **Driver**: NVIDIA 590+
- **CUDA**: 12.x (nvcc)
- **TensorRT**: 10.x
- **v4l2loopback**: 0.12+

## Installation

### System dependencies

```bash
# Build tools
sudo apt install build-essential cmake

# TensorRT (requires NVIDIA apt repo — see https://developer.nvidia.com/tensorrt)
sudo apt install tensorrt libnvinfer-dev libnvonnxparsers-dev

# V4L2
sudo apt install libv4l-dev v4l-utils v4l2loopback-dkms v4l2loopback-utils

# MJPEG decode
sudo apt install libturbojpeg0-dev
```

### Model preparation

The RVM ONNX model needs patching to work with TensorRT (the `downsample_ratio` dynamic input breaks TRT's Resize parser). A three-step pipeline produces a fully static-shape model:

```bash
# 1. Download RVM MobileNetV3 ONNX model
mkdir -p models
wget -O models/rvm_mobilenetv3_fp32.onnx \
  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx

# 2. Set up Python environment (requires astral uv)
uv venv .venv
source .venv/bin/activate
uv pip install onnx==1.16.2 onnx-graphsurgeon==0.5.2 onnxsim onnxruntime

# 3. Bake downsample_ratio=0.25 as constant
python scripts/patch_onnx.py \
  models/rvm_mobilenetv3_fp32.onnx \
  models/rvm_mobilenetv3_fp32_patched.onnx

# 4. Simplify (fold all shape nodes into static constants)
onnxsim models/rvm_mobilenetv3_fp32_patched.onnx \
  models/rvm_mobilenetv3_fp32_simplified.onnx \
  --overwrite-input-shape src:1,3,1080,1920 \
    r1i:1,16,135,240 r2i:1,20,68,120 r3i:1,40,34,60 r4i:1,64,17,30
```

The TensorRT engine (`.plan` file) is built automatically on first run, or you can pre-build it:

```bash
trtexec --onnx=models/rvm_mobilenetv3_fp32_simplified.onnx \
  --saveEngine=models/rvm.plan --fp16
```

Note: `.plan` files are GPU-specific and TensorRT-version-specific. Regenerate after driver or TensorRT updates.

### Build

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Usage

```bash
# Load v4l2loopback kernel module
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="AI-Camera" exclusive_caps=1

# Run with defaults (green screen background)
./build/rvm-vcam

# Run with custom background color
./build/rvm-vcam -b color -c 0,0,0

# Run with benchmark timing
./build/rvm-vcam --benchmark
```

### CLI Options

```
  -i, --input DEVICE        Camera device (default: /dev/video0)
  -o, --output DEVICE       v4l2loopback device (default: /dev/video10)
  -W, --width N             Frame width (default: 1920)
  -H, --height N            Frame height (default: 1080)
  -m, --model PATH          ONNX model path (default: models/rvm_mobilenetv3_fp32_simplified.onnx)
  -e, --engine PATH         TensorRT plan cache (default: models/rvm.plan)
  -d, --downsample RATIO    RVM downsample ratio (default: 0.25)
  -b, --background MODE     green|color (default: green)
  -c, --color R,G,B         Background color for 'color' mode (default: 0,177,64)
      --no-fp16             Disable FP16 (use FP32 throughout)
      --benchmark           Print per-frame timing every 100 frames
  -h, --help                Show help
```

### Verify output

```bash
# List camera devices — "AI-Camera" should appear
v4l2-ctl --list-devices

# Preview output with ffplay
ffplay /dev/video10
```

## License

[Apache License 2.0](LICENSE)
