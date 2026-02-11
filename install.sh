#!/usr/bin/env bash
#
# v4l-background-buster installer
#
# Installs system dependencies, builds the project, downloads and prepares
# the RVM model, and sets up v4l2loopback.
#
# Usage: ./install.sh [--skip-deps] [--skip-model] [--skip-build]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────

MODEL_URL="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"
MODEL_DIR="models"
MODEL_ORIG="$MODEL_DIR/rvm_mobilenetv3_fp32.onnx"
MODEL_PATCHED="$MODEL_DIR/rvm_mobilenetv3_fp32_patched.onnx"
MODEL_SIMPLIFIED="$MODEL_DIR/rvm_mobilenetv3_fp32_simplified.onnx"
VENV_DIR=".venv"
BUILD_DIR="build"

SKIP_DEPS=false
SKIP_MODEL=false
SKIP_BUILD=false

# ── Argument parsing ─────────────────────────────────────────────────────────

for arg in "$@"; do
    case "$arg" in
        --skip-deps)  SKIP_DEPS=true ;;
        --skip-model) SKIP_MODEL=true ;;
        --skip-build) SKIP_BUILD=true ;;
        --help|-h)
            echo "Usage: $0 [--skip-deps] [--skip-model] [--skip-build]"
            echo ""
            echo "  --skip-deps   Skip apt dependency installation"
            echo "  --skip-model  Skip model download and preparation"
            echo "  --skip-build  Skip C++ build"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

info()  { echo -e "\033[1;34m==>\033[0m \033[1m$*\033[0m"; }
ok()    { echo -e "\033[1;32m  ✓\033[0m $*"; }
warn()  { echo -e "\033[1;33m  !\033[0m $*"; }
err()   { echo -e "\033[1;31m  ✗\033[0m $*"; }
die()   { err "$*"; exit 1; }

check_cmd() {
    command -v "$1" &>/dev/null
}

# ── Pre-flight checks ────────────────────────────────────────────────────────

info "Checking system requirements"

# NVIDIA driver
if check_cmd nvidia-smi; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    ok "NVIDIA GPU: $GPU_NAME (driver $DRIVER_VER)"
else
    die "nvidia-smi not found — install NVIDIA drivers first"
fi

# CUDA compiler
if check_cmd nvcc; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9.]+')
    ok "CUDA: $CUDA_VER"
else
    die "nvcc not found — install CUDA toolkit first"
fi

# cmake
if check_cmd cmake; then
    CMAKE_VER=$(cmake --version | head -1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')
    ok "cmake: $CMAKE_VER"
else
    die "cmake not found (will be installed with dependencies)"
fi

echo ""

# ── System dependencies ──────────────────────────────────────────────────────

if [ "$SKIP_DEPS" = false ]; then
    info "Installing system dependencies"

    PACKAGES=(
        build-essential
        cmake
        libv4l-dev
        v4l-utils
        v4l2loopback-dkms
        v4l2loopback-utils
        libturbojpeg0-dev
        wget
    )

    # TensorRT packages (from NVIDIA repo)
    TRT_PACKAGES=(
        libnvinfer-dev
        libnvonnxparsers-dev
    )

    # Check if TensorRT is already available
    if ldconfig -p 2>/dev/null | grep -q libnvinfer; then
        ok "TensorRT already installed"
    else
        warn "TensorRT not found in ldconfig — adding TRT packages to install list"
        warn "Make sure the NVIDIA apt repository is configured"
        warn "See: https://developer.nvidia.com/tensorrt"
        PACKAGES+=("${TRT_PACKAGES[@]}")
    fi

    sudo apt-get update -qq
    sudo apt-get install -y -qq "${PACKAGES[@]}"
    ok "System dependencies installed"
    echo ""
fi

# ── Build ─────────────────────────────────────────────────────────────────────

if [ "$SKIP_BUILD" = false ]; then
    info "Building rvm-vcam"

    mkdir -p "$BUILD_DIR"
    cmake -B "$BUILD_DIR" -S . -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" -j"$(nproc)"

    if [ -f "$BUILD_DIR/rvm-vcam" ]; then
        ok "Built: $BUILD_DIR/rvm-vcam"
    else
        die "Build failed — rvm-vcam binary not found"
    fi
    echo ""
fi

# ── Model preparation ────────────────────────────────────────────────────────

if [ "$SKIP_MODEL" = false ]; then
    info "Preparing RVM model"
    mkdir -p "$MODEL_DIR"

    # 1. Download
    if [ -f "$MODEL_SIMPLIFIED" ]; then
        ok "Simplified model already exists: $MODEL_SIMPLIFIED"
    else
        if [ ! -f "$MODEL_ORIG" ]; then
            info "Downloading RVM MobileNetV3 ONNX model"
            wget -q --show-progress -O "$MODEL_ORIG" "$MODEL_URL"
            ok "Downloaded: $MODEL_ORIG"
        else
            ok "Original model already exists: $MODEL_ORIG"
        fi

        # 2. Set up Python venv
        if ! check_cmd uv; then
            info "Installing uv (Python package manager)"
            # F15: Pin to specific version instead of latest
            curl -LsSf https://astral.sh/uv/0.6.2/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"
        fi

        if [ ! -d "$VENV_DIR" ]; then
            info "Creating Python virtual environment"
            uv venv "$VENV_DIR"
        fi

        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        uv pip install -q onnx==1.16.2 onnx-graphsurgeon==0.5.2 onnxsim onnxruntime
        ok "Python environment ready"

        # 3. Patch: bake downsample_ratio as constant
        info "Patching model (baking downsample_ratio=0.25)"
        python scripts/patch_onnx.py "$MODEL_ORIG" "$MODEL_PATCHED"
        ok "Patched: $MODEL_PATCHED"

        # 4. Simplify: fold all shape nodes into static constants
        info "Simplifying model (folding shapes to static)"
        onnxsim "$MODEL_PATCHED" "$MODEL_SIMPLIFIED" \
            --overwrite-input-shape \
            src:1,3,1080,1920 \
            r1i:1,16,135,240 \
            r2i:1,20,68,120 \
            r3i:1,40,34,60 \
            r4i:1,64,17,30
        ok "Simplified: $MODEL_SIMPLIFIED"

        deactivate 2>/dev/null || true
    fi

    # 5. Build TensorRT engine
    if [ -f "$MODEL_DIR/rvm.plan" ]; then
        ok "TensorRT engine already exists: $MODEL_DIR/rvm.plan"
        warn "Delete it to force rebuild (engine is GPU- and TRT-version-specific)"
    else
        if check_cmd trtexec; then
            info "Building TensorRT engine (this takes 2-5 minutes)"
            trtexec \
                --onnx="$MODEL_SIMPLIFIED" \
                --saveEngine="$MODEL_DIR/rvm.plan" \
                --fp16 \
                2>&1 | tail -5
            ok "Engine built: $MODEL_DIR/rvm.plan"
        else
            warn "trtexec not found — engine will be built on first run"
        fi
    fi
    echo ""
fi

# ── v4l2loopback ─────────────────────────────────────────────────────────────

info "Checking v4l2loopback"

if lsmod | grep -q v4l2loopback; then
    ok "v4l2loopback module already loaded"
else
    warn "v4l2loopback not loaded — loading now"
    sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="AI-Camera" exclusive_caps=1
    ok "v4l2loopback loaded (video_nr=10)"
fi

echo ""

# ── Done ──────────────────────────────────────────────────────────────────────

info "Installation complete!"
echo ""
echo "  Run with:  ./build/rvm-vcam"
echo "  Benchmark: ./build/rvm-vcam --benchmark"
echo "  Help:      ./build/rvm-vcam --help"
echo ""
echo "  Note: v4l2loopback must be loaded before each reboot:"
echo "    sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=\"AI-Camera\" exclusive_caps=1"
echo ""
