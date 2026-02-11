#include "cuda_kernels.h"
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// YUYV → RGB FP16 BCHW [0,1]
// YUYV packs 2 pixels in 4 bytes: [Y0, U, Y1, V]
// ---------------------------------------------------------------------------
__global__ void yuyvToRgbFp16Kernel(const uint8_t* __restrict__ yuyv,
                                     __half* __restrict__ rgb,
                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixIdx = y * width + x;
    // Each YUYV macro-pixel is 4 bytes for 2 horizontal pixels
    int macroX = x / 2;
    int byteOff = (y * width / 2 + macroX) * 4;

    uint8_t Y = yuyv[byteOff + (x & 1) * 2];  // Y0 or Y1
    uint8_t U = yuyv[byteOff + 1];
    uint8_t V = yuyv[byteOff + 3];

    // BT.601 YUV→RGB
    float fY = static_cast<float>(Y);
    float fU = static_cast<float>(U) - 128.0f;
    float fV = static_cast<float>(V) - 128.0f;

    float R = fY + 1.402f * fV;
    float G = fY - 0.344136f * fU - 0.714136f * fV;
    float B = fY + 1.772f * fU;

    // Clamp and normalize to [0,1]
    R = fminf(fmaxf(R, 0.0f), 255.0f) / 255.0f;
    G = fminf(fmaxf(G, 0.0f), 255.0f) / 255.0f;
    B = fminf(fmaxf(B, 0.0f), 255.0f) / 255.0f;

    // Write BCHW: channel-first layout
    int planeSize = width * height;
    rgb[0 * planeSize + pixIdx] = __float2half(R);
    rgb[1 * planeSize + pixIdx] = __float2half(G);
    rgb[2 * planeSize + pixIdx] = __float2half(B);
}

void launchYuyvToRgbFp16(const uint8_t* d_yuyv, __half* d_rgb,
                          int width, int height, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    yuyvToRgbFp16Kernel<<<grid, block, 0, stream>>>(d_yuyv, d_rgb, width, height);
}

// ---------------------------------------------------------------------------
// Composite FGR+PHA → YUYV
// Combines alpha blending and RGB→YUYV in one kernel to avoid intermediate buffer.
// Processes 2 horizontal pixels per thread (YUYV macro-pixel).
// ---------------------------------------------------------------------------
__global__ void compositeToYuyvKernel(const __half* __restrict__ fgr,
                                       const __half* __restrict__ pha,
                                       uint8_t* __restrict__ yuyv,
                                       int width, int height,
                                       uint8_t bgR, uint8_t bgG, uint8_t bgB) {
    // Each thread handles 2 horizontal pixels (one YUYV macro-pixel)
    int macroX = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int halfW = width / 2;
    if (macroX >= halfW || y >= height) return;

    int planeSize = width * height;
    float bgRf = bgR / 255.0f;
    float bgGf = bgG / 255.0f;
    float bgBf = bgB / 255.0f;

    float R[2], G[2], B[2];
    for (int i = 0; i < 2; i++) {
        int x = macroX * 2 + i;
        int idx = y * width + x;

        float a = __half2float(pha[idx]);
        float fr = __half2float(fgr[0 * planeSize + idx]);
        float fg = __half2float(fgr[1 * planeSize + idx]);
        float fb = __half2float(fgr[2 * planeSize + idx]);

        // Alpha composite: fg * alpha + bg * (1 - alpha)
        R[i] = fminf(fmaxf(fr * a + bgRf * (1.0f - a), 0.0f), 1.0f) * 255.0f;
        G[i] = fminf(fmaxf(fg * a + bgGf * (1.0f - a), 0.0f), 1.0f) * 255.0f;
        B[i] = fminf(fmaxf(fb * a + bgBf * (1.0f - a), 0.0f), 1.0f) * 255.0f;
    }

    // BT.601 RGB→YUV
    auto toY = [](float r, float g, float b) -> uint8_t {
        float y = 16.0f + 65.481f * r / 255.0f + 128.553f * g / 255.0f + 24.966f * b / 255.0f;
        return static_cast<uint8_t>(fminf(fmaxf(y, 0.0f), 255.0f));
    };
    auto toU = [](float r, float g, float b) -> uint8_t {
        float u = 128.0f - 37.797f * r / 255.0f - 74.203f * g / 255.0f + 112.0f * b / 255.0f;
        return static_cast<uint8_t>(fminf(fmaxf(u, 0.0f), 255.0f));
    };
    auto toV = [](float r, float g, float b) -> uint8_t {
        float v = 128.0f + 112.0f * r / 255.0f - 93.786f * g / 255.0f - 18.214f * b / 255.0f;
        return static_cast<uint8_t>(fminf(fmaxf(v, 0.0f), 255.0f));
    };

    uint8_t Y0 = toY(R[0], G[0], B[0]);
    uint8_t Y1 = toY(R[1], G[1], B[1]);
    // Average U and V across the pixel pair
    uint8_t U = static_cast<uint8_t>((static_cast<int>(toU(R[0], G[0], B[0])) +
                                       static_cast<int>(toU(R[1], G[1], B[1]))) / 2);
    uint8_t Vv = static_cast<uint8_t>((static_cast<int>(toV(R[0], G[0], B[0])) +
                                        static_cast<int>(toV(R[1], G[1], B[1]))) / 2);

    int outOff = (y * halfW + macroX) * 4;
    yuyv[outOff + 0] = Y0;
    yuyv[outOff + 1] = U;
    yuyv[outOff + 2] = Y1;
    yuyv[outOff + 3] = Vv;
}

void launchCompositeToYuyv(const __half* d_fgr, const __half* d_pha,
                           uint8_t* d_yuyv, int width, int height,
                           uint8_t bgR, uint8_t bgG, uint8_t bgB,
                           cudaStream_t stream) {
    int halfW = width / 2;
    dim3 block(32, 8);
    dim3 grid((halfW + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    compositeToYuyvKernel<<<grid, block, 0, stream>>>(
        d_fgr, d_pha, d_yuyv, width, height, bgR, bgG, bgB);
}
