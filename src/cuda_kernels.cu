#include "cuda_kernels.h"

// ---------------------------------------------------------------------------
// YUYV → RGB FP32 BCHW [0,1]
// YUYV packs 2 pixels in 4 bytes: [Y0, U, Y1, V]
// ---------------------------------------------------------------------------
__global__ void yuyvToRgbFp32Kernel(const uint8_t* __restrict__ yuyv,
                                     float* __restrict__ rgb,
                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixIdx = y * width + x;
    int macroX = x / 2;
    int byteOff = (y * width / 2 + macroX) * 4;

    uint8_t Y = yuyv[byteOff + (x & 1) * 2];
    uint8_t U = yuyv[byteOff + 1];
    uint8_t V = yuyv[byteOff + 3];

    // #8: Limited-range BT.601 conversion (Y: 16-235, UV: 16-240)
    float fY = (static_cast<float>(Y) - 16.0f) * (255.0f / 219.0f);
    float fU = (static_cast<float>(U) - 128.0f) * (255.0f / 224.0f);
    float fV = (static_cast<float>(V) - 128.0f) * (255.0f / 224.0f);

    float R = fminf(fmaxf(fY + 1.402f * fV, 0.0f), 255.0f) / 255.0f;
    float G = fminf(fmaxf(fY - 0.344136f * fU - 0.714136f * fV, 0.0f), 255.0f) / 255.0f;
    float B = fminf(fmaxf(fY + 1.772f * fU, 0.0f), 255.0f) / 255.0f;

    int planeSize = width * height;
    rgb[0 * planeSize + pixIdx] = R;
    rgb[1 * planeSize + pixIdx] = G;
    rgb[2 * planeSize + pixIdx] = B;
}

void launchYuyvToRgbFp32(const uint8_t* d_yuyv, float* d_rgb,
                          int width, int height, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    yuyvToRgbFp32Kernel<<<grid, block, 0, stream>>>(d_yuyv, d_rgb, width, height);
}

// ---------------------------------------------------------------------------
// RGB uint8 HWC → RGB FP32 BCHW [0,1]
// ---------------------------------------------------------------------------
__global__ void rgbToFp32Kernel(const uint8_t* __restrict__ rgb_u8,
                                 float* __restrict__ rgb_fp32,
                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixIdx = y * width + x;
    int srcIdx = pixIdx * 3;
    int planeSize = width * height;

    rgb_fp32[0 * planeSize + pixIdx] = rgb_u8[srcIdx + 0] / 255.0f;
    rgb_fp32[1 * planeSize + pixIdx] = rgb_u8[srcIdx + 1] / 255.0f;
    rgb_fp32[2 * planeSize + pixIdx] = rgb_u8[srcIdx + 2] / 255.0f;
}

void launchRgbToFp32(const uint8_t* d_rgb_u8, float* d_rgb_fp32,
                      int width, int height, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    rgbToFp32Kernel<<<grid, block, 0, stream>>>(d_rgb_u8, d_rgb_fp32, width, height);
}

// ---------------------------------------------------------------------------
// Composite FGR+PHA → YUYV (float32 inputs)
// Processes 2 horizontal pixels per thread (YUYV macro-pixel).
// ---------------------------------------------------------------------------
__global__ void compositeToYuyvKernel(const float* __restrict__ fgr,
                                       const float* __restrict__ pha,
                                       uint8_t* __restrict__ yuyv,
                                       int width, int height,
                                       uint8_t bgR, uint8_t bgG, uint8_t bgB) {
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

        float a = pha[idx];
        float fr = fgr[0 * planeSize + idx];
        float fg = fgr[1 * planeSize + idx];
        float fb = fgr[2 * planeSize + idx];

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

void launchCompositeToYuyv(const float* d_fgr, const float* d_pha,
                           uint8_t* d_yuyv, int width, int height,
                           uint8_t bgR, uint8_t bgG, uint8_t bgB,
                           cudaStream_t stream) {
    int halfW = width / 2;
    dim3 block(32, 8);
    dim3 grid((halfW + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    compositeToYuyvKernel<<<grid, block, 0, stream>>>(
        d_fgr, d_pha, d_yuyv, width, height, bgR, bgG, bgB);
}
