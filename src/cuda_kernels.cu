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
// Adaptive alpha EMA temporal smoothing
// Edge pixels (alpha ~0.5) get heavy smoothing to reduce shimmer.
// Confident pixels (alpha ~0 or ~1) get minimal smoothing to avoid lag.
// Large frame-to-frame changes (motion) also reduce smoothing.
// ---------------------------------------------------------------------------
__global__ void alphaEmaKernel(float* __restrict__ pha,
                                float* __restrict__ phaPrev,
                                int width, int height, float baseFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float current = pha[idx];
    float prev = phaPrev[idx];

    // Confidence: how far from 0.5 (edge). Range [0, 0.5].
    float confidence = fabsf(current - 0.5f);

    // High confidence (near 0 or 1) → factor close to 1.0 (almost no smoothing)
    // Low confidence (near 0.5, edge) → factor = baseFactor (heavy smoothing)
    float factor = baseFactor + (1.0f - baseFactor) * confidence * 2.0f;

    // Reduce smoothing on large frame-to-frame change (motion)
    float delta = fabsf(current - prev);
    if (delta > 0.3f)
        factor = fminf(factor + 0.3f, 1.0f);

    float smoothed = prev * (1.0f - factor) + current * factor;
    pha[idx] = smoothed;
    phaPrev[idx] = smoothed;
}

void launchAlphaEma(float* d_pha, float* d_phaPrev,
                    int width, int height, float factor, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    alphaEmaKernel<<<grid, block, 0, stream>>>(d_pha, d_phaPrev, width, height, factor);
}

// ---------------------------------------------------------------------------
// Fast Guided Filter (He & Sun 2015) for alpha refinement
// ---------------------------------------------------------------------------

// Fused: RGB uint8 HWC → float luminance + nearest-neighbor downsample
__global__ void rgbToLuminanceDownsampleKernel(const uint8_t* __restrict__ rgb,
                                                float* __restrict__ lum,
                                                int fullW, int fullH,
                                                int lrW, int lrH, int subsample) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= lrW || y >= lrH) return;

    int sx = x * subsample;
    int sy = y * subsample;
    int srcIdx = (sy * fullW + sx) * 3;

    float r = rgb[srcIdx + 0] / 255.0f;
    float g = rgb[srcIdx + 1] / 255.0f;
    float b = rgb[srcIdx + 2] / 255.0f;

    lum[y * lrW + x] = 0.299f * r + 0.587f * g + 0.114f * b;
}

// Nearest-neighbor downsample for float buffer
__global__ void downsampleNearestKernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int fullW, int lrW, int lrH, int subsample) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= lrW || y >= lrH) return;

    output[y * lrW + x] = input[y * subsample * fullW + x * subsample];
}

// Separable box filter — horizontal pass (per-pixel, O(radius) per pixel)
__global__ void boxFilterHorizontalKernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int ksize = 2 * radius + 1;
    float sum = 0.0f;
    for (int dx = -radius; dx <= radius; dx++) {
        int sx = min(max(x + dx, 0), width - 1);
        sum += input[y * width + sx];
    }
    output[y * width + x] = sum / static_cast<float>(ksize);
}

// Separable box filter — vertical pass (per-pixel, O(radius) per pixel)
__global__ void boxFilterVerticalKernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int ksize = 2 * radius + 1;
    float sum = 0.0f;
    for (int dy = -radius; dy <= radius; dy++) {
        int sy = min(max(y + dy, 0), height - 1);
        sum += input[sy * width + x];
    }
    output[y * width + x] = sum / static_cast<float>(ksize);
}

// Elementwise multiply: out[i] = a[i] * b[i]
__global__ void elementwiseMultiplyKernel(const float* __restrict__ a,
                                           const float* __restrict__ b,
                                           float* __restrict__ out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] * b[i];
}

// Compute guided filter coefficients a and b from means
__global__ void computeCoefficientsKernel(const float* __restrict__ mean_I,
                                           const float* __restrict__ mean_p,
                                           const float* __restrict__ mean_Ip,
                                           const float* __restrict__ mean_II,
                                           float* __restrict__ a,
                                           float* __restrict__ b,
                                           int width, int height, float eps) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float mI = mean_I[idx];
    float mp = mean_p[idx];
    float var_I = mean_II[idx] - mI * mI;
    float cov_Ip = mean_Ip[idx] - mI * mp;

    a[idx] = cov_Ip / (var_I + eps);
    b[idx] = mp - a[idx] * mI;
}

// Bilinear upsample a,b from low-res and apply: output = a_up * I_full + b_up
__global__ void upsampleAndApplyKernel(const float* __restrict__ mean_a,
                                        const float* __restrict__ mean_b,
                                        const uint8_t* __restrict__ rgb,
                                        float* __restrict__ alpha,
                                        int lrW, int lrH,
                                        int fullW, int fullH, int subsample) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= fullW || y >= fullH) return;

    // Bilinear coordinates in low-res space
    float fx = (static_cast<float>(x) + 0.5f) / subsample - 0.5f;
    float fy = (static_cast<float>(y) + 0.5f) / subsample - 0.5f;
    int x0 = max(0, static_cast<int>(floorf(fx)));
    int y0 = max(0, static_cast<int>(floorf(fy)));
    int x1 = min(x0 + 1, lrW - 1);
    int y1 = min(y0 + 1, lrH - 1);
    float wx = fx - x0;
    float wy = fy - y0;

    float w00 = (1.0f - wx) * (1.0f - wy);
    float w10 = wx * (1.0f - wy);
    float w01 = (1.0f - wx) * wy;
    float w11 = wx * wy;

    float a = mean_a[y0*lrW+x0]*w00 + mean_a[y0*lrW+x1]*w10
            + mean_a[y1*lrW+x0]*w01 + mean_a[y1*lrW+x1]*w11;
    float bv = mean_b[y0*lrW+x0]*w00 + mean_b[y0*lrW+x1]*w10
             + mean_b[y1*lrW+x0]*w01 + mean_b[y1*lrW+x1]*w11;

    // Compute luminance from full-res RGB
    int srcIdx = (y * fullW + x) * 3;
    float lum = (0.299f * rgb[srcIdx] + 0.587f * rgb[srcIdx+1]
               + 0.114f * rgb[srcIdx+2]) / 255.0f;

    alpha[y * fullW + x] = fminf(fmaxf(a * lum + bv, 0.0f), 1.0f);
}

// Init/free/launch for guided filter
void guidedFilterInit(GuidedFilterState& state, int fullW, int fullH, int subsample) {
    state.fullW = fullW;
    state.fullH = fullH;
    state.subsample = subsample;
    state.lrW = fullW / subsample;
    state.lrH = fullH / subsample;

    size_t lrBytes = static_cast<size_t>(state.lrW) * state.lrH * sizeof(float);

    cudaMalloc(&state.d_guide_lr, lrBytes);
    cudaMalloc(&state.d_alpha_lr, lrBytes);
    cudaMalloc(&state.d_mean_I, lrBytes);
    cudaMalloc(&state.d_mean_p, lrBytes);
    cudaMalloc(&state.d_mean_Ip, lrBytes);
    cudaMalloc(&state.d_mean_II, lrBytes);
    cudaMalloc(&state.d_a, lrBytes);
    cudaMalloc(&state.d_b, lrBytes);
    cudaMalloc(&state.d_mean_a, lrBytes);
    cudaMalloc(&state.d_mean_b, lrBytes);
    cudaMalloc(&state.d_tmp, lrBytes);
}

void guidedFilterFree(GuidedFilterState& state) {
    cudaFree(state.d_guide_lr);
    cudaFree(state.d_alpha_lr);
    cudaFree(state.d_mean_I);
    cudaFree(state.d_mean_p);
    cudaFree(state.d_mean_Ip);
    cudaFree(state.d_mean_II);
    cudaFree(state.d_a);
    cudaFree(state.d_b);
    cudaFree(state.d_mean_a);
    cudaFree(state.d_mean_b);
    cudaFree(state.d_tmp);
    state = {};
}

void launchGuidedFilterAlpha(GuidedFilterState& state,
                              const uint8_t* d_rgb,
                              float* d_alpha,
                              int radius, float eps,
                              cudaStream_t stream) {
    int lrW = state.lrW, lrH = state.lrH;
    int fullW = state.fullW, fullH = state.fullH;
    int s = state.subsample;
    int lrN = lrW * lrH;

    dim3 block(32, 8);
    dim3 lrGrid((lrW + 31) / 32, (lrH + 7) / 8);
    dim3 fullGrid((fullW + 31) / 32, (fullH + 7) / 8);

    // 1. Downsample RGB → luminance at 1/s res
    rgbToLuminanceDownsampleKernel<<<lrGrid, block, 0, stream>>>(
        d_rgb, state.d_guide_lr, fullW, fullH, lrW, lrH, s);

    // 2. Downsample alpha to 1/s res
    downsampleNearestKernel<<<lrGrid, block, 0, stream>>>(
        d_alpha, state.d_alpha_lr, fullW, lrW, lrH, s);

    // 3. Box filter I → mean_I
    boxFilterHorizontalKernel<<<lrGrid, block, 0, stream>>>(
        state.d_guide_lr, state.d_tmp, lrW, lrH, radius);
    boxFilterVerticalKernel<<<lrGrid, block, 0, stream>>>(
        state.d_tmp, state.d_mean_I, lrW, lrH, radius);

    // 4. Box filter p → mean_p
    boxFilterHorizontalKernel<<<lrGrid, block, 0, stream>>>(
        state.d_alpha_lr, state.d_tmp, lrW, lrH, radius);
    boxFilterVerticalKernel<<<lrGrid, block, 0, stream>>>(
        state.d_tmp, state.d_mean_p, lrW, lrH, radius);

    // 5. Compute I*p → d_mean_Ip, then box filter
    int blockN = 256;
    int gridN = (lrN + blockN - 1) / blockN;
    elementwiseMultiplyKernel<<<gridN, blockN, 0, stream>>>(
        state.d_guide_lr, state.d_alpha_lr, state.d_mean_Ip, lrN);
    boxFilterHorizontalKernel<<<lrGrid, block, 0, stream>>>(
        state.d_mean_Ip, state.d_tmp, lrW, lrH, radius);
    boxFilterVerticalKernel<<<lrGrid, block, 0, stream>>>(
        state.d_tmp, state.d_mean_Ip, lrW, lrH, radius);

    // 6. Compute I*I → d_mean_II, then box filter
    elementwiseMultiplyKernel<<<gridN, blockN, 0, stream>>>(
        state.d_guide_lr, state.d_guide_lr, state.d_mean_II, lrN);
    boxFilterHorizontalKernel<<<lrGrid, block, 0, stream>>>(
        state.d_mean_II, state.d_tmp, lrW, lrH, radius);
    boxFilterVerticalKernel<<<lrGrid, block, 0, stream>>>(
        state.d_tmp, state.d_mean_II, lrW, lrH, radius);

    // 7. Compute coefficients a, b
    computeCoefficientsKernel<<<lrGrid, block, 0, stream>>>(
        state.d_mean_I, state.d_mean_p, state.d_mean_Ip, state.d_mean_II,
        state.d_a, state.d_b, lrW, lrH, eps);

    // 8. Box filter a → mean_a
    boxFilterHorizontalKernel<<<lrGrid, block, 0, stream>>>(
        state.d_a, state.d_tmp, lrW, lrH, radius);
    boxFilterVerticalKernel<<<lrGrid, block, 0, stream>>>(
        state.d_tmp, state.d_mean_a, lrW, lrH, radius);

    // 9. Box filter b → mean_b
    boxFilterHorizontalKernel<<<lrGrid, block, 0, stream>>>(
        state.d_b, state.d_tmp, lrW, lrH, radius);
    boxFilterVerticalKernel<<<lrGrid, block, 0, stream>>>(
        state.d_tmp, state.d_mean_b, lrW, lrH, radius);

    // 10. Upsample and apply: alpha = a_up * I_full + b_up
    upsampleAndApplyKernel<<<fullGrid, block, 0, stream>>>(
        state.d_mean_a, state.d_mean_b, d_rgb, d_alpha,
        lrW, lrH, fullW, fullH, s);
}

// ---------------------------------------------------------------------------
// Despill: suppress background color contamination at semi-transparent edges
// ---------------------------------------------------------------------------
__global__ void despillKernel(float* __restrict__ fgr,
                               const float* __restrict__ pha,
                               int width, int height,
                               float bgR, float bgG, float bgB,
                               float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float alpha = pha[idx];

    // Only process semi-transparent pixels (edges)
    if (alpha >= 0.95f || alpha <= 0.05f) return;

    int ps = width * height;
    float r = fgr[0 * ps + idx];
    float g = fgr[1 * ps + idx];
    float b = fgr[2 * ps + idx];

    float limit;
    if (bgG > bgR && bgG > bgB) {
        // Green-dominant background: suppress green spill
        limit = fmaxf(r, b);
        if (g > limit)
            g = limit + (g - limit) * (1.0f - strength);
    } else if (bgB > bgR && bgB > bgG) {
        // Blue-dominant background: suppress blue spill
        limit = fmaxf(r, g);
        if (b > limit)
            b = limit + (b - limit) * (1.0f - strength);
    } else {
        // Red-dominant background: suppress red spill
        limit = fmaxf(g, b);
        if (r > limit)
            r = limit + (r - limit) * (1.0f - strength);
    }

    fgr[0 * ps + idx] = r;
    fgr[1 * ps + idx] = g;
    fgr[2 * ps + idx] = b;
}

void launchDespill(float* d_fgr, const float* d_pha,
                   int width, int height,
                   float bgR, float bgG, float bgB,
                   float strength, cudaStream_t stream) {
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    despillKernel<<<grid, block, 0, stream>>>(d_fgr, d_pha, width, height,
                                               bgR, bgG, bgB, strength);
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
