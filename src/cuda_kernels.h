#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// YUYV (packed) → RGB float32 BCHW normalized [0,1]
void launchYuyvToRgbFp32(const uint8_t* d_yuyv, float* d_rgb,
                          int width, int height, cudaStream_t stream);

// RGB uint8 (interleaved, HWC) → RGB float32 BCHW normalized [0,1]
void launchRgbToFp32(const uint8_t* d_rgb_u8, float* d_rgb_fp32,
                      int width, int height, cudaStream_t stream);

// Alpha EMA temporal smoothing: smoothed = prev*(1-factor) + current*factor
// Updates both pha (current) and phaPrev in-place
void launchAlphaEma(float* d_pha, float* d_phaPrev,
                    int width, int height, float factor, cudaStream_t stream);

// Fast guided filter for alpha refinement (He & Sun 2015)
struct GuidedFilterState {
    float* d_guide_lr = nullptr;   // luminance at 1/s res
    float* d_alpha_lr = nullptr;   // alpha at 1/s res
    float* d_mean_I = nullptr;
    float* d_mean_p = nullptr;
    float* d_mean_Ip = nullptr;
    float* d_mean_II = nullptr;
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_mean_a = nullptr;
    float* d_mean_b = nullptr;
    float* d_tmp = nullptr;        // scratch for separable box filter
    int lrW = 0, lrH = 0;
    int fullW = 0, fullH = 0;
    int subsample = 4;
};

void guidedFilterInit(GuidedFilterState& state, int fullW, int fullH, int subsample);
void guidedFilterFree(GuidedFilterState& state);
void launchGuidedFilterAlpha(GuidedFilterState& state,
                              const uint8_t* d_rgb,
                              float* d_alpha,
                              int radius, float eps,
                              cudaStream_t stream);

// Despill: suppress background color contamination in FGR at semi-transparent edges
void launchDespill(float* d_fgr, const float* d_pha,
                   int width, int height,
                   float bgR, float bgG, float bgB,
                   float strength, cudaStream_t stream);

// RVM outputs (fgr, pha) → composite with background → YUYV output
// fgr: [1,3,H,W] float32, pha: [1,1,H,W] float32
// output: YUYV packed uint8
void launchCompositeToYuyv(const float* d_fgr, const float* d_pha,
                           uint8_t* d_yuyv, int width, int height,
                           float bgRf, float bgGf, float bgBf,
                           cudaStream_t stream);
