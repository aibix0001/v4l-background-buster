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

// RVM outputs (fgr, pha) → composite with background → YUYV output
// fgr: [1,3,H,W] float32, pha: [1,1,H,W] float32
// output: YUYV packed uint8
void launchCompositeToYuyv(const float* d_fgr, const float* d_pha,
                           uint8_t* d_yuyv, int width, int height,
                           uint8_t bgR, uint8_t bgG, uint8_t bgB,
                           cudaStream_t stream);
