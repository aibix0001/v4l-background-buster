#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// YUYV (packed) → RGB float16 BCHW normalized [0,1]
void launchYuyvToRgbFp16(const uint8_t* d_yuyv, __half* d_rgb,
                          int width, int height, cudaStream_t stream);

// RVM outputs (fgr, pha) → composite with background → YUYV output
// fgr: [1,3,H,W] float16, pha: [1,1,H,W] float16
// output: YUYV packed uint8
void launchCompositeToYuyv(const __half* d_fgr, const __half* d_pha,
                           uint8_t* d_yuyv, int width, int height,
                           uint8_t bgR, uint8_t bgG, uint8_t bgB,
                           cudaStream_t stream);
