#include "pipeline.h"
#include "cuda_kernels.h"
#include <cstdio>
#include <cstring>
#include <time.h>
#include <linux/videodev2.h>

static std::string resolveResolutionTag(int w, int h) {
    if (w == 1920 && h == 1080) return "1080p";
    if (w == 1280 && h == 720) return "720p";
    if (w == 2560 && h == 1440) return "1440p";
    if (w == 3840 && h == 2160) return "2160p";
    return std::to_string(w) + "x" + std::to_string(h);
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

Pipeline::Pipeline(const PipelineConfig& cfg)
    : cfg_(cfg),
      capture_(cfg.inputDevice, cfg.width, cfg.height),
      bgRf_(cfg.bgR / 255.0f), bgGf_(cfg.bgG / 255.0f), bgBf_(cfg.bgB / 255.0f) {}

Pipeline::~Pipeline() {
    // Stop capture thread
    stopCapture_ = true;
    cvConsumed_.notify_all();
    cvReady_.notify_all();
    if (captureThread_.joinable())
        captureThread_.join();

    freeGpuMemory();
    if (stream_) cudaStreamDestroy(stream_);
    if (evStart_) cudaEventDestroy(evStart_);
    if (evStop_) cudaEventDestroy(evStop_);
    for (int s = 0; s < NUM_SLOTS; s++) {
        if (slotDoneEvent_[s]) cudaEventDestroy(slotDoneEvent_[s]);
        if (slots_[s].nvState) nvjpegJpegStateDestroy(slots_[s].nvState);
        if (slots_[s].nvStream) nvjpegJpegStreamDestroy(slots_[s].nvStream);
        if (slots_[s].nvDeviceBuf) nvjpegBufferDeviceDestroy(slots_[s].nvDeviceBuf);
        if (slots_[s].nvPinnedBuf) nvjpegBufferPinnedDestroy(slots_[s].nvPinnedBuf);
    }
    if (nvjpegParams_) nvjpegDecodeParamsDestroy(nvjpegParams_);
    if (nvjpegDecoder_) nvjpegDecoderDestroy(nvjpegDecoder_);
    if (nvjpegHandle_) nvjpegDestroy(nvjpegHandle_);
}

bool Pipeline::init() {
    if (!capture_.init()) return false;

    captureFmt_ = capture_.pixelFormat();
    if (captureFmt_ != V4L2_PIX_FMT_YUYV && captureFmt_ != V4L2_PIX_FMT_MJPEG) {
        char fourcc[5] = {};
        memcpy(fourcc, &captureFmt_, 4);
        fprintf(stderr, "Unsupported capture format: %s (need YUYV or MJPEG)\n", fourcc);
        return false;
    }

    cfg_.width = capture_.width();
    cfg_.height = capture_.height();
    fprintf(stderr, "Using resolution: %dx%d\n", cfg_.width, cfg_.height);

    if (captureFmt_ == V4L2_PIX_FMT_MJPEG) {
        // Init shared nvJPEG handles
        nvjpegStatus_t st;
        st = nvjpegCreateEx(NVJPEG_BACKEND_GPU_HYBRID, nullptr, nullptr, 0, &nvjpegHandle_);
        if (st != NVJPEG_STATUS_SUCCESS) {
            fprintf(stderr, "nvjpegCreateEx failed: %d\n", st);
            return false;
        }
        st = nvjpegDecoderCreate(nvjpegHandle_, NVJPEG_BACKEND_GPU_HYBRID, &nvjpegDecoder_);
        if (st != NVJPEG_STATUS_SUCCESS) {
            fprintf(stderr, "nvjpegDecoderCreate failed: %d\n", st);
            return false;
        }
        st = nvjpegDecodeParamsCreate(nvjpegHandle_, &nvjpegParams_);
        if (st != NVJPEG_STATUS_SUCCESS) {
            fprintf(stderr, "nvjpegDecodeParamsCreate failed: %d\n", st);
            return false;
        }
        nvjpegDecodeParamsSetOutputFormat(nvjpegParams_, NVJPEG_OUTPUT_RGBI);

        // Per-slot nvJPEG state
        for (int s = 0; s < NUM_SLOTS; s++) {
            auto& slot = slots_[s];
            st = nvjpegDecoderStateCreate(nvjpegHandle_, nvjpegDecoder_, &slot.nvState);
            if (st != NVJPEG_STATUS_SUCCESS) {
                fprintf(stderr, "nvjpegDecoderStateCreate[%d] failed: %d\n", s, st);
                return false;
            }
            st = nvjpegBufferPinnedCreate(nvjpegHandle_, nullptr, &slot.nvPinnedBuf);
            if (st != NVJPEG_STATUS_SUCCESS) {
                fprintf(stderr, "nvjpegBufferPinnedCreate[%d] failed: %d\n", s, st);
                return false;
            }
            st = nvjpegBufferDeviceCreate(nvjpegHandle_, nullptr, &slot.nvDeviceBuf);
            if (st != NVJPEG_STATUS_SUCCESS) {
                fprintf(stderr, "nvjpegBufferDeviceCreate[%d] failed: %d\n", s, st);
                return false;
            }
            st = nvjpegStateAttachPinnedBuffer(slot.nvState, slot.nvPinnedBuf);
            if (st != NVJPEG_STATUS_SUCCESS) {
                fprintf(stderr, "nvjpegStateAttachPinnedBuffer[%d] failed: %d\n", s, st);
                return false;
            }
            st = nvjpegStateAttachDeviceBuffer(slot.nvState, slot.nvDeviceBuf);
            if (st != NVJPEG_STATUS_SUCCESS) {
                fprintf(stderr, "nvjpegStateAttachDeviceBuffer[%d] failed: %d\n", s, st);
                return false;
            }
            st = nvjpegJpegStreamCreate(nvjpegHandle_, &slot.nvStream);
            if (st != NVJPEG_STATUS_SUCCESS) {
                fprintf(stderr, "nvjpegJpegStreamCreate[%d] failed: %d\n", s, st);
                return false;
            }
        }
        fprintf(stderr, "MJPEG capture — using nvJPEG GPU-hybrid decoder (double-buffered)\n");
    }

    // Auto-resolve model paths based on negotiated resolution
    if (cfg_.onnxPath.empty()) {
        std::string tag = resolveResolutionTag(cfg_.width, cfg_.height);
        cfg_.onnxPath = "models/rvm_" + tag + ".onnx";
        fprintf(stderr, "Auto-resolved ONNX model: %s\n", cfg_.onnxPath.c_str());
    }
    if (cfg_.planPath.empty()) {
        std::string tag = resolveResolutionTag(cfg_.width, cfg_.height);
        cfg_.planPath = "models/rvm_" + tag + ".plan";
        fprintf(stderr, "Auto-resolved TRT plan: %s\n", cfg_.planPath.c_str());
    }

    output_ = std::make_unique<V4L2Output>(cfg_.outputDevice, cfg_.width, cfg_.height, V4L2_PIX_FMT_YUYV);
    if (!output_->init()) return false;

    if (!engine_.loadOrBuild(cfg_.onnxPath, cfg_.planPath, cfg_.fp16,
                             cfg_.width, cfg_.height))
        return false;
    engine_.printBindings();

    CUDA_CHECK(cudaStreamCreate(&stream_));
    if (cfg_.benchmark) {
        CUDA_CHECK(cudaEventCreate(&evStart_));
        CUDA_CHECK(cudaEventCreate(&evStop_));
    }
    // Create slot-done events and seed them so first cudaEventSynchronize returns immediately
    for (int s = 0; s < NUM_SLOTS; s++) {
        CUDA_CHECK(cudaEventCreateWithFlags(&slotDoneEvent_[s], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(slotDoneEvent_[s], stream_));
    }

    if (!allocateGpuMemory()) return false;
    if (!capture_.startStreaming()) return false;

    // Start capture thread
    captureThread_ = std::thread(&Pipeline::captureThreadFunc, this);

    fprintf(stderr, "Pipeline initialized successfully (double-buffered)\n");
    return true;
}

bool Pipeline::allocateGpuMemory() {
    int W = cfg_.width;
    int H = cfg_.height;

    if (W < 1 || W > 8192 || H < 1 || H > 8192) {
        fprintf(stderr, "Invalid dimensions %dx%d (must be 1..8192)\n", W, H);
        return false;
    }
    size_t pixels = static_cast<size_t>(W) * H;

    rgbBytes_ = pixels * 3;
    yuyvBytes_ = pixels * 2;
    stagingSize_ = (rgbBytes_ > yuyvBytes_) ? rgbBytes_ : yuyvBytes_;
    size_t srcBytes = pixels * 3 * sizeof(float);
    size_t fgrBytes = pixels * 3 * sizeof(float);
    size_t phaBytes = pixels * 1 * sizeof(float);

    // Recurrent state dimensions (scaled by downsample ratio)
    int intH = static_cast<int>(H * cfg_.downsampleRatio);
    int intW = static_cast<int>(W * cfg_.downsampleRatio);
    int divs[] = {2, 4, 8, 16};
    int chs[] = {16, 20, 40, 64};
    for (int i = 0; i < 4; i++) {
        recDims_[i].ch = chs[i];
        recDims_[i].h = (intH + divs[i] - 1) / divs[i];
        recDims_[i].w = (intW + divs[i] - 1) / divs[i];
        recDims_[i].bytes = 1 * chs[i] * recDims_[i].h * recDims_[i].w * sizeof(float);
    }

    // Per-slot allocations (double-buffered inputs)
    for (int s = 0; s < NUM_SLOTS; s++) {
        auto& slot = slots_[s];
        CUDA_CHECK(cudaHostAlloc(&slot.h_staging, stagingSize_, cudaHostAllocDefault));
        if (captureFmt_ == V4L2_PIX_FMT_MJPEG) {
            CUDA_CHECK(cudaMalloc(&slot.d_input, rgbBytes_));
            memset(&slot.nvOutput, 0, sizeof(slot.nvOutput));
            slot.nvOutput.channel[0] = slot.d_input;
            slot.nvOutput.pitch[0] = W * 3;  // interleaved RGB
        } else {
            CUDA_CHECK(cudaMalloc(&slot.d_input, yuyvBytes_));
        }
        CUDA_CHECK(cudaMalloc(&slot.d_src, srcBytes));
    }

    // Single-buffered allocations (outputs, consumed before next frame)
    CUDA_CHECK(cudaHostAlloc(&h_output_, yuyvBytes_, cudaHostAllocWriteCombined));
    CUDA_CHECK(cudaMalloc(&d_fgr_, fgrBytes));
    CUDA_CHECK(cudaMalloc(&d_pha_, phaBytes));
    if (cfg_.alphaSmoothing < 1.0f) {
        CUDA_CHECK(cudaMalloc(&d_phaPrev_, phaBytes));
    }
    CUDA_CHECK(cudaMalloc(&d_outputYuyv_, yuyvBytes_));

    for (int s = 0; s < 2; s++) {
        for (int i = 0; i < 4; i++) {
            CUDA_CHECK(cudaMalloc(&d_rec_[s][i], recDims_[i].bytes));
            CUDA_CHECK(cudaMemset(d_rec_[s][i], 0, recDims_[i].bytes));
        }
    }

    // Guided filter scratch buffers (operates at 1/4 resolution)
    if (cfg_.refineAlpha) {
        guidedFilterInit(gfState_, W, H, 4);
        CUDA_CHECK(cudaGetLastError());
    }

    fprintf(stderr, "GPU memory allocated (double-buffered, float32 I/O)\n");
    return true;
}

void Pipeline::freeGpuMemory() {
    for (int s = 0; s < NUM_SLOTS; s++) {
        auto& slot = slots_[s];
        if (slot.h_staging) { cudaFreeHost(slot.h_staging); slot.h_staging = nullptr; }
        if (slot.d_input) { cudaFree(slot.d_input); slot.d_input = nullptr; }
        if (slot.d_src) { cudaFree(slot.d_src); slot.d_src = nullptr; }
    }
    if (h_output_) { cudaFreeHost(h_output_); h_output_ = nullptr; }
    if (d_fgr_) { cudaFree(d_fgr_); d_fgr_ = nullptr; }
    if (d_pha_) { cudaFree(d_pha_); d_pha_ = nullptr; }
    if (d_phaPrev_) { cudaFree(d_phaPrev_); d_phaPrev_ = nullptr; }
    if (d_outputYuyv_) { cudaFree(d_outputYuyv_); d_outputYuyv_ = nullptr; }
    for (int s = 0; s < 2; s++)
        for (int i = 0; i < 4; i++)
            if (d_rec_[s][i]) { cudaFree(d_rec_[s][i]); d_rec_[s][i] = nullptr; }
    guidedFilterFree(gfState_);
}

// ---------------------------------------------------------------------------
// Capture thread: V4L2 DQBUF + memcpy + CPU Huffman decode (MJPEG)
// Runs concurrently with GPU processing on the main thread.
// ---------------------------------------------------------------------------
void Pipeline::captureThreadFunc() {
    int slot = 0;

    while (!stopCapture_) {
        auto& s = slots_[slot];

        // Wait until this slot has been consumed by the main thread
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cvConsumed_.wait(lock, [&] { return !s.ready || stopCapture_.load(); });
            if (stopCapture_) break;
        }

        // Wait for GPU to finish with this slot's device buffers
        cudaEventSynchronize(slotDoneEvent_[slot]);

        // V4L2 capture (blocking, with poll timeout)
        size_t frameSize = 0;
        const uint8_t* mmapPtr = capture_.dequeueLatestFrame(frameSize);
        if (!mmapPtr) {
            std::lock_guard<std::mutex> lock(mtx_);
            captureError_ = true;
            s.ready = true;
            s.valid = false;
            cvReady_.notify_one();
            break;
        }

        // Copy frame data to pinned staging, requeue mmap buffer
        if (frameSize > stagingSize_) {
            capture_.requeueBuffer();
            std::lock_guard<std::mutex> lock(mtx_);
            s.ready = true;
            s.valid = false;
            cvReady_.notify_one();
            slot = 1 - slot;
            continue;
        }
        memcpy(s.h_staging, mmapPtr, frameSize);
        capture_.requeueBuffer();
        s.frameSize = frameSize;

        // CPU-phase decode (MJPEG only — this overlaps with GPU work)
        if (captureFmt_ == V4L2_PIX_FMT_MJPEG) {
            // Parse JPEG header
            nvjpegStatus_t st = nvjpegJpegStreamParse(nvjpegHandle_,
                s.h_staging, frameSize, 0, 0, s.nvStream);
            if (st != NVJPEG_STATUS_SUCCESS) {
                std::lock_guard<std::mutex> lock(mtx_);
                s.ready = true;
                s.valid = false;
                cvReady_.notify_one();
                slot = 1 - slot;
                continue;
            }

            // Validate dimensions
            unsigned int jpegW, jpegH;
            nvjpegJpegStreamGetFrameDimensions(s.nvStream, &jpegW, &jpegH);
            if (static_cast<int>(jpegW) != cfg_.width ||
                static_cast<int>(jpegH) != cfg_.height) {
                std::lock_guard<std::mutex> lock(mtx_);
                s.ready = true;
                s.valid = false;
                cvReady_.notify_one();
                slot = 1 - slot;
                continue;
            }

            // CPU Huffman decode (the main overlap win)
            st = nvjpegDecodeJpegHost(nvjpegHandle_, nvjpegDecoder_, s.nvState,
                                       nvjpegParams_, s.nvStream);
            if (st != NVJPEG_STATUS_SUCCESS) {
                std::lock_guard<std::mutex> lock(mtx_);
                s.ready = true;
                s.valid = false;
                cvReady_.notify_one();
                slot = 1 - slot;
                continue;
            }
        }

        // Signal slot ready
        {
            std::lock_guard<std::mutex> lock(mtx_);
            s.ready = true;
            s.valid = true;
            cvReady_.notify_one();
        }

        slot = 1 - slot;
    }
}

// ---------------------------------------------------------------------------
// Main-thread frame processing
// ---------------------------------------------------------------------------
bool Pipeline::processFrame() {
    struct timespec wallStart{}, wallEnd{};
    if (cfg_.benchmark) {
        cudaEventRecord(evStart_, stream_);
        clock_gettime(CLOCK_MONOTONIC, &wallStart);
    }

    // 1. Wait for capture thread to fill current slot
    auto& slot = slots_[processSlotIdx_];
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cvReady_.wait(lock, [&] { return slot.ready || captureError_; });
        if (captureError_) return false;
    }

    if (!slot.valid) {
        // Skip invalid frame, release slot for capture thread
        slot.ready = false;
        cvConsumed_.notify_one();
        processSlotIdx_ = 1 - processSlotIdx_;
        if (++consecutiveSkips_ > 30) {
            fprintf(stderr, "Too many consecutive skipped frames (%d), aborting\n", consecutiveSkips_);
            return false;
        }
        return true;
    }

    // 2. GPU-phase decode and preprocess
    if (captureFmt_ == V4L2_PIX_FMT_MJPEG) {
        // Transfer coefficients to GPU + GPU IDCT → slot.d_input (interleaved RGB)
        nvjpegStatus_t st = nvjpegDecodeJpegTransferToDevice(nvjpegHandle_, nvjpegDecoder_,
                                                              slot.nvState, slot.nvStream, stream_);
        if (st != NVJPEG_STATUS_SUCCESS) {
            fprintf(stderr, "nvjpegDecodeJpegTransferToDevice failed: %d\n", st);
            return false;
        }
        st = nvjpegDecodeJpegDevice(nvjpegHandle_, nvjpegDecoder_, slot.nvState,
                                     &slot.nvOutput, stream_);
        if (st != NVJPEG_STATUS_SUCCESS) {
            fprintf(stderr, "nvjpegDecodeJpegDevice failed: %d\n", st);
            return false;
        }
        launchRgbToFp32(slot.d_input, slot.d_src, cfg_.width, cfg_.height, stream_);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // YUYV: H2D transfer + preprocess kernel
        CUDA_CHECK(cudaMemcpyAsync(slot.d_input, slot.h_staging, yuyvBytes_,
                                   cudaMemcpyHostToDevice, stream_));
        launchYuyvToRgbFp32(slot.d_input, slot.d_src, cfg_.width, cfg_.height, stream_);
        CUDA_CHECK(cudaGetLastError());
    }

    // Record event: GPU queued all reads from this slot's input buffers
    CUDA_CHECK(cudaEventRecord(slotDoneEvent_[processSlotIdx_], stream_));

    // Release slot for capture thread (it can start refilling while GPU runs)
    int consumedSlot = processSlotIdx_;
    slot.ready = false;
    cvConsumed_.notify_one();
    processSlotIdx_ = 1 - processSlotIdx_;

    // Reset skip counter
    consecutiveSkips_ = 0;

    // 3. TensorRT inference (reads slot.d_src, which is in-flight on stream)
    auto* ctx = engine_.context();
    int next = 1 - recIdx_;

    ctx->setTensorAddress("src", slots_[consumedSlot].d_src);
    ctx->setTensorAddress("r1i", d_rec_[recIdx_][0]);
    ctx->setTensorAddress("r2i", d_rec_[recIdx_][1]);
    ctx->setTensorAddress("r3i", d_rec_[recIdx_][2]);
    ctx->setTensorAddress("r4i", d_rec_[recIdx_][3]);
    ctx->setTensorAddress("fgr", d_fgr_);
    ctx->setTensorAddress("pha", d_pha_);
    ctx->setTensorAddress("r1o", d_rec_[next][0]);
    ctx->setTensorAddress("r2o", d_rec_[next][1]);
    ctx->setTensorAddress("r3o", d_rec_[next][2]);
    ctx->setTensorAddress("r4o", d_rec_[next][3]);

    if (!ctx->enqueueV3(stream_)) {
        fprintf(stderr, "TensorRT enqueueV3 failed\n");
        return false;
    }
    recIdx_ = next;

    // 3b. Periodic recurrent state reset to prevent drift
    if (cfg_.resetInterval > 0 && ++framesSinceReset_ >= cfg_.resetInterval) {
        for (int i = 0; i < 4; i++)
            cudaMemsetAsync(d_rec_[recIdx_][i], 0, recDims_[i].bytes, stream_);
        framesSinceReset_ = 0;
    }

    // 3c. Alpha temporal smoothing
    if (cfg_.alphaSmoothing < 1.0f) {
        if (firstFrame_) {
            CUDA_CHECK(cudaMemcpyAsync(d_phaPrev_, d_pha_,
                static_cast<size_t>(cfg_.width) * cfg_.height * sizeof(float),
                cudaMemcpyDeviceToDevice, stream_));
            firstFrame_ = false;
        } else {
            launchAlphaEma(d_pha_, d_phaPrev_, cfg_.width, cfg_.height,
                           cfg_.alphaSmoothing, stream_);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // 3d. Guided filter: refine alpha using high-res RGB as guide
    if (cfg_.refineAlpha) {
        launchGuidedFilterAlpha(gfState_, slots_[consumedSlot].d_input, d_pha_,
                                 cfg_.gfRadius, cfg_.gfEps, cfg_.perfLevel, stream_);
        CUDA_CHECK(cudaGetLastError());
    }

    // 3e + 4. Despill and composite + color convert
    if (cfg_.perfLevel >= 1) {
        // Fused: despill + composite + YUYV in one kernel
        launchDespillCompositeToYuyv(d_fgr_, d_pha_, d_outputYuyv_,
                                      cfg_.width, cfg_.height,
                                      bgRf_, bgGf_, bgBf_,
                                      cfg_.despillStrength, stream_);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Separate kernels (v0.3 baseline)
        if (cfg_.despillStrength > 0.0f) {
            launchDespill(d_fgr_, d_pha_, cfg_.width, cfg_.height,
                          bgRf_, bgGf_, bgBf_,
                          cfg_.despillStrength, stream_);
            CUDA_CHECK(cudaGetLastError());
        }
        launchCompositeToYuyv(d_fgr_, d_pha_, d_outputYuyv_,
                              cfg_.width, cfg_.height,
                              bgRf_, bgGf_, bgBf_, stream_);
        CUDA_CHECK(cudaGetLastError());
    }

    // 5. Download and write
    CUDA_CHECK(cudaMemcpyAsync(h_output_, d_outputYuyv_, yuyvBytes_,
                               cudaMemcpyDeviceToHost, stream_));

    if (cfg_.benchmark)
        cudaEventRecord(evStop_, stream_);

    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (!output_->writeFrame(h_output_, yuyvBytes_)) {
        fprintf(stderr, "Failed to write frame to output device\n");
        return false;
    }

    if (cfg_.benchmark) {
        cudaEventSynchronize(evStop_);
        cudaEventElapsedTime(&lastFrameMs_, evStart_, evStop_);
        clock_gettime(CLOCK_MONOTONIC, &wallEnd);
        lastWallMs_ = (wallEnd.tv_sec - wallStart.tv_sec) * 1000.0f
                     + (wallEnd.tv_nsec - wallStart.tv_nsec) / 1e6f;
    }

    return true;
}
