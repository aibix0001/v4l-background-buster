#include "pipeline.h"
#include "cuda_kernels.h"
#include <cstdio>
#include <cstring>
#include <time.h>
#include <linux/videodev2.h>

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
      capture_(cfg.inputDevice, cfg.width, cfg.height) {}

Pipeline::~Pipeline() {
    freeGpuMemory();
    if (stream_) cudaStreamDestroy(stream_);
    if (evStart_) cudaEventDestroy(evStart_);
    if (evStop_) cudaEventDestroy(evStop_);
    if (tjHandle_) tjDestroy(tjHandle_);
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
        tjHandle_ = tjInitDecompress();
        if (!tjHandle_) {
            fprintf(stderr, "Failed to init turbojpeg decompressor\n");
            return false;
        }
        fprintf(stderr, "MJPEG capture — using turbojpeg for decode\n");
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

    if (!allocateGpuMemory()) return false;
    if (!capture_.startStreaming()) return false;

    fprintf(stderr, "Pipeline initialized successfully\n");
    return true;
}

bool Pipeline::allocateGpuMemory() {
    int W = cfg_.width;
    int H = cfg_.height;

    // F2: Validate dimensions and use size_t arithmetic to prevent overflow
    if (W < 1 || W > 8192 || H < 1 || H > 8192) {
        fprintf(stderr, "Invalid dimensions %dx%d (must be 1..8192)\n", W, H);
        return false;
    }
    size_t pixels = static_cast<size_t>(W) * H;

    rgbBytes_ = pixels * 3;
    yuyvBytes_ = pixels * 2;
    size_t srcBytes = pixels * 3 * sizeof(float);  // RGB FP32 BCHW
    size_t fgrBytes = pixels * 3 * sizeof(float);
    size_t phaBytes = pixels * 1 * sizeof(float);

    // Recurrent state dimensions (from simplified model's static shapes)
    int intH = H / 4;
    int intW = W / 4;
    int divs[] = {2, 4, 8, 16};
    int chs[] = {16, 20, 40, 64};
    for (int i = 0; i < 4; i++) {
        recDims_[i].ch = chs[i];
        recDims_[i].h = (intH + divs[i] - 1) / divs[i];
        recDims_[i].w = (intW + divs[i] - 1) / divs[i];
        recDims_[i].bytes = 1 * chs[i] * recDims_[i].h * recDims_[i].w * sizeof(float);
    }

    CUDA_CHECK(cudaHostAlloc(&h_rgb_, rgbBytes_, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_output_, yuyvBytes_, cudaHostAllocDefault));

    // #14: Only allocate the input buffer needed for the negotiated format
    if (captureFmt_ == V4L2_PIX_FMT_MJPEG) {
        CUDA_CHECK(cudaMalloc(&d_inputRgb_, rgbBytes_));
    } else {
        CUDA_CHECK(cudaMalloc(&d_inputYuyv_, yuyvBytes_));
    }
    CUDA_CHECK(cudaMalloc(&d_src_, srcBytes));
    CUDA_CHECK(cudaMalloc(&d_fgr_, fgrBytes));
    CUDA_CHECK(cudaMalloc(&d_pha_, phaBytes));
    CUDA_CHECK(cudaMalloc(&d_outputYuyv_, yuyvBytes_));

    for (int s = 0; s < 2; s++) {
        for (int i = 0; i < 4; i++) {
            CUDA_CHECK(cudaMalloc(&d_rec_[s][i], recDims_[i].bytes));
            CUDA_CHECK(cudaMemset(d_rec_[s][i], 0, recDims_[i].bytes));
        }
    }

    fprintf(stderr, "GPU memory allocated (float32 I/O)\n");
    return true;
}

void Pipeline::freeGpuMemory() {
    if (h_rgb_) { cudaFreeHost(h_rgb_); h_rgb_ = nullptr; }
    if (h_output_) { cudaFreeHost(h_output_); h_output_ = nullptr; }
    if (d_inputRgb_) { cudaFree(d_inputRgb_); d_inputRgb_ = nullptr; }
    if (d_inputYuyv_) { cudaFree(d_inputYuyv_); d_inputYuyv_ = nullptr; }
    if (d_src_) { cudaFree(d_src_); d_src_ = nullptr; }
    if (d_fgr_) { cudaFree(d_fgr_); d_fgr_ = nullptr; }
    if (d_pha_) { cudaFree(d_pha_); d_pha_ = nullptr; }
    if (d_outputYuyv_) { cudaFree(d_outputYuyv_); d_outputYuyv_ = nullptr; }
    for (int s = 0; s < 2; s++)
        for (int i = 0; i < 4; i++)
            if (d_rec_[s][i]) { cudaFree(d_rec_[s][i]); d_rec_[s][i] = nullptr; }
}

bool Pipeline::processFrame() {
    struct timespec wallStart{}, wallEnd{};
    if (cfg_.benchmark) {
        cudaEventRecord(evStart_, stream_);
        clock_gettime(CLOCK_MONOTONIC, &wallStart);
    }

    // 1. Capture (#6: drain stale frames, keep latest)
    size_t frameSize = 0;
    const uint8_t* mmapPtr = capture_.dequeueLatestFrame(frameSize);
    if (!mmapPtr) return false;

    // 2. Decode and upload
    if (captureFmt_ == V4L2_PIX_FMT_MJPEG) {
        // F14: Copy JPEG data to h_output_ (unused until step 5) to avoid
        // const_cast on the mmap'd V4L2 buffer. Requeue mmap buffer early.
        if (frameSize > yuyvBytes_) {
            fprintf(stderr, "MJPEG frame too large: %zu > %zu — skipping\n", frameSize, yuyvBytes_);
            capture_.requeueBuffer();
            if (++consecutiveSkips_ > 30) {
                fprintf(stderr, "Too many consecutive skipped frames (%d), aborting\n", consecutiveSkips_);
                return false;
            }
            return true;
        }
        memcpy(h_output_, mmapPtr, frameSize);
        capture_.requeueBuffer();

        auto* jpegBuf = h_output_;
        int jpegSubsamp, jpegW, jpegH;
        if (tjDecompressHeader2(tjHandle_, jpegBuf, frameSize,
                                &jpegW, &jpegH, &jpegSubsamp) != 0) {
            // Corrupt JPEG frame — skip it (common on first frames)
            if (++consecutiveSkips_ > 30) {
                fprintf(stderr, "Too many consecutive corrupt frames (%d), aborting\n", consecutiveSkips_);
                return false;
            }
            return true;  // not fatal, just skip
        }
        // F5: Validate JPEG dimensions match expected resolution
        if (jpegW != cfg_.width || jpegH != cfg_.height) {
            fprintf(stderr, "JPEG frame size mismatch: got %dx%d, expected %dx%d — skipping\n",
                    jpegW, jpegH, cfg_.width, cfg_.height);
            if (++consecutiveSkips_ > 30) {
                fprintf(stderr, "Too many consecutive skipped frames (%d), aborting\n", consecutiveSkips_);
                return false;
            }
            return true;  // skip, not fatal
        }
        if (tjDecompress2(tjHandle_, jpegBuf, frameSize,
                          h_rgb_, cfg_.width, 0, cfg_.height,
                          TJPF_RGB, TJFLAG_FASTDCT) != 0) {
            if (++consecutiveSkips_ > 30) {
                fprintf(stderr, "Too many consecutive corrupt frames (%d), aborting\n", consecutiveSkips_);
                return false;
            }
            return true;  // skip corrupt frame
        }

        CUDA_CHECK(cudaMemcpyAsync(d_inputRgb_, h_rgb_, rgbBytes_,
                                   cudaMemcpyHostToDevice, stream_));
        launchRgbToFp32(d_inputRgb_, d_src_, cfg_.width, cfg_.height, stream_);
        CUDA_CHECK(cudaGetLastError());  // F9: check kernel launch
    } else {
        // F4: Validate frame size before memcpy
        if (frameSize > yuyvBytes_) {
            fprintf(stderr, "YUYV frame too large: %zu > %zu — skipping\n", frameSize, yuyvBytes_);
            capture_.requeueBuffer();
            if (++consecutiveSkips_ > 30) {
                fprintf(stderr, "Too many consecutive skipped frames (%d), aborting\n", consecutiveSkips_);
                return false;
            }
            return true;
        }
        memcpy(h_rgb_, mmapPtr, frameSize);
        capture_.requeueBuffer();

        CUDA_CHECK(cudaMemcpyAsync(d_inputYuyv_, h_rgb_, yuyvBytes_,
                                   cudaMemcpyHostToDevice, stream_));
        launchYuyvToRgbFp32(d_inputYuyv_, d_src_, cfg_.width, cfg_.height, stream_);
        CUDA_CHECK(cudaGetLastError());  // F9: check kernel launch
    }

    // Successfully decoded a frame — reset skip counter
    consecutiveSkips_ = 0;

    // 3. TensorRT inference
    auto* ctx = engine_.context();
    int next = 1 - recIdx_;

    ctx->setTensorAddress("src", d_src_);
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

    // 4. Composite + color convert
    launchCompositeToYuyv(d_fgr_, d_pha_, d_outputYuyv_,
                          cfg_.width, cfg_.height,
                          cfg_.bgR, cfg_.bgG, cfg_.bgB, stream_);
    CUDA_CHECK(cudaGetLastError());  // F9: check kernel launch

    // 5. Download and write
    CUDA_CHECK(cudaMemcpyAsync(h_output_, d_outputYuyv_, yuyvBytes_,
                               cudaMemcpyDeviceToHost, stream_));

    // F18: Record stop event before sync so timing measures GPU pipeline only,
    // not the write() syscall or CPU sync overhead
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
