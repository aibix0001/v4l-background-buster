#include "pipeline.h"
#include "cuda_kernels.h"
#include <cstdio>
#include <cstring>
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
}

bool Pipeline::init() {
    // Init V4L2 capture
    if (!capture_.init()) return false;
    if (capture_.pixelFormat() != V4L2_PIX_FMT_YUYV) {
        char fourcc[5] = {};
        uint32_t pf = capture_.pixelFormat();
        memcpy(fourcc, &pf, 4);
        fprintf(stderr, "Unsupported capture format: %s (only YUYV supported)\n", fourcc);
        return false;
    }

    // Update dimensions from what the camera actually gave us
    cfg_.width = capture_.width();
    cfg_.height = capture_.height();
    fprintf(stderr, "Using resolution: %dx%d\n", cfg_.width, cfg_.height);

    // Init V4L2 output with actual dimensions
    output_ = std::make_unique<V4L2Output>(cfg_.outputDevice, cfg_.width, cfg_.height, V4L2_PIX_FMT_YUYV);
    if (!output_->init()) return false;

    // Init TensorRT
    if (!engine_.loadOrBuild(cfg_.onnxPath, cfg_.planPath, cfg_.fp16,
                             cfg_.width, cfg_.height))
        return false;
    engine_.printBindings();

    // CUDA stream + events
    CUDA_CHECK(cudaStreamCreate(&stream_));
    if (cfg_.benchmark) {
        CUDA_CHECK(cudaEventCreate(&evStart_));
        CUDA_CHECK(cudaEventCreate(&evStop_));
    }

    // Allocate GPU + pinned memory
    if (!allocateGpuMemory()) return false;

    // Start capture
    if (!capture_.startStreaming()) return false;

    fprintf(stderr, "Pipeline initialized successfully\n");
    return true;
}

bool Pipeline::allocateGpuMemory() {
    int W = cfg_.width;
    int H = cfg_.height;
    int pixels = W * H;

    captureBytes_ = pixels * 2;       // YUYV: 2 bytes/pixel
    outputBytes_ = pixels * 2;
    size_t srcBytes = pixels * 3 * sizeof(__half);   // RGB FP16 BCHW
    size_t fgrBytes = pixels * 3 * sizeof(__half);
    size_t phaBytes = pixels * 1 * sizeof(__half);

    // Compute recurrent state dimensions
    int intH = H / 4;  // downsample_ratio=0.25
    int intW = W / 4;
    int divs[] = {2, 4, 8, 16};
    int chs[] = {16, 20, 40, 64};
    for (int i = 0; i < 4; i++) {
        recDims_[i].ch = chs[i];
        recDims_[i].h = (intH + divs[i] - 1) / divs[i];
        recDims_[i].w = (intW + divs[i] - 1) / divs[i];
        recDims_[i].bytes = 1 * chs[i] * recDims_[i].h * recDims_[i].w * sizeof(__half);
    }

    // Pinned host memory
    CUDA_CHECK(cudaHostAlloc(&h_capture_, captureBytes_, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_output_, outputBytes_, cudaHostAllocDefault));

    // Device memory
    CUDA_CHECK(cudaMalloc(&d_inputYuyv_, captureBytes_));
    CUDA_CHECK(cudaMalloc(&d_src_, srcBytes));
    CUDA_CHECK(cudaMalloc(&d_fgr_, fgrBytes));
    CUDA_CHECK(cudaMalloc(&d_pha_, phaBytes));
    CUDA_CHECK(cudaMalloc(&d_outputYuyv_, outputBytes_));

    // Recurrent state ping-pong buffers
    for (int s = 0; s < 2; s++) {
        for (int i = 0; i < 4; i++) {
            CUDA_CHECK(cudaMalloc(&d_rec_[s][i], recDims_[i].bytes));
            CUDA_CHECK(cudaMemset(d_rec_[s][i], 0, recDims_[i].bytes));
        }
    }

    // Downsample ratio scalar on device
    CUDA_CHECK(cudaMalloc(&d_dsRatio_, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_dsRatio_, &cfg_.downsampleRatio, sizeof(float),
                          cudaMemcpyHostToDevice));

    fprintf(stderr, "GPU memory allocated\n");
    return true;
}

void Pipeline::freeGpuMemory() {
    if (h_capture_) { cudaFreeHost(h_capture_); h_capture_ = nullptr; }
    if (h_output_) { cudaFreeHost(h_output_); h_output_ = nullptr; }
    if (d_inputYuyv_) { cudaFree(d_inputYuyv_); d_inputYuyv_ = nullptr; }
    if (d_src_) { cudaFree(d_src_); d_src_ = nullptr; }
    if (d_fgr_) { cudaFree(d_fgr_); d_fgr_ = nullptr; }
    if (d_pha_) { cudaFree(d_pha_); d_pha_ = nullptr; }
    if (d_outputYuyv_) { cudaFree(d_outputYuyv_); d_outputYuyv_ = nullptr; }
    for (int s = 0; s < 2; s++)
        for (int i = 0; i < 4; i++)
            if (d_rec_[s][i]) { cudaFree(d_rec_[s][i]); d_rec_[s][i] = nullptr; }
    if (d_dsRatio_) { cudaFree(d_dsRatio_); d_dsRatio_ = nullptr; }
}

bool Pipeline::processFrame() {
    if (cfg_.benchmark)
        cudaEventRecord(evStart_, stream_);

    // 1. Capture frame from V4L2
    size_t frameSize = 0;
    const uint8_t* mmapPtr = capture_.dequeueFrame(frameSize);
    if (!mmapPtr) return false;

    // Copy from mmap buffer to pinned host memory
    memcpy(h_capture_, mmapPtr, frameSize);
    capture_.requeueBuffer();

    // 2. Upload to GPU
    CUDA_CHECK(cudaMemcpyAsync(d_inputYuyv_, h_capture_, captureBytes_,
                               cudaMemcpyHostToDevice, stream_));

    // 3. Preprocess: YUYV → RGB FP16 BCHW
    launchYuyvToRgbFp16(d_inputYuyv_, d_src_, cfg_.width, cfg_.height, stream_);

    // 4. TensorRT inference with recurrent state ping-pong
    auto* ctx = engine_.context();
    int next = 1 - recIdx_;

    // Set input shapes (required for dynamic dimensions)
    nvinfer1::Dims4 srcShape{1, 3, cfg_.height, cfg_.width};
    ctx->setInputShape("src", srcShape);

    const char* riNames[] = {"r1i", "r2i", "r3i", "r4i"};
    for (int i = 0; i < 4; i++) {
        nvinfer1::Dims4 rShape{1, recDims_[i].ch,
                               firstFrame_ ? 1 : recDims_[i].h,
                               firstFrame_ ? 1 : recDims_[i].w};
        ctx->setInputShape(riNames[i], rShape);
    }

    nvinfer1::Dims dsDim;
    dsDim.nbDims = 1;
    dsDim.d[0] = 1;
    ctx->setInputShape("downsample_ratio", dsDim);

    // Set tensor addresses
    ctx->setTensorAddress("src", d_src_);
    ctx->setTensorAddress("r1i", firstFrame_ ? d_rec_[0][0] : d_rec_[recIdx_][0]);
    ctx->setTensorAddress("r2i", firstFrame_ ? d_rec_[0][1] : d_rec_[recIdx_][1]);
    ctx->setTensorAddress("r3i", firstFrame_ ? d_rec_[0][2] : d_rec_[recIdx_][2]);
    ctx->setTensorAddress("r4i", firstFrame_ ? d_rec_[0][3] : d_rec_[recIdx_][3]);
    ctx->setTensorAddress("downsample_ratio", d_dsRatio_);
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
    firstFrame_ = false;

    // 5. Composite + color convert: FGR+PHA → YUYV
    launchCompositeToYuyv(d_fgr_, d_pha_, d_outputYuyv_,
                          cfg_.width, cfg_.height,
                          cfg_.bgR, cfg_.bgG, cfg_.bgB, stream_);

    // 6. Download to host
    CUDA_CHECK(cudaMemcpyAsync(h_output_, d_outputYuyv_, outputBytes_,
                               cudaMemcpyDeviceToHost, stream_));

    // 7. Sync and write to v4l2loopback
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    output_->writeFrame(h_output_, outputBytes_);

    // Benchmark timing
    if (cfg_.benchmark) {
        cudaEventRecord(evStop_, stream_);
        cudaEventSynchronize(evStop_);
        cudaEventElapsedTime(&lastFrameMs_, evStart_, evStop_);
    }

    return true;
}
