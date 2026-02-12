#pragma once
#include "v4l2_capture.h"
#include "v4l2_output.h"
#include "trt_engine.h"
#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

struct PipelineConfig {
    std::string inputDevice = "/dev/video0";
    std::string outputDevice = "/dev/video10";
    int width = 1920;
    int height = 1080;
    std::string onnxPath;   // empty = auto-resolve from resolution
    std::string planPath;   // empty = auto-resolve from resolution
    bool fp16 = true;
    float downsampleRatio = 0.5f;
    uint8_t bgR = 0, bgG = 177, bgB = 64;  // green screen default
    float alphaSmoothing = 1.0f;  // 1.0 = no smoothing
    float despillStrength = 0.8f; // 0.0 = off, 1.0 = full suppression
    bool refineAlpha = true;     // guided filter alpha refinement
    int gfRadius = 3;            // guided filter radius (at low-res)
    float gfEps = 0.005f;        // guided filter regularization
    int resetInterval = 100;     // zero recurrent states every N frames (0 = disable)
    int perfLevel = 0;           // 0=baseline, 1=fused kernels, 2=+shmem+output thread, 3=+CUDA graphs
    bool benchmark = false;
};

class Pipeline {
public:
    Pipeline(const PipelineConfig& cfg);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    bool init();
    bool processFrame();  // returns false on fatal error

    float lastFrameMs() const { return lastFrameMs_; }
    float lastWallMs() const { return lastWallMs_; }

private:
    static constexpr int NUM_SLOTS = 2;

    // Double-buffered input slot for CPU-GPU overlap
    struct FrameSlot {
        uint8_t* h_staging = nullptr;  // pinned host staging (JPEG or YUYV data)
        uint8_t* d_input = nullptr;    // device: decoded RGB (MJPEG) or raw YUYV
        float*   d_src = nullptr;      // device: preprocessed FP32 BCHW [1,3,H,W]

        // Per-slot nvJPEG decode state (MJPEG only)
        nvjpegJpegState_t nvState = nullptr;
        nvjpegJpegStream_t nvStream = nullptr;
        nvjpegBufferPinned_t nvPinnedBuf = nullptr;
        nvjpegBufferDevice_t nvDeviceBuf = nullptr;
        nvjpegImage_t nvOutput = {};

        size_t frameSize = 0;
        bool ready = false;
        bool valid = false;
    };

    PipelineConfig cfg_;
    V4L2Capture capture_;
    std::unique_ptr<V4L2Output> output_;
    TrtEngine engine_;

    cudaStream_t stream_ = nullptr;
    uint32_t captureFmt_ = 0;

    // Shared nvJPEG handles (not per-slot)
    nvjpegHandle_t nvjpegHandle_ = nullptr;
    nvjpegJpegDecoder_t nvjpegDecoder_ = nullptr;
    nvjpegDecodeParams_t nvjpegParams_ = nullptr;

    // Double-buffered input slots
    FrameSlot slots_[NUM_SLOTS];
    cudaEvent_t slotDoneEvent_[NUM_SLOTS] = {};
    int processSlotIdx_ = 0;

    // Capture thread and synchronization
    std::thread captureThread_;
    std::mutex mtx_;
    std::condition_variable cvReady_;
    std::condition_variable cvConsumed_;
    std::atomic<bool> stopCapture_{false};
    bool captureError_ = false;  // accessed under mtx_

    // Pre-computed background color as float [0,1]
    float bgRf_ = 0.0f, bgGf_ = 0.0f, bgBf_ = 0.0f;

    // Pinned host output (single-buffered, consumed before next frame)
    uint8_t* h_output_ = nullptr;

    // Device memory: TRT outputs and final YUYV (single-buffered)
    float*   d_fgr_ = nullptr;
    float*   d_pha_ = nullptr;
    float*   d_phaPrev_ = nullptr;
    bool     firstFrame_ = true;
    uint8_t* d_outputYuyv_ = nullptr;

    // Recurrent states: ping-pong buffers [2 sets][4 states] (float32)
    float* d_rec_[2][4] = {};
    int recIdx_ = 0;
    int framesSinceReset_ = 0;

    struct RecDim { int ch; int h; int w; size_t bytes; };
    RecDim recDims_[4] = {};

    GuidedFilterState gfState_;

    size_t rgbBytes_ = 0;
    size_t yuyvBytes_ = 0;
    size_t stagingSize_ = 0;

    int consecutiveSkips_ = 0;

    float lastFrameMs_ = 0.0f;
    float lastWallMs_ = 0.0f;
    cudaEvent_t evStart_ = nullptr, evStop_ = nullptr;

    bool allocateGpuMemory();
    void freeGpuMemory();
    void captureThreadFunc();
};
