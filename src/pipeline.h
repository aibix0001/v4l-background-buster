#pragma once
#include "v4l2_capture.h"
#include "v4l2_output.h"
#include "trt_engine.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>

struct PipelineConfig {
    std::string inputDevice = "/dev/video0";
    std::string outputDevice = "/dev/video10";
    int width = 1920;
    int height = 1080;
    std::string onnxPath = "models/rvm_mobilenetv3_fp32.onnx";
    std::string planPath = "models/rvm.plan";
    bool fp16 = true;
    float downsampleRatio = 0.25f;
    uint8_t bgR = 0, bgG = 177, bgB = 64;  // green screen default
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

    // Stats from last frame (if benchmark enabled)
    float lastFrameMs() const { return lastFrameMs_; }

private:
    PipelineConfig cfg_;
    V4L2Capture capture_;
    std::unique_ptr<V4L2Output> output_;
    TrtEngine engine_;

    cudaStream_t stream_ = nullptr;

    // Pinned host memory
    uint8_t* h_capture_ = nullptr;   // capture frame (host, pinned)
    uint8_t* h_output_ = nullptr;    // output frame (host, pinned)

    // Device memory
    uint8_t* d_inputYuyv_ = nullptr;  // raw YUYV on GPU
    __half*  d_src_ = nullptr;        // preprocessed RGB FP16 BCHW
    __half*  d_fgr_ = nullptr;        // foreground output
    __half*  d_pha_ = nullptr;        // alpha output
    uint8_t* d_outputYuyv_ = nullptr; // composited YUYV on GPU

    // Recurrent states: ping-pong buffers [2 sets][4 states]
    __half* d_rec_[2][4] = {};
    int recIdx_ = 0;                  // current read index
    bool firstFrame_ = true;

    // Downsample ratio on device
    float* d_dsRatio_ = nullptr;

    // Recurrent state dimensions
    struct RecDim { int ch; int h; int w; size_t bytes; };
    RecDim recDims_[4] = {};

    size_t captureBytes_ = 0;
    size_t outputBytes_ = 0;

    float lastFrameMs_ = 0.0f;
    cudaEvent_t evStart_ = nullptr, evStop_ = nullptr;

    bool allocateGpuMemory();
    void freeGpuMemory();
};
