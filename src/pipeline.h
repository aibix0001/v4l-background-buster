#pragma once
#include "v4l2_capture.h"
#include "v4l2_output.h"
#include "trt_engine.h"
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <memory>
#include <string>

struct PipelineConfig {
    std::string inputDevice = "/dev/video0";
    std::string outputDevice = "/dev/video10";
    int width = 1920;
    int height = 1080;
    std::string onnxPath = "models/rvm_mobilenetv3_fp32_simplified.onnx";
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

    float lastFrameMs() const { return lastFrameMs_; }
    float lastWallMs() const { return lastWallMs_; }

private:
    PipelineConfig cfg_;
    V4L2Capture capture_;
    std::unique_ptr<V4L2Output> output_;
    TrtEngine engine_;

    cudaStream_t stream_ = nullptr;
    uint32_t captureFmt_ = 0;

    // nvJPEG decoupled decoder
    nvjpegHandle_t nvjpegHandle_ = nullptr;
    nvjpegJpegState_t nvjpegState_ = nullptr;
    nvjpegJpegDecoder_t nvjpegDecoder_ = nullptr;
    nvjpegJpegStream_t nvjpegStream_ = nullptr;
    nvjpegDecodeParams_t nvjpegParams_ = nullptr;
    nvjpegBufferPinned_t nvjpegPinnedBuf_ = nullptr;
    nvjpegBufferDevice_t nvjpegDeviceBuf_ = nullptr;
    nvjpegImage_t nvjpegOutput_ = {};

    // Pinned staging buffer for JPEG compressed data
    uint8_t* h_jpegStaging_ = nullptr;
    size_t jpegStagingSize_ = 0;

    // Pinned host memory
    uint8_t* h_rgb_ = nullptr;
    uint8_t* h_output_ = nullptr;

    // Device memory (TRT I/O is float32 even with FP16 internal compute)
    uint8_t* d_inputRgb_ = nullptr;
    uint8_t* d_inputYuyv_ = nullptr;
    float*   d_src_ = nullptr;        // preprocessed RGB FP32 BCHW
    float*   d_fgr_ = nullptr;        // foreground output FP32
    float*   d_pha_ = nullptr;        // alpha output FP32
    uint8_t* d_outputYuyv_ = nullptr;

    // Recurrent states: ping-pong buffers [2 sets][4 states] (float32)
    float* d_rec_[2][4] = {};
    int recIdx_ = 0;

    struct RecDim { int ch; int h; int w; size_t bytes; };
    RecDim recDims_[4] = {};

    size_t rgbBytes_ = 0;
    size_t yuyvBytes_ = 0;

    int consecutiveSkips_ = 0;  // #10: track consecutive corrupt frames

    float lastFrameMs_ = 0.0f;
    float lastWallMs_ = 0.0f;
    cudaEvent_t evStart_ = nullptr, evStop_ = nullptr;

    bool allocateGpuMemory();
    void freeGpuMemory();
};
