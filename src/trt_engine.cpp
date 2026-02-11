#include "trt_engine.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>
#include <vector>

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------
void TrtEngine::Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING)
        fprintf(stderr, "[TRT %d] %s\n", static_cast<int>(severity), msg);
}

// ---------------------------------------------------------------------------
// TrtEngine
// ---------------------------------------------------------------------------
TrtEngine::TrtEngine() = default;
TrtEngine::~TrtEngine() = default;

bool TrtEngine::loadOrBuild(const std::string& onnxPath, const std::string& planPath,
                            bool fp16, int width, int height) {
    // Try loading cached plan first
    std::ifstream planFile(planPath, std::ios::binary | std::ios::ate);
    if (planFile.good() && planFile.tellg() > 0) {
        fprintf(stderr, "Loading cached TensorRT engine from %s\n", planPath.c_str());
        planFile.close();
        if (loadFromPlan(planPath)) {
            context_.reset(engine_->createExecutionContext());
            if (context_) {
                fprintf(stderr, "Engine loaded successfully\n");
                return true;
            }
        }
        fprintf(stderr, "Cached plan invalid, rebuilding...\n");
    }

    // Build from ONNX
    fprintf(stderr, "Building TensorRT engine from %s (this takes 2-5 minutes)...\n",
            onnxPath.c_str());
    if (!buildFromOnnx(onnxPath, planPath, fp16, width, height))
        return false;

    context_.reset(engine_->createExecutionContext());
    return context_ != nullptr;
}

bool TrtEngine::buildFromOnnx(const std::string& onnxPath, const std::string& planPath,
                              bool fp16, int width, int height) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder, Deleter>(
        nvinfer1::createInferBuilder(logger_));
    if (!builder) return false;

    // Explicit batch
    const uint32_t flags = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition, Deleter>(
        builder->createNetworkV2(flags));
    if (!network) return false;

    auto parser = std::unique_ptr<nvonnxparser::IParser, Deleter>(
        nvonnxparser::createParser(*network, logger_));
    if (!parser) return false;

    if (!parser->parseFromFile(onnxPath.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        fprintf(stderr, "Failed to parse ONNX model\n");
        for (int i = 0; i < parser->getNbErrors(); i++)
            fprintf(stderr, "  Parser error: %s\n", parser->getError(i)->desc());
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig, Deleter>(
        builder->createBuilderConfig());
    if (!config) return false;

    // 2 GB workspace
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 2ULL << 30);

    if (fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        fprintf(stderr, "FP16 enabled\n");
    }

    // Optimization profile for RVM's dynamic inputs
    auto profile = builder->createOptimizationProfile();

    // Compute recurrent state spatial dims at downsample_ratio=0.25
    int intH = height / 4;  // 270 for 1080
    int intW = width / 4;   // 480 for 1920

    // src: fixed size [1,3,H,W]
    nvinfer1::Dims4 srcDim{1, 3, height, width};
    profile->setDimensions("src", nvinfer1::OptProfileSelector::kMIN, srcDim);
    profile->setDimensions("src", nvinfer1::OptProfileSelector::kOPT, srcDim);
    profile->setDimensions("src", nvinfer1::OptProfileSelector::kMAX, srcDim);

    // Recurrent states: r1i-r4i
    struct RecState { const char* name; int ch; int divH; int divW; };
    RecState recStates[] = {
        {"r1i", 16, 2, 2},
        {"r2i", 20, 4, 4},
        {"r3i", 40, 8, 8},
        {"r4i", 64, 16, 16},
    };
    for (auto& rs : recStates) {
        int rH = (intH + rs.divH - 1) / rs.divH;  // ceil division
        int rW = (intW + rs.divW - 1) / rs.divW;
        nvinfer1::Dims4 minDim{1, rs.ch, 1, 1};
        nvinfer1::Dims4 optDim{1, rs.ch, rH, rW};
        nvinfer1::Dims4 maxDim{1, rs.ch, rH, rW};
        profile->setDimensions(rs.name, nvinfer1::OptProfileSelector::kMIN, minDim);
        profile->setDimensions(rs.name, nvinfer1::OptProfileSelector::kOPT, optDim);
        profile->setDimensions(rs.name, nvinfer1::OptProfileSelector::kMAX, maxDim);
        fprintf(stderr, "  %s: opt [1,%d,%d,%d]\n", rs.name, rs.ch, rH, rW);
    }

    // downsample_ratio: scalar [1]
    nvinfer1::Dims dsDim;
    dsDim.nbDims = 1;
    dsDim.d[0] = 1;
    profile->setDimensions("downsample_ratio", nvinfer1::OptProfileSelector::kMIN, dsDim);
    profile->setDimensions("downsample_ratio", nvinfer1::OptProfileSelector::kOPT, dsDim);
    profile->setDimensions("downsample_ratio", nvinfer1::OptProfileSelector::kMAX, dsDim);

    config->addOptimizationProfile(profile);

    // Build serialized engine
    auto plan = std::unique_ptr<nvinfer1::IHostMemory, Deleter>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        fprintf(stderr, "Engine build failed\n");
        return false;
    }

    // Save plan to disk
    std::ofstream out(planPath, std::ios::binary);
    if (out.good()) {
        out.write(static_cast<const char*>(plan->data()), plan->size());
        out.close();
        fprintf(stderr, "Engine saved to %s (%zu bytes)\n", planPath.c_str(), plan->size());
    }

    // Deserialize
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(plan->data(), plan->size()));
    return engine_ != nullptr;
}

bool TrtEngine::loadFromPlan(const std::string& planPath) {
    std::ifstream in(planPath, std::ios::binary | std::ios::ate);
    size_t size = in.tellg();
    in.seekg(0);
    std::vector<char> data(size);
    in.read(data.data(), size);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
    return engine_ != nullptr;
}

void TrtEngine::printBindings() const {
    if (!engine_) return;
    int n = engine_->getNbIOTensors();
    fprintf(stderr, "Engine has %d IO tensors:\n", n);
    for (int i = 0; i < n; i++) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto dims = engine_->getTensorShape(name);
        auto dtype = engine_->getTensorDataType(name);

        fprintf(stderr, "  [%d] %s (%s) dtype=%d dims=(",
                i, name,
                mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT",
                static_cast<int>(dtype));
        for (int d = 0; d < dims.nbDims; d++)
            fprintf(stderr, "%s%ld", d ? "," : "", static_cast<long>(dims.d[d]));
        fprintf(stderr, ")\n");
    }
}
