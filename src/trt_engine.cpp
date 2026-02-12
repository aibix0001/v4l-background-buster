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
        // F1: Warn that plan files are untrusted binary blobs deserialized by TRT
        fprintf(stderr, "WARNING: Loading cached TensorRT plan from %s\n", planPath.c_str());
        fprintf(stderr, "  Plan files contain serialized GPU code. Only load plans you built yourself.\n");
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

    // 4 GB workspace
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 4ULL << 30);
    config->setBuilderOptimizationLevel(5);

    if (fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        fprintf(stderr, "FP16 enabled\n");
    }

    // The simplified ONNX model has all static shapes — no optimization profile needed.

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
    // F7: Validate file is readable and non-empty
    if (!in.good()) {
        fprintf(stderr, "Cannot open plan file: %s\n", planPath.c_str());
        return false;
    }
    auto pos = in.tellg();
    if (pos <= 0) {
        fprintf(stderr, "Plan file is empty or unreadable: %s\n", planPath.c_str());
        return false;
    }
    size_t size = static_cast<size_t>(pos);
    in.seekg(0);
    std::vector<char> data(size);
    in.read(data.data(), size);
    // F7: Verify read completed successfully
    if (!in.good() || static_cast<size_t>(in.gcount()) != size) {
        fprintf(stderr, "Failed to read plan file (read %zd of %zu bytes)\n",
                static_cast<ssize_t>(in.gcount()), size);
        return false;
    }

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
