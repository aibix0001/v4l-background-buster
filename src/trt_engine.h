#pragma once
#include <NvInfer.h>
#include <memory>
#include <string>
#include <vector>

class TrtEngine {
public:
    TrtEngine();
    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // Load cached plan if valid, else build from ONNX and cache.
    bool loadOrBuild(const std::string& onnxPath, const std::string& planPath,
                     bool fp16, int width, int height);

    nvinfer1::IExecutionContext* context() { return context_.get(); }
    nvinfer1::ICudaEngine* engine() { return engine_.get(); }

    // Print all tensor names, shapes, types for debugging
    void printBindings() const;

private:
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    };

    struct Deleter {
        template <typename T>
        void operator()(T* p) const { if (p) delete p; }
    };

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime, Deleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, Deleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, Deleter> context_;

    bool buildFromOnnx(const std::string& onnxPath, const std::string& planPath,
                       bool fp16, int width, int height);
    bool loadFromPlan(const std::string& planPath);
};
