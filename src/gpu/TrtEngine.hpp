#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <memory>

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

class TrtEngine {
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;
    Logger logger_;
    
public:
    TrtEngine() = default;
    
    ~TrtEngine() {
        if (stream_) cudaStreamDestroy(stream_);
        // Unique pointers handle the rest, but we need to be careful with destruction order
        // context -> engine -> runtime
        context_.reset();
        engine_.reset();
        runtime_.reset();
    }

    // Load engine, enabling DLA if specified in build
    bool initialize(const std::string& engine_path, int dla_core = -1) {
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file.good()) {
            std::cerr << "Error: Could not read engine file: " << engine_path << std::endl;
            return false;
        }
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            std::cerr << "Error: Could not read engine content." << std::endl;
            return false;
        }

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (dla_core >= 0) {
            runtime_->setDLACore(dla_core);
            std::cout << "Enabled DLA Core " << dla_core << " for " << engine_path << std::endl;
        }

        engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
        if (!engine_) {
            std::cerr << "Error: Failed to deserialize engine." << std::endl;
            return false;
        }

        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            std::cerr << "Error: Failed to create execution context." << std::endl;
            return false;
        }

        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            std::cerr << "Error: Failed to create CUDA stream." << std::endl;
            return false;
        }
        
        return true;
    }

    bool inference(void** bindings, int batch_size = 1) {
        if (!context_ || !stream_) return false;
        // Enqueue V2 is asynchronous
        return context_->enqueueV2(bindings, stream_, nullptr);
    }

    cudaStream_t get_stream() { return stream_; }
    
    int get_binding_index(const char* name) {
        return engine_->getBindingIndex(name);
    }
    
    nvinfer1::Dims get_binding_dims(int index) {
        return engine_->getBindingDimensions(index);
    }
};
