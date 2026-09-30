#pragma once
#include <ultratrack/core/result.hpp>
#include <ultratrack/core/types.hpp>
#include <memory>
#include <vector>

namespace ultratrack {

class IDetectorBackend {
public:
    virtual ~IDetectorBackend() = default;
    virtual Result<std::vector<Detection>> detect(const Frame& frame) = 0;
};

enum class DetectorBackend { OPENCV_DNN, ONNX_RUNTIME, TENSORRT };

} // namespace ultratrack
