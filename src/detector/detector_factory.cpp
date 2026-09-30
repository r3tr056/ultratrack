#include <ultratrack/detector/opencv_dnn_backend.hpp>

namespace ultratrack {

Result<std::unique_ptr<IDetectorBackend>> create_detector_backend(
    DetectorBackend backend, const OpenCVDNNBackend::Config& cfg) {
    switch (backend) {
        case DetectorBackend::OPENCV_DNN:
            return OpenCVDNNBackend::create(cfg);
        case DetectorBackend::ONNX_RUNTIME:
            return Status(ErrorCode::NOT_IMPLEMENTED, "ONNX Runtime backend not implemented");
        case DetectorBackend::TENSORRT:
            return Status(ErrorCode::NOT_IMPLEMENTED, "TensorRT backend not implemented");
        default:
            return Status(ErrorCode::NOT_IMPLEMENTED, "unknown detector backend");
    }
}

} // namespace ultratrack
