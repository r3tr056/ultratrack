#pragma once
#include <ultratrack/detector/detector_backend.hpp>
#include <opencv2/dnn.hpp>
#include <string>

namespace ultratrack {

class OpenCVDNNBackend : public IDetectorBackend {
public:
    struct Config {
        std::string model_path;
        cv::Size input_size{640, 640};
        float confidence_threshold = 0.3f;
        float nms_threshold = 0.5f;
        bool use_cuda = false;
    };

    static Result<std::unique_ptr<IDetectorBackend>> create(const Config& cfg);
    Result<std::vector<Detection>> detect(const Frame& frame) override;

private:
    OpenCVDNNBackend(cv::dnn::Net net, const Config& cfg);
    cv::dnn::Net net_;
    Config cfg_;
};

/// Factory function for creating detector backends.
/// Currently only DetectorBackend::OPENCV_DNN is supported.
Result<std::unique_ptr<IDetectorBackend>> create_detector_backend(
    DetectorBackend backend, const OpenCVDNNBackend::Config& cfg);

} // namespace ultratrack
