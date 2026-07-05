#include <catch2/catch_test_macros.hpp>
#include <ultratrack/detector/opencv_dnn_backend.hpp>
#include <filesystem>
#include <opencv2/core.hpp>

using namespace ultratrack;

TEST_CASE("OpenCVDNNBackend rejects empty model path", "[detector]") {
    OpenCVDNNBackend::Config cfg{""};
    auto backend = OpenCVDNNBackend::create(cfg);
    REQUIRE(!backend.has_value());
    REQUIRE(backend.error().code() == ErrorCode::MODEL_LOAD_FAILED);
}

TEST_CASE("OpenCVDNNBackend rejects missing model file", "[detector]") {
    OpenCVDNNBackend::Config cfg{"nonexistent_model.onnx"};
    auto backend = OpenCVDNNBackend::create(cfg);
    REQUIRE(!backend.has_value());
    REQUIRE(backend.error().code() == ErrorCode::MODEL_LOAD_FAILED);
}

TEST_CASE("Factory returns not implemented for unimplemented backends", "[detector]") {
    OpenCVDNNBackend::Config cfg{""};
    auto backend = create_detector_backend(DetectorBackend::ONNX_RUNTIME, cfg);
    REQUIRE(!backend.has_value());
    REQUIRE(backend.error().code() == ErrorCode::NOT_IMPLEMENTED);
}

TEST_CASE("OpenCVDNNBackend detects empty frame", "[detector]") {
    const std::string model_path = "models/test.onnx";
    if (!std::filesystem::exists(model_path)) {
        SKIP("Model file not found: " << model_path);
    }

    OpenCVDNNBackend::Config cfg{model_path};
    auto backend = OpenCVDNNBackend::create(cfg);
    REQUIRE(backend.has_value());

    Frame frame;
    frame.width = 640;
    frame.height = 480;
    auto result = backend.value()->detect(frame);
    REQUIRE(!result.has_value());
    REQUIRE(result.error().code() == ErrorCode::EMPTY_FRAME);
}

TEST_CASE("OpenCVDNNBackend runs inference when model is present", "[detector]") {
    const std::string model_path = "models/test.onnx";
    if (!std::filesystem::exists(model_path)) {
        SKIP("Model file not found: " << model_path);
    }

    OpenCVDNNBackend::Config cfg{model_path};
    auto backend = OpenCVDNNBackend::create(cfg);
    REQUIRE(backend.has_value());

    Frame frame;
    frame.data = cv::Mat::zeros(480, 640, CV_8UC3);
    frame.width = frame.data.cols;
    frame.height = frame.data.rows;
    frame.format = FrameFormat::BGR;

    auto result = backend.value()->detect(frame);
    REQUIRE(result.has_value());

    const auto& detections = result.value();
    for (const auto& d : detections) {
        REQUIRE(d.bbox.width > 0);
        REQUIRE(d.bbox.height > 0);
        REQUIRE(d.confidence >= 0.0f);
        REQUIRE(d.confidence <= 1.0f);
        REQUIRE(d.class_id >= 0);
    }
}
