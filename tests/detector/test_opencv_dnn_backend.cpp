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

TEST_CASE("OpenCVDNNBackend rejects invalid input size", "[detector]") {
    OpenCVDNNBackend::Config cfg{"nonexistent_model.onnx"};

    cfg.input_size = cv::Size(0, 0);
    auto backend = OpenCVDNNBackend::create(cfg);
    REQUIRE(!backend.has_value());
    REQUIRE(backend.error().code() == ErrorCode::INVALID_ARGUMENT);

    cfg.input_size = cv::Size(-1, 640);
    backend = OpenCVDNNBackend::create(cfg);
    REQUIRE(!backend.has_value());
    REQUIRE(backend.error().code() == ErrorCode::INVALID_ARGUMENT);
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

TEST_CASE("OpenCVDNNBackend creates with CUDA flag enabled", "[detector]") {
    const std::string model_path = "models/test.onnx";
    if (!std::filesystem::exists(model_path)) {
        SKIP("Model file not found: " << model_path);
    }

    OpenCVDNNBackend::Config cfg{model_path};
    cfg.use_cuda = true;
    auto backend = OpenCVDNNBackend::create(cfg);
    REQUIRE(backend.has_value());
}

TEST_CASE("OpenCVDNNBackend handles single-class output (dims == 5)", "[detector]") {
    // Layout [1, dims, anchors] with dims == 5 (no class-score entries).
    int sizes[] = {1, 5, 2};
    cv::Mat output(3, sizes, CV_32F);
    float* data = output.ptr<float>();
    // Anchor 0: [cx=100, cy=100, w=40, h=40, obj_conf=0.9]
    data[0] = 100.0f;
    data[1] = 100.0f;
    data[2] = 40.0f;
    data[3] = 40.0f;
    data[4] = 0.9f;
    // Anchor 1: [cx=200, cy=200, w=50, h=50, obj_conf=0.8]
    data[5] = 200.0f;
    data[6] = 200.0f;
    data[7] = 50.0f;
    data[8] = 50.0f;
    data[9] = 0.8f;

    OpenCVDNNBackend::Config cfg{"models/test.onnx"};
    cfg.input_size = cv::Size(640, 640);
    cfg.confidence_threshold = 0.3f;

    Frame frame;
    frame.width = 640;
    frame.height = 480;
    frame.format = FrameFormat::BGR;

    auto result = OpenCVDNNBackend::parseYOLOOutput(output, frame, cfg);
    REQUIRE(result.has_value());

    const auto& detections = result.value();
    REQUIRE(detections.size() == 2);
    for (const auto& d : detections) {
        REQUIRE(d.bbox.width > 0);
        REQUIRE(d.bbox.height > 0);
        REQUIRE(d.confidence >= cfg.confidence_threshold);
        REQUIRE(d.confidence <= 1.0f);
        REQUIRE(d.class_id == 0);
    }
}
