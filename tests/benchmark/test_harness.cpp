#include <catch2/catch_test_macros.hpp>
#include <ultratrack/benchmark/harness.hpp>
#include <ultratrack/ultratrack_sdk.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <string>

using namespace ultratrack;

namespace {

std::filesystem::path get_test_data_dir() {
#ifdef ULTRATRACK_TEST_DATA_DIR
    return std::filesystem::path(ULTRATRACK_TEST_DATA_DIR);
#else
    return std::filesystem::path("tests/data");
#endif
}

std::filesystem::path get_mot_dummy_dir() {
    return get_test_data_dir() / "mot_dummy";
}

void ensure_dummy_images(const std::filesystem::path& dir, int count = 10) {
    std::filesystem::create_directories(dir);
    cv::Mat blank(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 1; i <= count; ++i) {
        std::ostringstream oss;
        oss << "frame_" << std::setw(4) << std::setfill('0') << i << ".jpg";
        auto path = dir / oss.str();
        if (!std::filesystem::exists(path)) {
            REQUIRE(cv::imwrite(path.string(), blank));
        }
    }
}

void write_invalid_gt(const std::filesystem::path& path) {
    std::ofstream out(path);
    REQUIRE(out);
    out << "1,1,100,100,50,80\n";
    out << "not,a,valid,line\n";
    out << "2,1,102,100,50,80\n";
}

struct SdkLicenseGuard {
    SdkLicenseGuard() {
        if (!UltraTrackerSDK::is_licensed()) {
            SDKConfig cfg;
            cfg.license_key = "TRIAL";
            REQUIRE(UltraTrackerSDK::initialize(cfg).ok());
            owned_ = true;
        }
    }
    ~SdkLicenseGuard() {
        if (owned_) UltraTrackerSDK::shutdown();
    }
private:
    bool owned_ = false;
};

} // namespace

TEST_CASE("BenchmarkHarness loads MOT sequence directory", "[benchmark]") {
    BenchmarkHarness harness;
    auto mot_dir = get_mot_dummy_dir();
    ensure_dummy_images(mot_dir);

    auto seq = harness.loadSequence(mot_dir.string());
    REQUIRE(seq.has_value());
    REQUIRE(seq.value().frames.size() == 10);
}

TEST_CASE("BenchmarkHarness loadSequence reports empty directory", "[benchmark]") {
    BenchmarkHarness harness;
    auto empty_dir = get_test_data_dir() / "empty_sequence";
    std::filesystem::create_directories(empty_dir);

    // Remove any stray images from previous runs.
    for (const auto& entry : std::filesystem::directory_iterator(empty_dir)) {
        std::filesystem::remove_all(entry.path());
    }

    auto seq = harness.loadSequence(empty_dir.string());
    REQUIRE_FALSE(seq.has_value());
    REQUIRE(seq.error().code() == ErrorCode::INVALID_ARGUMENT);
}

TEST_CASE("BenchmarkHarness loadSequenceWithGT reports missing gt file", "[benchmark]") {
    BenchmarkHarness harness;
    auto mot_dir = get_test_data_dir() / "mot_no_gt";
    ensure_dummy_images(mot_dir);

    auto seq = harness.loadSequenceWithGT(mot_dir.string(), (mot_dir / "missing_gt.txt").string());
    REQUIRE_FALSE(seq.has_value());
    REQUIRE(seq.error().code() == ErrorCode::INVALID_ARGUMENT);
}

TEST_CASE("BenchmarkHarness loadSequenceWithGT reports invalid gt format", "[benchmark]") {
    BenchmarkHarness harness;
    auto mot_dir = get_test_data_dir() / "mot_invalid_gt";
    ensure_dummy_images(mot_dir);
    write_invalid_gt(mot_dir / "gt.txt");

    auto seq = harness.loadSequenceWithGT(mot_dir.string(), (mot_dir / "gt.txt").string());
    REQUIRE_FALSE(seq.has_value());
    REQUIRE(seq.error().code() == ErrorCode::INVALID_ARGUMENT);
}

TEST_CASE("BenchmarkHarness loadSequenceWithGT loads ground truth", "[benchmark]") {
    BenchmarkHarness harness;
    auto mot_dir = get_mot_dummy_dir();
    ensure_dummy_images(mot_dir);

    auto seq = harness.loadSequenceWithGT(mot_dir.string(), (mot_dir / "gt.txt").string());
    REQUIRE(seq.has_value());
    REQUIRE(seq.value().frames.size() == 10);
    REQUIRE(seq.value().ground_truth.size() == 20);
}

TEST_CASE("BenchmarkHarness runVariant returns latency metrics", "[benchmark]") {
    SdkLicenseGuard guard;
    BenchmarkHarness harness;
    auto mot_dir = get_mot_dummy_dir();
    ensure_dummy_images(mot_dir);

    auto seq = harness.loadSequence(mot_dir.string());
    REQUIRE(seq.has_value());

    TrackerSettings settings;
    settings.detector_model_path = "models/nonexistent.onnx";
    settings.mode = TrackingMode::BALANCED;

    auto result = harness.runVariant("ultratrack_balanced", settings, seq.value());
    REQUIRE(result.variant_name == "ultratrack_balanced");
    // With an invalid model the session cannot be created; the harness reports -1.0.
    REQUIRE(result.metrics.avg_latency_ms == -1.0);
}

TEST_CASE("BenchmarkHarness runVariant reports real latency with valid session", "[benchmark]") {
    SdkLicenseGuard guard;
    BenchmarkHarness harness;
    auto mot_dir = get_mot_dummy_dir();
    ensure_dummy_images(mot_dir);

    auto seq = harness.loadSequence(mot_dir.string());
    REQUIRE(seq.has_value());

    TrackerSettings settings;
    settings.detector_model_path = "models/yolov11n.onnx";
    settings.mode = TrackingMode::BALANCED;

    auto result = harness.runVariant("ultratrack_balanced", settings, seq.value());
    REQUIRE(result.variant_name == "ultratrack_balanced");
    if (result.metrics.avg_latency_ms < 0.0) {
        SKIP("Detector model not available; runVariant returned fallback metrics");
    }
    REQUIRE(result.metrics.avg_latency_ms >= 0.0);
    REQUIRE(result.metrics.p95_latency_ms >= 0.0);
}

TEST_CASE("BenchmarkHarness report does not throw", "[benchmark]") {
    BenchmarkHarness harness;
    BenchmarkResult r1{"variant_a", {}};
    BenchmarkResult r2{"variant_b", {}};
    REQUIRE_NOTHROW(harness.report({r1, r2}));
}
