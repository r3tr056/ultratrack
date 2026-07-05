#include <catch2/catch_test_macros.hpp>
#include <ultratrack/tracking_session.hpp>
#include <ultratrack/ultratrack_sdk.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <vector>

using namespace ultratrack;

namespace {

struct SdkLicenseGuard {
    SdkLicenseGuard() {
        SDKConfig cfg;
        cfg.license_key = "TRIAL";
        REQUIRE(UltraTrackerSDK::initialize(cfg).ok());
    }
    ~SdkLicenseGuard() { UltraTrackerSDK::shutdown(); }
};

TrackerSettings default_tracker_settings() {
    TrackerSettings ts;
    ts.detector_model_path = "models/yolov11n.onnx";
    ts.backend = DetectorBackend::OPENCV_DNN;
    ts.mode = TrackingMode::FAST;
    return ts;
}

Frame make_test_frame(FrameFormat format = FrameFormat::BGR) {
    Frame frame;
    frame.width = 640;
    frame.height = 480;
    frame.format = format;
    frame.data = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(frame.data, cv::Point(100, 100), cv::Point(140, 140), cv::Scalar(255, 255, 255), -1);
    return frame;
}

} // namespace

TEST_CASE("SDK initializes with trial license", "[public_api]") {
    SDKConfig cfg;
    cfg.license_key = "TRIAL";
    auto status = UltraTrackerSDK::initialize(cfg);
    REQUIRE(status.ok());
    REQUIRE(UltraTrackerSDK::is_licensed());
    UltraTrackerSDK::shutdown();
}

TEST_CASE("SDK LogLevel maps to spdlog level", "[public_api]") {
    UltraTrackerSDK::shutdown();

    const std::vector<std::pair<LogLevel, spdlog::level::level_enum>> mappings = {
        {LogLevel::TRACE, spdlog::level::trace},
        {LogLevel::DEBUG, spdlog::level::debug},
        {LogLevel::INFO, spdlog::level::info},
        {LogLevel::WARN, spdlog::level::warn},
        {LogLevel::ERROR, spdlog::level::err},
        {LogLevel::CRITICAL, spdlog::level::critical},
    };

    for (const auto& [level, expected] : mappings) {
        SDKConfig cfg;
        cfg.license_key = "TRIAL";
        cfg.log_level = level;
        REQUIRE(UltraTrackerSDK::initialize(cfg).ok());
        REQUIRE(spdlog::default_logger()->level() == expected);
        UltraTrackerSDK::shutdown();
    }
}

TEST_CASE("SDK double initialize returns error", "[public_api]") {
    SDKConfig cfg;
    cfg.license_key = "TRIAL";
    REQUIRE(UltraTrackerSDK::initialize(cfg).ok());

    auto status = UltraTrackerSDK::initialize(cfg);
    REQUIRE_FALSE(status.ok());
    REQUIRE(status.code() == ErrorCode::ALREADY_INITIALIZED);

    UltraTrackerSDK::shutdown();
}

TEST_CASE("SDK shutdown clears license state", "[public_api]") {
    SDKConfig cfg;
    cfg.license_key = "TRIAL";
    REQUIRE(UltraTrackerSDK::initialize(cfg).ok());
    REQUIRE(UltraTrackerSDK::is_licensed());

    UltraTrackerSDK::shutdown();
    REQUIRE_FALSE(UltraTrackerSDK::is_licensed());
}

TEST_CASE("TrackingSession creation fails when SDK is not licensed", "[public_api]") {
    UltraTrackerSDK::shutdown();

    auto session = TrackingSession::create(default_tracker_settings());
    REQUIRE_FALSE(session.has_value());
    REQUIRE(session.error().code() == ErrorCode::LICENSE_INVALID);
}

TEST_CASE("TrackingSession processes synthetic frame", "[public_api]") {
    SdkLicenseGuard guard;

    auto session = TrackingSession::create(default_tracker_settings());
    if (!session.has_value()) {
        SKIP("Detector model not available: " << session.error().message());
    }

    auto out = session.value()->processFrame(make_test_frame());
    REQUIRE(out.has_value());
}

TEST_CASE("TrackingSession rejects unsupported frame format", "[public_api]") {
    SdkLicenseGuard guard;

    auto session = TrackingSession::create(default_tracker_settings());
    if (!session.has_value()) {
        SKIP("Detector model not available: " << session.error().message());
    }

    auto out = session.value()->processFrame(make_test_frame(FrameFormat::GRAY));
    REQUIRE_FALSE(out.has_value());
    REQUIRE(out.error().code() == ErrorCode::NOT_IMPLEMENTED);
}

TEST_CASE("TrackingSession selectPrimaryTarget is reflected in output", "[public_api]") {
    SdkLicenseGuard guard;

    auto session = TrackingSession::create(default_tracker_settings());
    if (!session.has_value()) {
        SKIP("Detector model not available: " << session.error().message());
    }

    auto select_status = session.value()->selectPrimaryTarget(42);
    REQUIRE(select_status.ok());

    auto out = session.value()->processFrame(make_test_frame());
    REQUIRE(out.has_value());
    REQUIRE(out.value().primary_track_id == 42);
}

TEST_CASE("TrackingSession reset clears tracks and primary target", "[public_api]") {
    SdkLicenseGuard guard;

    auto session = TrackingSession::create(default_tracker_settings());
    if (!session.has_value()) {
        SKIP("Detector model not available: " << session.error().message());
    }

    // Establish some state.
    REQUIRE(session.value()->processFrame(make_test_frame()).has_value());
    REQUIRE(session.value()->selectPrimaryTarget(7).ok());

    auto reset_status = session.value()->reset();
    REQUIRE(reset_status.ok());

    REQUIRE(session.value()->activeTracks().empty());

    auto out = session.value()->processFrame(make_test_frame());
    REQUIRE(out.has_value());
    REQUIRE(out.value().primary_track_id == 0);
}
