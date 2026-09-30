#include <catch2/catch_test_macros.hpp>
#include <ultratrack/tracker/kcf_tracker.hpp>

using namespace ultratrack;

TEST_CASE("KCFTracker initializes and predicts", "[tracker]") {
    KCFTracker::Config cfg;
    cfg.template_size = cv::Size(64, 64);
    auto tracker = KCFTracker::create(cfg);
    REQUIRE(tracker.has_value());

    Track track;
    track.bbox = Rect2f(100, 100, 40, 40);
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(frame, cv::Point(100, 100), cv::Point(140, 140), cv::Scalar(255, 255, 255), -1);

    auto status = tracker.value()->init(track, frame);
    REQUIRE(status.ok());

    auto pred = tracker.value()->predict(track, frame);
    REQUIRE(pred.has_value());
    REQUIRE(pred.value().width > 0);
}

TEST_CASE("KCFTracker rejects invalid configuration", "[tracker]") {
    KCFTracker::Config cfg;
    cfg.template_size = cv::Size(0, 64);
    auto tracker = KCFTracker::create(cfg);
    REQUIRE(!tracker.has_value());
}

TEST_CASE("KCFTracker init rejects empty frame", "[tracker]") {
    KCFTracker::Config cfg;
    cfg.template_size = cv::Size(64, 64);
    auto tracker = KCFTracker::create(cfg);
    REQUIRE(tracker.has_value());

    Track track;
    track.bbox = Rect2f(10, 10, 20, 20);
    cv::Mat empty_frame;

    auto status = tracker.value()->init(track, empty_frame);
    REQUIRE(!status.ok());
    REQUIRE(status.code() == ErrorCode::EMPTY_FRAME);
}

TEST_CASE("KCFTracker predict fails without initialization", "[tracker]") {
    KCFTracker::Config cfg;
    cfg.template_size = cv::Size(64, 64);
    auto tracker = KCFTracker::create(cfg);
    REQUIRE(tracker.has_value());

    Track track;
    track.bbox = Rect2f(100, 100, 40, 40);
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    auto pred = tracker.value()->predict(track, frame);
    REQUIRE(!pred.has_value());
    REQUIRE(pred.error().code() == ErrorCode::TRACKING_LOST);
}

TEST_CASE("KCFTracker supports non-square template size", "[tracker]") {
    KCFTracker::Config cfg;
    cfg.template_size = cv::Size(96, 64);
    auto tracker = KCFTracker::create(cfg);
    REQUIRE(tracker.has_value());

    Track track;
    track.bbox = Rect2f(100, 100, 40, 40);
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(frame, cv::Point(100, 100), cv::Point(140, 140), cv::Scalar(255, 255, 255), -1);

    REQUIRE(tracker.value()->init(track, frame).ok());
    REQUIRE(!track.correlation_filter.empty());

    auto pred = tracker.value()->predict(track, frame);
    REQUIRE(pred.has_value());
    REQUIRE(pred.value().width > 0);
}

TEST_CASE("KCFTracker update rejects invalid bbox", "[tracker]") {
    KCFTracker::Config cfg;
    cfg.template_size = cv::Size(64, 64);
    auto tracker = KCFTracker::create(cfg);
    REQUIRE(tracker.has_value());

    Track track;
    track.bbox = Rect2f(100, 100, 40, 40);
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(frame, cv::Point(100, 100), cv::Point(140, 140), cv::Scalar(255, 255, 255), -1);

    REQUIRE(tracker.value()->init(track, frame).ok());
    Rect2f old_bbox = track.bbox;

    Rect2f invalid_bbox(-100, -100, 40, 40);
    auto status = tracker.value()->update(track, frame, invalid_bbox);
    REQUIRE(!status.ok());
    REQUIRE(status.code() == ErrorCode::INVALID_PATCH_SIZE);
    REQUIRE(track.bbox.x == old_bbox.x);
    REQUIRE(track.bbox.y == old_bbox.y);
}

TEST_CASE("KCFTracker predict rejects empty frame", "[tracker]") {
    KCFTracker::Config cfg;
    cfg.template_size = cv::Size(64, 64);
    auto tracker = KCFTracker::create(cfg);
    REQUIRE(tracker.has_value());

    Track track;
    track.bbox = Rect2f(100, 100, 40, 40);
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(frame, cv::Point(100, 100), cv::Point(140, 140), cv::Scalar(255, 255, 255), -1);

    REQUIRE(tracker.value()->init(track, frame).ok());

    cv::Mat empty_frame;
    auto pred = tracker.value()->predict(track, empty_frame);
    REQUIRE(!pred.has_value());
    REQUIRE(pred.error().code() == ErrorCode::EMPTY_FRAME);
}

TEST_CASE("KCFTracker update rejects empty frame", "[tracker]") {
    KCFTracker::Config cfg;
    cfg.template_size = cv::Size(64, 64);
    auto tracker = KCFTracker::create(cfg);
    REQUIRE(tracker.has_value());

    Track track;
    track.bbox = Rect2f(100, 100, 40, 40);
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(frame, cv::Point(100, 100), cv::Point(140, 140), cv::Scalar(255, 255, 255), -1);

    REQUIRE(tracker.value()->init(track, frame).ok());
    Rect2f old_bbox = track.bbox;

    cv::Mat empty_frame;
    auto status = tracker.value()->update(track, empty_frame, old_bbox);
    REQUIRE(!status.ok());
    REQUIRE(status.code() == ErrorCode::EMPTY_FRAME);
}

TEST_CASE("KCFTracker catches OpenCV exceptions and returns Status", "[tracker]") {
    KCFTracker::Config cfg;
    cfg.template_size = cv::Size(64, 64);
    auto tracker = KCFTracker::create(cfg);
    REQUIRE(tracker.has_value());

    Track track;
    track.bbox = Rect2f(10, 10, 50, 50);
    // CV_16SC3 is not supported by cv::COLOR_BGR2GRAY and will trigger a cv::Exception.
    cv::Mat bad_frame = cv::Mat::zeros(100, 100, CV_16SC3);

    auto init_status = tracker.value()->init(track, bad_frame);
    REQUIRE(!init_status.ok());
    REQUIRE(init_status.code() == ErrorCode::FEATURE_EXTRACTION_FAILED);

    // Initialize on a valid frame so we can exercise predict/update exception paths.
    cv::Mat frame = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::rectangle(frame, cv::Point(10, 10), cv::Point(60, 60), cv::Scalar(255, 255, 255), -1);
    REQUIRE(tracker.value()->init(track, frame).ok());

    auto pred = tracker.value()->predict(track, bad_frame);
    REQUIRE(!pred.has_value());
    REQUIRE(pred.error().code() == ErrorCode::TRACKING_LOST);

    auto update_status = tracker.value()->update(track, bad_frame, track.bbox);
    REQUIRE(!update_status.ok());
    REQUIRE(update_status.code() == ErrorCode::FEATURE_EXTRACTION_FAILED);
}

TEST_CASE("KCFTracker update refreshes the filter", "[tracker]") {
    KCFTracker::Config cfg;
    cfg.template_size = cv::Size(64, 64);
    cfg.learning_rate = 0.5f;
    auto tracker = KCFTracker::create(cfg);
    REQUIRE(tracker.has_value());

    Track track;
    track.bbox = Rect2f(100, 100, 40, 40);
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(frame, cv::Point(100, 100), cv::Point(140, 140), cv::Scalar(255, 255, 255), -1);

    REQUIRE(tracker.value()->init(track, frame).ok());
    REQUIRE(!track.correlation_filter.empty());

    cv::Mat old_filter = track.correlation_filter.clone();

    // Use a patch with different intensity so the filter actually changes.
    cv::Mat shifted_frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(shifted_frame, cv::Point(110, 110), cv::Point(150, 150),
                  cv::Scalar(128, 128, 128), -1);
    Rect2f updated_bbox(110, 110, 40, 40);

    auto status = tracker.value()->update(track, shifted_frame, updated_bbox);
    REQUIRE(status.ok());
    REQUIRE(!track.correlation_filter.empty());
    REQUIRE(cv::norm(old_filter, track.correlation_filter, cv::NORM_L2) > 0.0f);
}
