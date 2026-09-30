#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <ultratrack/tracker/kalman.hpp>

using namespace ultratrack;

TEST_CASE("KalmanFilter initializes from bbox", "[tracker]") {
    Rect2f bbox(100.0f, 100.0f, 40.0f, 60.0f);
    auto kf_result = KalmanFilter::create(bbox);
    REQUIRE(kf_result.has_value());
    auto& kf = kf_result.value();

    auto result = kf.stateBBox();
    REQUIRE(result.has_value());
    Rect2f state = result.value();
    REQUIRE(state.x == Catch::Approx(bbox.x));
    REQUIRE(state.y == Catch::Approx(bbox.y));
    REQUIRE(state.width == Catch::Approx(bbox.width));
    REQUIRE(state.height == Catch::Approx(bbox.height));
}

TEST_CASE("KalmanFilter predict moves the state", "[tracker]") {
    Rect2f bbox(100.0f, 100.0f, 40.0f, 40.0f);
    auto kf_result = KalmanFilter::create(bbox);
    REQUIRE(kf_result.has_value());
    auto& kf = kf_result.value();

    // Give the filter a velocity by updating with a shifted measurement.
    Rect2f m1(110.0f, 105.0f, 40.0f, 40.0f);
    REQUIRE(kf.update(m1).ok());
    REQUIRE(kf.predict().ok());

    auto result = kf.stateBBox();
    REQUIRE(result.has_value());
    Rect2f predicted = result.value();
    REQUIRE(predicted.x > bbox.x);
    REQUIRE(predicted.y > bbox.y);
}

TEST_CASE("KalmanFilter update corrects the state", "[tracker]") {
    Rect2f bbox(100.0f, 100.0f, 40.0f, 40.0f);
    auto kf_result = KalmanFilter::create(bbox);
    REQUIRE(kf_result.has_value());
    auto& kf = kf_result.value();

    REQUIRE(kf.predict().ok());
    Rect2f measurement(120.0f, 110.0f, 42.0f, 38.0f);
    REQUIRE(kf.update(measurement).ok());

    auto result = kf.stateBBox();
    REQUIRE(result.has_value());
    Rect2f corrected = result.value();
    REQUIRE(corrected.x > bbox.x);
    REQUIRE(corrected.x < measurement.x);
    REQUIRE(corrected.width > bbox.width);
    REQUIRE(corrected.width < measurement.width);
}

TEST_CASE("KalmanFilter handles zero-size bbox", "[tracker]") {
    Rect2f bbox(50.0f, 50.0f, 0.0f, 0.0f);
    auto kf_result = KalmanFilter::create(bbox);
    REQUIRE(kf_result.has_value());
    auto& kf = kf_result.value();

    auto result = kf.stateBBox();
    REQUIRE(result.has_value());
    Rect2f state = result.value();
    REQUIRE(state.x == Catch::Approx(bbox.x));
    REQUIRE(state.y == Catch::Approx(bbox.y));
    REQUIRE(state.width == Catch::Approx(0.0f));
    REQUIRE(state.height == Catch::Approx(0.0f));
}

TEST_CASE("KalmanFilter public methods do not throw on invalid input", "[tracker]") {
    Rect2f bbox(50.0f, 50.0f, 0.0f, 0.0f);
    auto kf_result = KalmanFilter::create(bbox);
    REQUIRE(kf_result.has_value());
    auto& kf = kf_result.value();

    REQUIRE_NOTHROW(kf.predict());
    REQUIRE(kf.predict().ok());

    Status update_status;
    REQUIRE_NOTHROW(update_status = kf.update(bbox));
    REQUIRE(update_status.code() != ErrorCode::UNKNOWN);

    auto state_result = kf.stateBBox();
    REQUIRE_NOTHROW(state_result = kf.stateBBox());
    REQUIRE(state_result.has_value());
}
