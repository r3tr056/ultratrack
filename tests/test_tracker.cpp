// tests/test_tracker.cpp
#include "ultratrack.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iostream>

using namespace ultratrack;

TEST_CASE("Tracker config modes", "[tracker]") {
    TrackerConfig config;
    config.mode = TrackingMode::ACCURATE;
    config.displacement.kappa = 0.8f;
    config.scale.num_scales = 3;

    REQUIRE(config.mode == TrackingMode::ACCURATE);
    REQUIRE(config.displacement.kappa == 0.8f);
    REQUIRE(config.scale.num_scales == 3);
}

TEST_CASE("Displacement config integration", "[tracker]") {
    DisplacementConfig config;
    config.enabled = true;
    config.kappa = 0.9f;
    config.min_threshold = 1.0f;
    config.max_threshold = 150.0f;

    DisplacementPredictor pred(config);

    REQUIRE(pred.get_config().kappa == 0.9f);
    REQUIRE(pred.get_config().min_threshold == 1.0f);
}

TEST_CASE("Scale config integration", "[tracker]") {
    ScaleConfig config;
    config.num_scales = 5;
    config.scale_step = 0.02f;
    config.scale_penalty = 0.98f;

    ScaleEstimator est(config);

    REQUIRE(est.get_config().num_scales == 5);
    REQUIRE(est.get_scale_pool().size() == 5);
}

TEST_CASE("Feature extractor integration", "[tracker]") {
    MultiFeatureExtractor ext(TrackingMode::ACCURATE);

    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(frame, cv::Rect(100, 100, 64, 64), cv::Scalar(0, 0, 255), -1);

    cv::Mat patch = frame(cv::Rect(100, 100, 64, 64));
    cv::Mat features = ext.extract(patch);

    REQUIRE(!features.empty());
}

TEST_CASE("Tracking pipeline components", "[tracker]") {
    TrackerConfig config;
    config.mode = TrackingMode::BALANCED;

    auto feature_ext = std::make_unique<MultiFeatureExtractor>(config.mode, config.feature);
    auto scale_est = std::make_unique<ScaleEstimator>(config.scale);
    auto disp_pred = std::make_unique<DisplacementPredictor>(config.displacement);

    REQUIRE(feature_ext->total_dimensions() == 32);
    REQUIRE(scale_est->get_scale_pool().size() == 3);
    REQUIRE(!disp_pred->is_ready());
}
