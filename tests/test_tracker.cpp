// tests/test_tracker.cpp
#include "ultratrack.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>
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

// --- Additional regression and integration tests ---

TEST_CASE("tracker config defaults", "[tracker][regression]") {
    TrackerConfig config;
    REQUIRE(config.mode == TrackingMode::ACCURATE);
    REQUIRE(std::abs(config.learning_rate - 0.01f) < 0.001f);
    REQUIRE(std::abs(config.lambda - 0.01f) < 0.001f);
    REQUIRE(std::abs(config.sigma - 2.0f) < 0.001f);
    REQUIRE(config.displacement.enabled == true);
    REQUIRE(std::abs(config.displacement.kappa - 0.8f) < 0.001f);
    REQUIRE(config.scale.num_scales == 3);
}

TEST_CASE("mode switching preserves config", "[tracker][regression]") {
    TrackerConfig config;
    config.mode = TrackingMode::FAST;

    auto feature_ext = std::make_unique<MultiFeatureExtractor>(config.mode, config.feature);

    REQUIRE(feature_ext->get_mode() == TrackingMode::FAST);
    REQUIRE(feature_ext->total_dimensions() == 31);

    // Switch to ACCURATE
    feature_ext->set_mode(TrackingMode::ACCURATE);
    REQUIRE(feature_ext->get_mode() == TrackingMode::ACCURATE);
    REQUIRE(feature_ext->total_dimensions() == 42);

    // Switch back to FAST
    feature_ext->set_mode(TrackingMode::FAST);
    REQUIRE(feature_ext->get_mode() == TrackingMode::FAST);
    REQUIRE(feature_ext->total_dimensions() == 31);
}

TEST_CASE("displacement with scale", "[tracker][regression]") {
    // Test that displacement predictor and scale estimator work together
    DisplacementPredictor pred;
    ScaleEstimator scale_est;

    // Simulate motion
    pred.predict({100, 100});
    pred.predict({110, 105});

    auto predicted = pred.predict({120, 110});

    // Verify prediction applied
    REQUIRE(predicted.x > 120.0f);  // Should be ahead in motion direction
    REQUIRE(predicted.y > 110.0f);

    // Scale should be 1.0 initially
    REQUIRE(std::abs(scale_est.get_current_scale() - 1.0f) < 0.001f);
}

TEST_CASE("multi feature with real patch", "[tracker][regression]") {
    // Create a synthetic "real-world" patch with color gradient
    cv::Mat patch(64, 64, CV_8UC3);
    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            patch.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(x * 4),   // B gradient
                static_cast<uchar>(y * 4),   // G gradient
                128                           // R constant
            );
        }
    }

    // Test all three modes produce valid features
    for (auto mode : {TrackingMode::FAST, TrackingMode::BALANCED, TrackingMode::ACCURATE}) {
        MultiFeatureExtractor ext(mode);
        cv::Mat features = ext.extract(patch);
        REQUIRE(!features.empty());

        // Verify no NaN or Inf values
        cv::Mat flat = features.reshape(1, features.total() * features.channels());
        for (int i = 0; i < flat.rows; i++) {
            float val = flat.at<float>(i, 0);
            REQUIRE(!std::isnan(val));
            REQUIRE(!std::isinf(val));
        }
    }
}

TEST_CASE("track copy semantics", "[tracker][regression]") {
    // Test that Track copy constructor properly copies all fields including unique_ptr
    Track original;
    original.id = 42;
    original.bbox = cv::Rect2f(10, 20, 30, 40);
    original.confidence = 0.95f;
    original.age = 5;
    original.hits = 3;
    original.time_since_update = 0;
    original.is_activated = true;
    original.current_scale = 1.5f;
    original.base_size = cv::Size2f(30, 40);
    original.predicted_center = cv::Point2f(25, 40);
    original.displacement_predictor = std::make_unique<DisplacementPredictor>();
    original.state = (cv::Mat_<float>(8, 1) << 25, 40, 30, 40, 0, 0, 0, 0);
    original.covariance = cv::Mat::eye(8, 8, CV_32F);

    // Feed some history to the displacement predictor
    original.displacement_predictor->predict({10, 10});
    original.displacement_predictor->predict({20, 20});

    // Copy
    Track copy(original);

    REQUIRE(copy.id == original.id);
    REQUIRE(copy.bbox == original.bbox);
    REQUIRE(copy.confidence == original.confidence);
    REQUIRE(copy.current_scale == original.current_scale);
    REQUIRE(copy.displacement_predictor != nullptr);
    REQUIRE(copy.displacement_predictor.get() != original.displacement_predictor.get()); // Different pointer
    REQUIRE(copy.displacement_predictor->is_ready());  // History preserved
}

TEST_CASE("all modes dimensions", "[tracker][regression]") {
    // Verify dimension counts match spec
    REQUIRE(MultiFeatureExtractor(TrackingMode::FAST).total_dimensions() == 31);
    REQUIRE(MultiFeatureExtractor(TrackingMode::BALANCED).total_dimensions() == 32);
    REQUIRE(MultiFeatureExtractor(TrackingMode::ACCURATE).total_dimensions() == 42);
}
