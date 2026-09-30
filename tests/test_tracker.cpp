// tests/test_tracker.cpp
#include "ultratrack.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
#include <memory>

using namespace ultratrack;

void test_tracker_config_modes() {
    std::cout << "  Note: Skipping model-dependent tests (no ONNX model)\n";
    
    // Test config creation
    TrackerConfig config;
    config.mode = TrackingMode::ACCURATE;
    config.displacement.kappa = 0.8f;
    config.scale.num_scales = 3;
    
    assert(config.mode == TrackingMode::ACCURATE);
    assert(config.displacement.kappa == 0.8f);
    assert(config.scale.num_scales == 3);
    
    std::cout << "  PASS: test_tracker_config_modes\n";
}

void test_displacement_config() {
    DisplacementConfig config;
    config.enabled = true;
    config.kappa = 0.9f;
    config.min_threshold = 1.0f;
    config.max_threshold = 150.0f;
    
    DisplacementPredictor pred(config);
    
    assert(pred.get_config().kappa == 0.9f);
    assert(pred.get_config().min_threshold == 1.0f);
    
    std::cout << "  PASS: test_displacement_config\n";
}

void test_scale_config() {
    ScaleConfig config;
    config.num_scales = 5;
    config.scale_step = 0.02f;
    config.scale_penalty = 0.98f;
    
    ScaleEstimator est(config);
    
    assert(est.get_config().num_scales == 5);
    assert(est.get_scale_pool().size() == 5);
    
    std::cout << "  PASS: test_scale_config\n";
}

void test_feature_extractor_integration() {
    MultiFeatureExtractor ext(TrackingMode::ACCURATE);
    
    // Create synthetic frame
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(frame, cv::Rect(100, 100, 64, 64), cv::Scalar(0, 0, 255), -1);
    
    cv::Mat patch = frame(cv::Rect(100, 100, 64, 64));
    cv::Mat features = ext.extract(patch);
    
    assert(!features.empty());
    
    std::cout << "  PASS: test_feature_extractor_integration\n";
}

void test_tracking_pipeline_components() {
    // Test that all components can be instantiated together
    TrackerConfig config;
    config.mode = TrackingMode::BALANCED;

    auto feature_ext = std::make_unique<MultiFeatureExtractor>(config.mode, config.feature);
    auto scale_est = std::make_unique<ScaleEstimator>(config.scale);
    auto disp_pred = std::make_unique<DisplacementPredictor>(config.displacement);

    assert(feature_ext->total_dimensions() == 32);
    assert(scale_est->get_scale_pool().size() == 3);
    assert(!disp_pred->is_ready());

    std::cout << "  PASS: test_tracking_pipeline_components\n";
}

// --- Additional regression and integration tests ---

void test_tracker_config_defaults() {
    TrackerConfig config;
    assert(config.mode == TrackingMode::ACCURATE);
    assert(std::abs(config.learning_rate - 0.01f) < 0.001f);
    assert(std::abs(config.lambda - 0.01f) < 0.001f);
    assert(std::abs(config.sigma - 2.0f) < 0.001f);
    assert(config.displacement.enabled == true);
    assert(std::abs(config.displacement.kappa - 0.8f) < 0.001f);
    assert(config.scale.num_scales == 3);

    std::cout << "  PASS: test_tracker_config_defaults\n";
}

void test_mode_switching_preserves_config() {
    TrackerConfig config;
    config.mode = TrackingMode::FAST;

    auto feature_ext = std::make_unique<MultiFeatureExtractor>(config.mode, config.feature);

    assert(feature_ext->get_mode() == TrackingMode::FAST);
    assert(feature_ext->total_dimensions() == 31);

    // Switch to ACCURATE
    feature_ext->set_mode(TrackingMode::ACCURATE);
    assert(feature_ext->get_mode() == TrackingMode::ACCURATE);
    assert(feature_ext->total_dimensions() == 42);

    // Switch back to FAST
    feature_ext->set_mode(TrackingMode::FAST);
    assert(feature_ext->get_mode() == TrackingMode::FAST);
    assert(feature_ext->total_dimensions() == 31);

    std::cout << "  PASS: test_mode_switching_preserves_config\n";
}

void test_displacement_with_scale() {
    // Test that displacement predictor and scale estimator work together
    DisplacementPredictor pred;
    ScaleEstimator scale_est;

    // Simulate motion
    pred.predict({100, 100});
    pred.predict({110, 105});

    auto predicted = pred.predict({120, 110});

    // Verify prediction applied
    assert(predicted.x > 120.0f);  // Should be ahead in motion direction
    assert(predicted.y > 110.0f);

    // Scale should be 1.0 initially
    assert(std::abs(scale_est.get_current_scale() - 1.0f) < 0.001f);

    std::cout << "  PASS: test_displacement_with_scale\n";
}

void test_multi_feature_with_real_patch() {
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
        assert(!features.empty());

        // Verify no NaN or Inf values
        cv::Mat flat = features.reshape(1, features.total() * features.channels());
        for (int i = 0; i < flat.rows; i++) {
            float val = flat.at<float>(i, 0);
            assert(!std::isnan(val));
            assert(!std::isinf(val));
        }
    }

    std::cout << "  PASS: test_multi_feature_with_real_patch\n";
}

void test_track_copy_semantics() {
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

    assert(copy.id == original.id);
    assert(copy.bbox == original.bbox);
    assert(copy.confidence == original.confidence);
    assert(copy.current_scale == original.current_scale);
    assert(copy.displacement_predictor != nullptr);
    assert(copy.displacement_predictor.get() != original.displacement_predictor.get()); // Different pointer
    assert(copy.displacement_predictor->is_ready());  // History preserved

    std::cout << "  PASS: test_track_copy_semantics\n";
}

void test_all_modes_dimensions() {
    // Verify dimension counts match spec
    assert(MultiFeatureExtractor(TrackingMode::FAST).total_dimensions() == 31);
    assert(MultiFeatureExtractor(TrackingMode::BALANCED).total_dimensions() == 32);
    assert(MultiFeatureExtractor(TrackingMode::ACCURATE).total_dimensions() == 42);

    std::cout << "  PASS: test_all_modes_dimensions\n";
}

int main() {
    std::cout << "Running tracker integration tests...\n";

    test_tracker_config_modes();
    test_displacement_config();
    test_scale_config();
    test_feature_extractor_integration();
    test_tracking_pipeline_components();
    test_tracker_config_defaults();
    test_mode_switching_preserves_config();
    test_displacement_with_scale();
    test_multi_feature_with_real_patch();
    test_track_copy_semantics();
    test_all_modes_dimensions();

    std::cout << "All tracker integration tests passed!\n";
    return 0;
}
