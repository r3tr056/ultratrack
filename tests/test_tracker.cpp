// tests/test_tracker.cpp
#include "ultratrack.hpp"
#include <cassert>
#include <iostream>

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

int main() {
    std::cout << "Running tracker integration tests...\n";
    
    test_tracker_config_modes();
    test_displacement_config();
    test_scale_config();
    test_feature_extractor_integration();
    test_tracking_pipeline_components();
    
    std::cout << "All tracker integration tests passed!\n";
    return 0;
}
