// tests/test_scale.cpp
#include "tracking/scale_estimator.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace ultratrack;

void test_three_scale_pool() {
    ScaleConfig config;
    config.num_scales = 3;
    config.scale_step = 0.04f;
    
    ScaleEstimator est(config);
    auto pool = est.get_scale_pool();
    
    assert(pool.size() == 3);
    assert(std::abs(pool[0] - 0.92f) < 0.001f);  // 1 - 2*0.04
    assert(std::abs(pool[1] - 1.00f) < 0.001f);
    assert(std::abs(pool[2] - 1.08f) < 0.001f);  // 1 + 2*0.04
    
    std::cout << "  PASS: test_three_scale_pool\n";
}

void test_five_scale_pool() {
    ScaleConfig config;
    config.num_scales = 5;
    config.scale_step = 0.02f;
    
    ScaleEstimator est(config);
    auto pool = est.get_scale_pool();
    
    assert(pool.size() == 5);
    assert(std::abs(pool[2] - 1.00f) < 0.001f);  // Center is 1.0
    
    std::cout << "  PASS: test_five_scale_pool\n";
}

void test_initial_scale() {
    ScaleEstimator est;
    assert(std::abs(est.get_current_scale() - 1.0f) < 0.001f);
    
    std::cout << "  PASS: test_initial_scale\n";
}

void test_reset() {
    ScaleEstimator est;
    
    // Manually set a different scale (via estimation would require frames)
    est.reset();
    
    assert(std::abs(est.get_current_scale() - 1.0f) < 0.001f);
    
    std::cout << "  PASS: test_reset\n";
}

void test_config_update() {
    ScaleEstimator est;
    
    ScaleConfig new_config;
    new_config.num_scales = 7;
    new_config.scale_step = 0.03f;
    
    est.set_config(new_config);
    
    auto pool = est.get_scale_pool();
    assert(pool.size() == 7);
    
    std::cout << "  PASS: test_config_update\n";
}

void test_scale_clamping() {
    ScaleConfig config;
    config.min_scale = 0.5f;
    config.max_scale = 2.0f;

    ScaleEstimator est(config);

    // The clamping happens during estimate(), which needs a frame
    // Just verify config is stored
    assert(std::abs(est.get_config().min_scale - 0.5f) < 0.001f);
    assert(std::abs(est.get_config().max_scale - 2.0f) < 0.001f);

    std::cout << "  PASS: test_scale_clamping\n";
}

// --- Additional regression tests ---

void test_seven_scale_pool() {
    ScaleConfig config;
    config.num_scales = 7;
    config.scale_step = 0.03f;

    ScaleEstimator est(config);
    auto pool = est.get_scale_pool();

    assert(pool.size() == 7);
    // Center should be 1.0
    assert(std::abs(pool[3] - 1.00f) < 0.001f);
    // Symmetric around 1.0
    assert(std::abs(pool[0] - (1.0f - 3 * 0.03f)) < 0.001f);
    assert(std::abs(pool[6] - (1.0f + 3 * 0.03f)) < 0.001f);

    std::cout << "  PASS: test_seven_scale_pool\n";
}

void test_scale_pool_symmetry() {
    ScaleConfig config;
    config.num_scales = 5;
    config.scale_step = 0.04f;

    ScaleEstimator est(config);
    auto pool = est.get_scale_pool();

    assert(pool.size() == 5);
    // Pool should be symmetric: pool[i] + pool[n-1-i] == 2.0
    for (size_t i = 0; i < pool.size() / 2; i++) {
        assert(std::abs((pool[i] + pool[pool.size() - 1 - i]) - 2.0f) < 0.001f);
    }

    std::cout << "  PASS: test_scale_pool_symmetry\n";
}

void test_estimate_with_empty_frame() {
    ScaleEstimator est;
    cv::Mat empty_frame;
    cv::Point2f pos(50, 50);
    cv::Size2f base(100, 100);
    cv::Mat filter;
    cv::Mat model;

    float result = est.estimate(empty_frame, pos, base, filter, model);

    // Should return current scale (1.0) without crash
    assert(std::abs(result - 1.0f) < 0.001f);

    std::cout << "  PASS: test_estimate_with_empty_frame\n";
}

void test_default_config_values() {
    ScaleConfig config;
    assert(config.num_scales == 3);
    assert(std::abs(config.scale_step - 0.04f) < 0.001f);
    assert(std::abs(config.min_scale - 0.2f) < 0.001f);
    assert(std::abs(config.max_scale - 5.0f) < 0.001f);
    assert(std::abs(config.scale_penalty - 0.975f) < 0.001f);

    std::cout << "  PASS: test_default_config_values\n";
}

int main() {
    std::cout << "Running scale estimator tests...\n";

    test_three_scale_pool();
    test_five_scale_pool();
    test_initial_scale();
    test_reset();
    test_config_update();
    test_scale_clamping();
    test_seven_scale_pool();
    test_scale_pool_symmetry();
    test_estimate_with_empty_frame();
    test_default_config_values();

    std::cout << "All scale estimator tests passed!\n";
    return 0;
}
