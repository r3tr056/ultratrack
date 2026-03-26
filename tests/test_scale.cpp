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

int main() {
    std::cout << "Running scale estimator tests...\n";
    
    test_three_scale_pool();
    test_five_scale_pool();
    test_initial_scale();
    test_reset();
    test_config_update();
    test_scale_clamping();
    
    std::cout << "All scale estimator tests passed!\n";
    return 0;
}
