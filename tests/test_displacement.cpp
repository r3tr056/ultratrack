// tests/test_displacement.cpp
#include "tracking/displacement_predictor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace ultratrack;

void test_initial_state() {
    DisplacementPredictor pred;
    assert(!pred.is_ready());
    std::cout << "  PASS: test_initial_state\n";
}

void test_history_buildup() {
    DisplacementPredictor pred;
    
    pred.predict({100, 100});
    assert(!pred.is_ready());
    
    pred.predict({110, 100});
    assert(pred.is_ready());
    
    std::cout << "  PASS: test_history_buildup\n";
}

void test_linear_motion_prediction() {
    DisplacementPredictor pred;
    
    // Build history: moving right at 10 pixels/frame
    pred.predict({100, 100});  // t-2
    pred.predict({110, 100});  // t-1
    
    // Current position, expect prediction ahead
    auto predicted = pred.predict({120, 100});  // t
    
    // Expected: 120 + 0.8 * (110 - 100) = 128
    float expected_x = 120 + 0.8f * 10;
    assert(std::abs(predicted.x - expected_x) < 1.0f);
    assert(std::abs(predicted.y - 100) < 1.0f);
    
    std::cout << "  PASS: test_linear_motion_prediction\n";
}

void test_stationary_no_prediction() {
    DisplacementConfig config;
    config.min_threshold = 2.0f;
    DisplacementPredictor pred(config);
    
    // Stationary target (displacement < min_threshold)
    pred.predict({100, 100});
    pred.predict({100.5f, 100});  // 0.5 pixel movement
    
    auto predicted = pred.predict({101, 100});
    
    // Should return current position (no prediction)
    assert(std::abs(predicted.x - 101) < 0.5f);
    
    std::cout << "  PASS: test_stationary_no_prediction\n";
}

void test_reset() {
    DisplacementPredictor pred;
    
    pred.predict({100, 100});
    pred.predict({110, 100});
    assert(pred.is_ready());
    
    pred.reset();
    assert(!pred.is_ready());
    
    std::cout << "  PASS: test_reset\n";
}

void test_disabled_prediction() {
    DisplacementConfig config;
    config.enabled = false;
    DisplacementPredictor pred(config);
    
    pred.predict({100, 100});
    pred.predict({110, 100});
    
    auto predicted = pred.predict({120, 100});
    
    // Should return current position when disabled
    assert(std::abs(predicted.x - 120) < 0.01f);
    
    std::cout << "  PASS: test_disabled_prediction\n";
}

int main() {
    std::cout << "Running displacement predictor tests...\n";
    
    test_initial_state();
    test_history_buildup();
    test_linear_motion_prediction();
    test_stationary_no_prediction();
    test_reset();
    test_disabled_prediction();
    
    std::cout << "All displacement predictor tests passed!\n";
    return 0;
}
