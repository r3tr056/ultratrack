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

// --- Additional regression tests ---

void test_max_threshold_clamping() {
    DisplacementConfig config;
    config.max_threshold = 50.0f;
    DisplacementPredictor pred(config);

    // Build history with large displacement (>max_threshold)
    pred.predict({100, 100});
    pred.predict({200, 100});  // 100px displacement > max_threshold=50

    auto predicted = pred.predict({300, 100});

    // Should return current position (displacement exceeds max threshold)
    assert(std::abs(predicted.x - 300) < 0.5f);
    assert(std::abs(predicted.y - 100) < 0.5f);

    std::cout << "  PASS: test_max_threshold_clamping\n";
}

void test_diagonal_motion() {
    DisplacementPredictor pred;

    pred.predict({100, 100});
    pred.predict({110, 110});  // Moving diagonally

    auto predicted = pred.predict({120, 120});

    // Expected: 120 + 0.8*10 = 128, 120 + 0.8*10 = 128
    assert(std::abs(predicted.x - 128) < 1.0f);
    assert(std::abs(predicted.y - 128) < 1.0f);

    std::cout << "  PASS: test_diagonal_motion\n";
}

void test_velocity_tracking() {
    DisplacementPredictor pred;

    pred.predict({100, 100});
    pred.predict({115, 100});

    pred.predict({130, 100});  // displacement = 15

    auto vel = pred.get_velocity();
    assert(std::abs(vel.x - 15.0f) < 0.5f);

    std::cout << "  PASS: test_velocity_tracking\n";
}

void test_config_update() {
    DisplacementPredictor pred;

    DisplacementConfig new_config;
    new_config.kappa = 0.5f;
    new_config.min_threshold = 5.0f;
    pred.set_config(new_config);

    assert(std::abs(pred.get_config().kappa - 0.5f) < 0.001f);
    assert(std::abs(pred.get_config().min_threshold - 5.0f) < 0.001f);

    std::cout << "  PASS: test_config_update\n";
}

void test_reset_then_predict() {
    DisplacementPredictor pred;

    pred.predict({100, 100});
    pred.predict({110, 100});
    assert(pred.is_ready());

    pred.reset();
    assert(!pred.is_ready());

    // After reset, should return same position until history rebuilds
    auto predicted = pred.predict({200, 200});
    assert(std::abs(predicted.x - 200) < 0.01f);

    std::cout << "  PASS: test_reset_then_predict\n";
}

int main() {
    std::cout << "Running displacement predictor tests...\n";

    test_initial_state();
    test_history_buildup();
    test_linear_motion_prediction();
    test_stationary_no_prediction();
    test_reset();
    test_disabled_prediction();
    test_max_threshold_clamping();
    test_diagonal_motion();
    test_velocity_tracking();
    test_config_update();
    test_reset_then_predict();

    std::cout << "All displacement predictor tests passed!\n";
    return 0;
}
