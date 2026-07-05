// tests/test_displacement.cpp
#include "tracking/displacement_predictor.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>

using namespace ultratrack;

TEST_CASE("Displacement predictor initial state", "[displacement]") {
    DisplacementPredictor pred;
    REQUIRE(!pred.is_ready());
}

TEST_CASE("Displacement predictor history buildup", "[displacement]") {
    DisplacementPredictor pred;

    pred.predict({100, 100});
    REQUIRE(!pred.is_ready());

    pred.predict({110, 100});
    REQUIRE(pred.is_ready());
}

TEST_CASE("Displacement predictor linear motion", "[displacement]") {
    DisplacementPredictor pred;

    // Build history: moving right at 10 pixels/frame
    pred.predict({100, 100});  // t-2
    pred.predict({110, 100});  // t-1

    // Current position, expect prediction ahead
    auto predicted = pred.predict({120, 100});  // t

    // Expected: 120 + 0.8 * (110 - 100) = 128
    float expected_x = 120 + 0.8f * 10;
    REQUIRE(std::abs(predicted.x - expected_x) < 1.0f);
    REQUIRE(std::abs(predicted.y - 100) < 1.0f);
}

TEST_CASE("Displacement predictor stationary target", "[displacement]") {
    DisplacementConfig config;
    config.min_threshold = 2.0f;
    DisplacementPredictor pred(config);

    // Stationary target (displacement < min_threshold)
    pred.predict({100, 100});
    pred.predict({100.5f, 100});  // 0.5 pixel movement

    auto predicted = pred.predict({101, 100});

    // Should return current position (no prediction)
    REQUIRE(std::abs(predicted.x - 101) < 0.5f);
}

TEST_CASE("Displacement predictor reset", "[displacement]") {
    DisplacementPredictor pred;

    pred.predict({100, 100});
    pred.predict({110, 100});
    REQUIRE(pred.is_ready());

    pred.reset();
    REQUIRE(!pred.is_ready());
}

TEST_CASE("Displacement predictor disabled", "[displacement]") {
    DisplacementConfig config;
    config.enabled = false;
    DisplacementPredictor pred(config);

    pred.predict({100, 100});
    pred.predict({110, 100});

    auto predicted = pred.predict({120, 100});

    // Should return current position when disabled
    REQUIRE(std::abs(predicted.x - 120) < 0.01f);
}
