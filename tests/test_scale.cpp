// tests/test_scale.cpp
#include "tracking/scale_estimator.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>

using namespace ultratrack;

TEST_CASE("Three-scale pool", "[scale]") {
    ScaleConfig config;
    config.num_scales = 3;
    config.scale_step = 0.04f;

    ScaleEstimator est(config);
    auto pool = est.get_scale_pool();

    REQUIRE(pool.size() == 3);
    REQUIRE(std::abs(pool[0] - 0.92f) < 0.001f);  // 1 - 2*0.04
    REQUIRE(std::abs(pool[1] - 1.00f) < 0.001f);
    REQUIRE(std::abs(pool[2] - 1.08f) < 0.001f);  // 1 + 2*0.04
}

TEST_CASE("Five-scale pool", "[scale]") {
    ScaleConfig config;
    config.num_scales = 5;
    config.scale_step = 0.02f;

    ScaleEstimator est(config);
    auto pool = est.get_scale_pool();

    REQUIRE(pool.size() == 5);
    REQUIRE(std::abs(pool[2] - 1.00f) < 0.001f);  // Center is 1.0
}

TEST_CASE("Initial scale", "[scale]") {
    ScaleEstimator est;
    REQUIRE(std::abs(est.get_current_scale() - 1.0f) < 0.001f);
}

TEST_CASE("Scale estimator reset", "[scale]") {
    ScaleEstimator est;
    est.reset();
    REQUIRE(std::abs(est.get_current_scale() - 1.0f) < 0.001f);
}

TEST_CASE("Scale config update", "[scale]") {
    ScaleEstimator est;

    ScaleConfig new_config;
    new_config.num_scales = 7;
    new_config.scale_step = 0.03f;

    est.set_config(new_config);

    auto pool = est.get_scale_pool();
    REQUIRE(pool.size() == 7);
}

TEST_CASE("Scale clamping config", "[scale]") {
    ScaleConfig config;
    config.min_scale = 0.5f;
    config.max_scale = 2.0f;

    ScaleEstimator est(config);

    REQUIRE(std::abs(est.get_config().min_scale - 0.5f) < 0.001f);
    REQUIRE(std::abs(est.get_config().max_scale - 2.0f) < 0.001f);
}
