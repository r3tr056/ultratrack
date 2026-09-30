// tests/test_scale.cpp
#include <ultratrack/tracker/scale_estimator.hpp>
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

// --- Additional regression tests ---

TEST_CASE("seven scale pool", "[scale][regression]") {
    ScaleConfig config;
    config.num_scales = 7;
    config.scale_step = 0.03f;

    ScaleEstimator est(config);
    auto pool = est.get_scale_pool();

    REQUIRE(pool.size() == 7);
    // Center should be 1.0
    REQUIRE(std::abs(pool[3] - 1.00f) < 0.001f);
    // Symmetric around 1.0
    REQUIRE(std::abs(pool[0] - (1.0f - 3 * 0.03f)) < 0.001f);
    REQUIRE(std::abs(pool[6] - (1.0f + 3 * 0.03f)) < 0.001f);
}

TEST_CASE("scale pool symmetry", "[scale][regression]") {
    ScaleConfig config;
    config.num_scales = 5;
    config.scale_step = 0.04f;

    ScaleEstimator est(config);
    auto pool = est.get_scale_pool();

    REQUIRE(pool.size() == 5);
    // Pool should be symmetric: pool[i] + pool[n-1-i] == 2.0
    for (size_t i = 0; i < pool.size() / 2; i++) {
        REQUIRE(std::abs((pool[i] + pool[pool.size() - 1 - i]) - 2.0f) < 0.001f);
    }
}

TEST_CASE("estimate with empty frame", "[scale][regression]") {
    ScaleEstimator est;
    cv::Mat empty_frame;
    cv::Point2f pos(50, 50);
    cv::Size2f base(100, 100);
    cv::Mat filter;
    cv::Mat model;

    float result = est.estimate(empty_frame, pos, base, filter, model);

    // Should return current scale (1.0) without crash
    REQUIRE(std::abs(result - 1.0f) < 0.001f);
}

TEST_CASE("default config values", "[scale][regression]") {
    ScaleConfig config;
    REQUIRE(config.num_scales == 3);
    REQUIRE(std::abs(config.scale_step - 0.04f) < 0.001f);
    REQUIRE(std::abs(config.min_scale - 0.2f) < 0.001f);
    REQUIRE(std::abs(config.max_scale - 5.0f) < 0.001f);
    REQUIRE(std::abs(config.scale_penalty - 0.975f) < 0.001f);
}
