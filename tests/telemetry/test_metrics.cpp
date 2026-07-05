#include <catch2/catch_test_macros.hpp>
#include <ultratrack/telemetry/metrics.hpp>
#include <thread>
#include <chrono>

using namespace ultratrack;

TEST_CASE("StageTimer records latency", "[telemetry]") {
    MetricsCollector collector;
    {
        StageTimer t(collector, "detect");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    auto metrics = collector.snapshot();
    REQUIRE(metrics.count("detect_count") == 1);
    REQUIRE(metrics.at("detect_count") >= 1.0);
    REQUIRE(metrics.at("detect_p50_ms") >= 0.0);
}

TEST_CASE("Empty snapshot returns no metrics", "[telemetry]") {
    MetricsCollector collector;
    auto metrics = collector.snapshot();
    REQUIRE(metrics.empty());
}

TEST_CASE("Multiple stages are tracked independently", "[telemetry]") {
    MetricsCollector collector;
    collector.record("detect", 10.0);
    collector.record("detect", 20.0);
    collector.record("track", 5.0);

    auto metrics = collector.snapshot();

    REQUIRE(metrics.count("detect_count") == 1);
    REQUIRE(metrics.at("detect_count") == 2.0);
    REQUIRE(metrics.at("detect_avg_ms") == 15.0);

    REQUIRE(metrics.count("track_count") == 1);
    REQUIRE(metrics.at("track_count") == 1.0);
    REQUIRE(metrics.at("track_avg_ms") == 5.0);

    REQUIRE(metrics.count("detect_p50_ms") == 1);
    REQUIRE(metrics.count("track_p50_ms") == 1);
}

TEST_CASE("Percentile is computed from ordered values", "[telemetry]") {
    MetricsCollector collector;
    collector.record("stage", 30.0);
    collector.record("stage", 10.0);
    collector.record("stage", 20.0);

    auto metrics = collector.snapshot();
    REQUIRE(metrics.at("stage_count") == 3.0);
    REQUIRE(metrics.at("stage_p50_ms") == 20.0);
    REQUIRE(metrics.at("stage_avg_ms") == 20.0);
}
