#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <mutex>

namespace ultratrack {

/// Thread-safe collector for per-stage latency measurements.
class MetricsCollector {
public:
    /// Records a latency sample for the given stage.
    void record(const std::string& stage, double latency_ms);

    /// Returns a snapshot of aggregated metrics.
    /// Keys are formatted as "<stage>_count", "<stage>_p50_ms", "<stage>_avg_ms".
    std::unordered_map<std::string, double> snapshot() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::vector<double>> data_;
};

/// RAII helper that records elapsed time on destruction.
class StageTimer {
public:
    StageTimer(MetricsCollector& collector, std::string stage);
    ~StageTimer();

private:
    MetricsCollector& collector_;
    std::string stage_;
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace ultratrack
