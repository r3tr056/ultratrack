#include <ultratrack/telemetry/metrics.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>

namespace ultratrack {

void MetricsCollector::record(const std::string& stage, double latency_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    data_[stage].push_back(latency_ms);
}

std::unordered_map<std::string, double> MetricsCollector::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_map<std::string, double> out;
    for (const auto& [stage, vals] : data_) {
        if (vals.empty()) continue;

        out[stage + "_count"] = static_cast<double>(vals.size());

        std::vector<double> sorted(vals);
        std::sort(sorted.begin(), sorted.end());
        out[stage + "_p50_ms"] = sorted[sorted.size() / 2];

        double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
        out[stage + "_avg_ms"] = sum / vals.size();
    }
    return out;
}

StageTimer::StageTimer(MetricsCollector& collector, std::string stage)
    : collector_(collector)
    , stage_(std::move(stage))
    , start_(std::chrono::high_resolution_clock::now()) {}

StageTimer::~StageTimer() {
    try {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        collector_.record(stage_, ms);
    } catch (const std::exception& e) {
        // Destructors must not throw. Log and swallow any exception from recording.
        spdlog::error("StageTimer::~StageTimer failed for stage '{}': {}", stage_, e.what());
    } catch (...) {
        spdlog::error("StageTimer::~StageTimer failed for stage '{}': unknown error", stage_);
    }
}

} // namespace ultratrack
