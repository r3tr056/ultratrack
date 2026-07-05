#pragma once
#include <ultratrack/core/result.hpp>
#include <ultratrack/core/types.hpp>
#include <ultratrack/tracking_session.hpp>
#include <string>
#include <vector>

namespace ultratrack {

struct GroundTruthBox {
    int frame_id;
    int track_id;
    Rect2f bbox;
};

struct MOTSequence {
    std::vector<Frame> frames;
    std::vector<GroundTruthBox> ground_truth;
};

struct BenchmarkMetrics {
    double mota = 0.0;
    double motp = 0.0;
    double idf1 = 0.0;
    double hota = 0.0;
    double avg_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
};

struct BenchmarkResult {
    std::string variant_name;
    BenchmarkMetrics metrics;
};

class BenchmarkHarness {
public:
    Result<MOTSequence> loadSequence(const std::string& image_dir);
    Result<MOTSequence> loadSequenceWithGT(const std::string& image_dir,
                                            const std::string& gt_file);
    BenchmarkResult runVariant(const std::string& name,
                                const TrackerSettings& settings,
                                const MOTSequence& seq);
    void report(const std::vector<BenchmarkResult>& results);
};

} // namespace ultratrack
