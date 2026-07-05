#include <ultratrack/benchmark/harness.hpp>
#include <ultratrack/ultratrack_sdk.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <chrono>

namespace ultratrack {

Result<MOTSequence> BenchmarkHarness::loadSequence(const std::string& image_dir) {
    try {
        std::vector<cv::String> files;
        cv::glob(image_dir + "/*.jpg", files);
        if (files.empty()) cv::glob(image_dir + "/*.png", files);
        if (files.empty()) {
            return Status(ErrorCode::INVALID_ARGUMENT, "no images found");
        }

        // cv::glob order is filesystem-dependent; enforce lexicographic frame order.
        std::sort(files.begin(), files.end());

        MOTSequence seq;
        for (const auto& f : files) {
            cv::Mat img = cv::imread(f, cv::IMREAD_UNCHANGED);
            if (img.empty()) continue;

            // Ensure consistent BGR format for the benchmark harness.
            if (img.channels() == 1) {
                cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
            } else if (img.channels() == 4) {
                cv::cvtColor(img, img, cv::COLOR_BGRA2BGR);
            }

            Frame frame;
            frame.width = img.cols;
            frame.height = img.rows;
            frame.format = FrameFormat::BGR;
            frame.data = std::move(img);
            seq.frames.push_back(frame);
        }

        if (seq.frames.empty()) {
            return Status(ErrorCode::INVALID_ARGUMENT, "no images could be loaded");
        }
        return seq;
    } catch (const std::exception& e) {
        return Status(ErrorCode::INTERNAL_ERROR,
                      std::string("loadSequence failed: ") + e.what());
    }
}

Result<MOTSequence> BenchmarkHarness::loadSequenceWithGT(const std::string& image_dir,
                                                          const std::string& gt_file) {
    auto seq = loadSequence(image_dir);
    if (!seq.has_value()) return seq.error();

    try {
        std::ifstream in(gt_file);
        if (!in) {
            return Status(ErrorCode::INVALID_ARGUMENT, "cannot open gt file");
        }

        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::stringstream ss(line);
            GroundTruthBox gt;
            char sep = '\0';
            float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;
            ss >> gt.frame_id >> sep >> gt.track_id >> sep >> x >> sep >> y >> sep >> w >> sep >> h;

            if (ss.fail() || sep != ',') {
                return Status(ErrorCode::INVALID_ARGUMENT,
                              "invalid gt format: " + line);
            }

            gt.bbox = Rect2f(x, y, w, h);
            seq.value().ground_truth.push_back(gt);
        }
        return seq.value();
    } catch (const std::exception& e) {
        return Status(ErrorCode::INTERNAL_ERROR,
                      std::string("loadSequenceWithGT failed: ") + e.what());
    }
}

BenchmarkResult BenchmarkHarness::runVariant(const std::string& name,
                                              const TrackerSettings& settings,
                                              const MOTSequence& seq) {
    BenchmarkResult result{name, {}};
    bool initialized_here = false;

    try {
        if (!UltraTrackerSDK::is_licensed()) {
            SDKConfig cfg;
            cfg.license_key = "TRIAL";
            auto status = UltraTrackerSDK::initialize(cfg);
            if (!status.ok()) {
                result.metrics.avg_latency_ms = -1.0;
                return result;
            }
            initialized_here = true;
        }

        auto session = TrackingSession::create(settings);
        if (!session.has_value()) {
            result.metrics.avg_latency_ms = -1.0;
            if (initialized_here) UltraTrackerSDK::shutdown();
            return result;
        }

        std::vector<double> latencies;
        latencies.reserve(seq.frames.size());
        for (const auto& frame : seq.frames) {
            auto start = std::chrono::high_resolution_clock::now();
            auto out = session.value()->processFrame(frame);
            auto end = std::chrono::high_resolution_clock::now();
            latencies.push_back(
                std::chrono::duration<double, std::milli>(end - start).count());
            (void)out;
        }

        if (!latencies.empty()) {
            result.metrics.avg_latency_ms =
                std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                static_cast<double>(latencies.size());
            std::sort(latencies.begin(), latencies.end());
            // Use a floor index for the 95th percentile to stay simple and
            // deterministic for small sequences; for large N this converges
            // to the standard p95.
            const size_t p95_index = static_cast<size_t>(latencies.size() * 0.95);
            result.metrics.p95_latency_ms = latencies[std::min(p95_index, latencies.size() - 1)];
        }

        // MOTA/HOTA/IDF1 placeholders; integrate trackeval or motmetrics in follow-up.
        result.metrics.mota = 0.0;
        result.metrics.motp = 0.0;
        result.metrics.hota = 0.0;
        result.metrics.idf1 = 0.0;

        if (initialized_here) UltraTrackerSDK::shutdown();
        return result;
    } catch (const std::exception& e) {
        spdlog::warn("BenchmarkHarness::runVariant '{}' failed: {}", name, e.what());
        result.metrics.avg_latency_ms = -1.0;
        if (initialized_here) UltraTrackerSDK::shutdown();
        return result;
    }
}

void BenchmarkHarness::report(const std::vector<BenchmarkResult>& results) {
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& r : results) {
        std::cout << r.variant_name << ": avg=" << r.metrics.avg_latency_ms
                  << "ms p95=" << r.metrics.p95_latency_ms
                  << "ms MOTA=" << r.metrics.mota
                  << " MOTP=" << r.metrics.motp
                  << " HOTA=" << r.metrics.hota
                  << " IDF1=" << r.metrics.idf1 << "\n";
    }
}

} // namespace ultratrack
