#pragma once

#include <ultratrack/core/result.hpp>
#include <ultratrack/core/types.hpp>
#include <ultratrack/tracker/features/feature_extractor.hpp>
#include <opencv2/core.hpp>
#include <memory>

namespace ultratrack {

/// Single-track KCF correlation-filter tracker.
class KCFTracker {
public:
    struct Config {
        cv::Size template_size{128, 128};
        float learning_rate = 0.02f;
        float lambda = 0.01f;
        float sigma = 2.0f;
        TrackingMode feature_mode = TrackingMode::ACCURATE;
        FeatureConfig feature_config;
    };

    static Result<std::unique_ptr<KCFTracker>> create(const Config& cfg);

    Status init(Track& track, const cv::Mat& frame);
    Result<Rect2f> predict(Track& track, const cv::Mat& frame);
    Status update(Track& track, const cv::Mat& frame, const Rect2f& detected_bbox);

private:
    explicit KCFTracker(const Config& cfg);

    cv::Mat createFilter(const cv::Mat& patch);
    cv::Mat createHannWindow(int size);

    Config cfg_;
    std::unique_ptr<MultiFeatureExtractor> feature_extractor_;
    cv::Mat hann_window_;
    cv::Mat gaussian_target_;
    cv::Mat gaussian_target_fft_;
};

} // namespace ultratrack
