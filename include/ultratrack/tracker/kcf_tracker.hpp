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

    /// Extract per-channel features from a patch resized to template_size.
    /// Falls back to a single grayscale channel if feature extraction fails.
    std::vector<cv::Mat> extractFeatureChannels(const cv::Mat& patch) const;

    /// Merge per-channel complex (CV_32FC2) filters into one multi-channel
    /// CV_32FC(2*C) Mat stored in Track::correlation_filter.
    static cv::Mat mergeComplexFilters(const std::vector<cv::Mat>& filters);

    /// Split a multi-channel CV_32FC(2*C) filter into per-channel CV_32FC2 Mats.
    static std::vector<cv::Mat> splitComplexFilters(const cv::Mat& filter);

    Config cfg_;
    std::unique_ptr<MultiFeatureExtractor> feature_extractor_;
    cv::Mat hann_window_;
    cv::Mat gaussian_target_;
    cv::Mat gaussian_target_fft_;
};

} // namespace ultratrack
