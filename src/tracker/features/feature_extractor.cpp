// src/tracker/features/feature_extractor.cpp
#include <ultratrack/tracker/features/feature_extractor.hpp>
#include <ultratrack/tracker/features/hog_feature.hpp>
#include <ultratrack/tracker/features/gray_feature.hpp>
#include <ultratrack/tracker/features/cn_feature.hpp>
#include "errors.hpp"

namespace ultratrack {

MultiFeatureExtractor::MultiFeatureExtractor(TrackingMode mode, const FeatureConfig& config)
    : mode_(mode), config_(config) {
    rebuild_extractors();
}

MultiFeatureExtractor::~MultiFeatureExtractor() = default;

void MultiFeatureExtractor::rebuild_extractors() {
    extractors_.clear();
    
    // HOG is always included
    extractors_.push_back(std::make_unique<HOGFeature>(config_));
    
    if (mode_ == TrackingMode::BALANCED || mode_ == TrackingMode::ACCURATE) {
        extractors_.push_back(std::make_unique<GrayFeature>());
    }
    
    if (mode_ == TrackingMode::ACCURATE) {
        extractors_.push_back(std::make_unique<CNFeature>());
    }
}

cv::Mat MultiFeatureExtractor::extract(const cv::Mat& patch) const {
    validate_patch(patch, "MultiFeatureExtractor::extract");

    std::vector<cv::Mat> features;
    features.reserve(extractors_.size());

    for (const auto& extractor : extractors_) {
        cv::Mat feat = extractor->extract(patch);
        if (!feat.empty()) {
            features.push_back(feat);
        }
    }

    if (features.empty()) {
        return cv::Mat();
    }

    if (features.size() == 1) {
        return features[0];
    }

    // BUG FIX: Feature extractors return different spatial sizes:
    //   HOG: cell-based grid (e.g., 16x16 for a 64x64 patch with cell_size=4)
    //   Gray: pixel-based (64x64)
    //   CN:   pixel-based (64x64)
    // We must resize all features to a common spatial size before concatenation.
    // Use the first feature's spatial size as the target (typically HOG, the smallest).
    int target_rows = features[0].rows;
    int target_cols = features[0].cols;

    // Resize all features to the target spatial size
    std::vector<cv::Mat> resized_features;
    resized_features.reserve(features.size());
    int total_channels = 0;

    for (const auto& feat : features) {
        total_channels += feat.channels();
        if (feat.rows == target_rows && feat.cols == target_cols) {
            resized_features.push_back(feat);
        } else {
            cv::Mat resized;
            cv::resize(feat, resized, cv::Size(target_cols, target_rows), 0, 0, cv::INTER_LINEAR);
            resized_features.push_back(resized);
        }
    }

    cv::Mat result(target_rows, target_cols, CV_32FC(total_channels));

    // Copy each feature's channels into the result
    int channel_offset = 0;
    for (const auto& feat : resized_features) {
        int feat_channels = feat.channels();

        for (int y = 0; y < target_rows; y++) {
            const float* src_ptr = feat.ptr<float>(y);
            float* dst_ptr = result.ptr<float>(y);

            for (int x = 0; x < target_cols; x++) {
                for (int c = 0; c < feat_channels; c++) {
                    dst_ptr[x * total_channels + channel_offset + c] =
                        src_ptr[x * feat_channels + c];
                }
            }
        }

        channel_offset += feat_channels;
    }

    return result;
}

int MultiFeatureExtractor::total_dimensions() const {
    int total = 0;
    for (const auto& extractor : extractors_) {
        total += extractor->dimensions();
    }
    return total;
}

void MultiFeatureExtractor::set_mode(TrackingMode mode) {
    if (mode_ != mode) {
        mode_ = mode;
        rebuild_extractors();
    }
}

} // namespace ultratrack
