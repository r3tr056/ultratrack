// src/features/feature_extractor.cpp
#include "feature_extractor.hpp"
#include "features/hog_feature.hpp"
#include "features/gray_feature.hpp"
#include "features/cn_feature.hpp"
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
    
    // Concatenate all features along channel dimension
    cv::Mat result;
    cv::merge(features, result);
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
