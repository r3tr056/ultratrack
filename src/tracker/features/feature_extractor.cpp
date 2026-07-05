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
    
    // Concatenate multi-channel feature maps
    // All features should have same size but may have different channel counts
    int total_channels = 0;
    for (const auto& feat : features) {
        total_channels += feat.channels();
    }
    
    int rows = features[0].rows;
    int cols = features[0].cols;
    cv::Mat result(rows, cols, CV_32FC(total_channels));
    
    // Copy each feature's channels into the result
    int channel_offset = 0;
    for (const auto& feat : features) {
        int feat_channels = feat.channels();
        
        for (int y = 0; y < rows; y++) {
            const float* src_ptr = feat.ptr<float>(y);
            float* dst_ptr = result.ptr<float>(y);
            
            for (int x = 0; x < cols; x++) {
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
