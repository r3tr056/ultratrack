// src/tracker/scale_estimator.cpp
#include <ultratrack/tracker/scale_estimator.hpp>
#include "errors.hpp"
#include <algorithm>
#include <cmath>

namespace ultratrack {

ScaleEstimator::ScaleEstimator(const ScaleConfig& config) : config_(config) {
    rebuild_scale_pool();
}

void ScaleEstimator::rebuild_scale_pool() {
    scale_pool_.clear();
    
    // Build scale pool: {1-2δ, 1-δ, ..., 1, ..., 1+δ, 1+2δ}
    // For 3 scales: {1-2δ, 1, 1+2δ}
    int half = config_.num_scales / 2;
    
    for (int i = -half; i <= half; i++) {
        if (config_.num_scales == 3 && i != 0) {
            // For 3-scale, use 2*delta steps
            scale_pool_.push_back(1.0f + 2.0f * i * config_.scale_step);
        } else {
            scale_pool_.push_back(1.0f + i * config_.scale_step);
        }
    }
    
    // For 3-scale pool, ensure we have exactly {1-2δ, 1, 1+2δ}
    if (config_.num_scales == 3) {
        scale_pool_.clear();
        scale_pool_.push_back(1.0f - 2.0f * config_.scale_step);
        scale_pool_.push_back(1.0f);
        scale_pool_.push_back(1.0f + 2.0f * config_.scale_step);
    }
}

cv::Mat ScaleEstimator::extract_patch_at_scale(const cv::Mat& frame,
                                                const cv::Point2f& center,
                                                const cv::Size2f& base_size,
                                                float scale) {
    float scaled_w = base_size.width * scale;
    float scaled_h = base_size.height * scale;
    
    cv::Rect2f roi(center.x - scaled_w / 2,
                   center.y - scaled_h / 2,
                   scaled_w, scaled_h);
    
    // Clamp to frame bounds
    cv::Rect safe_roi = cv::Rect(roi) & cv::Rect(0, 0, frame.cols, frame.rows);
    
    if (safe_roi.area() <= 0) {
        return cv::Mat();
    }
    
    return frame(safe_roi).clone();
}

float ScaleEstimator::compute_response(const cv::Mat& patch,
                                        const cv::Mat& correlation_filter,
                                        const cv::Mat& model_patch) {
    if (patch.empty() || correlation_filter.empty()) {
        return -std::numeric_limits<float>::max();
    }
    
    // Resize patch to match model size
    cv::Mat resized;
    cv::resize(patch, resized, model_patch.size());
    
    // Convert to grayscale float
    cv::Mat gray;
    if (resized.channels() == 3) {
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = resized;
    }
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);
    
    // Compute correlation response
    cv::Mat patch_fft, response_fft, response;
    cv::dft(gray, patch_fft, cv::DFT_COMPLEX_OUTPUT);
    cv::mulSpectrums(correlation_filter, patch_fft, response_fft, 0, true);
    cv::idft(response_fft, response, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    
    // Return peak response
    double max_val;
    cv::minMaxLoc(response, nullptr, &max_val);
    
    return static_cast<float>(max_val);
}

float ScaleEstimator::estimate(const cv::Mat& frame,
                                const cv::Point2f& position,
                                const cv::Size2f& base_size,
                                const cv::Mat& correlation_filter,
                                const cv::Mat& model_patch) {
    if (frame.empty()) {
        return current_scale_;
    }
    
    float best_scale = current_scale_;
    float best_response = -std::numeric_limits<float>::max();
    
    for (float scale_factor : scale_pool_) {
        float test_scale = current_scale_ * scale_factor;
        
        // Clamp to valid range
        test_scale = std::clamp(test_scale, config_.min_scale, config_.max_scale);
        
        // Extract patch at test scale
        cv::Mat patch = extract_patch_at_scale(frame, position, base_size, test_scale);
        if (patch.empty()) continue;
        
        // Compute response
        float response = compute_response(patch, correlation_filter, model_patch);
        
        // Apply scale penalty for non-unity scales
        if (std::abs(scale_factor - 1.0f) > 0.001f) {
            response *= config_.scale_penalty;
        }
        
        if (response > best_response) {
            best_response = response;
            best_scale = test_scale;
        }
    }
    
    current_scale_ = best_scale;
    return current_scale_;
}

void ScaleEstimator::reset() {
    current_scale_ = 1.0f;
}

void ScaleEstimator::set_config(const ScaleConfig& config) {
    config_ = config;
    rebuild_scale_pool();
}

} // namespace ultratrack
