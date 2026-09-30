// src/tracker/displacement_predictor.cpp
#include <ultratrack/tracker/displacement_predictor.hpp>
#include <cmath>

namespace ultratrack {

DisplacementPredictor::DisplacementPredictor(const DisplacementConfig& config)
    : config_(config), history_count_(0), last_velocity_(0, 0) {
    pos_history_[0] = cv::Point2f(0, 0);
    pos_history_[1] = cv::Point2f(0, 0);
}

cv::Point2f DisplacementPredictor::predict(const cv::Point2f& current_pos) {
    cv::Point2f predicted = current_pos;
    
    if (config_.enabled && history_count_ >= 2) {
        // Compute displacement from last two frames
        cv::Point2f displacement = pos_history_[1] - pos_history_[0];
        float disp_magnitude = std::sqrt(displacement.x * displacement.x + 
                                          displacement.y * displacement.y);
        
        // Apply prediction only if displacement is within thresholds
        if (disp_magnitude > config_.min_threshold && 
            disp_magnitude < config_.max_threshold) {
            // P_t = P_{t-1} + kappa * (P_{t-1} - P_{t-2})
            predicted = current_pos + config_.kappa * displacement;
            last_velocity_ = displacement;
        } else {
            last_velocity_ = cv::Point2f(0, 0);
        }
    }
    
    // Update history
    pos_history_[0] = pos_history_[1];
    pos_history_[1] = current_pos;
    if (history_count_ < 2) {
        history_count_++;
    }
    
    return predicted;
}

void DisplacementPredictor::reset() {
    history_count_ = 0;
    pos_history_[0] = cv::Point2f(0, 0);
    pos_history_[1] = cv::Point2f(0, 0);
    last_velocity_ = cv::Point2f(0, 0);
}

void DisplacementPredictor::set_config(const DisplacementConfig& config) {
    config_ = config;
}

} // namespace ultratrack
