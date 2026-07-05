// include/ultratrack/tracker/displacement_predictor.hpp
#ifndef ULTRATRACK_DISPLACEMENT_PREDICTOR_HPP
#define ULTRATRACK_DISPLACEMENT_PREDICTOR_HPP

#include <opencv2/core.hpp>

namespace ultratrack {

struct DisplacementConfig {
    float kappa = 0.8f;           // Velocity coefficient
    float min_threshold = 2.0f;   // Minimum displacement (pixels)
    float max_threshold = 100.0f; // Maximum displacement (pixels)
    bool enabled = true;
};

class DisplacementPredictor {
public:
    explicit DisplacementPredictor(const DisplacementConfig& config = {});
    
    // Predict next position and update history
    cv::Point2f predict(const cv::Point2f& current_pos);
    
    // Reset history
    void reset();
    
    // Configuration
    void set_config(const DisplacementConfig& config);
    DisplacementConfig get_config() const { return config_; }
    
    // Get last computed velocity
    cv::Point2f get_velocity() const { return last_velocity_; }
    
    // Check if predictor has enough history
    bool is_ready() const { return history_count_ >= 2; }

private:
    DisplacementConfig config_;
    cv::Point2f pos_history_[2];  // [0] = t-2, [1] = t-1
    int history_count_ = 0;
    cv::Point2f last_velocity_;
};

} // namespace ultratrack

#endif // ULTRATRACK_DISPLACEMENT_PREDICTOR_HPP
