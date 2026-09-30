// include/ultratrack/tracker/scale_estimator.hpp
#ifndef ULTRATRACK_SCALE_ESTIMATOR_HPP
#define ULTRATRACK_SCALE_ESTIMATOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ultratrack {

struct ScaleConfig {
    float scale_step = 0.04f;     // Scale increment delta
    int num_scales = 3;           // Number of scales (3, 5, or 7)
    float min_scale = 0.2f;       // Minimum allowed scale
    float max_scale = 5.0f;       // Maximum allowed scale
    float scale_penalty = 0.975f; // Penalty for scale change
};

class ScaleEstimator {
public:
    explicit ScaleEstimator(const ScaleConfig& config = {});
    
    // Estimate optimal scale
    float estimate(const cv::Mat& frame,
                   const cv::Point2f& position,
                   const cv::Size2f& base_size,
                   const cv::Mat& correlation_filter,
                   const cv::Mat& model_patch);
    
    // Get current scale pool
    std::vector<float> get_scale_pool() const { return scale_pool_; }
    
    // Get accumulated scale
    float get_current_scale() const { return current_scale_; }
    
    // Reset scale to 1.0
    void reset();
    
    // Configuration
    void set_config(const ScaleConfig& config);
    ScaleConfig get_config() const { return config_; }

private:
    ScaleConfig config_;
    float current_scale_ = 1.0f;
    std::vector<float> scale_pool_;
    
    void rebuild_scale_pool();
    
    cv::Mat extract_patch_at_scale(const cv::Mat& frame,
                                    const cv::Point2f& center,
                                    const cv::Size2f& base_size,
                                    float scale);
    
    float compute_response(const cv::Mat& patch,
                           const cv::Mat& correlation_filter,
                           const cv::Mat& model_patch);
};

} // namespace ultratrack

#endif // ULTRATRACK_SCALE_ESTIMATOR_HPP
