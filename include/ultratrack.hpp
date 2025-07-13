// ultratrack.hpp
#ifndef ULTRATRACK_HPP
#define ULTRATRACK_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <memory>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#ifdef _WIN32
    #include <intrin.h>
    #include <immintrin.h>  // AVX2 support
#elif defined(__ARM_NEON)
    #include <arm_neon.h>   // NEON support
#else
    #include <immintrin.h>  // AVX2/SSE support
#endif

#ifdef USE_CUDA
#include <cufft.h>
#include <cuda_runtime.h>
#endif

namespace ultratrack {

struct Detection {
    cv::Rect2f bbox;
    float confidence;
    int class_id;
    cv::Mat feature;  // Deep appearance feature
};

struct Track {
    int id;
    cv::Rect2f bbox;
    cv::Mat state;           // Kalman state [x, y, w, h, vx, vy, vw, vh]
    cv::Mat covariance;      // State covariance
    cv::Mat correlation_filter; // Frequency domain filter
    cv::Mat appearance_model;   // Deep feature template
    float confidence;
    int age;
    int hits;
    int time_since_update;
    bool is_activated;
    
    // Multi-scale support
    std::vector<cv::Mat> scale_filters;
    float current_scale;
};

class UltraTracker {
private:
    // Core components
    cv::dnn::Net detection_net_;
    cv::Size input_size_;
    float conf_threshold_;
    float nms_threshold_;
    
    // Tracking components
    std::vector<Track> active_tracks_;
    std::vector<Track> lost_tracks_;
    int next_track_id_;
    
    // Kalman filter matrices
    cv::Mat transition_matrix_;
    cv::Mat measurement_matrix_;
    cv::Mat process_noise_;
    cv::Mat measurement_noise_;
    
    // Correlation filter parameters
    float learning_rate_;
    float sigma_;
    float lambda_;
    cv::Size template_size_;
    
    // GPU acceleration
    #ifdef USE_CUDA
    cufftHandle fft_plan_;
    float* d_template_;
    float* d_search_;
    float* d_response_;
    #endif
    
    // Feature extraction
    cv::dnn::Net feature_net_;
    
    // Performance optimization
    cv::Mat hann_window_;
    cv::Mat gaussian_target_;
    
public:
    UltraTracker(const std::string& model_path, 
                 const std::string& feature_model_path = "");
    ~UltraTracker();
    
    void update(const cv::Mat& frame, std::vector<Detection>& detections);
    std::vector<Track> get_active_tracks() const;
    
    // Configuration
    void set_confidence_threshold(float threshold) { conf_threshold_ = threshold; }
    void set_learning_rate(float rate) { learning_rate_ = rate; }
    void set_nms_threshold(float threshold);
    void set_template_size(const cv::Size& size);
    
    // Additional utility functions
    void reset_tracker();
    std::vector<Track> get_all_tracks() const;
    size_t get_track_count() const;
    
    // Performance getters
    float get_confidence_threshold() const { return conf_threshold_; }
    float get_learning_rate() const { return learning_rate_; }
    float get_nms_threshold() const { return nms_threshold_; }
    cv::Size get_template_size() const { return template_size_; }
    
private:
    // Detection pipeline
    std::vector<Detection> detect_objects(const cv::Mat& frame);
    cv::Mat extract_features(const cv::Mat& patch);
    
    // Tracking core
    void predict_tracks();
    void associate_detections(const std::vector<Detection>& detections, const cv::Mat& frame);
    void update_tracks(const std::vector<Detection>& matched_detections,
                      const std::vector<int>& matched_track_ids);
    void create_new_tracks(const std::vector<Detection>& unmatched_detections, const cv::Mat& frame);
    void remove_lost_tracks();
    
    // Correlation filter tracking
    cv::Mat create_correlation_filter(const cv::Mat& patch);
    cv::Mat track_correlation_filter(const Track& track, const cv::Mat& frame);
    void update_correlation_filter(Track& track, const cv::Mat& patch);
    
    // Association algorithms
    std::vector<std::pair<int, int>> hungarian_assignment(const cv::Mat& cost_matrix);
    cv::Mat compute_cost_matrix(const std::vector<Track>& tracks,
                               const std::vector<Detection>& detections);
    
    // Optimization utilities
    cv::Mat fft2d(const cv::Mat& input);
    cv::Mat ifft2d(const cv::Mat& input);
    void simd_correlation(const float* a, const float* b, float* result, int size);
    
    // Kalman filter operations
    void init_kalman_matrices();
    void predict_kalman(Track& track);
    void update_kalman(Track& track, const cv::Rect2f& detection);
    
    // Input validation and error handling
    void validate_input(const cv::Mat& frame);
    bool is_bbox_valid(const cv::Rect2f& bbox, const cv::Size& frame_size);
    
    // Helper functions
    cv::Mat create_hann_window(int size);
};

} // namespace ultratrack

#endif // ULTRATRACK_HPP
