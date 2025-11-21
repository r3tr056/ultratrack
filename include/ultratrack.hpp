// ultratrack.hpp - Header definitions

#ifndef ULTRATRACK_HPP
#define ULTRATRACK_HPP

#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#ifdef _WIN32
#include <immintrin.h>
#include <intrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

#ifdef USE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <cufft.h>
#endif

namespace ultratrack {

struct Detection {
    cv::Rect2f bbox;
    float confidence;
    int class_id;
    cv::Mat feature;
};

struct Track {
    unsigned long long id; // Prevent overflow
    cv::Rect2f bbox;
    cv::Mat state;
    cv::Mat covariance;
    cv::Mat correlation_filter;
    cv::Mat appearance_model;
    float confidence;
    int age;
    int hits;
    int time_since_update;
    bool is_activated;
    std::vector<cv::Mat> scale_filters;
    float current_scale;
};

class UltraTracker {
private:
    cv::dnn::Net detection_net_;
    cv::Size input_size_;
    float conf_threshold_;
    float nms_threshold_;
    std::vector<Track> active_tracks_;
    std::vector<Track> lost_tracks_;
    unsigned long long next_track_id_;
    cv::Mat transition_matrix_;
    cv::Mat measurement_matrix_;
    cv::Mat process_noise_;
    cv::Mat measurement_noise_;
    float learning_rate_;
    float sigma_;
    float lambda_;
    cv::Size template_size_;
#ifdef USE_CUDA
    cufftHandle fft_plan_;
    float* d_template_;
    float* d_search_;
    float* d_response_;
#endif
    cv::dnn::Net feature_net_;
    cv::Mat hann_window_;
    cv::Mat gaussian_target_;
    std::vector<cv::Mat> gallery_features_; // HOG features from training gallery

public:
    UltraTracker(const std::string& model_path, const std::string& feature_model_path = "");
    ~UltraTracker();
    void update(const cv::Mat& frame, std::vector<Detection>& detections);
    std::vector<Track> get_active_tracks() const;
    void set_confidence_threshold(float threshold) { conf_threshold_ = threshold; }
    void set_learning_rate(float rate) { learning_rate_ = rate; }
    void set_nms_threshold(float threshold);
    void set_template_size(const cv::Size& size);
    void set_nms_threshold(float threshold);
    void set_template_size(const cv::Size& size);
    void reset_tracker();
    void load_gallery(const std::string& path);
    std::vector<Track> get_all_tracks() const;
    size_t get_track_count() const;
    float get_confidence_threshold() const { return conf_threshold_; }
    float get_learning_rate() const { return learning_rate_; }
    float get_nms_threshold() const { return nms_threshold_; }
    cv::Size get_template_size() const { return template_size_; }

private:
    std::vector<Detection> detect_objects(const cv::Mat& frame);
    cv::Mat extract_features(const cv::Mat& patch);
    void predict_tracks();
    void associate_detections(const std::vector<Detection>& detections, const cv::Mat& frame);
    void create_new_tracks(const std::vector<Detection>& unmatched_detections, const cv::Mat& frame);
    void remove_lost_tracks();
    cv::Mat create_correlation_filter(const cv::Mat& patch);
    cv::Mat track_correlation_filter(const Track& track, const cv::Mat& frame);
    void update_correlation_filter(Track& track, const cv::Mat& patch);
    std::vector<std::pair<int, int>> hungarian_assignment(const cv::Mat& cost_matrix);
    cv::Mat compute_cost_matrix(const std::vector<Track>& tracks, const std::vector<Detection>& detections);
    cv::Mat fft2d(const cv::Mat& input);
    cv::Mat ifft2d(const cv::Mat& input);
    void simd_correlation(const float* a, const float* b, float* result, int size);
    void simd_mul_spectrums(const float* a, const float* b, float* result, int size, bool conj_b);
    void simd_div_spectrums(const float* num, const float* den, float* result, int size);
    void simd_hann_window(const float* src, const float* win, float* dst, int size);
    void init_kalman_matrices();
    void predict_kalman(Track& track);
    void update_kalman(Track& track, const cv::Rect2f& detection);
    void validate_input(const cv::Mat& frame);
    bool is_bbox_valid(const cv::Rect2f& bbox, const cv::Size& frame_size);
    cv::Mat create_hann_window(int size);
};

} // namespace ultratrack

#endif // ULTRATRACK_HPP
