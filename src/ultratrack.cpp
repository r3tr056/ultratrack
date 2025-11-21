#include "ultratrack.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <mutex>

#define DEBUG_LOG(msg) std::cout << "[DEBUG] " << msg << std::endl


namespace ultratrack {

std::mutex tracks_mutex;

UltraTracker::UltraTracker(const std::string& model_path, const std::string& feature_model_path)
    : input_size_(640, 640), conf_threshold_(0.3f), nms_threshold_(0.5f), next_track_id_(1ULL), // Use unsigned long long to prevent overflow
      learning_rate_(0.01f), sigma_(2.0f), lambda_(0.01f), template_size_(128, 128) {
    try {
        detection_net_ = cv::dnn::readNetFromONNX(model_path);

        // Conditional CUDA setup with runtime check
#ifdef USE_CUDA
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            detection_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            detection_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::clog << "Using CUDA backend for detection." << std::endl;
        } else {
#endif
            detection_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            detection_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            std::clog << "Using CPU backend for detection (CUDA not available)." << std::endl;
#ifdef USE_CUDA
        }
#endif

        if (!feature_model_path.empty()) {
            try {
                feature_net_ = cv::dnn::readNetFromONNX(feature_model_path);
#ifdef USE_CUDA
                if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
                    feature_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                    feature_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                } else {
#endif
                    feature_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                    feature_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#ifdef USE_CUDA
                }
#endif
            } catch (const cv::Exception& e) {
                std::clog << "Warning: Could not load feature model: " << e.what() << std::endl;
                std::clog << "Continuing without deep features." << std::endl;
            }
        }
    } catch (const cv::Exception& e) {
        throw std::runtime_error("Failed to initialize detection network: " + std::string(e.what()));
    }
    init_kalman_matrices();

    cv::Mat hann_1d = create_hann_window(template_size_.width);
    cv::mulTransposed(hann_1d, hann_window_, false);
    hann_window_.convertTo(hann_window_, CV_32FC1);
    
    cv::Mat gaussian_1d_x = cv::getGaussianKernel(template_size_.width, sigma_, CV_32FC1);
    cv::Mat gaussian_1d_y = cv::getGaussianKernel(template_size_.height, sigma_, CV_32FC1);
    gaussian_target_ = gaussian_1d_y * gaussian_1d_x.t();

    CV_Assert(gaussian_target_.depth() == CV_32F);
#ifdef USE_CUDA
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        cufftPlan2d(&fft_plan_, template_size_.height, template_size_.width, CUFFT_R2C);
        cudaMalloc((void**)&d_template_, template_size_.area() * sizeof(float));
        cudaMalloc((void**)&d_search_, template_size_.area() * sizeof(float));
        cudaMalloc((void**)&d_response_, template_size_.area() * sizeof(float));
    } else {
        std::clog << "Warning: USE_CUDA defined but no devices found. Using CPU FFT." << std::endl;
    }
#endif
}

UltraTracker::~UltraTracker() {
#ifdef USE_CUDA
    cufftDestroy(fft_plan_);
    cudaFree(d_template_);
    cudaFree(d_search_);
    cudaFree(d_response_);
#endif
}

void UltraTracker::update(const cv::Mat& frame, std::vector<Detection>& detections) {
    validate_input(frame);
    auto start_time = std::chrono::high_resolution_clock::now();
    try {
        detections = detect_objects(frame);
        predict_tracks();
        associate_detections(detections, frame);
        remove_lost_tracks();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        if (duration.count() > 1) {
            if (conf_threshold_ < 0.7f) conf_threshold_ += 0.05f;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in tracker update: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error in tracker update: " << e.what() << std::endl;
        throw;
    }
}

std::vector<Detection> UltraTracker::detect_objects(const cv::Mat& frame) {
    std::vector<Detection> detections;
    try {
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, input_size_, cv::Scalar(), true, false);
        detection_net_.setInput(blob);
        std::vector<cv::Mat> outputs;
        detection_net_.forward(outputs, detection_net_.getUnconnectedOutLayersNames());
        if (outputs.empty()) {
            std::clog << "Warning: No outputs from detection network" << std::endl;
            return detections;
        }
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        const float* data = (float*)outputs[0].data;
        const int dimensions = outputs[0].size[1];
        const int rows = outputs[0].size[2];
        if (dimensions < 5) {
            std::cerr << "Error: Invalid network output dimensions: " << dimensions << std::endl;
            return detections;
        }
        float x_factor = frame.cols / static_cast<float>(input_size_.width);
        float y_factor = frame.rows / static_cast<float>(input_size_.height);
        for (int i = 0; i < rows; ++i) {
            const float* row = data + i * dimensions;
            float confidence = row[4];
            if (confidence >= conf_threshold_) {
                auto classes_scores = row + 5;
                cv::Mat scores(1, dimensions - 5, CV_32FC1, (void*)classes_scores);
                cv::Point class_id_point;
                double max_class_score;
                cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);
                if (max_class_score > conf_threshold_) {
                    float x = row[0];
                    float y = row[1];
                    float w = row[2];
                    float h = row[3];
                    // Strengthen bbox validation
                    if (w <= 0 || h <= 0) continue; // Skip invalid dimensions
                    cv::Rect bbox(static_cast<int>((x - w / 2) * x_factor),
                                  static_cast<int>((y - h / 2) * y_factor),
                                  static_cast<int>(w * x_factor),
                                  static_cast<int>(h * y_factor));
                    if (is_bbox_valid(cv::Rect2f(bbox), frame.size())) {
                        boxes.push_back(bbox);
                        confidences.push_back(confidence);
                        class_ids.push_back(class_id_point.x);
                    }
                }
            }
        }
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);
        for (int idx : indices) {
            Detection det;
            det.bbox = cv::Rect2f(boxes[idx]);
            det.confidence = confidences[idx];
            det.class_id = class_ids[idx];
            if (!feature_net_.empty()) {
                cv::Rect safe_bbox = boxes[idx] & cv::Rect(0, 0, frame.cols, frame.rows);
                if (safe_bbox.area() > 0) {
                    try {
                        cv::Mat patch = frame(safe_bbox);
                        det.feature = extract_features(patch);
                    } catch (const cv::Exception& e) {
                        std::clog << "Warning: Feature extraction failed: " << e.what() << std::endl;
                    }
                }
            }
            detections.push_back(det);
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in object detection: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Error in object detection: " << e.what() << std::endl;
        throw;
    }
    return detections;
}

cv::Mat UltraTracker::extract_features(const cv::Mat& patch) {
    if (feature_net_.empty() || patch.empty()) return cv::Mat();
    try {
        cv::Mat blob;
        cv::dnn::blobFromImage(patch, blob, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(), true, false);
        feature_net_.setInput(blob);
        cv::Mat features = feature_net_.forward();
        if (features.empty()) {
            std::clog << "Warning: Empty features from feature network" << std::endl;
            return cv::Mat();
        }
        cv::normalize(features, features, 1.0, 0.0, cv::NORM_L2);
        return features.reshape(1, 1).clone();
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in feature extraction: " << e.what() << std::endl;
        return cv::Mat();
    } catch (const std::exception& e) {
        std::cerr << "Error in feature extraction: " << e.what() << std::endl;
        return cv::Mat();
    }
}

void UltraTracker::predict_tracks() {
    std::lock_guard<std::mutex> lock(tracks_mutex); // Thread safety
    for (auto& track : active_tracks_) {
        predict_kalman(track);
        track.age++;
        track.time_since_update++;
    }
}

void UltraTracker::associate_detections(const std::vector<Detection>& detections, const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(tracks_mutex); // Thread safety
    std::vector<Detection> high_conf_dets, low_conf_dets;
    for (const auto& det : detections) {
        if (det.confidence >= 0.6f) {
            high_conf_dets.push_back(det);
        } else {
            low_conf_dets.push_back(det);
        }
    }
    cv::Mat cost_matrix = compute_cost_matrix(active_tracks_, high_conf_dets);
    auto matches = hungarian_assignment(cost_matrix);
    std::vector<bool> track_matched(active_tracks_.size(), false);
    std::vector<bool> det_matched(high_conf_dets.size(), false);
    std::vector<size_t> unmatched_track_ids;
    std::vector<Detection> unmatched_detections;
    for (const auto& match : matches) {
        size_t track_idx = match.first;
        size_t det_idx = match.second;
        if (cost_matrix.at<float>(track_idx, det_idx) < 0.8f) {
            update_kalman(active_tracks_[track_idx], high_conf_dets[det_idx].bbox);
            active_tracks_[track_idx].hits++;
            active_tracks_[track_idx].time_since_update = 0;
            active_tracks_[track_idx].confidence = high_conf_dets[det_idx].confidence;
            cv::Rect safe_bbox = cv::Rect(high_conf_dets[det_idx].bbox) & cv::Rect(0, 0, frame.cols, frame.rows);
            if (safe_bbox.area() > 0) {
                cv::Mat patch = frame(safe_bbox);
                update_correlation_filter(active_tracks_[track_idx], patch);
            }
            track_matched[track_idx] = true;
            det_matched[det_idx] = true;
        }
    }
    for (size_t i = 0; i < active_tracks_.size(); i++) {
        if (!track_matched[i]) unmatched_track_ids.push_back(i);
    }
    for (size_t i = 0; i < high_conf_dets.size(); i++) {
        if (!det_matched[i]) unmatched_detections.push_back(high_conf_dets[i]);
    }
    if (!low_conf_dets.empty() && !unmatched_track_ids.empty()) {
        std::vector<Track> unmatched_tracks;
        for (size_t idx : unmatched_track_ids) {
            unmatched_tracks.push_back(active_tracks_[idx]);
        }
        cv::Mat cost_matrix_2 = compute_cost_matrix(unmatched_tracks, low_conf_dets);
        auto matches_2 = hungarian_assignment(cost_matrix_2);
        for (const auto& match : matches_2) {
            size_t orig_track_idx = unmatched_track_ids[match.first];
            size_t det_idx = match.second;
            if (cost_matrix_2.at<float>(match.first, det_idx) < 0.7f) {
                update_kalman(active_tracks_[orig_track_idx], low_conf_dets[det_idx].bbox);
                active_tracks_[orig_track_idx].hits++;
                active_tracks_[orig_track_idx].time_since_update = 0;
                active_tracks_[orig_track_idx].confidence = low_conf_dets[det_idx].confidence;
                unmatched_track_ids.erase(std::remove(unmatched_track_ids.begin(), unmatched_track_ids.end(), orig_track_idx), unmatched_track_ids.end());
            }
        }
    }
    for (size_t i = 0; i < high_conf_dets.size(); i++) {
        if (!det_matched[i]) {
            unmatched_detections.push_back(high_conf_dets[i]);
        }
    }
    create_new_tracks(unmatched_detections, frame);
}

cv::Mat UltraTracker::compute_cost_matrix(const std::vector<Track>& tracks, const std::vector<Detection>& detections) {
    cv::Mat cost_matrix(static_cast<int>(tracks.size()), static_cast<int>(detections.size()), CV_32F, cv::Scalar(1.0f));
    for (size_t i = 0; i < tracks.size(); i++) {
        for (size_t j = 0; j < detections.size(); j++) {
            cv::Rect2f track_bbox = tracks[i].bbox;
            cv::Rect2f det_bbox = detections[j].bbox;
            float intersection = (track_bbox & det_bbox).area();
            float union_area = track_bbox.area() + det_bbox.area() - intersection;
            float iou = (union_area > 0) ? intersection / union_area : 0.0f;
            float cost = 1.0f - iou;
            if (!tracks[i].appearance_model.empty() && !detections[j].feature.empty()) {
                // Ensure size match for matchTemplate
                if (tracks[i].appearance_model.size() != detections[j].feature.size()) continue;
                cv::Mat similarity;
                cv::matchTemplate(tracks[i].appearance_model, detections[j].feature, similarity, cv::TM_CCOEFF_NORMED);
                float app_sim = similarity.at<float>(0, 0);
                cost = 0.7f * cost + 0.3f * (1.0f - app_sim);
            }
            cost_matrix.at<float>(static_cast<int>(i), static_cast<int>(j)) = cost;
        }
    }
    return cost_matrix;
}

std::vector<std::pair<int, int>> UltraTracker::hungarian_assignment(const cv::Mat& cost_matrix) {
    // Full Kuhn-Munkres (Hungarian) algorithm - basic implementation for accuracy
    std::vector<std::pair<int, int>> assignments;
    if (cost_matrix.rows == 0 || cost_matrix.cols == 0) return assignments;

    int n = std::max(cost_matrix.rows, cost_matrix.cols);
    cv::Mat cost = cv::Mat::zeros(n, n, CV_32F);
    cost_matrix.copyTo(cost(cv::Rect(0, 0, cost_matrix.cols, cost_matrix.rows)));

    // Step 1: Row reduction
    for (int row = 0; row < n; row++) {
        float min_val = *std::min_element(cost.ptr<float>(row), cost.ptr<float>(row) + n);
        for (int col = 0; col < n; col++) cost.at<float>(row, col) -= min_val;
    }

    // Step 2: Column reduction
    for (int col = 0; col < n; col++) {
        float min_val = std::numeric_limits<float>::max();
        for (int row = 0; row < n; row++) min_val = std::min(min_val, cost.at<float>(row, col));
        for (int row = 0; row < n; row++) cost.at<float>(row, col) -= min_val;
    }

    // Simplified assignment (for full, consider integrating a library like Munkres-cpp in production)
    // This is improved from placeholder but still not optimal; replace if needed
    std::vector<int> assignment(n, -1);
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            if (cost.at<float>(row, col) == 0 && assignment[row] == -1) {
                assignment[row] = col;
                break;
            }
        }
    }
    for (int row = 0; row < cost_matrix.rows; row++) {
        if (assignment[row] != -1 && assignment[row] < cost_matrix.cols) {
            assignments.emplace_back(row, assignment[row]);
        }
    }
    return assignments;
}

void UltraTracker::create_new_tracks(const std::vector<Detection>& unmatched_detections, const cv::Mat& frame) {
    for (const auto& detection : unmatched_detections) {
        if (detection.confidence < 0.7f) continue;
        Track new_track;
        new_track.id = next_track_id_++;
        new_track.bbox = detection.bbox;
        new_track.confidence = detection.confidence;
        new_track.age = 1;
        new_track.hits = 1;
        new_track.time_since_update = 0;
        new_track.is_activated = true;
        new_track.current_scale = 1.0f;
        new_track.scale_filters = {}; // Initialize empty
        new_track.state = (cv::Mat_<float>(8, 1) << detection.bbox.x + detection.bbox.width / 2,
                          detection.bbox.y + detection.bbox.height / 2, detection.bbox.width, detection.bbox.height, 0, 0, 0, 0);
        new_track.covariance = cv::Mat::eye(8, 8, CV_32F) * 10.0f;
        if (!detection.feature.empty()) {
            new_track.appearance_model = detection.feature.clone();
        }
        cv::Rect safe_bbox = cv::Rect(detection.bbox) & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safe_bbox.area() > 0 && !frame.empty()) {
            cv::Mat patch = frame(safe_bbox);
            new_track.correlation_filter = create_correlation_filter(patch);
        }
        {
            std::lock_guard<std::mutex> lock(tracks_mutex);
            active_tracks_.push_back(new_track);
        }
    }
}

void UltraTracker::remove_lost_tracks() {
    std::lock_guard<std::mutex> lock(tracks_mutex);
    auto it = std::remove_if(active_tracks_.begin(), active_tracks_.end(),
                             [](const Track& track) {
                                 return track.time_since_update > 30 || (track.time_since_update > 10 && track.hits < 3);
                             });
    active_tracks_.erase(it, active_tracks_.end());
}

void UltraTracker::init_kalman_matrices() {
    transition_matrix_ = (cv::Mat_<float>(8, 8) << 1, 0, 0, 0, 1, 0, 0, 0,
                          0, 1, 0, 0, 0, 1, 0, 0,
                          0, 0, 1, 0, 0, 0, 1, 0,
                          0, 0, 0, 1, 0, 0, 0, 1,
                          0, 0, 0, 0, 1, 0, 0, 0,
                          0, 0, 0, 0, 0, 1, 0, 0,
                          0, 0, 0, 0, 0, 0, 1, 0,
                          0, 0, 0, 0, 0, 0, 0, 1);
    measurement_matrix_ = (cv::Mat_<float>(4, 8) << 1, 0, 0, 0, 0, 0, 0, 0,
                           0, 1, 0, 0, 0, 0, 0, 0,
                           0, 0, 1, 0, 0, 0, 0, 0,
                           0, 0, 0, 1, 0, 0, 0, 0);
    process_noise_ = cv::Mat::eye(8, 8, CV_32F);
    cv::Mat q_diag = (cv::Mat_<float>(8, 1) << 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01);
    for (int i = 0; i < 8; i++) process_noise_.at<float>(i, i) = q_diag.at<float>(i, 0);
    measurement_noise_ = cv::Mat::eye(4, 4, CV_32F) * 1.0f;
}

void UltraTracker::predict_kalman(Track& track) {
    track.state = transition_matrix_ * track.state;
    cv::Mat temp = transition_matrix_ * track.covariance;
    track.covariance = temp * transition_matrix_.t() + process_noise_;
    track.bbox.x = track.state.at<float>(0) - track.state.at<float>(2) / 2;
    track.bbox.y = track.state.at<float>(1) - track.state.at<float>(3) / 2;
    track.bbox.width = track.state.at<float>(2);
    track.bbox.height = track.state.at<float>(3);
}

void UltraTracker::update_kalman(Track& track, const cv::Rect2f& detection) {
    cv::Mat measurement = (cv::Mat_<float>(4, 1) << detection.x + detection.width / 2,
                           detection.y + detection.height / 2, detection.width, detection.height);
    cv::Mat innovation = measurement - measurement_matrix_ * track.state;
    cv::Mat temp = measurement_matrix_ * track.covariance;
    cv::Mat innovation_cov = temp * measurement_matrix_.t() + measurement_noise_;
    cv::Mat kalman_gain = track.covariance * measurement_matrix_.t() * innovation_cov.inv();
    track.state = track.state + kalman_gain * innovation;
    cv::Mat identity = cv::Mat::eye(8, 8, CV_32F);
    track.covariance = (identity - kalman_gain * measurement_matrix_) * track.covariance;
    track.bbox.x = track.state.at<float>(0) - track.state.at<float>(2) / 2;
    track.bbox.y = track.state.at<float>(1) - track.state.at<float>(3) / 2;
    track.bbox.width = track.state.at<float>(2);
    track.bbox.height = track.state.at<float>(3);
}

std::vector<Track> UltraTracker::get_active_tracks() const {
    std::lock_guard<std::mutex> lock(tracks_mutex);
    std::vector<Track> result;
    std::copy_if(active_tracks_.begin(), active_tracks_.end(), std::back_inserter(result),
                 [](const Track& track) { return track.is_activated && track.time_since_update < 2; });
    return result;
}


cv::Mat UltraTracker::create_correlation_filter(const cv::Mat& patch) {
    if (patch.empty()) return cv::Mat();
    
    cv::Mat resized_patch;
    cv::resize(patch, resized_patch, template_size_);
    
    // Explicit conversion to float
    cv::Mat float_patch;
    resized_patch.convertTo(float_patch, CV_32FC1, 1.0 / 255.0);
    
    cv::Mat windowed_patch;
    if (float_patch.channels() == 3) {
        cv::cvtColor(float_patch, windowed_patch, cv::COLOR_BGR2GRAY);
        windowed_patch.convertTo(windowed_patch, CV_32FC1);
    } else {
        windowed_patch = float_patch.clone();
    }
    
    // Explicit conversion to float
    cv::Mat hann_32f;
    hann_window_.convertTo(hann_32f, CV_32FC1);
    // windowed_patch = windowed_patch.mul(hann_32f);
    simd_hann_window(windowed_patch.ptr<float>(), hann_32f.ptr<float>(), windowed_patch.ptr<float>(), windowed_patch.total());
    
    cv::Mat patch_fft = fft2d(windowed_patch);
    cv::Mat target_fft = fft2d(gaussian_target_);
    
    // Ensure FFT outputs are CV_32FC2
    if (patch_fft.type() != CV_32FC2) patch_fft.convertTo(patch_fft, CV_32FC2);
    if (target_fft.type() != CV_32FC2) target_fft.convertTo(target_fft, CV_32FC2);
    
    cv::Mat numerator, denominator, filter;
    // cv::mulSpectrums(target_fft, patch_fft, numerator, 0, true);
    numerator.create(target_fft.size(), target_fft.type());
    simd_mul_spectrums(target_fft.ptr<float>(), patch_fft.ptr<float>(), numerator.ptr<float>(), target_fft.total(), true);

    // cv::mulSpectrums(patch_fft, patch_fft, denominator, 0, true);
    denominator.create(patch_fft.size(), patch_fft.type());
    simd_mul_spectrums(patch_fft.ptr<float>(), patch_fft.ptr<float>(), denominator.ptr<float>(), patch_fft.total(), true);
    
    // Explicit complex-number handling
    cv::Mat lambda_mat;
    lambda_mat = cv::Scalar::all(lambda_);
    cv::add(denominator, lambda_mat, denominator);
    
    // cv::divide(numerator, denominator, filter);
    filter.create(numerator.size(), numerator.type());
    simd_div_spectrums(numerator.ptr<float>(), denominator.ptr<float>(), filter.ptr<float>(), numerator.total());
    return filter;
}


cv::Mat UltraTracker::track_correlation_filter(const Track& track, const cv::Mat& frame) {
    if (track.correlation_filter.empty() || frame.empty()) return cv::Mat();
    cv::Rect2f search_bbox = track.bbox;
    float scale_factor = 2.0f;
    search_bbox.x -= search_bbox.width * (scale_factor - 1.0f) / 2.0f;
    search_bbox.y -= search_bbox.height * (scale_factor - 1.0f) / 2.0f;
    search_bbox.width *= scale_factor;
    search_bbox.height *= scale_factor;
    cv::Rect safe_search = cv::Rect(search_bbox) & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safe_search.area() <= 0) return cv::Mat();
    cv::Mat search_patch = frame(safe_search);
    cv::Mat resized_search;
    cv::resize(search_patch, resized_search, template_size_);
    cv::Mat float_search;
    resized_search.convertTo(float_search, CV_32F, 1.0 / 255.0);
    cv::Mat gray_search;
    if (float_search.channels() == 3) {
        cv::cvtColor(float_search, gray_search, cv::COLOR_BGR2GRAY);
    } else {
        gray_search = float_search.clone();
    }
    // gray_search = gray_search.mul(hann_window_);
    simd_hann_window(gray_search.ptr<float>(), hann_window_.ptr<float>(), gray_search.ptr<float>(), gray_search.total());

    cv::Mat search_fft = fft2d(gray_search);
    cv::Mat response_fft;
    // cv::mulSpectrums(track.correlation_filter, search_fft, response_fft, 0, true);
    response_fft.create(track.correlation_filter.size(), track.correlation_filter.type());
    simd_mul_spectrums(track.correlation_filter.ptr<float>(), search_fft.ptr<float>(), response_fft.ptr<float>(), track.correlation_filter.total(), true);
    
    return ifft2d(response_fft);
}

void UltraTracker::update_correlation_filter(Track& track, const cv::Mat& patch) {
    if (patch.empty()) return;
    try {
        cv::Mat new_filter = create_correlation_filter(patch);
        if (new_filter.empty()) return;
        if (track.correlation_filter.empty()) {
            track.correlation_filter = new_filter.clone();
        } else {
            track.correlation_filter = (1.0f - learning_rate_) * track.correlation_filter + learning_rate_ * new_filter;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Error updating correlation filter: " << e.what() << std::endl;
    }
}

void UltraTracker::validate_input(const cv::Mat& frame) {
    if (frame.empty()) throw std::invalid_argument("Input frame is empty");
    if (frame.channels() != 3 && frame.channels() != 1) throw std::invalid_argument("Input frame must be 1 or 3 channel image");
    if (frame.type() != CV_8UC3 && frame.type() != CV_8UC1) throw std::invalid_argument("Input frame must be 8-bit unsigned integer type");
}

bool UltraTracker::is_bbox_valid(const cv::Rect2f& bbox, const cv::Size& frame_size) {
    return bbox.x >= 0 && bbox.y >= 0 && bbox.width > 0 && bbox.height > 0 &&
           bbox.x + bbox.width <= frame_size.width && bbox.y + bbox.height <= frame_size.height;
}

void UltraTracker::reset_tracker() {
    std::lock_guard<std::mutex> lock(tracks_mutex);
    active_tracks_.clear();
    lost_tracks_.clear();
    next_track_id_ = 1ULL;
}

void UltraTracker::load_gallery(const std::string& path) {
    std::vector<cv::String> filenames;
    cv::glob(path + "/*.jpg", filenames, false);
    if (filenames.empty()) {
        cv::glob(path + "/*.png", filenames, false);
    }
    
    std::cout << "[UltraTracker] Loading gallery from " << path << ". Found " << filenames.size() << " images." << std::endl;
    
    if (filenames.empty()) {
        std::cerr << "[UltraTracker] Warning: No images found in gallery path." << std::endl;
        return;
    }
    
    // Extract features from gallery images
    cv::HOGDescriptor hog(
        cv::Size(64, 128),  // winSize
        cv::Size(16, 16),   // blockSize
        cv::Size(8, 8),     // blockStride
        cv::Size(8, 8),     // cellSize
        9                   // nbins
    );
    
    gallery_features_.clear();
    
    for (const auto& filename : filenames) {
        cv::Mat img = cv::imread(filename);
        if (img.empty()) {
            std::cerr << "[UltraTracker] Warning: Could not read " << filename << std::endl;
            continue;
        }
        
        // Resize to standard size for HOG
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(64, 128));
        
        // Extract HOG features
        std::vector<float> descriptors;
        hog.compute(resized, descriptors);
        
        cv::Mat feature_mat(1, descriptors.size(), CV_32F);
        for (size_t i = 0; i < descriptors.size(); ++i) {
            feature_mat.at<float>(0, i) = descriptors[i];
        }
        
        // Normalize
        cv::normalize(feature_mat, feature_mat, 1.0, 0.0, cv::NORM_L2);
        
        gallery_features_.push_back(feature_mat);
        std::cout << "  - Loaded and processed " << filename << " (feature dim: " << descriptors.size() << ")" << std::endl;
    }
    
    std::cout << "[UltraTracker] Gallery loaded with " << gallery_features_.size() << " templates." << std::endl;
}

std::vector<Track> UltraTracker::get_all_tracks() const {
    std::lock_guard<std::mutex> lock(tracks_mutex);
    return active_tracks_;
}

size_t UltraTracker::get_track_count() const {
    std::lock_guard<std::mutex> lock(tracks_mutex);
    return active_tracks_.size();
}

void UltraTracker::set_nms_threshold(float threshold) {
    if (threshold >= 0.0f && threshold <= 1.0f) nms_threshold_ = threshold;
}

void UltraTracker::set_template_size(const cv::Size& size) {
    if (size.width > 0 && size.height > 0) {
        template_size_ = size;
        cv::Mat hann_1d = create_hann_window(template_size_.width);
        cv::mulTransposed(hann_1d, hann_window_, false);
        if (hann_window_.depth() != CV_32F)
            hann_window_.convertTo(hann_window_, CV_32F);
        cv::Mat gaussian_1d_x = cv::getGaussianKernel(template_size_.width, sigma_, CV_32F);
        cv::Mat gaussian_1d_y = cv::getGaussianKernel(template_size_.height, sigma_, CV_32F);
        gaussian_target_ = gaussian_1d_y * gaussian_1d_x.t();
    }
}

cv::Mat UltraTracker::create_hann_window(int size) {
    cv::Mat hann(1, size, CV_32F);
    float* data = hann.ptr<float>();
    for (int i = 0; i < size; i++) {
        data[i] = 0.5f * (1.0f - std::cos(2.0f * CV_PI * i / (size - 1)));
    }
    return hann;
}

} // namespace ultratrack
