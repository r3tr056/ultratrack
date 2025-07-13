// ultratrack.cpp
#include "ultratrack.hpp"
#include <algorithm>
#include <numeric>

namespace ultratrack {

UltraTracker::UltraTracker(const std::string& model_path, 
                           const std::string& feature_model_path)
    : input_size_(640, 640)
    , conf_threshold_(0.3f)
    , nms_threshold_(0.5f)
    , next_track_id_(1)
    , learning_rate_(0.01f)
    , sigma_(2.0f)
    , lambda_(0.01f)
    , template_size_(128, 128) {
    
    try {
        // Initialize YOLOv11 Nano detection network
        detection_net_ = cv::dnn::readNetFromONNX(model_path);
        
        // Try to set CUDA backend, fallback to CPU if not available
        try {
            detection_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            detection_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } catch (const cv::Exception&) {
            std::cout << "CUDA not available, using CPU backend for detection." << std::endl;
            detection_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            detection_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
        
        // Initialize feature extraction network (ResNet50 backbone)
        if (!feature_model_path.empty()) {
            try {
                feature_net_ = cv::dnn::readNetFromONNX(feature_model_path);
                
                try {
                    feature_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                    feature_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                } catch (const cv::Exception&) {
                    std::cout << "CUDA not available, using CPU backend for features." << std::endl;
                    feature_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                    feature_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                }
            } catch (const cv::Exception& e) {
                std::cout << "Warning: Could not load feature model: " << e.what() << std::endl;
                std::cout << "Continuing without deep features." << std::endl;
            }
        }
        
    } catch (const cv::Exception& e) {
        throw std::runtime_error("Failed to initialize detection network: " + std::string(e.what()));
    }
    
    // Initialize Kalman filter matrices
    init_kalman_matrices();
    
    // Create Hann window for correlation filter
    cv::Mat hann_1d = create_hann_window(template_size_.width);
    cv::Mat hann_2d;
    cv::mulTransposed(hann_1d, hann_2d, false);
    hann_window_ = hann_2d;
    
    // Create Gaussian target
    cv::Mat gaussian_1d_x = cv::getGaussianKernel(template_size_.width, sigma_, CV_32F);
    cv::Mat gaussian_1d_y = cv::getGaussianKernel(template_size_.height, sigma_, CV_32F);
    gaussian_target_ = gaussian_1d_y * gaussian_1d_x.t();
    
    #ifdef USE_CUDA
    // Initialize CUDA FFT
    try {
        cufftPlan2d(&fft_plan_, template_size_.height, template_size_.width, CUFFT_R2C);
        cudaMalloc(&d_template_, template_size_.area() * sizeof(float));
        cudaMalloc(&d_search_, template_size_.area() * sizeof(float));
        cudaMalloc(&d_response_, template_size_.area() * sizeof(float));
    } catch (const std::exception& e) {
        std::cout << "Warning: CUDA FFT initialization failed: " << e.what() << std::endl;
        std::cout << "Using CPU FFT fallback." << std::endl;
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
    // Input validation
    validate_input(frame);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Step 1: Object detection
        detections = detect_objects(frame);
        
        // Step 2: Predict existing tracks
        predict_tracks();
        
        // Step 3: Association using ByteTrack strategy
        associate_detections(detections, frame);
        
        // Step 4: Remove lost tracks
        remove_lost_tracks();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Performance optimization: aim for <1ms processing time
        if (duration.count() > 1000) {
            // Adaptive parameter adjustment for performance
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
        cv::dnn::blobFromImage(frame, blob, 1.0/255.0, input_size_, cv::Scalar(), true, false);
        
        detection_net_.setInput(blob);
        std::vector<cv::Mat> outputs;
        detection_net_.forward(outputs, detection_net_.getUnconnectedOutLayersNames());
        
        if (outputs.empty()) {
            std::cerr << "Warning: No outputs from detection network" << std::endl;
            return detections;
        }
        
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        
        // Parse YOLOv11 output
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
                    
                    cv::Rect bbox(
                        static_cast<int>((x - w/2) * x_factor),
                        static_cast<int>((y - h/2) * y_factor),
                        static_cast<int>(w * x_factor),
                        static_cast<int>(h * y_factor)
                    );
                    
                    // Validate bbox
                    if (is_bbox_valid(cv::Rect2f(bbox), frame.size())) {
                        boxes.push_back(bbox);
                        confidences.push_back(confidence);
                        class_ids.push_back(class_id_point.x);
                    }
                }
            }
        }
        
        // Apply NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);
        
        for (int idx : indices) {
            Detection det;
            det.bbox = cv::Rect2f(boxes[idx]);
            det.confidence = confidences[idx];
            det.class_id = class_ids[idx];
            
            // Extract deep appearance features
            if (!feature_net_.empty()) {
                cv::Rect safe_bbox = cv::Rect(det.bbox) & cv::Rect(0, 0, frame.cols, frame.rows);
                if (safe_bbox.area() > 0) {
                    try {
                        cv::Mat patch = frame(safe_bbox);
                        det.feature = extract_features(patch);
                    } catch (const cv::Exception& e) {
                        std::cerr << "Warning: Feature extraction failed: " << e.what() << std::endl;
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
        cv::dnn::blobFromImage(patch, blob, 1.0/255.0, cv::Size(224, 224), cv::Scalar(), true, false);
        
        feature_net_.setInput(blob);
        cv::Mat features;
        feature_net_.forward(features);
        
        if (features.empty()) {
            std::cerr << "Warning: Empty features from feature network" << std::endl;
            return cv::Mat();
        }
        
        // L2 normalize features
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
    for (auto& track : active_tracks_) {
        predict_kalman(track);
        track.age++;
        track.time_since_update++;
    }
}

void UltraTracker::associate_detections(const std::vector<Detection>& detections, const cv::Mat& frame) {
    // Separate high and low confidence detections (ByteTrack strategy)
    std::vector<Detection> high_conf_dets, low_conf_dets;
    for (const auto& det : detections) {
        if (det.confidence >= 0.6f) {
            high_conf_dets.push_back(det);
        } else {
            low_conf_dets.push_back(det);
        }
    }
    
    // First association: high confidence detections with active tracks
    cv::Mat cost_matrix = compute_cost_matrix(active_tracks_, high_conf_dets);
    auto matches = hungarian_assignment(cost_matrix);
    
    std::vector<int> matched_track_ids, unmatched_track_ids;
    std::vector<Detection> matched_detections, unmatched_detections;
    
    // Process matches from first stage
    std::vector<bool> track_matched(static_cast<size_t>(active_tracks_.size()), false);
    std::vector<bool> det_matched(static_cast<size_t>(high_conf_dets.size()), false);
    
    for (const auto& match : matches) {
        int track_idx = match.first;
        int det_idx = match.second;
        
        if (cost_matrix.at<float>(track_idx, det_idx) < 0.8f) {  // IoU threshold
            update_kalman(active_tracks_[track_idx], high_conf_dets[det_idx].bbox);
            active_tracks_[track_idx].hits++;
            active_tracks_[track_idx].time_since_update = 0;
            active_tracks_[track_idx].confidence = high_conf_dets[det_idx].confidence;
            
            // Update correlation filter with matched detection
            cv::Rect safe_bbox = cv::Rect(high_conf_dets[det_idx].bbox) & cv::Rect(0, 0, frame.cols, frame.rows);
            if (safe_bbox.area() > 0) {
                cv::Mat patch = frame(safe_bbox);
                update_correlation_filter(active_tracks_[track_idx], patch);
            }
            
            track_matched[static_cast<size_t>(track_idx)] = true;
            det_matched[static_cast<size_t>(det_idx)] = true;
        }
    }
    
    // Collect unmatched tracks and detections
    for (size_t i = 0; i < active_tracks_.size(); i++) {
        if (!track_matched[i]) {
            unmatched_track_ids.push_back(static_cast<int>(i));
        }
    }
    
    for (size_t i = 0; i < high_conf_dets.size(); i++) {
        if (!det_matched[i]) {
            unmatched_detections.push_back(high_conf_dets[i]);
        }
    }
    
    // Second association: low confidence detections with remaining tracks
    if (!low_conf_dets.empty() && !unmatched_track_ids.empty()) {
        std::vector<Track> unmatched_tracks;
        for (size_t idx : unmatched_track_ids) {
            unmatched_tracks.push_back(active_tracks_[idx]);
        }
        
        cv::Mat cost_matrix_2 = compute_cost_matrix(unmatched_tracks, low_conf_dets);
        auto matches_2 = hungarian_assignment(cost_matrix_2);
        
        for (const auto& match : matches_2) {
            size_t track_idx = unmatched_track_ids[match.first];
            int det_idx = match.second;
            
            if (cost_matrix_2.at<float>(static_cast<int>(match.first), det_idx) < 0.7f) {
                update_kalman(active_tracks_[track_idx], low_conf_dets[det_idx].bbox);
                active_tracks_[track_idx].hits++;
                active_tracks_[track_idx].time_since_update = 0;
                active_tracks_[track_idx].confidence = low_conf_dets[det_idx].confidence;
                
                // Remove from unmatched lists
                unmatched_track_ids.erase(
                    std::remove(unmatched_track_ids.begin(), unmatched_track_ids.end(), track_idx),
                    unmatched_track_ids.end()
                );
            }
        }
    }
    
    // Add remaining high confidence detections to unmatched
    for (size_t i = 0; i < high_conf_dets.size(); i++) {
        if (!det_matched[i]) {
            unmatched_detections.push_back(high_conf_dets[i]);
        }
    }
    
    // Create new tracks for unmatched detections
    create_new_tracks(unmatched_detections, frame);
}

cv::Mat UltraTracker::compute_cost_matrix(const std::vector<Track>& tracks,
                                         const std::vector<Detection>& detections) {
    cv::Mat cost_matrix(static_cast<int>(tracks.size()), static_cast<int>(detections.size()), CV_32F);
    
    for (size_t i = 0; i < tracks.size(); i++) {
        for (size_t j = 0; j < detections.size(); j++) {
            // Compute IoU distance
            cv::Rect2f track_bbox = tracks[i].bbox;
            cv::Rect2f det_bbox = detections[j].bbox;
            
            float intersection = (track_bbox & det_bbox).area();
            float union_area = track_bbox.area() + det_bbox.area() - intersection;
            float iou = intersection / union_area;
            
            float cost = 1.0f - iou;  // Convert IoU to distance
            
            // Add appearance similarity if features available
            if (!tracks[i].appearance_model.empty() && !detections[j].feature.empty()) {
                cv::Mat similarity;
                cv::matchTemplate(tracks[i].appearance_model, detections[j].feature, 
                                similarity, cv::TM_CCOEFF_NORMED);
                float app_sim = similarity.at<float>(0, 0);
                cost = 0.7f * cost + 0.3f * (1.0f - app_sim);  // Weighted combination
            }
            
            cost_matrix.at<float>(static_cast<int>(i), static_cast<int>(j)) = cost;
        }
    }
    
    return cost_matrix;
}

void UltraTracker::create_new_tracks(const std::vector<Detection>& unmatched_detections, const cv::Mat& frame) {
    for (const auto& detection : unmatched_detections) {
        if (detection.confidence < 0.7f) continue;  // Only create tracks for high confidence
        
        Track new_track;
        new_track.id = next_track_id_++;
        new_track.bbox = detection.bbox;
        new_track.confidence = detection.confidence;
        new_track.age = 1;
        new_track.hits = 1;
        new_track.time_since_update = 0;
        new_track.is_activated = true;
        new_track.current_scale = 1.0f;
        
        // Initialize Kalman state [x, y, w, h, vx, vy, vw, vh]
        new_track.state = (cv::Mat_<float>(8, 1) << 
            detection.bbox.x + detection.bbox.width/2,  // center x
            detection.bbox.y + detection.bbox.height/2, // center y
            detection.bbox.width,                       // width
            detection.bbox.height,                      // height
            0, 0, 0, 0);                               // velocities
        
        new_track.covariance = cv::Mat::eye(8, 8, CV_32F) * 10.0f;
        
        // Store appearance model
        if (!detection.feature.empty()) {
            new_track.appearance_model = detection.feature.clone();
        }
        
        // Initialize correlation filter
        cv::Rect safe_bbox = cv::Rect(detection.bbox) & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safe_bbox.area() > 0 && !frame.empty()) {
            cv::Mat patch = frame(safe_bbox);
            new_track.correlation_filter = create_correlation_filter(patch);
        }
        
        active_tracks_.push_back(new_track);
    }
}

void UltraTracker::remove_lost_tracks() {
    auto it = std::remove_if(active_tracks_.begin(), active_tracks_.end(),
        [](const Track& track) {
            return track.time_since_update > 30 || 
                   (track.time_since_update > 10 && track.hits < 3);
        });
    
    active_tracks_.erase(it, active_tracks_.end());
}

// Kalman Filter Implementation
void UltraTracker::init_kalman_matrices() {
    // State transition matrix (constant velocity model)
    transition_matrix_ = (cv::Mat_<float>(8, 8) <<
        1, 0, 0, 0, 1, 0, 0, 0,
        0, 1, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1);
    
    // Measurement matrix (observe position and size)
    measurement_matrix_ = (cv::Mat_<float>(4, 8) <<
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0);
    
    // Process noise
    process_noise_ = cv::Mat::eye(8, 8, CV_32F);
    cv::Mat q_diag = (cv::Mat_<float>(8, 1) << 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01);
    for (int i = 0; i < 8; i++) {
        process_noise_.at<float>(i, i) = q_diag.at<float>(i, 0);
    }
    
    // Measurement noise
    measurement_noise_ = cv::Mat::eye(4, 4, CV_32F) * 1.0f;
}

void UltraTracker::predict_kalman(Track& track) {
    // Predict state: x = F * x
    track.state = transition_matrix_ * track.state;
    
    // Predict covariance: P = F * P * F^T + Q
    cv::Mat temp = transition_matrix_ * track.covariance;
    track.covariance = temp * transition_matrix_.t() + process_noise_;
    
    // Update bbox from predicted state
    track.bbox.x = track.state.at<float>(0, 0) - track.state.at<float>(2, 0) / 2;
    track.bbox.y = track.state.at<float>(1, 0) - track.state.at<float>(3, 0) / 2;
    track.bbox.width = track.state.at<float>(2, 0);
    track.bbox.height = track.state.at<float>(3, 0);
}

void UltraTracker::update_kalman(Track& track, const cv::Rect2f& detection) {
    // Measurement vector
    cv::Mat measurement = (cv::Mat_<float>(4, 1) <<
        detection.x + detection.width/2,   // center x
        detection.y + detection.height/2,  // center y
        detection.width,                   // width
        detection.height);                 // height
    
    // Innovation: y = z - H * x
    cv::Mat innovation = measurement - measurement_matrix_ * track.state;
    
    // Innovation covariance: S = H * P * H^T + R
    cv::Mat temp = measurement_matrix_ * track.covariance;
    cv::Mat innovation_cov = temp * measurement_matrix_.t() + measurement_noise_;
    
    // Kalman gain: K = P * H^T * S^-1
    cv::Mat kalman_gain = track.covariance * measurement_matrix_.t() * innovation_cov.inv();
    
    // Update state: x = x + K * y
    track.state = track.state + kalman_gain * innovation;
    
    // Update covariance: P = (I - K * H) * P
    cv::Mat identity = cv::Mat::eye(8, 8, CV_32F);
    track.covariance = (identity - kalman_gain * measurement_matrix_) * track.covariance;
    
    // Update bbox
    track.bbox.x = track.state.at<float>(0, 0) - track.state.at<float>(2, 0) / 2;
    track.bbox.y = track.state.at<float>(1, 0) - track.state.at<float>(3, 0) / 2;
    track.bbox.width = track.state.at<float>(2, 0);
    track.bbox.height = track.state.at<float>(3, 0);
}

// Hungarian Algorithm Implementation (simplified)
std::vector<std::pair<int, int>> UltraTracker::hungarian_assignment(const cv::Mat& cost_matrix) {
    std::vector<std::pair<int, int>> assignments;
    
    if (cost_matrix.rows == 0 || cost_matrix.cols == 0) {
        return assignments;
    }
    
    // Simple greedy assignment for performance (can be replaced with full Hungarian)
    std::vector<bool> row_assigned(cost_matrix.rows, false);
    std::vector<bool> col_assigned(cost_matrix.cols, false);
    
    while (true) {
        float min_cost = std::numeric_limits<float>::max();
        int min_row = -1, min_col = -1;
        
        for (int i = 0; i < cost_matrix.rows; i++) {
            if (row_assigned[i]) continue;
            for (int j = 0; j < cost_matrix.cols; j++) {
                if (col_assigned[j]) continue;
                if (cost_matrix.at<float>(i, j) < min_cost) {
                    min_cost = cost_matrix.at<float>(i, j);
                    min_row = i;
                    min_col = j;
                }
            }
        }
        
        if (min_row == -1 || min_cost > 1.0f) break;  // No more valid assignments
        
        assignments.emplace_back(min_row, min_col);
        row_assigned[min_row] = true;
        col_assigned[min_col] = true;
    }
    
    return assignments;
}

std::vector<Track> UltraTracker::get_active_tracks() const {
    std::vector<Track> result;
    std::copy_if(active_tracks_.begin(), active_tracks_.end(), std::back_inserter(result),
                [](const Track& track) { return track.is_activated && track.time_since_update < 2; });
    return result;
}

// Correlation Filter Implementation
cv::Mat UltraTracker::create_correlation_filter(const cv::Mat& patch) {
    if (patch.empty()) return cv::Mat();
    
    // Resize patch to template size
    cv::Mat resized_patch;
    cv::resize(patch, resized_patch, template_size_);
    
    // Convert to float and normalize
    cv::Mat float_patch;
    resized_patch.convertTo(float_patch, CV_32F, 1.0/255.0);
    
    // Apply Hann window
    cv::Mat windowed_patch;
    if (float_patch.channels() == 3) {
        cv::cvtColor(float_patch, windowed_patch, cv::COLOR_BGR2GRAY);
    } else {
        windowed_patch = float_patch.clone();
    }
    
    windowed_patch = windowed_patch.mul(hann_window_);
    
    // Compute FFT of patch
    cv::Mat patch_fft = fft2d(windowed_patch);
    
    // Compute FFT of Gaussian target
    cv::Mat target_fft = fft2d(gaussian_target_);
    
    // Create filter: H = (Y* ⊙ X) / (X* ⊙ X + λ)
    cv::Mat numerator, denominator, filter;
    cv::mulSpectrums(target_fft, patch_fft, numerator, 0, true);
    cv::mulSpectrums(patch_fft, patch_fft, denominator, 0, true);
    
    // Add regularization term
    cv::Scalar lambda_scalar(lambda_, lambda_);
    denominator += lambda_scalar;
    
    // Compute filter
    cv::divide(numerator, denominator, filter);
    
    return filter;
}

cv::Mat UltraTracker::track_correlation_filter(const Track& track, const cv::Mat& frame) {
    if (track.correlation_filter.empty() || frame.empty()) {
        return cv::Mat();
    }
    
    // Extract search region (expanded bbox)
    cv::Rect2f search_bbox = track.bbox;
    float scale_factor = 2.0f; // Search region scale
    
    search_bbox.x -= search_bbox.width * (scale_factor - 1.0f) / 2.0f;
    search_bbox.y -= search_bbox.height * (scale_factor - 1.0f) / 2.0f;
    search_bbox.width *= scale_factor;
    search_bbox.height *= scale_factor;
    
    // Ensure search region is within frame bounds
    cv::Rect safe_search = cv::Rect(search_bbox) & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safe_search.area() <= 0) {
        return cv::Mat();
    }
    
    cv::Mat search_patch = frame(safe_search);
    
    // Resize to template size
    cv::Mat resized_search;
    cv::resize(search_patch, resized_search, template_size_);
    
    // Convert to float and normalize
    cv::Mat float_search;
    resized_search.convertTo(float_search, CV_32F, 1.0/255.0);
    
    // Convert to grayscale if needed
    cv::Mat gray_search;
    if (float_search.channels() == 3) {
        cv::cvtColor(float_search, gray_search, cv::COLOR_BGR2GRAY);
    } else {
        gray_search = float_search.clone();
    }
    
    // Apply Hann window
    gray_search = gray_search.mul(hann_window_);
    
    // Compute FFT of search patch
    cv::Mat search_fft = fft2d(gray_search);
    
    // Apply correlation filter
    cv::Mat response_fft;
    cv::mulSpectrums(track.correlation_filter, search_fft, response_fft, 0, true);
    
    // Compute inverse FFT to get response map
    cv::Mat response = ifft2d(response_fft);
    
    return response;
}

void UltraTracker::update_correlation_filter(Track& track, const cv::Mat& patch) {
    if (patch.empty()) return;
    
    try {
        // Create new filter from current patch
        cv::Mat new_filter = create_correlation_filter(patch);
        
        if (new_filter.empty()) return;
        
        // Update filter with learning rate
        if (track.correlation_filter.empty()) {
            track.correlation_filter = new_filter.clone();
        } else {
            // Linear interpolation: H = (1-α)*H_old + α*H_new
            track.correlation_filter = (1.0f - learning_rate_) * track.correlation_filter + 
                                      learning_rate_ * new_filter;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Error updating correlation filter: " << e.what() << std::endl;
    }
}

// Additional utility functions for production readiness
void UltraTracker::validate_input(const cv::Mat& frame) {
    if (frame.empty()) {
        throw std::invalid_argument("Input frame is empty");
    }
    if (frame.channels() != 3 && frame.channels() != 1) {
        throw std::invalid_argument("Input frame must be 1 or 3 channel image");
    }
    if (frame.type() != CV_8UC3 && frame.type() != CV_8UC1) {
        throw std::invalid_argument("Input frame must be 8-bit unsigned integer type");
    }
}

bool UltraTracker::is_bbox_valid(const cv::Rect2f& bbox, const cv::Size& frame_size) {
    return bbox.x >= 0 && bbox.y >= 0 && 
           bbox.x + bbox.width <= frame_size.width &&
           bbox.y + bbox.height <= frame_size.height &&
           bbox.width > 0 && bbox.height > 0;
}

void UltraTracker::reset_tracker() {
    active_tracks_.clear();
    lost_tracks_.clear();
    next_track_id_ = 1;
}

std::vector<Track> UltraTracker::get_all_tracks() const {
    return active_tracks_;
}

size_t UltraTracker::get_track_count() const {
    return active_tracks_.size();
}

void UltraTracker::set_nms_threshold(float threshold) {
    if (threshold >= 0.0f && threshold <= 1.0f) {
        nms_threshold_ = threshold;
    }
}

void UltraTracker::set_template_size(const cv::Size& size) {
    if (size.width > 0 && size.height > 0) {
        template_size_ = size;
        // Recreate Hann window for new template size
        cv::Mat hann_1d = create_hann_window(template_size_.width);
        cv::Mat hann_2d;
        cv::mulTransposed(hann_1d, hann_2d, false);
        hann_window_ = hann_2d;
        
        // Recreate Gaussian target
        cv::Mat gaussian_1d_x = cv::getGaussianKernel(template_size_.width, sigma_, CV_32F);
        cv::Mat gaussian_1d_y = cv::getGaussianKernel(template_size_.height, sigma_, CV_32F);
        gaussian_target_ = gaussian_1d_y * gaussian_1d_x.t();
    }
}

cv::Mat UltraTracker::create_hann_window(int size) {
    cv::Mat hann(1, size, CV_32F);
    float* data = hann.ptr<float>();
    
    for (int i = 0; i < size; i++) {
        data[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(CV_PI) * i / (size - 1)));
    }
    
    return hann;
}

} // namespace ultratrack
