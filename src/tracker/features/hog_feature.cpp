// src/tracker/features/hog_feature.cpp
#include <ultratrack/tracker/features/hog_feature.hpp>
#include "errors.hpp"
#include <cmath>

namespace ultratrack {

HOGFeature::HOGFeature(const FeatureConfig& config) : config_(config) {}

cv::Mat HOGFeature::extract(const cv::Mat& patch) const {
    validate_patch(patch, "HOGFeature::extract");
    
    // Convert to grayscale if needed
    cv::Mat gray;
    if (patch.channels() == 3) {
        cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = patch.clone();
    }
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);
    
    // Compute gradients
    cv::Mat magnitude, orientation;
    compute_gradients(gray, magnitude, orientation);
    
    // Compute HOG cells
    cv::Mat hog = compute_hog_cells(magnitude, orientation, config_.cell_size, config_.num_bins);
    
    return hog;
}

void HOGFeature::compute_gradients(const cv::Mat& gray, cv::Mat& magnitude, cv::Mat& orientation) const {
    cv::Mat gx, gy;
    
    // Sobel gradients
    cv::Sobel(gray, gx, CV_32F, 1, 0, 1);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 1);
    
    // Magnitude and orientation
    cv::cartToPolar(gx, gy, magnitude, orientation, true);  // degrees
}

cv::Mat HOGFeature::compute_hog_cells(const cv::Mat& magnitude, const cv::Mat& orientation,
                                       int cell_size, int num_bins) const {
    int cells_x = magnitude.cols / cell_size;
    int cells_y = magnitude.rows / cell_size;
    
    if (cells_x < 1 || cells_y < 1) {
        return cv::Mat();
    }
    
    // FHOG uses 18 contrast-sensitive + 9 contrast-insensitive + 4 texture = 31 dims
    // Simplified: 9 unsigned + 18 signed + 4 = 31
    const int feat_dim = 31;
    cv::Mat features = cv::Mat::zeros(cells_y, cells_x, CV_32FC(feat_dim));
    
    float bin_width = 360.0f / num_bins;
    
    for (int cy = 0; cy < cells_y; cy++) {
        for (int cx = 0; cx < cells_x; cx++) {
            std::vector<float> hist(feat_dim, 0.0f);
            
            // Accumulate over cell
            for (int py = 0; py < cell_size; py++) {
                for (int px = 0; px < cell_size; px++) {
                    int y = cy * cell_size + py;
                    int x = cx * cell_size + px;
                    
                    if (y >= magnitude.rows || x >= magnitude.cols) continue;
                    
                    float mag = magnitude.at<float>(y, x);
                    float angle = orientation.at<float>(y, x);
                    
                    // Unsigned bin (0-180)
                    int bin_unsigned = static_cast<int>(fmod(angle, 180.0f) / (180.0f / num_bins)) % num_bins;
                    hist[bin_unsigned] += mag;
                    
                    // Signed bin (0-360) 
                    int bin_signed = static_cast<int>(angle / bin_width) % (num_bins * 2);
                    hist[num_bins + bin_signed] += mag;
                }
            }
            
            // Texture features (gradients of histogram)
            float texture_sum = 0;
            for (int i = 0; i < num_bins; i++) {
                texture_sum += hist[i];
            }
            hist[27] = texture_sum / (cell_size * cell_size);
            hist[28] = hist[27] * hist[27];
            hist[29] = std::sqrt(hist[27]);
            hist[30] = hist[27] > 0.1f ? 1.0f : 0.0f;
            
            // Normalize
            float norm = 0;
            for (int i = 0; i < feat_dim; i++) norm += hist[i] * hist[i];
            norm = std::sqrt(norm) + 1e-5f;
            for (int i = 0; i < feat_dim; i++) hist[i] /= norm;
            
            // Store
            float* ptr = features.ptr<float>(cy, cx);
            for (int i = 0; i < feat_dim; i++) {
                ptr[i] = hist[i];
            }
        }
    }
    
    return features;
}

} // namespace ultratrack
