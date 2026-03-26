// src/features/gray_feature.cpp
#include "features/gray_feature.hpp"
#include "errors.hpp"

namespace ultratrack {

cv::Mat GrayFeature::extract(const cv::Mat& patch) const {
    validate_patch(patch, "GrayFeature::extract");
    
    cv::Mat gray;
    if (patch.channels() == 3) {
        cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = patch.clone();
    }
    
    // Normalize to [0, 1]
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);
    
    // Subtract mean for zero-centering
    cv::Scalar mean = cv::mean(gray);
    gray -= mean[0];
    
    return gray;
}

} // namespace ultratrack
