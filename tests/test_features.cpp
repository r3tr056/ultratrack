// tests/test_features.cpp
#include "feature_extractor.hpp"
#include "features/hog_feature.hpp"
#include "features/gray_feature.hpp"
#include "features/cn_feature.hpp"
#include <cassert>
#include <iostream>

using namespace ultratrack;

void test_hog_dimensions() {
    HOGFeature hog;
    assert(hog.dimensions() == 31);
    assert(hog.name() == "HOG");
    
    std::cout << "  PASS: test_hog_dimensions\n";
}

void test_hog_extraction() {
    HOGFeature hog;
    
    // Create test patch
    cv::Mat patch = cv::Mat::zeros(64, 64, CV_8UC3);
    cv::rectangle(patch, cv::Rect(16, 16, 32, 32), cv::Scalar(255, 255, 255), -1);
    
    cv::Mat features = hog.extract(patch);
    
    assert(!features.empty());
    assert(features.channels() == 31);
    
    std::cout << "  PASS: test_hog_extraction\n";
}

void test_gray_dimensions() {
    GrayFeature gray;
    assert(gray.dimensions() == 1);
    assert(gray.name() == "Gray");
    
    std::cout << "  PASS: test_gray_dimensions\n";
}

void test_gray_extraction() {
    GrayFeature gray;
    
    cv::Mat patch = cv::Mat::ones(64, 64, CV_8UC3) * 128;
    cv::Mat features = gray.extract(patch);
    
    assert(!features.empty());
    assert(features.channels() == 1);
    assert(features.type() == CV_32FC1);
    
    std::cout << "  PASS: test_gray_extraction\n";
}

void test_cn_dimensions() {
    CNFeature cn;
    assert(cn.dimensions() == 10);
    assert(cn.name() == "ColorNames");
    
    std::cout << "  PASS: test_cn_dimensions\n";
}

void test_cn_extraction() {
    CNFeature cn;
    
    // Create a red patch
    cv::Mat patch = cv::Mat(64, 64, CV_8UC3, cv::Scalar(0, 0, 255));  // BGR red
    cv::Mat features = cn.extract(patch);
    
    assert(!features.empty());
    assert(features.channels() == 10);
    
    std::cout << "  PASS: test_cn_extraction\n";
}

void test_multi_feature_fast_mode() {
    MultiFeatureExtractor ext(TrackingMode::FAST);
    
    assert(ext.total_dimensions() == 31);  // HOG only
    assert(ext.get_mode() == TrackingMode::FAST);
    
    std::cout << "  PASS: test_multi_feature_fast_mode\n";
}

void test_multi_feature_balanced_mode() {
    MultiFeatureExtractor ext(TrackingMode::BALANCED);
    
    assert(ext.total_dimensions() == 32);  // HOG + Gray
    assert(ext.get_mode() == TrackingMode::BALANCED);
    
    std::cout << "  PASS: test_multi_feature_balanced_mode\n";
}

void test_multi_feature_accurate_mode() {
    MultiFeatureExtractor ext(TrackingMode::ACCURATE);
    
    assert(ext.total_dimensions() == 42);  // HOG + Gray + CN
    assert(ext.get_mode() == TrackingMode::ACCURATE);
    
    std::cout << "  PASS: test_multi_feature_accurate_mode\n";
}

void test_mode_switching() {
    MultiFeatureExtractor ext(TrackingMode::FAST);
    assert(ext.total_dimensions() == 31);
    
    ext.set_mode(TrackingMode::ACCURATE);
    assert(ext.total_dimensions() == 42);
    
    ext.set_mode(TrackingMode::BALANCED);
    assert(ext.total_dimensions() == 32);
    
    std::cout << "  PASS: test_mode_switching\n";
}

void test_multi_feature_extraction() {
    MultiFeatureExtractor ext(TrackingMode::ACCURATE);
    
    cv::Mat patch = cv::Mat::zeros(64, 64, CV_8UC3);
    cv::rectangle(patch, cv::Rect(16, 16, 32, 32), cv::Scalar(0, 0, 255), -1);
    
    cv::Mat features = ext.extract(patch);
    
    assert(!features.empty());
    
    std::cout << "  PASS: test_multi_feature_extraction\n";
}

int main() {
    std::cout << "Running feature extraction tests...\n";
    
    test_hog_dimensions();
    test_hog_extraction();
    test_gray_dimensions();
    test_gray_extraction();
    test_cn_dimensions();
    test_cn_extraction();
    test_multi_feature_fast_mode();
    test_multi_feature_balanced_mode();
    test_multi_feature_accurate_mode();
    test_mode_switching();
    test_multi_feature_extraction();
    
    std::cout << "All feature extraction tests passed!\n";
    return 0;
}
