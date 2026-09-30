// tests/test_features.cpp
#include "feature_extractor.hpp"
#include "features/hog_feature.hpp"
#include "features/gray_feature.hpp"
#include "features/cn_feature.hpp"
#include "errors.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

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

// --- Regression tests for discovered bugs ---

void test_feature_size_mismatch_regression() {
    // BUG #1 REGRESSION: HOG returns cell-based (16x16), Gray returns pixel-based (64x64),
    // CN returns pixel-based (64x64). MultiFeatureExtractor must resize them to a common
    // spatial size before concatenation.
    MultiFeatureExtractor ext(TrackingMode::ACCURATE);

    cv::Mat patch = cv::Mat::zeros(64, 64, CV_8UC3);
    cv::rectangle(patch, cv::Rect(16, 16, 32, 32), cv::Scalar(128, 64, 255), -1);

    cv::Mat features = ext.extract(patch);

    assert(!features.empty());
    // Result should have 42 channels (31 HOG + 1 Gray + 10 CN)
    assert(features.channels() == 42);
    // All features should be resized to HOG's cell-based grid size
    // For 64x64 with cell_size=4: 16x16 cells
    assert(features.rows == 16);
    assert(features.cols == 16);

    std::cout << "  PASS: test_feature_size_mismatch_regression\n";
}

void test_balanced_mode_feature_size_mismatch() {
    // Same regression check for BALANCED mode (HOG + Gray)
    MultiFeatureExtractor ext(TrackingMode::BALANCED);

    cv::Mat patch = cv::Mat::zeros(64, 64, CV_8UC3);
    cv::rectangle(patch, cv::Rect(10, 10, 40, 40), cv::Scalar(200, 200, 200), -1);

    cv::Mat features = ext.extract(patch);

    assert(!features.empty());
    assert(features.channels() == 32);  // 31 HOG + 1 Gray
    // Spatial size should match HOG output
    assert(features.rows == 16);
    assert(features.cols == 16);

    std::cout << "  PASS: test_balanced_mode_feature_size_mismatch\n";
}

void test_hog_feature_output_size() {
    // Verify HOG output spatial dimensions are cell-based
    FeatureConfig config;
    config.cell_size = 4;
    HOGFeature hog(config);

    cv::Mat patch = cv::Mat::zeros(64, 64, CV_8UC3);
    cv::rectangle(patch, cv::Rect(10, 10, 40, 40), cv::Scalar(255, 255, 255), -1);
    cv::Mat features = hog.extract(patch);

    assert(!features.empty());
    assert(features.rows == 64 / config.cell_size);  // 16
    assert(features.cols == 64 / config.cell_size);  // 16
    assert(features.channels() == 31);

    std::cout << "  PASS: test_hog_feature_output_size\n";
}

void test_gray_feature_output_size() {
    // Verify Gray output is pixel-based
    GrayFeature gray;

    cv::Mat patch = cv::Mat::zeros(64, 64, CV_8UC3);
    cv::Mat features = gray.extract(patch);

    assert(!features.empty());
    assert(features.rows == 64);
    assert(features.cols == 64);
    assert(features.channels() == 1);

    std::cout << "  PASS: test_gray_feature_output_size\n";
}

void test_cn_feature_output_size() {
    // Verify CN output is pixel-based
    CNFeature cn;

    cv::Mat patch = cv::Mat(64, 64, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat features = cn.extract(patch);

    assert(!features.empty());
    assert(features.rows == 64);
    assert(features.cols == 64);
    assert(features.channels() == 10);

    std::cout << "  PASS: test_cn_feature_output_size\n";
}

void test_different_patch_sizes() {
    // Test with various patch sizes to ensure robustness
    MultiFeatureExtractor ext(TrackingMode::ACCURATE);

    std::vector<int> sizes = {32, 48, 64, 96, 128};
    for (int sz : sizes) {
        cv::Mat patch = cv::Mat::zeros(sz, sz, CV_8UC3);
        cv::rectangle(patch, cv::Rect(sz/4, sz/4, sz/2, sz/2), cv::Scalar(100, 200, 50), -1);

        cv::Mat features = ext.extract(patch);

        assert(!features.empty());
        assert(features.channels() == 42);
        // HOG cell-based output: sz / cell_size
        int expected_cells = sz / 4;
        assert(features.rows == expected_cells);
        assert(features.cols == expected_cells);
    }

    std::cout << "  PASS: test_different_patch_sizes\n";
}

void test_empty_patch_throws() {
    MultiFeatureExtractor ext(TrackingMode::FAST);

    bool threw = false;
    try {
        cv::Mat empty;
        ext.extract(empty);
    } catch (const TrackerException& e) {
        threw = true;
        assert(e.code() == ErrorCode::EMPTY_FRAME);
    }
    assert(threw);

    std::cout << "  PASS: test_empty_patch_throws\n";
}

void test_too_small_patch_throws() {
    MultiFeatureExtractor ext(TrackingMode::FAST);

    bool threw = false;
    try {
        cv::Mat tiny(2, 2, CV_8UC3, cv::Scalar(128, 128, 128));
        ext.extract(tiny);
    } catch (const TrackerException& e) {
        threw = true;
        assert(e.code() == ErrorCode::INVALID_PATCH_SIZE);
    }
    assert(threw);

    std::cout << "  PASS: test_too_small_patch_throws\n";
}

void test_cn_grayscale_input() {
    // Test CN feature with grayscale input (should convert internally)
    CNFeature cn;

    cv::Mat gray_patch(64, 64, CV_8UC1, cv::Scalar(128));
    cv::Mat features = cn.extract(gray_patch);

    assert(!features.empty());
    assert(features.channels() == 10);

    std::cout << "  PASS: test_cn_grayscale_input\n";
}

void test_hog_grayscale_input() {
    // Test HOG feature with grayscale input
    HOGFeature hog;

    cv::Mat gray_patch(64, 64, CV_8UC1, cv::Scalar(128));
    cv::Mat features = hog.extract(gray_patch);

    assert(!features.empty());
    assert(features.channels() == 31);

    std::cout << "  PASS: test_hog_grayscale_input\n";
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

    // Regression tests
    test_feature_size_mismatch_regression();
    test_balanced_mode_feature_size_mismatch();
    test_hog_feature_output_size();
    test_gray_feature_output_size();
    test_cn_feature_output_size();
    test_different_patch_sizes();
    test_empty_patch_throws();
    test_too_small_patch_throws();
    test_cn_grayscale_input();
    test_hog_grayscale_input();

    std::cout << "All feature extraction tests passed!\n";
    return 0;
}
