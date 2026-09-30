// tests/test_features.cpp
#include <ultratrack/tracker/features/feature_extractor.hpp>
#include <ultratrack/tracker/features/hog_feature.hpp>
#include <ultratrack/tracker/features/gray_feature.hpp>
#include <ultratrack/tracker/features/cn_feature.hpp>
#include <catch2/catch_test_macros.hpp>
#include "errors.hpp"
#include <iostream>
#include <cmath>

using namespace ultratrack;

TEST_CASE("HOG dimensions and name", "[features]") {
    HOGFeature hog;
    REQUIRE(hog.dimensions() == 31);
    REQUIRE(hog.name() == "HOG");
}

TEST_CASE("HOG extraction", "[features]") {
    HOGFeature hog;

    cv::Mat patch = cv::Mat::zeros(64, 64, CV_8UC3);
    cv::rectangle(patch, cv::Rect(16, 16, 32, 32), cv::Scalar(255, 255, 255), -1);

    cv::Mat features = hog.extract(patch);

    REQUIRE(!features.empty());
    REQUIRE(features.channels() == 31);
}

TEST_CASE("Gray dimensions and name", "[features]") {
    GrayFeature gray;
    REQUIRE(gray.dimensions() == 1);
    REQUIRE(gray.name() == "Gray");
}

TEST_CASE("Gray extraction", "[features]") {
    GrayFeature gray;

    cv::Mat patch = cv::Mat::ones(64, 64, CV_8UC3) * 128;
    cv::Mat features = gray.extract(patch);

    REQUIRE(!features.empty());
    REQUIRE(features.channels() == 1);
    REQUIRE(features.type() == CV_32FC1);
}

TEST_CASE("CN dimensions and name", "[features]") {
    CNFeature cn;
    REQUIRE(cn.dimensions() == 10);
    REQUIRE(cn.name() == "ColorNames");
}

TEST_CASE("CN extraction", "[features]") {
    CNFeature cn;

    // Create a red patch
    cv::Mat patch = cv::Mat(64, 64, CV_8UC3, cv::Scalar(0, 0, 255));  // BGR red
    cv::Mat features = cn.extract(patch);

    REQUIRE(!features.empty());
    REQUIRE(features.channels() == 10);
}

TEST_CASE("Multi-feature extractor FAST mode", "[features]") {
    MultiFeatureExtractor ext(TrackingMode::FAST);

    REQUIRE(ext.total_dimensions() == 31);  // HOG only
    REQUIRE(ext.get_mode() == TrackingMode::FAST);
}

TEST_CASE("Multi-feature extractor BALANCED mode", "[features]") {
    MultiFeatureExtractor ext(TrackingMode::BALANCED);

    REQUIRE(ext.total_dimensions() == 32);  // HOG + Gray
    REQUIRE(ext.get_mode() == TrackingMode::BALANCED);
}

TEST_CASE("Multi-feature extractor ACCURATE mode", "[features]") {
    MultiFeatureExtractor ext(TrackingMode::ACCURATE);

    REQUIRE(ext.total_dimensions() == 42);  // HOG + Gray + CN
    REQUIRE(ext.get_mode() == TrackingMode::ACCURATE);
}

TEST_CASE("Multi-feature extractor mode switching", "[features]") {
    MultiFeatureExtractor ext(TrackingMode::FAST);
    REQUIRE(ext.total_dimensions() == 31);

    ext.set_mode(TrackingMode::ACCURATE);
    REQUIRE(ext.total_dimensions() == 42);

    ext.set_mode(TrackingMode::BALANCED);
    REQUIRE(ext.total_dimensions() == 32);
}

TEST_CASE("Multi-feature extraction", "[features]") {
    MultiFeatureExtractor ext(TrackingMode::ACCURATE);

    cv::Mat patch = cv::Mat::zeros(64, 64, CV_8UC3);
    cv::rectangle(patch, cv::Rect(16, 16, 32, 32), cv::Scalar(0, 0, 255), -1);

    cv::Mat features = ext.extract(patch);

    REQUIRE(!features.empty());
}

// --- Regression tests for discovered bugs ---

TEST_CASE("feature size mismatch regression", "[features][regression]") {
    // BUG #1 REGRESSION: HOG returns cell-based (16x16), Gray returns pixel-based (64x64),
    // CN returns pixel-based (64x64). MultiFeatureExtractor must resize them to a common
    // spatial size before concatenation.
    MultiFeatureExtractor ext(TrackingMode::ACCURATE);

    cv::Mat patch = cv::Mat::zeros(64, 64, CV_8UC3);
    cv::rectangle(patch, cv::Rect(16, 16, 32, 32), cv::Scalar(128, 64, 255), -1);

    cv::Mat features = ext.extract(patch);

    REQUIRE(!features.empty());
    // Result should have 42 channels (31 HOG + 1 Gray + 10 CN)
    REQUIRE(features.channels() == 42);
    // All features should be resized to HOG's cell-based grid size
    // For 64x64 with cell_size=4: 16x16 cells
    REQUIRE(features.rows == 16);
    REQUIRE(features.cols == 16);
}

TEST_CASE("balanced mode feature size mismatch", "[features][regression]") {
    // Same regression check for BALANCED mode (HOG + Gray)
    MultiFeatureExtractor ext(TrackingMode::BALANCED);

    cv::Mat patch = cv::Mat::zeros(64, 64, CV_8UC3);
    cv::rectangle(patch, cv::Rect(10, 10, 40, 40), cv::Scalar(200, 200, 200), -1);

    cv::Mat features = ext.extract(patch);

    REQUIRE(!features.empty());
    REQUIRE(features.channels() == 32);  // 31 HOG + 1 Gray
    // Spatial size should match HOG output
    REQUIRE(features.rows == 16);
    REQUIRE(features.cols == 16);
}

TEST_CASE("hog feature output size", "[features][regression]") {
    // Verify HOG output spatial dimensions are cell-based
    FeatureConfig config;
    config.cell_size = 4;
    HOGFeature hog(config);

    cv::Mat patch = cv::Mat::zeros(64, 64, CV_8UC3);
    cv::rectangle(patch, cv::Rect(10, 10, 40, 40), cv::Scalar(255, 255, 255), -1);
    cv::Mat features = hog.extract(patch);

    REQUIRE(!features.empty());
    REQUIRE(features.rows == 64 / config.cell_size);  // 16
    REQUIRE(features.cols == 64 / config.cell_size);  // 16
    REQUIRE(features.channels() == 31);
}

TEST_CASE("gray feature output size", "[features][regression]") {
    // Verify Gray output is pixel-based
    GrayFeature gray;

    cv::Mat patch = cv::Mat::zeros(64, 64, CV_8UC3);
    cv::Mat features = gray.extract(patch);

    REQUIRE(!features.empty());
    REQUIRE(features.rows == 64);
    REQUIRE(features.cols == 64);
    REQUIRE(features.channels() == 1);
}

TEST_CASE("cn feature output size", "[features][regression]") {
    // Verify CN output is pixel-based
    CNFeature cn;

    cv::Mat patch = cv::Mat(64, 64, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat features = cn.extract(patch);

    REQUIRE(!features.empty());
    REQUIRE(features.rows == 64);
    REQUIRE(features.cols == 64);
    REQUIRE(features.channels() == 10);
}

TEST_CASE("different patch sizes", "[features][regression]") {
    // Test with various patch sizes to ensure robustness
    MultiFeatureExtractor ext(TrackingMode::ACCURATE);

    std::vector<int> sizes = {32, 48, 64, 96, 128};
    for (int sz : sizes) {
        cv::Mat patch = cv::Mat::zeros(sz, sz, CV_8UC3);
        cv::rectangle(patch, cv::Rect(sz/4, sz/4, sz/2, sz/2), cv::Scalar(100, 200, 50), -1);

        cv::Mat features = ext.extract(patch);

        REQUIRE(!features.empty());
        REQUIRE(features.channels() == 42);
        // HOG cell-based output: sz / cell_size
        int expected_cells = sz / 4;
        REQUIRE(features.rows == expected_cells);
        REQUIRE(features.cols == expected_cells);
    }
}

TEST_CASE("empty patch throws", "[features][regression]") {
    MultiFeatureExtractor ext(TrackingMode::FAST);

    bool threw = false;
    try {
        cv::Mat empty;
        ext.extract(empty);
    } catch (const TrackerException& e) {
        threw = true;
        REQUIRE(e.code() == ErrorCode::EMPTY_FRAME);
    }
    REQUIRE(threw);
}

TEST_CASE("too small patch throws", "[features][regression]") {
    MultiFeatureExtractor ext(TrackingMode::FAST);

    bool threw = false;
    try {
        cv::Mat tiny(2, 2, CV_8UC3, cv::Scalar(128, 128, 128));
        ext.extract(tiny);
    } catch (const TrackerException& e) {
        threw = true;
        REQUIRE(e.code() == ErrorCode::INVALID_PATCH_SIZE);
    }
    REQUIRE(threw);
}

TEST_CASE("cn grayscale input", "[features][regression]") {
    // Test CN feature with grayscale input (should convert internally)
    CNFeature cn;

    cv::Mat gray_patch(64, 64, CV_8UC1, cv::Scalar(128));
    cv::Mat features = cn.extract(gray_patch);

    REQUIRE(!features.empty());
    REQUIRE(features.channels() == 10);
}

TEST_CASE("hog grayscale input", "[features][regression]") {
    // Test HOG feature with grayscale input
    HOGFeature hog;

    cv::Mat gray_patch(64, 64, CV_8UC1, cv::Scalar(128));
    cv::Mat features = hog.extract(gray_patch);

    REQUIRE(!features.empty());
    REQUIRE(features.channels() == 31);
}
