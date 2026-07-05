// tests/test_features.cpp
#include "feature_extractor.hpp"
#include "features/hog_feature.hpp"
#include "features/gray_feature.hpp"
#include "features/cn_feature.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iostream>

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
