#pragma once
#include <opencv2/core.hpp>
#include <cstdint>

namespace ultratrack {

using Rect2f = cv::Rect2f;
using Point2f = cv::Point2f;
using Size2f = cv::Size2f;

enum class FrameFormat { BGR, RGB, GRAY, NV12, GPU_HANDLE };

struct Frame {
    int width = 0;
    int height = 0;
    FrameFormat format = FrameFormat::BGR;
    cv::Mat data;        // CPU data
    void* gpu_handle = nullptr; // opaque GPU handle
    int64_t timestamp_us = 0;
};

struct Detection {
    uint64_t id = 0;
    Rect2f bbox;
    float confidence = 0.0f;
    int class_id = -1;
    cv::Mat feature;
};

enum class TrackState { TENTATIVE, CONFIRMED, LOST };

struct Track {
    uint64_t id = 0;
    Rect2f bbox;
    float confidence = 0.0f;
    TrackState state = TrackState::TENTATIVE;
    int age = 0;
    int hits = 0;
    int time_since_update = 0;
    cv::Mat kalman_state;
    cv::Mat kalman_covariance;
    cv::Mat appearance_feature;
    cv::Mat correlation_filter;
};

} // namespace ultratrack
