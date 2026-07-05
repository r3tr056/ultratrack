#pragma once

#include <ultratrack/core/result.hpp>
#include <ultratrack/core/types.hpp>
#include <opencv2/core.hpp>

namespace ultratrack {

/// Simple 8-state Kalman filter for bounding-box tracking.
///
/// State vector: [center_x, center_y, width, height, v_center_x, v_center_y, v_width, v_height].
class KalmanFilter {
public:
    explicit KalmanFilter(const Rect2f& initial_bbox);

    /// Advance the state estimate by one time step.
    Status predict();

    /// Fuse a new measurement into the state estimate.
    Status update(const Rect2f& measured_bbox);

    /// Return the bounding box implied by the current state.
    Result<Rect2f> stateBBox() const;

private:
    void initMatrices();

    cv::Mat state_;          ///< 8x1 state vector.
    cv::Mat covariance_;     ///< 8x8 error covariance.
    cv::Mat transition_;     ///< 8x8 state transition matrix.
    cv::Mat measurement_;    ///< 4x8 measurement matrix.
    cv::Mat process_noise_;  ///< 8x8 process noise covariance.
    cv::Mat measurement_noise_; ///< 4x4 measurement noise covariance.
};

} // namespace ultratrack
