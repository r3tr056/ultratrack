#include <ultratrack/tracker/kalman.hpp>

namespace ultratrack {

KalmanFilter::KalmanFilter(const Rect2f& initial_bbox) {
    initMatrices();

    state_ = (cv::Mat_<float>(8, 1) <<
              initial_bbox.x + initial_bbox.width * 0.5f,
              initial_bbox.y + initial_bbox.height * 0.5f,
              initial_bbox.width,
              initial_bbox.height,
              0.0f, 0.0f, 0.0f, 0.0f);

    covariance_ = cv::Mat::eye(8, 8, CV_32F) * 10.0f;
}

void KalmanFilter::initMatrices() {
    transition_ = (cv::Mat_<float>(8, 8) <<
                   1, 0, 0, 0, 1, 0, 0, 0,
                   0, 1, 0, 0, 0, 1, 0, 0,
                   0, 0, 1, 0, 0, 0, 1, 0,
                   0, 0, 0, 1, 0, 0, 0, 1,
                   0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 0, 1, 0, 0,
                   0, 0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 0, 0, 1);

    measurement_ = (cv::Mat_<float>(4, 8) <<
                    1, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0, 0, 0);

    process_noise_ = cv::Mat::eye(8, 8, CV_32F);
    cv::Mat q_diag = (cv::Mat_<float>(8, 1) << 1, 1, 1, 1, 0.01f, 0.01f, 0.01f, 0.01f);
    for (int i = 0; i < 8; ++i) {
        process_noise_.at<float>(i, i) = q_diag.at<float>(i, 0);
    }

    measurement_noise_ = cv::Mat::eye(4, 4, CV_32F) * 1.0f;
}

Status KalmanFilter::predict() {
    try {
        state_ = transition_ * state_;
        cv::Mat temp = transition_ * covariance_;
        covariance_ = temp * transition_.t() + process_noise_;
        return Status();
    } catch (const cv::Exception& e) {
        return Status(ErrorCode::INTERNAL_ERROR, std::string("Kalman predict failed: ") + e.what());
    } catch (const std::exception& e) {
        return Status(ErrorCode::INTERNAL_ERROR, std::string("Kalman predict failed: ") + e.what());
    } catch (...) {
        return Status(ErrorCode::INTERNAL_ERROR, "Kalman predict failed: unknown error");
    }
}

Status KalmanFilter::update(const Rect2f& measured_bbox) {
    try {
        cv::Mat measurement = (cv::Mat_<float>(4, 1) <<
                               measured_bbox.x + measured_bbox.width * 0.5f,
                               measured_bbox.y + measured_bbox.height * 0.5f,
                               measured_bbox.width,
                               measured_bbox.height);

        cv::Mat innovation = measurement - measurement_ * state_;
        cv::Mat temp = measurement_ * covariance_;
        cv::Mat innovation_cov = temp * measurement_.t() + measurement_noise_;
        cv::Mat innovation_cov_inv = innovation_cov.inv(cv::DECOMP_SVD);
        if (innovation_cov_inv.empty()) {
            return Status(ErrorCode::INTERNAL_ERROR, "Kalman update failed: innovation covariance is singular");
        }

        cv::Mat kalman_gain = covariance_ * measurement_.t() * innovation_cov_inv;

        state_ = state_ + kalman_gain * innovation;
        cv::Mat identity = cv::Mat::eye(8, 8, CV_32F);
        covariance_ = (identity - kalman_gain * measurement_) * covariance_;
        return Status();
    } catch (const cv::Exception& e) {
        return Status(ErrorCode::INTERNAL_ERROR, std::string("Kalman update failed: ") + e.what());
    } catch (const std::exception& e) {
        return Status(ErrorCode::INTERNAL_ERROR, std::string("Kalman update failed: ") + e.what());
    } catch (...) {
        return Status(ErrorCode::INTERNAL_ERROR, "Kalman update failed: unknown error");
    }
}

Result<Rect2f> KalmanFilter::stateBBox() const {
    try {
        const float cx = state_.at<float>(0);
        const float cy = state_.at<float>(1);
        const float w = state_.at<float>(2);
        const float h = state_.at<float>(3);
        return Rect2f(cx - w * 0.5f, cy - h * 0.5f, w, h);
    } catch (const cv::Exception& e) {
        return Status(ErrorCode::INTERNAL_ERROR, std::string("Kalman stateBBox failed: ") + e.what());
    } catch (const std::exception& e) {
        return Status(ErrorCode::INTERNAL_ERROR, std::string("Kalman stateBBox failed: ") + e.what());
    } catch (...) {
        return Status(ErrorCode::INTERNAL_ERROR, "Kalman stateBBox failed: unknown error");
    }
}

} // namespace ultratrack
