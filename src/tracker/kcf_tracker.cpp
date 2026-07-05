#include <ultratrack/tracker/kcf_tracker.hpp>
#include "simd_helpers.hpp"

#include <opencv2/imgproc.hpp>

namespace ultratrack {

Result<std::unique_ptr<KCFTracker>> KCFTracker::create(const Config& cfg) {
    if (cfg.template_size.width <= 0 || cfg.template_size.height <= 0) {
        return Status(ErrorCode::INVALID_ARGUMENT, "template size must be positive");
    }
    return std::unique_ptr<KCFTracker>(new KCFTracker(cfg));
}

KCFTracker::KCFTracker(const Config& cfg) : cfg_(cfg) {
    feature_extractor_ = std::make_unique<MultiFeatureExtractor>(cfg.feature_mode, cfg.feature_config);

    cv::Mat hann_1d = createHannWindow(cfg.template_size.width);
    cv::mulTransposed(hann_1d, hann_window_, true);
    hann_window_.convertTo(hann_window_, CV_32FC1);

    cv::Mat gx = cv::getGaussianKernel(cfg.template_size.width, cfg.sigma, CV_32F);
    cv::Mat gy = cv::getGaussianKernel(cfg.template_size.height, cfg.sigma, CV_32F);
    gaussian_target_ = gy * gx.t();
}

cv::Mat KCFTracker::createHannWindow(int size) {
    return internal::create_hann_window(size);
}

Status KCFTracker::init(Track& track, const cv::Mat& frame) {
    if (frame.empty()) {
        return Status(ErrorCode::EMPTY_FRAME, "empty frame");
    }

    cv::Rect safe = cv::Rect(track.bbox) & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safe.area() <= 0) {
        return Status(ErrorCode::INVALID_PATCH_SIZE, "invalid bbox");
    }

    cv::Mat patch = frame(safe);
    track.correlation_filter = createFilter(patch);
    if (track.correlation_filter.empty()) {
        return Status(ErrorCode::FEATURE_EXTRACTION_FAILED, "failed to create correlation filter");
    }
    return Status();
}

Result<Rect2f> KCFTracker::predict(Track& track, const cv::Mat& frame) {
    if (track.correlation_filter.empty()) {
        return Status(ErrorCode::TRACKING_LOST, "no correlation filter");
    }
    if (frame.empty()) {
        return Status(ErrorCode::EMPTY_FRAME, "empty frame");
    }

    // Build a search region centred on the current bbox, twice the object size.
    cv::Rect2f search_bbox = track.bbox;
    constexpr float scale_factor = 2.0f;
    search_bbox.x -= search_bbox.width * (scale_factor - 1.0f) / 2.0f;
    search_bbox.y -= search_bbox.height * (scale_factor - 1.0f) / 2.0f;
    search_bbox.width *= scale_factor;
    search_bbox.height *= scale_factor;

    cv::Rect safe_search = cv::Rect(search_bbox) & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safe_search.area() <= 0) {
        return Status(ErrorCode::INVALID_PATCH_SIZE, "invalid search region");
    }

    cv::Mat search_patch = frame(safe_search);
    cv::Mat resized_search;
    cv::resize(search_patch, resized_search, cfg_.template_size);

    cv::Mat gray_search;
    if (resized_search.channels() == 3) {
        cv::cvtColor(resized_search, gray_search, cv::COLOR_BGR2GRAY);
    } else {
        gray_search = resized_search.clone();
    }

    cv::Mat float_search;
    gray_search.convertTo(float_search, CV_32F, 1.0 / 255.0);

    internal::simd_hann_window(float_search.ptr<float>(), hann_window_.ptr<float>(),
                               float_search.ptr<float>(), static_cast<int>(float_search.total()));

    cv::Mat search_fft = internal::fft2d(float_search);
    if (search_fft.empty()) {
        return Status(ErrorCode::FEATURE_EXTRACTION_FAILED, "search fft failed");
    }

    cv::Mat response_fft(track.correlation_filter.size(), track.correlation_filter.type());
    internal::simd_mul_spectrums(track.correlation_filter.ptr<float>(), search_fft.ptr<float>(),
                                 response_fft.ptr<float>(),
                                 static_cast<int>(track.correlation_filter.total()), true);

    cv::Mat response = internal::ifft2d(response_fft);
    if (response.empty()) {
        return Status(ErrorCode::FEATURE_EXTRACTION_FAILED, "response ifft failed");
    }

    // Locate the peak in the correlation response.
    double min_val = 0.0;
    double max_val = 0.0;
    cv::Point min_loc;
    cv::Point max_loc;
    cv::minMaxLoc(response, &min_val, &max_val, &min_loc, &max_loc);

    const float center_x = static_cast<float>(response.cols) * 0.5f;
    const float center_y = static_cast<float>(response.rows) * 0.5f;
    const float dx_map = static_cast<float>(max_loc.x) - center_x;
    const float dy_map = static_cast<float>(max_loc.y) - center_y;

    // Map the displacement from response space back to image space.
    const float scale_x = static_cast<float>(safe_search.width) / static_cast<float>(response.cols);
    const float scale_y = static_cast<float>(safe_search.height) / static_cast<float>(response.rows);

    const float dx_frame = dx_map * scale_x;
    const float dy_frame = dy_map * scale_y;

    const cv::Point2f old_center(track.bbox.x + track.bbox.width * 0.5f,
                                 track.bbox.y + track.bbox.height * 0.5f);
    const cv::Point2f new_center(old_center.x + dx_frame, old_center.y + dy_frame);

    track.bbox.x = new_center.x - track.bbox.width * 0.5f;
    track.bbox.y = new_center.y - track.bbox.height * 0.5f;

    return track.bbox;
}

Status KCFTracker::update(Track& track, const cv::Mat& frame, const Rect2f& detected_bbox) {
    track.bbox = detected_bbox;

    cv::Rect safe = cv::Rect(detected_bbox) & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safe.area() <= 0) {
        return Status(ErrorCode::INVALID_PATCH_SIZE, "invalid update bbox");
    }

    cv::Mat patch = frame(safe);
    cv::Mat new_filter = createFilter(patch);
    if (new_filter.empty()) {
        return Status(ErrorCode::FEATURE_EXTRACTION_FAILED, "filter update failed");
    }

    if (track.correlation_filter.empty()) {
        track.correlation_filter = new_filter.clone();
    } else {
        track.correlation_filter = (1.0f - cfg_.learning_rate) * track.correlation_filter +
                                   cfg_.learning_rate * new_filter;
    }
    return Status();
}

cv::Mat KCFTracker::createFilter(const cv::Mat& patch) {
    if (patch.empty()) {
        return cv::Mat();
    }

    cv::Mat resized_patch;
    cv::resize(patch, resized_patch, cfg_.template_size);

    cv::Mat gray_patch;
    if (resized_patch.channels() == 3) {
        cv::cvtColor(resized_patch, gray_patch, cv::COLOR_BGR2GRAY);
    } else {
        gray_patch = resized_patch.clone();
    }

    cv::Mat float_patch;
    gray_patch.convertTo(float_patch, CV_32FC1, 1.0 / 255.0);

    cv::Mat windowed_patch = float_patch.clone();

    internal::simd_hann_window(windowed_patch.ptr<float>(), hann_window_.ptr<float>(),
                               windowed_patch.ptr<float>(),
                               static_cast<int>(windowed_patch.total()));

    cv::Mat patch_fft = internal::fft2d(windowed_patch);
    cv::Mat target_fft = internal::fft2d(gaussian_target_);

    if (patch_fft.empty() || target_fft.empty()) {
        return cv::Mat();
    }

    if (patch_fft.type() != CV_32FC2) patch_fft.convertTo(patch_fft, CV_32FC2);
    if (target_fft.type() != CV_32FC2) target_fft.convertTo(target_fft, CV_32FC2);

    cv::Mat numerator(target_fft.size(), target_fft.type());
    internal::simd_mul_spectrums(target_fft.ptr<float>(), patch_fft.ptr<float>(),
                                 numerator.ptr<float>(),
                                 static_cast<int>(target_fft.total()), true);

    cv::Mat denominator(patch_fft.size(), patch_fft.type());
    internal::simd_mul_spectrums(patch_fft.ptr<float>(), patch_fft.ptr<float>(),
                                 denominator.ptr<float>(),
                                 static_cast<int>(patch_fft.total()), true);

    cv::add(denominator, cv::Scalar::all(cfg_.lambda), denominator);

    cv::Mat filter(numerator.size(), numerator.type());
    internal::simd_div_spectrums(numerator.ptr<float>(), denominator.ptr<float>(),
                                 filter.ptr<float>(),
                                 static_cast<int>(numerator.total()));
    return filter;
}

} // namespace ultratrack
