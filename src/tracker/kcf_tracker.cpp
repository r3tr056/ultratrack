#include <ultratrack/tracker/kcf_tracker.hpp>
#include "simd_helpers.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace ultratrack {

Result<std::unique_ptr<KCFTracker>> KCFTracker::create(const Config& cfg) {
    if (cfg.template_size.width <= 0 || cfg.template_size.height <= 0) {
        return Status(ErrorCode::INVALID_ARGUMENT, "template size must be positive");
    }
    try {
        return std::unique_ptr<KCFTracker>(new KCFTracker(cfg));
    } catch (const cv::Exception& e) {
        return Status(ErrorCode::FEATURE_EXTRACTION_FAILED,
                      std::string("failed to create KCF tracker: ") + e.what());
    } catch (const std::exception& e) {
        return Status(ErrorCode::FEATURE_EXTRACTION_FAILED,
                      std::string("failed to create KCF tracker: ") + e.what());
    } catch (...) {
        return Status(ErrorCode::INTERNAL_ERROR,
                      "failed to create KCF tracker: unknown error");
    }
}

KCFTracker::KCFTracker(const Config& cfg) : cfg_(cfg) {
    feature_extractor_ = std::make_unique<MultiFeatureExtractor>(cfg.feature_mode, cfg.feature_config);

    cv::Mat hann_x = internal::create_hann_window(cfg.template_size.width);
    cv::Mat hann_y = internal::create_hann_window(cfg.template_size.height);
    hann_window_ = hann_y.t() * hann_x;
    hann_window_.convertTo(hann_window_, CV_32FC1);

    cv::Mat gx = cv::getGaussianKernel(cfg.template_size.width, cfg.sigma, CV_32F);
    cv::Mat gy = cv::getGaussianKernel(cfg.template_size.height, cfg.sigma, CV_32F);
    gaussian_target_ = gy * gx.t();

    gaussian_target_fft_ = internal::fft2d(gaussian_target_);
}

Status KCFTracker::init(Track& track, const cv::Mat& frame) {
    try {
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
    } catch (const cv::Exception& e) {
        return Status(ErrorCode::FEATURE_EXTRACTION_FAILED, e.what());
    } catch (const std::exception& e) {
        return Status(ErrorCode::FEATURE_EXTRACTION_FAILED, e.what());
    } catch (...) {
        return Status(ErrorCode::INTERNAL_ERROR, "unknown error");
    }
}

Result<Rect2f> KCFTracker::predict(Track& track, const cv::Mat& frame) {
    try {
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

        auto channels = extractFeatureChannels(resized_search);
        if (channels.empty()) {
            return Status(ErrorCode::FEATURE_EXTRACTION_FAILED, "feature extraction failed");
        }

        auto filters = splitComplexFilters(track.correlation_filter);
        if (filters.empty() || filters.size() != channels.size()) {
            return Status(ErrorCode::FEATURE_EXTRACTION_FAILED,
                          "filter/feature channel count mismatch");
        }

        cv::Mat response_fft;
        for (size_t c = 0; c < channels.size(); ++c) {
            internal::simd_hann_window(channels[c].ptr<float>(), hann_window_.ptr<float>(),
                                       channels[c].ptr<float>(),
                                       static_cast<int>(channels[c].total()));

            cv::Mat search_fft = internal::fft2d(channels[c]);
            if (search_fft.empty()) {
                return Status(ErrorCode::FEATURE_EXTRACTION_FAILED, "search fft failed");
            }
            if (search_fft.type() != CV_32FC2) {
                search_fft.convertTo(search_fft, CV_32FC2);
            }

            cv::Mat channel_response(filters[c].size(), filters[c].type());
            internal::simd_mul_spectrums(filters[c].ptr<float>(), search_fft.ptr<float>(),
                                         channel_response.ptr<float>(),
                                         static_cast<int>(filters[c].total()), true);

            if (response_fft.empty()) {
                response_fft = channel_response;
            } else {
                cv::add(response_fft, channel_response, response_fft);
            }
        }

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

        // The response map dimensions are FFT-padded, not necessarily template_size.
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

        Rect2f predicted_bbox(new_center.x - track.bbox.width * 0.5f,
                              new_center.y - track.bbox.height * 0.5f,
                              track.bbox.width,
                              track.bbox.height);

        // Clamp the predicted bbox to the frame bounds.
        predicted_bbox.x = std::max(0.0f, std::min(predicted_bbox.x,
                                    static_cast<float>(frame.cols)));
        predicted_bbox.y = std::max(0.0f, std::min(predicted_bbox.y,
                                    static_cast<float>(frame.rows)));
        predicted_bbox.width = std::max(0.0f, std::min(predicted_bbox.width,
                                       static_cast<float>(frame.cols) - predicted_bbox.x));
        predicted_bbox.height = std::max(0.0f, std::min(predicted_bbox.height,
                                        static_cast<float>(frame.rows) - predicted_bbox.y));

        track.bbox = predicted_bbox;
        return track.bbox;
    } catch (const cv::Exception& e) {
        return Status(ErrorCode::TRACKING_LOST, e.what());
    } catch (const std::exception& e) {
        return Status(ErrorCode::TRACKING_LOST, e.what());
    } catch (...) {
        return Result<Rect2f>(Status(ErrorCode::INTERNAL_ERROR, "unknown error"));
    }
}

Status KCFTracker::update(Track& track, const cv::Mat& frame, const Rect2f& detected_bbox) {
    try {
        if (frame.empty()) {
            return Status(ErrorCode::EMPTY_FRAME, "empty frame");
        }

        cv::Rect safe = cv::Rect(detected_bbox) & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safe.area() <= 0) {
            return Status(ErrorCode::INVALID_PATCH_SIZE, "invalid update bbox");
        }

        cv::Mat patch = frame(safe);
        cv::Mat new_filter = createFilter(patch);
        if (new_filter.empty()) {
            return Status(ErrorCode::FEATURE_EXTRACTION_FAILED, "filter update failed");
        }

        // Only commit the new bbox after the filter was created successfully.
        // Store the clamped rectangle so the canonical bbox stays inside the frame.
        track.bbox = Rect2f(safe);

        if (track.correlation_filter.empty() ||
            track.correlation_filter.channels() != new_filter.channels()) {
            // Replace the filter on first update or when the feature representation changed.
            track.correlation_filter = new_filter.clone();
        } else {
            track.correlation_filter = (1.0f - cfg_.learning_rate) * track.correlation_filter +
                                       cfg_.learning_rate * new_filter;
        }
        return Status();
    } catch (const cv::Exception& e) {
        return Status(ErrorCode::FEATURE_EXTRACTION_FAILED, e.what());
    } catch (const std::exception& e) {
        return Status(ErrorCode::FEATURE_EXTRACTION_FAILED, e.what());
    } catch (...) {
        return Status(ErrorCode::INTERNAL_ERROR, "unknown error");
    }
}

cv::Mat KCFTracker::createFilter(const cv::Mat& patch) {
    if (patch.empty()) {
        return cv::Mat();
    }

    cv::Mat resized_patch;
    cv::resize(patch, resized_patch, cfg_.template_size);

    auto channels = extractFeatureChannels(resized_patch);
    if (channels.empty()) {
        return cv::Mat();
    }

    std::vector<cv::Mat> filters;
    filters.reserve(channels.size());

    for (auto& ch : channels) {
        internal::simd_hann_window(ch.ptr<float>(), hann_window_.ptr<float>(),
                                   ch.ptr<float>(),
                                   static_cast<int>(ch.total()));

        cv::Mat patch_fft = internal::fft2d(ch);
        if (patch_fft.empty() || gaussian_target_fft_.empty()) {
            return cv::Mat();
        }
        if (patch_fft.type() != CV_32FC2) {
            patch_fft.convertTo(patch_fft, CV_32FC2);
        }

        cv::Mat numerator(gaussian_target_fft_.size(), gaussian_target_fft_.type());
        internal::simd_mul_spectrums(gaussian_target_fft_.ptr<float>(), patch_fft.ptr<float>(),
                                     numerator.ptr<float>(),
                                     static_cast<int>(gaussian_target_fft_.total()), true);

        cv::Mat denominator(patch_fft.size(), patch_fft.type());
        internal::simd_mul_spectrums(patch_fft.ptr<float>(), patch_fft.ptr<float>(),
                                     denominator.ptr<float>(),
                                     static_cast<int>(patch_fft.total()), true);

        cv::add(denominator, cv::Scalar(cfg_.lambda, 0.0f), denominator);

        cv::Mat filter(numerator.size(), numerator.type());
        internal::simd_div_spectrums(numerator.ptr<float>(), denominator.ptr<float>(),
                                     filter.ptr<float>(),
                                     static_cast<int>(numerator.total()));
        filters.push_back(filter);
    }

    return mergeComplexFilters(filters);
}

std::vector<cv::Mat> KCFTracker::extractFeatureChannels(const cv::Mat& patch) const {
    std::vector<cv::Mat> channels;

    cv::Mat features;
    try {
        features = feature_extractor_->extract(patch);
    } catch (const cv::Exception&) {
        features = cv::Mat();
    } catch (const std::exception&) {
        features = cv::Mat();
    }

    if (!features.empty() && features.channels() > 0) {
        cv::split(features, channels);
        for (auto& ch : channels) {
            if (ch.size() != cfg_.template_size) {
                cv::Mat resized;
                cv::resize(ch, resized, cfg_.template_size);
                ch = resized;
            }
            ch.convertTo(ch, CV_32F);
        }
    }

    if (channels.empty()) {
        // Grayscale fallback when the configured feature extractor cannot produce features.
        cv::Mat gray;
        if (patch.channels() == 3) {
            cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = patch.clone();
        }
        gray.convertTo(gray, CV_32F, 1.0 / 255.0);
        channels.push_back(gray);
    }

    return channels;
}

cv::Mat KCFTracker::mergeComplexFilters(const std::vector<cv::Mat>& filters) {
    if (filters.empty()) {
        return cv::Mat();
    }

    std::vector<cv::Mat> all_planes;
    all_planes.reserve(filters.size() * 2);
    for (const auto& f : filters) {
        std::vector<cv::Mat> planes;
        cv::split(f, planes);
        all_planes.insert(all_planes.end(), planes.begin(), planes.end());
    }

    cv::Mat merged;
    cv::merge(all_planes, merged);
    return merged;
}

std::vector<cv::Mat> KCFTracker::splitComplexFilters(const cv::Mat& filter) {
    if (filter.empty() || filter.channels() % 2 != 0) {
        return {};
    }

    std::vector<cv::Mat> all_planes;
    cv::split(filter, all_planes);

    std::vector<cv::Mat> filters;
    filters.reserve(all_planes.size() / 2);
    for (size_t i = 0; i + 1 < all_planes.size(); i += 2) {
        cv::Mat f;
        cv::merge(std::vector<cv::Mat>{all_planes[i], all_planes[i + 1]}, f);
        filters.push_back(f);
    }
    return filters;
}

} // namespace ultratrack
