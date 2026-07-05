#include <ultratrack/tracking_session.hpp>
#include <ultratrack/detector/opencv_dnn_backend.hpp>
#include <ultratrack/tracking_engine/track_manager.hpp>
#include <ultratrack/ultratrack_sdk.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <memory>
#include <vector>

namespace ultratrack {

class TrackingSessionImpl {
public:
    TrackerSettings settings;
    std::unique_ptr<IDetectorBackend> detector;
    std::unique_ptr<TrackManager> track_manager;
    uint64_t primary_track_id = 0;
};

Result<std::unique_ptr<TrackingSession>> TrackingSession::create(const TrackerSettings& settings) {
    try {
        if (!UltraTrackerSDK::is_licensed()) {
            return Status(ErrorCode::LICENSE_INVALID, "SDK not licensed");
        }

        auto session = std::unique_ptr<TrackingSession>(new TrackingSession(settings));

        OpenCVDNNBackend::Config dcfg;
        dcfg.model_path = settings.detector_model_path;
        dcfg.confidence_threshold = settings.confidence_threshold;
        dcfg.nms_threshold = settings.nms_threshold;
        auto det = OpenCVDNNBackend::create(dcfg);
        if (!det.has_value()) {
            return det.error();
        }
        session->impl_->detector = std::move(det.value());

        TrackManager::Config tcfg;
        tcfg.kcf.feature_mode = settings.mode;
        auto tm = TrackManager::create(tcfg);
        if (!tm.has_value()) {
            return tm.error();
        }
        session->impl_->track_manager = std::move(tm.value());

        return session;
    } catch (const std::exception& e) {
        return Status(ErrorCode::INTERNAL_ERROR, std::string("TrackingSession::create failed: ") + e.what());
    }
}

TrackingSession::TrackingSession(const TrackerSettings& settings) : settings_(settings) {
    impl_ = std::make_unique<TrackingSessionImpl>();
    impl_->settings = settings;
}

TrackingSession::~TrackingSession() = default;

Result<TrackingOutput> TrackingSession::processFrame(const Frame& frame) {
    try {
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat bgr;
        if (frame.format == FrameFormat::BGR) {
            bgr = frame.data;
        } else if (frame.format == FrameFormat::RGB) {
            cv::cvtColor(frame.data, bgr, cv::COLOR_RGB2BGR);
        } else {
            return Status(ErrorCode::NOT_IMPLEMENTED, "frame format not supported");
        }

        auto detections = impl_->detector->detect(frame);
        if (!detections.has_value()) {
            return detections.error();
        }

        auto status = impl_->track_manager->update(detections.value(), bgr);
        if (!status.ok()) {
            return status;
        }

        TrackingOutput out;
        out.tracks = impl_->track_manager->activeTracks();
        out.primary_track_id = impl_->primary_track_id;
        bgr.copyTo(out.annotated_frame);
        for (const auto& t : out.tracks) {
            cv::Scalar color = (t.state == TrackState::CONFIRMED) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
            cv::rectangle(out.annotated_frame, t.bbox, color, 2);
        }

        auto end = std::chrono::high_resolution_clock::now();
        out.latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return out;
    } catch (const std::exception& e) {
        return Status(ErrorCode::INTERNAL_ERROR, std::string("processFrame failed: ") + e.what());
    }
}

Result<void> TrackingSession::selectPrimaryTarget(uint64_t track_id) {
    try {
        impl_->primary_track_id = track_id;
        return Status();
    } catch (const std::exception& e) {
        return Status(ErrorCode::INTERNAL_ERROR, std::string("selectPrimaryTarget failed: ") + e.what());
    }
}

Result<void> TrackingSession::reset() {
    try {
        impl_->track_manager->reset();
        impl_->primary_track_id = 0;
        return Status();
    } catch (const std::exception& e) {
        return Status(ErrorCode::INTERNAL_ERROR, std::string("reset failed: ") + e.what());
    }
}

const std::vector<Track>& TrackingSession::activeTracks() const {
    return impl_->track_manager->activeTracks();
}

} // namespace ultratrack
