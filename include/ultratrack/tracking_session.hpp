#pragma once
#include <ultratrack/core/result.hpp>
#include <ultratrack/core/types.hpp>
#include <ultratrack/detector/detector_backend.hpp>
#include <ultratrack/tracker/features/feature_extractor.hpp>
#include <memory>
#include <vector>

namespace ultratrack {

struct TrackerSettings {
    std::string detector_model_path;
    DetectorBackend backend = DetectorBackend::OPENCV_DNN;
    TrackingMode mode = TrackingMode::BALANCED;
    float confidence_threshold = 0.3f;
    float nms_threshold = 0.5f;
    std::string reid_model_path;
};

struct TrackingOutput {
    std::vector<Track> tracks;
    uint64_t primary_track_id = 0;
    cv::Mat annotated_frame;
    double latency_ms = 0.0;
};

class TrackingSession {
public:
    static Result<std::unique_ptr<TrackingSession>> create(const TrackerSettings& settings);

    Result<TrackingOutput> processFrame(const Frame& frame);
    Result<void> selectPrimaryTarget(uint64_t track_id);
    Result<void> reset();

    /// Returns the current set of active tracks.
    /// This function is noexcept. If the internal tracker throws, an empty
    /// vector is returned and the exception is swallowed.
    const std::vector<Track>& activeTracks() const noexcept;

    ~TrackingSession();

private:
    explicit TrackingSession(const TrackerSettings& settings);
    TrackerSettings settings_;
    std::unique_ptr<class TrackingSessionImpl> impl_;
};

} // namespace ultratrack
