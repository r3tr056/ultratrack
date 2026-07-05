#pragma once
#include <ultratrack/core/result.hpp>
#include <ultratrack/core/types.hpp>
#include <ultratrack/tracker/kcf_tracker.hpp>
#include <ultratrack/tracking_engine/track_lifecycle.hpp>
#include <vector>
#include <memory>

namespace ultratrack {

class TrackManager {
public:
    struct Config {
        LifecycleConfig lifecycle;
        KCFTracker::Config kcf;
    };

    static Result<std::unique_ptr<TrackManager>> create(const Config& cfg);
    Status update(const std::vector<Detection>& detections, const cv::Mat& frame);
    const std::vector<Track>& activeTracks() const;
    void reset();

private:
    explicit TrackManager(const Config& cfg);
    Config cfg_;
    std::vector<Track> tracks_;
    std::unique_ptr<KCFTracker> kcf_;
    uint64_t next_id_ = 1;

    void predictAll(const cv::Mat& frame);
    void createNewTracks(const std::vector<Detection>& unmatched, const cv::Mat& frame);
};

} // namespace ultratrack
