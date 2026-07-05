#include <ultratrack/tracking_engine/track_lifecycle.hpp>

namespace ultratrack {

TrackLifecycle::TrackLifecycle(const LifecycleConfig& cfg) : cfg_(cfg) {}

void TrackLifecycle::onMatched(Track& track) {
    track.hits++;
    track.time_since_update = 0;
    if (track.state == TrackState::TENTATIVE && shouldConfirm(track)) {
        track.state = TrackState::CONFIRMED;
    }
}

void TrackLifecycle::onMissed(Track& track) {
    track.time_since_update++;
    if (track.state == TrackState::TENTATIVE && track.time_since_update > cfg_.max_tentative_age) {
        track.state = TrackState::LOST;
    } else if (track.state == TrackState::CONFIRMED && track.time_since_update > cfg_.max_age) {
        track.state = TrackState::LOST;
    }
}

bool TrackLifecycle::shouldRemove(const Track& track) const {
    return track.state == TrackState::LOST;
}

bool TrackLifecycle::shouldConfirm(const Track& track) const {
    return track.hits >= cfg_.confirmation_threshold;
}

} // namespace ultratrack
