#include <ultratrack/tracking_engine/track_manager.hpp>
#include <ultratrack/tracking_engine/data_association.hpp>
#include <ultratrack/core/math.hpp>
#include <algorithm>

namespace ultratrack {

Result<std::unique_ptr<TrackManager>> TrackManager::create(const Config& cfg) {
    try {
        auto kcf = KCFTracker::create(cfg.kcf);
        if (!kcf.has_value()) {
            return kcf.error();
        }
        auto tm = std::unique_ptr<TrackManager>(new TrackManager(cfg));
        tm->kcf_ = std::move(kcf.value());
        return tm;
    } catch (const std::exception& e) {
        return Status(ErrorCode::INTERNAL_ERROR,
                      std::string("failed to create TrackManager: ") + e.what());
    } catch (...) {
        return Status(ErrorCode::INTERNAL_ERROR, "failed to create TrackManager: unknown error");
    }
}

TrackManager::TrackManager(const Config& cfg) : cfg_(cfg) {}

Status TrackManager::update(const std::vector<Detection>& detections, const cv::Mat& frame) {
    try {
        predictAll(frame);

        DataAssociation::Config assoc_cfg;
        assoc_cfg.iou_threshold = 0.3f;
        DataAssociation association(assoc_cfg);
        auto assoc_result = association.associate(tracks_, detections);

        std::vector<bool> det_matched(detections.size(), false);
        TrackLifecycle life(cfg_.lifecycle);

        for (const auto& match : assoc_result.matches) {
            size_t i = match.first;
            size_t j = match.second;
            life.onMatched(tracks_[i]);
            tracks_[i].bbox = detections[j].bbox;
            tracks_[i].confidence = detections[j].confidence;
            kcf_->update(tracks_[i], frame, detections[j].bbox);
            det_matched[j] = true;
        }

        // Mark missed tracks
        for (size_t i = 0; i < tracks_.size(); ++i) {
            bool matched = false;
            for (const auto& m : assoc_result.matches) {
                if (m.first == i) {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                life.onMissed(tracks_[i]);
            }
        }

        // Remove lost tracks
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
            [&life](const Track& t) { return life.shouldRemove(t); }), tracks_.end());

        // Create new tracks from unmatched detections
        std::vector<Detection> unmatched;
        for (size_t j = 0; j < detections.size(); ++j) {
            if (!det_matched[j]) unmatched.push_back(detections[j]);
        }
        createNewTracks(unmatched, frame);

        return Status();
    } catch (const std::exception& e) {
        return Status(ErrorCode::INTERNAL_ERROR, std::string("TrackManager::update failed: ") + e.what());
    } catch (...) {
        return Status(ErrorCode::INTERNAL_ERROR, "TrackManager::update failed: unknown error");
    }
}

void TrackManager::predictAll(const cv::Mat& frame) {
    (void)frame;
    for (auto& track : tracks_) {
        track.age++;
        // Note: time_since_update is managed by TrackLifecycle::onMatched/onMissed
        // so it is not incremented here. This avoids double-counting missed frames.
        // KCF prediction is intentionally skipped in this skeleton; bbox refinement
        // will be integrated once the data association flow is complete (Task 7).
    }
}

void TrackManager::createNewTracks(const std::vector<Detection>& unmatched, const cv::Mat& frame) {
    for (const auto& det : unmatched) {
        if (det.confidence < 0.5f) continue;
        Track track;
        track.id = next_id_++;
        track.bbox = det.bbox;
        track.confidence = det.confidence;
        track.state = TrackState::TENTATIVE;
        track.age = 1;
        track.hits = 1;
        track.time_since_update = 0;
        kcf_->init(track, frame);
        tracks_.push_back(track);
    }
}

const std::vector<Track>& TrackManager::activeTracks() const { return tracks_; }
void TrackManager::reset() { tracks_.clear(); next_id_ = 1; }

} // namespace ultratrack
