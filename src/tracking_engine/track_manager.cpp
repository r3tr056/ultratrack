#include <ultratrack/tracking_engine/track_manager.hpp>
#include <ultratrack/tracking_engine/data_association.hpp>
#include <ultratrack/core/math.hpp>
#include <spdlog/spdlog.h>
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
        auto predicted_missed = predictAll(frame);

        DataAssociation::Config assoc_cfg;
        assoc_cfg.iou_threshold = 0.3f;
        DataAssociation association(assoc_cfg);
        auto assoc_result = association.associate(tracks_, detections);

        std::vector<bool> track_matched(tracks_.size(), false);
        std::vector<bool> det_matched(detections.size(), false);
        TrackLifecycle life(cfg_.lifecycle);

        for (const auto& match : assoc_result.matches) {
            size_t i = match.first;
            size_t j = match.second;
            auto status = kcf_->update(tracks_[i], frame, detections[j].bbox);
            if (!status.ok()) {
                spdlog::warn("KCF update failed for track {}: {}", tracks_[i].id, status.message());
                // Treat the track as unmatched: leave bbox/confidence unchanged and
                // let the post-association loop count this frame as a miss.
                track_matched[i] = false;
                det_matched[j] = true;
                continue;
            }
            life.onMatched(tracks_[i]);
            tracks_[i].bbox = detections[j].bbox;
            tracks_[i].confidence = detections[j].confidence;
            track_matched[i] = true;
            det_matched[j] = true;
        }

        // Mark missed tracks, skipping those already handled by a failed prediction.
        for (size_t i = 0; i < tracks_.size(); ++i) {
            if (!track_matched[i] && !predicted_missed[i]) {
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

std::vector<bool> TrackManager::predictAll(const cv::Mat& frame) {
    std::vector<bool> predicted_missed(tracks_.size(), false);
    TrackLifecycle life(cfg_.lifecycle);
    for (size_t i = 0; i < tracks_.size(); ++i) {
        auto& track = tracks_[i];
        track.age++;

        auto result = kcf_->predict(track, frame);
        if (!result.has_value()) {
            spdlog::warn("KCF predict failed for track {}: {}", track.id, result.error().message());
            life.onMissed(track);
            predicted_missed[i] = true;
            continue;
        }

        Rect2f predicted_bbox = result.value();
        if (predicted_bbox.area() <= 0.0f) {
            life.onMissed(track);
            predicted_missed[i] = true;
            continue;
        }

        track.bbox = predicted_bbox;
    }
    return predicted_missed;
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
        auto status = kcf_->init(track, frame);
        if (!status.ok()) {
            // If the KCF tracker cannot initialize on this patch, do not create a track.
            continue;
        }
        tracks_.push_back(track);
    }
}

const std::vector<Track>& TrackManager::activeTracks() const { return tracks_; }
void TrackManager::reset() { tracks_.clear(); next_id_ = 1; }

} // namespace ultratrack
