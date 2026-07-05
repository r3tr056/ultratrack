#pragma once
#include <ultratrack/core/types.hpp>
#include <vector>
#include <utility>

namespace ultratrack {

struct AssociationResult {
    std::vector<std::pair<size_t, size_t>> matches; // track idx -> detection idx
    std::vector<size_t> unmatched_tracks;
    std::vector<size_t> unmatched_detections;
};

class DataAssociation {
public:
    struct Config {
        float iou_threshold = 0.3f;
        float reid_weight = 0.0f; // Phase 1: IoU only
        float motion_weight = 0.0f;
    };

    explicit DataAssociation(const Config& cfg);
    AssociationResult associate(const std::vector<Track>& tracks,
                                 const std::vector<Detection>& detections);

private:
    Config cfg_;
    std::vector<std::pair<int, int>> hungarian(const cv::Mat& cost_matrix);
};

} // namespace ultratrack
