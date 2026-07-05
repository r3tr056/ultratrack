#include <ultratrack/tracking_engine/data_association.hpp>
#include <ultratrack/core/math.hpp>
#include <spdlog/spdlog.h>
#include <limits>
#include <algorithm>
#include <cmath>

namespace ultratrack {

DataAssociation::DataAssociation(const Config& cfg) : cfg_(cfg) {}

AssociationResult DataAssociation::associate(const std::vector<Track>& tracks,
                                              const std::vector<Detection>& detections) {
    AssociationResult result;
    try {
        if (tracks.empty() || detections.empty()) {
            for (size_t i = 0; i < tracks.size(); ++i) result.unmatched_tracks.push_back(i);
            for (size_t j = 0; j < detections.size(); ++j) result.unmatched_detections.push_back(j);
            return result;
        }

        cv::Mat cost(static_cast<int>(tracks.size()), static_cast<int>(detections.size()), CV_32F);
        for (size_t i = 0; i < tracks.size(); ++i) {
            for (size_t j = 0; j < detections.size(); ++j) {
                float iou = IoU(tracks[i].bbox, detections[j].bbox);
                cost.at<float>(static_cast<int>(i), static_cast<int>(j)) = 1.0f - iou;
            }
        }

        auto matches = hungarian(cost);
        std::vector<bool> track_used(tracks.size(), false);
        std::vector<bool> det_used(detections.size(), false);

        for (const auto& m : matches) {
            size_t ti = static_cast<size_t>(m.first);
            size_t di = static_cast<size_t>(m.second);
            if (IoU(tracks[ti].bbox, detections[di].bbox) >= cfg_.iou_threshold) {
                result.matches.emplace_back(ti, di);
                track_used[ti] = true;
                det_used[di] = true;
            }
        }

        for (size_t i = 0; i < tracks.size(); ++i) {
            if (!track_used[i]) result.unmatched_tracks.push_back(i);
        }
        for (size_t j = 0; j < detections.size(); ++j) {
            if (!det_used[j]) result.unmatched_detections.push_back(j);
        }
    } catch (const std::exception& e) {
        // Exception safety across the library boundary: fall back to returning
        // every input as unmatched so callers receive a valid, conservative result.
        spdlog::error("DataAssociation::associate failed: {}", e.what());
        result.matches.clear();
        result.unmatched_tracks.clear();
        result.unmatched_detections.clear();
        for (size_t i = 0; i < tracks.size(); ++i) result.unmatched_tracks.push_back(i);
        for (size_t j = 0; j < detections.size(); ++j) result.unmatched_detections.push_back(j);
    } catch (...) {
        spdlog::error("DataAssociation::associate failed: unknown error");
        result.matches.clear();
        result.unmatched_tracks.clear();
        result.unmatched_detections.clear();
        for (size_t i = 0; i < tracks.size(); ++i) result.unmatched_tracks.push_back(i);
        for (size_t j = 0; j < detections.size(); ++j) result.unmatched_detections.push_back(j);
    }

    return result;
}

std::vector<std::pair<int, int>> DataAssociation::hungarian(const cv::Mat& cost_matrix) {
    std::vector<std::pair<int, int>> assignments;
    if (cost_matrix.rows == 0 || cost_matrix.cols == 0) return assignments;

    int n = std::max(cost_matrix.rows, cost_matrix.cols);
    cv::Mat cost(n, n, CV_32F, cv::Scalar(1.0f));
    cost_matrix.copyTo(cost(cv::Rect(0, 0, cost_matrix.cols, cost_matrix.rows)));

    for (int row = 0; row < n; ++row) {
        float min_val = *std::min_element(cost.ptr<float>(row), cost.ptr<float>(row) + n);
        for (int col = 0; col < n; ++col) cost.at<float>(row, col) -= min_val;
    }
    for (int col = 0; col < n; ++col) {
        float min_val = std::numeric_limits<float>::max();
        for (int row = 0; row < n; ++row) min_val = std::min(min_val, cost.at<float>(row, col));
        for (int row = 0; row < n; ++row) cost.at<float>(row, col) -= min_val;
    }

    std::vector<int> assignment(n, -1);
    std::vector<bool> col_used(n, false);

    for (int row = 0; row < n; ++row) {
        int zero_col = -1, zero_count = 0;
        for (int col = 0; col < n; ++col) {
            if (!col_used[col] && std::abs(cost.at<float>(row, col)) < 1e-6f) {
                zero_col = col;
                zero_count++;
            }
        }
        if (zero_count == 1) {
            assignment[row] = zero_col;
            col_used[zero_col] = true;
        }
    }
    for (int row = 0; row < n; ++row) {
        if (assignment[row] != -1) continue;
        for (int col = 0; col < n; ++col) {
            if (!col_used[col] && std::abs(cost.at<float>(row, col)) < 1e-6f) {
                assignment[row] = col;
                col_used[col] = true;
                break;
            }
        }
    }
    for (int row = 0; row < n; ++row) {
        if (assignment[row] != -1) continue;
        float min_cost = std::numeric_limits<float>::max();
        int best_col = -1;
        for (int col = 0; col < n; ++col) {
            if (!col_used[col] && cost.at<float>(row, col) < min_cost) {
                min_cost = cost.at<float>(row, col);
                best_col = col;
            }
        }
        if (best_col != -1) {
            assignment[row] = best_col;
            col_used[best_col] = true;
        }
    }

    for (int row = 0; row < cost_matrix.rows; ++row) {
        if (assignment[row] != -1 && assignment[row] < cost_matrix.cols) {
            assignments.emplace_back(row, assignment[row]);
        }
    }
    return assignments;
}

} // namespace ultratrack
