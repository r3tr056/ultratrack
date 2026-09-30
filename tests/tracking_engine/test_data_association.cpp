#include <catch2/catch_test_macros.hpp>
#include <ultratrack/tracking_engine/data_association.hpp>

using namespace ultratrack;

TEST_CASE("DataAssociation matches overlapping tracks", "[tracking_engine]") {
    DataAssociation::Config cfg;
    DataAssociation assoc(cfg);

    std::vector<Track> tracks;
    Track t; t.id = 1; t.bbox = Rect2f(100, 100, 40, 40);
    tracks.push_back(t);

    std::vector<Detection> dets;
    dets.push_back({0, Rect2f(102, 101, 40, 40), 0.9f, 0, {}});

    auto result = assoc.associate(tracks, dets);
    REQUIRE(result.matches.size() == 1);
    REQUIRE(result.matches[0].first == 0);
    REQUIRE(result.matches[0].second == 0);
    REQUIRE(result.unmatched_tracks.empty());
    REQUIRE(result.unmatched_detections.empty());
}

TEST_CASE("DataAssociation leaves distant detections unmatched", "[tracking_engine]") {
    DataAssociation::Config cfg;
    DataAssociation assoc(cfg);

    std::vector<Track> tracks;
    Track t; t.id = 1; t.bbox = Rect2f(100, 100, 40, 40);
    tracks.push_back(t);

    std::vector<Detection> dets;
    dets.push_back({0, Rect2f(400, 400, 40, 40), 0.9f, 0, {}});

    auto result = assoc.associate(tracks, dets);
    REQUIRE(result.matches.empty());
    REQUIRE(result.unmatched_tracks.size() == 1);
    REQUIRE(result.unmatched_detections.size() == 1);
}

TEST_CASE("DataAssociation handles empty tracks", "[tracking_engine]") {
    DataAssociation::Config cfg;
    DataAssociation assoc(cfg);

    std::vector<Track> tracks;
    std::vector<Detection> dets;
    dets.push_back({0, Rect2f(100, 100, 40, 40), 0.9f, 0, {}});

    auto result = assoc.associate(tracks, dets);
    REQUIRE(result.matches.empty());
    REQUIRE(result.unmatched_tracks.empty());
    REQUIRE(result.unmatched_detections.size() == 1);
}

TEST_CASE("DataAssociation handles empty detections", "[tracking_engine]") {
    DataAssociation::Config cfg;
    DataAssociation assoc(cfg);

    std::vector<Track> tracks;
    Track t; t.id = 1; t.bbox = Rect2f(100, 100, 40, 40);
    tracks.push_back(t);
    std::vector<Detection> dets;

    auto result = assoc.associate(tracks, dets);
    REQUIRE(result.matches.empty());
    REQUIRE(result.unmatched_tracks.size() == 1);
    REQUIRE(result.unmatched_detections.empty());
}

TEST_CASE("DataAssociation handles both empty inputs", "[tracking_engine]") {
    DataAssociation::Config cfg;
    DataAssociation assoc(cfg);

    auto result = assoc.associate({}, {});
    REQUIRE(result.matches.empty());
    REQUIRE(result.unmatched_tracks.empty());
    REQUIRE(result.unmatched_detections.empty());
}

TEST_CASE("DataAssociation resolves multiple competing detections by cost", "[tracking_engine]") {
    DataAssociation::Config cfg;
    cfg.iou_threshold = 0.1f;
    DataAssociation assoc(cfg);

    std::vector<Track> tracks;
    Track t; t.id = 1; t.bbox = Rect2f(100, 100, 40, 40);
    tracks.push_back(t);

    std::vector<Detection> dets;
    // Detection 0 overlaps well.
    dets.push_back({0, Rect2f(102, 101, 40, 40), 0.9f, 0, {}});
    // Detection 1 overlaps a little; should be unmatched because detection 0 wins.
    dets.push_back({0, Rect2f(145, 145, 40, 40), 0.8f, 0, {}});

    auto result = assoc.associate(tracks, dets);
    REQUIRE(result.matches.size() == 1);
    REQUIRE(result.matches[0].first == 0);
    REQUIRE(result.matches[0].second == 0);
    REQUIRE(result.unmatched_tracks.empty());
    REQUIRE(result.unmatched_detections.size() == 1);
    REQUIRE(result.unmatched_detections[0] == 1);
}

TEST_CASE("DataAssociation resolves multiple competing tracks by cost", "[tracking_engine]") {
    DataAssociation::Config cfg;
    cfg.iou_threshold = 0.1f;
    DataAssociation assoc(cfg);

    std::vector<Track> tracks;
    Track t0; t0.id = 1; t0.bbox = Rect2f(100, 100, 40, 40);
    Track t1; t1.id = 2; t1.bbox = Rect2f(300, 300, 40, 40);
    tracks.push_back(t0);
    tracks.push_back(t1);

    std::vector<Detection> dets;
    // One detection closer to track 0.
    dets.push_back({0, Rect2f(102, 101, 40, 40), 0.9f, 0, {}});

    auto result = assoc.associate(tracks, dets);
    REQUIRE(result.matches.size() == 1);
    REQUIRE(result.matches[0].first == 0);
    REQUIRE(result.matches[0].second == 0);
    REQUIRE(result.unmatched_tracks.size() == 1);
    REQUIRE(result.unmatched_tracks[0] == 1);
    REQUIRE(result.unmatched_detections.empty());
}

TEST_CASE("DataAssociation prefers optimal assignment in 2x2 case", "[tracking_engine]") {
    DataAssociation::Config cfg;
    cfg.iou_threshold = 0.0f;
    DataAssociation assoc(cfg);

    std::vector<Track> tracks;
    Track t0; t0.id = 1; t0.bbox = Rect2f(100, 100, 40, 40);
    Track t1; t1.id = 2; t1.bbox = Rect2f(300, 300, 40, 40);
    tracks.push_back(t0);
    tracks.push_back(t1);

    std::vector<Detection> dets;
    // The best pairing overall is t0-d0 and t1-d1 even though a greedy
    // row-by-row approach might pair both detections with t0.
    dets.push_back({0, Rect2f(101, 101, 40, 40), 0.9f, 0, {}});
    dets.push_back({0, Rect2f(301, 301, 40, 40), 0.9f, 0, {}});

    auto result = assoc.associate(tracks, dets);
    REQUIRE(result.matches.size() == 2);
    std::vector<std::pair<size_t, size_t>> expected = {{0, 0}, {1, 1}};
    REQUIRE(result.matches == expected);
    REQUIRE(result.unmatched_tracks.empty());
    REQUIRE(result.unmatched_detections.empty());
}

TEST_CASE("DataAssociation supports rectangular cost matrices", "[tracking_engine]") {
    DataAssociation::Config cfg;
    cfg.iou_threshold = 0.0f;
    DataAssociation assoc(cfg);

    std::vector<Track> tracks;
    for (int i = 0; i < 3; ++i) {
        Track t;
        t.id = i + 1;
        t.bbox = Rect2f(100.0f * i, 100.0f * i, 40, 40);
        tracks.push_back(t);
    }

    std::vector<Detection> dets;
    dets.push_back({0, Rect2f(0, 0, 40, 40), 0.9f, 0, {}});
    dets.push_back({0, Rect2f(200, 200, 40, 40), 0.9f, 0, {}});

    auto result = assoc.associate(tracks, dets);
    REQUIRE(result.matches.size() == 2);
    REQUIRE(result.unmatched_tracks.size() == 1);
    REQUIRE(result.unmatched_detections.empty());
}
