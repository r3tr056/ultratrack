#include <catch2/catch_test_macros.hpp>
#include <ultratrack/tracking_engine/track_lifecycle.hpp>
#include <ultratrack/tracking_engine/track_manager.hpp>

using namespace ultratrack;

TEST_CASE("TrackManager creates tentative tracks from detections", "[tracking_engine]") {
    TrackManager::Config cfg;
    auto tm = TrackManager::create(cfg);
    REQUIRE(tm.has_value());

    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    std::vector<Detection> dets;
    dets.push_back({0, Rect2f(100, 100, 40, 40), 0.9f, 0, {}});

    auto status = tm.value()->update(dets, frame);
    REQUIRE(status.ok());
    REQUIRE(tm.value()->activeTracks().size() == 1);
    REQUIRE(tm.value()->activeTracks()[0].state == TrackState::TENTATIVE);
}

TEST_CASE("TrackManager confirms track after enough hits", "[tracking_engine]") {
    TrackManager::Config cfg;
    cfg.lifecycle.confirmation_threshold = 3;
    auto tm = TrackManager::create(cfg);
    REQUIRE(tm.has_value());

    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    for (int i = 0; i < 5; ++i) {
        std::vector<Detection> dets;
        dets.push_back({0, Rect2f(100.0f + i, 100.0f, 40, 40), 0.9f, 0, {}});
        tm.value()->update(dets, frame);
    }
    REQUIRE(tm.value()->activeTracks()[0].state == TrackState::CONFIRMED);
}

TEST_CASE("TrackManager removes tentative track after max tentative age misses", "[tracking_engine]") {
    TrackManager::Config cfg;
    cfg.lifecycle.max_tentative_age = 3;
    auto tm = TrackManager::create(cfg);
    REQUIRE(tm.has_value());

    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    // Create a tentative track.
    std::vector<Detection> dets;
    dets.push_back({0, Rect2f(100, 100, 40, 40), 0.9f, 0, {}});
    REQUIRE(tm.value()->update(dets, frame).ok());
    REQUIRE(tm.value()->activeTracks().size() == 1);

    // Miss for max_tentative_age + 1 frames; the lifecycle uses >, not >=.
    for (int i = 0; i < cfg.lifecycle.max_tentative_age + 1; ++i) {
        REQUIRE(tm.value()->update({}, frame).ok());
    }

    REQUIRE(tm.value()->activeTracks().empty());
}

TEST_CASE("TrackManager removes confirmed track after max age misses", "[tracking_engine]") {
    TrackManager::Config cfg;
    cfg.lifecycle.confirmation_threshold = 2;
    cfg.lifecycle.max_age = 3;
    auto tm = TrackManager::create(cfg);
    REQUIRE(tm.has_value());

    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    // Confirm a track.
    for (int i = 0; i < cfg.lifecycle.confirmation_threshold; ++i) {
        std::vector<Detection> dets;
        dets.push_back({0, Rect2f(100.0f + i, 100.0f, 40, 40), 0.9f, 0, {}});
        REQUIRE(tm.value()->update(dets, frame).ok());
    }
    REQUIRE(tm.value()->activeTracks().size() == 1);
    REQUIRE(tm.value()->activeTracks()[0].state == TrackState::CONFIRMED);

    // Miss for max_age + 1 frames; the lifecycle uses >, not >=.
    for (int i = 0; i < cfg.lifecycle.max_age + 1; ++i) {
        REQUIRE(tm.value()->update({}, frame).ok());
    }

    REQUIRE(tm.value()->activeTracks().empty());
}

TEST_CASE("TrackManager reset clears tracks and id counter", "[tracking_engine]") {
    TrackManager::Config cfg;
    auto tm = TrackManager::create(cfg);
    REQUIRE(tm.has_value());

    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    std::vector<Detection> dets;
    dets.push_back({0, Rect2f(100, 100, 40, 40), 0.9f, 0, {}});
    REQUIRE(tm.value()->update(dets, frame).ok());
    REQUIRE(tm.value()->activeTracks().size() == 1);
    REQUIRE(tm.value()->activeTracks()[0].id == 1);

    tm.value()->reset();
    REQUIRE(tm.value()->activeTracks().empty());

    // New tracks should reuse ids starting from 1.
    REQUIRE(tm.value()->update(dets, frame).ok());
    REQUIRE(tm.value()->activeTracks().size() == 1);
    REQUIRE(tm.value()->activeTracks()[0].id == 1);
}

TEST_CASE("TrackManager handles empty detection list", "[tracking_engine]") {
    TrackManager::Config cfg;
    auto tm = TrackManager::create(cfg);
    REQUIRE(tm.has_value());

    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    REQUIRE(tm.value()->update({}, frame).ok());
    REQUIRE(tm.value()->activeTracks().empty());
}

TEST_CASE("TrackManager skips tracks when KCF init fails", "[tracking_engine]") {
    TrackManager::Config cfg;
    auto tm = TrackManager::create(cfg);
    REQUIRE(tm.has_value());

    // Empty frame causes KCFTracker::init to fail, so no track should be created.
    cv::Mat empty_frame;
    std::vector<Detection> dets;
    dets.push_back({0, Rect2f(100, 100, 40, 40), 0.9f, 0, {}});

    auto status = tm.value()->update(dets, empty_frame);
    REQUIRE(status.ok());
    REQUIRE(tm.value()->activeTracks().empty());
}

TEST_CASE("TrackLifecycle confirms tentative track", "[tracking_engine]") {
    LifecycleConfig cfg;
    cfg.confirmation_threshold = 3;
    TrackLifecycle life(cfg);

    Track track;
    track.state = TrackState::TENTATIVE;
    track.hits = 0;

    life.onMatched(track);
    REQUIRE(track.state == TrackState::TENTATIVE);
    REQUIRE(track.hits == 1);
    REQUIRE(track.time_since_update == 0);

    life.onMatched(track);
    REQUIRE(track.state == TrackState::TENTATIVE);

    life.onMatched(track);
    REQUIRE(track.state == TrackState::CONFIRMED);
    REQUIRE(track.hits == 3);
}

TEST_CASE("TrackLifecycle marks tentative track lost after max tentative age", "[tracking_engine]") {
    LifecycleConfig cfg;
    cfg.max_tentative_age = 2;
    TrackLifecycle life(cfg);

    Track track;
    track.state = TrackState::TENTATIVE;
    track.time_since_update = 0;

    life.onMissed(track);
    REQUIRE(track.state == TrackState::TENTATIVE);
    REQUIRE(track.time_since_update == 1);

    life.onMissed(track);
    REQUIRE(track.state == TrackState::TENTATIVE);
    REQUIRE(track.time_since_update == 2);

    life.onMissed(track);
    REQUIRE(track.state == TrackState::LOST);
    REQUIRE(track.time_since_update == 3);
    REQUIRE(life.shouldRemove(track));
}

TEST_CASE("TrackLifecycle marks confirmed track lost after max age", "[tracking_engine]") {
    LifecycleConfig cfg;
    cfg.max_age = 3;
    TrackLifecycle life(cfg);

    Track track;
    track.state = TrackState::CONFIRMED;
    track.time_since_update = 0;

    for (int i = 0; i < cfg.max_age; ++i) {
        life.onMissed(track);
        REQUIRE(track.state == TrackState::CONFIRMED);
    }

    life.onMissed(track);
    REQUIRE(track.state == TrackState::LOST);
    REQUIRE(life.shouldRemove(track));
}

TEST_CASE("TrackManager predicts and updates track bboxes", "[tracking_engine]") {
    TrackManager::Config cfg;
    auto tm = TrackManager::create(cfg);
    REQUIRE(tm.has_value());

    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(frame, cv::Point(100, 100), cv::Point(140, 140), cv::Scalar(255, 255, 255), -1);

    std::vector<Detection> dets;
    dets.push_back({0, Rect2f(100, 100, 40, 40), 0.9f, 0, {}});
    REQUIRE(tm.value()->update(dets, frame).ok());
    REQUIRE(tm.value()->activeTracks().size() == 1);
    REQUIRE(tm.value()->activeTracks()[0].age == 1);

    // A second update with no detections exercises predictAll: the track ages and
    // is counted as missed once, without duplicating misses from update's logic.
    REQUIRE(tm.value()->update({}, frame).ok());

    const auto& tracks = tm.value()->activeTracks();
    REQUIRE(tracks.size() == 1);
    REQUIRE(tracks[0].age == 2);
    REQUIRE(tracks[0].time_since_update == 1);
    REQUIRE(tracks[0].bbox.width >= 0);
    REQUIRE(tracks[0].bbox.height >= 0);
}

TEST_CASE("TrackManager marks track missed when prediction fails", "[tracking_engine]") {
    TrackManager::Config cfg;
    auto tm = TrackManager::create(cfg);
    REQUIRE(tm.has_value());

    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(frame, cv::Point(100, 100), cv::Point(140, 140), cv::Scalar(255, 255, 255), -1);

    std::vector<Detection> dets;
    dets.push_back({0, Rect2f(100, 100, 40, 40), 0.9f, 0, {}});
    REQUIRE(tm.value()->update(dets, frame).ok());
    REQUIRE(tm.value()->activeTracks().size() == 1);
    REQUIRE(tm.value()->activeTracks()[0].time_since_update == 0);

    // Empty frame causes KCF predict to fail; the track should be counted as missed once.
    cv::Mat empty_frame;
    REQUIRE(tm.value()->update(dets, empty_frame).ok());
    REQUIRE(tm.value()->activeTracks().size() == 1);
    REQUIRE(tm.value()->activeTracks()[0].time_since_update == 1);
}

TEST_CASE("TrackManager does not overwrite bbox when KCF update fails", "[tracking_engine]") {
    TrackManager::Config cfg;
    auto tm = TrackManager::create(cfg);
    REQUIRE(tm.has_value());

    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(frame, cv::Point(100, 100), cv::Point(140, 140), cv::Scalar(255, 255, 255), -1);

    std::vector<Detection> dets;
    dets.push_back({0, Rect2f(100, 100, 40, 40), 0.9f, 0, {}});
    REQUIRE(tm.value()->update(dets, frame).ok());
    REQUIRE(tm.value()->activeTracks().size() == 1);
    Rect2f old_bbox = tm.value()->activeTracks()[0].bbox;

    // A frame type that triggers an OpenCV exception causes KCF update (and predict)
    // to fail. The detection still overlaps the previous bbox, so association produces
    // a match, but update rejects the patch and the track is treated as unmatched.
    cv::Mat bad_frame = cv::Mat::zeros(480, 640, CV_16SC3);
    std::vector<Detection> bad_dets;
    bad_dets.push_back({0, Rect2f(100, 100, 40, 40), 0.9f, 0, {}});
    REQUIRE(tm.value()->update(bad_dets, bad_frame).ok());

    const auto& tracks = tm.value()->activeTracks();
    REQUIRE(tracks.size() == 1);
    REQUIRE(tracks[0].bbox.x == old_bbox.x);
    REQUIRE(tracks[0].bbox.y == old_bbox.y);
    REQUIRE(tracks[0].bbox.width == old_bbox.width);
    REQUIRE(tracks[0].bbox.height == old_bbox.height);
    REQUIRE(tracks[0].time_since_update == 1);
}
