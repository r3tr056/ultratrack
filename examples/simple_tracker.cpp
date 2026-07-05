// examples/simple_tracker.cpp
#include <ultratrack/ultratrack_sdk.hpp>
#include <ultratrack/tracking_session.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace ultratrack;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: simple_tracker <model.onnx> [video.mp4]\n";
        return 1;
    }

    SDKConfig sdk_cfg;
    sdk_cfg.license_key = "TRIAL";
    auto s = UltraTrackerSDK::initialize(sdk_cfg);
    if (!s.ok()) {
        std::cerr << "SDK init failed: " << s.message() << "\n";
        return 1;
    }

    TrackerSettings ts;
    ts.detector_model_path = argv[1];
    ts.backend = DetectorBackend::OPENCV_DNN;
    ts.mode = TrackingMode::BALANCED;

    auto session = TrackingSession::create(ts);
    if (!session.has_value()) {
        std::cerr << "Session create failed: " << session.error().message() << "\n";
        return 1;
    }

    cv::VideoCapture cap(argc > 2 ? argv[2] : "0");
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video source\n";
        return 1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        Frame f;
        f.width = frame.cols;
        f.height = frame.rows;
        f.format = FrameFormat::BGR;
        f.data = frame;
        auto out = session.value()->processFrame(f);
        if (out.has_value()) {
            cv::imshow("UltraTrack", out.value().annotated_frame);
            if (cv::waitKey(1) == 27) break;
        }
    }

    UltraTrackerSDK::shutdown();
    return 0;
}
