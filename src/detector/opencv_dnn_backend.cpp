#include <ultratrack/detector/opencv_dnn_backend.hpp>
#include <spdlog/spdlog.h>

namespace ultratrack {

Result<std::unique_ptr<IDetectorBackend>> OpenCVDNNBackend::create(const Config& cfg) {
    if (cfg.model_path.empty()) {
        return Status(ErrorCode::MODEL_LOAD_FAILED, "model path is empty");
    }
    try {
        auto net = cv::dnn::readNetFromONNX(cfg.model_path);
        if (net.empty()) {
            return Status(ErrorCode::MODEL_LOAD_FAILED, "failed to load ONNX model");
        }

        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        return std::unique_ptr<IDetectorBackend>(new OpenCVDNNBackend(std::move(net), cfg));
    } catch (const std::exception& e) {
        return Status(ErrorCode::MODEL_LOAD_FAILED, e.what());
    }
}

OpenCVDNNBackend::OpenCVDNNBackend(cv::dnn::Net net, const Config& cfg)
    : net_(std::move(net)), cfg_(cfg) {}

Result<std::vector<Detection>> OpenCVDNNBackend::detect(const Frame& frame) {
    if (frame.data.empty()) {
        return Status(ErrorCode::EMPTY_FRAME, "input frame is empty");
    }
    try {
        cv::Mat blob;
        cv::dnn::blobFromImage(frame.data, blob, 1.0 / 255.0, cfg_.input_size,
                               cv::Scalar(), true, false);
        net_.setInput(blob);
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());
        if (outputs.empty()) {
            return std::vector<Detection>{};
        }

        // Parse YOLO output [1, dims, anchors] -> first mat
        const float* data = reinterpret_cast<float*>(outputs[0].data);
        const int dims = outputs[0].size[1];
        const int rows = outputs[0].size[2];
        if (dims < 5) {
            return Status(ErrorCode::INFERENCE_FAILED,
                          "invalid network output dimensions: " + std::to_string(dims));
        }

        const float xf = frame.width / static_cast<float>(cfg_.input_size.width);
        const float yf = frame.height / static_cast<float>(cfg_.input_size.height);

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        for (int i = 0; i < rows; ++i) {
            const float* row = data + i * dims;
            float obj_conf = row[4];
            if (obj_conf < cfg_.confidence_threshold) continue;

            cv::Mat scores(1, dims - 5, CV_32FC1, const_cast<float*>(row + 5));
            cv::Point cls;
            double max_score;
            cv::minMaxLoc(scores, nullptr, &max_score, nullptr, &cls);
            if (max_score < cfg_.confidence_threshold) continue;

            float cx = row[0], cy = row[1], w = row[2], h = row[3];
            if (w <= 0 || h <= 0) continue;

            cv::Rect box(static_cast<int>((cx - w / 2) * xf),
                         static_cast<int>((cy - h / 2) * yf),
                         static_cast<int>(w * xf),
                         static_cast<int>(h * yf));
            boxes.push_back(box);
            confidences.push_back(obj_conf);
            class_ids.push_back(cls.x);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, cfg_.confidence_threshold,
                          cfg_.nms_threshold, indices);

        std::vector<Detection> detections;
        for (int idx : indices) {
            Detection d;
            d.bbox = Rect2f(boxes[idx]);
            d.confidence = confidences[idx];
            d.class_id = class_ids[idx];
            detections.push_back(d);
        }
        return detections;
    } catch (const std::exception& e) {
        return Status(ErrorCode::INFERENCE_FAILED, e.what());
    }
}

} // namespace ultratrack
