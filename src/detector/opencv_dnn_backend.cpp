#include <ultratrack/detector/opencv_dnn_backend.hpp>

namespace ultratrack {

Result<std::unique_ptr<IDetectorBackend>> OpenCVDNNBackend::create(const Config& cfg) {
    if (cfg.model_path.empty()) {
        return Status(ErrorCode::MODEL_LOAD_FAILED, "model path is empty");
    }
    if (cfg.input_size.width <= 0 || cfg.input_size.height <= 0) {
        return Status(ErrorCode::INVALID_ARGUMENT,
                      "input_size dimensions must be positive");
    }
    try {
        auto net = cv::dnn::readNetFromONNX(cfg.model_path);
        if (net.empty()) {
            return Status(ErrorCode::MODEL_LOAD_FAILED, "failed to load ONNX model");
        }

        if (cfg.use_cuda) {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } else {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

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

        // Parse YOLO output. Two common layouts exist:
        //   [1, dims, anchors]  (cols are anchor entries)
        //   [1, anchors, dims]  (rows are anchor entries)
        if (outputs[0].dims != 3 || outputs[0].size[0] != 1 ||
            outputs[0].type() != CV_32F) {
            return Status(ErrorCode::INFERENCE_FAILED,
                          "unexpected network output shape: expected 3-D CV_32F tensor "
                          "with batch size 1");
        }

        cv::Mat output = outputs[0];
        int dims = 0;
        int rows = 0;
        if (output.size[1] > output.size[2] && output.size[1] >= 5) {
            // Layout [1, dims, anchors]: second dimension holds the box/class vector.
            dims = output.size[1];
            rows = output.size[2];
        } else if (output.size[2] > output.size[1] && output.size[2] >= 5) {
            // Layout [1, anchors, dims]: transpose so dims is the inner dimension.
            cv::Mat transposed;
            const int perm[3] = {0, 2, 1};
            cv::transposeND(output, std::vector<int>(perm, perm + 3), transposed);
            output = transposed;
            dims = output.size[1];
            rows = output.size[2];
        } else {
            return Status(ErrorCode::INFERENCE_FAILED,
                          "unrecognized network output shape: [" +
                              std::to_string(output.size[0]) + ", " +
                              std::to_string(output.size[1]) + ", " +
                              std::to_string(output.size[2]) + "]");
        }

        const float* data = output.ptr<float>();
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

            int cls = 0;
            double max_score = row[5];
            for (int k = 1; k < dims - 5; ++k) {
                if (row[5 + k] > max_score) {
                    max_score = row[5 + k];
                    cls = k;
                }
            }
            if (max_score < cfg_.confidence_threshold) continue;

            float cx = row[0], cy = row[1], w = row[2], h = row[3];
            if (w <= 0 || h <= 0) continue;

            cv::Rect box(static_cast<int>((cx - w / 2) * xf),
                         static_cast<int>((cy - h / 2) * yf),
                         static_cast<int>(w * xf),
                         static_cast<int>(h * yf));
            boxes.push_back(box);
            confidences.push_back(obj_conf);
            class_ids.push_back(cls);
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
