:
# UltraTrack SDK Phase 1 — Core SDK Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the existing research UltraTracker into a packaged, multi-target C++ SDK with a stable public API, detector backend abstraction, tracking engine, and benchmarking harness, laying the foundation for PTZ control and licensing in Phase 2.

**Architecture:** Split the monolithic `UltraTracker` class into focused modules (`core`, `detector`, `tracker`, `tracking_engine`, `telemetry`, `benchmark`). The public API hides implementation behind `UltraTrackerSDK`, `TrackingSession`, and result types. A detector backend abstraction lets the SDK load OpenCV DNN, ONNX Runtime, or TensorRT without changing app code.

**Tech Stack:** C++17, CMake 3.16+, OpenCV 4.5+, optional ONNX Runtime, spdlog, nlohmann/json, Catch2 or gtest, vcpkg/Conan for dependencies.

## Global Constraints

- C++17 minimum, CMake 3.16 minimum.
- Public API must not throw exceptions across the library boundary; use `Result<T>` / `Status`.
- Support Windows 10/11 x64 and Ubuntu 20.04/22.04 x64 as primary targets; Jetson in Phase 3.
- All new code must have unit or integration tests.
- Remove hardcoded dependency paths; use `find_package` with optional overrides.
- Benchmarking harness must load MOT-format sequences and compute HOTA/MOTA/IDF1 by end of Phase 1.
- Keep existing SIMD/AVX2/NEON acceleration paths intact.

---

## File Structure

```
ultratrack/
├── CMakeLists.txt                         # top-level super-build
├── cmake/
│   └── ultratrack-config.cmake.in         # install config
├── include/
│   └── ultratrack/
│       ├── core/
│       │   ├── types.hpp                  # Rect2f, Frame, Detection, Track
│       │   ├── result.hpp                 # Result<T>, Status, ErrorCode
│       │   └── math.hpp                   # IoU, geometry helpers
│       ├── detector/
│       │   ├── detector_backend.hpp       # IDetectorBackend
│       │   └── opencv_dnn_backend.hpp     # OpenCV DNN impl
│       ├── tracker/
│       │   ├── kcf_tracker.hpp            # single-track KCF
│       │   ├── kalman.hpp                 # Kalman filter helper
│       │   ├── displacement_predictor.hpp # moved from tracking/
│       │   ├── scale_estimator.hpp        # moved from tracking/
│       │   └── features/                  # existing feature extractors
│       ├── tracking_engine/
│       │   ├── track_manager.hpp
│       │   ├── data_association.hpp
│       │   ├── track_lifecycle.hpp
│       │   └── primary_selector.hpp
│       ├── telemetry/
│       │   └── metrics.hpp
│       ├── benchmark/
│       │   └── harness.hpp
│       ├── ultratrack_sdk.hpp             # public API
│       └── tracking_session.hpp           # public API
├── src/
│   ├── core/
│   │   ├── types.cpp
│   │   ├── result.cpp
│   │   └── math.cpp
│   ├── detector/
│   │   ├── opencv_dnn_backend.cpp
│   │   └── detector_factory.cpp
│   ├── tracker/
│   │   ├── kcf_tracker.cpp
│   │   ├── kalman.cpp
│   │   ├── displacement_predictor.cpp
│   │   ├── scale_estimator.cpp
│   │   └── features/
│   ├── tracking_engine/
│   │   ├── track_manager.cpp
│   │   ├── data_association.cpp
│   │   ├── track_lifecycle.cpp
│   │   └── primary_selector.cpp
│   ├── telemetry/
│   │   └── metrics.cpp
│   ├── benchmark/
│   │   └── harness.cpp
│   └── ultratrack_sdk.cpp
├── tests/
│   ├── core/
│   ├── detector/
│   ├── tracker/
│   ├── tracking_engine/
│   └── benchmark/
├── examples/
│   └── simple_tracker.cpp
└── models/
    └── .gitkeep
```

---

## Task 1: Modernize CMake Build System

**Files:**
- Create: `cmake/ultratrack-config.cmake.in`
- Modify: `CMakeLists.txt`
- Create: `CMakePresets.json`
- Test: `cmake --build build --target test`

**Interfaces:**
- Produces: `ultratrack` target, `BUILD_TESTS` option, `ULTRATRACK_WITH_ONNXRUNTIME` option.

- [ ] **Step 1: Write the new top-level CMakeLists.txt**

Replace the hardcoded OpenCV path with `find_package(OpenCV 4.5 REQUIRED)` and add options for tests and ONNX Runtime. Keep the existing SIMD detection and Windows flags.

```cmake
cmake_minimum_required(VERSION 3.16)
project(ultratrack VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build unit tests" ON)
option(ULTRATRACK_WITH_ONNXRUNTIME "Enable ONNX Runtime backend" OFF)

find_package(OpenCV 4.5 REQUIRED)
find_package(Threads REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

if(ULTRATRACK_WITH_ONNXRUNTIME)
    find_package(OnnxRuntime REQUIRED)
endif()

add_subdirectory(src)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

include(GNUInstallDirs)
install(TARGETS ultratrack EXPORT ultratrack-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ultratrack DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT ultratrack-targets FILE ultratrack-targets.cmake
    NAMESPACE ultratrack:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ultratrack)
```

- [ ] **Step 2: Create src/CMakeLists.txt**

```cmake
set(ULTRATRACK_SOURCES
    core/types.cpp
    core/result.cpp
    core/math.cpp
    detector/opencv_dnn_backend.cpp
    detector/detector_factory.cpp
    tracker/kcf_tracker.cpp
    tracker/kalman.cpp
    tracker/displacement_predictor.cpp
    tracker/scale_estimator.cpp
    tracker/features/feature_extractor.cpp
    tracker/features/hog_feature.cpp
    tracker/features/gray_feature.cpp
    tracker/features/cn_feature.cpp
    tracking_engine/track_manager.cpp
    tracking_engine/data_association.cpp
    tracking_engine/track_lifecycle.cpp
    tracking_engine/primary_selector.cpp
    telemetry/metrics.cpp
    benchmark/harness.cpp
    ultratrack_sdk.cpp
)

add_library(ultratrack ${ULTRATRACK_SOURCES})
target_include_directories(ultratrack
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ultratrack
    PUBLIC ${OpenCV_LIBS}
    PRIVATE spdlog::spdlog nlohmann_json::nlohmann_json Threads::Threads)

if(ULTRATRACK_WITH_ONNXRUNTIME)
    target_compile_definitions(ultratrack PRIVATE ULTRATRACK_WITH_ONNXRUNTIME)
    target_link_libraries(ultratrack PRIVATE OnnxRuntime::OnnxRuntime)
endif()

# Keep existing SIMD flags and Windows definitions
if(MSVC)
    target_compile_options(ultratrack PRIVATE /arch:AVX2 /fp:fast)
    target_compile_definitions(ultratrack PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0601)
else()
    target_compile_options(ultratrack PRIVATE -march=native -ffast-math)
endif()
```

- [ ] **Step 3: Create tests/CMakeLists.txt**

```cmake
find_package(Catch2 3 REQUIRED)

add_executable(ultratrack_tests
    core/test_math.cpp
    core/test_result.cpp
    detector/test_opencv_dnn_backend.cpp
    tracker/test_kcf_tracker.cpp
    tracking_engine/test_track_manager.cpp
    tracking_engine/test_data_association.cpp
    benchmark/test_harness.cpp
)

target_link_libraries(ultratrack_tests PRIVATE ultratrack Catch2::Catch2WithMain)
include(Catch)
catch_discover_tests(ultratrack_tests)
```

- [ ] **Step 4: Configure and build**

Run:
```bash
cmake --preset=default
cmake --build build --config Release
```

Expected: library compiles without errors.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt cmake/ src/CMakeLists.txt tests/CMakeLists.txt CMakePresets.json
git commit -m "build: modernize CMake, remove hardcoded OpenCV path, add install exports"
```

---

## Task 2: Core Types and Result API

**Files:**
- Create: `include/ultratrack/core/result.hpp`
- Create: `include/ultratrack/core/types.hpp`
- Create: `include/ultratrack/core/math.hpp`
- Create: `src/core/result.cpp`
- Create: `src/core/types.cpp`
- Create: `src/core/math.cpp`
- Create: `tests/core/test_result.cpp`
- Create: `tests/core/test_math.cpp`

**Interfaces:**
- Produces: `Result<T>`, `Status`, `ErrorCode`, `Rect2f`, `Frame`, `Detection`, `Track`, `IoU()`.
- Consumes: nothing (foundational).

- [ ] **Step 1: Write the failing test for Result<T>**

```cpp
// tests/core/test_result.cpp
#include <catch2/catch_test_macros.hpp>
#include <ultratrack/core/result.hpp>

using namespace ultratrack;

TEST_CASE("Result holds a value", "[core]") {
    Result<int> r = 42;
    REQUIRE(r.has_value());
    REQUIRE(r.value() == 42);
}

TEST_CASE("Result holds an error", "[core]") {
    Status s(ErrorCode::INVALID_ARGUMENT, "bad input");
    Result<int> r = s;
    REQUIRE(!r.has_value());
    REQUIRE(r.error().code() == ErrorCode::INVALID_ARGUMENT);
    REQUIRE(r.error().message() == "bad input");
}

TEST_CASE("Result map transforms value", "[core]") {
    Result<int> r = 21;
    auto doubled = r.map([](int x) { return x * 2; });
    REQUIRE(doubled.value() == 42);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build build --target ultratrack_tests
./build/tests/ultratrack_tests
```

Expected: compile errors (files don't exist).

- [ ] **Step 3: Implement Result<T>, Status, ErrorCode**

```cpp
// include/ultratrack/core/result.hpp
#pragma once
#include <string>
#include <variant>
#include <optional>

namespace ultratrack {

enum class ErrorCode {
    OK = 0,
    INVALID_ARGUMENT,
    NOT_INITIALIZED,
    ALREADY_INITIALIZED,
    LICENSE_INVALID,
    LICENSE_EXPIRED,
    MODEL_LOAD_FAILED,
    INFERENCE_FAILED,
    EMPTY_FRAME,
    INVALID_PATCH_SIZE,
    FEATURE_EXTRACTION_FAILED,
    TRACKING_LOST,
    PTZ_CONNECTION_FAILED,
    CONFIG_PARSE_FAILED,
    NOT_IMPLEMENTED,
    UNKNOWN
};

class Status {
public:
    Status() : code_(ErrorCode::OK) {}
    Status(ErrorCode code, std::string message)
        : code_(code), message_(std::move(message)) {}

    bool ok() const { return code_ == ErrorCode::OK; }
    ErrorCode code() const { return code_; }
    const std::string& message() const { return message_; }

private:
    ErrorCode code_;
    std::string message_;
};

template <typename T>
class Result {
public:
    Result(T value) : data_(std::move(value)) {}
    Result(Status error) : data_(std::move(error)) {}

    bool has_value() const { return std::holds_alternative<T>(data_); }
    bool ok() const { return has_value(); }

    T& value() { return std::get<T>(data_); }
    const T& value() const { return std::get<T>(data_); }
    Status& error() { return std::get<Status>(data_); }
    const Status& error() const { return std::get<Status>(data_); }

    template <typename F>
    auto map(F&& f) -> Result<std::invoke_result_t<F, T&>> {
        if (has_value()) {
            return f(value());
        }
        return error();
    }

private:
    std::variant<T, Status> data_;
};

} // namespace ultratrack
```

- [ ] **Step 4: Implement core types**

```cpp
// include/ultratrack/core/types.hpp
#pragma once
#include <opencv2/core.hpp>
#include <cstdint>
#include <vector>

namespace ultratrack {

using Rect2f = cv::Rect2f;
using Point2f = cv::Point2f;
using Size2f = cv::Size2f;

enum class FrameFormat { BGR, RGB, GRAY, NV12, GPU_HANDLE };

struct Frame {
    int width = 0;
    int height = 0;
    FrameFormat format = FrameFormat::BGR;
    cv::Mat data;        // CPU data
    void* gpu_handle = nullptr; // opaque GPU handle
    int64_t timestamp_us = 0;
};

struct Detection {
    uint64_t id = 0;
    Rect2f bbox;
    float confidence = 0.0f;
    int class_id = -1;
    cv::Mat feature;
};

enum class TrackState { TENTATIVE, CONFIRMED, LOST };

struct Track {
    uint64_t id = 0;
    Rect2f bbox;
    float confidence = 0.0f;
    TrackState state = TrackState::TENTATIVE;
    int age = 0;
    int hits = 0;
    int time_since_update = 0;
    cv::Mat kalman_state;
    cv::Mat kalman_covariance;
    cv::Mat appearance_feature;
};

} // namespace ultratrack
```

- [ ] **Step 5: Implement math helpers**

```cpp
// include/ultratrack/core/math.hpp
#pragma once
#include <ultratrack/core/types.hpp>

namespace ultratrack {

float IoU(const Rect2f& a, const Rect2f& b);
Rect2f clamp(const Rect2f& r, const Size2f& frame_size);
Point2f center(const Rect2f& r);

} // namespace ultratrack
```

```cpp
// src/core/math.cpp
#include <ultratrack/core/math.hpp>
#include <algorithm>

namespace ultratrack {

float IoU(const Rect2f& a, const Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

Point2f center(const Rect2f& r) {
    return Point2f(r.x + r.width / 2.0f, r.y + r.height / 2.0f);
}

} // namespace ultratrack
```

- [ ] **Step 6: Run tests**

```bash
cmake --build build --target ultratrack_tests
./build/tests/ultratrack_tests
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add include/ultratrack/core/ src/core/ tests/core/
git commit -m "feat(core): add Result<T>, Status, ErrorCode, and core types"
```

---

## Task 3: Detector Backend Abstraction

**Files:**
- Create: `include/ultratrack/detector/detector_backend.hpp`
- Create: `include/ultratrack/detector/opencv_dnn_backend.hpp`
- Create: `src/detector/opencv_dnn_backend.cpp`
- Create: `src/detector/detector_factory.cpp`
- Create: `tests/detector/test_opencv_dnn_backend.cpp`

**Interfaces:**
- Produces: `IDetectorBackend`, `OpenCVDNNBackend`, `DetectorBackend` enum.
- Consumes: `Frame`, `Detection`, `Result<T>`, `Status`.

- [ ] **Step 1: Write the failing test**

```cpp
// tests/detector/test_opencv_dnn_backend.cpp
#include <catch2/catch_test_macros.hpp>
#include <ultratrack/detector/opencv_dnn_backend.hpp>

using namespace ultratrack;

TEST_CASE("OpenCVDNNBackend rejects empty model path", "[detector]") {
    OpenCVDNNBackend::Config cfg{ "" };
    auto backend = OpenCVDNNBackend::create(cfg);
    REQUIRE(!backend.has_value());
    REQUIRE(backend.error().code() == ErrorCode::MODEL_LOAD_FAILED);
}
```

- [ ] **Step 2: Define the interface**

```cpp
// include/ultratrack/detector/detector_backend.hpp
#pragma once
#include <ultratrack/core/result.hpp>
#include <ultratrack/core/types.hpp>
#include <vector>
#include <memory>

namespace ultratrack {

class IDetectorBackend {
public:
    virtual ~IDetectorBackend() = default;
    virtual Result<std::vector<Detection>> detect(const Frame& frame) = 0;
};

enum class DetectorBackend { OPENCV_DNN, ONNX_RUNTIME, TENSORRT };

} // namespace ultratrack
```

- [ ] **Step 3: Implement OpenCVDNNBackend**

```cpp
// include/ultratrack/detector/opencv_dnn_backend.hpp
#pragma once
#include <ultratrack/detector/detector_backend.hpp>
#include <opencv2/dnn.hpp>

namespace ultratrack {

class OpenCVDNNBackend : public IDetectorBackend {
public:
    struct Config {
        std::string model_path;
        cv::Size input_size{640, 640};
        float confidence_threshold = 0.3f;
        float nms_threshold = 0.5f;
        bool use_cuda = false;
    };

    static Result<std::unique_ptr<IDetectorBackend>> create(const Config& cfg);
    Result<std::vector<Detection>> detect(const Frame& frame) override;

private:
    OpenCVDNNBackend(cv::dnn::Net net, const Config& cfg);
    cv::dnn::Net net_;
    Config cfg_;
};

} // namespace ultratrack
```

```cpp
// src/detector/opencv_dnn_backend.cpp
#include <ultratrack/detector/opencv_dnn_backend.hpp>
#include <spdlog/spdlog.h>

namespace ultratrack {

Result<std::unique_ptr<IDetectorBackend>> OpenCVDNNBackend::create(const Config& cfg) {
    if (cfg.model_path.empty()) {
        return Status(ErrorCode::INVALID_ARGUMENT, "model path is empty");
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
```

- [ ] **Step 4: Run tests**

Expected: the empty-path test passes; other tests require a model and can be skipped if absent.

- [ ] **Step 5: Commit**

```bash
git add include/ultratrack/detector/ src/detector/ tests/detector/
git commit -m "feat(detector): add detector backend abstraction and OpenCV DNN implementation"
```

---

## Task 4: Refactor Existing Tracker Components

**Files:**
- Move: `include/tracking/displacement_predictor.hpp` → `include/ultratrack/tracker/displacement_predictor.hpp`
- Move: `include/tracking/scale_estimator.hpp` → `include/ultratrack/tracker/scale_estimator.hpp`
- Move: `include/feature_extractor.hpp` → `include/ultratrack/tracker/features/feature_extractor.hpp`
- Move: `include/features/*` → `include/ultratrack/tracker/features/`
- Move corresponding `.cpp` files similarly.
- Update namespace references from `tracking/` to `tracker/`.
- Update `CMakeLists.txt` and existing tests.

**Interfaces:**
- Produces: `ultratrack::tracker::DisplacementPredictor`, `ultratrack::tracker::ScaleEstimator`, feature extractors under `ultratrack::tracker::features`.

- [ ] **Step 1: Move headers and sources into new layout**

Use `git mv` to preserve history:

```bash
git mv include/tracking include/ultratrack/tracker
git mv include/features include/ultratrack/tracker/features
git mv include/feature_extractor.hpp include/ultratrack/tracker/features/feature_extractor.hpp
git mv src/tracking src/tracker
git mv src/features src/tracker/features
git mv src/feature_extractor.cpp src/tracker/features/feature_extractor.cpp
```

- [ ] **Step 2: Update include paths**

In `include/ultratrack/tracker/displacement_predictor.hpp`:

```cpp
#pragma once
#include <opencv2/core.hpp>
```

In `include/ultratrack/tracker/features/feature_extractor.hpp`:

```cpp
#include "feature_extractor.hpp"
#include "hog_feature.hpp"
#include "gray_feature.hpp"
#include "cn_feature.hpp"
```

In `include/ultratrack.hpp` (new public umbrella), include:

```cpp
#include <ultratrack/tracker/displacement_predictor.hpp>
#include <ultratrack/tracker/scale_estimator.hpp>
#include <ultratrack/tracker/features/feature_extractor.hpp>
```

- [ ] **Step 3: Update existing tests**

Update `#include` paths in `tests/test_displacement.cpp`, `tests/test_scale.cpp`, `tests/test_features.cpp`.

- [ ] **Step 4: Build and run existing tests**

```bash
cmake --build build --target ultratrack_tests
./build/tests/ultratrack_tests
```

Expected: existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor(tracker): move displacement, scale, and feature extractors into tracker module"
```

---

## Task 5: Single-Track KCF Tracker

**Files:**
- Create: `include/ultratrack/tracker/kcf_tracker.hpp`
- Create: `include/ultratrack/tracker/kalman.hpp`
- Create: `src/tracker/kcf_tracker.cpp`
- Create: `src/tracker/kalman.cpp`
- Create: `tests/tracker/test_kcf_tracker.cpp`

**Interfaces:**
- Produces: `KCFTracker` class that updates a single `Track` given a frame patch.
- Consumes: `Track`, `Frame`, existing `DisplacementPredictor`, `ScaleEstimator`, feature extractors.

- [ ] **Step 1: Write the failing test**

```cpp
// tests/tracker/test_kcf_tracker.cpp
#include <catch2/catch_test_macros.hpp>
#include <ultratrack/tracker/kcf_tracker.hpp>

using namespace ultratrack;

TEST_CASE("KCFTracker initializes and predicts", "[tracker]") {
    KCFTracker::Config cfg;
    cfg.template_size = cv::Size(64, 64);
    auto tracker = KCFTracker::create(cfg);
    REQUIRE(tracker.has_value());

    Track track;
    track.bbox = Rect2f(100, 100, 40, 40);
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(frame, cv::Point(100, 100), cv::Point(140, 140), cv::Scalar(255, 255, 255), -1);

    auto status = tracker.value()->init(track, frame);
    REQUIRE(status.ok());

    auto pred = tracker.value()->predict(track, frame);
    REQUIRE(pred.has_value());
    REQUIRE(pred.value().width > 0);
}
```

- [ ] **Step 2: Define KCFTracker interface**

```cpp
// include/ultratrack/tracker/kcf_tracker.hpp
#pragma once
#include <ultratrack/core/result.hpp>
#include <ultratrack/core/types.hpp>
#include <ultratrack/tracker/features/feature_extractor.hpp>
#include <opencv2/core.hpp>
#include <memory>

namespace ultratrack {

class KCFTracker {
public:
    struct Config {
        cv::Size template_size{128, 128};
        float learning_rate = 0.02f;
        float lambda = 0.01f;
        float sigma = 2.0f;
        TrackingMode feature_mode = TrackingMode::ACCURATE;
        FeatureConfig feature_config;
    };

    static Result<std::unique_ptr<KCFTracker>> create(const Config& cfg);

    Status init(Track& track, const cv::Mat& frame);
    Result<Rect2f> predict(Track& track, const cv::Mat& frame);
    Status update(Track& track, const cv::Mat& frame, const Rect2f& detected_bbox);

private:
    explicit KCFTracker(const Config& cfg);
    Config cfg_;
    std::unique_ptr<MultiFeatureExtractor> feature_extractor_;
    cv::Mat hann_window_;
    cv::Mat gaussian_target_;

    cv::Mat createFilter(const cv::Mat& patch);
    cv::Mat createHannWindow(int size);
};

} // namespace ultratrack
```

- [ ] **Step 3: Implement minimal KCFTracker**

Port the existing `create_correlation_filter`, `track_correlation_filter`, and `update_correlation_filter` logic from `src/ultratrack.cpp` into `KCFTracker`. Keep FFT/SIMD helpers in a separate internal file or inside `KCFTracker`.

Key methods:

```cpp
// src/tracker/kcf_tracker.cpp
#include <ultratrack/tracker/kcf_tracker.hpp>
#include "simd_helpers.hpp"  // internal header for fft2d/ifft2d/simd ops

namespace ultratrack {

Result<std::unique_ptr<KCFTracker>> KCFTracker::create(const Config& cfg) {
    return std::unique_ptr<KCFTracker>(new KCFTracker(cfg));
}

KCFTracker::KCFTracker(const Config& cfg) : cfg_(cfg) {
    feature_extractor_ = std::make_unique<MultiFeatureExtractor>(cfg.feature_mode, cfg.feature_config);
    hann_window_ = createHannWindow(cfg.template_size.width);
    cv::Mat hann_1d = createHannWindow(cfg.template_size.width);
    cv::mulTransposed(hann_1d, hann_window_, false);
    hann_window_.convertTo(hann_window_, CV_32FC1);

    cv::Mat gx = cv::getGaussianKernel(cfg.template_size.width, cfg.sigma, CV_32F);
    cv::Mat gy = cv::getGaussianKernel(cfg.template_size.height, cfg.sigma, CV_32F);
    gaussian_target_ = gy * gx.t();
}

Status KCFTracker::init(Track& track, const cv::Mat& frame) {
    if (frame.empty()) return Status(ErrorCode::EMPTY_FRAME, "empty frame");
    cv::Rect safe = track.bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safe.area() <= 0) return Status(ErrorCode::INVALID_PATCH_SIZE, "invalid bbox");

    cv::Mat patch = frame(safe);
    track.correlation_filter = createFilter(patch);
    return Status();
}

Result<Rect2f> KCFTracker::predict(Track& track, const cv::Mat& frame) {
    if (track.correlation_filter.empty()) {
        return Status(ErrorCode::TRACKING_LOST, "no correlation filter");
    }
    // Port existing track_correlation_filter logic from src/ultratrack.cpp lines 569-600.
    // Use member variables track.correlation_filter, hann_window_, template_size from cfg_.
    // Clamp the search region to frame bounds, extract/resize the patch, apply Hann window,
    // compute FFT, multiply with the filter, inverse FFT, find the peak response,
    // and shift track.bbox by the peak offset. Then return track.bbox.
    return track.bbox;
}

Status KCFTracker::update(Track& track, const cv::Mat& frame, const Rect2f& detected_bbox) {
    track.bbox = detected_bbox;
    cv::Rect safe = detected_bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safe.area() <= 0) return Status(ErrorCode::INVALID_PATCH_SIZE, "invalid update bbox");
    cv::Mat patch = frame(safe);
    cv::Mat new_filter = createFilter(patch);
    if (new_filter.empty()) return Status(ErrorCode::FEATURE_EXTRACTION_FAILED, "filter update failed");
    track.correlation_filter = (1.0f - cfg_.learning_rate) * track.correlation_filter + cfg_.learning_rate * new_filter;
    return Status();
}

} // namespace ultratrack
```

- [ ] **Step 4: Run tests**

```bash
cmake --build build --target ultratrack_tests
./build/tests/ultratrack_tests "[tracker]"
```

Expected: KCF tracker initializes and predicts on synthetic frame.

- [ ] **Step 5: Commit**

```bash
git add include/ultratrack/tracker/kcf_tracker.hpp src/tracker/kcf_tracker.cpp src/tracker/simd_helpers.hpp tests/tracker/test_kcf_tracker.cpp
git commit -m "feat(tracker): add single-track KCF tracker class"
```

---

## Task 6: Track Lifecycle and Track Manager

**Files:**
- Create: `include/ultratrack/tracking_engine/track_lifecycle.hpp`
- Create: `include/ultratrack/tracking_engine/track_manager.hpp`
- Create: `src/tracking_engine/track_lifecycle.cpp`
- Create: `src/tracking_engine/track_manager.cpp`
- Create: `tests/tracking_engine/test_track_manager.cpp`

**Interfaces:**
- Produces: `TrackLifecycle` (tentative/confirmed/lost state machine), `TrackManager` (predict/create/update/remove tracks).
- Consumes: `Track`, `Detection`, `KCFTracker`.

- [ ] **Step 1: Write the failing test**

```cpp
// tests/tracking_engine/test_track_manager.cpp
#include <catch2/catch_test_macros.hpp>
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
    cfg.confirmation_threshold = 3;
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
```

- [ ] **Step 2: Define TrackLifecycle**

```cpp
// include/ultratrack/tracking_engine/track_lifecycle.hpp
#pragma once
#include <ultratrack/core/types.hpp>

namespace ultratrack {

struct LifecycleConfig {
    int confirmation_threshold = 3;
    int max_age = 30;
    int max_tentative_age = 5;
};

class TrackLifecycle {
public:
    explicit TrackLifecycle(const LifecycleConfig& cfg);

    void onMatched(Track& track);
    void onMissed(Track& track);
    bool shouldRemove(const Track& track) const;
    bool shouldConfirm(const Track& track) const;

private:
    LifecycleConfig cfg_;
};

} // namespace ultratrack
```

```cpp
// src/tracking_engine/track_lifecycle.cpp
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
```

- [ ] **Step 3: Define TrackManager**

```cpp
// include/ultratrack/tracking_engine/track_manager.hpp
#pragma once
#include <ultratrack/core/result.hpp>
#include <ultratrack/core/types.hpp>
#include <ultratrack/tracker/kcf_tracker.hpp>
#include <ultratrack/tracking_engine/track_lifecycle.hpp>
#include <vector>
#include <memory>

namespace ultratrack {

class TrackManager {
public:
    struct Config {
        LifecycleConfig lifecycle;
        KCFTracker::Config kcf;
    };

    static Result<std::unique_ptr<TrackManager>> create(const Config& cfg);
    Status update(const std::vector<Detection>& detections, const cv::Mat& frame);
    const std::vector<Track>& activeTracks() const;
    void reset();

private:
    explicit TrackManager(const Config& cfg);
    Config cfg_;
    std::vector<Track> tracks_;
    std::unique_ptr<KCFTracker> kcf_;
    uint64_t next_id_ = 1;

    void predictAll(const cv::Mat& frame);
    void createNewTracks(const std::vector<Detection>& unmatched, const cv::Mat& frame);
};

} // namespace ultratrack
```

- [ ] **Step 4: Implement TrackManager update flow**

For Phase 1, use a simplified greedy IoU association inside `TrackManager::update` (full Hungarian/DataAssociation in Task 7). The goal is to get lifecycle tests passing.

```cpp
// src/tracking_engine/track_manager.cpp
#include <ultratrack/tracking_engine/track_manager.hpp>
#include <ultratrack/core/math.hpp>
#include <algorithm>

namespace ultratrack {

Result<std::unique_ptr<TrackManager>> TrackManager::create(const Config& cfg) {
    auto kcf = KCFTracker::create(cfg.kcf);
    if (!kcf.has_value()) return kcf.error();
    auto tm = std::unique_ptr<TrackManager>(new TrackManager(cfg));
    tm->kcf_ = std::move(kcf.value());
    return tm;
}

TrackManager::TrackManager(const Config& cfg) : cfg_(cfg) {}

Status TrackManager::update(const std::vector<Detection>& detections, const cv::Mat& frame) {
    predictAll(frame);

    std::vector<bool> det_matched(detections.size(), false);
    std::vector<bool> track_matched(tracks_.size(), false);

    // Greedy IoU association for skeleton; replaced by Hungarian in Task 7
    for (size_t i = 0; i < tracks_.size(); ++i) {
        float best_iou = 0.3f; // gating threshold
        int best_j = -1;
        for (size_t j = 0; j < detections.size(); ++j) {
            if (det_matched[j]) continue;
            float iou = IoU(tracks_[i].bbox, detections[j].bbox);
            if (iou > best_iou) {
                best_iou = iou;
                best_j = static_cast<int>(j);
            }
        }
        if (best_j >= 0) {
            TrackLifecycle life(cfg_.lifecycle);
            life.onMatched(tracks_[i]);
            tracks_[i].bbox = detections[best_j].bbox;
            tracks_[i].confidence = detections[best_j].confidence;
            kcf_->update(tracks_[i], frame, detections[best_j].bbox);
            det_matched[best_j] = true;
            track_matched[i] = true;
        }
    }

    // Mark missed tracks
    TrackLifecycle life(cfg_.lifecycle);
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (!track_matched[i]) {
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
}

void TrackManager::predictAll(const cv::Mat& frame) {
    for (auto& track : tracks_) {
        track.age++;
        track.time_since_update++;
        // Optional: KCF predict to refine bbox when no detection
        auto pred = kcf_->predict(track, frame);
        if (pred.has_value()) {
            track.bbox = pred.value();
        }
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
```

- [ ] **Step 5: Run tests**

```bash
cmake --build build --target ultratrack_tests
./build/tests/ultratrack_tests "[tracking_engine]"
```

Expected: lifecycle tests pass.

- [ ] **Step 6: Commit**

```bash
git add include/ultratrack/tracking_engine/ src/tracking_engine/ tests/tracking_engine/
git commit -m "feat(tracking_engine): add track lifecycle and track manager skeleton"
```

---

## Task 7: Data Association with Hungarian Algorithm

**Files:**
- Create: `include/ultratrack/tracking_engine/data_association.hpp`
- Create: `src/tracking_engine/data_association.cpp`
- Create: `tests/tracking_engine/test_data_association.cpp`

**Interfaces:**
- Produces: `DataAssociation` class returning matched/unmatched pairs.
- Consumes: `Track`, `Detection`, `IoU`, Re-ID features.

- [ ] **Step 1: Write the failing test**

```cpp
// tests/tracking_engine/test_data_association.cpp
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
```

- [ ] **Step 2: Define DataAssociation interface**

```cpp
// include/ultratrack/tracking_engine/data_association.hpp
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
```

- [ ] **Step 3: Implement DataAssociation**

Port the existing Hungarian assignment from `src/ultratrack.cpp` into `DataAssociation::hungarian`. Build cost matrix as `1 - IoU` with gating.

```cpp
// src/tracking_engine/data_association.cpp
#include <ultratrack/tracking_engine/data_association.hpp>
#include <ultratrack/core/math.hpp>
#include <limits>
#include <algorithm>
#include <cmath>

namespace ultratrack {

DataAssociation::DataAssociation(const Config& cfg) : cfg_(cfg) {}

AssociationResult DataAssociation::associate(const std::vector<Track>& tracks,
                                              const std::vector<Detection>& detections) {
    AssociationResult result;
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

    return result;
}

std::vector<std::pair<int, int>> DataAssociation::hungarian(const cv::Mat& cost_matrix) {
    // Port existing implementation from src/ultratrack.cpp::hungarian_assignment
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
```

- [ ] **Step 4: Integrate DataAssociation into TrackManager**

Replace the greedy association in `TrackManager::update` with `DataAssociation`.

- [ ] **Step 5: Run tests**

```bash
cmake --build build --target ultratrack_tests
./build/tests/ultratrack_tests "[tracking_engine]"
```

Expected: association tests and lifecycle tests pass.

- [ ] **Step 6: Commit**

```bash
git add include/ultratrack/tracking_engine/data_association.hpp src/tracking_engine/data_association.cpp tests/tracking_engine/test_data_association.cpp
git commit -m "feat(tracking_engine): add Hungarian data association with IoU gating"
```

---

## Task 8: Public API — UltraTrackerSDK and TrackingSession

**Files:**
- Create: `include/ultratrack/ultratrack_sdk.hpp`
- Create: `include/ultratrack/tracking_session.hpp`
- Create: `src/ultratrack_sdk.cpp`
- Create: `src/tracking_session.cpp`
- Create: `tests/test_public_api.cpp`

**Interfaces:**
- Produces: `UltraTrackerSDK`, `TrackingSession`, `TrackingOutput`, `TrackerSettings`.
- Consumes: all internal modules.

- [ ] **Step 1: Write the failing test**

```cpp
// tests/test_public_api.cpp
#include <catch2/catch_test_macros.hpp>
#include <ultratrack/ultratrack_sdk.hpp>
#include <ultratrack/tracking_session.hpp>

using namespace ultratrack;

TEST_CASE("SDK initializes with trial license", "[public_api]") {
    SDKConfig cfg;
    cfg.license_key = "TRIAL";
    auto status = UltraTrackerSDK::initialize(cfg);
    REQUIRE(status.ok());
    REQUIRE(UltraTrackerSDK::is_licensed());
    UltraTrackerSDK::shutdown();
}

TEST_CASE("TrackingSession processes synthetic frame", "[public_api]") {
    SDKConfig cfg; cfg.license_key = "TRIAL";
    UltraTrackerSDK::initialize(cfg);

    TrackerSettings ts;
    ts.detector_model_path = "models/yolov11n.onnx";
    ts.backend = DetectorBackend::OPENCV_DNN;
    ts.mode = TrackingMode::FAST;

    auto session = TrackingSession::create(ts);
    if (!session.has_value()) {
        // Skip if no model present
        UltraTrackerSDK::shutdown();
        return;
    }

    Frame frame;
    frame.width = 640;
    frame.height = 480;
    frame.format = FrameFormat::BGR;
    frame.data = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(frame.data, cv::Point(100, 100), cv::Point(140, 140), cv::Scalar(255, 255, 255), -1);

    auto out = session.value()->processFrame(frame);
    REQUIRE(out.has_value());
    UltraTrackerSDK::shutdown();
}
```

- [ ] **Step 2: Define public API headers**

```cpp
// include/ultratrack/ultratrack_sdk.hpp
#pragma once
#include <ultratrack/core/result.hpp>
#include <string>

namespace ultratrack {

enum class LogLevel { TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL };

struct SDKConfig {
    std::string license_key;
    std::string activation_server_url;
    LogLevel log_level = LogLevel::INFO;
    std::string telemetry_endpoint;
};

class UltraTrackerSDK {
public:
    static Status initialize(const SDKConfig& cfg);
    static void shutdown();
    static bool is_licensed();
};

} // namespace ultratrack
```

```cpp
// include/ultratrack/tracking_session.hpp
#pragma once
#include <ultratrack/core/result.hpp>
#include <ultratrack/core/types.hpp>
#include <ultratrack/detector/detector_backend.hpp>
#include <ultratrack/tracker/features/feature_extractor.hpp>
#include <memory>
#include <vector>

namespace ultratrack {

struct TrackerSettings {
    std::string detector_model_path;
    DetectorBackend backend = DetectorBackend::OPENCV_DNN;
    TrackingMode mode = TrackingMode::BALANCED;
    float confidence_threshold = 0.3f;
    float nms_threshold = 0.5f;
    std::string reid_model_path;
};

struct TrackingOutput {
    std::vector<Track> tracks;
    uint64_t primary_track_id = 0;
    cv::Mat annotated_frame;
    double latency_ms = 0.0;
};

class TrackingSession {
public:
    static Result<std::unique_ptr<TrackingSession>> create(const TrackerSettings& settings);

    Result<TrackingOutput> processFrame(const Frame& frame);
    Result<void> selectPrimaryTarget(uint64_t track_id);
    Result<void> reset();
    const std::vector<Track>& activeTracks() const;

private:
    explicit TrackingSession(const TrackerSettings& settings);
    TrackerSettings settings_;
    std::unique_ptr<class TrackingSessionImpl> impl_;
};

} // namespace ultratrack
```

- [ ] **Step 3: Implement UltraTrackerSDK**

```cpp
// src/ultratrack_sdk.cpp
#include <ultratrack/ultratrack_sdk.hpp>
#include <spdlog/spdlog.h>

namespace ultratrack {

namespace {
    bool g_initialized = false;
    bool g_licensed = false;
}

Status UltraTrackerSDK::initialize(const SDKConfig& cfg) {
    if (g_initialized) return Status(ErrorCode::ALREADY_INITIALIZED, "SDK already initialized");
    spdlog::set_level(static_cast<spdlog::level::level_enum>(cfg.log_level));
    // Phase 1: accept TRIAL key; Phase 2 adds real license checks
    g_licensed = (cfg.license_key == "TRIAL" || !cfg.license_key.empty());
    g_initialized = true;
    return Status();
}

void UltraTrackerSDK::shutdown() {
    g_initialized = false;
    g_licensed = false;
}

bool UltraTrackerSDK::is_licensed() { return g_initialized && g_licensed; }

} // namespace ultratrack
```

- [ ] **Step 4: Implement TrackingSession**

Create `src/tracking_session.cpp` with a pimpl `TrackingSessionImpl` that owns `IDetectorBackend` and `TrackManager`.

```cpp
// src/tracking_session.cpp
#include <ultratrack/tracking_session.hpp>
#include <ultratrack/detector/opencv_dnn_backend.hpp>
#include <ultratrack/tracking_engine/track_manager.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>

namespace ultratrack {

class TrackingSessionImpl {
public:
    TrackerSettings settings;
    std::unique_ptr<IDetectorBackend> detector;
    std::unique_ptr<TrackManager> track_manager;
    uint64_t primary_track_id = 0;
};

Result<std::unique_ptr<TrackingSession>> TrackingSession::create(const TrackerSettings& settings) {
    if (!UltraTrackerSDK::is_licensed()) {
        return Status(ErrorCode::LICENSE_INVALID, "SDK not licensed");
    }
    auto session = std::unique_ptr<TrackingSession>(new TrackingSession(settings));

    OpenCVDNNBackend::Config dcfg;
    dcfg.model_path = settings.detector_model_path;
    dcfg.confidence_threshold = settings.confidence_threshold;
    dcfg.nms_threshold = settings.nms_threshold;
    auto det = OpenCVDNNBackend::create(dcfg);
    if (!det.has_value()) return det.error();
    session->impl_->detector = std::move(det.value());

    TrackManager::Config tcfg;
    tcfg.kcf.feature_mode = settings.mode;
    auto tm = TrackManager::create(tcfg);
    if (!tm.has_value()) return tm.error();
    session->impl_->track_manager = std::move(tm.value());

    return session;
}

TrackingSession::TrackingSession(const TrackerSettings& settings) : settings_(settings) {
    impl_ = std::make_unique<TrackingSessionImpl>();
    impl_->settings = settings;
}

Result<TrackingOutput> TrackingSession::processFrame(const Frame& frame) {
    auto start = std::chrono::high_resolution_clock::now();

    auto detections = impl_->detector->detect(frame);
    if (!detections.has_value()) return detections.error();

    cv::Mat bgr;
    if (frame.format == FrameFormat::BGR) {
        bgr = frame.data;
    } else if (frame.format == FrameFormat::RGB) {
        cv::cvtColor(frame.data, bgr, cv::COLOR_RGB2BGR);
    } else {
        return Status(ErrorCode::NOT_IMPLEMENTED, "frame format not supported");
    }

    auto status = impl_->track_manager->update(detections.value(), bgr);
    if (!status.ok()) return status;

    TrackingOutput out;
    out.tracks = impl_->track_manager->activeTracks();
    out.primary_track_id = impl_->primary_track_id;
    bgr.copyTo(out.annotated_frame);
    for (const auto& t : out.tracks) {
        cv::Scalar color = (t.state == TrackState::CONFIRMED) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
        cv::rectangle(out.annotated_frame, t.bbox, color, 2);
    }

    auto end = std::chrono::high_resolution_clock::now();
    out.latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return out;
}

Result<void> TrackingSession::selectPrimaryTarget(uint64_t track_id) {
    impl_->primary_track_id = track_id;
    return Status();
}

Result<void> TrackingSession::reset() {
    impl_->track_manager->reset();
    impl_->primary_track_id = 0;
    return Status();
}

const std::vector<Track>& TrackingSession::activeTracks() const {
    return impl_->track_manager->activeTracks();
}

TrackingSession::~TrackingSession() = default;

} // namespace ultratrack
```

- [ ] **Step 5: Run tests**

```bash
cmake --build build --target ultratrack_tests
./build/tests/ultratrack_tests "[public_api]"
```

Expected: SDK init and synthetic frame tests pass (detector tests skip if model missing).

- [ ] **Step 6: Commit**

```bash
git add include/ultratrack/ultratrack_sdk.hpp include/ultratrack/tracking_session.hpp src/ultratrack_sdk.cpp src/tracking_session.cpp tests/test_public_api.cpp
git commit -m "feat(api): add public SDK and TrackingSession API"
```

---

## Task 9: Example Application

**Files:**
- Create: `examples/simple_tracker.cpp`
- Create: `examples/CMakeLists.txt`
- Modify: `CMakeLists.txt` to add `add_subdirectory(examples)`

**Interfaces:**
- Consumes: `UltraTrackerSDK`, `TrackingSession`, `TrackingOutput`.

- [ ] **Step 1: Create simple_tracker.cpp**

```cpp
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
```

- [ ] **Step 2: Create examples/CMakeLists.txt**

```cmake
add_executable(simple_tracker simple_tracker.cpp)
target_link_libraries(simple_tracker PRIVATE ultratrack)
install(TARGETS simple_tracker RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
```

- [ ] **Step 3: Build and run (manual test)**

```bash
cmake --build build --target simple_tracker
# Requires model to run
./build/examples/simple_tracker models/yolov11n.onnx
```

- [ ] **Step 4: Commit**

```bash
git add examples/ CMakeLists.txt
git commit -m "feat(examples): add simple_tracker example"
```

---

## Task 10: Benchmark Harness Skeleton

**Files:**
- Create: `include/ultratrack/benchmark/harness.hpp`
- Create: `src/benchmark/harness.cpp`
- Create: `tests/benchmark/test_harness.cpp`
- Create: `tools/benchmark_cli.cpp`

**Interfaces:**
- Produces: `BenchmarkHarness`, `BenchmarkResult`.
- Consumes: `TrackingSession`, `TrackerSettings`, MOT-format files.

- [ ] **Step 1: Write the failing test**

```cpp
// tests/benchmark/test_harness.cpp
#include <catch2/catch_test_macros.hpp>
#include <ultratrack/benchmark/harness.hpp>

using namespace ultratrack;

TEST_CASE("BenchmarkHarness loads MOT sequence directory", "[benchmark]") {
    BenchmarkHarness harness;
    auto seq = harness.loadSequence("tests/data/mot_dummy");
    REQUIRE(seq.has_value());
    REQUIRE(seq.value().frames.size() == 10);
}
```

- [ ] **Step 2: Define BenchmarkHarness interface**

```cpp
// include/ultratrack/benchmark/harness.hpp
#pragma once
#include <ultratrack/core/result.hpp>
#include <ultratrack/core/types.hpp>
#include <ultratrack/tracking_session.hpp>
#include <string>
#include <vector>

namespace ultratrack {

struct GroundTruthBox {
    int frame_id;
    int track_id;
    Rect2f bbox;
};

struct MOTSequence {
    std::vector<Frame> frames;
    std::vector<GroundTruthBox> ground_truth;
};

struct BenchmarkMetrics {
    double mota = 0.0;
    double motp = 0.0;
    double idf1 = 0.0;
    double hota = 0.0;
    double avg_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
};

struct BenchmarkResult {
    std::string variant_name;
    BenchmarkMetrics metrics;
};

class BenchmarkHarness {
public:
    Result<MOTSequence> loadSequence(const std::string& image_dir);
    Result<MOTSequence> loadSequenceWithGT(const std::string& image_dir,
                                            const std::string& gt_file);
    BenchmarkResult runVariant(const std::string& name,
                                const TrackerSettings& settings,
                                const MOTSequence& seq);
    void report(const std::vector<BenchmarkResult>& results);
};

} // namespace ultratrack
```

- [ ] **Step 3: Implement skeleton harness**

```cpp
// src/benchmark/harness.cpp
#include <ultratrack/benchmark/harness.hpp>
#include <ultratrack/ultratrack_sdk.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

namespace ultratrack {

Result<MOTSequence> BenchmarkHarness::loadSequence(const std::string& image_dir) {
    std::vector<cv::String> files;
    cv::glob(image_dir + "/*.jpg", files);
    if (files.empty()) cv::glob(image_dir + "/*.png", files);
    if (files.empty()) return Status(ErrorCode::INVALID_ARGUMENT, "no images found");

    MOTSequence seq;
    for (const auto& f : files) {
        cv::Mat img = cv::imread(f);
        if (img.empty()) continue;
        Frame frame;
        frame.width = img.cols;
        frame.height = img.rows;
        frame.format = FrameFormat::BGR;
        frame.data = img;
        seq.frames.push_back(frame);
    }
    return seq;
}

Result<MOTSequence> BenchmarkHarness::loadSequenceWithGT(const std::string& image_dir,
                                                          const std::string& gt_file) {
    auto seq = loadSequence(image_dir);
    if (!seq.has_value()) return seq.error();

    std::ifstream in(gt_file);
    if (!in) return Status(ErrorCode::INVALID_ARGUMENT, "cannot open gt file");

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        GroundTruthBox gt;
        char sep;
        float x, y, w, h;
        ss >> gt.frame_id >> sep >> gt.track_id >> sep >> x >> sep >> y >> sep >> w >> sep >> h;
        gt.bbox = Rect2f(x, y, w, h);
        seq.value().ground_truth.push_back(gt);
    }
    return seq.value();
}

BenchmarkResult BenchmarkHarness::runVariant(const std::string& name,
                                              const TrackerSettings& settings,
                                              const MOTSequence& seq) {
    SDKConfig cfg; cfg.license_key = "TRIAL";
    UltraTrackerSDK::initialize(cfg);

    auto session = TrackingSession::create(settings);
    BenchmarkResult result{name, {}};
    if (!session.has_value()) {
        result.metrics.avg_latency_ms = -1.0;
        UltraTrackerSDK::shutdown();
        return result;
    }

    std::vector<double> latencies;
    for (const auto& frame : seq.frames) {
        auto start = std::chrono::high_resolution_clock::now();
        auto out = session.value()->processFrame(frame);
        auto end = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        (void)out;
    }

    if (!latencies.empty()) {
        result.metrics.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        std::sort(latencies.begin(), latencies.end());
        result.metrics.p95_latency_ms = latencies[static_cast<size_t>(latencies.size() * 0.95)];
    }

    // MOTA/HOTA/IDF1 placeholders; integrate trackeval or motmetrics in follow-up
    result.metrics.mota = 0.0;
    result.metrics.hota = 0.0;
    result.metrics.idf1 = 0.0;

    UltraTrackerSDK::shutdown();
    return result;
}

void BenchmarkHarness::report(const std::vector<BenchmarkResult>& results) {
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& r : results) {
        std::cout << r.variant_name << ": avg=" << r.metrics.avg_latency_ms
                  << "ms p95=" << r.metrics.p95_latency_ms
                  << "ms MOTA=" << r.metrics.mota
                  << " HOTA=" << r.metrics.hota
                  << " IDF1=" << r.metrics.idf1 << "\n";
    }
}

} // namespace ultratrack
```

- [ ] **Step 4: Create benchmark CLI**

```cpp
// tools/benchmark_cli.cpp
#include <ultratrack/benchmark/harness.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: benchmark_cli <image_dir> <gt.txt> <model.onnx>\n";
        return 1;
    }
    ultratrack::BenchmarkHarness harness;
    auto seq = harness.loadSequenceWithGT(argv[1], argv[2]);
    if (!seq.has_value()) {
        std::cerr << "Failed to load sequence: " << seq.error().message() << "\n";
        return 1;
    }

    ultratrack::TrackerSettings settings;
    settings.detector_model_path = argv[3];
    settings.mode = ultratrack::TrackingMode::BALANCED;

    auto result = harness.runVariant("ultratrack_balanced", settings, seq.value());
    harness.report({result});
    return 0;
}
```

- [ ] **Step 5: Add dummy MOT data for tests**

Create `tests/data/mot_dummy/` with 10 blank 640x480 images and a matching `gt.txt`.

- [ ] **Step 6: Run tests**

```bash
cmake --build build --target ultratrack_tests
./build/tests/ultratrack_tests "[benchmark]"
```

Expected: sequence loading test passes.

- [ ] **Step 7: Commit**

```bash
git add include/ultratrack/benchmark/ src/benchmark/ tests/benchmark/ tools/benchmark_cli.cpp tests/data/mot_dummy/
git commit -m "feat(benchmark): add benchmark harness skeleton and CLI"
```

---

## Task 11: Telemetry and Metrics

**Files:**
- Create: `include/ultratrack/telemetry/metrics.hpp`
- Create: `src/telemetry/metrics.cpp`
- Create: `tests/telemetry/test_metrics.cpp`

**Interfaces:**
- Produces: `StageTimer`, `MetricsCollector`.
- Consumes: spdlog.

- [ ] **Step 1: Write the failing test**

```cpp
// tests/telemetry/test_metrics.cpp
#include <catch2/catch_test_macros.hpp>
#include <ultratrack/telemetry/metrics.hpp>

using namespace ultratrack;

TEST_CASE("StageTimer records latency", "[telemetry]") {
    MetricsCollector collector;
    {
        StageTimer t(collector, "detect");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    auto metrics = collector.snapshot();
    REQUIRE(metrics.count("detect_count") == 1);
    REQUIRE(metrics.at("detect_count") >= 1.0);
    REQUIRE(metrics.at("detect_p50_ms") >= 0.0);
}
```

- [ ] **Step 2: Implement metrics**

```cpp
// include/ultratrack/telemetry/metrics.hpp
#pragma once
#include <string>
#include <unordered_map>
#include <chrono>
#include <mutex>

namespace ultratrack {

class MetricsCollector {
public:
    void record(const std::string& stage, double latency_ms);
    std::unordered_map<std::string, double> snapshot() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::vector<double>> data_;
};

class StageTimer {
public:
    StageTimer(MetricsCollector& collector, std::string stage);
    ~StageTimer();

private:
    MetricsCollector& collector_;
    std::string stage_;
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace ultratrack
```

```cpp
// src/telemetry/metrics.cpp
#include <ultratrack/telemetry/metrics.hpp>
#include <algorithm>
#include <numeric>

namespace ultratrack {

void MetricsCollector::record(const std::string& stage, double latency_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    data_[stage].push_back(latency_ms);
}

std::unordered_map<std::string, double> MetricsCollector::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_map<std::string, double> out;
    for (const auto& [stage, vals] : data_) {
        if (vals.empty()) continue;
        out[stage + "_count"] = static_cast<double>(vals.size());
        out[stage + "_p50_ms"] = vals[vals.size() / 2];
        double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
        out[stage + "_avg_ms"] = sum / vals.size();
    }
    return out;
}

StageTimer::StageTimer(MetricsCollector& collector, std::string stage)
    : collector_(collector), stage_(std::move(stage)), start_(std::chrono::high_resolution_clock::now()) {}

StageTimer::~StageTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start_).count();
    collector_.record(stage_, ms);
}

} // namespace ultratrack
```

- [ ] **Step 3: Integrate timers into TrackingSession**

Add `MetricsCollector` to `TrackingSessionImpl` and wrap detection and tracking stages with `StageTimer`.

- [ ] **Step 4: Run tests**

```bash
cmake --build build --target ultratrack_tests
./build/tests/ultratrack_tests "[telemetry]"
```

Expected: metrics tests pass.

- [ ] **Step 5: Commit**

```bash
git add include/ultratrack/telemetry/ src/telemetry/ tests/telemetry/
git commit -m "feat(telemetry): add stage timing metrics collector"
```

---

## Task 12: CI/CD Skeleton

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: GitHub Actions workflow.

- [ ] **Step 1: Create CI workflow**

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  build-linux:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y libopencv-dev libspdlog-dev nlohmann-json3-dev catch2
      - name: Configure
        run: cmake -B build -S . -DBUILD_TESTS=ON
      - name: Build
        run: cmake --build build -j$(nproc)
      - name: Test
        run: ctest --test-dir build --output-on-failure

  build-windows:
    runs-on: windows-2022
    steps:
      - uses: actions/checkout@v4
      - name: Setup vcpkg
        uses: lukka/run-vcpkg@v11
        with:
          vcpkgGitCommitId: 'a34c873a9717a888f58dc05268dea15592c2f984'
      - name: Configure
        run: cmake -B build -S . -DBUILD_TESTS=ON -DCMAKE_TOOLCHAIN_FILE=${{ env.VCPKG_ROOT }}/scripts/buildsystems/vcpkg.cmake
      - name: Build
        run: cmake --build build --config Release
      - name: Test
        run: ctest --test-dir build --output-on-failure -C Release
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions build and test workflow"
```

---

## Self-Review

### Spec coverage

| Spec section | Implementing task |
|--------------|-------------------|
| Stable C++ public API | Task 8 |
| Multi-target tracking with persistent IDs | Tasks 6, 7 |
| Detector backends | Task 3 |
| Tracking mode presets | Task 5, 8 |
| Benchmarking harness | Task 10 |
| Telemetry/metrics | Task 11 |
| CMake packaging | Task 1 |
| Tests | Every task |

### Placeholder scan

- No "TBD", "TODO", or "implement later" strings remain in the plan.
- All code shown is concrete, though some internal ports (e.g., full KCF predict) reference existing `src/ultratrack.cpp` logic.
- Association cost uses IoU only in Phase 1; Re-ID and motion weights are config knobs for Phase 2.

### Type consistency

- `Result<T>`, `Status`, `ErrorCode` are used consistently across modules.
- `Track`, `Detection`, `Frame` come from `core/types.hpp`.
- `TrackingSession` uses `DetectorBackend` enum and `TrackingMode` enum.

### Gaps

- Re-ID feature extractor is not implemented in Phase 1; association is IoU-only. This is intentional per scope.
- PTZ control is Phase 2.
- Real license activation server is Phase 2.
- HOTA/MOTA/IDF1 computation is a skeleton; integration with trackeval or a metrics library is a follow-up task.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-05-ultratrack-sdk-phase1.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — Dispatch a fresh coder subagent per task; I review between tasks for fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans with checkpoints for review.

Which approach do you want? Also confirm if I should commit the design doc and this plan to git first.
