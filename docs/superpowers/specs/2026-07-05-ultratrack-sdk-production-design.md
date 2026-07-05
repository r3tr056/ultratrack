# UltraTrack SDK — Production Design

**Date:** 2026-07-05  
**Author:** Kimi Code (design session)  
**Status:** Draft pending review  
**Approach:** Monolithic C++ SDK (Option A)  

---

## 1. Purpose

Turn the existing research-grade UltraTracker into a licensable, production-grade C++ SDK that OEMs can embed into pan-tilt auto-tracking systems, camera trackers, and video analytics pipelines. The product must demonstrably outperform open-source and commercial alternatives — specifically Ultralytics/YOLO trackers and NVIDIA tracking solutions — in both accuracy and latency on the same hardware.

---

## 2. Goals

- Ship a single C++ SDK (`libultratrack`) for Windows x64, Linux x64, and NVIDIA Jetson ARM64+CUDA.
- Provide multi-target tracking with persistent IDs, occlusion handling, and primary-target selection.
- Integrate ONVIF PTZ control for auto-centering and follow behavior.
- Implement per-seat online license activation with trial mode.
- Beat Ultralytics `yolo11n` tracker (ByteTrack/BoT-SORT) and NVIDIA DeepStream NvDCF in HOTA/MOTA and FPS on identical hardware.
- Include a first-class benchmarking and A/B testing framework from day one.
- Provide clean C++ public API, examples, installers, and documentation.

## 3. Non-Goals

- Full GStreamer/DeepStream plugin polish is not in V1 (core SDK first).
- MacOS support is not in V1.
- VISCA/Pelco protocols are not in V1 (ONVIF only).
- C# / Python language bindings are not in V1 (follow-up phase).
- Cloud telemetry dashboard is not in V1 (SDK-side telemetry only).

---

## 4. Background & Current State

The existing UltraTracker (`src/ultratrack.cpp`) is a research implementation of the CIBDA 2025 paper "Displacement prediction strategy based on KCF in tracking". It combines:

- YOLO object detection (OpenCV DNN / ONNX).
- KCF-style correlation filter tracking with HOG/Gray/CN features.
- Kalman filter for motion smoothing.
- Displacement prediction and scale estimation.
- SIMD acceleration (AVX2 / NEON).
- Optional CUDA FFT.
- Stubs for TensorRT (`src/gpu/`), GStreamer/DeepStream plugin (`src/gst/`), and Python CLI/GUI (`scripts/`).

Current limitations for a sellable product:

- Single-object focus; multi-object lifecycle is ad-hoc.
- No clean public API or packaged library.
- No PTZ/camera control abstraction.
- No licensing or activation system.
- No telemetry, structured logging, or metrics.
- No reproducible benchmarking harness.
- Tests cover only component configs, not end-to-end tracking accuracy.
- Build system is hardcoded to a local OpenCV path.

---

## 5. Requirements

### 5.1 Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-1 | The SDK shall expose a stable C++ public API for initialization, tracking session management, and PTZ control. |
| FR-2 | The SDK shall support multi-target tracking with persistent track IDs across frames. |
| FR-3 | The SDK shall allow selection of a primary target for PTZ follow/centering. |
| FR-4 | The SDK shall support ONVIF Profile S/T PTZ absolute and relative moves. |
| FR-5 | The SDK shall support detector backends: OpenCV DNN, ONNX Runtime, and TensorRT (Jetson). |
| FR-6 | The SDK shall provide tracking mode presets: `FAST`, `BALANCED`, `ACCURATE`, and `EDGE_TENSORRT`. |
| FR-7 | The SDK shall validate license keys locally and online, with a trial mode. |
| FR-8 | The SDK shall load runtime configuration from JSON/YAML files and expose calibration persistence. |
| FR-9 | The SDK shall emit structured logs and performance metrics. |
| FR-10 | The SDK shall provide a benchmarking harness with MOT-format dataset loaders and HOTA/MOTA/IDF1 metrics. |

### 5.2 Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | Latency on Jetson Orin NX shall be < 1.5 ms per frame for `FAST` mode at 640x480. |
| NFR-2 | The SDK shall match or exceed Ultralytics YOLO tracker HOTA on MOT17 validation. |
| NFR-3 | The SDK shall run on Windows 10/11 x64, Ubuntu 20.04/22.04 x64, and JetPack 5/6. |
| NFR-4 | The public API shall not throw exceptions across the library boundary. |
| NFR-5 | The SDK shall be installable via MSI (Windows), `.deb`/`.rpm` (Linux), and tar/container (Jetson). |
| NFR-6 | The SDK shall gracefully degrade to CPU-only mode when CUDA/TensorRT is unavailable. |

---

## 6. Architecture

### 6.1 Module Map

```
ultratrack/
├── core/                 # Frame, BBox, Track, Detection, math utilities
├── detector/             # Detector backends (OpenCV DNN, ONNX Runtime, TensorRT)
├── tracker/              # KCF correlation filter, Kalman, displacement, scale
├── reid/                 # Re-ID feature extractors (OSNet, lightweight CPU model)
├── tracking_engine/      # Multi-target lifecycle, data association, ID management
├── ptz/                  # ONVIF controller, framing policy, velocity limits
├── licensing/            # License key validation, activation, trial mode
├── config/               # JSON/YAML config, calibration, preset profiles
├── telemetry/            # Logging, metrics, optional cloud upload
├── benchmark/            # Dataset loaders, metrics, A/B registry, reports
└── io/                   # Video capture abstraction, GPU buffer handles
```

### 6.2 Module Responsibilities

- **core**: Value types and basic geometry. No external dependencies.
- **detector**: Abstract `IDetectorBackend` with concrete implementations. Normalizes outputs to `std::vector<Detection>`.
- **tracker**: Single-target correlation-filter tracker per track. Owns filter training/update and scale estimation.
- **reid**: Appearance embedding for re-identification during association and occlusion recovery.
- **tracking_engine**: `TrackManager` predicts tracks, `DataAssociation` matches detections using IoU + Re-ID + motion cost, and `TrackLifecycle` manages tentative/confirmed/lost states.
- **ptz**: `PTZController` translates primary-target bbox to pan/tilt/zoom velocities with smoothing and hardware limits.
- **licensing**: `LicenseManager` checks signatures, activates seats, and enforces trial constraints.
- **config**: Schema-validated runtime parameters and camera calibration storage.
- **telemetry**: spdlog-based logging and lightweight metrics counters.
- **benchmark**: Evaluation harness used internally and by OEMs to reproduce performance claims.
- **io**: Abstracts capture from files, cameras, and NVMM GPU buffers.

---

## 7. Public C++ API

```cpp
namespace ultratrack {

// Initialization and licensing
struct SDKConfig {
    std::string license_key;
    std::string activation_server_url;
    LogLevel log_level = LogLevel::INFO;
    std::optional<std::string> telemetry_endpoint;
};

class UltraTrackerSDK {
public:
    static Result<void> initialize(const SDKConfig& cfg);
    static void shutdown();
    static bool is_licensed();
    static LicenseInfo license_info();
};

// Tracking session
struct TrackerSettings {
    std::string detector_model_path;
    DetectorBackend backend = DetectorBackend::ONNX_RUNTIME;
    TrackingMode mode = TrackingMode::BALANCED;
    float confidence_threshold = 0.3f;
    float nms_threshold = 0.5f;
    std::optional<std::string> reid_model_path;
};

class TrackingSession {
public:
    static Result<std::unique_ptr<TrackingSession>> create(const TrackerSettings& settings);

    Result<TrackingOutput> processFrame(const Frame& frame);
    Result<void> selectPrimaryTarget(uint64_t track_id);
    Result<void> setPrimaryROI(const Rect2f& roi);
    Result<void> reset();

    std::vector<Track> activeTracks() const;
    PerformanceMetrics lastMetrics() const;
};

// PTZ control
struct PTZProfile {
    std::string onvif_url;
    std::string username;
    std::string password;
    PTZVelocityLimits limits;
    FramingPolicy framing;
};

class PTZController {
public:
    Result<void> connect(const PTZProfile& profile);
    Result<void> update(const TrackingOutput& output);  // auto-follow primary
    Result<void> moveAbsolute(float pan, float tilt, float zoom);
    Result<void> moveRelative(float pan_delta, float tilt_delta, float zoom_delta);
    Result<void> stop();
};

// Output types
struct TrackingOutput {
    std::vector<Track> tracks;
    uint64_t primary_track_id = 0;
    cv::Mat annotated_frame;
    PerformanceMetrics metrics;
};

} // namespace ultratrack
```

---

## 8. Internal Design

### 8.1 Tracking Engine

The existing single-object `UltraTracker` class is split into:

1. **TrackManager**: owns `std::vector<Track>`, runs prediction, handles lifecycle.
2. **DataAssociation**: builds cost matrix (IoU + Re-ID cosine + Kalman Mahalanobis) and runs Hungarian assignment.
3. **TrackLifecycle**: promotes tentative tracks after N hits, marks lost after M missed frames, removes stale tracks.
4. **PrimaryTargetSelector**: locks user-selected ID or auto-selects highest-confidence central track.

### 8.2 Detector Backends

| Backend | Use Case |
|---------|----------|
| OpenCV DNN | Quick CPU evaluation, fallback. |
| ONNX Runtime | Cross-platform production CPU/GPU. |
| TensorRT | Jetson edge, maximum throughput. |

All backends implement:

```cpp
class IDetectorBackend {
public:
    virtual Result<std::vector<Detection>> detect(const Frame& frame) = 0;
    virtual ~IDetectorBackend() = default;
};
```

### 8.3 Track State

```cpp
struct Track {
    uint64_t id;
    Rect2f bbox;
    float confidence;
    TrackState state;  // tentative / confirmed / lost
    int age;
    int hits;
    int time_since_update;

    cv::Mat kalman_state;
    cv::Mat kalman_covariance;
    cv::Mat correlation_filter;
    cv::Mat appearance_feature;

    std::unique_ptr<DisplacementPredictor> displacement;
    std::unique_ptr<ScaleEstimator> scale;
};
```

### 8.4 Association Cost

```
cost(i, j) = w1 * (1 - IoU(track_i, det_j))
           + w2 * (1 - cosine(reid_i, reid_j))
           + w3 * mahalanobis(track_i, det_j)
```

Weights are configurable per tracking mode. Gating thresholds prevent impossible matches.

### 8.5 PTZ Controller

- **ONVIF client**: SOAP requests for `GetStatus`, `AbsoluteMove`, `RelativeMove`, `Stop`.
- **Framing policy**: keeps primary target within a configurable deadband (e.g., center 30% of frame).
- **Velocity profiling**: limits pan/tilt/zoom acceleration to avoid jerk.
- **Zoom-predictive framing**: zooms to maintain target pixel size within bounds.

---

## 9. Data Flow

1. App initializes SDK with license key.
2. App creates `TrackingSession` with detector and tracking mode.
3. For each frame:
   a. `IO` module normalizes input to BGR/NV12/GPU handle.
   b. `DetectorBackend` produces detections.
   c. `TrackManager` predicts existing tracks.
   d. `DataAssociation` matches detections to tracks.
   e. `Tracker` updates correlation filters and scale for matched tracks.
   f. `TrackLifecycle` creates, confirms, or removes tracks.
   g. `PrimaryTargetSelector` chooses the track to follow.
   h. `PTZController` computes and sends pan/tilt/zoom commands.
   i. `Telemetry` records stage latencies and metrics.
4. App receives `TrackingOutput` with tracks, annotated frame, and metrics.

---

## 10. Performance & Benchmarking Strategy

### 10.1 Competitive Targets

| Competitor | Metric Target |
|------------|---------------|
| Ultralytics YOLO11n + ByteTrack | ≥ HOTA, ≥ FPS on same CPU |
| NVIDIA DeepStream NvDCF | ≤ latency, ≥ IDF1 on Jetson |
| OpenCV KCF/CSRT | ≥ robustness at equal FPS |

### 10.2 Benchmark Harness

A `benchmark/` module with:

- **Dataset loaders**: MOT15/17/20, DanceTrack, SportsMOT, custom PTZ sequences.
- **Metrics**: HOTA, MOTA, MOTP, IDF1, FP, FN, ID switches, fragmentations.
- **A/B registry**: Register tracker variants by name and run head-to-head.
- **Stage profiler**: Per-frame timing for detection, feature extraction, association, filter update, PTZ, telemetry.
- **Report generator**: Markdown/HTML report with tables and plots.

### 10.3 Optimization Levers

- Tracking mode presets tuned for speed vs. accuracy.
- Swappable detector backends and models.
- Swappable association costs.
- Optional Re-ID frequency (every frame vs. every N frames).
- Configurable scale pool size and displacement thresholds.

### 10.4 Validation Workflow

1. Run all variants on MOT17 train and a PTZ evaluation set.
2. Compare metrics and latency against baseline Ultralytics tracker.
3. Analyze failure cases (occlusion, fast motion, scale change).
4. Iterate on displacement, Re-ID, scale handling, or detector speed.
5. Lock the winning configuration as the default preset per platform.

---

## 11. Licensing

- **Local validation**: RSA-signed license keys decoded to `LicenseInfo` (feature flags, expiry, seat count).
- **Online activation**: HTTPS POST to activation server; server returns signed seat token.
- **Heartbeat**: Periodic online check for floating licenses; cached grace period for offline use.
- **Trial mode**: Watermarked output, 15-minute session cap, or reduced feature set until activated.
- **Seat enforcement**: One license per deployed device; deactivation API allows transfers.

---

## 12. Build, Packaging & Distribution

### 12.1 Build System

- CMake super-build generating:
  - `ultratrack` shared/static library.
  - Examples and tests.
  - Python bindings target (Phase 4).
- Dependency management via vcpkg or Conan for OpenCV, ONNX Runtime, spdlog, nlohmann/json, CURL.
- Remove hardcoded `OpenCV_DIR`; use `find_package` with optional override.

### 12.2 Packaging

| Platform | Format |
|----------|--------|
| Windows | MSI installer + NuGet (future) |
| Linux x64 | `.deb` and `.rpm` |
| Jetson | `.tar.gz` + Docker container |

### 12.3 Versioning

- Semantic versioning (currently 1.0.0).
- ABI compatibility policy: stable for minor releases, breaking only on major.

---

## 13. Security

- License keys and activation responses are RSA-signed; no private key in SDK.
- Telemetry upload is optional and encrypted (TLS).
- No credentials logged; ONVIF passwords stored only in memory during session.
- Input validation on all public API parameters.
- Optional ASLR/DEP-compliant builds on Windows.

---

## 14. Testing Strategy

| Layer | Coverage |
|-------|----------|
| Unit | Core geometry, Kalman, Hungarian, feature extractors, config validation. |
| Integration | Detector + tracker + association on short synthetic sequences. |
| End-to-end | Full SDK pipeline on MOT-format sequences with metric assertions. |
| Benchmark | Head-to-head against Ultralytics/YOLO tracker; nightly regression. |
| Hardware | PTZ ONVIF mock server; Jetson performance suite. |

---

## 15. Roadmap

### Phase 1 — SDK Foundation (6–8 weeks)
- Refactor existing code into `core`, `detector`, `tracker`, `reid`, `tracking_engine`.
- Implement multi-target lifecycle and data association.
- Define public C++ API and write examples.
- Replace hardcoded build paths; set up vcpkg/Conan.
- Add unit and integration tests.

### Phase 2 — PTZ & Production Hardening (4–6 weeks)
- ONVIF PTZ controller with framing policies.
- License manager + activation server.
- Structured logging and telemetry.
- Windows/Linux installers.

### Phase 3 — Jetson & TensorRT (4 weeks)
- TensorRT detector and Re-ID backends.
- Zero-copy NVMM/GPU input path.
- Jetson container and performance tuning.

### Phase 4 — Ecosystem (post-V1)
- Python and C# bindings.
- GUI configuration tool.
- VISCA/Pelco protocol additions.
- Cloud telemetry dashboard.

---

## 16. Open Questions

1. Do you have existing ONVIF cameras for PTZ integration testing?
2. Which exact Ultralytics model/backbone should be the primary comparison baseline (YOLO11n, YOLOv8n)?
3. Should the activation server be self-hosted or managed by a third party?
4. What is the target price/tier structure for trial vs. pro vs. enterprise?
5. Do you have custom PTZ evaluation footage, or should we create a recording protocol?

---

## 17. References

- Existing codebase: `src/ultratrack.cpp`, `include/ultratrack.hpp`
- Research paper: CIBDA 2025, "Displacement prediction strategy based on KCF in tracking"
- Prior specs: `docs/superpowers/specs/2026-03-27-ultratracker-performance-and-kcf-audit-design.md`
