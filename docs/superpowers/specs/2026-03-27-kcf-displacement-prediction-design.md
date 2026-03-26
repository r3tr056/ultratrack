# KCF Displacement Prediction Tracker - Design Specification

**Date:** 2026-03-27  
**Status:** Approved  
**Based on:** "Displacement prediction strategy based on KCF in tracking" (Guo, CIBDA 2025)

## 1. Overview

### 1.1 Goal

Implement the improved KCF tracking algorithm from the research paper, adding displacement prediction, 3-scale pool estimation, and multi-feature fusion (HOG + Gray + Color Names) to the existing UltraTracker system with configurable runtime modes.

### 1.2 Success Criteria

- Achieve ~84.5% precision and ~63.5% success rate on OTB100 benchmark (paper's results)
- Maintain ~92 FPS in ACCURATE mode, ~150 FPS in FAST mode
- Runtime switching between FAST/BALANCED/ACCURATE modes without restart
- All existing UltraTracker functionality continues to work

### 1.3 Non-Goals

- Deep learning-based tracking methods (out of scope)
- GPU-only implementation (CPU must work)
- Real-time re-detection after complete track loss

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        UltraTracker                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ DisplacementPred │  │   ScaleEstimator │  │ KCFCorrelation│ │
│  │    (new module)  │  │   (3-scale pool) │  │   (enhanced)  │ │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬───────┘ │
│           │                     │                     │         │
│           └─────────────────────┴─────────────────────┘         │
│                                 │                               │
│                    ┌────────────▼────────────┐                  │
│                    │   MultiFeatureExtractor │                  │
│                    │   (configurable)        │                  │
│                    └────────────┬────────────┘                  │
│           ┌─────────────────────┼─────────────────────┐         │
│           │                     │                     │         │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐ │
│  │  HOGFeature     │  │  GrayFeature    │  │   CNFeature     │ │
│  │  (31-dim FHOG)  │  │  (1-dim)        │  │  (10-dim)       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `MultiFeatureExtractor` | Orchestrates feature extraction based on tracking mode |
| `HOGFeature` | Extracts 31-dimensional FHOG gradient features |
| `GrayFeature` | Extracts 1-dimensional grayscale intensity |
| `CNFeature` | Extracts 10-dimensional Color Names probabilities |
| `DisplacementPredictor` | Predicts target position based on velocity history |
| `ScaleEstimator` | Estimates optimal scale using 3-scale pool |

## 3. Tracking Modes

| Mode | Features | Dimensions | Target FPS | Use Case |
|------|----------|------------|------------|----------|
| `FAST` | HOG | 31 | 150+ | Real-time, resource-constrained |
| `BALANCED` | HOG + Gray | 32 | 100-120 | General purpose |
| `ACCURATE` | HOG + Gray + CN | 42 | 90-100 | High accuracy requirements |

## 4. Component Specifications

### 4.1 Feature Extraction System

#### 4.1.1 Interface

```cpp
namespace ultratrack {

enum class TrackingMode {
    FAST,      // HOG only (31-dim)
    BALANCED,  // HOG + Gray (32-dim)
    ACCURATE   // HOG + Gray + CN (42-dim)
};

struct FeatureConfig {
    int cell_size = 4;        // HOG cell size in pixels
    int num_bins = 9;         // HOG orientation bins
    bool use_simd = true;     // Enable SIMD acceleration
};

class IFeatureExtractor {
public:
    virtual ~IFeatureExtractor() = default;
    virtual cv::Mat extract(const cv::Mat& patch) const = 0;
    virtual int dimensions() const = 0;
    virtual std::string name() const = 0;
};

class MultiFeatureExtractor {
public:
    explicit MultiFeatureExtractor(TrackingMode mode, 
                                   const FeatureConfig& config = {});
    cv::Mat extract(const cv::Mat& patch) const;
    int total_dimensions() const;
    void set_mode(TrackingMode mode);
    TrackingMode get_mode() const;
private:
    std::vector<std::unique_ptr<IFeatureExtractor>> extractors_;
    TrackingMode mode_;
    FeatureConfig config_;
};

}
```

#### 4.1.2 HOG Feature (31 dimensions)

- Uses FHOG variant with 9 orientation bins
- 4x4 pixel cells, 2x2 cell blocks
- Includes both signed and unsigned gradients
- Output: 31-dimensional feature vector per cell
- SIMD-accelerated gradient computation

#### 4.1.3 Gray Feature (1 dimension)

- Simple normalized grayscale intensity
- Range: [0, 1] after normalization
- Complements HOG by capturing absolute intensity

#### 4.1.4 Color Names Feature (10 dimensions)

- Maps RGB to 10 color name probabilities
- Colors: black, blue, brown, grey, green, orange, pink, purple, red, white
- Uses precomputed 32x32x32 lookup table (32KB)
- Lookup table stored in `data/cn_lookup.bin`

**Lookup Table Format:**

- Binary file: 32768 entries × 10 floats × 4 bytes = 1.31 MB
- Index formula: `idx = (r >> 3) * 1024 + (g >> 3) * 32 + (b >> 3)`
- Each entry: 10 consecutive floats representing color name probabilities
- Byte order: Little-endian (native x86/x64)

**Lookup Table Generation:**

```cpp
// Generates CN lookup if file not found
void CNFeature::generate_lookup_table() {
    // Based on van de Weijer et al. "Learning Color Names for Real-World Applications"
    // Maps 5-bit RGB (32^3 = 32768 entries) to 10 color probabilities
    for (int r = 0; r < 32; r++) {
        for (int g = 0; g < 32; g++) {
            for (int b = 0; b < 32; b++) {
                int idx = r * 1024 + g * 32 + b;
                lookup_table_[idx] = compute_cn_probabilities(r*8, g*8, b*8);
            }
        }
    }
}
```

### 4.2 Displacement Prediction

#### 4.2.1 Algorithm (Paper Equation 21)

```
P_t = P_{t-1} + κ(P_{t-1} - P_{t-2}),  if μ₁ < |P_{t-1} - P_{t-2}| < μ₂
P_t = P_{t-1},                          otherwise
```

Where:
- `P_t` = predicted position at frame t
- `κ` = velocity coefficient (default: 0.8)
- `μ₁` = minimum displacement threshold (default: 2.0 pixels)
- `μ₂` = maximum displacement threshold (default: 100.0 pixels)

#### 4.2.2 Interface

```cpp
namespace ultratrack {

struct DisplacementConfig {
    float kappa = 0.8f;           // Velocity coefficient
    float min_threshold = 2.0f;   // μ₁ - ignore tiny movements (pixels)
    float max_threshold = 100.0f; // μ₂ - ignore extreme jumps (pixels)
    bool enabled = true;
};

class DisplacementPredictor {
public:
    explicit DisplacementPredictor(const DisplacementConfig& config = {});
    cv::Point2f predict(const cv::Point2f& current_pos);
    void reset();
    void set_config(const DisplacementConfig& config);
    DisplacementConfig get_config() const;
    cv::Point2f get_velocity() const;
private:
    DisplacementConfig config_;
    cv::Point2f pos_history_[2];  // [0] = t-2, [1] = t-1
    int history_count_ = 0;
    cv::Point2f last_velocity_;
};

}
```

#### 4.2.3 State Machine

```
INIT (history_count=0)
    │
    ▼ first position
SINGLE (history_count=1)
    │
    ▼ second position
READY (history_count=2) ◄──┐
    │                      │
    ▼ predict & update     │
READY ─────────────────────┘
    │
    ▼ reset()
INIT
```

### 4.3 Scale Estimation

#### 4.3.1 Scale Pool (Paper Equation 23)

```
S_pool = {1 - 2δ, 1, 1 + 2δ}
```

Where δ (scale_step) defaults to 0.04 (4%).

Example: `{0.92, 1.0, 1.08}`

#### 4.3.2 Interface

```cpp
namespace ultratrack {

struct ScaleConfig {
    float scale_step = 0.04f;     // δ - scale increment
    int num_scales = 3;           // 3 for fast, 5 or 7 for accurate
    float min_scale = 0.2f;       // Minimum allowed scale
    float max_scale = 5.0f;       // Maximum allowed scale
    float scale_penalty = 0.975f; // Penalize scale changes
    bool use_scale_filter = true;
};

class ScaleEstimator {
public:
    explicit ScaleEstimator(const ScaleConfig& config = {});
    float estimate(const cv::Mat& frame,
                   const cv::Point2f& position,
                   const cv::Size2f& base_size,
                   const cv::Mat& correlation_filter);
    std::vector<float> get_scale_pool() const;
    void set_config(const ScaleConfig& config);
    ScaleConfig get_config() const;
    float get_current_scale() const;
    void reset();
private:
    ScaleConfig config_;
    float current_scale_ = 1.0f;
    std::vector<float> scale_pool_;
    void rebuild_scale_pool();
    float compute_scale_response(const cv::Mat& frame,
                                  const cv::Point2f& position,
                                  const cv::Size2f& size,
                                  float scale,
                                  const cv::Mat& correlation_filter);
};

}
```

#### 4.3.3 Scale Estimation Algorithm

```
For each scale s in scale_pool:
    1. Compute test_scale = current_scale * s
    2. Clamp to [min_scale, max_scale]
    3. Extract patch at test_scale
    4. Compute correlation response
    5. Apply scale_penalty if s != 1.0
    6. Track best response and scale

Update current_scale = best_scale
Return current_scale
```

### 4.4 Multi-Channel Correlation Filter

#### 4.4.1 Multi-Channel Gaussian Kernel (Paper Equation 18)

```cpp
cv::Mat compute_multi_channel_kernel(const cv::Mat& x, const cv::Mat& z, float sigma) {
    // x, z: [H x W x C] multi-channel feature maps
    int num_channels = x.channels();
    float xx = 0, zz = 0;
    cv::Mat xz_sum = cv::Mat::zeros(x.rows, x.cols, CV_32FC2);
    
    for (int c = 0; c < num_channels; c++) {
        cv::Mat x_c, z_c;
        cv::extractChannel(x, x_c, c);
        cv::extractChannel(z, z_c, c);
        
        xx += cv::sum(x_c.mul(x_c))[0];
        zz += cv::sum(z_c.mul(z_c))[0];
        
        cv::Mat xf, zf, xzf;
        cv::dft(x_c, xf, cv::DFT_COMPLEX_OUTPUT);
        cv::dft(z_c, zf, cv::DFT_COMPLEX_OUTPUT);
        cv::mulSpectrums(xf, zf, xzf, 0, true);
        xz_sum += xzf;
    }
    
    cv::Mat xz;
    cv::idft(xz_sum, xz, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    
    cv::Mat kernel;
    float denom = sigma * sigma * x.total() * num_channels;
    cv::exp(-(xx + zz - 2 * xz) / denom, kernel);
    return kernel;
}
```

#### 4.4.2 Filter Training (Paper Equation 13)

```
α̂ = ŷ / (k̂_xx + λ)
```

Where:
- `ŷ` = DFT of Gaussian target
- `k̂_xx` = DFT of kernel autocorrelation
- `λ` = regularization parameter (default: 0.01)

#### 4.4.3 Filter Update

```
α = (1 - η) * α_old + η * α_new
```

Where η = learning_rate (default: 0.01)

### 4.5 UltraTracker Integration

#### 4.5.1 Modified Track Struct

```cpp
struct Track {
    // Existing fields
    unsigned long long id;
    cv::Rect2f bbox;
    cv::Mat state;
    cv::Mat covariance;
    cv::Mat correlation_filter;
    cv::Mat appearance_model;
    float confidence;
    int age;
    int hits;
    int time_since_update;
    bool is_activated;
    
    // NEW fields
    std::unique_ptr<DisplacementPredictor> displacement_predictor;
    cv::Point2f predicted_center;
    float current_scale = 1.0f;
    cv::Size2f base_size;              // Original target size
    cv::Mat multi_channel_filter;      // Multi-channel correlation filter
};
```

#### 4.5.2 Modified TrackerConfig

```cpp
struct TrackerConfig {
    TrackingMode mode = TrackingMode::ACCURATE;
    DisplacementConfig displacement;
    ScaleConfig scale;
    FeatureConfig feature;
    float learning_rate = 0.01f;
    float lambda = 0.01f;
    float sigma = 2.0f;
};
```

#### 4.5.3 Modified UltraTracker Class

```cpp
class UltraTracker {
public:
    // Existing constructors + new one
    UltraTracker(const std::string& model_path,
                 const TrackerConfig& config = {});
    
    // NEW methods
    void set_tracking_mode(TrackingMode mode);
    TrackingMode get_tracking_mode() const;
    void set_config(const TrackerConfig& config);
    TrackerConfig get_config() const;

private:
    // NEW members
    TrackerConfig config_;
    std::unique_ptr<MultiFeatureExtractor> feature_extractor_;
    std::unique_ptr<ScaleEstimator> scale_estimator_;
    
    // NEW methods
    cv::Mat create_multi_channel_filter(const cv::Mat& patch);
    cv::Mat track_with_displacement_prediction(Track& track, const cv::Mat& frame);
    void update_track_scale(Track& track, const cv::Mat& frame);
    cv::Mat extract_patch_at_scale(const cv::Mat& frame,
                                    const cv::Point2f& center,
                                    const cv::Size2f& base_size,
                                    float scale);
};
```

#### 4.5.4 Updated Tracking Flow

```
update(frame, detections):
    1. For each active track:
       a. Displacement prediction → predicted_center
       b. Extract search patch at predicted_center
       c. Multi-feature extraction (based on mode)
       d. Compute correlation response
       e. Find peak → refined position
       f. Scale estimation (3-scale pool)
       g. Update track bbox with new position and scale
    
    2. Associate tracks with detections (Hungarian algorithm)
    
    3. For matched tracks:
       a. Update Kalman filter
       b. Update correlation filter with learning rate
    
    4. Create new tracks for unmatched detections
    
    5. Remove lost tracks
```

## 5. Error Handling

### 5.1 Error Codes

```cpp
enum class ErrorCode {
    OK = 0,
    INVALID_PATCH_SIZE,
    EMPTY_FRAME,
    FEATURE_EXTRACTION_FAILED,
    SCALE_ESTIMATION_FAILED,
    CN_LOOKUP_LOAD_FAILED,
    INVALID_CONFIG
};

class TrackerException : public std::runtime_error {
public:
    TrackerException(ErrorCode code, const std::string& msg);
    ErrorCode code() const;
private:
    ErrorCode code_;
};
```

### 5.2 Validation Functions

```cpp
void validate_patch(const cv::Mat& patch, const std::string& context);
void validate_config(const TrackerConfig& config);
```

### 5.3 Graceful Degradation

- If CN lookup fails to load: Fall back to BALANCED mode (HOG + Gray)
- If feature extraction fails: Skip frame, use Kalman prediction only
- If scale estimation fails: Keep previous scale

## 6. File Structure

```
ultratrack/
├── CMakeLists.txt              # Modified
├── include/
│   ├── ultratrack.hpp          # Modified
│   ├── feature_extractor.hpp   # NEW
│   ├── errors.hpp              # NEW
│   ├── features/
│   │   ├── hog_feature.hpp     # NEW
│   │   ├── gray_feature.hpp    # NEW
│   │   └── cn_feature.hpp      # NEW
│   └── tracking/
│       ├── displacement_predictor.hpp  # NEW
│       └── scale_estimator.hpp         # NEW
├── src/
│   ├── main.cpp
│   ├── ultratrack.cpp          # Modified
│   ├── simd_optimization.cpp   # Modified
│   ├── features/
│   │   ├── feature_extractor.cpp  # NEW
│   │   ├── hog_feature.cpp        # NEW
│   │   ├── gray_feature.cpp       # NEW
│   │   └── cn_feature.cpp         # NEW
│   └── tracking/
│       ├── displacement_predictor.cpp  # NEW
│       └── scale_estimator.cpp         # NEW
├── data/
│   └── cn_lookup.bin           # NEW
└── tests/
    ├── test_features.cpp       # NEW
    ├── test_displacement.cpp   # NEW
    ├── test_scale.cpp          # NEW
    ├── test_tracker.cpp        # NEW
    └── benchmark.cpp           # NEW
```

## 7. Testing Strategy

### 7.1 Unit Tests

| Test File | Coverage |
|-----------|----------|
| `test_features.cpp` | HOG, Gray, CN extraction; multi-feature fusion |
| `test_displacement.cpp` | Prediction accuracy; edge cases; reset behavior |
| `test_scale.cpp` | Scale pool generation; estimation accuracy |
| `test_tracker.cpp` | Full pipeline integration |

### 7.2 Benchmark Tests

| Test | Metric |
|------|--------|
| FPS by mode | FAST > 150, BALANCED > 100, ACCURATE > 90 |
| Precision | Compare against paper's 84.5% |
| Success rate | Compare against paper's 63.5% |

### 7.3 Sample Test Cases

```cpp
// Displacement prediction - linear motion
TEST(DisplacementPredictor, PredictLinearMotion) {
    DisplacementPredictor pred;
    pred.predict({100, 100});
    pred.predict({110, 100});
    auto predicted = pred.predict({120, 100});
    EXPECT_NEAR(predicted.x, 128, 2);  // 120 + 0.8*10
    EXPECT_NEAR(predicted.y, 100, 1);
}

// Scale pool generation
TEST(ScaleEstimator, ThreeScalePool) {
    ScaleConfig config{.scale_step = 0.04f, .num_scales = 3};
    ScaleEstimator est(config);
    auto pool = est.get_scale_pool();
    ASSERT_EQ(pool.size(), 3);
    EXPECT_FLOAT_EQ(pool[0], 0.92f);
    EXPECT_FLOAT_EQ(pool[1], 1.00f);
    EXPECT_FLOAT_EQ(pool[2], 1.08f);
}

// Multi-feature dimensions
TEST(MultiFeatureExtractor, AccurateModeHas42Dims) {
    MultiFeatureExtractor ext(TrackingMode::ACCURATE);
    EXPECT_EQ(ext.total_dimensions(), 42);
}
```

## 8. Build System Changes

### 8.1 CMakeLists.txt Additions

```cmake
# New source files
set(FEATURE_SOURCES
    ${SRC_DIR}/features/feature_extractor.cpp
    ${SRC_DIR}/features/hog_feature.cpp
    ${SRC_DIR}/features/gray_feature.cpp
    ${SRC_DIR}/features/cn_feature.cpp
)

set(TRACKING_SOURCES
    ${SRC_DIR}/tracking/displacement_predictor.cpp
    ${SRC_DIR}/tracking/scale_estimator.cpp
)

# Add to SOURCES
set(SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/ultratrack.cpp
    ${SRC_DIR}/simd_optimization.cpp
    ${FEATURE_SOURCES}
    ${TRACKING_SOURCES}
)

# Copy CN lookup table
configure_file(
    ${CMAKE_SOURCE_DIR}/data/cn_lookup.bin
    ${CMAKE_BINARY_DIR}/Release/data/cn_lookup.bin
    COPYONLY
)

# Optional testing
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
    add_executable(ultratrack_tests
        tests/test_features.cpp
        tests/test_displacement.cpp
        tests/test_scale.cpp
        tests/test_tracker.cpp
        ${FEATURE_SOURCES}
        ${TRACKING_SOURCES}
    )
    target_link_libraries(ultratrack_tests GTest::gtest_main ${OpenCV_LIBS})
    gtest_discover_tests(ultratrack_tests)
endif()
```

## 9. Expected Performance

| Metric | KCF (baseline) | SAMF | This Implementation |
|--------|----------------|------|---------------------|
| Speed (FPS) | 404 | 48 | 92 (ACCURATE) |
| Precision | 78.0% | 84.3% | 84.5% |
| Success Rate | 52.1% | 64.6% | 63.5% |

## 10. References

1. Guo, S. (2025). "Displacement prediction strategy based on KCF in tracking." CIBDA 2025.
2. Henriques, J.F. et al. (2014). "High-speed tracking with kernelized correlation filters." TPAMI.
3. Li, Y. & Zhu, J. (2014). "A scale adaptive kernel correlation filter tracker with feature integration." ECCV Workshops.
4. van de Weijer, J. et al. (2009). "Learning Color Names for Real-World Applications." TIP.
