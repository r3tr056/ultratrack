# KCF Displacement Prediction Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the improved KCF tracking algorithm with displacement prediction, 3-scale pool, and multi-feature fusion (HOG+Gray+CN).

**Architecture:** Modular feature extraction pipeline with pluggable extractors, separate displacement predictor and scale estimator modules, integrated into existing UltraTracker class with runtime mode switching.

**Tech Stack:** C++17, OpenCV 4.x, SIMD (AVX2/SSE4.2/NEON)

---

## File Structure Overview

```
include/
  errors.hpp                    # Task 1
  feature_extractor.hpp         # Task 2
  features/
    hog_feature.hpp             # Task 3
    gray_feature.hpp            # Task 4
    cn_feature.hpp              # Task 5
  tracking/
    displacement_predictor.hpp  # Task 6
    scale_estimator.hpp         # Task 7
  ultratrack.hpp                # Task 9 (modify)

src/
  features/
    feature_extractor.cpp       # Task 2
    hog_feature.cpp             # Task 3
    gray_feature.cpp            # Task 4
    cn_feature.cpp              # Task 5
  tracking/
    displacement_predictor.cpp  # Task 6
    scale_estimator.cpp         # Task 7
  ultratrack.cpp                # Task 9 (modify)

data/
  cn_lookup.bin                 # Task 5 (generated)

tests/
  test_displacement.cpp         # Task 6
  test_scale.cpp                # Task 7
  test_features.cpp             # Task 8
  test_tracker.cpp              # Task 10

CMakeLists.txt                  # Task 11 (modify)
```

---

## Task 1: Create Error Handling Infrastructure

**Files:**
- Create: `include/errors.hpp`

- [ ] **Step 1: Create the errors header file**

```cpp
// include/errors.hpp
#ifndef ULTRATRACK_ERRORS_HPP
#define ULTRATRACK_ERRORS_HPP

#include <stdexcept>
#include <string>

namespace ultratrack {

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
    TrackerException(ErrorCode code, const std::string& msg)
        : std::runtime_error(msg), code_(code) {}
    
    ErrorCode code() const { return code_; }
    
private:
    ErrorCode code_;
};

inline void validate_patch(const cv::Mat& patch, const std::string& context) {
    if (patch.empty()) {
        throw TrackerException(ErrorCode::EMPTY_FRAME, 
            context + ": patch is empty");
    }
    if (patch.cols < 4 || patch.rows < 4) {
        throw TrackerException(ErrorCode::INVALID_PATCH_SIZE,
            context + ": patch too small (min 4x4)");
    }
}

} // namespace ultratrack

#endif // ULTRATRACK_ERRORS_HPP
```

- [ ] **Step 2: Verify file compiles**

Run: `cl /c /std:c++17 /I"include" /I"C:/Users/Ankur/sdk/opencv/build/include" include/errors.hpp /Fo:NUL` (Windows) or create a simple test compile.

Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add include/errors.hpp
git commit -m "feat: add error handling infrastructure for tracker"
```

---

## Task 2: Create Feature Extractor Interface and Factory

**Files:**
- Create: `include/feature_extractor.hpp`
- Create: `src/features/feature_extractor.cpp`

- [ ] **Step 1: Create the feature extractor header**

```cpp
// include/feature_extractor.hpp
#ifndef ULTRATRACK_FEATURE_EXTRACTOR_HPP
#define ULTRATRACK_FEATURE_EXTRACTOR_HPP

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>

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
    explicit MultiFeatureExtractor(TrackingMode mode = TrackingMode::ACCURATE,
                                   const FeatureConfig& config = {});
    ~MultiFeatureExtractor();
    
    cv::Mat extract(const cv::Mat& patch) const;
    int total_dimensions() const;
    void set_mode(TrackingMode mode);
    TrackingMode get_mode() const { return mode_; }
    const FeatureConfig& get_config() const { return config_; }

private:
    void rebuild_extractors();
    
    std::vector<std::unique_ptr<IFeatureExtractor>> extractors_;
    TrackingMode mode_;
    FeatureConfig config_;
};

} // namespace ultratrack

#endif // ULTRATRACK_FEATURE_EXTRACTOR_HPP
```

- [ ] **Step 2: Create the feature extractor implementation**

```cpp
// src/features/feature_extractor.cpp
#include "feature_extractor.hpp"
#include "features/hog_feature.hpp"
#include "features/gray_feature.hpp"
#include "features/cn_feature.hpp"
#include "errors.hpp"

namespace ultratrack {

MultiFeatureExtractor::MultiFeatureExtractor(TrackingMode mode, const FeatureConfig& config)
    : mode_(mode), config_(config) {
    rebuild_extractors();
}

MultiFeatureExtractor::~MultiFeatureExtractor() = default;

void MultiFeatureExtractor::rebuild_extractors() {
    extractors_.clear();
    
    // HOG is always included
    extractors_.push_back(std::make_unique<HOGFeature>(config_));
    
    if (mode_ == TrackingMode::BALANCED || mode_ == TrackingMode::ACCURATE) {
        extractors_.push_back(std::make_unique<GrayFeature>());
    }
    
    if (mode_ == TrackingMode::ACCURATE) {
        extractors_.push_back(std::make_unique<CNFeature>());
    }
}

cv::Mat MultiFeatureExtractor::extract(const cv::Mat& patch) const {
    validate_patch(patch, "MultiFeatureExtractor::extract");
    
    std::vector<cv::Mat> features;
    features.reserve(extractors_.size());
    
    for (const auto& extractor : extractors_) {
        cv::Mat feat = extractor->extract(patch);
        if (!feat.empty()) {
            features.push_back(feat);
        }
    }
    
    if (features.empty()) {
        return cv::Mat();
    }
    
    // Concatenate all features along channel dimension
    cv::Mat result;
    cv::merge(features, result);
    return result;
}

int MultiFeatureExtractor::total_dimensions() const {
    int total = 0;
    for (const auto& extractor : extractors_) {
        total += extractor->dimensions();
    }
    return total;
}

void MultiFeatureExtractor::set_mode(TrackingMode mode) {
    if (mode_ != mode) {
        mode_ = mode;
        rebuild_extractors();
    }
}

} // namespace ultratrack
```

- [ ] **Step 3: Commit**

```bash
git add include/feature_extractor.hpp src/features/feature_extractor.cpp
git commit -m "feat: add feature extractor interface and multi-feature factory"
```

---

## Task 3: Implement HOG Feature Extractor

**Files:**
- Create: `include/features/hog_feature.hpp`
- Create: `src/features/hog_feature.cpp`

- [ ] **Step 1: Create HOG feature header**

```cpp
// include/features/hog_feature.hpp
#ifndef ULTRATRACK_HOG_FEATURE_HPP
#define ULTRATRACK_HOG_FEATURE_HPP

#include "feature_extractor.hpp"

namespace ultratrack {

class HOGFeature : public IFeatureExtractor {
public:
    explicit HOGFeature(const FeatureConfig& config = {});
    ~HOGFeature() override = default;
    
    cv::Mat extract(const cv::Mat& patch) const override;
    int dimensions() const override { return 31; }
    std::string name() const override { return "HOG"; }

private:
    FeatureConfig config_;
    
    void compute_gradients(const cv::Mat& gray, cv::Mat& magnitude, cv::Mat& orientation) const;
    cv::Mat compute_hog_cells(const cv::Mat& magnitude, const cv::Mat& orientation, 
                               int cell_size, int num_bins) const;
};

} // namespace ultratrack

#endif // ULTRATRACK_HOG_FEATURE_HPP
```

- [ ] **Step 2: Create HOG feature implementation**

```cpp
// src/features/hog_feature.cpp
#include "features/hog_feature.hpp"
#include "errors.hpp"
#include <cmath>

namespace ultratrack {

HOGFeature::HOGFeature(const FeatureConfig& config) : config_(config) {}

cv::Mat HOGFeature::extract(const cv::Mat& patch) const {
    validate_patch(patch, "HOGFeature::extract");
    
    // Convert to grayscale if needed
    cv::Mat gray;
    if (patch.channels() == 3) {
        cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = patch.clone();
    }
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);
    
    // Compute gradients
    cv::Mat magnitude, orientation;
    compute_gradients(gray, magnitude, orientation);
    
    // Compute HOG cells
    cv::Mat hog = compute_hog_cells(magnitude, orientation, config_.cell_size, config_.num_bins);
    
    return hog;
}

void HOGFeature::compute_gradients(const cv::Mat& gray, cv::Mat& magnitude, cv::Mat& orientation) const {
    cv::Mat gx, gy;
    
    // Sobel gradients
    cv::Sobel(gray, gx, CV_32F, 1, 0, 1);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 1);
    
    // Magnitude and orientation
    cv::cartToPolar(gx, gy, magnitude, orientation, true);  // degrees
}

cv::Mat HOGFeature::compute_hog_cells(const cv::Mat& magnitude, const cv::Mat& orientation,
                                       int cell_size, int num_bins) const {
    int cells_x = magnitude.cols / cell_size;
    int cells_y = magnitude.rows / cell_size;
    
    if (cells_x < 1 || cells_y < 1) {
        return cv::Mat();
    }
    
    // FHOG uses 18 contrast-sensitive + 9 contrast-insensitive + 4 texture = 31 dims
    // Simplified: 9 unsigned + 18 signed + 4 = 31
    const int feat_dim = 31;
    cv::Mat features = cv::Mat::zeros(cells_y, cells_x, CV_32FC(feat_dim));
    
    float bin_width = 360.0f / num_bins;
    
    for (int cy = 0; cy < cells_y; cy++) {
        for (int cx = 0; cx < cells_x; cx++) {
            std::vector<float> hist(feat_dim, 0.0f);
            
            // Accumulate over cell
            for (int py = 0; py < cell_size; py++) {
                for (int px = 0; px < cell_size; px++) {
                    int y = cy * cell_size + py;
                    int x = cx * cell_size + px;
                    
                    if (y >= magnitude.rows || x >= magnitude.cols) continue;
                    
                    float mag = magnitude.at<float>(y, x);
                    float angle = orientation.at<float>(y, x);
                    
                    // Unsigned bin (0-180)
                    int bin_unsigned = static_cast<int>(fmod(angle, 180.0f) / (180.0f / num_bins)) % num_bins;
                    hist[bin_unsigned] += mag;
                    
                    // Signed bin (0-360) 
                    int bin_signed = static_cast<int>(angle / bin_width) % (num_bins * 2);
                    hist[num_bins + bin_signed] += mag;
                }
            }
            
            // Texture features (gradients of histogram)
            float texture_sum = 0;
            for (int i = 0; i < num_bins; i++) {
                texture_sum += hist[i];
            }
            hist[27] = texture_sum / (cell_size * cell_size);
            hist[28] = hist[27] * hist[27];
            hist[29] = std::sqrt(hist[27]);
            hist[30] = hist[27] > 0.1f ? 1.0f : 0.0f;
            
            // Normalize
            float norm = 0;
            for (int i = 0; i < feat_dim; i++) norm += hist[i] * hist[i];
            norm = std::sqrt(norm) + 1e-5f;
            for (int i = 0; i < feat_dim; i++) hist[i] /= norm;
            
            // Store
            float* ptr = features.ptr<float>(cy, cx);
            for (int i = 0; i < feat_dim; i++) {
                ptr[i] = hist[i];
            }
        }
    }
    
    return features;
}

} // namespace ultratrack
```

- [ ] **Step 3: Commit**

```bash
git add include/features/hog_feature.hpp src/features/hog_feature.cpp
git commit -m "feat: implement HOG feature extractor (31-dim FHOG)"
```

---

## Task 4: Implement Gray Feature Extractor

**Files:**
- Create: `include/features/gray_feature.hpp`
- Create: `src/features/gray_feature.cpp`

- [ ] **Step 1: Create Gray feature header**

```cpp
// include/features/gray_feature.hpp
#ifndef ULTRATRACK_GRAY_FEATURE_HPP
#define ULTRATRACK_GRAY_FEATURE_HPP

#include "feature_extractor.hpp"

namespace ultratrack {

class GrayFeature : public IFeatureExtractor {
public:
    GrayFeature() = default;
    ~GrayFeature() override = default;
    
    cv::Mat extract(const cv::Mat& patch) const override;
    int dimensions() const override { return 1; }
    std::string name() const override { return "Gray"; }
};

} // namespace ultratrack

#endif // ULTRATRACK_GRAY_FEATURE_HPP
```

- [ ] **Step 2: Create Gray feature implementation**

```cpp
// src/features/gray_feature.cpp
#include "features/gray_feature.hpp"
#include "errors.hpp"

namespace ultratrack {

cv::Mat GrayFeature::extract(const cv::Mat& patch) const {
    validate_patch(patch, "GrayFeature::extract");
    
    cv::Mat gray;
    if (patch.channels() == 3) {
        cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = patch.clone();
    }
    
    // Normalize to [0, 1]
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);
    
    // Subtract mean for zero-centering
    cv::Scalar mean = cv::mean(gray);
    gray -= mean[0];
    
    return gray;
}

} // namespace ultratrack
```

- [ ] **Step 3: Commit**

```bash
git add include/features/gray_feature.hpp src/features/gray_feature.cpp
git commit -m "feat: implement grayscale feature extractor (1-dim)"
```

---

## Task 5: Implement Color Names Feature Extractor

**Files:**
- Create: `include/features/cn_feature.hpp`
- Create: `src/features/cn_feature.cpp`
- Create: `data/` directory (lookup table generated at runtime if not present)

- [ ] **Step 1: Create CN feature header**

```cpp
// include/features/cn_feature.hpp
#ifndef ULTRATRACK_CN_FEATURE_HPP
#define ULTRATRACK_CN_FEATURE_HPP

#include "feature_extractor.hpp"
#include <array>

namespace ultratrack {

class CNFeature : public IFeatureExtractor {
public:
    CNFeature();
    ~CNFeature() override = default;
    
    cv::Mat extract(const cv::Mat& patch) const override;
    int dimensions() const override { return 10; }
    std::string name() const override { return "ColorNames"; }
    
    // Color name indices
    enum ColorName {
        BLACK = 0, BLUE, BROWN, GREY, GREEN,
        ORANGE, PINK, PURPLE, RED, WHITE
    };

private:
    static constexpr int LOOKUP_SIZE = 32768;  // 32^3
    std::array<std::array<float, 10>, LOOKUP_SIZE> lookup_table_;
    bool lookup_loaded_ = false;
    
    void load_or_generate_lookup();
    bool load_lookup_from_file(const std::string& path);
    void generate_lookup_table();
    void save_lookup_to_file(const std::string& path) const;
    std::array<float, 10> compute_cn_probabilities(int r, int g, int b) const;
};

} // namespace ultratrack

#endif // ULTRATRACK_CN_FEATURE_HPP
```

- [ ] **Step 2: Create CN feature implementation**

```cpp
// src/features/cn_feature.cpp
#include "features/cn_feature.hpp"
#include "errors.hpp"
#include <fstream>
#include <cmath>
#include <filesystem>

namespace ultratrack {

CNFeature::CNFeature() {
    load_or_generate_lookup();
}

void CNFeature::load_or_generate_lookup() {
    // Try multiple paths
    std::vector<std::string> paths = {
        "data/cn_lookup.bin",
        "../data/cn_lookup.bin",
        "../../data/cn_lookup.bin"
    };
    
    for (const auto& path : paths) {
        if (load_lookup_from_file(path)) {
            lookup_loaded_ = true;
            return;
        }
    }
    
    // Generate if not found
    generate_lookup_table();
    lookup_loaded_ = true;
    
    // Try to save for next time
    std::filesystem::create_directories("data");
    save_lookup_to_file("data/cn_lookup.bin");
}

bool CNFeature::load_lookup_from_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.read(reinterpret_cast<char*>(lookup_table_.data()), 
              LOOKUP_SIZE * 10 * sizeof(float));
    
    return file.good();
}

void CNFeature::save_lookup_to_file(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return;
    
    file.write(reinterpret_cast<const char*>(lookup_table_.data()),
               LOOKUP_SIZE * 10 * sizeof(float));
}

void CNFeature::generate_lookup_table() {
    // Generate CN lookup based on van de Weijer et al.
    for (int r = 0; r < 32; r++) {
        for (int g = 0; g < 32; g++) {
            for (int b = 0; b < 32; b++) {
                int idx = r * 1024 + g * 32 + b;
                lookup_table_[idx] = compute_cn_probabilities(r * 8, g * 8, b * 8);
            }
        }
    }
}

std::array<float, 10> CNFeature::compute_cn_probabilities(int r, int g, int b) const {
    // Simplified color name assignment based on heuristics
    // Full implementation would use learned probabilities from van de Weijer
    std::array<float, 10> probs = {0};
    
    float rf = r / 255.0f;
    float gf = g / 255.0f;
    float bf = b / 255.0f;
    
    float max_rgb = std::max({rf, gf, bf});
    float min_rgb = std::min({rf, gf, bf});
    float chroma = max_rgb - min_rgb;
    float lightness = (max_rgb + min_rgb) / 2.0f;
    
    // Black
    if (lightness < 0.15f) {
        probs[BLACK] = 1.0f - lightness / 0.15f;
    }
    // White
    if (lightness > 0.85f && chroma < 0.1f) {
        probs[WHITE] = (lightness - 0.85f) / 0.15f;
    }
    // Grey
    if (chroma < 0.15f && lightness > 0.15f && lightness < 0.85f) {
        probs[GREY] = 1.0f - chroma / 0.15f;
    }
    
    // Chromatic colors
    if (chroma > 0.1f) {
        float hue = 0;
        if (max_rgb == rf) {
            hue = 60.0f * fmod((gf - bf) / chroma, 6.0f);
        } else if (max_rgb == gf) {
            hue = 60.0f * ((bf - rf) / chroma + 2.0f);
        } else {
            hue = 60.0f * ((rf - gf) / chroma + 4.0f);
        }
        if (hue < 0) hue += 360.0f;
        
        float sat_weight = std::min(1.0f, chroma / 0.5f);
        
        // Assign based on hue ranges
        if (hue < 15 || hue >= 345) probs[RED] = sat_weight;
        else if (hue < 45) probs[ORANGE] = sat_weight;
        else if (hue < 75) { probs[ORANGE] = sat_weight * 0.5f; probs[BROWN] = sat_weight * 0.5f; }
        else if (hue < 165) probs[GREEN] = sat_weight;
        else if (hue < 195) { probs[GREEN] = sat_weight * 0.5f; probs[BLUE] = sat_weight * 0.5f; }
        else if (hue < 255) probs[BLUE] = sat_weight;
        else if (hue < 285) probs[PURPLE] = sat_weight;
        else if (hue < 330) probs[PINK] = sat_weight;
        else probs[RED] = sat_weight;
        
        // Brown adjustment
        if (lightness < 0.4f && (hue > 15 && hue < 45)) {
            probs[BROWN] = sat_weight * (0.4f - lightness) / 0.4f;
            probs[ORANGE] *= lightness / 0.4f;
        }
    }
    
    // Normalize
    float sum = 0;
    for (float p : probs) sum += p;
    if (sum > 0) {
        for (float& p : probs) p /= sum;
    } else {
        probs[GREY] = 1.0f;  // Default to grey
    }
    
    return probs;
}

cv::Mat CNFeature::extract(const cv::Mat& patch) const {
    validate_patch(patch, "CNFeature::extract");
    
    if (!lookup_loaded_) {
        throw TrackerException(ErrorCode::CN_LOOKUP_LOAD_FAILED,
            "Color Names lookup table not loaded");
    }
    
    cv::Mat bgr;
    if (patch.channels() == 1) {
        cv::cvtColor(patch, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = patch;
    }
    
    // Output: 10-channel feature map
    cv::Mat result(patch.rows, patch.cols, CV_32FC(10));
    
    for (int y = 0; y < bgr.rows; y++) {
        const cv::Vec3b* row_in = bgr.ptr<cv::Vec3b>(y);
        float* row_out = result.ptr<float>(y);
        
        for (int x = 0; x < bgr.cols; x++) {
            int b = row_in[x][0] >> 3;  // 5-bit quantization
            int g = row_in[x][1] >> 3;
            int r = row_in[x][2] >> 3;
            
            int idx = r * 1024 + g * 32 + b;
            const auto& probs = lookup_table_[idx];
            
            for (int c = 0; c < 10; c++) {
                row_out[x * 10 + c] = probs[c];
            }
        }
    }
    
    return result;
}

} // namespace ultratrack
```

- [ ] **Step 3: Commit**

```bash
git add include/features/cn_feature.hpp src/features/cn_feature.cpp
git commit -m "feat: implement Color Names feature extractor (10-dim)"
```

---

## Task 6: Implement Displacement Predictor

**Files:**
- Create: `include/tracking/displacement_predictor.hpp`
- Create: `src/tracking/displacement_predictor.cpp`
- Create: `tests/test_displacement.cpp`

- [ ] **Step 1: Create displacement predictor header**

```cpp
// include/tracking/displacement_predictor.hpp
#ifndef ULTRATRACK_DISPLACEMENT_PREDICTOR_HPP
#define ULTRATRACK_DISPLACEMENT_PREDICTOR_HPP

#include <opencv2/core.hpp>

namespace ultratrack {

struct DisplacementConfig {
    float kappa = 0.8f;           // Velocity coefficient
    float min_threshold = 2.0f;   // Minimum displacement (pixels)
    float max_threshold = 100.0f; // Maximum displacement (pixels)
    bool enabled = true;
};

class DisplacementPredictor {
public:
    explicit DisplacementPredictor(const DisplacementConfig& config = {});
    
    // Predict next position and update history
    cv::Point2f predict(const cv::Point2f& current_pos);
    
    // Reset history
    void reset();
    
    // Configuration
    void set_config(const DisplacementConfig& config);
    DisplacementConfig get_config() const { return config_; }
    
    // Get last computed velocity
    cv::Point2f get_velocity() const { return last_velocity_; }
    
    // Check if predictor has enough history
    bool is_ready() const { return history_count_ >= 2; }

private:
    DisplacementConfig config_;
    cv::Point2f pos_history_[2];  // [0] = t-2, [1] = t-1
    int history_count_ = 0;
    cv::Point2f last_velocity_;
};

} // namespace ultratrack

#endif // ULTRATRACK_DISPLACEMENT_PREDICTOR_HPP
```

- [ ] **Step 2: Create displacement predictor implementation**

```cpp
// src/tracking/displacement_predictor.cpp
#include "tracking/displacement_predictor.hpp"
#include <cmath>

namespace ultratrack {

DisplacementPredictor::DisplacementPredictor(const DisplacementConfig& config)
    : config_(config), history_count_(0), last_velocity_(0, 0) {
    pos_history_[0] = cv::Point2f(0, 0);
    pos_history_[1] = cv::Point2f(0, 0);
}

cv::Point2f DisplacementPredictor::predict(const cv::Point2f& current_pos) {
    cv::Point2f predicted = current_pos;
    
    if (config_.enabled && history_count_ >= 2) {
        // Compute displacement from last two frames
        cv::Point2f displacement = pos_history_[1] - pos_history_[0];
        float disp_magnitude = std::sqrt(displacement.x * displacement.x + 
                                          displacement.y * displacement.y);
        
        // Apply prediction only if displacement is within thresholds
        if (disp_magnitude > config_.min_threshold && 
            disp_magnitude < config_.max_threshold) {
            // P_t = P_{t-1} + kappa * (P_{t-1} - P_{t-2})
            predicted = current_pos + config_.kappa * displacement;
            last_velocity_ = displacement;
        } else {
            last_velocity_ = cv::Point2f(0, 0);
        }
    }
    
    // Update history
    pos_history_[0] = pos_history_[1];
    pos_history_[1] = current_pos;
    if (history_count_ < 2) {
        history_count_++;
    }
    
    return predicted;
}

void DisplacementPredictor::reset() {
    history_count_ = 0;
    pos_history_[0] = cv::Point2f(0, 0);
    pos_history_[1] = cv::Point2f(0, 0);
    last_velocity_ = cv::Point2f(0, 0);
}

void DisplacementPredictor::set_config(const DisplacementConfig& config) {
    config_ = config;
}

} // namespace ultratrack
```

- [ ] **Step 3: Create displacement predictor test**

```cpp
// tests/test_displacement.cpp
#include "tracking/displacement_predictor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace ultratrack;

void test_initial_state() {
    DisplacementPredictor pred;
    assert(!pred.is_ready());
    std::cout << "  PASS: test_initial_state\n";
}

void test_history_buildup() {
    DisplacementPredictor pred;
    
    pred.predict({100, 100});
    assert(!pred.is_ready());
    
    pred.predict({110, 100});
    assert(pred.is_ready());
    
    std::cout << "  PASS: test_history_buildup\n";
}

void test_linear_motion_prediction() {
    DisplacementPredictor pred;
    
    // Build history: moving right at 10 pixels/frame
    pred.predict({100, 100});  // t-2
    pred.predict({110, 100});  // t-1
    
    // Current position, expect prediction ahead
    auto predicted = pred.predict({120, 100});  // t
    
    // Expected: 120 + 0.8 * (110 - 100) = 128
    float expected_x = 120 + 0.8f * 10;
    assert(std::abs(predicted.x - expected_x) < 1.0f);
    assert(std::abs(predicted.y - 100) < 1.0f);
    
    std::cout << "  PASS: test_linear_motion_prediction\n";
}

void test_stationary_no_prediction() {
    DisplacementConfig config;
    config.min_threshold = 2.0f;
    DisplacementPredictor pred(config);
    
    // Stationary target (displacement < min_threshold)
    pred.predict({100, 100});
    pred.predict({100.5f, 100});  // 0.5 pixel movement
    
    auto predicted = pred.predict({101, 100});
    
    // Should return current position (no prediction)
    assert(std::abs(predicted.x - 101) < 0.5f);
    
    std::cout << "  PASS: test_stationary_no_prediction\n";
}

void test_reset() {
    DisplacementPredictor pred;
    
    pred.predict({100, 100});
    pred.predict({110, 100});
    assert(pred.is_ready());
    
    pred.reset();
    assert(!pred.is_ready());
    
    std::cout << "  PASS: test_reset\n";
}

void test_disabled_prediction() {
    DisplacementConfig config;
    config.enabled = false;
    DisplacementPredictor pred(config);
    
    pred.predict({100, 100});
    pred.predict({110, 100});
    
    auto predicted = pred.predict({120, 100});
    
    // Should return current position when disabled
    assert(std::abs(predicted.x - 120) < 0.01f);
    
    std::cout << "  PASS: test_disabled_prediction\n";
}

int main() {
    std::cout << "Running displacement predictor tests...\n";
    
    test_initial_state();
    test_history_buildup();
    test_linear_motion_prediction();
    test_stationary_no_prediction();
    test_reset();
    test_disabled_prediction();
    
    std::cout << "All displacement predictor tests passed!\n";
    return 0;
}
```

- [ ] **Step 4: Commit**

```bash
git add include/tracking/displacement_predictor.hpp src/tracking/displacement_predictor.cpp tests/test_displacement.cpp
git commit -m "feat: implement displacement predictor with velocity-based position estimation"
```

---

## Task 7: Implement Scale Estimator

**Files:**
- Create: `include/tracking/scale_estimator.hpp`
- Create: `src/tracking/scale_estimator.cpp`
- Create: `tests/test_scale.cpp`

- [ ] **Step 1: Create scale estimator header**

```cpp
// include/tracking/scale_estimator.hpp
#ifndef ULTRATRACK_SCALE_ESTIMATOR_HPP
#define ULTRATRACK_SCALE_ESTIMATOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ultratrack {

struct ScaleConfig {
    float scale_step = 0.04f;     // Scale increment delta
    int num_scales = 3;           // Number of scales (3, 5, or 7)
    float min_scale = 0.2f;       // Minimum allowed scale
    float max_scale = 5.0f;       // Maximum allowed scale
    float scale_penalty = 0.975f; // Penalty for scale change
};

class ScaleEstimator {
public:
    explicit ScaleEstimator(const ScaleConfig& config = {});
    
    // Estimate optimal scale
    float estimate(const cv::Mat& frame,
                   const cv::Point2f& position,
                   const cv::Size2f& base_size,
                   const cv::Mat& correlation_filter,
                   const cv::Mat& model_patch);
    
    // Get current scale pool
    std::vector<float> get_scale_pool() const { return scale_pool_; }
    
    // Get accumulated scale
    float get_current_scale() const { return current_scale_; }
    
    // Reset scale to 1.0
    void reset();
    
    // Configuration
    void set_config(const ScaleConfig& config);
    ScaleConfig get_config() const { return config_; }

private:
    ScaleConfig config_;
    float current_scale_ = 1.0f;
    std::vector<float> scale_pool_;
    
    void rebuild_scale_pool();
    
    cv::Mat extract_patch_at_scale(const cv::Mat& frame,
                                    const cv::Point2f& center,
                                    const cv::Size2f& base_size,
                                    float scale);
    
    float compute_response(const cv::Mat& patch,
                           const cv::Mat& correlation_filter,
                           const cv::Mat& model_patch);
};

} // namespace ultratrack

#endif // ULTRATRACK_SCALE_ESTIMATOR_HPP
```

- [ ] **Step 2: Create scale estimator implementation**

```cpp
// src/tracking/scale_estimator.cpp
#include "tracking/scale_estimator.hpp"
#include "errors.hpp"
#include <algorithm>
#include <cmath>

namespace ultratrack {

ScaleEstimator::ScaleEstimator(const ScaleConfig& config) : config_(config) {
    rebuild_scale_pool();
}

void ScaleEstimator::rebuild_scale_pool() {
    scale_pool_.clear();
    
    // Build scale pool: {1-2δ, 1-δ, ..., 1, ..., 1+δ, 1+2δ}
    // For 3 scales: {1-2δ, 1, 1+2δ}
    int half = config_.num_scales / 2;
    
    for (int i = -half; i <= half; i++) {
        if (config_.num_scales == 3 && i != 0) {
            // For 3-scale, use 2*delta steps
            scale_pool_.push_back(1.0f + 2.0f * i * config_.scale_step);
        } else {
            scale_pool_.push_back(1.0f + i * config_.scale_step);
        }
    }
    
    // For 3-scale pool, ensure we have exactly {1-2δ, 1, 1+2δ}
    if (config_.num_scales == 3) {
        scale_pool_.clear();
        scale_pool_.push_back(1.0f - 2.0f * config_.scale_step);
        scale_pool_.push_back(1.0f);
        scale_pool_.push_back(1.0f + 2.0f * config_.scale_step);
    }
}

cv::Mat ScaleEstimator::extract_patch_at_scale(const cv::Mat& frame,
                                                const cv::Point2f& center,
                                                const cv::Size2f& base_size,
                                                float scale) {
    float scaled_w = base_size.width * scale;
    float scaled_h = base_size.height * scale;
    
    cv::Rect2f roi(center.x - scaled_w / 2,
                   center.y - scaled_h / 2,
                   scaled_w, scaled_h);
    
    // Clamp to frame bounds
    cv::Rect safe_roi = cv::Rect(roi) & cv::Rect(0, 0, frame.cols, frame.rows);
    
    if (safe_roi.area() <= 0) {
        return cv::Mat();
    }
    
    return frame(safe_roi).clone();
}

float ScaleEstimator::compute_response(const cv::Mat& patch,
                                        const cv::Mat& correlation_filter,
                                        const cv::Mat& model_patch) {
    if (patch.empty() || correlation_filter.empty()) {
        return -std::numeric_limits<float>::max();
    }
    
    // Resize patch to match model size
    cv::Mat resized;
    cv::resize(patch, resized, model_patch.size());
    
    // Convert to grayscale float
    cv::Mat gray;
    if (resized.channels() == 3) {
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = resized;
    }
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);
    
    // Compute correlation response
    cv::Mat patch_fft, response_fft, response;
    cv::dft(gray, patch_fft, cv::DFT_COMPLEX_OUTPUT);
    cv::mulSpectrums(correlation_filter, patch_fft, response_fft, 0, true);
    cv::idft(response_fft, response, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    
    // Return peak response
    double max_val;
    cv::minMaxLoc(response, nullptr, &max_val);
    
    return static_cast<float>(max_val);
}

float ScaleEstimator::estimate(const cv::Mat& frame,
                                const cv::Point2f& position,
                                const cv::Size2f& base_size,
                                const cv::Mat& correlation_filter,
                                const cv::Mat& model_patch) {
    if (frame.empty()) {
        return current_scale_;
    }
    
    float best_scale = current_scale_;
    float best_response = -std::numeric_limits<float>::max();
    
    for (float scale_factor : scale_pool_) {
        float test_scale = current_scale_ * scale_factor;
        
        // Clamp to valid range
        test_scale = std::clamp(test_scale, config_.min_scale, config_.max_scale);
        
        // Extract patch at test scale
        cv::Mat patch = extract_patch_at_scale(frame, position, base_size, test_scale);
        if (patch.empty()) continue;
        
        // Compute response
        float response = compute_response(patch, correlation_filter, model_patch);
        
        // Apply scale penalty for non-unity scales
        if (std::abs(scale_factor - 1.0f) > 0.001f) {
            response *= config_.scale_penalty;
        }
        
        if (response > best_response) {
            best_response = response;
            best_scale = test_scale;
        }
    }
    
    current_scale_ = best_scale;
    return current_scale_;
}

void ScaleEstimator::reset() {
    current_scale_ = 1.0f;
}

void ScaleEstimator::set_config(const ScaleConfig& config) {
    config_ = config;
    rebuild_scale_pool();
}

} // namespace ultratrack
```

- [ ] **Step 3: Create scale estimator test**

```cpp
// tests/test_scale.cpp
#include "tracking/scale_estimator.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace ultratrack;

void test_three_scale_pool() {
    ScaleConfig config;
    config.num_scales = 3;
    config.scale_step = 0.04f;
    
    ScaleEstimator est(config);
    auto pool = est.get_scale_pool();
    
    assert(pool.size() == 3);
    assert(std::abs(pool[0] - 0.92f) < 0.001f);  // 1 - 2*0.04
    assert(std::abs(pool[1] - 1.00f) < 0.001f);
    assert(std::abs(pool[2] - 1.08f) < 0.001f);  // 1 + 2*0.04
    
    std::cout << "  PASS: test_three_scale_pool\n";
}

void test_five_scale_pool() {
    ScaleConfig config;
    config.num_scales = 5;
    config.scale_step = 0.02f;
    
    ScaleEstimator est(config);
    auto pool = est.get_scale_pool();
    
    assert(pool.size() == 5);
    assert(std::abs(pool[2] - 1.00f) < 0.001f);  // Center is 1.0
    
    std::cout << "  PASS: test_five_scale_pool\n";
}

void test_initial_scale() {
    ScaleEstimator est;
    assert(std::abs(est.get_current_scale() - 1.0f) < 0.001f);
    
    std::cout << "  PASS: test_initial_scale\n";
}

void test_reset() {
    ScaleEstimator est;
    
    // Manually set a different scale (via estimation would require frames)
    est.reset();
    
    assert(std::abs(est.get_current_scale() - 1.0f) < 0.001f);
    
    std::cout << "  PASS: test_reset\n";
}

void test_config_update() {
    ScaleEstimator est;
    
    ScaleConfig new_config;
    new_config.num_scales = 7;
    new_config.scale_step = 0.03f;
    
    est.set_config(new_config);
    
    auto pool = est.get_scale_pool();
    assert(pool.size() == 7);
    
    std::cout << "  PASS: test_config_update\n";
}

void test_scale_clamping() {
    ScaleConfig config;
    config.min_scale = 0.5f;
    config.max_scale = 2.0f;
    
    ScaleEstimator est(config);
    
    // The clamping happens during estimate(), which needs a frame
    // Just verify config is stored
    assert(std::abs(est.get_config().min_scale - 0.5f) < 0.001f);
    assert(std::abs(est.get_config().max_scale - 2.0f) < 0.001f);
    
    std::cout << "  PASS: test_scale_clamping\n";
}

int main() {
    std::cout << "Running scale estimator tests...\n";
    
    test_three_scale_pool();
    test_five_scale_pool();
    test_initial_scale();
    test_reset();
    test_config_update();
    test_scale_clamping();
    
    std::cout << "All scale estimator tests passed!\n";
    return 0;
}
```

- [ ] **Step 4: Commit**

```bash
git add include/tracking/scale_estimator.hpp src/tracking/scale_estimator.cpp tests/test_scale.cpp
git commit -m "feat: implement 3-scale pool estimator for adaptive scale tracking"
```

---

## Task 8: Create Feature Extraction Tests

**Files:**
- Create: `tests/test_features.cpp`

- [ ] **Step 1: Create feature extraction tests**

```cpp
// tests/test_features.cpp
#include "feature_extractor.hpp"
#include "features/hog_feature.hpp"
#include "features/gray_feature.hpp"
#include "features/cn_feature.hpp"
#include <cassert>
#include <iostream>

using namespace ultratrack;

void test_hog_dimensions() {
    HOGFeature hog;
    assert(hog.dimensions() == 31);
    assert(hog.name() == "HOG");
    
    std::cout << "  PASS: test_hog_dimensions\n";
}

void test_hog_extraction() {
    HOGFeature hog;
    
    // Create test patch
    cv::Mat patch = cv::Mat::zeros(64, 64, CV_8UC3);
    cv::rectangle(patch, cv::Rect(16, 16, 32, 32), cv::Scalar(255, 255, 255), -1);
    
    cv::Mat features = hog.extract(patch);
    
    assert(!features.empty());
    assert(features.channels() == 31);
    
    std::cout << "  PASS: test_hog_extraction\n";
}

void test_gray_dimensions() {
    GrayFeature gray;
    assert(gray.dimensions() == 1);
    assert(gray.name() == "Gray");
    
    std::cout << "  PASS: test_gray_dimensions\n";
}

void test_gray_extraction() {
    GrayFeature gray;
    
    cv::Mat patch = cv::Mat::ones(64, 64, CV_8UC3) * 128;
    cv::Mat features = gray.extract(patch);
    
    assert(!features.empty());
    assert(features.channels() == 1);
    assert(features.type() == CV_32FC1);
    
    std::cout << "  PASS: test_gray_extraction\n";
}

void test_cn_dimensions() {
    CNFeature cn;
    assert(cn.dimensions() == 10);
    assert(cn.name() == "ColorNames");
    
    std::cout << "  PASS: test_cn_dimensions\n";
}

void test_cn_extraction() {
    CNFeature cn;
    
    // Create a red patch
    cv::Mat patch = cv::Mat(64, 64, CV_8UC3, cv::Scalar(0, 0, 255));  // BGR red
    cv::Mat features = cn.extract(patch);
    
    assert(!features.empty());
    assert(features.channels() == 10);
    
    std::cout << "  PASS: test_cn_extraction\n";
}

void test_multi_feature_fast_mode() {
    MultiFeatureExtractor ext(TrackingMode::FAST);
    
    assert(ext.total_dimensions() == 31);  // HOG only
    assert(ext.get_mode() == TrackingMode::FAST);
    
    std::cout << "  PASS: test_multi_feature_fast_mode\n";
}

void test_multi_feature_balanced_mode() {
    MultiFeatureExtractor ext(TrackingMode::BALANCED);
    
    assert(ext.total_dimensions() == 32);  // HOG + Gray
    assert(ext.get_mode() == TrackingMode::BALANCED);
    
    std::cout << "  PASS: test_multi_feature_balanced_mode\n";
}

void test_multi_feature_accurate_mode() {
    MultiFeatureExtractor ext(TrackingMode::ACCURATE);
    
    assert(ext.total_dimensions() == 42);  // HOG + Gray + CN
    assert(ext.get_mode() == TrackingMode::ACCURATE);
    
    std::cout << "  PASS: test_multi_feature_accurate_mode\n";
}

void test_mode_switching() {
    MultiFeatureExtractor ext(TrackingMode::FAST);
    assert(ext.total_dimensions() == 31);
    
    ext.set_mode(TrackingMode::ACCURATE);
    assert(ext.total_dimensions() == 42);
    
    ext.set_mode(TrackingMode::BALANCED);
    assert(ext.total_dimensions() == 32);
    
    std::cout << "  PASS: test_mode_switching\n";
}

void test_multi_feature_extraction() {
    MultiFeatureExtractor ext(TrackingMode::ACCURATE);
    
    cv::Mat patch = cv::Mat::zeros(64, 64, CV_8UC3);
    cv::rectangle(patch, cv::Rect(16, 16, 32, 32), cv::Scalar(0, 0, 255), -1);
    
    cv::Mat features = ext.extract(patch);
    
    assert(!features.empty());
    
    std::cout << "  PASS: test_multi_feature_extraction\n";
}

int main() {
    std::cout << "Running feature extraction tests...\n";
    
    test_hog_dimensions();
    test_hog_extraction();
    test_gray_dimensions();
    test_gray_extraction();
    test_cn_dimensions();
    test_cn_extraction();
    test_multi_feature_fast_mode();
    test_multi_feature_balanced_mode();
    test_multi_feature_accurate_mode();
    test_mode_switching();
    test_multi_feature_extraction();
    
    std::cout << "All feature extraction tests passed!\n";
    return 0;
}
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_features.cpp
git commit -m "test: add comprehensive feature extraction unit tests"
```

---

## Task 9: Integrate Components into UltraTracker

**Files:**
- Modify: `include/ultratrack.hpp`
- Modify: `src/ultratrack.cpp`

- [ ] **Step 1: Update ultratrack.hpp with new includes and types**

Add after line 10 (after opencv includes):

```cpp
#include "feature_extractor.hpp"
#include "tracking/displacement_predictor.hpp"
#include "tracking/scale_estimator.hpp"
#include "errors.hpp"
```

- [ ] **Step 2: Add TrackerConfig struct after Detection struct (around line 34)**

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

- [ ] **Step 3: Update Track struct to add new fields (around line 50)**

Add before the closing brace of Track struct:

```cpp
    // Displacement prediction
    std::unique_ptr<DisplacementPredictor> displacement_predictor;
    cv::Point2f predicted_center;
    cv::Size2f base_size;
    cv::Mat multi_channel_filter;
```

- [ ] **Step 4: Update UltraTracker class - add new constructor and methods**

Add after existing constructor declaration:

```cpp
    UltraTracker(const std::string& model_path, const TrackerConfig& config,
                 const std::string& feature_model_path = "");
    
    // Mode switching
    void set_tracking_mode(TrackingMode mode);
    TrackingMode get_tracking_mode() const;
    void set_tracker_config(const TrackerConfig& config);
    TrackerConfig get_tracker_config() const;
```

- [ ] **Step 5: Add new private members to UltraTracker class**

Add in private section:

```cpp
    TrackerConfig tracker_config_;
    std::unique_ptr<MultiFeatureExtractor> feature_extractor_;
    std::unique_ptr<ScaleEstimator> scale_estimator_;
    
    cv::Mat create_multi_channel_filter(const cv::Mat& patch);
    void update_track_with_displacement(Track& track, const cv::Mat& frame);
    cv::Mat extract_multi_channel_features(const cv::Mat& patch);
```

- [ ] **Step 6: Update ultratrack.cpp - add new constructor**

Add after existing constructor:

```cpp
UltraTracker::UltraTracker(const std::string& model_path, const TrackerConfig& config,
                           const std::string& feature_model_path)
    : UltraTracker(model_path, feature_model_path) {
    tracker_config_ = config;
    feature_extractor_ = std::make_unique<MultiFeatureExtractor>(config.mode, config.feature);
    scale_estimator_ = std::make_unique<ScaleEstimator>(config.scale);
    learning_rate_ = config.learning_rate;
    lambda_ = config.lambda;
    sigma_ = config.sigma;
}
```

- [ ] **Step 7: Add mode switching methods**

```cpp
void UltraTracker::set_tracking_mode(TrackingMode mode) {
    tracker_config_.mode = mode;
    if (feature_extractor_) {
        feature_extractor_->set_mode(mode);
    }
}

TrackingMode UltraTracker::get_tracking_mode() const {
    return tracker_config_.mode;
}

void UltraTracker::set_tracker_config(const TrackerConfig& config) {
    tracker_config_ = config;
    if (feature_extractor_) {
        feature_extractor_->set_mode(config.mode);
    }
    if (scale_estimator_) {
        scale_estimator_->set_config(config.scale);
    }
    learning_rate_ = config.learning_rate;
    lambda_ = config.lambda;
    sigma_ = config.sigma;
}

TrackerConfig UltraTracker::get_tracker_config() const {
    return tracker_config_;
}
```

- [ ] **Step 8: Add multi-channel filter creation**

```cpp
cv::Mat UltraTracker::create_multi_channel_filter(const cv::Mat& patch) {
    if (patch.empty() || !feature_extractor_) {
        return create_correlation_filter(patch);  // Fallback to original
    }
    
    cv::Mat features = feature_extractor_->extract(patch);
    if (features.empty()) {
        return create_correlation_filter(patch);
    }
    
    // Create filter for multi-channel features
    cv::Mat resized;
    cv::resize(features, resized, template_size_);
    
    // Apply Hann window to each channel
    std::vector<cv::Mat> channels;
    cv::split(resized, channels);
    
    cv::Mat filter_sum = cv::Mat::zeros(template_size_, CV_32FC2);
    
    for (auto& channel : channels) {
        channel = channel.mul(hann_window_);
        
        cv::Mat channel_fft = fft2d(channel);
        cv::Mat target_fft = fft2d(gaussian_target_);
        
        cv::Mat numerator, denominator;
        cv::mulSpectrums(target_fft, channel_fft, numerator, 0, true);
        cv::mulSpectrums(channel_fft, channel_fft, denominator, 0, true);
        
        denominator += cv::Scalar::all(lambda_);
        
        cv::Mat channel_filter;
        cv::divide(numerator, denominator, channel_filter);
        
        filter_sum += channel_filter;
    }
    
    // Average across channels
    filter_sum /= static_cast<float>(channels.size());
    
    return filter_sum;
}
```

- [ ] **Step 9: Update create_new_tracks to use displacement predictor**

In `create_new_tracks` function, after `new_track.correlation_filter = ...`, add:

```cpp
        // Initialize displacement predictor
        new_track.displacement_predictor = std::make_unique<DisplacementPredictor>(
            tracker_config_.displacement);
        new_track.base_size = cv::Size2f(detection.bbox.width, detection.bbox.height);
        new_track.predicted_center = cv::Point2f(
            detection.bbox.x + detection.bbox.width / 2,
            detection.bbox.y + detection.bbox.height / 2);
        
        // Create multi-channel filter if available
        if (feature_extractor_) {
            new_track.multi_channel_filter = create_multi_channel_filter(patch);
        }
```

- [ ] **Step 10: Update predict_tracks to use displacement prediction**

In `predict_tracks` function, inside the loop, add after `predict_kalman(track)`:

```cpp
        // Displacement prediction
        if (track.displacement_predictor) {
            cv::Point2f current_center(
                track.bbox.x + track.bbox.width / 2,
                track.bbox.y + track.bbox.height / 2);
            track.predicted_center = track.displacement_predictor->predict(current_center);
        }
```

- [ ] **Step 11: Commit**

```bash
git add include/ultratrack.hpp src/ultratrack.cpp
git commit -m "feat: integrate displacement prediction, scale estimation, and multi-feature extraction into UltraTracker"
```

---

## Task 10: Create Integration Tests

**Files:**
- Create: `tests/test_tracker.cpp`

- [ ] **Step 1: Create tracker integration tests**

```cpp
// tests/test_tracker.cpp
#include "ultratrack.hpp"
#include <cassert>
#include <iostream>

using namespace ultratrack;

void test_tracker_config_modes() {
    std::cout << "  Note: Skipping model-dependent tests (no ONNX model)\n";
    
    // Test config creation
    TrackerConfig config;
    config.mode = TrackingMode::ACCURATE;
    config.displacement.kappa = 0.8f;
    config.scale.num_scales = 3;
    
    assert(config.mode == TrackingMode::ACCURATE);
    assert(config.displacement.kappa == 0.8f);
    assert(config.scale.num_scales == 3);
    
    std::cout << "  PASS: test_tracker_config_modes\n";
}

void test_displacement_config() {
    DisplacementConfig config;
    config.enabled = true;
    config.kappa = 0.9f;
    config.min_threshold = 1.0f;
    config.max_threshold = 150.0f;
    
    DisplacementPredictor pred(config);
    
    assert(pred.get_config().kappa == 0.9f);
    assert(pred.get_config().min_threshold == 1.0f);
    
    std::cout << "  PASS: test_displacement_config\n";
}

void test_scale_config() {
    ScaleConfig config;
    config.num_scales = 5;
    config.scale_step = 0.02f;
    config.scale_penalty = 0.98f;
    
    ScaleEstimator est(config);
    
    assert(est.get_config().num_scales == 5);
    assert(est.get_scale_pool().size() == 5);
    
    std::cout << "  PASS: test_scale_config\n";
}

void test_feature_extractor_integration() {
    MultiFeatureExtractor ext(TrackingMode::ACCURATE);
    
    // Create synthetic frame
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(frame, cv::Rect(100, 100, 64, 64), cv::Scalar(0, 0, 255), -1);
    
    cv::Mat patch = frame(cv::Rect(100, 100, 64, 64));
    cv::Mat features = ext.extract(patch);
    
    assert(!features.empty());
    
    std::cout << "  PASS: test_feature_extractor_integration\n";
}

void test_tracking_pipeline_components() {
    // Test that all components can be instantiated together
    TrackerConfig config;
    config.mode = TrackingMode::BALANCED;
    
    auto feature_ext = std::make_unique<MultiFeatureExtractor>(config.mode, config.feature);
    auto scale_est = std::make_unique<ScaleEstimator>(config.scale);
    auto disp_pred = std::make_unique<DisplacementPredictor>(config.displacement);
    
    assert(feature_ext->total_dimensions() == 32);
    assert(scale_est->get_scale_pool().size() == 3);
    assert(!disp_pred->is_ready());
    
    std::cout << "  PASS: test_tracking_pipeline_components\n";
}

int main() {
    std::cout << "Running tracker integration tests...\n";
    
    test_tracker_config_modes();
    test_displacement_config();
    test_scale_config();
    test_feature_extractor_integration();
    test_tracking_pipeline_components();
    
    std::cout << "All tracker integration tests passed!\n";
    return 0;
}
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_tracker.cpp
git commit -m "test: add tracker integration tests for new components"
```

---

## Task 11: Update Build System

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add new source files to CMakeLists.txt**

After line 104 (after existing SOURCES definition), replace the SOURCES block:

```cmake
# Feature extraction sources
set(FEATURE_SOURCES
    ${SRC_DIR}/features/feature_extractor.cpp
    ${SRC_DIR}/features/hog_feature.cpp
    ${SRC_DIR}/features/gray_feature.cpp
    ${SRC_DIR}/features/cn_feature.cpp
)

# Tracking module sources
set(TRACKING_SOURCES
    ${SRC_DIR}/tracking/displacement_predictor.cpp
    ${SRC_DIR}/tracking/scale_estimator.cpp
)

set(SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/ultratrack.cpp
    ${SRC_DIR}/simd_optimization.cpp
    ${FEATURE_SOURCES}
    ${TRACKING_SOURCES}
)

set(HEADERS
    ${INC_DIR}/ultratrack.hpp
    ${INC_DIR}/version.hpp
    ${INC_DIR}/feature_extractor.hpp
    ${INC_DIR}/errors.hpp
    ${INC_DIR}/features/hog_feature.hpp
    ${INC_DIR}/features/gray_feature.hpp
    ${INC_DIR}/features/cn_feature.hpp
    ${INC_DIR}/tracking/displacement_predictor.hpp
    ${INC_DIR}/tracking/scale_estimator.hpp
)
```

- [ ] **Step 2: Add data directory copy command**

After the create_models_dir target (around line 156):

```cmake
# Create data directory and copy lookup table if exists
add_custom_target(create_data_dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_BINARY_DIR}/Release/data
    COMMENT "Creating data directory"
)

if(EXISTS ${CMAKE_SOURCE_DIR}/data/cn_lookup.bin)
    configure_file(
        ${CMAKE_SOURCE_DIR}/data/cn_lookup.bin
        ${CMAKE_BINARY_DIR}/Release/data/cn_lookup.bin
        COPYONLY
    )
endif()
```

- [ ] **Step 3: Add test targets**

Before the install section (around line 160):

```cmake
# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
option(BUILD_TESTS "Build unit tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    
    # Test executables
    add_executable(test_displacement
        tests/test_displacement.cpp
        ${SRC_DIR}/tracking/displacement_predictor.cpp
    )
    target_include_directories(test_displacement PRIVATE ${INC_DIR} ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(test_displacement PRIVATE ${OpenCV_LIBS})
    add_test(NAME DisplacementTests COMMAND test_displacement)
    
    add_executable(test_scale
        tests/test_scale.cpp
        ${SRC_DIR}/tracking/scale_estimator.cpp
    )
    target_include_directories(test_scale PRIVATE ${INC_DIR} ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(test_scale PRIVATE ${OpenCV_LIBS})
    add_test(NAME ScaleTests COMMAND test_scale)
    
    add_executable(test_features
        tests/test_features.cpp
        ${FEATURE_SOURCES}
    )
    target_include_directories(test_features PRIVATE ${INC_DIR} ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(test_features PRIVATE ${OpenCV_LIBS})
    add_test(NAME FeatureTests COMMAND test_features)
    
    add_executable(test_tracker
        tests/test_tracker.cpp
        ${FEATURE_SOURCES}
        ${TRACKING_SOURCES}
    )
    target_include_directories(test_tracker PRIVATE ${INC_DIR} ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(test_tracker PRIVATE ${OpenCV_LIBS})
    add_test(NAME TrackerTests COMMAND test_tracker)
endif()
```

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: update CMakeLists.txt with new feature and tracking modules"
```

---

## Task 12: Create Directory Structure and Build Verification

**Files:**
- Create directories

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p include/features
mkdir -p include/tracking
mkdir -p src/features
mkdir -p src/tracking
mkdir -p data
mkdir -p tests
```

- [ ] **Step 2: Build and verify**

```bash
mkdir -p build
cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . --config Release
```

Expected: Build succeeds without errors

- [ ] **Step 3: Run tests**

```bash
ctest -C Release --output-on-failure
```

Expected: All tests pass

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete KCF displacement prediction tracker implementation"
```

---

## Summary

| Task | Component | Files | Est. Time |
|------|-----------|-------|-----------|
| 1 | Error handling | `errors.hpp` | 5 min |
| 2 | Feature interface | `feature_extractor.hpp/cpp` | 10 min |
| 3 | HOG feature | `hog_feature.hpp/cpp` | 15 min |
| 4 | Gray feature | `gray_feature.hpp/cpp` | 5 min |
| 5 | CN feature | `cn_feature.hpp/cpp` | 20 min |
| 6 | Displacement predictor | `displacement_predictor.hpp/cpp`, test | 15 min |
| 7 | Scale estimator | `scale_estimator.hpp/cpp`, test | 15 min |
| 8 | Feature tests | `test_features.cpp` | 10 min |
| 9 | UltraTracker integration | `ultratrack.hpp/cpp` | 25 min |
| 10 | Integration tests | `test_tracker.cpp` | 10 min |
| 11 | Build system | `CMakeLists.txt` | 10 min |
| 12 | Build verification | - | 10 min |

**Total estimated time: ~2.5 hours**
