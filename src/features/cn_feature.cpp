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
