// simd_optimization.cpp - SIMD-accelerated functions

#include "ultratrack.hpp"
#include "tracker/simd_helpers.hpp"

namespace ultratrack {

void UltraTracker::simd_correlation(const float* a, const float* b, float* result, int size) {
    internal::simd_correlation(a, b, result, size);
}

cv::Mat UltraTracker::fft2d(const cv::Mat& input) {
    return internal::fft2d(input);
}

cv::Mat UltraTracker::ifft2d(const cv::Mat& input) {
    return internal::ifft2d(input);
}

void UltraTracker::simd_mul_spectrums(const float* a, const float* b, float* result, int size, bool conj_b) {
    internal::simd_mul_spectrums(a, b, result, size, conj_b);
}

void UltraTracker::simd_div_spectrums(const float* num, const float* den, float* result, int size) {
    internal::simd_div_spectrums(num, den, result, size);
}

void UltraTracker::simd_hann_window(const float* src, const float* win, float* dst, int size) {
    internal::simd_hann_window(src, win, dst, size);
}

} // namespace ultratrack
