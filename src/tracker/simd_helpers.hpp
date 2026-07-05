#pragma once

#include <opencv2/core.hpp>
#include <cstddef>

#ifdef _WIN32
#include <immintrin.h>
#include <intrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

#ifdef USE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif

namespace ultratrack {
namespace internal {

// 1-D Hann window suitable for building a 2-D window via mulTransposed.
cv::Mat create_hann_window(int size);

// Element-wise multiplication (used for applying the Hann window).
void simd_correlation(const float* a, const float* b, float* result, int size);

// Complex spectrum multiplication: result = a * conj(b) if conj_b == true.
void simd_mul_spectrums(const float* a, const float* b, float* result, int size, bool conj_b);

// Complex spectrum division: result = num / den.
void simd_div_spectrums(const float* num, const float* den, float* result, int size);

// Convenience wrapper around simd_correlation for windowing.
inline void simd_hann_window(const float* src, const float* win, float* dst, int size) {
    simd_correlation(src, win, dst, size);
}

// Forward 2-D FFT with optional CUDA fallback.
cv::Mat fft2d(const cv::Mat& input);

// Inverse 2-D FFT returning a real response map.
cv::Mat ifft2d(const cv::Mat& input);

} // namespace internal
} // namespace ultratrack
