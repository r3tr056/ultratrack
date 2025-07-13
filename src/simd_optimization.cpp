// simd_optimization.cpp
#include "ultratrack.hpp"

#ifdef _WIN32
    #include <intrin.h>
    #include <immintrin.h>  // AVX2 support
#elif defined(__ARM_NEON)
    #include <arm_neon.h>   // NEON support
#else
    #include <immintrin.h>  // AVX2/SSE support
#endif

namespace ultratrack {

void UltraTracker::simd_correlation(const float* a, const float* b, float* result, int size) {
    if (!a || !b || !result || size <= 0) return;
    
    #ifdef __AVX2__
    // AVX2 optimized correlation
    const int simd_size = 8;
    const int simd_end = size - (size % simd_size);
    
    for (int i = 0; i < simd_end; i += simd_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);  // Use unaligned load for safety
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);   // Use unaligned store for safety
    }
    
    // Handle remaining elements
    for (int i = simd_end; i < size; i++) {
        result[i] = a[i] * b[i];
    }
    
    #elif defined(__ARM_NEON)
    // NEON optimized correlation for ARM processors
    const int simd_size = 4;
    const int simd_end = size - (size % simd_size);
    
    for (int i = 0; i < simd_end; i += simd_size) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vr = vmulq_f32(va, vb);
        vst1q_f32(&result[i], vr);
    }
    
    // Handle remaining elements
    for (int i = simd_end; i < size; i++) {
        result[i] = a[i] * b[i];
    }
    
    #else
    // Fallback scalar implementation with loop unrolling
    int i = 0;
    const int unroll_end = size - (size % 4);
    
    // Unroll by 4 for better performance
    for (; i < unroll_end; i += 4) {
        result[i] = a[i] * b[i];
        result[i + 1] = a[i + 1] * b[i + 1];
        result[i + 2] = a[i + 2] * b[i + 2];
        result[i + 3] = a[i + 3] * b[i + 3];
    }
    
    // Handle remaining elements
    for (; i < size; i++) {
        result[i] = a[i] * b[i];
    }
    #endif
}

cv::Mat UltraTracker::fft2d(const cv::Mat& input) {
    if (input.empty()) return cv::Mat();
    
    try {
        #ifdef USE_CUDA
        // GPU FFT implementation
        cv::Mat padded;
        int m = cv::getOptimalDFTSize(input.rows);
        int n = cv::getOptimalDFTSize(input.cols);
        cv::copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, 
                          cv::BORDER_CONSTANT, cv::Scalar::all(0));
        
        // Check if CUDA is available before using GPU
        try {
            cv::cuda::GpuMat gpu_input, gpu_output;
            gpu_input.upload(padded);
            cv::cuda::dft(gpu_input, gpu_output, padded.size(), cv::DFT_COMPLEX_OUTPUT);
            
            cv::Mat result;
            gpu_output.download(result);
            return result;
        } catch (const cv::Exception&) {
            // Fallback to CPU if CUDA fails
        }
        #endif
        
        // CPU FFT implementation (fallback or default)
        cv::Mat padded;
        int m = cv::getOptimalDFTSize(input.rows);
        int n = cv::getOptimalDFTSize(input.cols);
        cv::copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, 
                          cv::BORDER_CONSTANT, cv::Scalar::all(0));
        
        cv::Mat result;
        cv::dft(padded, result, cv::DFT_COMPLEX_OUTPUT);
        return result;
        
    } catch (const cv::Exception& e) {
        std::cerr << "FFT2D error: " << e.what() << std::endl;
        return cv::Mat();
    }
}

cv::Mat UltraTracker::ifft2d(const cv::Mat& input) {
    if (input.empty()) return cv::Mat();
    
    try {
        #ifdef USE_CUDA
        try {
            cv::cuda::GpuMat gpu_input, gpu_output;
            gpu_input.upload(input);
            cv::cuda::dft(gpu_input, gpu_output, input.size(), 
                         cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
            
            cv::Mat result;
            gpu_output.download(result);
            return result;
        } catch (const cv::Exception&) {
            // Fallback to CPU if CUDA fails
        }
        #endif
        
        // CPU FFT implementation (fallback or default)
        cv::Mat result;
        cv::dft(input, result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        return result;
        
    } catch (const cv::Exception& e) {
        std::cerr << "IFFT2D error: " << e.what() << std::endl;
        return cv::Mat();
    }
}

} // namespace ultratrack
