// simd_optimization.cpp - SIMD-accelerated functions

#include "ultratrack.hpp"

#ifdef _WIN32
#include <immintrin.h>
#include <intrin.h> // AVX2 support
#elif defined(__ARM_NEON)
#include <arm_neon.h> // NEON support
#else
#include <immintrin.h> // AVX2/SSE support
#endif

namespace ultratrack {

void UltraTracker::simd_correlation(const float* a, const float* b, float* result, int size) {
    if (!a || !b || !result || size <= 0) return;

    // Check alignment for safety (prevent crashes on unaligned data)
    bool a_aligned = (reinterpret_cast<uintptr_t>(a) % 32 == 0);
    bool b_aligned = (reinterpret_cast<uintptr_t>(b) % 32 == 0);
    bool result_aligned = (reinterpret_cast<uintptr_t>(result) % 32 == 0);

#ifdef __AVX2__
    const int simd_size = 8; // 256-bit / 32-bit float = 8
    const int simd_end = size - (size % simd_size);
    for (int i = 0; i < simd_end; i += simd_size) {
        __m256 va = (a_aligned) ? _mm256_load_ps(&a[i]) : _mm256_loadu_ps(&a[i]);
        __m256 vb = (b_aligned) ? _mm256_load_ps(&b[i]) : _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        if (result_aligned) {
            _mm256_store_ps(&result[i], vr);
        } else {
            _mm256_storeu_ps(&result[i], vr);
        }
    }
    for (int i = simd_end; i < size; i++) {
        result[i] = a[i] * b[i];
    }
#elif defined(__ARM_NEON)
    const int simd_size = 4; // 128-bit / 32-bit float = 4
    const int simd_end = size - (size % simd_size);
    for (int i = 0; i < simd_end; i += simd_size) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vr = vmulq_f32(va, vb);
        vst1q_f32(&result[i], vr);
    }
    for (int i = simd_end; i < size; i++) {
        result[i] = a[i] * b[i];
    }
#else
    // Fallback with loop unrolling
    int i = 0;
    const int unroll_end = size - (size % 4);
    for (; i < unroll_end; i += 4) {
        result[i] = a[i] * b[i];
        result[i + 1] = a[i + 1] * b[i + 1];
        result[i + 2] = a[i + 2] * b[i + 2];
        result[i + 3] = a[i + 3] * b[i + 3];
    }
    for (; i < size; i++) {
        result[i] = a[i] * b[i];
    }
#endif
}

cv::Mat UltraTracker::fft2d(const cv::Mat& input) {
    if (input.empty()) return cv::Mat();
    try {
#ifdef USE_CUDA
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            cv::Mat padded;
            int m = cv::getOptimalDFTSize(input.rows);
            int n = cv::getOptimalDFTSize(input.cols);
            cv::copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
            cv::cuda::GpuMat gpu_input, gpu_output;
            gpu_input.upload(padded);
            cv::cuda::dft(gpu_input, gpu_output, padded.size(), cv::DFT_COMPLEX_OUTPUT);
            cv::Mat result;
            gpu_output.download(result);
            return result;
        } else {
            std::clog << "No CUDA devices found. Falling back to CPU FFT." << std::endl;
        }
#endif
        // CPU fallback
        cv::Mat padded;
        int m = cv::getOptimalDFTSize(input.rows);
        int n = cv::getOptimalDFTSize(input.cols);
        cv::copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
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
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            cv::cuda::GpuMat gpu_input, gpu_output;
            gpu_input.upload(input);
            cv::cuda::dft(gpu_input, gpu_output, input.size(), cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
            cv::Mat result;
            gpu_output.download(result);
            return result;
        } else {
            std::clog << "No CUDA devices found. Falling back to CPU IFFT." << std::endl;
        }
#endif
        // CPU fallback
        cv::Mat result;
        cv::dft(input, result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        return result;
    } catch (const cv::Exception& e) {
        std::cerr << "IFFT2D error: " << e.what() << std::endl;
        return cv::Mat();
    }
}


void UltraTracker::simd_mul_spectrums(const float* a, const float* b, float* result, int size, bool conj_b) {
    if (!a || !b || !result || size <= 0) return;

    // size is number of complex elements. float array length is 2 * size.
    int len = size * 2;

#ifdef __AVX2__
    const int simd_size = 8; // 4 complex numbers per register (8 floats)
    const int simd_end = len - (len % simd_size);
    
    for (int i = 0; i < simd_end; i += simd_size) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);

        __m256 va_re = _mm256_moveldup_ps(va); // [r0, r0, r1, r1...]
        __m256 va_im = _mm256_movehdup_ps(va); // [i0, i0, i1, i1...]
        __m256 vb_re = _mm256_moveldup_ps(vb);
        __m256 vb_im = _mm256_movehdup_ps(vb);

        __m256 res_re, res_im;

        if (conj_b) {
            // (ar + i*ai) * (br - i*bi) = (ar*br + ai*bi) + i(ai*br - ar*bi)
            res_re = _mm256_add_ps(_mm256_mul_ps(va_re, vb_re), _mm256_mul_ps(va_im, vb_im));
            res_im = _mm256_sub_ps(_mm256_mul_ps(va_im, vb_re), _mm256_mul_ps(va_re, vb_im));
        } else {
            // (ar + i*ai) * (br + i*bi) = (ar*br - ai*bi) + i(ar*bi + ai*br)
            res_re = _mm256_sub_ps(_mm256_mul_ps(va_re, vb_re), _mm256_mul_ps(va_im, vb_im));
            res_im = _mm256_add_ps(_mm256_mul_ps(va_re, vb_im), _mm256_mul_ps(va_im, vb_re));
        }

        __m256 result_vec = _mm256_blend_ps(res_re, res_im, 0xAA);
        _mm256_storeu_ps(&result[i], result_vec);
    }
    
    for (int i = simd_end; i < len; i += 2) {
        float ar = a[i], ai = a[i+1];
        float br = b[i], bi = b[i+1];
        if (conj_b) {
            result[i] = ar*br + ai*bi;
            result[i+1] = ai*br - ar*bi;
        } else {
            result[i] = ar*br - ai*bi;
            result[i+1] = ar*bi + ai*br;
        }
    }

#elif defined(__ARM_NEON)
    const int simd_size = 4; // 4 complex numbers (8 floats)
    const int simd_end = len - (len % 8); // Process 8 floats at a time (4 complex)

    for (int i = 0; i < simd_end; i += 8) {
        float32x4x2_t va = vld2q_f32(&a[i]); // .val[0] = real, .val[1] = imag
        float32x4x2_t vb = vld2q_f32(&b[i]);
        float32x4x2_t vres;

        if (conj_b) {
            // Re = ar*br + ai*bi
            vres.val[0] = vmlaq_f32(vmulq_f32(va.val[0], vb.val[0]), va.val[1], vb.val[1]);
            // Im = ai*br - ar*bi
            vres.val[1] = vmlsq_f32(vmulq_f32(va.val[1], vb.val[0]), va.val[0], vb.val[1]);
        } else {
            // Re = ar*br - ai*bi
            vres.val[0] = vmlsq_f32(vmulq_f32(va.val[0], vb.val[0]), va.val[1], vb.val[1]);
            // Im = ar*bi + ai*br
            vres.val[1] = vmlaq_f32(vmulq_f32(va.val[0], vb.val[1]), va.val[1], vb.val[0]);
        }
        vst2q_f32(&result[i], vres);
    }

    for (int i = simd_end; i < len; i += 2) {
        float ar = a[i], ai = a[i+1];
        float br = b[i], bi = b[i+1];
        if (conj_b) {
            result[i] = ar*br + ai*bi;
            result[i+1] = ai*br - ar*bi;
        } else {
            result[i] = ar*br - ai*bi;
            result[i+1] = ar*bi + ai*br;
        }
    }
#else
    for (int i = 0; i < len; i += 2) {
        float ar = a[i], ai = a[i+1];
        float br = b[i], bi = b[i+1];
        if (conj_b) {
            result[i] = ar*br + ai*bi;
            result[i+1] = ai*br - ar*bi;
        } else {
            result[i] = ar*br - ai*bi;
            result[i+1] = ar*bi + ai*br;
        }
    }
#endif
}

void UltraTracker::simd_div_spectrums(const float* num, const float* den, float* result, int size) {
    if (!num || !den || !result || size <= 0) return;
    int len = size * 2;

#ifdef __AVX2__
    const int simd_size = 8;
    const int simd_end = len - (len % simd_size);
    
    for (int i = 0; i < simd_end; i += simd_size) {
        __m256 vn = _mm256_loadu_ps(&num[i]);
        __m256 vd = _mm256_loadu_ps(&den[i]);

        __m256 vn_re = _mm256_moveldup_ps(vn);
        __m256 vn_im = _mm256_movehdup_ps(vn);
        __m256 vd_re = _mm256_moveldup_ps(vd);
        __m256 vd_im = _mm256_movehdup_ps(vd);

        // Denominator magnitude squared: d_mag2 = dr*dr + di*di
        __m256 d_mag2 = _mm256_add_ps(_mm256_mul_ps(vd_re, vd_re), _mm256_mul_ps(vd_im, vd_im));
        
        __m256 d_inv = _mm256_div_ps(_mm256_set1_ps(1.0f), d_mag2);

        // (nr + i*ni) / (dr + i*di) = [(nr*dr + ni*di) + i(ni*dr - nr*di)] / d_mag2
        
        __m256 res_re = _mm256_add_ps(_mm256_mul_ps(vn_re, vd_re), _mm256_mul_ps(vn_im, vd_im));
        res_re = _mm256_mul_ps(res_re, d_inv);

        __m256 res_im = _mm256_sub_ps(_mm256_mul_ps(vn_im, vd_re), _mm256_mul_ps(vn_re, vd_im));
        res_im = _mm256_mul_ps(res_im, d_inv);

        __m256 result_vec = _mm256_blend_ps(res_re, res_im, 0xAA);
        _mm256_storeu_ps(&result[i], result_vec);
    }

    for (int i = simd_end; i < len; i += 2) {
        float nr = num[i], ni = num[i+1];
        float dr = den[i], di = den[i+1];
        float d_mag2 = dr*dr + di*di;
        result[i] = (nr*dr + ni*di) / d_mag2;
        result[i+1] = (ni*dr - nr*di) / d_mag2;
    }

#elif defined(__ARM_NEON)
    const int simd_size = 4;
    const int simd_end = len - (len % 8);

    for (int i = 0; i < simd_end; i += 8) {
        float32x4x2_t vn = vld2q_f32(&num[i]);
        float32x4x2_t vd = vld2q_f32(&den[i]);
        float32x4x2_t vres;

        // d_mag2 = dr*dr + di*di
        float32x4_t d_mag2 = vmlaq_f32(vmulq_f32(vd.val[0], vd.val[0]), vd.val[1], vd.val[1]);
        
        float32x4_t one = vdupq_n_f32(1.0f);
#if defined(__aarch64__)
        float32x4_t d_inv = vdivq_f32(one, d_mag2);
#else
        float32x4_t rec = vrecpeq_f32(d_mag2);
        rec = vmulq_f32(vrecpsq_f32(d_mag2, rec), rec);
        rec = vmulq_f32(vrecpsq_f32(d_mag2, rec), rec);
        float32x4_t d_inv = rec;
#endif

        // Re = (nr*dr + ni*di) * inv
        float32x4_t re_num = vmlaq_f32(vmulq_f32(vn.val[0], vd.val[0]), vn.val[1], vd.val[1]);
        vres.val[0] = vmulq_f32(re_num, d_inv);

        // Im = (ni*dr - nr*di) * inv
        float32x4_t im_num = vmlsq_f32(vmulq_f32(vn.val[1], vd.val[0]), vn.val[0], vd.val[1]);
        vres.val[1] = vmulq_f32(im_num, d_inv);

        vst2q_f32(&result[i], vres);
    }

    for (int i = simd_end; i < len; i += 2) {
        float nr = num[i], ni = num[i+1];
        float dr = den[i], di = den[i+1];
        float d_mag2 = dr*dr + di*di;
        result[i] = (nr*dr + ni*di) / d_mag2;
        result[i+1] = (ni*dr - nr*di) / d_mag2;
    }
#else
    for (int i = 0; i < len; i += 2) {
        float nr = num[i], ni = num[i+1];
        float dr = den[i], di = den[i+1];
        float d_mag2 = dr*dr + di*di;
        result[i] = (nr*dr + ni*di) / d_mag2;
        result[i+1] = (ni*dr - nr*di) / d_mag2;
    }
#endif
}

void UltraTracker::simd_hann_window(const float* src, const float* win, float* dst, int size) {
    simd_correlation(src, win, dst, size);
}

} // namespace ultratrack
