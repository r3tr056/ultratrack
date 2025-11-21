#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>

// Forward declaration for NvBufSurface (simulated if not present)
// In a real DeepStream build, this would include "nvbufsurface.h"
#ifndef NVBUFSURFACE_H_
typedef struct NvBufSurfaceParams {
    uint32_t width;
    uint32_t height;
    uint32_t pitch;
    uint32_t colorFormat;
    uint32_t dataSize;
    void* dataPtr;
    // ... simplified for this implementation ...
} NvBufSurfaceParams;
#endif

__global__ void nv12_to_rgb_norm_kernel(
    uint8_t* src_y, uint8_t* src_uv, int src_pitch,
    float* dst_rgb, int dst_w, int dst_h, 
    float crop_x, float crop_y, float crop_w, float crop_h) 
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= dst_w || ty >= dst_h) return;

    // 1. Map destination pixel (tx, ty) to source crop coordinates
    float src_x_f = crop_x + (tx / (float)dst_w) * crop_w;
    float src_y_f = crop_y + (ty / (float)dst_h) * crop_h;
    int sx = (int)src_x_f;
    int sy = (int)src_y_f;

    // 2. Read NV12 (Y plane and UV plane)
    // Y plane is full resolution
    uint8_t y_val = src_y[sy * src_pitch + sx];
    
    // UV plane is half resolution (subsampled 2x2)
    // UV data is interleaved U, V, U, V...
    int uv_pitch = src_pitch; // Usually same pitch
    int uv_offset = (sy / 2) * uv_pitch + (sx / 2) * 2;
    
    uint8_t u_val = src_uv[uv_offset];
    uint8_t v_val = src_uv[uv_offset + 1];

    // 3. Color Conversion YUV -> RGB
    // Standard BT.601 conversion
    float y = (float)y_val;
    float u = (float)u_val - 128.0f;
    float v = (float)v_val - 128.0f;

    float r = y + 1.402f * v;
    float g = y - 0.344136f * u - 0.714136f * v;
    float b = y + 1.772f * u;

    // Clamp to 0-255
    r = (r < 0.0f) ? 0.0f : ((r > 255.0f) ? 255.0f : r);
    g = (g < 0.0f) ? 0.0f : ((g > 255.0f) ? 255.0f : g);
    b = (b < 0.0f) ? 0.0f : ((b > 255.0f) ? 255.0f : b);

    // 4. Normalize & Store (Planar: RRR...GGG...BBB...)
    // Normalization 0.0 - 1.0
    dst_rgb[ty * dst_w + tx] = r / 255.0f;
    dst_rgb[dst_w * dst_h + ty * dst_w + tx] = g / 255.0f;
    dst_rgb[2 * dst_w * dst_h + ty * dst_w + tx] = b / 255.0f;
}

// Host wrapper function
extern "C"
void gpu_crop_and_process(NvBufSurfaceParams* surf, float* trt_input, int dst_w, int dst_h, float cx, float cy, float cw, float ch, cudaStream_t stream) {
    // Assuming NV12 format: Y plane followed by UV plane
    // In NvBufSurface, dataPtr usually points to Y. UV offset depends on implementation.
    // For simplicity, we assume standard NV12 layout where UV follows Y immediately if contiguous,
    // but NvBufSurface often has separate offsets. 
    // In a real implementation, we would check surf->planeParams.offset[0] and [1].
    
    uint8_t* d_y = (uint8_t*)surf->dataPtr;
    // Approximation: UV starts after height * pitch. 
    // REAL IMPLEMENTATION MUST USE surf->planeParams.offset[1]
    uint8_t* d_uv = d_y + surf->height * surf->pitch; 

    dim3 block(32, 32);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    nv12_to_rgb_norm_kernel<<<grid, block, 0, stream>>>(
        d_y, d_uv, surf->pitch,
        trt_input, dst_w, dst_h,
        cx, cy, cw, ch
    );
}
