#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <stdexcept>
#include <iostream>

class GpuMemoryPool {
private:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };
    std::vector<Block> pool_;
    std::mutex mutex_;

public:
    GpuMemoryPool(size_t block_size, int block_count) {
        for (int i = 0; i < block_count; i++) {
            void* dev_ptr;
            if (cudaMalloc(&dev_ptr, block_size) != cudaSuccess) 
                throw std::runtime_error("CUDA OOM: Failed to allocate pool block");
            pool_.push_back({dev_ptr, block_size, false});
        }
        std::cout << "GpuMemoryPool initialized with " << block_count << " blocks of " << block_size << " bytes." << std::endl;
    }

    ~GpuMemoryPool() {
        for (auto& block : pool_) {
            if (block.ptr) cudaFree(block.ptr);
        }
    }

    void* acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& block : pool_) {
            if (!block.in_use) {
                block.in_use = true;
                return block.ptr;
            }
        }
        throw std::runtime_error("GPU Pool Exhausted - Frame Dropped");
    }

    void release(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& block : pool_) {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
        // If we get here, the pointer wasn't in our pool. 
        // In production, log a warning.
        std::cerr << "Warning: Attempted to release unknown pointer to GpuMemoryPool" << std::endl;
    }
};
