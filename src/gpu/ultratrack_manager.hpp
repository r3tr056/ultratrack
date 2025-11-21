#pragma once
#include "TrtEngine.hpp"
#include "GpuMemoryPool.hpp"
#include <memory>
#include <vector>

// Forward declaration
struct NvBufSurfaceParams;

class UltraTrackManager {
public:
    struct Config {
        std::string nanotrack_engine;
        std::string yolo_engine;
        std::string osnet_engine;
        int batch_size = 1;
    };

    UltraTrackManager(const Config& config);
    ~UltraTrackManager();

    // Main entry point for GStreamer
    void process_batch(void* nvbuf_surface); // void* to avoid NvBufSurface dependency in header if possible

private:
    Config config_;
    std::unique_ptr<TrtEngine> nanotrack_;
    std::unique_ptr<TrtEngine> yolo_;
    std::unique_ptr<TrtEngine> osnet_;
    std::unique_ptr<GpuMemoryPool> pool_;

    // Tracking state
    bool is_tracking_ = false;
    float target_bbox_[4]; // x, y, w, h
    
    // Few-shot Gallery
    std::vector<std::vector<float>> gallery_features_;
    const float reid_threshold_ = 0.7f;

    void run_detection(void* surface_params, cudaStream_t stream);
    void run_tracking(void* surface_params, cudaStream_t stream);
    
    // Helpers
    void load_gallery_from_disk(const std::string& path);
    void extract_feature(void* surface_params, const float* bbox, std::vector<float>& feature, cudaStream_t stream);
    float compute_similarity(const std::vector<float>& f1, const std::vector<float>& f2);
}