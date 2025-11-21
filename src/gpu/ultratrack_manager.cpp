#include "ultratrack_manager.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>

// Extern declaration for the CUDA kernel wrapper
extern "C" void gpu_crop_and_process(struct NvBufSurfaceParams* surf, float* trt_input, 
                                      int dst_w, int dst_h, 
                                      float cx, float cy, float cw, float ch, 
                                      cudaStream_t stream);

UltraTrackManager::UltraTrackManager(const Config& config) : config_(config) {
    pool_ = std::make_unique<GpuMemoryPool>(1024 * 1024 * 16, 4); // 16MB blocks,4 count
    
    nanotrack_ = std::make_unique<TrtEngine>();
    if (!nanotrack_->initialize(config.nanotrack_engine)) {
        std::cerr << "Warning: Failed to load NanoTrack engine." << std::endl;
    }
 
    yolo_ = std::make_unique<TrtEngine>();
    if (! yolo_->initialize(config.yolo_engine)) {
        std::cerr << "Warning: Failed to load YOLO engine." << std::endl;
    }
    
    // OSNet on DLA Core 0
    osnet_ = std::make_unique<TrtEngine>();
    if (!osnet_->initialize(config.osnet_engine, 0)) {
        std::cerr << "Warning: Failed to load OSNet engine (DLA)." << std::endl;
    }
}

UltraTrackManager::~UltraTrackManager() {
    // Resources cleaned up by unique_ptrs
}

void UltraTrackManager::process_batch(void* nvbuf_surface) {
    // Cast nvbuf_surface to NvBufSurface* and get first surface
    struct NvBufSurface* surf = (struct NvBufSurface*)nvbuf_surface;
    struct NvBufSurfaceParams* params = (struct NvBufSurfaceParams*)surf->surfaceList;
    
    if (is_tracking_) {
        run_tracking(params, nanotrack_->get_stream());
    } else {
        run_detection(params, yolo_->get_stream());
    }
}

void UltraTrackManager::run_detection(void* surface_params, cudaStream_t stream) {
    struct NvBufSurfaceParams* params = (struct NvBufSurfaceParams*)surface_params;
    
    // 1. Acquire memory for YOLO input (640x640)
    float* d_input = (float*)pool_->acquire();
    
    // 2. Preprocess: NV12 -> RGB + Resize to 640x640
    gpu_crop_and_process(params, d_input, 640, 640, 
                         0, 0, params->width, params->height, stream);
    
    // 3. Acquire output buffer
    // YOLOv11 output: [1, 84, 8400] -> [batch, 4+80, anchors]
    const int num_anchors = 8400;
    const int num_attrs = 84;  // 4 bbox + 80 classes
    float* d_output = (float*)pool_->acquire();
    
    // 4. Run inference
    void* bindings[] = {d_input, d_output};
    yolo_->inference(bindings);
    
    // 5. Post-process: NMS on GPU
    std::vector<float> host_output(num_anchors * num_attrs);
    cudaMemcpyAsync(host_output.data(), d_output, 
                    host_output.size() * sizeof(float), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // 6. Parse detections
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    
    const float conf_thresh = 0.25f;
    for (int i = 0; i < num_anchors; ++i) {
        const float* row = host_output.data() + i * num_attrs;
        
        // YOLO format: cx, cy, w, h, ...class_scores
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        
        // Find max class score
        auto max_it = std::max_element(row + 4, row + num_attrs);
        float class_score = *max_it;
        
        if (class_score > conf_thresh) {
            int class_id = std::distance(row + 4, max_it);
            
            boxes.push_back(cv::Rect2f(cx - w/2, cy - h/2, w, h));
            scores.push_back(class_score);
            class_ids.push_back(class_id);
        }
    }
    
    // 7. NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_thresh, 0.45f, indices);
    
    // 8. If we have detections, initialize tracking
    if (!indices.empty()) {
        // Take the highest scoring detection
        int best_idx = indices[0];
        target_bbox_[0] = boxes[best_idx].x;
        target_bbox_[1] = boxes[best_idx].y;
        target_bbox_[2] = boxes[best_idx].width;
        target_bbox_[3] = boxes[best_idx].height;
        
        // Extract Re-ID feature using OSNet
        std::vector<float> reid_feature;
        extract_feature(surface_params, target_bbox_, reid_feature, osnet_->get_stream());
        
        // Check against gallery
        if (!gallery_features_.empty()) {
            float max_sim = 0.0f;
            for (const auto& gallery_feat : gallery_features_) {
                float sim = compute_similarity(reid_feature, gallery_feat);
                max_sim = std::max(max_sim, sim);
            }
            
            if (max_sim > reid_threshold_) {
                is_tracking_ = true;
                std::cout << "[UltraTrack] Target matched! Similarity: " << max_sim 
                          << ". Switching to tracking mode." << std::endl;
            }
        } else {
            // No gallery - track any detection
            is_tracking_ = true;
        }
    }
    
    pool_->release(d_output);
    pool_->release(d_input);
}

void UltraTrackManager::run_tracking(void* surface_params, cudaStream_t stream) {
    struct NvBufSurfaceParams* params = (struct NvBufSurfaceParams*)surface_params;
    
    // 1. Compute search region (2x bbox centered on predicted position)
    float search_scale = 2.5f;
    float search_cx = target_bbox_[0] + target_bbox_[2] / 2.0f;
    float search_cy = target_bbox_[1] + target_bbox_[3] / 2.0f;
    float search_w = target_bbox_[2] * search_scale;
    float search_h = target_bbox_[3] * search_scale;
    
    // Crop coordinates
    float crop_x = search_cx - search_w / 2.0f;
    float crop_y = search_cy - search_h / 2.0f;
    
    // Clamp to frame
    crop_x = std::max(0.0f, std::min(crop_x, (float)params->width - search_w));
    crop_y = std::max(0.0f, std::min(crop_y, (float)params->height - search_h));
    
    // 2. Acquire memory for NanoTrack input (255x255 typical)
    float* d_input = (float*)pool_->acquire();
    
    gpu_crop_and_process(params, d_input, 255, 255, crop_x, crop_y, search_w, search_h, stream);
    
    // 3. Run NanoTrack inference
    // Output: [1, 4] -> [x, y, w, h] relative to search region
    float* d_output = (float*)pool_->acquire();
    void* bindings[] = {d_input, d_output};
    nanotrack_->inference(bindings);
    
    // 4. Download result
    float tracked_bbox[4];
    cudaMemcpyAsync(tracked_bbox, d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // 5. Convert from search region coords to frame coords
    target_bbox_[0] = crop_x + tracked_bbox[0] * search_w;
    target_bbox_[1] = crop_y + tracked_bbox[1] * search_h;
    target_bbox_[2] = tracked_bbox[2] * search_w;
    target_bbox_[3] = tracked_bbox[3] * search_h;
    
    // 6. Check if tracking lost (bbox too small or out of bounds)
    if (target_bbox_[2] < 10 || target_bbox_[3] < 10 ||
        target_bbox_[0] < 0 || target_bbox_[1] < 0 ||
        target_bbox_[0] + target_bbox_[2] > params->width ||
        target_bbox_[1] + target_bbox_[3] > params->height) {
        is_tracking_ = false;
        std::cout << "[UltraTrack] Tracking lost. Switching back to detection." << std::endl;
    }
    
    pool_->release(d_output);
    pool_->release(d_input);
}

void UltraTrackManager::load_gallery_from_disk(const std::string& path) {
    std::vector<cv::String> filenames;
    cv::glob(path + "/*.jpg", filenames);
    if (filenames.empty()) {
        cv::glob(path + "/*.png", filenames);
    }
    
    std::cout << "[UltraTrack] Loading gallery from " << path << std::endl;
    
    for (const auto& f : filenames) {
        cv::Mat img = cv::imread(f);
        if (img.empty()) continue;
        
        // Upload to GPU
        float* d_img = (float*)pool_->acquire();
        
        // Convert and copy
        cv::Mat img_float;
        img.convertTo(img_float, CV_32FC3, 1.0/255.0);
        cudaMemcpy(d_img, img_float.data, img_float.total() * img_float.elemSize(), 
                   cudaMemcpyHostToDevice);
        
        // Extract feature
        std::vector<float> feat;
        float bbox[4] = {0, 0, (float)img.cols, (float)img.rows};
        
        // Create a dummy NvBufSurfaceParams (in real code this would be proper)
        struct NvBufSurfaceParams dummy_params;
        dummy_params.width = img.cols;
        dummy_params.height = img.rows;
        dummy_params.dataPtr = d_img;
        
        extract_feature(&dummy_params, bbox, feat, osnet_->get_stream());
        gallery_features_.push_back(feat);
        
        pool_->release(d_img);
    }
    
    std::cout << "[UltraTrack] Gallery loaded: " << gallery_features_.size() << " features." << std::endl;
}

void UltraTrackManager::extract_feature(void* surface_params, const float* bbox, 
                                        std::vector<float>& feature, cudaStream_t stream) {
    struct NvBufSurfaceParams* params = (struct NvBufSurfaceParams*)surface_params;
    
    // 1. Acquire memory
    float* d_input = (float*)pool_->acquire();
    
    // 2. Crop and Resize to OSNet input (256x128)
    gpu_crop_and_process(params, d_input, 128, 256, bbox[0], bbox[1], bbox[2], bbox[3], stream);
    
    // 3. Inference
    float* d_output = (float*)pool_->acquire();
    void* bindings[] = {d_input, d_output};
    
    osnet_->inference(bindings);
    
    // 4. Download feature
    int feat_dim = 512; // OSNet output dimension
    feature.resize(feat_dim);
    cudaMemcpyAsync(feature.data(), d_output, feat_dim * sizeof(float), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Normalize
    float norm = 0.0f;
    for (float v : feature) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (float& v : feature) v /= norm;
    }
    
    pool_->release(d_output);
    pool_->release(d_input);
}

float UltraTrackManager::compute_similarity(const std::vector<float>& f1, const std::vector<float>& f2) {
    if (f1.size() != f2.size() || f1.empty()) return 0.0f;
    
    float dot = 0.0f;
    for (size_t i = 0; i < f1.size(); ++i) {
        dot += f1[i] * f2[i];
    }
    
    // Features are already normalized, so dot product = cosine similarity
    return dot;
}
