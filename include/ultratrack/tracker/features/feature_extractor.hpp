// include/ultratrack/tracker/features/feature_extractor.hpp
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
