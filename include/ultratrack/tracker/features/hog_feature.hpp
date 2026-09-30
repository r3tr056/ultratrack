// include/ultratrack/tracker/features/hog_feature.hpp
#ifndef ULTRATRACK_HOG_FEATURE_HPP
#define ULTRATRACK_HOG_FEATURE_HPP

#include "feature_extractor.hpp"

namespace ultratrack {

class HOGFeature : public IFeatureExtractor {
public:
    explicit HOGFeature(const FeatureConfig& config = {});
    ~HOGFeature() override = default;
    
    cv::Mat extract(const cv::Mat& patch) const override;
    int dimensions() const override { return 31; }
    std::string name() const override { return "HOG"; }

private:
    FeatureConfig config_;
    
    void compute_gradients(const cv::Mat& gray, cv::Mat& magnitude, cv::Mat& orientation) const;
    cv::Mat compute_hog_cells(const cv::Mat& magnitude, const cv::Mat& orientation, 
                               int cell_size, int num_bins) const;
};

} // namespace ultratrack

#endif // ULTRATRACK_HOG_FEATURE_HPP
