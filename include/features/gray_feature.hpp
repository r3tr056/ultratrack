// include/features/gray_feature.hpp
#ifndef ULTRATRACK_GRAY_FEATURE_HPP
#define ULTRATRACK_GRAY_FEATURE_HPP

#include "feature_extractor.hpp"

namespace ultratrack {

class GrayFeature : public IFeatureExtractor {
public:
    GrayFeature() = default;
    ~GrayFeature() override = default;
    
    cv::Mat extract(const cv::Mat& patch) const override;
    int dimensions() const override { return 1; }
    std::string name() const override { return "Gray"; }
};

} // namespace ultratrack

#endif // ULTRATRACK_GRAY_FEATURE_HPP
