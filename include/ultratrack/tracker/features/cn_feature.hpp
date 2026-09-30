// include/ultratrack/tracker/features/cn_feature.hpp
#ifndef ULTRATRACK_CN_FEATURE_HPP
#define ULTRATRACK_CN_FEATURE_HPP

#include "feature_extractor.hpp"
#include <array>

namespace ultratrack {

class CNFeature : public IFeatureExtractor {
public:
    CNFeature();
    ~CNFeature() override = default;
    
    cv::Mat extract(const cv::Mat& patch) const override;
    int dimensions() const override { return 10; }
    std::string name() const override { return "ColorNames"; }
    
    // Color name indices
    enum ColorName {
        BLACK = 0, BLUE, BROWN, GREY, GREEN,
        ORANGE, PINK, PURPLE, RED, WHITE
    };

private:
    static constexpr int LOOKUP_SIZE = 32768;  // 32^3
    std::vector<std::array<float, 10>> lookup_table_;
    bool lookup_loaded_ = false;
    
    void load_or_generate_lookup();
    bool load_lookup_from_file(const std::string& path);
    void generate_lookup_table();
    void save_lookup_to_file(const std::string& path) const;
    std::array<float, 10> compute_cn_probabilities(int r, int g, int b) const;
};

} // namespace ultratrack

#endif // ULTRATRACK_CN_FEATURE_HPP
