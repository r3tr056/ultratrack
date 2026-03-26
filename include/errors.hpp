// include/errors.hpp
#ifndef ULTRATRACK_ERRORS_HPP
#define ULTRATRACK_ERRORS_HPP

#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>

namespace ultratrack {

enum class ErrorCode {
    OK = 0,
    INVALID_PATCH_SIZE,
    EMPTY_FRAME,
    FEATURE_EXTRACTION_FAILED,
    SCALE_ESTIMATION_FAILED,
    CN_LOOKUP_LOAD_FAILED,
    INVALID_CONFIG
};

class TrackerException : public std::runtime_error {
public:
    TrackerException(ErrorCode code, const std::string& msg)
        : std::runtime_error(msg), code_(code) {}
    
    ErrorCode code() const { return code_; }
    
private:
    ErrorCode code_;
};

inline void validate_patch(const cv::Mat& patch, const std::string& context) {
    if (patch.empty()) {
        throw TrackerException(ErrorCode::EMPTY_FRAME, 
            context + ": patch is empty");
    }
    if (patch.cols < 4 || patch.rows < 4) {
        throw TrackerException(ErrorCode::INVALID_PATCH_SIZE,
            context + ": patch too small (min 4x4)");
    }
}

} // namespace ultratrack

#endif // ULTRATRACK_ERRORS_HPP
