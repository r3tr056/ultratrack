#pragma once
#include <ultratrack/core/types.hpp>

namespace ultratrack {

float IoU(const Rect2f& a, const Rect2f& b);
Rect2f clamp(const Rect2f& r, const Size2f& frame_size);
Point2f center(const Rect2f& r);

} // namespace ultratrack
