#include <ultratrack/core/math.hpp>
#include <algorithm>

namespace ultratrack {

float IoU(const Rect2f& a, const Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

Point2f center(const Rect2f& r) {
    return Point2f(r.x + r.width / 2.0f, r.y + r.height / 2.0f);
}

Rect2f clamp(const Rect2f& r, const Size2f& frame_size) {
    float x1 = std::max(0.0f, r.x);
    float y1 = std::max(0.0f, r.y);
    float x2 = std::min(frame_size.width, r.x + r.width);
    float y2 = std::min(frame_size.height, r.y + r.height);
    return Rect2f(x1, y1, std::max(0.0f, x2 - x1), std::max(0.0f, y2 - y1));
}

} // namespace ultratrack
