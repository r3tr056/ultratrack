#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <ultratrack/core/math.hpp>

using namespace ultratrack;

TEST_CASE("IoU of identical rectangles is 1", "[core]") {
    Rect2f r(0, 0, 10, 10);
    REQUIRE(IoU(r, r) == Catch::Approx(1.0f));
}

TEST_CASE("IoU of non-overlapping rectangles is 0", "[core]") {
    Rect2f a(0, 0, 10, 10);
    Rect2f b(20, 20, 10, 10);
    REQUIRE(IoU(a, b) == 0.0f);
}

TEST_CASE("IoU of partially overlapping rectangles", "[core]") {
    Rect2f a(0, 0, 10, 10);
    Rect2f b(5, 5, 10, 10);
    float inter = 25.0f;
    float uni = 100.0f + 100.0f - 25.0f;
    REQUIRE(IoU(a, b) == Catch::Approx(inter / uni));
}

TEST_CASE("Center of rectangle", "[core]") {
    Rect2f r(10, 20, 30, 40);
    Point2f c = center(r);
    REQUIRE(c.x == Catch::Approx(25.0f));
    REQUIRE(c.y == Catch::Approx(40.0f));
}

TEST_CASE("Clamp rectangle inside frame", "[core]") {
    Rect2f r(-5, -5, 20, 20);
    Size2f frame(100, 100);
    Rect2f clamped = clamp(r, frame);
    REQUIRE(clamped.x >= 0);
    REQUIRE(clamped.y >= 0);
    REQUIRE(clamped.br().x <= frame.width);
    REQUIRE(clamped.br().y <= frame.height);
}

TEST_CASE("Clamp does not change rectangle already inside frame", "[core]") {
    Rect2f r(10, 10, 20, 20);
    Size2f frame(100, 100);
    Rect2f clamped = clamp(r, frame);
    REQUIRE(clamped.x == 10);
    REQUIRE(clamped.y == 10);
    REQUIRE(clamped.width == 20);
    REQUIRE(clamped.height == 20);
}
