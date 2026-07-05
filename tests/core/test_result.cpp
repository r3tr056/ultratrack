#include <catch2/catch_test_macros.hpp>
#include <ultratrack/core/result.hpp>

using namespace ultratrack;

TEST_CASE("Result holds a value", "[core]") {
    Result<int> r = 42;
    REQUIRE(r.has_value());
    REQUIRE(r.value() == 42);
}

TEST_CASE("Result holds an error", "[core]") {
    Status s(ErrorCode::INVALID_ARGUMENT, "bad input");
    Result<int> r = s;
    REQUIRE(!r.has_value());
    REQUIRE(r.error().code() == ErrorCode::INVALID_ARGUMENT);
    REQUIRE(r.error().message() == "bad input");
}

TEST_CASE("Result value_or returns default on error", "[core]") {
    Status s(ErrorCode::INVALID_ARGUMENT, "bad input");
    Result<int> r = s;
    REQUIRE(!r.has_value());
    REQUIRE(r.value_or(0) == 0);
}

TEST_CASE("Result error_or returns default on value", "[core]") {
    Result<int> r = 42;
    REQUIRE(r.has_value());
    Status default_status(ErrorCode::UNKNOWN, "default");
    Status err = r.error_or(default_status);
    REQUIRE(err.code() == ErrorCode::UNKNOWN);
    REQUIRE(err.message() == "default");
}

TEST_CASE("Result map transforms value", "[core]") {
    Result<int> r = 21;
    auto doubled = r.map([](int x) { return x * 2; });
    REQUIRE(doubled.value() == 42);
}

TEST_CASE("Result map propagates error", "[core]") {
    Status s(ErrorCode::NOT_INITIALIZED, "not ready");
    Result<int> r = s;
    auto doubled = r.map([](int x) { return x * 2; });
    REQUIRE(!doubled.has_value());
    REQUIRE(doubled.error().code() == ErrorCode::NOT_INITIALIZED);
}

TEST_CASE("Default Status is OK", "[core]") {
    Status s;
    REQUIRE(s.ok());
    REQUIRE(s.code() == ErrorCode::OK);
}
