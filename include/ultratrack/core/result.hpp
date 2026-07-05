#pragma once
#include <string>
#include <variant>
#include <optional>

namespace ultratrack {

enum class ErrorCode {
    OK = 0,
    INVALID_ARGUMENT,
    NOT_INITIALIZED,
    ALREADY_INITIALIZED,
    LICENSE_INVALID,
    LICENSE_EXPIRED,
    MODEL_LOAD_FAILED,
    INFERENCE_FAILED,
    EMPTY_FRAME,
    INVALID_PATCH_SIZE,
    FEATURE_EXTRACTION_FAILED,
    TRACKING_LOST,
    PTZ_CONNECTION_FAILED,
    CONFIG_PARSE_FAILED,
    NOT_IMPLEMENTED,
    UNKNOWN
};

class Status {
public:
    Status() : code_(ErrorCode::OK) {}
    Status(ErrorCode code, std::string message)
        : code_(code), message_(std::move(message)) {}

    bool ok() const { return code_ == ErrorCode::OK; }
    ErrorCode code() const { return code_; }
    const std::string& message() const { return message_; }

private:
    ErrorCode code_;
    std::string message_;
};

template <typename T>
class Result {
public:
    Result(T value) : data_(std::move(value)) {}
    Result(Status error) : data_(std::move(error)) {}

    bool has_value() const { return std::holds_alternative<T>(data_); }
    bool ok() const { return has_value(); }

    T& value() { return std::get<T>(data_); }
    const T& value() const { return std::get<T>(data_); }
    Status& error() { return std::get<Status>(data_); }
    const Status& error() const { return std::get<Status>(data_); }

    template <typename F>
    auto map(F&& f) -> Result<std::invoke_result_t<F, T&>> {
        if (has_value()) {
            return f(value());
        }
        return error();
    }

private:
    std::variant<T, Status> data_;
};

} // namespace ultratrack
