#pragma once
#include <cassert>
#include <cstdlib>
#include <string>
#include <variant>

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
    UNKNOWN,
    INTERNAL_ERROR
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

    /// Returns the contained value.
    /// @pre The Result is in the value state (has_value() == true).
    ///      Calling this on an error Result terminates the program.
    T& value() noexcept {
        T* ptr = std::get_if<T>(&data_);
        if (ptr == nullptr) {
            assert(false && "Result<T>::value() called on an error Result");
            std::terminate();
        }
        return *ptr;
    }

    /// Returns the contained value.
    /// @pre The Result is in the value state (has_value() == true).
    ///      Calling this on an error Result terminates the program.
    const T& value() const noexcept {
        const T* ptr = std::get_if<T>(&data_);
        if (ptr == nullptr) {
            assert(false && "Result<T>::value() called on an error Result");
            std::terminate();
        }
        return *ptr;
    }

    /// Returns the contained error Status.
    /// @pre The Result is in the error state (has_value() == false).
    ///      Calling this on a value Result terminates the program.
    Status& error() noexcept {
        Status* ptr = std::get_if<Status>(&data_);
        if (ptr == nullptr) {
            assert(false && "Result<T>::error() called on a value Result");
            std::terminate();
        }
        return *ptr;
    }

    /// Returns the contained error Status.
    /// @pre The Result is in the error state (has_value() == false).
    ///      Calling this on a value Result terminates the program.
    const Status& error() const noexcept {
        const Status* ptr = std::get_if<Status>(&data_);
        if (ptr == nullptr) {
            assert(false && "Result<T>::error() called on a value Result");
            std::terminate();
        }
        return *ptr;
    }

    /// Returns the contained value or @p default_value if this is an error.
    T value_or(const T& default_value) const noexcept {
        const T* ptr = std::get_if<T>(&data_);
        return ptr != nullptr ? *ptr : default_value;
    }

    /// Returns the contained error or @p default_status if this is a value.
    Status error_or(const Status& default_status) const noexcept {
        const Status* ptr = std::get_if<Status>(&data_);
        return ptr != nullptr ? *ptr : default_status;
    }

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
