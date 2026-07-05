#pragma once
#include <cassert>
#include <cstdlib>
#include <string>
#include <type_traits>
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

/// Specialization of Result for operations that produce no value on success.
template <>
class Result<void> {
public:
    Result() : status_() {}
    Result(Status error) : status_(std::move(error)) {}

    bool has_value() const { return status_.ok(); }
    bool ok() const { return status_.ok(); }

    /// No-op for the success state.
    void value() const noexcept {
        assert(status_.ok() && "Result<void>::value() called on an error Result");
    }

    /// Returns the contained error Status.
    Status& error() noexcept { return status_; }
    const Status& error() const noexcept { return status_; }

    /// Returns the contained error or @p default_status if this is a value.
    Status error_or(const Status& default_status) const noexcept {
        return status_.ok() ? default_status : status_;
    }

    /// Maps a successful Result<void> to a value-producing callable.
    template <typename F>
    auto map(F&& f) -> std::enable_if_t<!std::is_void_v<std::invoke_result_t<F>>,
                                        Result<std::invoke_result_t<F>>> {
        if (status_.ok()) {
            return f();
        }
        return status_;
    }

    /// Maps a successful Result<void> to a void-returning callable.
    template <typename F>
    auto map(F&& f) -> std::enable_if_t<std::is_void_v<std::invoke_result_t<F>>, Result<void>> {
        if (status_.ok()) {
            f();
            return Result<void>();
        }
        return status_;
    }

private:
    Status status_;
};

} // namespace ultratrack
