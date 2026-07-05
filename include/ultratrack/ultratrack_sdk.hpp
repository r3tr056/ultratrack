#pragma once
#include <ultratrack/core/result.hpp>
#include <string>

namespace ultratrack {

enum class LogLevel { TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL };

struct SDKConfig {
    std::string license_key;
    std::string activation_server_url;
    LogLevel log_level = LogLevel::INFO;
    std::string telemetry_endpoint;
};

class UltraTrackerSDK {
public:
    static Status initialize(const SDKConfig& cfg);
    static void shutdown();
    static bool is_licensed();
};

} // namespace ultratrack
