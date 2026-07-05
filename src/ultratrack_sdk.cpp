#include <ultratrack/ultratrack_sdk.hpp>
#include <spdlog/spdlog.h>

namespace ultratrack {

namespace {
    bool g_initialized = false;
    bool g_licensed = false;

    spdlog::level::level_enum to_spdlog_level(LogLevel level) {
        switch (level) {
            case LogLevel::TRACE:    return spdlog::level::trace;
            case LogLevel::DEBUG:    return spdlog::level::debug;
            case LogLevel::INFO:     return spdlog::level::info;
            case LogLevel::WARN:     return spdlog::level::warn;
            case LogLevel::ERROR:    return spdlog::level::err;
            case LogLevel::CRITICAL: return spdlog::level::critical;
        }
        return spdlog::level::info;
    }
} // namespace

Status UltraTrackerSDK::initialize(const SDKConfig& cfg) {
    try {
        if (g_initialized) {
            return Status(ErrorCode::ALREADY_INITIALIZED, "SDK already initialized");
        }

        spdlog::set_level(to_spdlog_level(cfg.log_level));
        // Phase 1: accept TRIAL key; Phase 2 adds real license checks.
        g_licensed = (cfg.license_key == "TRIAL" || !cfg.license_key.empty());
        g_initialized = true;
        return Status();
    } catch (const std::exception& e) {
        return Status(ErrorCode::INTERNAL_ERROR, std::string("initialize failed: ") + e.what());
    }
}

void UltraTrackerSDK::shutdown() {
    g_initialized = false;
    g_licensed = false;
}

bool UltraTrackerSDK::is_licensed() { return g_initialized && g_licensed; }

} // namespace ultratrack
