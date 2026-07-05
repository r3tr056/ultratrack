#include <ultratrack/ultratrack_sdk.hpp>
#include <spdlog/spdlog.h>

namespace ultratrack {

namespace {
    bool g_initialized = false;
    bool g_licensed = false;
} // namespace

Status UltraTrackerSDK::initialize(const SDKConfig& cfg) {
    try {
        if (g_initialized) {
            return Status(ErrorCode::ALREADY_INITIALIZED, "SDK already initialized");
        }

        spdlog::set_level(static_cast<spdlog::level::level_enum>(cfg.log_level));
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
