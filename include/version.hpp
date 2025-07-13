#ifndef ULTRATRACK_VERSION_HPP
#define ULTRATRACK_VERSION_HPP

#define ULTRATRACK_VERSION_MAJOR 1
#define ULTRATRACK_VERSION_MINOR 0
#define ULTRATRACK_VERSION_PATCH 0
#define ULTRATRACK_VERSION_STRING "1.0.0"

namespace ultratrack {
    struct Version {
        static constexpr int major = ULTRATRACK_VERSION_MAJOR;
        static constexpr int minor = ULTRATRACK_VERSION_MINOR;
        static constexpr int patch = ULTRATRACK_VERSION_PATCH;
        static constexpr const char* string = ULTRATRACK_VERSION_STRING;
        
        static const char* get_version_string() {
            return ULTRATRACK_VERSION_STRING;
        }
        
        static const char* get_build_info() {
            return "UltraTracker " ULTRATRACK_VERSION_STRING 
                   " built on " __DATE__ " " __TIME__;
        }
    };
}

#endif // ULTRATRACK_VERSION_HPP
