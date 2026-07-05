cmake_minimum_required(VERSION 3.16)

# -----------------------------------------------------------------------------
# Default build type for single-config generators
# -----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

project(ultratrack VERSION 1.0.0 LANGUAGES CXX)

# -----------------------------------------------------------------------------
# C++ standard
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Build options
# -----------------------------------------------------------------------------
option(BUILD_TESTS "Build unit tests" ON)
option(ULTRATRACK_WITH_ONNXRUNTIME "Enable ONNX Runtime backend" OFF)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
find_package(OpenCV 4.5 REQUIRED)
find_package(Threads REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

if(ULTRATRACK_WITH_ONNXRUNTIME)
    find_package(OnnxRuntime REQUIRED)
endif()

# -----------------------------------------------------------------------------
# Install directories (must be defined before subdirectories use them)
# -----------------------------------------------------------------------------
include(GNUInstallDirs)

# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tools)

if(BUILD_TESTS)
    # Always pass --output-on-failure to the built-in `test` / `RUN_TESTS` target.
    # Note: CMake reserves the lowercase target name `test`, so we do not create
    # a custom one. On multi-config generators use `RUN_TESTS` or run `ctest`
    # directly (e.g. `ctest --test-dir build --output-on-failure -C Release`).
    set(CMAKE_CTEST_ARGUMENTS --output-on-failure)

    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Install / export
# -----------------------------------------------------------------------------

install(TARGETS ultratrack EXPORT ultratrack-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT ultratrack-targets FILE ultratrack-targets.cmake
    NAMESPACE ultratrack:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ultratrack)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/ultratrack-config.cmake.in
    ${CMAKE_BINARY_DIR}/ultratrack-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ultratrack)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/ultratrack-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_BINARY_DIR}/ultratrack-config.cmake
    ${CMAKE_BINARY_DIR}/ultratrack-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ultratrack)

# -----------------------------------------------------------------------------
# Configuration summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "UltraTrack configuration summary:")
message(STATUS "  OpenCV version:     ${OpenCV_VERSION}")
message(STATUS "  Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "  SIMD AVX2:          ${ULTRATRACK_SIMD_AVX2}")
message(STATUS "  SIMD SSE4.2:        ${ULTRATRACK_SIMD_SSE42}")
message(STATUS "  SIMD NEON:          ${ULTRATRACK_SIMD_NEON}")
message(STATUS "")
