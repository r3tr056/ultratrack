cmake_minimum_required(VERSION 3.16)
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
# Targets
# -----------------------------------------------------------------------------
add_subdirectory(src)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Install / export
# -----------------------------------------------------------------------------
include(GNUInstallDirs)

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
