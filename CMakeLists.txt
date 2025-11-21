cmake_minimum_required(VERSION 3.16)
project(ultratrack VERSION 1.0.0 LANGUAGES CXX)

# -----------------------------------------------------------------------------
# 1.  C++ standard and build-type
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS  OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -----------------------------------------------------------------------------
# 2.  Generic optimisation flags (kept from your original file)
# -----------------------------------------------------------------------------
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /arch:AVX2 /fp:fast")
    set(CMAKE_CXX_FLAGS_DEBUG   "/Od /Zi /D_DEBUG /W4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")      # parallel build
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math -funroll-loops -flto -fomit-frame-pointer")
    set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -D_DEBUG -Wall -Wextra")
endif()

# -----------------------------------------------------------------------------
# 3.  SIMD-flag detection  (fixed for MSVC)
# -----------------------------------------------------------------------------
include(CheckCXXCompilerFlag)

if(MSVC)
    # MSVC uses /arch:… instead of -m…
    check_cxx_compiler_flag("/arch:AVX2"  MSVC_AVX2_OK)
    check_cxx_compiler_flag("/arch:SSE2"  MSVC_SSE2_OK)     # SSE4.2 is implied
    if(MSVC_AVX2_OK)
        add_compile_options("/arch:AVX2")
        add_definitions(-D__AVX2__)
        set(COMPILER_SUPPORTS_AVX2 TRUE)
    elseif(MSVC_SSE2_OK)
        add_definitions(-D__SSE4_2__)        # best we can do on MSVC if AVX2 absent
        set(COMPILER_SUPPORTS_SSE42 TRUE)
    endif()
else()
    check_cxx_compiler_flag("-mavx2"  GCC_AVX2_OK)
    check_cxx_compiler_flag("-msse4.2" GCC_SSE42_OK)
    if(GCC_AVX2_OK)
        add_compile_options("-mavx2")
        add_definitions(-D__AVX2__)
        set(COMPILER_SUPPORTS_AVX2 TRUE)
    elseif(GCC_SSE42_OK)
        add_compile_options("-msse4.2")
        add_definitions(-D__SSE4_2__)
        set(COMPILER_SUPPORTS_SSE42 TRUE)
    endif()
endif()

# -----------------------------------------------------------------------------
# 4.  Silence CMP0146 ("FindCUDA module is removed") – keep your logic intact
# -----------------------------------------------------------------------------
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# -----------------------------------------------------------------------------
# 5.  OpenCV (unchanged, but prints a helpful hint if not found)
# -----------------------------------------------------------------------------
set(OpenCV_DIR "C:/Users/Ankur/sdk/opencv/build")   # keep custom path
find_package(OpenCV 4.5 REQUIRED CONFIG COMPONENTS
             core imgproc imgcodecs highgui dnn video videoio)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Install OpenCV 4.x and set OpenCV_DIR.")
endif()

find_package(Threads REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    config:   ${OpenCV_DIR}")
message(STATUS "    version:  ${OpenCV_VERSION}")
message(STATUS "    libraries:${OpenCV_LIBS}")
message(STATUS "    includes: ${OpenCV_INCLUDE_DIRS}")

# -----------------------------------------------------------------------------
# 6.  Optional CUDA (leave your manual search untouched)
# -----------------------------------------------------------------------------
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    set(CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
    message(STATUS "CUDA support enabled")
else()
    set(CUDA_LIBS "")
    message(STATUS "CUDA not found, using CPU-only mode")
endif()

# -----------------------------------------------------------------------------
# 7.  Sources / headers / include dirs
# -----------------------------------------------------------------------------
set(SRC_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INC_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/ultratrack.cpp
    ${SRC_DIR}/simd_optimization.cpp
)

set(HEADERS
    ${INC_DIR}/ultratrack.hpp
    ${INC_DIR}/version.hpp
)

add_executable(ultratrack ${SOURCES} ${HEADERS})

target_include_directories(ultratrack PRIVATE
    ${INC_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ultratrack PRIVATE
    ${OpenCV_LIBS}
    ${CUDA_LIBS}
    Threads::Threads
)

# -----------------------------------------------------------------------------
# 8.  Windows-specific tweaks (unchanged)
# -----------------------------------------------------------------------------
if(WIN32)
    target_compile_definitions(ultratrack PRIVATE
        _WIN32_WINNT=0x0601
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set_target_properties(ultratrack PROPERTIES
            WIN32_EXECUTABLE TRUE
            LINK_FLAGS "/SUBSYSTEM:CONSOLE"
        )
    endif()
endif()

# -----------------------------------------------------------------------------
# 9.  Runtime output & models folder
# -----------------------------------------------------------------------------
set_target_properties(ultratrack PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Release
)

add_custom_target(create_models_dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_BINARY_DIR}/Release/models
    COMMENT "Creating models directory"
)

# -----------------------------------------------------------------------------
# 10.  Install
# -----------------------------------------------------------------------------
install(TARGETS ultratrack
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY ${CMAKE_BINARY_DIR}/Release/models
        DESTINATION bin
        OPTIONAL)

# -----------------------------------------------------------------------------
# 11.  Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler:           ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  SIMD (AVX2):        ${COMPILER_SUPPORTS_AVX2}")
message(STATUS "  SIMD (SSE4.2):      ${COMPILER_SUPPORTS_SSE42}")
message(STATUS "  CUDA enabled:       ${CUDA_FOUND}")
message(STATUS "")

# -----------------------------------------------------------------------------
# 12.  GStreamer Plugin (UltraTrack Zero-Copy)
# -----------------------------------------------------------------------------
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(GST gstreamer-1.0 gstreamer-base-1.0)
    pkg_check_modules(GLIB glib-2.0)
endif()

if(CUDA_FOUND AND GST_FOUND AND GLIB_FOUND)
    message(STATUS "Building GStreamer Plugin (gstultratracker)...")
    
    set(GST_SOURCES
        src/gst/gstultratracker.cpp
        src/gpu/ultratrack_manager.cpp
        src/gpu/kernels.cu
    )
    
    add_library(gstultratracker SHARED ${GST_SOURCES})
    
    target_include_directories(gstultratracker PRIVATE
        src/gpu
        ${GST_INCLUDE_DIRS}
        ${GLIB_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        # Add DeepStream includes if available
        "/opt/nvidia/deepstream/deepstream/sources/includes" 
    )
    
    target_link_libraries(gstultratracker PRIVATE
        ${GST_LIBRARIES}
        ${GLIB_LIBRARIES}
        ${CUDA_LIBRARIES}
        nvinfer
        nvbufsurface
        cudart
    )
    
    install(TARGETS gstultratracker DESTINATION lib/gstreamer-1.0)
else()
    message(STATUS "Skipping GStreamer Plugin (Missing CUDA or GStreamer)")
endif()

