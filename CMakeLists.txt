cmake_minimum_required(VERSION 3.16)
project(ultratrack VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type to Release for better performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set compiler flags for optimization
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /arch:AVX2 /fp:fast")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /D_DEBUG /W4")
    # Enable parallel compilation
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math -funroll-loops")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -D_DEBUG -Wall -Wextra")
    # Enable additional optimizations for GCC/Clang
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto -fomit-frame-pointer")
endif()

# Detect architecture for SIMD optimizations
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)

if(COMPILER_SUPPORTS_AVX2)
    add_definitions(-D__AVX2__)
    if(NOT MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    endif()
elseif(COMPILER_SUPPORTS_SSE42)
    add_definitions(-D__SSE4_2__)
    if(NOT MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
    endif()
endif()

# Set the directory containing OpenCVConfig.cmake
set(OpenCV_DIR "C:/Users/Ankur/sdk/opencv/build")  # Adjust to match your actual path

# Find required packages
find_package(OpenCV REQUIRED CONFIG COMPONENTS core imgproc imgcodecs highgui dnn video videoio)
find_package(Threads REQUIRED)

# Print OpenCV information
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Optional CUDA support
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

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/main.cpp
    src/ultratrack.cpp
    src/simd_optimization.cpp
)

# Headers
set(HEADERS
    include/ultratrack.hpp
)

# Create executable
add_executable(ultratrack ${SOURCES} ${HEADERS})

# Set include directories
target_include_directories(ultratrack PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(ultratrack PRIVATE 
    ${OpenCV_LIBS}
    ${CUDA_LIBS}
    Threads::Threads
)

# Set target properties
set_target_properties(ultratrack PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Release
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(ultratrack PRIVATE 
        _WIN32_WINNT=0x0601
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
    # Set subsystem for release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set_target_properties(ultratrack PROPERTIES
            WIN32_EXECUTABLE TRUE
            LINK_FLAGS "/SUBSYSTEM:CONSOLE"
        )
    endif()
endif()

# Add custom target for creating models directory
add_custom_target(create_models_dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/Release/models
    COMMENT "Creating models directory"
)

# Installation
install(TARGETS ultratrack
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install models directory
install(DIRECTORY ${CMAKE_BINARY_DIR}/Release/models
    DESTINATION bin
    OPTIONAL
)

# Create a configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "  CXX flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "  CXX flags (Debug): ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "  SIMD support: ${COMPILER_SUPPORTS_AVX2} (AVX2), ${COMPILER_SUPPORTS_SSE42} (SSE4.2)")
message(STATUS "  CUDA support: ${CUDA_FOUND}")
message(STATUS "")
