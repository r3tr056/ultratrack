set(ULTRATRACK_SOURCES
    ultratrack.cpp
    simd_optimization.cpp
    core/result.cpp
    core/types.cpp
    core/math.cpp
    detector/opencv_dnn_backend.cpp
    detector/detector_factory.cpp
    tracker/features/feature_extractor.cpp
    tracker/features/hog_feature.cpp
    tracker/features/gray_feature.cpp
    tracker/features/cn_feature.cpp
    tracker/displacement_predictor.cpp
    tracker/scale_estimator.cpp
    tracker/kcf_tracker.cpp
    tracker/simd_helpers.cpp
    tracker/kalman.cpp
    tracking_engine/track_lifecycle.cpp
    tracking_engine/track_manager.cpp
    tracking_engine/data_association.cpp
)

add_library(ultratrack STATIC ${ULTRATRACK_SOURCES})

target_include_directories(ultratrack
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(ultratrack
    PUBLIC
        ${OpenCV_LIBS}
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        Threads::Threads)

if(ULTRATRACK_WITH_ONNXRUNTIME)
    target_compile_definitions(ultratrack PUBLIC ULTRATRACK_WITH_ONNXRUNTIME)
    target_link_libraries(ultratrack PUBLIC OnnxRuntime::OnnxRuntime)
endif()

# -----------------------------------------------------------------------------
# SIMD / floating-point flags (detected per-target, never applied globally)
# -----------------------------------------------------------------------------
include(CheckCXXCompilerFlag)

set(ULTRATRACK_SIMD_FLAGS)
set(ULTRATRACK_SIMD_DEFS)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|x64|i[3-6]86|x86")
    if(MSVC)
        check_cxx_compiler_flag("/arch:AVX2" ULTRATRACK_AVX2_OK)
        if(ULTRATRACK_AVX2_OK)
            list(APPEND ULTRATRACK_SIMD_FLAGS "/arch:AVX2")
            list(APPEND ULTRATRACK_SIMD_DEFS "__AVX2__")
            set(COMPILER_SUPPORTS_AVX2 TRUE)
        elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
            # x64 MSVC: SSE2/SSE4.2 are baseline, so /arch:SSE2 is rejected.
            # Define the macro so source code can use SSE4.2 intrinsics.
            list(APPEND ULTRATRACK_SIMD_DEFS "__SSE4_2__")
            set(COMPILER_SUPPORTS_SSE42 TRUE)
        else()
            # 32-bit MSVC fallback: explicitly enable SSE2. The SSE4.2
            # intrinsics are still available once the macro is defined.
            check_cxx_compiler_flag("/arch:SSE2" ULTRATRACK_SSE2_OK)
            if(ULTRATRACK_SSE2_OK)
                list(APPEND ULTRATRACK_SIMD_FLAGS "/arch:SSE2")
                list(APPEND ULTRATRACK_SIMD_DEFS "__SSE4_2__")
                set(COMPILER_SUPPORTS_SSE42 TRUE)
            endif()
        endif()
    else()
        check_cxx_compiler_flag("-mavx2" ULTRATRACK_AVX2_OK)
        if(ULTRATRACK_AVX2_OK)
            list(APPEND ULTRATRACK_SIMD_FLAGS "-mavx2")
            list(APPEND ULTRATRACK_SIMD_DEFS "__AVX2__")
            set(COMPILER_SUPPORTS_AVX2 TRUE)
        else()
            check_cxx_compiler_flag("-msse4.2" ULTRATRACK_SSE42_OK)
            if(ULTRATRACK_SSE42_OK)
                list(APPEND ULTRATRACK_SIMD_FLAGS "-msse4.2")
                list(APPEND ULTRATRACK_SIMD_DEFS "__SSE4_2__")
                set(COMPILER_SUPPORTS_SSE42 TRUE)
            endif()
        endif()
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64|arm|armv7")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(COMPILER_SUPPORTS_NEON TRUE)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        check_cxx_compiler_flag("-mfpu=neon" ULTRATRACK_NEON_OK)
        if(ULTRATRACK_NEON_OK)
            list(APPEND ULTRATRACK_SIMD_FLAGS "-mfpu=neon")
            set(COMPILER_SUPPORTS_NEON TRUE)
        endif()
    endif()
    if(COMPILER_SUPPORTS_NEON)
        list(APPEND ULTRATRACK_SIMD_DEFS "__ARM_NEON" "__ARM_NEON__")
    endif()
endif()

if(ULTRATRACK_SIMD_FLAGS)
    target_compile_options(ultratrack PRIVATE ${ULTRATRACK_SIMD_FLAGS})
endif()
if(ULTRATRACK_SIMD_DEFS)
    target_compile_definitions(ultratrack PRIVATE ${ULTRATRACK_SIMD_DEFS})
endif()

if(MSVC)
    target_compile_options(ultratrack PRIVATE /fp:fast)
    target_compile_definitions(ultratrack PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0601)
else()
    target_compile_options(ultratrack PRIVATE -ffast-math)
endif()

# CLI executable that consumes the library (preserves the original entry point)
add_executable(ultratrack_cli main.cpp)
target_link_libraries(ultratrack_cli PRIVATE ultratrack)
set_target_properties(ultratrack_cli PROPERTIES
    OUTPUT_NAME ultratrack
)
if(WIN32)
    target_compile_definitions(ultratrack_cli PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0601)
    set_target_properties(ultratrack_cli PROPERTIES WIN32_EXECUTABLE FALSE)
endif()

# -----------------------------------------------------------------------------
# Runtime data / models directories (created next to the executable)
# -----------------------------------------------------------------------------
add_custom_command(TARGET ultratrack_cli POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
            $<TARGET_FILE_DIR:ultratrack_cli>/models
            $<TARGET_FILE_DIR:ultratrack_cli>/data
    COMMENT "Creating runtime models/data directories next to ultratrack_cli"
)

if(EXISTS ${CMAKE_SOURCE_DIR}/data/cn_lookup.bin)
    add_custom_command(TARGET ultratrack_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_SOURCE_DIR}/data/cn_lookup.bin
                $<TARGET_FILE_DIR:ultratrack_cli>/data/cn_lookup.bin
        COMMENT "Copying cn_lookup.bin next to ultratrack_cli"
    )
endif()

# Make summary variables visible in the top-level CMakeLists.txt
set(ULTRATRACK_SIMD_AVX2  ${COMPILER_SUPPORTS_AVX2}  PARENT_SCOPE)
set(ULTRATRACK_SIMD_SSE42 ${COMPILER_SUPPORTS_SSE42} PARENT_SCOPE)
set(ULTRATRACK_SIMD_NEON  ${COMPILER_SUPPORTS_NEON}  PARENT_SCOPE)
