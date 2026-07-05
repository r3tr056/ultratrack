set(ULTRATRACK_SOURCES
    ultratrack.cpp
    simd_optimization.cpp
    features/feature_extractor.cpp
    features/hog_feature.cpp
    features/gray_feature.cpp
    features/cn_feature.cpp
    tracking/displacement_predictor.cpp
    tracking/scale_estimator.cpp
)

add_library(ultratrack STATIC ${ULTRATRACK_SOURCES})

target_include_directories(ultratrack
    PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(ultratrack
    PUBLIC ${OpenCV_LIBS}
    PRIVATE spdlog::spdlog nlohmann_json::nlohmann_json Threads::Threads)

if(ULTRATRACK_WITH_ONNXRUNTIME)
    target_compile_definitions(ultratrack PRIVATE ULTRATRACK_WITH_ONNXRUNTIME)
    target_link_libraries(ultratrack PRIVATE OnnxRuntime::OnnxRuntime)
endif()

# Preserve existing SIMD / Windows compiler flags
if(MSVC)
    target_compile_options(ultratrack PRIVATE /arch:AVX2 /fp:fast)
    target_compile_definitions(ultratrack PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0601)
else()
    target_compile_options(ultratrack PRIVATE -march=native -ffast-math)
endif()

# CLI executable that consumes the library (preserves original entry point)
add_executable(ultratrack_cli main.cpp)
target_link_libraries(ultratrack_cli PRIVATE ultratrack)
set_target_properties(ultratrack_cli PROPERTIES OUTPUT_NAME ultratrack)
if(WIN32)
    target_compile_definitions(ultratrack_cli PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0601)
    set_target_properties(ultratrack_cli PROPERTIES WIN32_EXECUTABLE FALSE)
endif()
