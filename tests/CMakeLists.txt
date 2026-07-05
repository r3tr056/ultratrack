find_package(Catch2 3 REQUIRED)

add_executable(ultratrack_tests
    test_public_api.cpp
    test_displacement.cpp
    test_scale.cpp
    test_features.cpp
    test_tracker.cpp
    benchmark/test_harness.cpp
    core/test_result.cpp
    core/test_math.cpp
    detector/test_opencv_dnn_backend.cpp
    tracker/test_kcf_tracker.cpp
    tracker/test_kalman.cpp
    tracking_engine/test_track_manager.cpp
    tracking_engine/test_data_association.cpp
)

target_link_libraries(ultratrack_tests PRIVATE ultratrack Catch2::Catch2WithMain)

# Make test data available in the build tree so tests can write generated fixtures
# next to the committed ground-truth file without polluting the source tree.
set(ULTRATRACK_TEST_DATA_DIR "${CMAKE_BINARY_DIR}/tests/data")
file(MAKE_DIRECTORY "${ULTRATRACK_TEST_DATA_DIR}/mot_dummy")
configure_file(
    "${CMAKE_SOURCE_DIR}/tests/data/mot_dummy/gt.txt"
    "${ULTRATRACK_TEST_DATA_DIR}/mot_dummy/gt.txt"
    COPYONLY)

target_compile_definitions(ultratrack_tests PRIVATE
    ULTRATRACK_TEST_DATA_DIR="${ULTRATRACK_TEST_DATA_DIR}")

include(Catch)
catch_discover_tests(ultratrack_tests)
