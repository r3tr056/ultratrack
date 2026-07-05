find_package(Catch2 3 REQUIRED)

add_executable(ultratrack_tests
    test_public_api.cpp
    test_displacement.cpp
    test_scale.cpp
    test_features.cpp
    test_tracker.cpp
    core/test_result.cpp
    core/test_math.cpp
    detector/test_opencv_dnn_backend.cpp
    tracker/test_kcf_tracker.cpp
    tracker/test_kalman.cpp
    tracking_engine/test_track_manager.cpp
    tracking_engine/test_data_association.cpp
)

target_link_libraries(ultratrack_tests PRIVATE ultratrack Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(ultratrack_tests)
