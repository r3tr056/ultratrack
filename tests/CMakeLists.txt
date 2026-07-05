find_package(Catch2 3 REQUIRED)

add_executable(ultratrack_tests
    test_displacement.cpp
    test_scale.cpp
    test_features.cpp
    test_tracker.cpp
    core/test_result.cpp
    core/test_math.cpp
)

target_link_libraries(ultratrack_tests PRIVATE ultratrack Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(ultratrack_tests)
