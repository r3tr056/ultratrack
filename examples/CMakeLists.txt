add_executable(simple_tracker simple_tracker.cpp)
target_link_libraries(simple_tracker PRIVATE ultratrack)
install(TARGETS simple_tracker RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
