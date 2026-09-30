add_executable(benchmark_cli benchmark_cli.cpp)
target_link_libraries(benchmark_cli PRIVATE ultratrack)

install(TARGETS benchmark_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
