add_executable(ExaMPM_Benchmarks exampm_benchmarks.cpp)
target_link_libraries(ExaMPM_Benchmarks PRIVATE exampm)

if(BUILD_TESTING)
  add_test(NAME benchmark_dense_smoke
    COMMAND ExaMPM_Benchmarks ${EXAMPM_TEST_BACKEND} dense 10000 2)
  set_tests_properties(benchmark_dense_smoke PROPERTIES
    LABELS "benchmark;smoke;${EXAMPM_TEST_BACKEND}"
    TIMEOUT 60)
endif()

install(TARGETS ExaMPM_Benchmarks DESTINATION ${CMAKE_INSTALL_BINDIR})
