add_executable( DamBreak dam_break.cpp )
target_link_libraries( DamBreak PRIVATE exampm)

add_executable( FreeFall free_fall.cpp )
target_link_libraries( FreeFall PRIVATE exampm)

if(BUILD_TESTING)
  set(EXAMPM_TEST_BACKEND "Serial" CACHE STRING
    "Execution backend used by ExaMPM example tests")
  set_property(CACHE EXAMPM_TEST_BACKEND PROPERTY STRINGS Serial OpenMP CUDA HIP)
  set(EXAMPM_EXAMPLE_TEST_ARGS 0.05 2 0 0.001 0.25 100 CACHE STRING
    "Arguments passed to ExaMPM example smoke tests before the backend")

  function(add_exampm_example_test test_name target_name)
    add_test(NAME ${test_name}
      COMMAND ${target_name} ${EXAMPM_EXAMPLE_TEST_ARGS} ${EXAMPM_TEST_BACKEND})
    set_tests_properties(${test_name} PROPERTIES
      LABELS "examples;${target_name};${EXAMPM_TEST_BACKEND}"
      TIMEOUT 300)
  endfunction()

  add_exampm_example_test(example_DamBreak DamBreak)
  add_exampm_example_test(example_FreeFall FreeFall)
endif()

install(TARGETS DamBreak FreeFall DESTINATION ${CMAKE_INSTALL_BINDIR})
