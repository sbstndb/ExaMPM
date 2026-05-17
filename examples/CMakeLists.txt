add_executable( DamBreak dam_break.cpp )
target_link_libraries( DamBreak PRIVATE exampm)

add_executable( FreeFall free_fall.cpp )
target_link_libraries( FreeFall PRIVATE exampm)

if(BUILD_TESTING)
  set(EXAMPM_TEST_BACKEND "Serial" CACHE STRING
    "Execution backend used by ExaMPM example tests")
  set(EXAMPM_TEST_ARGS 0.05 2 0 0.001 0.25 100)

  add_test(NAME DamBreak COMMAND DamBreak ${EXAMPM_TEST_ARGS} ${EXAMPM_TEST_BACKEND})
  set_tests_properties(DamBreak PROPERTIES
    LABELS "examples;DamBreak;${EXAMPM_TEST_BACKEND}"
    TIMEOUT 300)

  add_test(NAME FreeFall COMMAND FreeFall ${EXAMPM_TEST_ARGS} ${EXAMPM_TEST_BACKEND})
  set_tests_properties(FreeFall PROPERTIES
    LABELS "examples;FreeFall;${EXAMPM_TEST_BACKEND}"
    TIMEOUT 300)
endif()

install(TARGETS DamBreak FreeFall DESTINATION ${CMAKE_INSTALL_BINDIR})
