add_executable(exampm_unit_tests unit_tests.cpp)
target_link_libraries(exampm_unit_tests PRIVATE exampm)

add_test(NAME unit_core
  COMMAND exampm_unit_tests ${EXAMPM_TEST_BACKEND})
set_tests_properties(unit_core PROPERTIES
  LABELS "unit;core;${EXAMPM_TEST_BACKEND}"
  TIMEOUT 60)
