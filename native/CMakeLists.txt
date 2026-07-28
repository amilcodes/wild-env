cmake_minimum_required(VERSION 3.20)
project(aeolus_native LANGUAGES CXX)

add_library(aeolus_native SHARED src/reference_kernel.cpp)
target_include_directories(aeolus_native PUBLIC include)
target_compile_features(aeolus_native PUBLIC cxx_std_20)
target_compile_options(aeolus_native PRIVATE -Wall -Wextra -Wpedantic)

add_executable(aeolus_native_smoke tests/smoke.cpp)
target_link_libraries(aeolus_native_smoke PRIVATE aeolus_native)

enable_testing()
add_test(NAME aeolus_native_smoke COMMAND aeolus_native_smoke)
