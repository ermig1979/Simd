cmake_minimum_required(VERSION 3.15)
project(test_simd CXX)

find_package(simd REQUIRED)

add_executable(test_simd test_simd.cpp)
target_link_libraries(test_simd PRIVATE simd::simd)
target_compile_features(test_simd PRIVATE cxx_std_11)
