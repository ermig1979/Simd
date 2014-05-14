cmake_minimum_required(VERSION 2.8)

set(TRUNK_DIR ${CMAKE_SOURCE_DIR}/../..)

if(CMAKE_BUILD_TYPE STREQUAL "")
message("CMAKE_BUILD_TYPE variable is not defined. Set CMAKE_BUILD_TYPE = 'Release'.")
set(CMAKE_BUILD_TYPE "Release")
endif()

if((CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR))
set(EXECUTABLE_OUTPUT_PATH ${TRUNK_DIR}/bin/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${TRUNK_DIR}/bin/${CMAKE_BUILD_TYPE})
endif()

set(DEFAULT_CXX_FLAGS "${MAKE_CXX_FLAGS} -fPIC")

include_directories(${TRUNK_DIR}/src)

execute_process(COMMAND sh ${TRUNK_DIR}/prj/sh/GetVersion.sh)

project(SimdBase)
file(GLOB_RECURSE SIMD_BASE_SOURCES ${TRUNK_DIR}/src/Simd/SimdBase*.cpp)
add_library(SimdBase STATIC ${SIMD_BASE_SOURCES})

project(SimdSse2)
file(GLOB_RECURSE SIMD_SSE2_SOURCES ${TRUNK_DIR}/src/Simd/SimdSse2*.cpp)
set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS} -msse2")
add_library(SimdSse2 STATIC ${SIMD_SSE2_SOURCES})

project(SimdSsse3)
file(GLOB_RECURSE SIMD_SSSE3_SOURCES ${TRUNK_DIR}/src/Simd/SimdSsse3*.cpp)
set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS} -mssse3")
add_library(SimdSsse3 STATIC ${SIMD_SSSE3_SOURCES})

project(SimdSse41)
file(GLOB_RECURSE SIMD_SSE41_SOURCES ${TRUNK_DIR}/src/Simd/SimdSse41*.cpp)
set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS} -msse4.1")
add_library(SimdSse41 STATIC ${SIMD_SSE41_SOURCES})

project(SimdSse42)
file(GLOB_RECURSE SIMD_SSE42_SOURCES ${TRUNK_DIR}/src/Simd/SimdSse42*.cpp)
set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS} -msse4.2")
add_library(SimdSse42 STATIC ${SIMD_SSE42_SOURCES})

project(SimdAvx2)
file(GLOB_RECURSE SIMD_AVX2_SOURCES ${TRUNK_DIR}/src/Simd/SimdAvx2*.cpp)
set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS} -mavx2")
add_library(SimdAvx2 STATIC ${SIMD_AVX2_SOURCES})

project(Simd)
file(GLOB_RECURSE SIMD_SOURCES ${TRUNK_DIR}/src/Simd/SimdLib.cpp)
add_library(Simd SHARED ${SIMD_SOURCES})
target_link_libraries(Simd SimdBase SimdSse2 SimdSsse3 SimdSse41 SimdSse42 SimdAvx2)

project(Test)
set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS} -mavx2 -std=c++11")
file(GLOB_RECURSE TEST_SOURCES ${TRUNK_DIR}/src/Test/*.cpp)
add_executable(Test ${TEST_SOURCES})
target_link_libraries(Test Simd)

