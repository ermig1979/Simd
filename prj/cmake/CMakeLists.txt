cmake_minimum_required(VERSION 2.8)

set(TRUNK_DIR ${CMAKE_SOURCE_DIR}/../..)

if(CMAKE_BUILD_TYPE STREQUAL "")
 message("CMAKE_BUILD_TYPE variable is not defined. Set CMAKE_BUILD_TYPE = 'Release'.")
 set(CMAKE_BUILD_TYPE "Release")
endif()

if((NOT TOOLCHAIN STREQUAL "") AND (NOT ${TARGET} STREQUAL ""))
 message("Use crossplatform toolchain '${TOOLCHAIN}' for '${TARGET}' platform.")
 set(CMAKE_CXX_COMPILER ${TOOLCHAIN})
 set(CMAKE_SYSTEM_PROCESSOR ${TARGET})
else()
 message("Use default toolchain '${CMAKE_CXX_COMPILER}' for current '${CMAKE_SYSTEM_PROCESSOR}' platform.")
endif()

include_directories(${TRUNK_DIR}/src)

if((CHECK_VERSION STREQUAL "") OR (NOT (CHECK_VERSION STREQUAL "0")))
 execute_process(COMMAND bash ${TRUNK_DIR}/prj/sh/GetVersion.sh)
else()
 message("Skip version checking.")
endif()

message("CMAKE_SYSTEM_PROCESSOR = '${CMAKE_SYSTEM_PROCESSOR}'")

if((CMAKE_SYSTEM_PROCESSOR STREQUAL "i686") OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"))

set(COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3")

file(GLOB_RECURSE SIMD_BASE_SOURCES ${TRUNK_DIR}/src/Simd/SimdBase*.cpp)
add_library(SimdBase STATIC ${SIMD_BASE_SOURCES})
set_target_properties(SimdBase PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS}")

file(GLOB_RECURSE SIMD_SSE1_SOURCES ${TRUNK_DIR}/src/Simd/SimdSse1*.cpp)
add_library(SimdSse1 STATIC ${SIMD_SSE1_SOURCES})
set_target_properties(SimdSse1 PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -msse")

file(GLOB_RECURSE SIMD_SSE2_SOURCES ${TRUNK_DIR}/src/Simd/SimdSse2*.cpp)
add_library(SimdSse2 STATIC ${SIMD_SSE2_SOURCES})
set_target_properties(SimdSse2 PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -msse2")

file(GLOB_RECURSE SIMD_SSE3_SOURCES ${TRUNK_DIR}/src/Simd/SimdSse3*.cpp)
add_library(SimdSse3 STATIC ${SIMD_SSE3_SOURCES})
set_target_properties(SimdSse3 PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -msse3")

file(GLOB_RECURSE SIMD_SSSE3_SOURCES ${TRUNK_DIR}/src/Simd/SimdSsse3*.cpp)
add_library(SimdSsse3 STATIC ${SIMD_SSSE3_SOURCES})
set_target_properties(SimdSsse3 PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mssse3")

file(GLOB_RECURSE SIMD_SSE41_SOURCES ${TRUNK_DIR}/src/Simd/SimdSse41*.cpp)
add_library(SimdSse41 STATIC ${SIMD_SSE41_SOURCES})
set_target_properties(SimdSse41 PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -msse4.1")

file(GLOB_RECURSE SIMD_SSE42_SOURCES ${TRUNK_DIR}/src/Simd/SimdSse42*.cpp)
add_library(SimdSse42 STATIC ${SIMD_SSE42_SOURCES})
set_target_properties(SimdSse42 PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -msse4.2")

file(GLOB_RECURSE SIMD_AVX1_SOURCES ${TRUNK_DIR}/src/Simd/SimdAvx1*.cpp)
add_library(SimdAvx1 STATIC ${SIMD_AVX1_SOURCES})
set_target_properties(SimdAvx1 PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mavx")

file(GLOB_RECURSE SIMD_AVX2_SOURCES ${TRUNK_DIR}/src/Simd/SimdAvx2*.cpp)
add_library(SimdAvx2 STATIC ${SIMD_AVX2_SOURCES})
set_target_properties(SimdAvx2 PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mavx2 -mfma")

file(GLOB_RECURSE SIMD_SOURCES ${TRUNK_DIR}/src/Simd/SimdLib.cpp)
add_library(Simd STATIC ${SIMD_SOURCES})
target_link_libraries(Simd SimdBase SimdSse1 SimdSse2 SimdSse3 SimdSsse3 SimdSse41 SimdSse42 SimdAvx1 SimdAvx2)
set_target_properties(Simd PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mavx2")

add_library(TestCheckC STATIC ${TRUNK_DIR}/src/Test/TestCheckC.c)

file(GLOB_RECURSE TEST_SOURCES ${TRUNK_DIR}/src/Test/*.cpp)
add_executable(Test ${TEST_SOURCES})
target_link_libraries(Test Simd TestCheckC -lpthread)
if((NOT (TARGET STREQUAL "")) AND (NOT (TOOLCHAIN STREQUAL "")))
 set_target_properties(Test PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mavx2 -std=c++11")
else()
 set_target_properties(Test PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mtune=native -std=c++11")
endif()

elseif((CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc") OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64"))

set(COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3")

file(GLOB_RECURSE SIMD_BASE_SOURCES ${TRUNK_DIR}/src/Simd/SimdBase*.cpp)
add_library(SimdBase STATIC ${SIMD_BASE_SOURCES})
set_target_properties(SimdBase PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -std=c++03")

file(GLOB_RECURSE SIMD_VMX_SOURCES ${TRUNK_DIR}/src/Simd/SimdVmx*.cpp)
add_library(SimdVmx STATIC ${SIMD_VMX_SOURCES})
set_target_properties(SimdVmx PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -maltivec -std=c++03")

file(GLOB_RECURSE SIMD_VSX_SOURCES ${TRUNK_DIR}/src/Simd/SimdVsx*.cpp)
add_library(SimdVsx STATIC ${SIMD_VSX_SOURCES})
set_target_properties(SimdVsx PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mvsx -std=c++03")

file(GLOB_RECURSE SIMD_SOURCES ${TRUNK_DIR}/src/Simd/SimdLib.cpp)
add_library(Simd STATIC ${SIMD_SOURCES})
target_link_libraries(Simd SimdBase SimdVmx SimdVsx)
set_target_properties(Simd PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mvsx -std=c++03")

add_library(TestCheckC STATIC ${TRUNK_DIR}/src/Test/TestCheckC.c)

file(GLOB_RECURSE TEST_SOURCES ${TRUNK_DIR}/src/Test/*.cpp)
add_executable(Test ${TEST_SOURCES})
target_link_libraries(Test Simd TestCheckC -lpthread)
if((NOT (TARGET STREQUAL "")) AND (NOT (TOOLCHAIN STREQUAL "")))
 set_target_properties(Test PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mvsx -std=c++11")
else()
 set_target_properties(Test PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mtune=native -std=c++11")
endif()

elseif((CMAKE_SYSTEM_PROCESSOR STREQUAL "arm") OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"))

set(COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3")

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm")
set(CXX_NEON_FLAG "-mfpu=neon")
else()
set(CXX_NEON_FLAG "")
endif() 

file(GLOB_RECURSE SIMD_BASE_SOURCES ${TRUNK_DIR}/src/Simd/SimdBase*.cpp)
add_library(SimdBase STATIC ${SIMD_BASE_SOURCES})
set_target_properties(SimdBase PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS}")

file(GLOB_RECURSE SIMD_NEON_SOURCES ${TRUNK_DIR}/src/Simd/SimdNeon*.cpp)
add_library(SimdNeon STATIC ${SIMD_NEON_SOURCES})
set_target_properties(SimdNeon PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${CXX_NEON_FLAG}")

file(GLOB_RECURSE SIMD_SOURCES ${TRUNK_DIR}/src/Simd/SimdLib.cpp)
add_library(Simd STATIC ${SIMD_SOURCES})
target_link_libraries(Simd SimdBase SimdNeon)
set_target_properties(Simd PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${CXX_NEON_FLAG}")

add_library(TestCheckC STATIC ${TRUNK_DIR}/src/Test/TestCheckC.c)

file(GLOB_RECURSE TEST_SOURCES ${TRUNK_DIR}/src/Test/*.cpp)
add_executable(Test ${TEST_SOURCES})
target_link_libraries(Test Simd TestCheckC -lpthread)
if((NOT (TARGET STREQUAL "")) AND (NOT (TOOLCHAIN STREQUAL "")))
 set_target_properties(Test PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${CXX_NEON_FLAG} -std=c++11 -D_GLIBCXX_USE_NANOSLEEP")
else()
 set_target_properties(Test PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mtune=native -std=c++11 -D_GLIBCXX_USE_NANOSLEEP")
endif()

else()
message(FATAL_ERROR "Unknown value of CMAKE_SYSTEM_PROCESSOR!")
endif()
