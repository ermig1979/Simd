cmake_minimum_required(VERSION 2.8)

set(TRUNK_DIR ${CMAKE_SOURCE_DIR}/../..)

if(CMAKE_BUILD_TYPE STREQUAL "")
    message("CMAKE_BUILD_TYPE variable is not defined. Set CMAKE_BUILD_TYPE = 'Release'.")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if((NOT TOOLCHAIN STREQUAL "") AND (NOT ${TARGET} STREQUAL ""))
    message("Use crossplatform toolchain '${TOOLCHAIN}' for '${TARGET}' platform.")
    set(CMAKE_C_COMPILER ${TOOLCHAIN})
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

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O0")
else()
    message(FATAL_ERROR "Unknown value of CMAKE_BUILD_TYPE!")
endif()	
	
if((CMAKE_SYSTEM_PROCESSOR STREQUAL "i686") OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"))

	file(GLOB_RECURSE SIMD_BASE_SRC ${TRUNK_DIR}/src/Simd/SimdBase*.cpp)
	set_source_files_properties(${SIMD_BASE_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS}")

    file(GLOB_RECURSE SIMD_SSE1_SRC ${TRUNK_DIR}/src/Simd/SimdSse1*.cpp)
	set_source_files_properties(${SIMD_SSE1_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -msse")

    file(GLOB_RECURSE SIMD_SSE2_SRC ${TRUNK_DIR}/src/Simd/SimdSse2*.cpp)
	set_source_files_properties(${SIMD_SSE2_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -msse2")

    file(GLOB_RECURSE SIMD_SSE3_SRC ${TRUNK_DIR}/src/Simd/SimdSse3*.cpp)
	set_source_files_properties(${SIMD_SSE3_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -msse3")

    file(GLOB_RECURSE SIMD_SSSE3_SRC ${TRUNK_DIR}/src/Simd/SimdSsse3*.cpp)
	set_source_files_properties(${SIMD_SSSE3_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mssse3")

    file(GLOB_RECURSE SIMD_SSE41_SRC ${TRUNK_DIR}/src/Simd/SimdSse41*.cpp)
	set_source_files_properties(${SIMD_SSE41_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -msse4.1")

    file(GLOB_RECURSE SIMD_SSE42_SRC ${TRUNK_DIR}/src/Simd/SimdSse42*.cpp)
	set_source_files_properties(${SIMD_SSE42_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -msse4.2")

    file(GLOB_RECURSE SIMD_AVX1_SRC ${TRUNK_DIR}/src/Simd/SimdAvx1*.cpp)
	set_source_files_properties(${SIMD_AVX1_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mavx")

    file(GLOB_RECURSE SIMD_AVX2_SRC ${TRUNK_DIR}/src/Simd/SimdAvx2*.cpp)
	if (CMAKE_CXX_COMPILER MATCHES "clang")
		set_source_files_properties(${SIMD_AVX2_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mavx2 -mfma -mf16c")
	else()
		set_source_files_properties(${SIMD_AVX2_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mavx2 -mfma -mf16c -fabi-version=4")
	endif()

    file(GLOB_RECURSE SIMD_LIB_SRC ${TRUNK_DIR}/src/Simd/SimdLib.cpp)
	set_source_files_properties(${SIMD_LIB_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mavx2")
    add_library(Simd STATIC ${SIMD_LIB_SRC} ${SIMD_BASE_SRC} ${SIMD_SSE1_SRC} ${SIMD_SSE2_SRC} ${SIMD_SSE3_SRC} ${SIMD_SSSE3_SRC} ${SIMD_SSE41_SRC} ${SIMD_SSE42_SRC} ${SIMD_AVX1_SRC} ${SIMD_AVX2_SRC})

	file(GLOB_RECURSE TEST_SRC_C ${TRUNK_DIR}/src/Test/*.c)
    file(GLOB_RECURSE TEST_SRC_CPP ${TRUNK_DIR}/src/Test/*.cpp)
    if((NOT (TARGET STREQUAL "")) AND (NOT (TOOLCHAIN STREQUAL "")))
		set_source_files_properties(${TEST_SRC_CPP} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mavx2 -std=c++11")
    else()
		set_source_files_properties(${TEST_SRC_CPP} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mtune=native -std=c++11")
    endif()	
    add_executable(Test ${TEST_SRC_C} ${TEST_SRC_CPP})
    target_link_libraries(Test Simd -lpthread)

elseif((CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc") OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64"))

	file(GLOB_RECURSE SIMD_BASE_SRC ${TRUNK_DIR}/src/Simd/SimdBase*.cpp)
	set_source_files_properties(${SIMD_BASE_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS}")

    file(GLOB_RECURSE SIMD_VMX_SRC ${TRUNK_DIR}/src/Simd/SimdVmx*.cpp)
 	set_source_files_properties(${SIMD_VMX_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -maltivec")

    file(GLOB_RECURSE SIMD_VSX_SRC ${TRUNK_DIR}/src/Simd/SimdVsx*.cpp)
	set_source_files_properties(${SIMD_VSX_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mvsx")

	file(GLOB_RECURSE SIMD_LIB_SRC ${TRUNK_DIR}/src/Simd/SimdLib.cpp)
	set_source_files_properties(${SIMD_LIB_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mvsx")
    add_library(Simd STATIC ${SIMD_LIB_SRC} ${SIMD_BASE_SRC} ${SIMD_VMX_SRC} ${SIMD_VSX_SRC})

	file(GLOB_RECURSE TEST_SRC_C ${TRUNK_DIR}/src/Test/*.c)
    file(GLOB_RECURSE TEST_SRC_CPP ${TRUNK_DIR}/src/Test/*.cpp)
    if((NOT (TARGET STREQUAL "")) AND (NOT (TOOLCHAIN STREQUAL "")))
		set_source_files_properties(${TEST_SRC_CPP} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mvsx -std=c++11")
    else()
		set_source_files_properties(${TEST_SRC_CPP} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mtune=native -std=c++11")
    endif()	
    add_executable(Test ${TEST_SRC_C} ${TEST_SRC_CPP})
    target_link_libraries(Test Simd -lpthread)

elseif((CMAKE_SYSTEM_PROCESSOR STREQUAL "arm") OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"))

    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm")
        set(CXX_NEON_FLAG "-mfpu=neon")
    else()
        set(CXX_NEON_FLAG "")
    endif() 

    file(GLOB_RECURSE SIMD_BASE_SRC ${TRUNK_DIR}/src/Simd/SimdBase*.cpp)
	set_source_files_properties(${SIMD_BASE_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS}")

    file(GLOB_RECURSE SIMD_NEON_SRC ${TRUNK_DIR}/src/Simd/SimdNeon*.cpp)
	set_source_files_properties(${SIMD_NEON_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${CXX_NEON_FLAG}")
	
	file(GLOB_RECURSE SIMD_LIB_SRC ${TRUNK_DIR}/src/Simd/SimdLib.cpp)
	set_source_files_properties(${SIMD_LIB_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${CXX_NEON_FLAG}")
    add_library(Simd STATIC ${SIMD_LIB_SRC} ${SIMD_BASE_SRC} ${SIMD_NEON_SRC})

	file(GLOB_RECURSE TEST_SRC_C ${TRUNK_DIR}/src/Test/*.c)
    file(GLOB_RECURSE TEST_SRC_CPP ${TRUNK_DIR}/src/Test/*.cpp)
    if((NOT (TARGET STREQUAL "")) AND (NOT (TOOLCHAIN STREQUAL "")))
		set_source_files_properties(${TEST_SRC_CPP} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${CXX_NEON_FLAG} -std=c++11 -D_GLIBCXX_USE_NANOSLEEP")
    else()
		set_source_files_properties(${TEST_SRC_CPP} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mtune=native -std=c++11 -D_GLIBCXX_USE_NANOSLEEP")
    endif()	
    add_executable(Test ${TEST_SRC_C} ${TEST_SRC_CPP})
    target_link_libraries(Test Simd -lpthread)

elseif((CMAKE_SYSTEM_PROCESSOR STREQUAL "mips"))

    set(COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Ofast -EL")

    set(CXX_MSA_FLAG "-mmsa")#"-mmsa -mhard-float -mfp64")
	
	file(GLOB_RECURSE SIMD_BASE_SRC ${TRUNK_DIR}/src/Simd/SimdBase*.cpp)
	set_source_files_properties(${SIMD_BASE_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS}")

    file(GLOB_RECURSE SIMD_MSA_SRC ${TRUNK_DIR}/src/Simd/SimdMsa*.cpp)
	set_source_files_properties(${SIMD_MSA_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${CXX_MSA_FLAG}")
	
	file(GLOB_RECURSE SIMD_LIB_SRC ${TRUNK_DIR}/src/Simd/SimdLib.cpp)
	set_source_files_properties(${SIMD_LIB_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${CXX_MSA_FLAG}")
    add_library(Simd STATIC ${SIMD_LIB_SRC} ${SIMD_BASE_SRC} ${SIMD_MSA_SRC})

	file(GLOB_RECURSE TEST_SRC_C ${TRUNK_DIR}/src/Test/*.c)
	set_source_files_properties(${TEST_SRC_C} PROPERTIES COMPILE_FLAGS "${COMMON_C_FLAGS} -EL")
    file(GLOB_RECURSE TEST_SRC_CPP ${TRUNK_DIR}/src/Test/*.cpp)
    if((NOT (TARGET STREQUAL "")) AND (NOT (TOOLCHAIN STREQUAL "")))
		set_source_files_properties(${TEST_SRC_CPP} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${CXX_MSA_FLAG} -std=c++11")# -D_GLIBCXX_USE_NANOSLEEP")
    else()
		set_source_files_properties(${TEST_SRC_CPP} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mtune=native -std=c++11")# -D_GLIBCXX_USE_NANOSLEEP")
    endif()	
    add_executable(Test ${TEST_SRC_C} ${TEST_SRC_CPP})
    target_link_libraries(Test Simd -lpthread "${CMAKE_MODULE_LINKER_FLAGS} -EL")

else()
    message(FATAL_ERROR "Unknown value of CMAKE_SYSTEM_PROCESSOR!")
endif()
