# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

set(CXX_HVX_FLAG "-mhvx -mhvx-length=128B")

file(GLOB_RECURSE SIMD_BASE_SRC ${SIMD_ROOT}/src/Simd/SimdBase*.cpp)
set_source_files_properties(${SIMD_BASE_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS}")

file(GLOB_RECURSE SIMD_HVX_SRC ${SIMD_ROOT}/src/Simd/SimdHvx*.cpp)
set_source_files_properties(${SIMD_HVX_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${CXX_HVX_FLAG}")

file(GLOB_RECURSE SIMD_LIB_SRC ${SIMD_ROOT}/src/Simd/SimdLib.cpp)
set_source_files_properties(${SIMD_LIB_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${CXX_HVX_FLAG}")
add_library(Simd ${SIMD_LIB_TYPE} ${SIMD_LIB_SRC} ${SIMD_BASE_SRC} ${SIMD_HVX_SRC})

if(SIMD_TEST)
	# Work around QEMU Hexagon emulation bug: test code compiled at -O2 or
	# higher triggers misemulation of certain instruction sequences (the
	# stack-coloring pass at -O2 produces code that QEMU handles incorrectly).
	# The library itself remains at full optimization; only test harness code
	# is affected.  Cap test files at -O1 until the QEMU fix is available.
	string(REGEX REPLACE "-O[23s]" "-O1" TEST_CXX_FLAGS "${COMMON_CXX_FLAGS}")
	file(GLOB_RECURSE TEST_SRC_C ${SIMD_ROOT}/src/Test/*.c)
	file(GLOB_RECURSE TEST_SRC_CPP ${SIMD_ROOT}/src/Test/*.cpp)
	set_source_files_properties(${TEST_SRC_CPP} PROPERTIES COMPILE_FLAGS "${TEST_CXX_FLAGS} ${CXX_HVX_FLAG} -D_GLIBCXX_USE_NANOSLEEP")
	add_executable(Test ${TEST_SRC_C} ${TEST_SRC_CPP})
	target_link_libraries(Test Simd -lpthread -lstdc++ -lm)
	if(SIMD_OPENCV)
		target_compile_definitions(Test PUBLIC SIMD_OPENCV_ENABLE)
		target_link_libraries(Test ${OpenCV_LIBS})
		target_include_directories(Test PUBLIC ${OpenCV_INCLUDE_DIRS})
	endif()
endif()
