file(GLOB_RECURSE SIMD_BASE_SRC ${SIMD_ROOT}/src/Simd/SimdBase*.cpp)
set_source_files_properties(${SIMD_BASE_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS}")

file(GLOB_RECURSE SIMD_VMX_SRC ${SIMD_ROOT}/src/Simd/SimdVmx*.cpp)
 set_source_files_properties(${SIMD_VMX_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -maltivec")

file(GLOB_RECURSE SIMD_VSX_SRC ${SIMD_ROOT}/src/Simd/SimdVsx*.cpp)
set_source_files_properties(${SIMD_VSX_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mvsx")

file(GLOB_RECURSE SIMD_LIB_SRC ${SIMD_ROOT}/src/Simd/SimdLib.cpp)
set_source_files_properties(${SIMD_LIB_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mvsx")
add_library(Simd ${SIMD_LIB_TYPE} ${SIMD_LIB_SRC} ${SIMD_BASE_SRC} ${SIMD_VMX_SRC} ${SIMD_VSX_SRC})

if(SIMD_TEST)
	file(GLOB_RECURSE TEST_SRC_C ${SIMD_ROOT}/src/Test/*.c)
	file(GLOB_RECURSE TEST_SRC_CPP ${SIMD_ROOT}/src/Test/*.cpp)
	if(NOT ${SIMD_TARGET} STREQUAL "")
		set_source_files_properties(${TEST_SRC_CPP} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} -mvsx")
	else()
		set_source_files_properties(${TEST_SRC_CPP} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${SIMD_TEST_FLAGS} -mtune=native")
	endif()
	add_executable(Test ${TEST_SRC_C} ${TEST_SRC_CPP})
	target_link_libraries(Test Simd -lpthread -lstdc++ -lm)
endif()