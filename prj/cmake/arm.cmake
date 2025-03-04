if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (NOT(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.0.0")))
	set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} -Wno-psabi")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    if( NOT ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER MATCHES "clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")))
	  set(CXX_NEON_FLAG "-mfpu=neon -mfpu=neon-fp16")
	endif()
	if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
		set(CXX_NEON_FLAG "${CXX_NEON_FLAG} -mfp16-format=ieee")
	endif()
else()
	set(CXX_NEON_FLAG "")
endif()

if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER MATCHES "clang")  OR (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"))
	add_definitions(-DSIMD_NEON_FP16_DISABLE)
endif()

file(GLOB_RECURSE SIMD_BASE_SRC ${SIMD_ROOT}/src/Simd/SimdBase*.cpp)
set_source_files_properties(${SIMD_BASE_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS}")

file(GLOB_RECURSE SIMD_NEON_SRC ${SIMD_ROOT}/src/Simd/SimdNeon*.cpp)
set_source_files_properties(${SIMD_NEON_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${CXX_NEON_FLAG}")

file(GLOB_RECURSE SIMD_LIB_SRC ${SIMD_ROOT}/src/Simd/SimdLib.cpp)
set_source_files_properties(${SIMD_LIB_SRC} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${CXX_NEON_FLAG}")
add_library(Simd ${SIMD_LIB_TYPE} ${SIMD_LIB_SRC} ${SIMD_BASE_SRC} ${SIMD_NEON_SRC})

if(SIMD_TEST)
	file(GLOB_RECURSE TEST_SRC_C ${SIMD_ROOT}/src/Test/*.c)
	file(GLOB_RECURSE TEST_SRC_CPP ${SIMD_ROOT}/src/Test/*.cpp)
	if((NOT ${SIMD_TARGET} STREQUAL "") OR (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.0.0"))
		set_source_files_properties(${TEST_SRC_CPP} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${CXX_NEON_FLAG} -D_GLIBCXX_USE_NANOSLEEP")
	else()
		set_source_files_properties(${TEST_SRC_CPP} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${SIMD_TEST_FLAGS} -mtune=native -D_GLIBCXX_USE_NANOSLEEP")
	endif()
	add_executable(Test ${TEST_SRC_C} ${TEST_SRC_CPP})
	target_link_libraries(Test Simd -lpthread -lstdc++ -lm)
	if(SIMD_OPENCV)
		target_compile_definitions(Test PUBLIC SIMD_OPENCV_ENABLE)
		target_link_libraries(Test ${OpenCV_LIBS})
		target_include_directories(Test PUBLIC ${OpenCV_INCLUDE_DIRS})
	endif()
endif()
