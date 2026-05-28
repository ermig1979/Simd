# Install script for directory: /tmp/workspace/ermig1979/Simd/prj/cmake

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/tmp/workspace/ermig1979/Simd/build-lite/libSimd.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/Simd" TYPE FILE FILES
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdLib.h"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdAllocator.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdContour.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdDetection.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdDrawing.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdFont.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdFrame.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdImageMatcher.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdLib.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdMotion.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdNeural.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdParallel.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdPixel.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdPoint.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdPyramid.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdRectangle.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdShift.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdView.hpp"
    "/tmp/workspace/ermig1979/Simd/prj/cmake/../../src/Simd/SimdXml.hpp"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/simd/simdConfig.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/simd/simdConfig.cmake"
         "/tmp/workspace/ermig1979/Simd/build-lite/CMakeFiles/Export/463bfcc5261162aa9888662043fdb51c/simdConfig.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/simd/simdConfig-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/simd/simdConfig.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/simd" TYPE FILE FILES "/tmp/workspace/ermig1979/Simd/build-lite/CMakeFiles/Export/463bfcc5261162aa9888662043fdb51c/simdConfig.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/simd" TYPE FILE FILES "/tmp/workspace/ermig1979/Simd/build-lite/CMakeFiles/Export/463bfcc5261162aa9888662043fdb51c/simdConfig-release.cmake")
  endif()
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/tmp/workspace/ermig1979/Simd/build-lite/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/tmp/workspace/ermig1979/Simd/build-lite/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
