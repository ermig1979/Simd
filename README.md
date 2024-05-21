Introduction
============

The [Simd Library](http://ermig1979.github.io/Simd) is a free open source image processing and machine learning library, designed for C and C++ programmers. 
It provides many useful high performance algorithms for image processing such as: 
pixel format conversion, image scaling and filtration, extraction of statistic information from images, motion detection,
object detection and classification, neural network.

The algorithms are optimized with using of different SIMD CPU extensions. 
In particular the library supports following CPU extensions: 
SSE, AVX, AVX-512 and AMX for x86/x64, NEON for ARM.

The Simd Library has C API and also contains useful C++ classes and functions to facilitate access to C API. 
The library supports dynamic and static linking, 32-bit and 64-bit Windows and Linux, 
MSVS, G++ and Clang compilers, MSVS project and CMake build systems.

Library folder's structure
==========================

The Simd Library has next folder's structure:

* `simd/src/Simd/` - contains source codes of the library.
* `simd/src/Test/` - contains test framework of the library.
* `simd/src/Use/` - contains the use examples of the library.
* `simd/py/SimdPy/` - contains Python wrapper of the library.
* `simd/prj/vs2015/` - contains project files of Microsoft Visual Studio 2015.
* `simd/prj/vs2017/` - contains project files of Microsoft Visual Studio 2017.
* `simd/prj/vs2019/` - contains project files of Microsoft Visual Studio 2019.
* `simd/prj/vs2022/` - contains project files of Microsoft Visual Studio 2022.
* `simd/prj/cmd/` - contains additional scripts needed for building of the library in Windows.
* `simd/prj/cmake/` - contains files of CMake build systems.
* `simd/prj/sh/` - contains additional scripts needed for building of the library in Linux.
* `simd/prj/txt/` - contains text files needed for building of the library.
* `simd/data/cascade/` - contains OpenCV cascades (HAAR and LBP).
* `simd/data/image/` - contains image samples.
* `simd/data/network/` - contains examples of trained networks.
* `simd/docs/` - contains documentation of the library.

Building the library for Windows
================================

To build the library and test application for Windows 32/64 you need to use Microsoft Visual Studio 2022 (or 2015/2017/2019). 
The project files are in the directory: 

`simd/prj/vs2022/`

By default the library is built as a DLL (Dynamic Linked Library).
You also may build it as a static library. 
To do this you must change appropriate property (Configuration Type) of **Simd** project and also uncomment `#define SIMD_STATIC` in file:

`simd/src/Simd/SimdConfig.h`

Also in order to build the library you can use CMake and MinGW:

	mkdir build
	cd build
	cmake ..\prj\cmake -DSIMD_TOOLCHAIN="your_toolchain\bin\g++" -DSIMD_TARGET="x86_64" -DCMAKE_BUILD_TYPE="Release" -G "MinGW Makefiles"
	mingw32-make

Building the library for Linux
==============================

To build the library and test application for Linux 32/64 you need to use CMake build systems.
Files of CMake build systems are placed in the directory:

`simd/prj/cmake/`
	
The library can be built for x86/x64, ARM(32/64) platforms using the G++ or Clang compilers.
Using the native compiler (g++) for the current platform is simple:

	mkdir build
	cd build
	cmake ../prj/cmake -DSIMD_TOOLCHAIN="" -DSIMD_TARGET=""
	make
	
To build the library for ARM(32/64) platform you can also use a toolchain for cross compilation.
There is an example of using for ARM (32 bit):

	mkdir build
	cd build
	cmake ../prj/cmake -DSIMD_TOOLCHAIN="/your_toolchain/usr/bin/arm-linux-gnueabihf-g++" -DSIMD_TARGET="arm" -DCMAKE_BUILD_TYPE="Release"
	make

And for ARM (64 bit):

	mkdir build
	cd build
	cmake ../prj/cmake -DSIMD_TOOLCHAIN="/your_toolchain/usr/bin/aarch64-linux-gnu-g++" -DSIMD_TARGET="aarch64" -DCMAKE_BUILD_TYPE="Release"
	make

As result the library and the test application will be built in the current directory.

There are addition build parameters:

* `SIMD_AVX512` - Enable of AVX-512 (AVX-512F, AVX-512CD, AVX-512VL, AVX-512DQ, AVX-512BW) CPU extensions. It is switched on by default.
* `SIMD_AVX512VNNI` - Enable of AVX-512-VNNI CPU extensions. It is switched on by default.
* `SIMD_AMXBF16` - Enable of AMX-BF16, AMX-INT8 and AVX-512-BF16 CPU extensions. It is switched off by default.
* `SIMD_TEST` - Build test framework. It is switched on by default.
* `SIMD_INFO` - Print build information. It is switched on by default.
* `SIMD_PERF` - Enable of internal performance statistic. It is switched off by default.
* `SIMD_SHARED` - Build as SHARED library. It is switched off by default.
* `SIMD_GET_VERSION` - Call scipt to get Simd Library version. It is switched on by default.
* `SIMD_SYNET` - Enable optimizations for Synet framework. It is switched on by default.
* `SIMD_INT8_DEBUG` - Enable debug INT8 capabilities for Synet framework. It is switched off by default.
* `SIMD_HIDE` - Hide internal functions of Simd Library. It is switched off by default.
* `SIMD_RUNTIME` - Enable of runtime faster algorithm choise. It is switched on by default.
* `SIMD_TEST_FLAGS` - Addition compiler flags to build test framework.
* `SIMD_OPENCV` - Use OpenCV in test framework. It is switched off by default.
* `SIMD_INSTALL` - Enabling of install target. It is switched on by default.
* `SIMD_UNINSTALL` - Enabling of uninstall target. It is switched on by default.
* `SIMD_PYTHON` - Enabling of Simd Python wrapper. It is switched on by default.

Using the library
=================

If you use the library from C code you must include:
	
    #include "Simd/SimdLib.h"

And to use the library from C++ code you must include:

    #include "Simd/SimdLib.hpp"

In order to use [Simd::Detection](http://ermig1979.github.io/Simd/help/struct_simd_1_1_detection.html) you must include:

    #include "Simd/SimdDetection.hpp"
	
In order to use [Simd::Neural](http://ermig1979.github.io/Simd/help/namespace_simd_1_1_neural.html) you must include:

    #include "Simd/SimdNeural.hpp"
	
In order to use [Simd::Motion](http://ermig1979.github.io/Simd/help/namespace_simd_1_1_motion.html) you must include:

    #include "Simd/SimdMotion.hpp"

Package Managers
================

You can download and install simd using the [vcpkg](https://github.com/Microsoft/vcpkg) dependency manager:

    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh
    ./vcpkg integrate install
    ./vcpkg install simd

The simd port in vcpkg is kept up to date by Microsoft team members and community contributors. If the version is out of date, please [create an issue or pull request](https://github.com/Microsoft/vcpkg) on the vcpkg repository.

Interaction with OpenCV
=======================

If you need to use mutual conversion between Simd and OpenCV types you just have to define macro `SIMD_OPENCV_ENABLE` before including of Simd headers:
    
    #include <opencv2/core/core.hpp>
    #define SIMD_OPENCV_ENABLE
    #include "Simd/SimdLib.hpp"

And you can convert next types:
	
* `cv::Point`, `cv::Size` <--> `Simd::Point`.
* `cv::Rect` <--> `Simd::Rectangle`.
* `cv::Mat` <--> `Simd::View`.
	
Test Framework
==============

The test suite is needed for testing of correctness of work of the library and also for its performance testing.
There is a set of tests for every function from API of the library. 
There is an example of test application using:

	./Test -m=a -tt=1 -f=Sobel -ot=log.txt

Where next parameters were used:

* `-m=a` - a auto checking mode which includes performance testing (only for library built in Release mode). 
In this case different implementations of each functions will be compared between themselves 
(for example a scalar implementation and implementations with using of different SIMD instructions such as SSE2, AVX2, and other).
Also it can be `-m=s` (running of special tests).
* `-tt=1` - a number of test threads. Use -1 to set maximum parallelization.
* `-fi=Sobel` - an include filter. In current case will be tested only functions which contain word 'Sobel' in their names. 
If you miss this parameter then full testing will be performed.
You can use several filters - function name has to satisfy at least one of them.
* `-ot=log.txt` - a file name with test report (in TEXT file format). The test's report also will be output to console.
    
Also you can use parameters:

* `-help` or `-?` in order to print help message.
* `-r=../..` to set project root directory.
* `-pa=1` to print alignment statistics.
* `-pi=1` to print internal statistics (Cmake parameter SIMD_PERF must be ON).
* `-c=512` a number of channels in test image for performance testing.
* `-h=1080` a height of test image for performance testing.
* `-w=1920` a width of test image for performance testing.
* `-oh=log.html` - a file name with test report (in HTML file format).	
* `-s=sample.avi` a video source (See `Simd::Motion` test).
* `-o=output.avi` an annotated video output (See `Simd::Motion` test).
* `-wt=1` a thread number used to parallelize algorithms. Use -1 to set maximum parallelization.
* `-fe=Abs` an exclude filter to exclude some tests.
* `-mt=100` a minimal test execution time (in milliseconds).
* `-lc=1` to litter CPU cache between test runs.
* `-ri=city.jpg` a name of real image used in some tests. The image have to be placed in `./data/image` directory.
* `-tr=2` a number of test execution repeats.
* `-ts=1` to print statistics of time of tests execution.
* `-cc=1` to check c++ API.
* `-de=2` a flags of SIMD extensions which testing are disabled. Base - 1, 2 - SSE4.1/NEON, 4 - AVX2, 8 - AVX-512BW, 16 - AVX-512VNNI, 32 - AMX-BF16.
* `-wu=100` a time to warm up CPU before testing (in milliseconds).

