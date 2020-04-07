Introduction
============

The [Simd Library](http://ermig1979.github.io/Simd) is a free open source image processing and machine learning library, designed for C and C++ programmers. 
It provides many useful high performance algorithms for image processing such as: 
pixel format conversion, image scaling and filtration, extraction of statistic information from images, motion detection,
object detection (HAAR and LBP classifier cascades) and classification, neural network.

The algorithms are optimized with using of different SIMD CPU extensions. 
In particular the library supports following CPU extensions: 
SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2 and AVX-512 for x86/x64, VMX(Altivec) and VSX(Power7) for PowerPC (big-endian), NEON for ARM.

The Simd Library has C API and also contains useful C++ classes and functions to facilitate access to C API. 
The library supports dynamic and static linking, 32-bit and 64-bit Windows, Android and Linux, 
MSVS, G++ and Clang compilers, MSVS project and CMake build systems.

Library folder's structure
==========================

The Simd Library has next folder's structure:

* `simd/src/Simd/` - contains source codes of the library.
* `simd/src/Test/` - contains test framework of the library.
* `simd/src/Use/` - contains the use examples of the library.
* `simd/prj/vs2013/` - contains project files of Microsoft Visual Studio 2013.
* `simd/prj/vs2015/` - contains project files of Microsoft Visual Studio 2015.
* `simd/prj/vs2017w/` - contains project files of Microsoft Visual Studio 2017 (for Windows).
* `simd/prj/vs2017a/` - contains project files of Microsoft Visual Studio 2017 (for Android).
* `simd/prj/vs2019/` - contains project files of Microsoft Visual Studio 2019.
* `simd/prj/cmd/` - contains additional scripts needed for building of the library in Windows.
* `simd/prj/cmake/` - contains files of CMake build systems.
* `simd/prj/sh/` - contains additional scripts needed for building of the library in Linux.
* `simd/prj/txt/` - contains text files needed for building of the library.
* `simd/data/cascade/` - contains OpenCV cascades (HAAR and LBP).
* `simd/data/image/` - contains image samples.
* `simd/data/network/` - contains examples of trained networks.
* `simd/docs/` - contains documentation of the library.

The library building for Windows
================================

To build the library and test application for Windows 32/64 you need to use Microsoft Visual Studio 2019 (or 2013/2015/2017). 
The project files are in the directory: 

`simd/prj/vs2015/`

By default the library is built as a DLL (Dynamic Linked Library).
You also may build it as a static library. 
To do this you must change appropriate property (Configuration Type) of **Simd** project and also uncomment `#define SIMD_STATIC` in file:

`simd/src/Simd/SimdConfig.h`

Also in order to build the library you can use CMake and MinGW:

    cd .\prj\cmake
    cmake . -DTOOLCHAIN="your_toolchain\bin\g++" -DTARGET="x86_64" -DCMAKE_BUILD_TYPE="Release" -DLIBRARY="STATIC" -G "MinGW Makefiles"
    mingw32-make

The library building for Android
================================

To build the library and test application for Android(x86, x64, ARM, ARM64) you need to use Microsoft Visual Studio 2017. 
The project files are in the directory: 

`simd/prj/vs2017a/`

By default the library is built as a SO (Dynamic Library).

The library building for Linux
==============================

To build the library and test application for Linux 32/64 you need to use CMake build systems.
Files of CMake build systems are placed in the directory:

`simd/prj/cmake/`
	
The library can be built for x86/x64, PowerPC(64, big-endian) and ARM(32/64) platforms with using of G++ or Clang compilers.
With using of native compiler (g++) for current platform it is simple:

	cd ./prj/cmake
	cmake . -DTOOLCHAIN="" -DTARGET=""
	make
	
To build the library for PowerPC(64, big-endian) and ARM(32/64) platforms you can also use toolchain for cross compilation.
There is an example of using for PowerPC (64 bit, big-endian):

	cd ./prj/cmake
	cmake . -DTOOLCHAIN="/your_toolchain/usr/bin/powerpc-linux-gnu-g++" -DTARGET="ppc64" -DCMAKE_BUILD_TYPE="Release" -DLIBRARY="STATIC"
	make
	
For ARM (32 bit):

	cd ./prj/cmake
	cmake . -DTOOLCHAIN="/your_toolchain/usr/bin/arm-linux-gnueabihf-g++" -DTARGET="arm" -DCMAKE_BUILD_TYPE="Release" -DLIBRARY="STATIC"
	make
	
And for ARM (64 bit):

    cd ./prj/cmake
    cmake . -DTOOLCHAIN="/your_toolchain/usr/bin/aarch64-linux-gnu-g++" -DTARGET="aarch64" -DCMAKE_BUILD_TYPE="Release" -DLIBRARY="STATIC"
    make

As result the library and the test application will be built in the current directory.

The library using
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
	
Interaction with OpenCV
=======================

If you need use mutual conversion between Simd and OpenCV types you just have to define macro `SIMD_OPENCV_ENABLE` before including of Simd headers:
    
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
Also it can be `-m=c` (creation of test data for cross-platform testing), `-m=v` (cross-platform testing with using of early prepared test data)
and `-m=s` (running of special tests).
* `-tt=1` - a number of test threads.
* `-fi=Sobel` - an include filter. In current case will be tested only functions which contain word 'Sobel' in their names. 
If you miss this parameter then full testing will be performed.
You can use several filters - function name has to satisfy at least one of them.
* `-ot=log.txt` - a file name with test report (in TEXT file format). The test's report also will be output to console.
    
Also you can use parameters:

* `-help` or `-?` in order to print help message.
* `-r=../..` to set project root directory.
* `-pa=1` to print alignment statistics.
* `-c=512` a number of channels in test image for performance testing.
* `-h=1080` a height of test image for performance testing.
* `-w=1920` a width of test image for performance testing.
* `-oh=log.html` - a file name with test report (in HTML file format).	
* `-s=sample.avi` a video source (See `Simd::Motion` test).
* `-wt=1` a thread number used to parallelize algorithms.
* `-fe=Abs` an exclude filter to exclude some tests.
* `-mt=100` a minimal test execution time (in milliseconds).
