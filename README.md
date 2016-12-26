#Introduction

The [Simd Library](http://simd.sourceforge.net) is a free open source image processing library, designed for C and C++ programmers. 
It provides many useful high performance algorithms for image processing such as: 
pixel format conversion, image scaling and filtration, extraction of statistic information from images, motion detection,
object detection (HAAR and LBP classifier cascades) and classification, neural network.

The algorithms are optimized with using of different SIMD CPU extensions. 
In particular the library supports following CPU extensions: 
SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX and AVX2 for x86/x64, VMX(Altivec) and VSX(Power7) for PowerPC, NEON for ARM.

The Simd Library has C API and also contains useful C++ classes and functions to facilitate access to C API. 
The library supports dynamic and static linking, 32-bit and 64-bit Windows and Linux, 
MSVS and G++ compilers, MSVS project and CMake build systems.

#Library folder's structure

The Simd Library has next folder's structure:

* `simd/src/Simd/` - contains source codes of the library.
* `simd/src/Test/` - contains test framework of the library.
* `simd/prj/vs14/` - contains project files of Microsoft Visual Studio 2015.
* `simd/prj/cmd/` - contains additional scripts needed for building of the library in Windows.
* `simd/prj/cmake/` - contains files of CMake build systems.
* `simd/prj/sh/` - contains additional scripts needed for building of the library in Linux.
* `simd/prj/txt/` - contains text files needed for building of the library.
* `simd/data/cascade/` - contains OpenCV cascades (HAAR and LBP).
* `simd/data/image/` - contains image samples.
* `simd/data/network/` - contains examples of trained networks.
* `simd/doc/` - contains documentation of the library.
* `simd/doc/src/` - contains scripts for generation of this documentation by doxygen.

#The library building for Windows

To build the library and test application for Windows 32/64 you need to use Microsoft Visual Studio 2015. 
These project files are in the directory: 

`simd/prj/vs14/`

By default the library is built as a DLL (Dynamic Linked Library).
You also may build it as a static library. 
To do this you must change appropriate property (Configuration Type) of **Simd** project and also uncomment \#define SIMD_STATIC in file:

`simd/src/Simd/SimdConfig.h`

#The library building for Linux

To build the library and test application for Linux 32/64 you need to use CMake build systems.
Files of CMake build systems are placed in the directory:

`simd/prj/cmake/`
	
The library can be built for x86/x64, PowerPC and ARM platforms.
With using of native compiler (g++) for current platform it is simple:

	cd ./prj/cmake
	cmake . -DTOOLCHAIN="" -DTARGET=""
	make
	
To build the library for PowePC and ARM platforms you can also use toolchain for cross compilation.
There is an example of using for PowerPC:

	cd ./prj/cmake
	cmake . -DTOOLCHAIN="/path_to_your_toolchain/usr/bin/powerpc-linux-g++" -DTARGET="ppc64" -DCMAKE_BUILD_TYPE="Release"
	make
	
And for ARM:

	cd ./prj/cmake
	cmake . -DTOOLCHAIN="/path_to_your_toolchain/usr/bin/arm-linux-gnueabihf-g++" -DTARGET="arm" -DCMAKE_BUILD_TYPE="Release"
	make

As result the library and the test application will be built in the current directory.

#The library using

If you use the library from C code you must include:
	
    #include "Simd/SimdLib.h"

And to use the library from C++ code you must include:

    #include "Simd/SimdLib.hpp"

In order to use [Simd::Detection](http://simd.sourceforge.net/help/struct_simd_1_1_detection.html) you must include:

    #include "Simd/SimdDetection.hpp"
	
#Test Framework

The test suite is needed for testing of correctness of work of the library and also for its performance testing.
There is a set of tests for every function from API of the library. 
There is an example of test application using:

	./Test -m=a -t=1 -f=Sobel -o=log.txt

Where next parameters were used:

* `-m=a` - a auto checking mode which includes performance testing (only for library built in Release mode). 
In this case different implementations of each functions will be compared between themselves 
(for example a scalar implementation and implementations with using of different SIMD instructions such as SSE2, AVX2, and other).
Also it can be -m=c (creation of test data for cross-platform testing), -m=v (cross-platform testing with using of early prepared test data)
and -m=s (running of special tests).
* `-t=1` - a number of used threads (every thread runs all tests) for simulation of multi-thread loading.
* `-f=Sobel` - a filter. In current case will be tested only functions which contain word 'Sobel' in their names. 
If you miss this parameter then full testing will be performed.
You can use several filters - function name has to satisfy at least one of them.
* `-o=log.txt` - a file name with test report. The test's report also will be output to console.
    
Also you can use parameters:

* `-h` or `-?` in order to print help message.
* `-r=../..` to set project root directory.
* `-pa=1` to print alignment statistics.
	
