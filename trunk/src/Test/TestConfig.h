/*
* Simd Library Tests.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#ifndef __TestConfig_h__
#define __TestConfig_h__

#ifndef _DEBUG
#define TEST_PERFORMANCE_TEST_ENABLE
#endif

#include <stdlib.h>

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <sstream>
#include <limits>
#include <iomanip>

#define SIMD_STATIC
#include "Simd/SimdEnable.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdAlg.h"
#include "Simd/SimdUtils.h"

namespace Test
{
	typedef Simd::uchar uchar;
    typedef Simd::ushort ushort;
	typedef Simd::uint uint;
	typedef Simd::View View;
	typedef uint Histogram[Simd::HISTOGRAM_SIZE];
    typedef std::vector<uint> Sums;

#ifdef _DEBUG
	const int W = 128;
	const int H = 96;
#else
	const int W = 1920;
	const int H = 1080;
#endif

	const double MINIMAL_TEST_EXECUTION_TIME = 0.1;
}

#define TEST_ALIGN(size) \
	(((size_t)(size))%Simd::DEFAULT_MEMORY_ALIGN == 0 ? Simd::DEFAULT_MEMORY_ALIGN : 1)

#define TEST_CHECK_VALUE(name) \
    if(name##1 != name##2) \
    { \
        std::cout << "Error " << #name << ": (" << name##1  << " != " << name##2 << ")! " << std::endl; \
        return false; \
    }   

#endif//__TestConfig_h__
