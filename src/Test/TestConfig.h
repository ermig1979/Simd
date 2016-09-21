/*
* Tests for Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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

#ifdef NDEBUG
#define TEST_PERFORMANCE_TEST_ENABLE
#endif

#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <algorithm>
#include <sstream>
#include <limits>
#include <iomanip>
#include <memory>
#include <exception>
#include <stdexcept>
#include <thread>
#include <mutex>

#define SIMD_STATIC
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdEnable.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse1.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdSse3.h"
#include "Simd/SimdSsse3.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdSse42.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdVmx.h"
#include "Simd/SimdVsx.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdLib.hpp"

namespace Test
{
    typedef std::string String;
    typedef std::vector<String> Strings;
	typedef Simd::View<Simd::Allocator> View;
    typedef Simd::Point<ptrdiff_t> Point;
	typedef Point Size;
    typedef Simd::Rectangle<ptrdiff_t> Rect;
	typedef uint32_t Histogram[Simd::HISTOGRAM_SIZE];
    typedef std::vector<uint32_t> Sums;
    typedef std::vector<uint64_t> Sums64;
    typedef std::vector<float> Buffer32f;

#ifdef TEST_PERFORMANCE_TEST_ENABLE
	const int W = 1920;
	const int H = 1080;
#else
    const int W = 128;
    const int H = 96;
#endif

    const int E = 10;
    const int O = 9;

    const double MINIMAL_TEST_EXECUTION_TIME = 0.1;

    const int DW = 48;
    const int DH = 64;

    const float EPS = 0.001f;

    extern String ROOT_PATH;
}

#endif//__TestConfig_h__
