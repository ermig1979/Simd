/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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

#if defined(NDEBUG)
#define TEST_PERFORMANCE_TEST_ENABLE
#endif

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
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
#include "Simd/SimdConfig.h"
#include "Simd/SimdLib.hpp"

#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdEnable.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdAvx512vnni.h"
#include "Simd/SimdAvx512bf16.h"
#include "Simd/SimdAmx.h"
#include "Simd/SimdVmx.h"
#include "Simd/SimdVsx.h"
#include "Simd/SimdNeon.h"

namespace Test
{
    template <class T> class Tensor;

    typedef std::string String;
    typedef std::vector<String> Strings;
    typedef Simd::View<Simd::Allocator> View;
    typedef std::vector<View> Views;
    typedef Simd::Point<ptrdiff_t> Point;
    typedef std::vector<Point> Points;
    typedef Point Size;
    typedef Simd::Rectangle<ptrdiff_t> Rect;
    typedef uint32_t Histogram[Simd::HISTOGRAM_SIZE];
    typedef std::vector<int> Ints;
    typedef std::vector<uint32_t> Sums;
    typedef std::vector<uint64_t> Sums64;
    typedef std::vector<float, Simd::Allocator<float> > Buffer32f;
    typedef std::vector<uint8_t> Buffer8u;
    typedef std::vector<float*> FloatPtrs;
    typedef Tensor<float> Tensor32f;
    typedef Tensor<uint8_t> Tensor8u;
    typedef Tensor<int8_t> Tensor8i;
    typedef Tensor<int32_t> Tensor32i;

    const int E = 10;
    const int O = 9;

    extern double MINIMAL_TEST_EXECUTION_TIME;

    const int DW = 48;
    const int DH = 64;

    const float EPS = 0.001f;

    extern int C;
    extern int H;    
    extern int W;

    extern String ROOT_PATH;
    extern String SOURCE;
    extern String OUTPUT;
    extern String REAL_IMAGE;

    extern int LITTER_CPU_CACHE;

    enum DifferenceType
    {
        DifferenceAbsolute,
        DifferenceRelative,
        DifferenceBoth,
        DifferenceAny,
    };
}

#endif//__TestConfig_h__
