/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#ifndef __SimdTime_h__
#define __SimdTime_h__

#include "Simd/SimdDefs.h"

#if defined(_MSC_VER)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__GNUC__)
#include <sys/time.h>
#else
#error Platform is not supported!
#endif

namespace Simd
{
#if defined(_MSC_VER)
    SIMD_INLINE double Time()
    {
        LARGE_INTEGER counter, frequency;
        QueryPerformanceCounter(&counter);
        QueryPerformanceFrequency(&frequency);
        return double(counter.QuadPart) / double(frequency.QuadPart);
    }

    SIMD_INLINE int64_t TimeCounter()
    {
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        return counter.QuadPart;
    }

    SIMD_INLINE int64_t TimeFrequency()
    {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        return frequency.QuadPart;
    }
#elif defined(__GNUC__)
    SIMD_INLINE double Time()
    {
        timeval t;
        gettimeofday(&t, NULL);
        return t.tv_sec + t.tv_usec * 0.000001;
    }

    SIMD_INLINE int64_t TimeCounter()
    {
        timeval t;
        gettimeofday(&t, NULL);
        return int64_t(t.tv_sec)*1000000 + t.tv_usec;
    }

    SIMD_INLINE int64_t TimeFrequency()
    {
        return int64_t(1000000);
    }
#else
#error Platform is not supported!
#endif
}

#endif//__SimdTime_h__
