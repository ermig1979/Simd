/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#ifndef __TestUtils_h__
#define __TestUtils_h__

#include "Test/TestLog.h"

namespace Test
{
    void FillSequence(View & view);
    void FillPicture(View & view, uint64_t flag = 0x000000000000000F);

    SIMD_INLINE int Random(int range)
    {
        return ((::rand()&INT16_MAX)*range) / INT16_MAX;
    }

    SIMD_INLINE double Random()
    {
        return ((::rand()&INT16_MAX)*1.0) / INT16_MAX;
    }

    template<class T> inline void Fill(T * data, size_t size, T value)
    {
        for (size_t i = 0; i < size; ++i)
            data[i] = value;
    }

    void FillRandom(View & view, uint8_t lo = 0, uint8_t hi = 255);

    void FillRandom2(View & view, uint8_t lo = 0, uint8_t hi = 255, uint8_t step = 1);

    void FillRandomMask(View & view, uint8_t index);

    void FillRhombMask(View & mask, const Rect & rect, uint8_t index);

    void FillRandom16u(View& view, uint16_t lo = 0, uint16_t hi = UINT16_MAX);

    void FillRandom32f(View & view, float lo = 0, float hi = 4096.0f);

    void FillRandom(Buffer32f & buffer, float lo = 0, float hi = 4096.0f);

    void FillRandom(float * data, size_t size, float lo = 0, float hi = 4096.0f);

    void FillRandom(Tensor32f & tensor, float lo = -10.0f, float hi = 10.0f);

    void FillRandom(uint8_t * data, size_t size, uint8_t lo = 0, uint8_t hi = 255);

    void FillRandom(Tensor8u& tensor, uint8_t lo = 0, uint8_t hi = 255);

    void FillRandom(Tensor8i& tensor, int8_t lo = -128, int8_t hi = 127);

    void FillRandom(Tensor32i& tensor, int32_t lo, int32_t hi);

    void FillRandom(Tensor32f& tensor, float* min, float* max, size_t channels, int negative, float upper = 1.0f, float range = 0.01f);

    void SetSrc32fTo8u(const Tensor32f& src, const float* min, const float* max, size_t channels, int negative,
        SimdSynetCompatibilityType compatibility, float* shift, float* scale, Tensor8u& dst);

    void SetDstStat(size_t channels, int negative, SimdSynetCompatibilityType compatibility, 
        const Tensor32f& dst, float* min, float* max, float* scale, float* shift);

    bool Compare(const View & a, const View & b,
        int differenceMax = 0, bool printError = false, int errorCountMax = 0, int valueCycle = 0,
        const String & description = "");

    bool Compare(const Histogram a, const Histogram b,
        int differenceMax = 0, bool printError = false, int errorCountMax = 0, const String & description = "");

    bool Compare(const Sums & a, const Sums & b,
        int differenceMax = 0, bool printError = false, int errorCountMax = 0, const String & description = "");

    bool Compare(const Sums64 & a, const Sums64 & b,
        int differenceMax = 0, bool printError = false, int errorCountMax = 0, const String & description = "");

    bool Compare(const Rect & a, const Rect & b, bool printError = false);

    bool Compare(const Buffer32f & a, const Buffer32f & b, float differenceMax,
        bool printError, int errorCountMax, DifferenceType differenceType, const String & description = "");

    bool Compare(const Buffer32f & a, const Buffer32f & b, float differenceMax = EPS,
        bool printError = false, int errorCountMax = 0, bool relative = true, const String & description = "");

    bool CompareCycle(const Buffer32f & a, const Buffer32f & b, size_t cycle, float differenceMax = EPS,
        bool printError = false, int errorCountMax = 0, const String & description = "");

    bool Compare(const View & a, const View & b, float differenceMax, bool printError,
        int errorCountMax, DifferenceType differenceType, const String & description = "");

    bool Compare(const View & a, const View & b, float differenceMax = EPS, bool printError = false,
        int errorCountMax = 0, bool relative = true, const String & description = "");

    bool Compare(const float & a, const float & b, float differenceMax = EPS, bool printError = false, 
        DifferenceType differenceType = DifferenceRelative, const String & description = "");

    bool Compare(const uint8_t * data1, size_t size1, const uint8_t* data2, size_t size2, int differenceMax = 0,
        bool printError = false, int errorCountMax = 0, const String& description = "");
}

#define TEST_CHECK_VALUE(name) \
    if(name##1 != name##2) \
    { \
        TEST_LOG_SS(Error, "Error " << #name << ": (" << name##1  << " != " << name##2 << ")! "); \
        return false; \
    } 

#endif//__TestUtils_h__
