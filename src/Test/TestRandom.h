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
#ifndef __TestRandom_h__
#define __TestRandom_h__

#include "Test/TestLog.h"

#define TEST_RAND_VERSION 0

namespace Test
{
    SIMD_INLINE int Rand()
    {
        return ::rand();
    }

    SIMD_INLINE void Srand(unsigned int seed)
    {
        ::srand(seed);
    }

    SIMD_INLINE int RandMax()
    {
        return INT16_MAX;
    }

    void FillSequence(View& view);

    void FillPicture(View& view, uint64_t flag = 0x000000000000000F);

    void CreateTestImage(View& canvas, int rects, int labels);

    SIMD_INLINE int Random(int range)
    {
        return ((::rand() & INT16_MAX) * range) / INT16_MAX;
    }

    SIMD_INLINE double Random()
    {
        return ((::rand() & INT16_MAX) * 1.0) / INT16_MAX;
    }

    template<class T> inline void Fill(T* data, size_t size, T value)
    {
        for (size_t i = 0; i < size; ++i)
            data[i] = value;
    }

    void FillRandom(View& view, uint8_t lo = 0, uint8_t hi = 255);

    void FillRandom2(View& view, uint8_t lo = 0, uint8_t hi = 255, uint8_t step = 1);

    void FillRandomMask(View& view, uint8_t index);

    void FillRhombMask(View& mask, const Rect& rect, uint8_t index);

    void FillRandom16u(View& view, uint16_t lo = 0, uint16_t hi = UINT16_MAX);

    void FillRandom32f(View& view, float lo = 0, float hi = 4096.0f);

    void FillRandom(Buffer32f& buffer, float lo = 0, float hi = 4096.0f);

    void FillRandom(float* data, size_t size, float lo = 0, float hi = 4096.0f);

    void FillRandom(Tensor32f& tensor, float lo = -10.0f, float hi = 10.0f);

    void FillRandom(uint8_t* data, size_t size, uint8_t lo = 0, uint8_t hi = 255);

    void FillRandom(Tensor8u& tensor, uint8_t lo = 0, uint8_t hi = 255);

    void FillRandom(Tensor8i& tensor, int8_t lo = -128, int8_t hi = 127);

    void FillRandom(Tensor32i& tensor, int32_t lo, int32_t hi);

    void FillRandom(Tensor32f& tensor, float* min, float* max, size_t channels, int negative, float upper = 1.0f, float range = 0.01f);
}
#endif//__TestRandom_h__
