/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#ifndef __SimdWinograd_h__
#define __SimdWinograd_h__

#include "Simd/SimdMath.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void Winograd2x3SetFilter1(const float * src, float * dst, size_t stride)
        {
            const float r2 = 1.0f / 2.0f;
            const float r4 = 1.0f / 4.0f;
            dst[0 * stride] = src[0];
            dst[1 * stride] = (src[0] + src[2] + src[1])*r2;
            dst[2 * stride] = (src[0] + src[2] - src[1])*r2;
            dst[3 * stride] = src[2];
            dst[4 * stride] = (src[0] + src[6] + src[3])*r2;
            dst[5 * stride] = ((src[0] + src[6] + src[3]) + (src[2] + src[8] + src[5]) + (src[1] + src[7] + src[4]))*r4;
            dst[6 * stride] = ((src[0] + src[6] + src[3]) + (src[2] + src[8] + src[5]) - (src[1] + src[7] + src[4]))*r4;
            dst[7 * stride] = (src[2] + src[8] + src[5])*r2;
            dst[8 * stride] = (src[0] + src[6] - src[3])*r2;
            dst[9 * stride] = ((src[0] + src[6] - src[3]) + (src[2] + src[8] - src[5]) + (src[1] + src[7] - src[4]))*r4;
            dst[10 * stride] = ((src[0] + src[6] - src[3]) + (src[2] + src[8] - src[5]) - (src[1] + src[7] - src[4]))*r4;
            dst[11 * stride] = (src[2] + src[8] - src[5])*r2;
            dst[12 * stride] = src[6];
            dst[13 * stride] = (src[6] + src[8] + src[7])*r2;
            dst[14 * stride] = (src[6] + src[8] - src[7])*r2;
            dst[15 * stride] = src[8];
        }
    };
}

#endif//__SimdWinograd_h__
