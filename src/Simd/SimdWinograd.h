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
#include "Simd/SimdLoad.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void Winograd2x3iSetInput1(const float * src, size_t srcStride, float * dst)
        {
            float tmp[16];
            tmp[0] = src[0 * srcStride + 0];
            tmp[1] = src[0 * srcStride + 1];
            tmp[2] = src[0 * srcStride + 2];
            tmp[3] = src[0 * srcStride + 3];

            tmp[4] = src[1 * srcStride + 0];
            tmp[5] = src[1 * srcStride + 1];
            tmp[6] = src[1 * srcStride + 2];
            tmp[7] = src[1 * srcStride + 3];

            tmp[8] = src[2 * srcStride + 0];
            tmp[9] = src[2 * srcStride + 1];
            tmp[10] = src[2 * srcStride + 2];
            tmp[11] = src[2 * srcStride + 3];

            tmp[12] = src[3 * srcStride + 0];
            tmp[13] = src[3 * srcStride + 1];
            tmp[14] = src[3 * srcStride + 2];
            tmp[15] = src[3 * srcStride + 3];

            dst[0] = (tmp[0] - tmp[8]) - (tmp[2] - tmp[10]);
            dst[1] = (tmp[1] - tmp[9]) + (tmp[2] - tmp[10]);
            dst[2] = (tmp[2] - tmp[10]) - (tmp[1] - tmp[9]);
            dst[3] = (tmp[1] - tmp[9]) - (tmp[3] - tmp[11]);
            dst[4] = (tmp[4] + tmp[8]) - (tmp[6] + tmp[10]);
            dst[5] = (tmp[5] + tmp[9]) + (tmp[6] + tmp[10]);
            dst[6] = (tmp[6] + tmp[10]) - (tmp[5] + tmp[9]);
            dst[7] = (tmp[5] + tmp[9]) - (tmp[7] + tmp[11]);
            dst[8] = (tmp[8] - tmp[4]) - (tmp[10] - tmp[6]);
            dst[9] = (tmp[9] - tmp[5]) + (tmp[10] - tmp[6]);
            dst[10] = (tmp[10] - tmp[6]) - (tmp[9] - tmp[5]);
            dst[11] = (tmp[9] - tmp[5]) - (tmp[11] - tmp[7]);
            dst[12] = (tmp[4] - tmp[12]) - (tmp[6] - tmp[14]);
            dst[13] = (tmp[5] - tmp[13]) + (tmp[6] - tmp[14]);
            dst[14] = (tmp[6] - tmp[14]) - (tmp[5] - tmp[13]);
            dst[15] = (tmp[5] - tmp[13]) - (tmp[7] - tmp[15]);
        }

        SIMD_INLINE void Winograd2x3iSetInput1p(const float * src, size_t srcStride, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst)
        {
            float tmp[4 * 4] = { 0 };
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    tmp[row * 4 + col] = src[row * srcStride + col];
            Winograd2x3iSetInput1(tmp, 4, dst);
        }

        SIMD_INLINE void Winograd2x3pSetFilter1(const float * src, float * dst, size_t stride)
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

        SIMD_INLINE void Winograd4x3pSetFilter1(const float * src, float * dst, size_t stride)
        {
            const float r4 = float(1.0f / 4.0f);
            const float r6 = float(1.0f / 6.0f);
            const float r12 = float(1.0f / 12.0f);
            const float r24 = float(1.0f / 24.0f);
            float t[18];
            t[0] = r4 * src[0];
            t[1] = r4 * src[1];
            t[2] = r4 * src[2];
            t[3] = -r6 * (src[0] + src[3] + src[6]);
            t[4] = -r6 * (src[1] + src[4] + src[7]);
            t[5] = -r6 * (src[2] + src[5] + src[8]);
            t[6] = -r6 * (src[0] - src[3] + src[6]);
            t[7] = -r6 * (src[1] - src[4] + src[7]);
            t[8] = -r6 * (src[2] - src[5] + src[8]);
            t[9] = r24 * src[0] + r12 * src[3] + r6 * src[6];
            t[10] = r24 * src[1] + r12 * src[4] + r6 * src[7];
            t[11] = r24 * src[2] + r12 * src[5] + r6 * src[8];
            t[12] = r24 * src[0] - r12 * src[3] + r6 * src[6];
            t[13] = r24 * src[1] - r12 * src[4] + r6 * src[7];
            t[14] = r24 * src[2] - r12 * src[5] + r6 * src[8];
            t[15] = src[6];
            t[16] = src[7];
            t[17] = src[8];

            dst[stride*0] = r4 * t[0];
            dst[stride*1] = -r6 * (t[0] + t[1] + t[2]);
            dst[stride*2] = -r6 * (t[0] - t[1] + t[2]);
            dst[stride*3] = r24 * t[0] + r12 * t[1] + r6 * t[2];
            dst[stride*4] = r24 * t[0] - r12 * t[1] + r6 * t[2];
            dst[stride*5] = t[2];

            dst[stride*6] = r4 * t[3];
            dst[stride*7] = -r6 * (t[3] + t[4] + t[5]);
            dst[stride*8] = -r6 * (t[3] - t[4] + t[5]);
            dst[stride*9] = r24 * t[3] + r12 * t[4] + r6 * t[5];
            dst[stride*10] = r24 * t[3] - r12 * t[4] + r6 * t[5];
            dst[stride*11] = t[5];

            dst[stride*12] = r4 * t[6];
            dst[stride*13] = -r6 * (t[6] + t[7] + t[8]);
            dst[stride*14] = -r6 * (t[6] - t[7] + t[8]);
            dst[stride*15] = r24 * t[6] + r12 * t[7] + r6 * t[8];
            dst[stride*16] = r24 * t[6] - r12 * t[7] + r6 * t[8];
            dst[stride*17] = t[8];

            dst[stride*18] = r4 * t[9];
            dst[stride*19] = -r6 * (t[9] + t[10] + t[11]);
            dst[stride*20] = -r6 * (t[9] - t[10] + t[11]);
            dst[stride*21] = r24 * t[9] + r12 * t[10] + r6 * t[11];
            dst[stride*22] = r24 * t[9] - r12 * t[10] + r6 * t[11];
            dst[stride*23] = t[11];

            dst[stride*24] = r4 * t[12];
            dst[stride*25] = -r6 * (t[12] + t[13] + t[14]);
            dst[stride*26] = -r6 * (t[12] - t[13] + t[14]);
            dst[stride*27] = r24 * t[12] + r12 * t[13] + r6 * t[14];
            dst[stride*28] = r24 * t[12] - r12 * t[13] + r6 * t[14];
            dst[stride*29] = t[14];

            dst[stride*30] = r4 * t[15];
            dst[stride*31] = -r6 * (t[15] + t[16] + t[17]);
            dst[stride*32] = -r6 * (t[15] - t[16] + t[17]);
            dst[stride*33] = r24 * t[15] + r12 * t[16] + r6 * t[17];
            dst[stride*34] = r24 * t[15] - r12 * t[16] + r6 * t[17];
            dst[stride*35] = t[17];
        }
    }
}

#endif//__SimdWinograd_h__
