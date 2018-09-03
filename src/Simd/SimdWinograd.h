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

        SIMD_INLINE void Winograd2x3pSetInput1(const float * src, size_t srcStride, float * dst, size_t dstStride)
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

            dst[0 * dstStride] = (tmp[0] - tmp[8]) - (tmp[2] - tmp[10]);
            dst[1 * dstStride] = (tmp[1] - tmp[9]) + (tmp[2] - tmp[10]);
            dst[2 * dstStride] = (tmp[2] - tmp[10]) - (tmp[1] - tmp[9]);
            dst[3 * dstStride] = (tmp[1] - tmp[9]) - (tmp[3] - tmp[11]);
            dst[4 * dstStride] = (tmp[4] + tmp[8]) - (tmp[6] + tmp[10]);
            dst[5 * dstStride] = (tmp[5] + tmp[9]) + (tmp[6] + tmp[10]);
            dst[6 * dstStride] = (tmp[6] + tmp[10]) - (tmp[5] + tmp[9]);
            dst[7 * dstStride] = (tmp[5] + tmp[9]) - (tmp[7] + tmp[11]);
            dst[8 * dstStride] = (tmp[8] - tmp[4]) - (tmp[10] - tmp[6]);
            dst[9 * dstStride] = (tmp[9] - tmp[5]) + (tmp[10] - tmp[6]);
            dst[10 * dstStride] = (tmp[10] - tmp[6]) - (tmp[9] - tmp[5]);
            dst[11 * dstStride] = (tmp[9] - tmp[5]) - (tmp[11] - tmp[7]);
            dst[12 * dstStride] = (tmp[4] - tmp[12]) - (tmp[6] - tmp[14]);
            dst[13 * dstStride] = (tmp[5] - tmp[13]) + (tmp[6] - tmp[14]);
            dst[14 * dstStride] = (tmp[6] - tmp[14]) - (tmp[5] - tmp[13]);
            dst[15 * dstStride] = (tmp[5] - tmp[13]) - (tmp[7] - tmp[15]);
        }

        SIMD_INLINE void Winograd2x3pSetInput1p(const float * src, size_t srcStride, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            float tmp[4 * 4] = { 0 };
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    tmp[row * 4 + col] = src[row * srcStride + col];
            Winograd2x3pSetInput1(tmp, 4, dst, dstStride);
        }

        SIMD_INLINE void Winograd2x3pSetOutput1(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            float c1[16];
            c1[0] = src[0 * srcStride];
            c1[1] = src[1 * srcStride];
            c1[2] = src[2 * srcStride];
            c1[3] = src[3 * srcStride];
            c1[4] = src[4 * srcStride];
            c1[5] = src[5 * srcStride];
            c1[6] = src[6 * srcStride];
            c1[7] = src[7 * srcStride];
            c1[8] = src[8 * srcStride];
            c1[9] = src[9 * srcStride];
            c1[10] = src[10 * srcStride];
            c1[11] = src[11 * srcStride];
            c1[12] = src[12 * srcStride];
            c1[13] = src[13 * srcStride];
            c1[14] = src[14 * srcStride];
            c1[15] = src[15 * srcStride];

            float tmp[8];
            tmp[0] = c1[0] + c1[1] + c1[2];
            tmp[1] = c1[1] - c1[2] - c1[3];
            tmp[2] = c1[4] + c1[5] + c1[6];
            tmp[3] = c1[5] - c1[6] - c1[7];
            tmp[4] = c1[8] + c1[9] + c1[10];
            tmp[5] = c1[9] - c1[10] - c1[11];
            tmp[6] = c1[12] + c1[13] + c1[14];
            tmp[7] = c1[13] - c1[14] - c1[15];

            dst[0 * dstStride + 0] = tmp[0] + tmp[2] + tmp[4];
            dst[0 * dstStride + 1] = tmp[1] + tmp[3] + tmp[5];
            dst[1 * dstStride + 0] = tmp[2] - tmp[4] - tmp[6];
            dst[1 * dstStride + 1] = tmp[3] - tmp[5] - tmp[7];
        }

        SIMD_INLINE void Winograd2x3pSetOutput1p(const float * src, size_t srcStride, float * dst, size_t dstStride, size_t rowE, size_t colE)
        {
            float tmp[2 * 2];
            Winograd2x3pSetOutput1(src, srcStride, tmp, 2);
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    dst[row*dstStride + col] = tmp[row * 2 + col];
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

#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        SIMD_INLINE void Winograd2x3pSetInputLoad4(const float * src, __m128 * dst)
        {
            __m128 a0 = _mm_loadu_ps(src + 0);
            __m128 a1 = _mm_loadu_ps(src + 2);
            __m128 a2 = _mm_loadu_ps(src + 4);
            __m128 a3 = _mm_loadu_ps(src + 6);
            dst[0] = _mm_shuffle_ps(a0, a2, 0x88);
            dst[1] = _mm_shuffle_ps(a0, a2, 0xDD);
            dst[2] = _mm_shuffle_ps(a1, a3, 0x88);
            dst[3] = _mm_shuffle_ps(a1, a3, 0xDD);
        }

        SIMD_INLINE void Winograd2x3pSetInput4Store(const __m128 * t, float * dst, size_t dstStride)
        {
            _mm_storeu_ps(dst + 0 * dstStride, _mm_sub_ps(_mm_sub_ps(t[0], t[8]), _mm_sub_ps(t[2], t[10])));
            _mm_storeu_ps(dst + 1 * dstStride, _mm_add_ps(_mm_sub_ps(t[1], t[9]), _mm_sub_ps(t[2], t[10])));
            _mm_storeu_ps(dst + 2 * dstStride, _mm_sub_ps(_mm_sub_ps(t[2], t[10]), _mm_sub_ps(t[1], t[9])));
            _mm_storeu_ps(dst + 3 * dstStride, _mm_sub_ps(_mm_sub_ps(t[1], t[9]), _mm_sub_ps(t[3], t[11])));
            _mm_storeu_ps(dst + 4 * dstStride, _mm_sub_ps(_mm_add_ps(t[4], t[8]), _mm_add_ps(t[6], t[10])));
            _mm_storeu_ps(dst + 5 * dstStride, _mm_add_ps(_mm_add_ps(t[5], t[9]), _mm_add_ps(t[6], t[10])));
            _mm_storeu_ps(dst + 6 * dstStride, _mm_sub_ps(_mm_add_ps(t[6], t[10]), _mm_add_ps(t[5], t[9])));
            _mm_storeu_ps(dst + 7 * dstStride, _mm_sub_ps(_mm_add_ps(t[5], t[9]), _mm_add_ps(t[7], t[11])));
            _mm_storeu_ps(dst + 8 * dstStride, _mm_sub_ps(_mm_sub_ps(t[8], t[4]), _mm_sub_ps(t[10], t[6])));
            _mm_storeu_ps(dst + 9 * dstStride, _mm_add_ps(_mm_sub_ps(t[9], t[5]), _mm_sub_ps(t[10], t[6])));
            _mm_storeu_ps(dst + 10 * dstStride, _mm_sub_ps(_mm_sub_ps(t[10], t[6]), _mm_sub_ps(t[9], t[5])));
            _mm_storeu_ps(dst + 11 * dstStride, _mm_sub_ps(_mm_sub_ps(t[9], t[5]), _mm_sub_ps(t[11], t[7])));
            _mm_storeu_ps(dst + 12 * dstStride, _mm_sub_ps(_mm_sub_ps(t[4], t[12]), _mm_sub_ps(t[6], t[14])));
            _mm_storeu_ps(dst + 13 * dstStride, _mm_add_ps(_mm_sub_ps(t[5], t[13]), _mm_sub_ps(t[6], t[14])));
            _mm_storeu_ps(dst + 14 * dstStride, _mm_sub_ps(_mm_sub_ps(t[6], t[14]), _mm_sub_ps(t[5], t[13])));
            _mm_storeu_ps(dst + 15 * dstStride, _mm_sub_ps(_mm_sub_ps(t[5], t[13]), _mm_sub_ps(t[7], t[15])));
        }

        SIMD_INLINE void Winograd2x3pSetInput4(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m128 t[16];
            Winograd2x3pSetInputLoad4(src + 0 * srcStride, t + 0);
            Winograd2x3pSetInputLoad4(src + 1 * srcStride, t + 4);
            Winograd2x3pSetInputLoad4(src + 2 * srcStride, t + 8);
            Winograd2x3pSetInputLoad4(src + 3 * srcStride, t + 12);
            Winograd2x3pSetInput4Store(t, dst, dstStride);
        }

        SIMD_INLINE void Winograd2x3pSetInput4p(const float * src, size_t srcStride, size_t rowB, size_t rowE, float * dst, size_t dstStride)
        {
            __m128 t[16];
            __m128 * pt = t;
            for (size_t row = 0; row < 4; row++, pt += 4)
            {
                if (row >= rowB && row < rowE)
                    Winograd2x3pSetInputLoad4(src + row * srcStride, pt);
                else
                {
                    pt[0] = _mm_setzero_ps();
                    pt[1] = _mm_setzero_ps();
                    pt[2] = _mm_setzero_ps();
                    pt[3] = _mm_setzero_ps();
                }
            }
            Winograd2x3pSetInput4Store(t, dst, dstStride);
        }

        SIMD_INLINE void Load2t(const float * src, size_t srcStride, __m128 * dst)
        {
            __m128 s0 = _mm_loadu_ps(src + 0 * srcStride);
            __m128 s1 = _mm_loadu_ps(src + 1 * srcStride);
            __m128 s2 = _mm_loadu_ps(src + 2 * srcStride);
            __m128 s3 = _mm_loadu_ps(src + 3 * srcStride);
            dst[0] = _mm_add_ps(_mm_add_ps(s0, s1), s2);
            dst[1] = _mm_sub_ps(_mm_sub_ps(s1, s2), s3);
        }

        SIMD_INLINE void Winograd2x3pSetOutput4(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m128 t[8];
            Load2t(src + 0 * srcStride, srcStride, t + 0);
            Load2t(src + 4 * srcStride, srcStride, t + 2);
            Load2t(src + 8 * srcStride, srcStride, t + 4);
            Load2t(src + 12 * srcStride, srcStride, t + 6);

            __m128 d00 = _mm_add_ps(_mm_add_ps(t[0], t[2]), t[4]);
            __m128 d01 = _mm_add_ps(_mm_add_ps(t[1], t[3]), t[5]);
            __m128 d10 = _mm_sub_ps(_mm_sub_ps(t[2], t[4]), t[6]);
            __m128 d11 = _mm_sub_ps(_mm_sub_ps(t[3], t[5]), t[7]);

            _mm_storeu_ps(dst + 0 * dstStride + 0, _mm_unpacklo_ps(d00, d01));
            _mm_storeu_ps(dst + 0 * dstStride + 4, _mm_unpackhi_ps(d00, d01));
            _mm_storeu_ps(dst + 1 * dstStride + 0, _mm_unpacklo_ps(d10, d11));
            _mm_storeu_ps(dst + 1 * dstStride + 4, _mm_unpackhi_ps(d10, d11));
        }
    }
#endif //SIMD_SSE_ENABLE
}

#endif//__SimdWinograd_h__
