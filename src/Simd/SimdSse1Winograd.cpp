/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdWinograd.h"

namespace Simd
{
#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        SIMD_INLINE void Load4(const float * src, size_t step, __m128 * dst)
        {
            __m128 a0 = _mm_loadu_ps(src + 0 * step);
            __m128 a1 = _mm_loadu_ps(src + 1 * step);
            __m128 a2 = _mm_loadu_ps(src + 2 * step);
            __m128 a3 = _mm_loadu_ps(src + 3 * step);
            __m128 b0 = _mm_unpacklo_ps(a0, a2);
            __m128 b1 = _mm_unpackhi_ps(a0, a2);
            __m128 b2 = _mm_unpacklo_ps(a1, a3);
            __m128 b3 = _mm_unpackhi_ps(a1, a3);
            dst[0] = _mm_unpacklo_ps(b0, b2);
            dst[1] = _mm_unpackhi_ps(b0, b2);
            dst[2] = _mm_unpacklo_ps(b1, b3);
            dst[3] = _mm_unpackhi_ps(b1, b3);
        }

        SIMD_INLINE void Winograd2x3SetFilter4(const float * src, float * dst, size_t stride)
        {
            const __m128 r2 = _mm_set1_ps(1.0f / 2.0f);
            const __m128 r4 = _mm_set1_ps(1.0f / 4.0f);

            __m128 s[9];
            Load4(src + 0, 9, s + 0);
            Load4(src + 4, 9, s + 4);
            s[8] = _mm_setr_ps(src[8], src[17], src[26], src[35]);

            _mm_storeu_ps(dst + 0 * stride, s[0]);
            __m128 _0a2 = _mm_add_ps(s[0], s[2]);
            _mm_storeu_ps(dst + 1 * stride, _mm_mul_ps(_mm_add_ps(_0a2, s[1]), r2));
            _mm_storeu_ps(dst + 2 * stride, _mm_mul_ps(_mm_sub_ps(_0a2, s[1]), r2));
            _mm_storeu_ps(dst + 3 * stride, s[2]);

            __m128 _0a6a3 = _mm_add_ps(_mm_add_ps(s[0], s[6]), s[3]);
            _mm_storeu_ps(dst + 4 * stride, _mm_mul_ps(_0a6a3, r2));
            __m128 _2a8a5 = _mm_add_ps(_mm_add_ps(s[2], s[8]), s[5]);
            __m128 _1a7a4 = _mm_add_ps(_mm_add_ps(s[1], s[7]), s[4]);
            _mm_storeu_ps(dst + 5 * stride, _mm_mul_ps(_mm_add_ps(_mm_add_ps(_0a6a3, _2a8a5), _1a7a4), r4));
            _mm_storeu_ps(dst + 6 * stride, _mm_mul_ps(_mm_sub_ps(_mm_add_ps(_0a6a3, _2a8a5), _1a7a4), r4));
            _mm_storeu_ps(dst + 7 * stride, _mm_mul_ps(_2a8a5, r2));

            __m128 _0a6s3 = _mm_sub_ps(_mm_add_ps(s[0], s[6]), s[3]);
            _mm_storeu_ps(dst + 8 * stride, _mm_mul_ps(_0a6s3, r2));
            __m128 _2a8s5 = _mm_sub_ps(_mm_add_ps(s[2], s[8]), s[5]);
            __m128 _1a7s4 = _mm_sub_ps(_mm_add_ps(s[1], s[7]), s[4]);
            _mm_storeu_ps(dst + 9 * stride, _mm_mul_ps(_mm_add_ps(_mm_add_ps(_0a6s3, _2a8s5), _1a7s4), r4));
            _mm_storeu_ps(dst + 10 * stride, _mm_mul_ps(_mm_sub_ps(_mm_add_ps(_0a6s3, _2a8s5), _1a7s4), r4));
            _mm_storeu_ps(dst + 11 * stride, _mm_mul_ps(_2a8s5, r2));

            _mm_storeu_ps(dst + 12 * stride, s[6]);
            __m128 _6a8 = _mm_add_ps(s[6], s[8]);
            _mm_storeu_ps(dst + 13 * stride, _mm_mul_ps(_mm_add_ps(_6a8, s[7]), r2));
            _mm_storeu_ps(dst + 14 * stride, _mm_mul_ps(_mm_sub_ps(_6a8, s[7]), r2));
            _mm_storeu_ps(dst + 15 * stride, s[8]);
        }

        void Winograd2x3SetFilter(const float * src, size_t srcChannels, size_t dstChannels, float * dst, size_t dstStride)
        {
            size_t size = dstChannels * srcChannels;
            size_t size4 = AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4, src += 36, dst += 4)
                Winograd2x3SetFilter4(src, dst, dstStride);
            for (; i < size; i += 1, src += 9, dst += 1)
                Base::Winograd2x3SetFilter1(src, dst, dstStride);
        }

        SIMD_INLINE void Winograd2x3SetInput4(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m128 t[16];
            Load4(src + 0 * srcStride, 2, t + 0);
            Load4(src + 1 * srcStride, 2, t + 4);
            Load4(src + 2 * srcStride, 2, t + 8);
            Load4(src + 3 * srcStride, 2, t + 12);
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

        SIMD_INLINE void Winograd2x3SetInput4p(const float * src, size_t srcStride, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            float tmp[4 * 16] = { 0 };
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    tmp[row * 16 + col] = src[row * srcStride + col];
            Winograd2x3SetInput4(tmp, 16, dst, dstStride);
        }

        void Winograd2x3SetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, size_t dstStride, int pad)
        {
            size_t dstHeight = pad ? srcHeight : srcHeight - 2;
            size_t dstWidth = pad ? srcWidth : srcWidth - 2;
            size_t dstHeightFull = AlignLo(dstHeight, 2);
            size_t dstWidthFull = AlignLo(dstWidth, 2);
            size_t noseW = Simd::Min<size_t>(4, dstWidth + 1);
            size_t noseH = Simd::Min<size_t>(4, dstHeight + 1);
            size_t start = pad ? 2 : 0;
            if (pad)
            {
                if (dstHeight == dstHeightFull)
                    dstHeightFull -= 2;
                if (dstWidth == dstWidthFull)
                    dstWidthFull -= 2;
                src -= srcWidth + 1;
            }
            size_t tailW = dstWidth - dstWidthFull + (pad ? 1 : 2);
            size_t tailH = dstHeight - dstHeightFull + (pad ? 1 : 2);
            size_t dstWidthFull4 = dstWidthFull >= start ? AlignLo(dstWidthFull - start, 8) + start : start;
            for (size_t c = 0; c < srcChannels; ++c)
            {
                size_t row = 0, col = 0;
                if (pad)
                {
                    if (pad)
                        Base::Winograd2x3SetInput1p(src, srcWidth, 1, noseH, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull; col += 2)
                        Base::Winograd2x3SetInput1p(src + col, srcWidth, 1, noseH, 0, 4, dst++, dstStride);
                    if (col < dstWidth)
                        Base::Winograd2x3SetInput1p(src + col, srcWidth, 1, noseH, 0, tailW, dst++, dstStride);
                }
                for (row = start; row < dstHeightFull; row += 2)
                {
                    if (pad)
                        Base::Winograd2x3SetInput1p(src + row * srcWidth, srcWidth, 0, 4, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull4; col += 8, dst += 4)
                        Winograd2x3SetInput4(src + row * srcWidth + col, srcWidth, dst, dstStride);
                    for (; col < dstWidthFull; col += 2)
                        Base::Winograd2x3SetInput1(src + row * srcWidth + col, srcWidth, dst++, dstStride);
                    if (col < dstWidth)
                        Base::Winograd2x3SetInput1p(src + row * srcWidth + col, srcWidth, 0, 4, 0, tailW, dst++, dstStride);
                }
                if (row < dstHeight)
                {
                    if (pad)
                        Base::Winograd2x3SetInput1p(src + row * srcWidth, srcWidth, 0, tailH, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull; col += 2)
                        Base::Winograd2x3SetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, 4, dst++, dstStride);
                    if (col < dstWidth)
                        Base::Winograd2x3SetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, tailW, dst++, dstStride);
                }
                src += srcWidth * srcHeight;
            }
        }

        SIMD_INLINE void Winograd4x3SetFilter4Row(const __m128 * t, float * dst, size_t stride)
        {
            const __m128 r4 = _mm_set1_ps(1.0f / 4.0f);
            const __m128 r6 = _mm_set1_ps(1.0f / 6.0f);
            const __m128 mr6 = _mm_set1_ps(-1.0f / 6.0f);
            const __m128 r12 = _mm_set1_ps(1.0f / 12.0f);
            const __m128 r24 = _mm_set1_ps(1.0f / 24.0f);
            _mm_storeu_ps(dst + 0 * stride, _mm_mul_ps(r4, t[0]));
            __m128 t0 = _mm_add_ps(t[0], t[2]);
            _mm_storeu_ps(dst + 1 * stride, _mm_mul_ps(mr6, _mm_add_ps(t0, t[1])));
            _mm_storeu_ps(dst + 2 * stride, _mm_mul_ps(mr6, _mm_sub_ps(t0, t[1])));
            __m128 t1 = _mm_add_ps(_mm_mul_ps(r24, t[0]), _mm_mul_ps(r6, t[2]));
            __m128 t2 = _mm_mul_ps(r12, t[1]);
            _mm_storeu_ps(dst + 3 * stride, _mm_add_ps(t1, t2));
            _mm_storeu_ps(dst + 4 * stride, _mm_sub_ps(t1, t2));
            _mm_storeu_ps(dst + 5 * stride, t[2]);
        }

        SIMD_INLINE void Winograd4x3SetFilter4(const float * src, float * dst, size_t stride)
        {
            const __m128 r4 = _mm_set1_ps(1.0f / 4.0f);
            const __m128 r6 = _mm_set1_ps(1.0f / 6.0f);
            const __m128 mr6 = _mm_set1_ps(-1.0f / 6.0f);
            const __m128 r12 = _mm_set1_ps(1.0f / 12.0f);
            const __m128 r24 = _mm_set1_ps(1.0f / 24.0f);

            __m128 s[9];
            Load4(src + 0, 9, s + 0);
            Load4(src + 4, 9, s + 4);
            s[8] = _mm_setr_ps(src[8], src[17], src[26], src[35]);

            __m128 t[3];
            t[0] = _mm_mul_ps(r4, s[0]);
            t[1] = _mm_mul_ps(r4, s[1]);
            t[2] = _mm_mul_ps(r4, s[2]);
            Winograd4x3SetFilter4Row(t, dst + 0*stride, stride);

            t[0] = _mm_mul_ps(mr6, _mm_add_ps(_mm_add_ps(s[0], s[3]), s[6]));
            t[1] = _mm_mul_ps(mr6, _mm_add_ps(_mm_add_ps(s[1], s[4]), s[7]));
            t[2] = _mm_mul_ps(mr6, _mm_add_ps(_mm_add_ps(s[2], s[5]), s[8]));
            Winograd4x3SetFilter4Row(t, dst + 6*stride, stride);

            t[0] = _mm_mul_ps(mr6, _mm_add_ps(_mm_sub_ps(s[0], s[3]), s[6]));
            t[1] = _mm_mul_ps(mr6, _mm_add_ps(_mm_sub_ps(s[1], s[4]), s[7]));
            t[2] = _mm_mul_ps(mr6, _mm_add_ps(_mm_sub_ps(s[2], s[5]), s[8]));
            Winograd4x3SetFilter4Row(t, dst + 12 * stride, stride);

            t[0] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(r24, s[0]), _mm_mul_ps(r12, s[3])), _mm_mul_ps(r6, s[6]));
            t[1] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(r24, s[1]), _mm_mul_ps(r12, s[4])), _mm_mul_ps(r6, s[7]));
            t[2] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(r24, s[2]), _mm_mul_ps(r12, s[5])), _mm_mul_ps(r6, s[8]));
            Winograd4x3SetFilter4Row(t, dst + 18 * stride, stride);

            t[0] = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(r24, s[0]), _mm_mul_ps(r12, s[3])), _mm_mul_ps(r6, s[6]));
            t[1] = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(r24, s[1]), _mm_mul_ps(r12, s[4])), _mm_mul_ps(r6, s[7]));
            t[2] = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(r24, s[2]), _mm_mul_ps(r12, s[5])), _mm_mul_ps(r6, s[8]));
            Winograd4x3SetFilter4Row(t, dst + 24 * stride, stride);

            Winograd4x3SetFilter4Row(s + 6, dst + 30 * stride, stride);
        }

        void Winograd4x3SetFilter(const float * src, size_t srcChannels, size_t dstChannels, float * dst, size_t dstStride)
        {
            size_t size = dstChannels * srcChannels;
            size_t size4 = AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4, src += 36, dst += 4)
                Winograd4x3SetFilter4(src, dst, dstStride);
            for (; i < size; i += 1, src += 9, dst += 1)
                Base::Winograd4x3SetFilter1(src, dst, dstStride);
        }
    }
#endif// SIMD_SSE_ENABLE
}
