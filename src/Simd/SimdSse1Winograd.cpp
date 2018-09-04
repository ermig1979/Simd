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
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        SIMD_INLINE void Winograd2x3iSetInput1(const __m128 * src, float * dst)
        {
            static const __m128 _mpmm = _mm_setr_ps(-1.0f, 1.0f, -1.0f, -1.0f);
            __m128 t0 = _mm_sub_ps(src[0], src[2]);
            __m128 t1 = _mm_add_ps(src[1], src[2]);
            __m128 t2 = _mm_sub_ps(src[2], src[1]);
            __m128 t3 = _mm_sub_ps(src[1], src[3]);
            _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_shuffle_ps(t0, t0, 0x64), _mm_mul_ps(_mpmm, _mm_shuffle_ps(t0, t0, 0xDA))));
            _mm_storeu_ps(dst + 4, _mm_add_ps(_mm_shuffle_ps(t1, t1, 0x64), _mm_mul_ps(_mpmm, _mm_shuffle_ps(t1, t1, 0xDA))));
            _mm_storeu_ps(dst + 8, _mm_add_ps(_mm_shuffle_ps(t2, t2, 0x64), _mm_mul_ps(_mpmm, _mm_shuffle_ps(t2, t2, 0xDA))));
            _mm_storeu_ps(dst + 12, _mm_add_ps(_mm_shuffle_ps(t3, t3, 0x64), _mm_mul_ps(_mpmm, _mm_shuffle_ps(t3, t3, 0xDA))));
        }

        SIMD_INLINE void Winograd2x3iSetInput1(const float * src, size_t srcStride, float * dst)
        {
            __m128 s[4];
            s[0] = _mm_loadu_ps(src + 0 * srcStride);
            s[1] = _mm_loadu_ps(src + 1 * srcStride);
            s[2] = _mm_loadu_ps(src + 2 * srcStride);
            s[3] = _mm_loadu_ps(src + 3 * srcStride);
            Winograd2x3iSetInput1(s, dst);
        }

        SIMD_INLINE void Winograd2x3iSetInput1p(const float * src, size_t srcStride, size_t rowB, size_t rowE, float * dst)
        {
            __m128 s[4] = {_mm_setzero_ps(), _mm_setzero_ps() , _mm_setzero_ps() , _mm_setzero_ps() };
            for (size_t row = rowB; row < rowE; ++row)
                s[row] = _mm_loadu_ps(src + row * srcStride);
            Winograd2x3iSetInput1(s, dst);
        }

        SIMD_INLINE void Winograd2x3iSetInput1p(const float * src, size_t srcStride, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst)
        {
            float tmp[4 * 4] = { 0 };
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    tmp[row * 4 + col] = src[row * srcStride + col];
            Winograd2x3iSetInput1(tmp, 4, dst);
        }

        void Winograd2x3iSetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, int pad)
        {
            size_t dstHeight = pad ? srcHeight : srcHeight - 2;
            size_t dstWidth = pad ? srcWidth : srcWidth - 2;
            size_t dstStride = ((dstHeight + 1) / 2) * ((dstWidth + 1) / 2)*srcChannels;
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
            for (size_t c = 0; c < srcChannels; ++c)
            {
                size_t row = 0, col = 0;
                if (pad)
                {
                    if (pad)
                        Winograd2x3iSetInput1p(src, srcWidth, 1, noseH, 1, noseW, dst), dst += 16;
                    for (col = start; col < dstWidthFull; col += 2)
                        Winograd2x3iSetInput1p(src + col, srcWidth, 1, noseH, dst), dst += 16;
                    if (col < dstWidth)
                        Winograd2x3iSetInput1p(src + col, srcWidth, 1, noseH, 0, tailW, dst), dst += 16;
                }
                for (row = start; row < dstHeightFull; row += 2)
                {
                    if (pad)
                        Winograd2x3iSetInput1p(src + row * srcWidth, srcWidth, 0, 4, 1, noseW, dst), dst += 16;
                    for (col = start; col < dstWidthFull; col += 2)
                        Winograd2x3iSetInput1(src + row * srcWidth + col, srcWidth, dst), dst += 16;
                    if (col < dstWidth)
                        Winograd2x3iSetInput1p(src + row * srcWidth + col, srcWidth, 0, 4, 0, tailW, dst), dst += 16;
                }
                if (row < dstHeight)
                {
                    if (pad)
                        Winograd2x3iSetInput1p(src + row * srcWidth, srcWidth, 0, tailH, 1, noseW, dst), dst += 16;
                    for (col = start; col < dstWidthFull; col += 2)
                        Winograd2x3iSetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, dst), dst += 16;
                    if (col < dstWidth)
                        Winograd2x3iSetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, tailW, dst), dst += 16;
                }
                src += srcWidth * srcHeight;
            }
        }

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

        SIMD_INLINE void Winograd2x3pSetFilter4(const float * src, float * dst, size_t stride)
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

        void Winograd2x3pSetFilter(const float * src, size_t size, float * dst)
        {
            size_t size4 = AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4, src += 36, dst += 4)
                Winograd2x3pSetFilter4(src, dst, size);
            for (; i < size; i += 1, src += 9, dst += 1)
                Base::Winograd2x3pSetFilter1(src, dst, size);
        }

        void Winograd2x3pSetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, int pad)
        {
            size_t dstHeight = pad ? srcHeight : srcHeight - 2;
            size_t dstWidth = pad ? srcWidth : srcWidth - 2;
            size_t dstStride = ((dstHeight + 1) / 2) * ((dstWidth + 1) / 2)*srcChannels;
            size_t dstHeightBody = AlignLo(dstHeight, 2);
            size_t dstWidthBody = AlignLo(dstWidth, 2);
            size_t noseW = Simd::Min<size_t>(4, dstWidth + 1);
            size_t noseH = Simd::Min<size_t>(4, dstHeight + 1);
            size_t startEdge = pad ? 2 : 0, startBody = 0;
            if (pad)
            {
                if (dstHeight == dstHeightBody)
                    dstHeightBody -= 2;
                if (dstWidth == dstWidthBody)
                    dstWidthBody -= 2;
                src -= srcWidth + 1;
                startBody = dstWidth < 9 ? 2 : 8;
            }
            size_t tailW = dstWidth - dstWidthBody + (pad ? 1 : 2);
            size_t tailH = dstHeight - dstHeightBody + (pad ? 1 : 2);
            size_t dstWidthEdge8 = startEdge + (dstWidthBody >= startEdge ? AlignLo(dstWidthBody - startEdge, 8) : 0);
            size_t dstWidthBody8 = startBody + (dstWidthBody >= startBody ? AlignLo(dstWidthBody - startBody, 8) : 0);
            for (size_t c = 0; c < srcChannels; ++c)
            {
                size_t row = 0, col = 0;
                if (pad)
                {
                    if (pad)
                        Base::Winograd2x3pSetInput1p(src, srcWidth, 1, noseH, 1, noseW, dst++, dstStride);
                    for (col = startEdge; col < dstWidthEdge8; col += 8, dst += 4)
                        Winograd2x3pSetInput4PadEdgeRow(src + col, srcWidth, 1, noseH, dst, dstStride);
                    for (; col < dstWidthBody; col += 2)
                        Base::Winograd2x3pSetInput1p(src + col, srcWidth, 1, noseH, 0, 4, dst++, dstStride);
                    if (col < dstWidth)
                        Base::Winograd2x3pSetInput1p(src + col, srcWidth, 1, noseH, 0, tailW, dst++, dstStride);
                }
                for (row = startEdge; row < dstHeightBody; row += 2)
                {
                    if (pad)
                    {
                        if(startBody == 8)
                            Winograd2x3pSetInput4Nose(src + row * srcWidth, srcWidth, dst, dstStride), dst +=4;
                        else
                            Base::Winograd2x3pSetInput1p(src + row * srcWidth, srcWidth, 0, 4, 1, noseW, dst++, dstStride);
                    }
                    for (col = startBody; col < dstWidthBody8; col += 8, dst += 4)
                        Winograd2x3pSetInput4Body(src + row * srcWidth + col, srcWidth, dst, dstStride);
                    for (; col < dstWidthBody; col += 2)
                        Base::Winograd2x3pSetInput1(src + row * srcWidth + col, srcWidth, dst++, dstStride);
                    if (col < dstWidth)
                        Base::Winograd2x3pSetInput1p(src + row * srcWidth + col, srcWidth, 0, 4, 0, tailW, dst++, dstStride);
                }
                if (row < dstHeight)
                {
                    if (pad)
                        Base::Winograd2x3pSetInput1p(src + row * srcWidth, srcWidth, 0, tailH, 1, noseW, dst++, dstStride);
                    for (col = startEdge; col < dstWidthEdge8; col += 8, dst += 4)
                        Winograd2x3pSetInput4PadEdgeRow(src + row * srcWidth + col, srcWidth, 0, tailH, dst, dstStride);
                    for (; col < dstWidthBody; col += 2)
                        Base::Winograd2x3pSetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, 4, dst++, dstStride);
                    if (col < dstWidth)
                        Base::Winograd2x3pSetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, tailW, dst++, dstStride);
                }
                src += srcWidth * srcHeight;
            }
        }

        void Winograd2x3pSetOutput(const float * src, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth)
        {
            size_t srcStride = ((dstHeight + 1) / 2) * ((dstWidth + 1) / 2)*dstChannels;
            size_t dstHeightFull = AlignLo(dstHeight, 2);
            size_t dstWidthFull = AlignLo(dstWidth, 2);
            size_t dstWidthFull8 = AlignLo(dstWidth, 8);
            for (size_t c = 0; c < dstChannels; ++c)
            {
                size_t row, col;
                for (row = 0; row < dstHeightFull; row += 2)
                {
                    for (col = 0; col < dstWidthFull8; col += 8, src += 4)
                        Winograd2x3pSetOutput4(src, srcStride, dst + row * dstWidth + col, dstWidth);
                    for (; col < dstWidthFull; col += 2)
                        Base::Winograd2x3pSetOutput1(src++, srcStride, dst + row * dstWidth + col, dstWidth);
                    if (col < dstWidth)
                        Base::Winograd2x3pSetOutput1p(src++, srcStride, dst + row * dstWidth + col, dstWidth, 2, dstWidth - col);
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstWidthFull; col += 2)
                        Base::Winograd2x3pSetOutput1p(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, 2);
                    if (col < dstWidth)
                        Base::Winograd2x3pSetOutput1p(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, dstWidth - col);
                }
                dst += dstHeight * dstWidth;
            }
        }

        SIMD_INLINE void Winograd4x3pSetFilter4Row(const __m128 * t, float * dst, size_t stride)
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

        SIMD_INLINE void Winograd4x3pSetFilter4(const float * src, float * dst, size_t stride)
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
            Winograd4x3pSetFilter4Row(t, dst + 0*stride, stride);

            t[0] = _mm_mul_ps(mr6, _mm_add_ps(_mm_add_ps(s[0], s[3]), s[6]));
            t[1] = _mm_mul_ps(mr6, _mm_add_ps(_mm_add_ps(s[1], s[4]), s[7]));
            t[2] = _mm_mul_ps(mr6, _mm_add_ps(_mm_add_ps(s[2], s[5]), s[8]));
            Winograd4x3pSetFilter4Row(t, dst + 6*stride, stride);

            t[0] = _mm_mul_ps(mr6, _mm_add_ps(_mm_sub_ps(s[0], s[3]), s[6]));
            t[1] = _mm_mul_ps(mr6, _mm_add_ps(_mm_sub_ps(s[1], s[4]), s[7]));
            t[2] = _mm_mul_ps(mr6, _mm_add_ps(_mm_sub_ps(s[2], s[5]), s[8]));
            Winograd4x3pSetFilter4Row(t, dst + 12 * stride, stride);

            t[0] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(r24, s[0]), _mm_mul_ps(r12, s[3])), _mm_mul_ps(r6, s[6]));
            t[1] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(r24, s[1]), _mm_mul_ps(r12, s[4])), _mm_mul_ps(r6, s[7]));
            t[2] = _mm_add_ps(_mm_add_ps(_mm_mul_ps(r24, s[2]), _mm_mul_ps(r12, s[5])), _mm_mul_ps(r6, s[8]));
            Winograd4x3pSetFilter4Row(t, dst + 18 * stride, stride);

            t[0] = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(r24, s[0]), _mm_mul_ps(r12, s[3])), _mm_mul_ps(r6, s[6]));
            t[1] = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(r24, s[1]), _mm_mul_ps(r12, s[4])), _mm_mul_ps(r6, s[7]));
            t[2] = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(r24, s[2]), _mm_mul_ps(r12, s[5])), _mm_mul_ps(r6, s[8]));
            Winograd4x3pSetFilter4Row(t, dst + 24 * stride, stride);

            Winograd4x3pSetFilter4Row(s + 6, dst + 30 * stride, stride);
        }

        void Winograd4x3pSetFilter(const float * src, size_t size, float * dst)
        {
            size_t size4 = AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4, src += 36, dst += 4)
                Winograd4x3pSetFilter4(src, dst, size);
            for (; i < size; i += 1, src += 9, dst += 1)
                Base::Winograd4x3pSetFilter1(src, dst, size);
        }
    }
#endif// SIMD_SSE_ENABLE
}
