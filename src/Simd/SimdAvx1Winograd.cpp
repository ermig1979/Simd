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
#include "Simd/SimdSse1.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        SIMD_INLINE void Winograd2x3iSetInput1(const __m128 * src, float * dst)
        {
            static const __m256 _mpmm = _mm256_setr_ps(-1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f);
            __m256 t01 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_sub_ps(src[0], src[2])), _mm_add_ps(src[1], src[2]), 1);
            __m256 t23 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_sub_ps(src[2], src[1])), _mm_sub_ps(src[1], src[3]), 1);
            _mm256_storeu_ps(dst + 0, _mm256_add_ps(_mm256_permute_ps(t01, 0x64), _mm256_mul_ps(_mpmm, _mm256_permute_ps(t01, 0xDA))));
            _mm256_storeu_ps(dst + 8, _mm256_add_ps(_mm256_permute_ps(t23, 0x64), _mm256_mul_ps(_mpmm, _mm256_permute_ps(t23, 0xDA))));
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
            __m128 s[4] = { _mm_setzero_ps(), _mm_setzero_ps() , _mm_setzero_ps() , _mm_setzero_ps() };
            for (size_t row = rowB; row < rowE; ++row)
                s[row] = _mm_loadu_ps(src + row * srcStride);
            Winograd2x3iSetInput1(s, dst);
        }

        SIMD_INLINE void Winograd2x3iSetInput1p(const float * src, size_t srcStride, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst)
        {
            __m128 s[4] = { _mm_setzero_ps(), _mm_setzero_ps() , _mm_setzero_ps() , _mm_setzero_ps() };
            if (colB == 1)
                for (size_t row = rowB; row < rowE; ++row)
                    s[row] = Sse::LoadPadZeroNose1(src + row * srcStride);
            else if (colE == 2)
                for (size_t row = rowB; row < rowE; ++row)
                    s[row] = Sse::LoadPadZeroTail2(src + row * srcStride);
            else if (colE == 3)
                for (size_t row = rowB; row < rowE; ++row)
                    s[row] = Sse::LoadPadZeroTail1(src + row * srcStride);
            else
                for (size_t row = rowB; row < rowE; ++row)
                    _mm_loadu_ps(src + row * srcStride);
            Winograd2x3iSetInput1(s, dst);
        }

        SIMD_INLINE __m256 Winograd2x3iSetInput2Row(__m256 t, __m256 k)
        {
            return _mm256_add_ps(_mm256_permute_ps(t, 0x64), _mm256_mul_ps(k, _mm256_permute_ps(t, 0xDA)));
        }

        SIMD_INLINE void Winograd2x3iSetInput4(const float * src, size_t srcStride, float * dst, const __m256 & k)
        {
            __m256 s0 = _mm256_loadu_ps(src + 0 * srcStride);
            __m256 s1 = _mm256_loadu_ps(src + 1 * srcStride);
            __m256 s2 = _mm256_loadu_ps(src + 2 * srcStride);
            __m256 s3 = _mm256_loadu_ps(src + 3 * srcStride);
            __m256 t0 = Winograd2x3iSetInput2Row(_mm256_sub_ps(s0, s2), k);
            __m256 t1 = Winograd2x3iSetInput2Row(_mm256_add_ps(s1, s2), k);
            _mm256_storeu_ps(dst + 0, _mm256_permute2f128_ps(t0, t1, 0x20));
            _mm256_storeu_ps(dst + 32, _mm256_permute2f128_ps(t0, t1, 0x31));
            __m256 t2 = Winograd2x3iSetInput2Row(_mm256_sub_ps(s2, s1), k);
            __m256 t3 = Winograd2x3iSetInput2Row(_mm256_sub_ps(s1, s3), k);
            _mm256_storeu_ps(dst + 8, _mm256_permute2f128_ps(t2, t3, 0x20));
            _mm256_storeu_ps(dst + 40, _mm256_permute2f128_ps(t2, t3, 0x31));
        }

        SIMD_INLINE void Winograd2x3iSetInput4(const float * src, size_t srcStride, float * dst)
        {
            static const __m256 k = _mm256_setr_ps(-1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f);
            Winograd2x3iSetInput4(src + 0, srcStride, dst + 0, k);
            Winograd2x3iSetInput4(src + 2, srcStride, dst + 16, k);
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
            size_t dstWidthFull8 = start;// dstWidthFull >= start ? AlignLo(dstWidthFull - start, 8) + start : start;
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
                    for (col = start; col < dstWidthFull8; col += 8)
                        Winograd2x3iSetInput4(src + row * srcWidth + col, srcWidth, dst), dst += 64;
                    for (; col < dstWidthFull; col += 2)
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

        SIMD_INLINE void Winograd2x3iSetOutput4LoadRow(const float * src, __m128 * dst)
        {
            __m128 t[4];
            Load4(src, 16, t);
            dst[0] = _mm_add_ps(_mm_add_ps(t[0], t[1]), t[2]);
            dst[1] = _mm_sub_ps(_mm_sub_ps(t[1], t[2]), t[3]);
        }

        SIMD_INLINE void Winograd2x3iSetOutput4(const float * src, __m128 * dst)
        {
            __m128 t[8], d[4];
            Winograd2x3iSetOutput4LoadRow(src + 0, t + 0);
            Winograd2x3iSetOutput4LoadRow(src + 4, t + 2);
            Winograd2x3iSetOutput4LoadRow(src + 8, t + 4);
            Winograd2x3iSetOutput4LoadRow(src + 12, t + 6);
            d[0] = _mm_add_ps(_mm_add_ps(t[0], t[2]), t[4]);
            d[1] = _mm_add_ps(_mm_add_ps(t[1], t[3]), t[5]);
            d[2] = _mm_sub_ps(_mm_sub_ps(t[2], t[4]), t[6]);
            d[3] = _mm_sub_ps(_mm_sub_ps(t[3], t[5]), t[7]);
            dst[0] = _mm_unpacklo_ps(d[0], d[1]);
            dst[1] = _mm_unpackhi_ps(d[0], d[1]);
            dst[2] = _mm_unpacklo_ps(d[2], d[3]);
            dst[3] = _mm_unpackhi_ps(d[2], d[3]);
        }

        SIMD_INLINE void Winograd2x3iSetOutput4Body(const float * src, float * dst, size_t dstStride)
        {
            __m128 d[4];
            Winograd2x3iSetOutput4(src, d);
            _mm_storeu_ps(dst + 0 * dstStride + 0, d[0]);
            _mm_storeu_ps(dst + 0 * dstStride + 4, d[1]);
            _mm_storeu_ps(dst + 1 * dstStride + 0, d[2]);
            _mm_storeu_ps(dst + 1 * dstStride + 4, d[3]);
        }

        SIMD_INLINE void Winograd2x3iSetOutput4Edge(const float * src, float * dst)
        {
            __m128 t[6], d[2];
            Winograd2x3iSetOutput4LoadRow(src + 0, t + 0);
            Winograd2x3iSetOutput4LoadRow(src + 4, t + 2);
            Winograd2x3iSetOutput4LoadRow(src + 8, t + 4);
            d[0] = _mm_add_ps(_mm_add_ps(t[0], t[2]), t[4]);
            d[1] = _mm_add_ps(_mm_add_ps(t[1], t[3]), t[5]);
            _mm_storeu_ps(dst + 0, _mm_unpacklo_ps(d[0], d[1]));
            _mm_storeu_ps(dst + 4, _mm_unpackhi_ps(d[0], d[1]));
        }

        template<bool row, bool col> SIMD_INLINE void Winograd2x3iSetOutput4Edge(const float * src, float * dst, size_t dstStride, const __m128 & mask)
        {
            __m128 d[4];
            Winograd2x3iSetOutput4(src, d);
            _mm_storeu_ps(dst + 0, d[0]);
            if (col)
                _mm_storeu_ps(dst + 4, d[1]);
            else
                Sse::StoreMasked<false>(dst + 4, d[1], mask);
            if (row)
            {
                dst += dstStride;
                _mm_storeu_ps(dst + 0, d[2]);
                if (col)
                    _mm_storeu_ps(dst + 4, d[1]);
                else
                    Sse::StoreMasked<false>(dst + 4, d[3], mask);
            }
        }

        SIMD_INLINE void Load8(const float * src, __m256 * dst)
        {
            __m256 a0 = Load<false>(src + 0, src + 32);
            __m256 a1 = Load<false>(src + 16, src + 48);
            __m256 a2 = Load<false>(src + 64, src + 96);
            __m256 a3 = Load<false>(src + 80, src + 112);
            __m256 b0 = _mm256_unpacklo_ps(a0, a2);
            __m256 b1 = _mm256_unpackhi_ps(a0, a2);
            __m256 b2 = _mm256_unpacklo_ps(a1, a3);
            __m256 b3 = _mm256_unpackhi_ps(a1, a3);
            dst[0] = _mm256_unpacklo_ps(b0, b2);
            dst[1] = _mm256_unpackhi_ps(b0, b2);
            dst[2] = _mm256_unpacklo_ps(b1, b3);
            dst[3] = _mm256_unpackhi_ps(b1, b3);
        }

        SIMD_INLINE void Winograd2x3iSetOutput8LoadRow(const float * src, __m256 * dst)
        {
            __m256 t[4];
            Load8(src, t);
            dst[0] = _mm256_add_ps(_mm256_add_ps(t[0], t[1]), t[2]);
            dst[1] = _mm256_sub_ps(_mm256_sub_ps(t[1], t[2]), t[3]);
        }

        SIMD_INLINE void Winograd2x3iSetOutput8(const float * src, __m256 * dst)
        {
            __m256 t[8], d[4];
            Winograd2x3iSetOutput8LoadRow(src + 0, t + 0);
            Winograd2x3iSetOutput8LoadRow(src + 4, t + 2);
            Winograd2x3iSetOutput8LoadRow(src + 8, t + 4);
            Winograd2x3iSetOutput8LoadRow(src + 12, t + 6);
            d[0] = _mm256_add_ps(_mm256_add_ps(t[0], t[2]), t[4]);
            d[1] = _mm256_add_ps(_mm256_add_ps(t[1], t[3]), t[5]);
            d[2] = _mm256_sub_ps(_mm256_sub_ps(t[2], t[4]), t[6]);
            d[3] = _mm256_sub_ps(_mm256_sub_ps(t[3], t[5]), t[7]);
            dst[0] = _mm256_unpacklo_ps(d[0], d[1]);
            dst[1] = _mm256_unpackhi_ps(d[0], d[1]);
            dst[2] = _mm256_unpacklo_ps(d[2], d[3]);
            dst[3] = _mm256_unpackhi_ps(d[2], d[3]);
        }

        SIMD_INLINE void Winograd2x3iSetOutput8Body(const float * src, float * dst, size_t dstStride)
        {
            __m256 d[4];
            Winograd2x3iSetOutput8(src, d);
            _mm256_storeu_ps(dst + 0 * dstStride + 0, d[0]);
            _mm256_storeu_ps(dst + 0 * dstStride + 8, d[1]);
            _mm256_storeu_ps(dst + 1 * dstStride + 0, d[2]);
            _mm256_storeu_ps(dst + 1 * dstStride + 8, d[3]);
        }

        void Winograd2x3iSetOutput(const float * src, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth)
        {
            if (dstHeight < 2 || dstWidth < 16)
            {
                Sse::Winograd2x3iSetOutput(src, dst, dstChannels, dstHeight, dstWidth);
                return;
            }
            size_t tileH = (dstHeight + 1) / 2;
            size_t tileW = (dstWidth + 1) / 2;
            size_t dstH2 = AlignLo(dstHeight, 2);
            size_t dstW2 = AlignLo(dstWidth, 2);
            size_t dstW8 = AlignLo(dstWidth, 8);
            size_t dstW16 = AlignLo(dstWidth, 16);
            __m128 tailMask = Sse::LeftNotZero(4 + dstW2 - dstWidth);
            size_t tailCol = dstW2 < dstWidth ? dstWidth - 7 : dstWidth - 8;
            size_t tailRow = dstH2 < dstHeight ? dstHeight - 1 : dstHeight - 2;
            bool longTail = dstWidth - dstW8 > 4;
#if 0
            for (size_t c = 0; c < dstChannels; ++c)
            {
                size_t row = 0, tileY = 0;
                for (; row < dstH2; row += 2, tileY += 1)
                {
                    size_t col = 0, tileX = 0;
                    const float * s = src + tileY * tileW * 16;
                    float * d = dst + row * dstWidth;
                    for (; col < dstW8; col += 8, tileX += 64)
                        Winograd2x3iSetOutput4Body(s + tileX, d + col, dstWidth);
                    if (col < dstWidth)
                    {
                        if (longTail)
                            Winograd2x3iSetOutput4Edge<true, false>(s + (tileW - 4) * 16, d + tailCol, dstWidth, tailMask);
                        else
                        {
                            for (; col < dstW2; col += 2, tileX += 16)
                                Base::Winograd2x3iSetOutput1(s + tileX, d + col, dstWidth);
                            if (col < dstWidth)
                                Base::Winograd2x3iSetOutput1p(s + tileX, d + col, dstWidth, 2, dstWidth - col);
                        }
                    }
                }
                if (row < dstHeight)
                {
                    size_t col = 0, tileX = 0;
                    const float * s = src + (tileH - 1) * tileW * 16;
                    float * d = dst + (dstHeight - 1) * dstWidth;
                    for (; col < dstW8; col += 8, tileX += 64)
                        Winograd2x3iSetOutput4Edge(s + tileX, d + col);
                    if (col < dstWidth)
                    {
                        if (longTail)
                            Winograd2x3iSetOutput4Edge<false, false>(s + (tileW - 4) * 16, d + tailCol, dstWidth, tailMask);
                        else
                        {
                            for (; col < dstW2; col += 2, tileX += 16)
                                Base::Winograd2x3iSetOutput1p(s + tileX, d + col, dstWidth, dstHeight - row, 2);
                            if (col < dstWidth)
                                Base::Winograd2x3iSetOutput1p(s + tileX, d + col, dstWidth, dstHeight - row, dstWidth - col);
                        }
                    }
                }
                src += tileW * tileH * 16;
                dst += dstHeight * dstWidth;
            }
#else
            for (size_t c = 0; c < dstChannels; ++c)
            {
                size_t row = 0;
                for (; row < dstH2; row += 2)
                {
                    size_t col = 0;
                    for (; col < dstW16; col += 16)
                        Winograd2x3iSetOutput8Body(src, dst + row * dstWidth + col, dstWidth), src += 128;
                    for (; col < dstW8; col += 8)
                        Winograd2x3iSetOutput4Body(src, dst + row * dstWidth + col, dstWidth), src += 64;
                    for (; col < dstW2; col += 2)
                        Base::Winograd2x3iSetOutput1(src, dst + row * dstWidth + col, dstWidth), src += 16;
                    if (col < dstWidth)
                        Base::Winograd2x3iSetOutput1p(src, dst + row * dstWidth + col, dstWidth, 2, dstWidth - col), src += 16;
                }
                if (row < dstHeight)
                {
                    size_t col = 0;
                    for (; col < dstW8; col += 8)
                        Winograd2x3iSetOutput4Edge(src, dst + row * dstWidth + col), src += 64;
                    for (; col < dstW2; col += 2)
                        Base::Winograd2x3iSetOutput1p(src, dst + row * dstWidth + col, dstWidth, dstHeight - row, 2), src += 16;
                    if (col < dstWidth)
                        Base::Winograd2x3iSetOutput1p(src, dst + row * dstWidth + col, dstWidth, dstHeight - row, dstWidth - col), src += 16;
                }
                dst += dstHeight * dstWidth;
            }
#endif
        }

        SIMD_INLINE void Winograd2x3pSetInputLoad8Body(const float * src, __m256 * dst)
        {
            __m256 a0 = Load<false>(src + 0, src + 8);
            __m256 a1 = Load<false>(src + 2, src + 10);
            __m256 a2 = Load<false>(src + 4, src + 12);
            __m256 a3 = Load<false>(src + 6, src + 14);
            dst[0] = _mm256_shuffle_ps(a0, a2, 0x88);
            dst[1] = _mm256_shuffle_ps(a0, a2, 0xDD);
            dst[2] = _mm256_shuffle_ps(a1, a3, 0x88);
            dst[3] = _mm256_shuffle_ps(a1, a3, 0xDD);
        }

        SIMD_INLINE void Winograd2x3pSetInputLoad8Edge(const float * src, __m256 * dst, PadType pad)
        {
            __m256 a0 = Set(pad == PadNose1 ? Sse::LoadPadZeroNose1(src + 0) : _mm_loadu_ps(src + 0), _mm_loadu_ps(src + 8));
            __m256 a1 = Load<false>(src + 2, src + 10);
            __m256 a2 = Load<false>(src + 4, src + 12);
            __m256 a3 = Set(_mm_loadu_ps(src + 6), pad == PadTail2 ? Sse::LoadPadZeroTail2(src + 14) : (pad == PadTail1 ? Sse::LoadPadZeroTail1(src + 14) : _mm_loadu_ps(src + 14)));
            dst[0] = _mm256_shuffle_ps(a0, a2, 0x88);
            dst[1] = _mm256_shuffle_ps(a0, a2, 0xDD);
            dst[2] = _mm256_shuffle_ps(a1, a3, 0x88);
            dst[3] = _mm256_shuffle_ps(a1, a3, 0xDD);
        }

        SIMD_INLINE void Winograd2x3pSetInputLoad8Zero(__m256 * dst)
        {
            dst[0] = _mm256_setzero_ps();
            dst[1] = _mm256_setzero_ps();
            dst[2] = _mm256_setzero_ps();
            dst[3] = _mm256_setzero_ps();
        }

        SIMD_INLINE void Winograd2x3pSetInput8Store(const __m256 * t, float * dst, size_t dstStride)
        {
            _mm256_storeu_ps(dst + 0 * dstStride, _mm256_sub_ps(_mm256_sub_ps(t[0], t[8]), _mm256_sub_ps(t[2], t[10])));
            _mm256_storeu_ps(dst + 1 * dstStride, _mm256_add_ps(_mm256_sub_ps(t[1], t[9]), _mm256_sub_ps(t[2], t[10])));
            _mm256_storeu_ps(dst + 2 * dstStride, _mm256_sub_ps(_mm256_sub_ps(t[2], t[10]), _mm256_sub_ps(t[1], t[9])));
            _mm256_storeu_ps(dst + 3 * dstStride, _mm256_sub_ps(_mm256_sub_ps(t[1], t[9]), _mm256_sub_ps(t[3], t[11])));
            _mm256_storeu_ps(dst + 4 * dstStride, _mm256_sub_ps(_mm256_add_ps(t[4], t[8]), _mm256_add_ps(t[6], t[10])));
            _mm256_storeu_ps(dst + 5 * dstStride, _mm256_add_ps(_mm256_add_ps(t[5], t[9]), _mm256_add_ps(t[6], t[10])));
            _mm256_storeu_ps(dst + 6 * dstStride, _mm256_sub_ps(_mm256_add_ps(t[6], t[10]), _mm256_add_ps(t[5], t[9])));
            _mm256_storeu_ps(dst + 7 * dstStride, _mm256_sub_ps(_mm256_add_ps(t[5], t[9]), _mm256_add_ps(t[7], t[11])));
            _mm256_storeu_ps(dst + 8 * dstStride, _mm256_sub_ps(_mm256_sub_ps(t[8], t[4]), _mm256_sub_ps(t[10], t[6])));
            _mm256_storeu_ps(dst + 9 * dstStride, _mm256_add_ps(_mm256_sub_ps(t[9], t[5]), _mm256_sub_ps(t[10], t[6])));
            _mm256_storeu_ps(dst + 10 * dstStride, _mm256_sub_ps(_mm256_sub_ps(t[10], t[6]), _mm256_sub_ps(t[9], t[5])));
            _mm256_storeu_ps(dst + 11 * dstStride, _mm256_sub_ps(_mm256_sub_ps(t[9], t[5]), _mm256_sub_ps(t[11], t[7])));
            _mm256_storeu_ps(dst + 12 * dstStride, _mm256_sub_ps(_mm256_sub_ps(t[4], t[12]), _mm256_sub_ps(t[6], t[14])));
            _mm256_storeu_ps(dst + 13 * dstStride, _mm256_add_ps(_mm256_sub_ps(t[5], t[13]), _mm256_sub_ps(t[6], t[14])));
            _mm256_storeu_ps(dst + 14 * dstStride, _mm256_sub_ps(_mm256_sub_ps(t[6], t[14]), _mm256_sub_ps(t[5], t[13])));
            _mm256_storeu_ps(dst + 15 * dstStride, _mm256_sub_ps(_mm256_sub_ps(t[5], t[13]), _mm256_sub_ps(t[7], t[15])));
        }

        SIMD_INLINE void Winograd2x3pSetInput8Body(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m256 t[16];
            Winograd2x3pSetInputLoad8Body(src + 0 * srcStride, t + 0);
            Winograd2x3pSetInputLoad8Body(src + 1 * srcStride, t + 4);
            Winograd2x3pSetInputLoad8Body(src + 2 * srcStride, t + 8);
            Winograd2x3pSetInputLoad8Body(src + 3 * srcStride, t + 12);
            Winograd2x3pSetInput8Store(t, dst, dstStride);
        }

        SIMD_INLINE void Winograd2x3pSetInput8Edge(const float * src, size_t srcStride, PadType rowPad, PadType colPad, float * dst, size_t dstStride)
        {
            __m256 t[16];
            if (rowPad == PadNose1)
                Winograd2x3pSetInputLoad8Zero(t + 0);
            else
                Winograd2x3pSetInputLoad8Edge(src + 0 * srcStride, t + 0, colPad);
            Winograd2x3pSetInputLoad8Edge(src + 1 * srcStride, t + 4, colPad);
            if (rowPad == PadTail2)
                Winograd2x3pSetInputLoad8Zero(t + 8);
            else
                Winograd2x3pSetInputLoad8Edge(src + 2 * srcStride, t + 8, colPad);
            if (rowPad >= PadTail1)
                Winograd2x3pSetInputLoad8Zero(t + 12);
            else
                Winograd2x3pSetInputLoad8Edge(src + 3 * srcStride, t + 12, colPad);
            Winograd2x3pSetInput8Store(t, dst, dstStride);
        }

        void Winograd2x3pSetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, int pad)
        {
            if (srcHeight < 4 || srcWidth < 18)
            {
                Sse::Winograd2x3pSetInput(src, srcChannels, srcHeight, srcWidth, dst, pad);
                return;
            }
            size_t dstH = pad ? srcHeight : srcHeight - 2;
            size_t dstW = pad ? srcWidth : srcWidth - 2;
            size_t tileH = (dstH + 1) / 2;
            size_t tileW = (dstW + 1) / 2;
            size_t dstStride = srcChannels * tileH*tileW;

            size_t dstH2 = AlignLo(dstH, 2);
            size_t dstW2 = AlignLo(dstW, 2);
            size_t dstW16 = AlignLo(dstW, 16);
            if (pad && dstW16 == dstW)
                dstW16 -= 16;
            PadType rowPad = dstH2 < dstH ? PadTail1 : PadNone;
            PadType colPad = dstW2 < dstW ? PadTail1 : PadNone;
            size_t tailCol = dstW2 < dstW ? dstW - 15 : dstW - 16;
            size_t tailRow = dstH2 < dstH ? dstH - 1 : dstH - 2;
            bool specialColTail = dstW16 < dstW || pad;
            bool specialRowTail = dstH2 < dstH || pad;
            if (pad)
            {
                src -= srcWidth + 1;
                rowPad = dstH2 < dstH ? PadTail2 : PadTail1;
                colPad = dstW2 < dstW ? PadTail2 : PadTail1;
            }
            for (size_t c = 0; c < srcChannels; ++c)
            {
                size_t row = 0, tileY = 0;
                if (pad)
                {
                    size_t col = 0, tileX = 0;
                    const float * s = src + row * srcWidth;
                    float * d = dst + tileY * tileW;
                    if (pad)
                        Winograd2x3pSetInput8Edge(s + col, srcWidth, PadNose1, PadNose1, d + tileX, dstStride), col += 16, tileX += 8;
                    for (; col < dstW16; col += 16, tileX += 8)
                        Winograd2x3pSetInput8Edge(s + col, srcWidth, PadNose1, PadNone, d + tileX, dstStride);
                    if (specialColTail)
                        Winograd2x3pSetInput8Edge(s + tailCol, srcWidth, PadNose1, colPad, d + tileW - 8, dstStride);
                    row += 2, tileY += 1;
                }
                for (; row < dstH2; row += 2, tileY += 1)
                {
                    size_t col = 0, tileX = 0;
                    const float * s = src + row * srcWidth;
                    float * d = dst + tileY * tileW;
                    if (pad)
                        Winograd2x3pSetInput8Edge(s + col, srcWidth, PadNone, PadNose1, d + tileX, dstStride), col += 16, tileX += 8;
                    for (; col < dstW16; col += 16, tileX += 8)
                        Winograd2x3pSetInput8Body(s + col, srcWidth, d + tileX, dstStride);
                    if (specialColTail)
                        Winograd2x3pSetInput8Edge(s + tailCol, srcWidth, PadNone, colPad, d + tileW - 8, dstStride);
                }
                if (specialRowTail)
                {
                    size_t col = 0, tileX = 0;
                    const float * s = src + tailRow * srcWidth;
                    float * d = dst + (tileH - 1) * tileW;
                    if (pad)
                        Winograd2x3pSetInput8Edge(s + col, srcWidth, rowPad, PadNose1, d + tileX, dstStride), col += 16, tileX += 8;
                    for (; col < dstW16; col += 16, tileX += 8)
                        Winograd2x3pSetInput8Edge(s + col, srcWidth, rowPad, PadNone, d + tileX, dstStride);
                    if (specialColTail)
                        Winograd2x3pSetInput8Edge(s + tailCol, srcWidth, rowPad, colPad, d + tileW - 8, dstStride);
                }
                src += srcWidth * srcHeight;
                dst += tileW * tileH;
            }
        }

        SIMD_INLINE void Winograd2x3pSetOutputLoad2t(const float * src, size_t srcStride, __m256 * dst)
        {
            __m256 s0 = _mm256_loadu_ps(src + 0 * srcStride);
            __m256 s1 = _mm256_loadu_ps(src + 1 * srcStride);
            __m256 s2 = _mm256_loadu_ps(src + 2 * srcStride);
            __m256 s3 = _mm256_loadu_ps(src + 3 * srcStride);
            dst[0] = _mm256_add_ps(_mm256_add_ps(s0, s1), s2);
            dst[1] = _mm256_sub_ps(_mm256_sub_ps(s1, s2), s3);
        }

        SIMD_INLINE void Winograd2x3pSetOutput8(const float * src, size_t srcStride, __m256 * dst)
        {
            __m256 t[8], d[4], u[4];
            Winograd2x3pSetOutputLoad2t(src + 0 * srcStride, srcStride, t + 0);
            Winograd2x3pSetOutputLoad2t(src + 4 * srcStride, srcStride, t + 2);
            Winograd2x3pSetOutputLoad2t(src + 8 * srcStride, srcStride, t + 4);
            Winograd2x3pSetOutputLoad2t(src + 12 * srcStride, srcStride, t + 6);
            d[0] = _mm256_add_ps(_mm256_add_ps(t[0], t[2]), t[4]);
            d[1] = _mm256_add_ps(_mm256_add_ps(t[1], t[3]), t[5]);
            d[2] = _mm256_sub_ps(_mm256_sub_ps(t[2], t[4]), t[6]);
            d[3] = _mm256_sub_ps(_mm256_sub_ps(t[3], t[5]), t[7]);
            u[0] = _mm256_unpacklo_ps(d[0], d[1]);
            u[1] = _mm256_unpackhi_ps(d[0], d[1]);
            u[2] = _mm256_unpacklo_ps(d[2], d[3]);
            u[3] = _mm256_unpackhi_ps(d[2], d[3]);
            dst[0] = _mm256_permute2f128_ps(u[0], u[1], 0x20);
            dst[1] = _mm256_permute2f128_ps(u[0], u[1], 0x31);
            dst[2] = _mm256_permute2f128_ps(u[2], u[3], 0x20);
            dst[3] = _mm256_permute2f128_ps(u[2], u[3], 0x31);
        }

        SIMD_INLINE void Winograd2x3pSetOutput8Body(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m256 d[4];
            Winograd2x3pSetOutput8(src, srcStride, d);
            _mm256_storeu_ps(dst + 0 * dstStride + 0, d[0]);
            _mm256_storeu_ps(dst + 0 * dstStride + 8, d[1]);
            _mm256_storeu_ps(dst + 1 * dstStride + 0, d[2]);
            _mm256_storeu_ps(dst + 1 * dstStride + 8, d[3]);
        }

        SIMD_INLINE void Winograd2x3pSetOutput8Edge(const float * src, size_t srcStride, float * dst, size_t dstStride, bool lastRow, bool lastCol, const __m256 & mask)
        {
            __m256 d[4];
            Winograd2x3pSetOutput8(src, srcStride, d);
            _mm256_storeu_ps(dst + 0, d[0]);
            if (lastCol)
                _mm256_storeu_ps(dst + 8, d[1]);
            else
                StoreMasked<false>(dst + 8, d[1], mask);
            if (lastRow)
            {
                dst += dstStride;
                _mm256_storeu_ps(dst + 0, d[2]);
                if (lastCol)
                    _mm256_storeu_ps(dst + 8, d[1]);
                else
                    StoreMasked<false>(dst + 8, d[3], mask);
            }
        }

        void Winograd2x3pSetOutput(const float * src, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth)
        {
            if (dstHeight < 2 || dstWidth < 16)
            {
                Sse::Winograd2x3pSetOutput(src, dst, dstChannels, dstHeight, dstWidth);
                return;
            }
            size_t tileH = (dstHeight + 1) / 2;
            size_t tileW = (dstWidth + 1) / 2;
            size_t srcStride = dstChannels * tileH*tileW;
            size_t dstH2 = AlignLo(dstHeight, 2);
            size_t dstW2 = AlignLo(dstWidth, 2);
            size_t dstW16 = AlignLo(dstWidth, 16);
            __m256 tailMask = LeftNotZero(8 + dstW2 - dstWidth);
            size_t tailCol = dstW2 < dstWidth ? dstWidth - 15 : dstWidth - 16;
            size_t tailRow = dstH2 < dstHeight ? dstHeight - 1 : dstHeight - 2;
            for (size_t c = 0; c < dstChannels; ++c)
            {
                size_t row = 0, tileY = 0;
                for (; row < dstH2; row += 2, tileY += 1)
                {
                    size_t col = 0, tileX = 0;
                    const float * s = src + tileY * tileW;
                    float * d = dst + row * dstWidth;
                    for (; col < dstW16; col += 16, tileX += 8)
                        Winograd2x3pSetOutput8Body(s + tileX, srcStride, d + col, dstWidth);
                    if (col < dstWidth)
                        Winograd2x3pSetOutput8Edge(s + tileW - 8, srcStride, d + tailCol, dstWidth, true, false, tailMask);
                }
                if (row < dstHeight)
                {
                    size_t col = 0, tileX = 0;
                    const float * s = src + (tileH - 1) * tileW;
                    float * d = dst + (dstHeight - 1) * dstWidth;
                    for (; col < dstW16; col += 16, tileX += 8)
                        Winograd2x3pSetOutput8Edge(s + tileX, srcStride, d + col, dstWidth, false, true, tailMask);
                    if (col < dstWidth)
                        Winograd2x3pSetOutput8Edge(s + tileW - 8, srcStride, d + tailCol, dstWidth, false, false, tailMask);
                }
                src += tileW * tileH;
                dst += dstHeight * dstWidth;
            }
        }
    }
#endif// SIMD_AVX_ENABLE
}
