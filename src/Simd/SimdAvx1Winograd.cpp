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
            float tmp[4 * 4] = { 0 };
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    tmp[row * 4 + col] = src[row * srcStride + col];
            Winograd2x3iSetInput1(tmp, 4, dst);
        }

        SIMD_INLINE __m256 Winograd2x3iSetInput2Row(__m256 t, __m256 k)
        {
            return _mm256_add_ps(_mm256_permute_ps(t, 0x64), _mm256_mul_ps(k, _mm256_permute_ps(t, 0xDA)));
        }

        SIMD_INLINE void Winograd2x3iSetInput4(const float * src, size_t srcStride, float * dst)
        {
            static const __m256 k = _mm256_setr_ps(-1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f);
            __m256 s0 = _mm256_loadu_ps(src + 0 * srcStride);
            __m256 s1 = _mm256_loadu_ps(src + 1 * srcStride);
            __m256 s2 = _mm256_loadu_ps(src + 2 * srcStride);
            __m256 s3 = _mm256_loadu_ps(src + 3 * srcStride);
            __m256 t0 = Winograd2x3iSetInput2Row(_mm256_sub_ps(s0, s2), k);
            __m256 t1 = Winograd2x3iSetInput2Row(_mm256_add_ps(s1, s2), k);
            __m256 t2 = Winograd2x3iSetInput2Row(_mm256_sub_ps(s2, s1), k);
            __m256 t3 = Winograd2x3iSetInput2Row(_mm256_sub_ps(s1, s3), k);
            src += 2;
            __m256 s4 = _mm256_loadu_ps(src + 0 * srcStride);
            __m256 s5 = _mm256_loadu_ps(src + 1 * srcStride);
            __m256 s6 = _mm256_loadu_ps(src + 2 * srcStride);
            __m256 s7 = _mm256_loadu_ps(src + 3 * srcStride);
            __m256 t4 = Winograd2x3iSetInput2Row(_mm256_sub_ps(s4, s6), k);
            __m256 t5 = Winograd2x3iSetInput2Row(_mm256_add_ps(s5, s6), k);
            __m256 t6 = Winograd2x3iSetInput2Row(_mm256_sub_ps(s6, s5), k);
            __m256 t7 = Winograd2x3iSetInput2Row(_mm256_sub_ps(s5, s7), k);
            _mm256_storeu_ps(dst + 0, _mm256_permute2f128_ps(t0, t1, 0x20));
            _mm256_storeu_ps(dst + 8, _mm256_permute2f128_ps(t2, t3, 0x20));
            _mm256_storeu_ps(dst + 16, _mm256_permute2f128_ps(t4, t5, 0x20));
            _mm256_storeu_ps(dst + 24, _mm256_permute2f128_ps(t6, t7, 0x20));
            _mm256_storeu_ps(dst + 32, _mm256_permute2f128_ps(t0, t1, 0x31));
            _mm256_storeu_ps(dst + 40, _mm256_permute2f128_ps(t2, t3, 0x31));
            _mm256_storeu_ps(dst + 48, _mm256_permute2f128_ps(t4, t5, 0x31));
            _mm256_storeu_ps(dst + 56, _mm256_permute2f128_ps(t6, t7, 0x31));
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

        SIMD_INLINE void Winograd2x3pSetInputLoad4(const float * src, __m256 * dst)
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

        SIMD_INLINE void Winograd2x3pSetInput8(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m256 t[16];
            Winograd2x3pSetInputLoad4(src + 0 * srcStride, t + 0);
            Winograd2x3pSetInputLoad4(src + 1 * srcStride, t + 4);
            Winograd2x3pSetInputLoad4(src + 2 * srcStride, t + 8);
            Winograd2x3pSetInputLoad4(src + 3 * srcStride, t + 12);
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

        SIMD_INLINE void Winograd2x3pSetInput8p(const float * src, size_t srcStride, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            float tmp[4 * 32] = { 0 };
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    tmp[row * 32 + col] = src[row * srcStride + col];
            Winograd2x3pSetInput8(tmp, 32, dst, dstStride);
        }

        void Winograd2x3pSetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, int pad)
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
            size_t dstWidthFull8 = dstWidthFull >= start ? AlignLo(dstWidthFull - start, 8) + start : start;
            size_t dstWidthFull16 = dstWidthFull >= start ? AlignLo(dstWidthFull - start, 16) + start : start;
            for (size_t c = 0; c < srcChannels; ++c)
            {
                size_t row = 0, col = 0;
                if (pad)
                {
                    if (pad)
                        Base::Winograd2x3pSetInput1p(src, srcWidth, 1, noseH, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull8; col += 8, dst += 4)
                        Sse::Winograd2x3pSetInput4p(src + col, srcWidth, 1, noseH, dst, dstStride);
                    for (; col < dstWidthFull; col += 2)
                        Base::Winograd2x3pSetInput1p(src + col, srcWidth, 1, noseH, 0, 4, dst++, dstStride);
                    if (col < dstWidth)
                        Base::Winograd2x3pSetInput1p(src + col, srcWidth, 1, noseH, 0, tailW, dst++, dstStride);
                }
                for (row = start; row < dstHeightFull; row += 2)
                {
                    if (pad)
                        Base::Winograd2x3pSetInput1p(src + row * srcWidth, srcWidth, 0, 4, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull16; col += 16, dst += 8)
                        Winograd2x3pSetInput8(src + row * srcWidth + col, srcWidth, dst, dstStride);
                    for (; col < dstWidthFull8; col += 8, dst += 4)
                       Sse::Winograd2x3pSetInput4(src + row * srcWidth + col, srcWidth, dst, dstStride);
                    for (; col < dstWidthFull; col += 2)
                        Base::Winograd2x3pSetInput1(src + row * srcWidth + col, srcWidth, dst++, dstStride);
                    if (col < dstWidth)
                        Base::Winograd2x3pSetInput1p(src + row * srcWidth + col, srcWidth, 0, 4, 0, tailW, dst++, dstStride);
                }
                if (row < dstHeight)
                {
                    if (pad)
                        Base::Winograd2x3pSetInput1p(src + row * srcWidth, srcWidth, 0, tailH, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull8; col += 8, dst += 4)
                        Sse::Winograd2x3pSetInput4p(src + row * srcWidth + col, srcWidth, 0, tailH, dst, dstStride);
                    for (; col < dstWidthFull; col += 2)
                        Base::Winograd2x3pSetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, 4, dst++, dstStride);
                    if (col < dstWidth)
                        Base::Winograd2x3pSetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, tailW, dst++, dstStride);
                }
                src += srcWidth * srcHeight;
            }
        }
    }
#endif// SIMD_AVX_ENABLE
}
