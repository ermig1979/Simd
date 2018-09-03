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
        SIMD_INLINE void Winograd2x3iSetInput1(const float * src, size_t srcStride, float * dst)
        {
            static const __m256 _mpmm = _mm256_setr_ps(-1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f);
            __m128 s0 = _mm_loadu_ps(src + 0 * srcStride);
            __m128 s1 = _mm_loadu_ps(src + 1 * srcStride);
            __m128 s2 = _mm_loadu_ps(src + 2 * srcStride);
            __m128 s3 = _mm_loadu_ps(src + 3 * srcStride);
            __m256 t01 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_sub_ps(s0, s2)), _mm_add_ps(s1, s2), 1);
            __m256 t23 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_sub_ps(s2, s1)), _mm_sub_ps(s1, s3), 1);
            _mm256_storeu_ps(dst + 0, _mm256_add_ps(_mm256_permute_ps(t01, 0x64), _mm256_mul_ps(_mpmm, _mm256_permute_ps(t01, 0xDA))));
            _mm256_storeu_ps(dst + 8, _mm256_add_ps(_mm256_permute_ps(t23, 0x64), _mm256_mul_ps(_mpmm, _mm256_permute_ps(t23, 0xDA))));
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
                        Winograd2x3iSetInput1p(src + col, srcWidth, 1, noseH, 0, 4, dst), dst += 16;
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
                        Winograd2x3iSetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, 4, dst), dst += 16;
                    if (col < dstWidth)
                        Winograd2x3iSetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, tailW, dst), dst += 16;
                }
                src += srcWidth * srcHeight;
            }
        }
    }
#endif// SIMD_AVX_ENABLE
}
