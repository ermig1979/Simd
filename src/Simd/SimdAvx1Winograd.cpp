/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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

        SIMD_INLINE void Winograd2x3SetFilter8t(const float * src, float * dst, size_t stride)
        {
            const __m256 r2 = _mm256_set1_ps(1.0f / 2.0f);
            const __m256 r4 = _mm256_set1_ps(1.0f / 4.0f);

            __m256 s[9];
            s[0] = _mm256_loadu_ps(src + 0 * stride);
            s[1] = _mm256_loadu_ps(src + 1 * stride);
            s[2] = _mm256_loadu_ps(src + 2 * stride);
            s[3] = _mm256_loadu_ps(src + 3 * stride);
            s[4] = _mm256_loadu_ps(src + 4 * stride);
            s[5] = _mm256_loadu_ps(src + 5 * stride);
            s[6] = _mm256_loadu_ps(src + 6 * stride);
            s[7] = _mm256_loadu_ps(src + 7 * stride);
            s[8] = _mm256_loadu_ps(src + 8 * stride);

            _mm256_storeu_ps(dst + 0 * stride, s[0]);
            __m256 _0a2 = _mm256_add_ps(s[0], s[2]);
            _mm256_storeu_ps(dst + 1 * stride, _mm256_mul_ps(_mm256_add_ps(_0a2, s[1]), r2));
            _mm256_storeu_ps(dst + 2 * stride, _mm256_mul_ps(_mm256_sub_ps(_0a2, s[1]), r2));
            _mm256_storeu_ps(dst + 3 * stride, s[2]);

            __m256 _0a6a3 = _mm256_add_ps(_mm256_add_ps(s[0], s[6]), s[3]);
            _mm256_storeu_ps(dst + 4 * stride, _mm256_mul_ps(_0a6a3, r2));
            __m256 _2a8a5 = _mm256_add_ps(_mm256_add_ps(s[2], s[8]), s[5]);
            __m256 _1a7a4 = _mm256_add_ps(_mm256_add_ps(s[1], s[7]), s[4]);
            _mm256_storeu_ps(dst + 5 * stride, _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(_0a6a3, _2a8a5), _1a7a4), r4));
            _mm256_storeu_ps(dst + 6 * stride, _mm256_mul_ps(_mm256_sub_ps(_mm256_add_ps(_0a6a3, _2a8a5), _1a7a4), r4));
            _mm256_storeu_ps(dst + 7 * stride, _mm256_mul_ps(_2a8a5, r2));

            __m256 _0a6s3 = _mm256_sub_ps(_mm256_add_ps(s[0], s[6]), s[3]);
            _mm256_storeu_ps(dst + 8 * stride, _mm256_mul_ps(_0a6s3, r2));
            __m256 _2a8s5 = _mm256_sub_ps(_mm256_add_ps(s[2], s[8]), s[5]);
            __m256 _1a7s4 = _mm256_sub_ps(_mm256_add_ps(s[1], s[7]), s[4]);
            _mm256_storeu_ps(dst + 9 * stride, _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(_0a6s3, _2a8s5), _1a7s4), r4));
            _mm256_storeu_ps(dst + 10 * stride, _mm256_mul_ps(_mm256_sub_ps(_mm256_add_ps(_0a6s3, _2a8s5), _1a7s4), r4));
            _mm256_storeu_ps(dst + 11 * stride, _mm256_mul_ps(_2a8s5, r2));

            _mm256_storeu_ps(dst + 12 * stride, s[6]);
            __m256 _6a8 = _mm256_add_ps(s[6], s[8]);
            _mm256_storeu_ps(dst + 13 * stride, _mm256_mul_ps(_mm256_add_ps(_6a8, s[7]), r2));
            _mm256_storeu_ps(dst + 14 * stride, _mm256_mul_ps(_mm256_sub_ps(_6a8, s[7]), r2));
            _mm256_storeu_ps(dst + 15 * stride, s[8]);
        }

        void Winograd2x3SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
        {
            if (trans)
            {
                size_t size8 = AlignLo(size, 8), i = 0;
                for (; i < size8; i += 8)
                    Winograd2x3SetFilter8t(src + i, dst + i, size);
                for (; i < size; i += 1)
                    Base::Winograd2x3SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                Sse::Winograd2x3SetFilter(src, size, dst, trans);
            }
        }

        SIMD_INLINE void Winograd2x3SetInputLoad8n(const float * src, __m256 * dst)
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

        SIMD_INLINE void Winograd2x3SetInputLoad8n(const float * src, __m256 * dst, PadType pad)
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

        SIMD_INLINE void Winograd2x3SetInputLoad8z(__m256 * dst)
        {
            dst[0] = _mm256_setzero_ps();
            dst[1] = _mm256_setzero_ps();
            dst[2] = _mm256_setzero_ps();
            dst[3] = _mm256_setzero_ps();
        }

        SIMD_INLINE void Winograd2x3SetInput8Store(const __m256 * src, float * dst, size_t stride)
        {
            _mm256_storeu_ps(dst + 0 * stride, _mm256_sub_ps(_mm256_sub_ps(src[0], src[8]), _mm256_sub_ps(src[2], src[10])));
            _mm256_storeu_ps(dst + 1 * stride, _mm256_add_ps(_mm256_sub_ps(src[1], src[9]), _mm256_sub_ps(src[2], src[10])));
            _mm256_storeu_ps(dst + 2 * stride, _mm256_sub_ps(_mm256_sub_ps(src[2], src[10]), _mm256_sub_ps(src[1], src[9])));
            _mm256_storeu_ps(dst + 3 * stride, _mm256_sub_ps(_mm256_sub_ps(src[1], src[9]), _mm256_sub_ps(src[3], src[11])));
            _mm256_storeu_ps(dst + 4 * stride, _mm256_sub_ps(_mm256_add_ps(src[4], src[8]), _mm256_add_ps(src[6], src[10])));
            _mm256_storeu_ps(dst + 5 * stride, _mm256_add_ps(_mm256_add_ps(src[5], src[9]), _mm256_add_ps(src[6], src[10])));
            _mm256_storeu_ps(dst + 6 * stride, _mm256_sub_ps(_mm256_add_ps(src[6], src[10]), _mm256_add_ps(src[5], src[9])));
            _mm256_storeu_ps(dst + 7 * stride, _mm256_sub_ps(_mm256_add_ps(src[5], src[9]), _mm256_add_ps(src[7], src[11])));
            _mm256_storeu_ps(dst + 8 * stride, _mm256_sub_ps(_mm256_sub_ps(src[8], src[4]), _mm256_sub_ps(src[10], src[6])));
            _mm256_storeu_ps(dst + 9 * stride, _mm256_add_ps(_mm256_sub_ps(src[9], src[5]), _mm256_sub_ps(src[10], src[6])));
            _mm256_storeu_ps(dst + 10 * stride, _mm256_sub_ps(_mm256_sub_ps(src[10], src[6]), _mm256_sub_ps(src[9], src[5])));
            _mm256_storeu_ps(dst + 11 * stride, _mm256_sub_ps(_mm256_sub_ps(src[9], src[5]), _mm256_sub_ps(src[11], src[7])));
            _mm256_storeu_ps(dst + 12 * stride, _mm256_sub_ps(_mm256_sub_ps(src[4], src[12]), _mm256_sub_ps(src[6], src[14])));
            _mm256_storeu_ps(dst + 13 * stride, _mm256_add_ps(_mm256_sub_ps(src[5], src[13]), _mm256_sub_ps(src[6], src[14])));
            _mm256_storeu_ps(dst + 14 * stride, _mm256_sub_ps(_mm256_sub_ps(src[6], src[14]), _mm256_sub_ps(src[5], src[13])));
            _mm256_storeu_ps(dst + 15 * stride, _mm256_sub_ps(_mm256_sub_ps(src[5], src[13]), _mm256_sub_ps(src[7], src[15])));
        }

        SIMD_INLINE void Winograd2x3SetInput8n(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m256 t[16];
            Winograd2x3SetInputLoad8n(src + 0 * srcStride, t + 0);
            Winograd2x3SetInputLoad8n(src + 1 * srcStride, t + 4);
            Winograd2x3SetInputLoad8n(src + 2 * srcStride, t + 8);
            Winograd2x3SetInputLoad8n(src + 3 * srcStride, t + 12);
            Winograd2x3SetInput8Store(t, dst, dstStride);
        }

        SIMD_INLINE void Winograd2x3SetInput8n(const float * src, size_t srcStride, PadType rowPad, PadType colPad, float * dst, size_t dstStride)
        {
            __m256 t[16];
            if (rowPad == PadNose1)
                Winograd2x3SetInputLoad8z(t + 0);
            else
                Winograd2x3SetInputLoad8n(src + 0 * srcStride, t + 0, colPad);
            Winograd2x3SetInputLoad8n(src + 1 * srcStride, t + 4, colPad);
            if (rowPad == PadTail2)
                Winograd2x3SetInputLoad8z(t + 8);
            else
                Winograd2x3SetInputLoad8n(src + 2 * srcStride, t + 8, colPad);
            if (rowPad >= PadTail1)
                Winograd2x3SetInputLoad8z(t + 12);
            else
                Winograd2x3SetInputLoad8n(src + 3 * srcStride, t + 12, colPad);
            Winograd2x3SetInput8Store(t, dst, dstStride);
        }

        SIMD_INLINE void Winograd2x3SetInput8t(const float * src, size_t srcS, size_t srcC, __m256 dst[16])
        {
            dst[0] = _mm256_loadu_ps(src + 0 * srcS + 0 * srcC);
            dst[1] = _mm256_loadu_ps(src + 0 * srcS + 1 * srcC);
            dst[2] = _mm256_loadu_ps(src + 0 * srcS + 2 * srcC);
            dst[3] = _mm256_loadu_ps(src + 0 * srcS + 3 * srcC);
            dst[4] = _mm256_loadu_ps(src + 1 * srcS + 0 * srcC);
            dst[5] = _mm256_loadu_ps(src + 1 * srcS + 1 * srcC);
            dst[6] = _mm256_loadu_ps(src + 1 * srcS + 2 * srcC);
            dst[7] = _mm256_loadu_ps(src + 1 * srcS + 3 * srcC);
            dst[8] = _mm256_loadu_ps(src + 2 * srcS + 0 * srcC);
            dst[9] = _mm256_loadu_ps(src + 2 * srcS + 1 * srcC);
            dst[10] = _mm256_loadu_ps(src + 2 * srcS + 2 * srcC);
            dst[11] = _mm256_loadu_ps(src + 2 * srcS + 3 * srcC);
            dst[12] = _mm256_loadu_ps(src + 3 * srcS + 0 * srcC);
            dst[13] = _mm256_loadu_ps(src + 3 * srcS + 1 * srcC);
            dst[14] = _mm256_loadu_ps(src + 3 * srcS + 2 * srcC);
            dst[15] = _mm256_loadu_ps(src + 3 * srcS + 3 * srcC);
        }

        SIMD_INLINE void Winograd2x3SetInput8t(const float * src, size_t srcW, size_t srcC, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
            {
                __m256 tmp[16];
                Winograd2x3SetInput8t(src + c, srcS, srcC, tmp);
                Winograd2x3SetInput8Store(tmp, dst + c, dstStride);
            }
            if (srcCF < srcC)
            {
                __m256 tmp[16];
                Winograd2x3SetInput8t(src + srcC - F, srcS, srcC, tmp);
                Winograd2x3SetInput8Store(tmp, dst + srcC - F, dstStride);
            }
        }

        SIMD_INLINE void Winograd2x3SetInput8t(const float * src, size_t srcS, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, __m256 dst[16])
        {
            for (size_t i = 0; i < 16; ++i)
                dst[i] = _mm256_setzero_ps();
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    dst[row * 4 + col] = _mm256_loadu_ps(src + row * srcS + col * srcC);
        }

        SIMD_INLINE void Winograd2x3SetInput8t(const float * src, size_t srcW, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
            {
                __m256 tmp[16];
                Winograd2x3SetInput8t(src + c, srcS, srcC, rowB, rowE, colB, colE, tmp);
                Winograd2x3SetInput8Store(tmp, dst + c, dstStride);
            }
            if (srcCF < srcC)
            {
                __m256 tmp[16];
                Winograd2x3SetInput8t(src + srcC - F, srcS, srcC, rowB, rowE, colB, colE, tmp);
                Winograd2x3SetInput8Store(tmp, dst + srcC - F, dstStride);
            }
        }

        void Winograd2x3SetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, SimdBool pad, SimdBool trans)
        {
            if (trans ? (srcChannels < 8) : (srcHeight < 4 || srcWidth < 18))
            {
                Sse::Winograd2x3SetInput(src, srcChannels, srcHeight, srcWidth, dst, pad, trans);
                return;
            }
            size_t dstH = pad ? srcHeight : srcHeight - 2;
            size_t dstW = pad ? srcWidth : srcWidth - 2;
            size_t tileH = (dstH + 1) / 2;
            size_t tileW = (dstW + 1) / 2;
            size_t dstStride = srcChannels * tileH*tileW;
            size_t dstH2 = AlignLo(dstH, 2);
            size_t dstW2 = AlignLo(dstW, 2);
            if (trans)
            {
                size_t noseW = Simd::Min<size_t>(4, dstW + 1);
                size_t noseH = Simd::Min<size_t>(4, dstH + 1);
                size_t start = pad ? 2 : 0;
                if (pad)
                {
                    if (dstH == dstH2)
                        dstH2 -= 2;
                    if (dstW == dstW2)
                        dstW2 -= 2;
                    src -= (srcWidth + 1)*srcChannels;
                }
                size_t tailW = dstW - dstW2 + (pad ? 1 : 2);
                size_t tailH = dstH - dstH2 + (pad ? 1 : 2);
                size_t row = 0, col = 0;
                if (pad)
                {
                    if (pad)
                        Winograd2x3SetInput8t(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstW2; col += 2)
                        Winograd2x3SetInput8t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 4, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        Winograd2x3SetInput8t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                for (row = start; row < dstH2; row += 2)
                {
                    if (pad)
                        Winograd2x3SetInput8t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 4, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstW2; col += 2)
                        Winograd2x3SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        Winograd2x3SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 4, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                if (row < dstH)
                {
                    if (pad)
                        Winograd2x3SetInput8t(src + row * srcWidth* srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstW2; col += 2)
                        Winograd2x3SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 4, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        Winograd2x3SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
            }
            else
            {
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
                    if (dstH2 == dstH)
                        dstH2 -= 2;
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
                            Winograd2x3SetInput8n(s + col, srcWidth, PadNose1, PadNose1, d + tileX, dstStride), col += 16, tileX += 8;
                        for (; col < dstW16; col += 16, tileX += 8)
                            Winograd2x3SetInput8n(s + col, srcWidth, PadNose1, PadNone, d + tileX, dstStride);
                        if (specialColTail)
                            Winograd2x3SetInput8n(s + tailCol, srcWidth, PadNose1, colPad, d + tileW - 8, dstStride);
                        row += 2, tileY += 1;
                    }
                    for (; row < dstH2; row += 2, tileY += 1)
                    {
                        size_t col = 0, tileX = 0;
                        const float * s = src + row * srcWidth;
                        float * d = dst + tileY * tileW;
                        if (pad)
                            Winograd2x3SetInput8n(s + col, srcWidth, PadNone, PadNose1, d + tileX, dstStride), col += 16, tileX += 8;
                        for (; col < dstW16; col += 16, tileX += 8)
                            Winograd2x3SetInput8n(s + col, srcWidth, d + tileX, dstStride);
                        if (specialColTail)
                            Winograd2x3SetInput8n(s + tailCol, srcWidth, PadNone, colPad, d + tileW - 8, dstStride);
                    }
                    if (specialRowTail)
                    {
                        size_t col = 0, tileX = 0;
                        const float * s = src + tailRow * srcWidth;
                        float * d = dst + (tileH - 1) * tileW;
                        if (pad)
                            Winograd2x3SetInput8n(s + col, srcWidth, rowPad, PadNose1, d + tileX, dstStride), col += 16, tileX += 8;
                        for (; col < dstW16; col += 16, tileX += 8)
                            Winograd2x3SetInput8n(s + col, srcWidth, rowPad, PadNone, d + tileX, dstStride);
                        if (specialColTail)
                            Winograd2x3SetInput8n(s + tailCol, srcWidth, rowPad, colPad, d + tileW - 8, dstStride);
                    }
                    src += srcWidth * srcHeight;
                    dst += tileW * tileH;
                }
            }
        }

        SIMD_INLINE void Winograd2x3SetOutputLoad2t(const float * src, size_t srcStride, __m256 * dst)
        {
            __m256 s0 = _mm256_loadu_ps(src + 0 * srcStride);
            __m256 s1 = _mm256_loadu_ps(src + 1 * srcStride);
            __m256 s2 = _mm256_loadu_ps(src + 2 * srcStride);
            __m256 s3 = _mm256_loadu_ps(src + 3 * srcStride);
            dst[0] = _mm256_add_ps(_mm256_add_ps(s0, s1), s2);
            dst[1] = _mm256_sub_ps(_mm256_sub_ps(s1, s2), s3);
        }

        SIMD_INLINE void Winograd2x3SetOutput8(const float * src, size_t srcStride, __m256 * dst)
        {
            __m256 t[8], d[4], u[4];
            Winograd2x3SetOutputLoad2t(src + 0 * srcStride, srcStride, t + 0);
            Winograd2x3SetOutputLoad2t(src + 4 * srcStride, srcStride, t + 2);
            Winograd2x3SetOutputLoad2t(src + 8 * srcStride, srcStride, t + 4);
            Winograd2x3SetOutputLoad2t(src + 12 * srcStride, srcStride, t + 6);
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

        SIMD_INLINE void Winograd2x3SetOutput8Body(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m256 d[4];
            Winograd2x3SetOutput8(src, srcStride, d);
            _mm256_storeu_ps(dst + 0 * dstStride + 0, d[0]);
            _mm256_storeu_ps(dst + 0 * dstStride + 8, d[1]);
            _mm256_storeu_ps(dst + 1 * dstStride + 0, d[2]);
            _mm256_storeu_ps(dst + 1 * dstStride + 8, d[3]);
        }

        SIMD_INLINE void Winograd2x3SetOutput8Edge(const float * src, size_t srcStride, float * dst, size_t dstStride, bool lastRow, bool lastCol, const __m256 & mask)
        {
            __m256 d[4];
            Winograd2x3SetOutput8(src, srcStride, d);
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

        void Winograd2x3SetOutput(const float * src, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            if (dstHeight < 2 || dstWidth < 16 || trans)
            {
                Sse::Winograd2x3SetOutput(src, dst, dstChannels, dstHeight, dstWidth, trans);
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
                        Winograd2x3SetOutput8Body(s + tileX, srcStride, d + col, dstWidth);
                    if (col < dstWidth)
                        Winograd2x3SetOutput8Edge(s + tileW - 8, srcStride, d + tailCol, dstWidth, true, false, tailMask);
                }
                if (row < dstHeight)
                {
                    size_t col = 0, tileX = 0;
                    const float * s = src + (tileH - 1) * tileW;
                    float * d = dst + (dstHeight - 1) * dstWidth;
                    for (; col < dstW16; col += 16, tileX += 8)
                        Winograd2x3SetOutput8Edge(s + tileX, srcStride, d + col, dstWidth, false, true, tailMask);
                    if (col < dstWidth)
                        Winograd2x3SetOutput8Edge(s + tileW - 8, srcStride, d + tailCol, dstWidth, false, false, tailMask);
                }
                src += tileW * tileH;
                dst += dstHeight * dstWidth;
            }
        }

        SIMD_INLINE void Winograd4x3SetFilter8Row(const __m256 * t, float * dst, size_t stride)
        {
            const __m256 r4 = _mm256_set1_ps(1.0f / 4.0f);
            const __m256 r6 = _mm256_set1_ps(1.0f / 6.0f);
            const __m256 mr6 = _mm256_set1_ps(-1.0f / 6.0f);
            const __m256 r12 = _mm256_set1_ps(1.0f / 12.0f);
            const __m256 r24 = _mm256_set1_ps(1.0f / 24.0f);
            _mm256_storeu_ps(dst + 0 * stride, _mm256_mul_ps(r4, t[0]));
            __m256 t0 = _mm256_add_ps(t[0], t[2]);
            _mm256_storeu_ps(dst + 1 * stride, _mm256_mul_ps(mr6, _mm256_add_ps(t0, t[1])));
            _mm256_storeu_ps(dst + 2 * stride, _mm256_mul_ps(mr6, _mm256_sub_ps(t0, t[1])));
            __m256 t1 = _mm256_add_ps(_mm256_mul_ps(r24, t[0]), _mm256_mul_ps(r6, t[2]));
            __m256 t2 = _mm256_mul_ps(r12, t[1]);
            _mm256_storeu_ps(dst + 3 * stride, _mm256_add_ps(t1, t2));
            _mm256_storeu_ps(dst + 4 * stride, _mm256_sub_ps(t1, t2));
            _mm256_storeu_ps(dst + 5 * stride, t[2]);
        }

        SIMD_INLINE void Winograd4x3SetFilter8All(const __m256 * s, float * dst, size_t stride)
        {
            const __m256 r4 = _mm256_set1_ps(1.0f / 4.0f);
            const __m256 r6 = _mm256_set1_ps(1.0f / 6.0f);
            const __m256 mr6 = _mm256_set1_ps(-1.0f / 6.0f);
            const __m256 r12 = _mm256_set1_ps(1.0f / 12.0f);
            const __m256 r24 = _mm256_set1_ps(1.0f / 24.0f);

            __m256 t[3];
            t[0] = _mm256_mul_ps(r4, s[0]);
            t[1] = _mm256_mul_ps(r4, s[1]);
            t[2] = _mm256_mul_ps(r4, s[2]);
            Winograd4x3SetFilter8Row(t, dst + 0 * stride, stride);

            t[0] = _mm256_mul_ps(mr6, _mm256_add_ps(_mm256_add_ps(s[0], s[3]), s[6]));
            t[1] = _mm256_mul_ps(mr6, _mm256_add_ps(_mm256_add_ps(s[1], s[4]), s[7]));
            t[2] = _mm256_mul_ps(mr6, _mm256_add_ps(_mm256_add_ps(s[2], s[5]), s[8]));
            Winograd4x3SetFilter8Row(t, dst + 6 * stride, stride);

            t[0] = _mm256_mul_ps(mr6, _mm256_add_ps(_mm256_sub_ps(s[0], s[3]), s[6]));
            t[1] = _mm256_mul_ps(mr6, _mm256_add_ps(_mm256_sub_ps(s[1], s[4]), s[7]));
            t[2] = _mm256_mul_ps(mr6, _mm256_add_ps(_mm256_sub_ps(s[2], s[5]), s[8]));
            Winograd4x3SetFilter8Row(t, dst + 12 * stride, stride);

            t[0] = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r24, s[0]), _mm256_mul_ps(r12, s[3])), _mm256_mul_ps(r6, s[6]));
            t[1] = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r24, s[1]), _mm256_mul_ps(r12, s[4])), _mm256_mul_ps(r6, s[7]));
            t[2] = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r24, s[2]), _mm256_mul_ps(r12, s[5])), _mm256_mul_ps(r6, s[8]));
            Winograd4x3SetFilter8Row(t, dst + 18 * stride, stride);

            t[0] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(r24, s[0]), _mm256_mul_ps(r12, s[3])), _mm256_mul_ps(r6, s[6]));
            t[1] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(r24, s[1]), _mm256_mul_ps(r12, s[4])), _mm256_mul_ps(r6, s[7]));
            t[2] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(r24, s[2]), _mm256_mul_ps(r12, s[5])), _mm256_mul_ps(r6, s[8]));
            Winograd4x3SetFilter8Row(t, dst + 24 * stride, stride);

            Winograd4x3SetFilter8Row(s + 6, dst + 30 * stride, stride);
        }

        SIMD_INLINE void Winograd4x3SetFilter8t(const float * src, float * dst, size_t stride)
        {
            __m256 s[9];
            s[0] = _mm256_loadu_ps(src + 0 * stride);
            s[1] = _mm256_loadu_ps(src + 1 * stride);
            s[2] = _mm256_loadu_ps(src + 2 * stride);
            s[3] = _mm256_loadu_ps(src + 3 * stride);
            s[4] = _mm256_loadu_ps(src + 4 * stride);
            s[5] = _mm256_loadu_ps(src + 5 * stride);
            s[6] = _mm256_loadu_ps(src + 6 * stride);
            s[7] = _mm256_loadu_ps(src + 7 * stride);
            s[8] = _mm256_loadu_ps(src + 8 * stride);
            Winograd4x3SetFilter8All(s, dst + 0 * stride, stride);
        }

        void Winograd4x3SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
        {
            if (trans)
            {
                size_t size8 = AlignLo(size, 8), i = 0;
                for (; i < size8; i += 8)
                    Winograd4x3SetFilter8t(src + i, dst + i, size);
                for (; i < size; i += 1)
                    Base::Winograd4x3SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                Sse::Winograd4x3SetFilter(src, size, dst, trans);
            }
        }
    }
#endif// SIMD_AVX_ENABLE
}
