/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdSse2.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_AVX_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx
    {
        SIMD_INLINE void WinogradKernel3x3Block2x2SetFilter8t(const float* src, float* dst, size_t stride)
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

        void WinogradKernel3x3Block2x2SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            if (trans)
            {
                size_t size8 = AlignLo(size, 8), i = 0;
                for (; i < size8; i += 8)
                    WinogradKernel3x3Block2x2SetFilter8t(src + i, dst + i, size);
                for (; i < size; i += 1)
                    Base::WinogradKernel3x3Block2x2SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                Sse2::WinogradKernel3x3Block2x2SetFilter(src, size, dst, trans);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInputLoad8n(const float * src, __m256 * dst)
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

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInputLoad8n(const float * src, __m256 * dst, PadType pad)
        {
            __m256 a0 = Set(pad == PadNose1 ? Sse2::LoadPadZeroNose1(src + 0) : _mm_loadu_ps(src + 0), _mm_loadu_ps(src + 8));
            __m256 a1 = Load<false>(src + 2, src + 10);
            __m256 a2 = Load<false>(src + 4, src + 12);
            __m256 a3 = Set(_mm_loadu_ps(src + 6), pad == PadTail2 ? Sse2::LoadPadZeroTail2(src + 14) : (pad == PadTail1 ? Sse2::LoadPadZeroTail1(src + 14) : _mm_loadu_ps(src + 14)));
            dst[0] = _mm256_shuffle_ps(a0, a2, 0x88);
            dst[1] = _mm256_shuffle_ps(a0, a2, 0xDD);
            dst[2] = _mm256_shuffle_ps(a1, a3, 0x88);
            dst[3] = _mm256_shuffle_ps(a1, a3, 0xDD);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInputLoad8z(__m256 * dst)
        {
            dst[0] = _mm256_setzero_ps();
            dst[1] = _mm256_setzero_ps();
            dst[2] = _mm256_setzero_ps();
            dst[3] = _mm256_setzero_ps();
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput8Store(const __m256 * src, float * dst, size_t stride)
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

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput8n(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m256 t[16];
            WinogradKernel3x3Block2x2SetInputLoad8n(src + 0 * srcStride, t + 0);
            WinogradKernel3x3Block2x2SetInputLoad8n(src + 1 * srcStride, t + 4);
            WinogradKernel3x3Block2x2SetInputLoad8n(src + 2 * srcStride, t + 8);
            WinogradKernel3x3Block2x2SetInputLoad8n(src + 3 * srcStride, t + 12);
            WinogradKernel3x3Block2x2SetInput8Store(t, dst, dstStride);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput8n(const float * src, size_t srcStride, PadType rowPad, PadType colPad, float * dst, size_t dstStride)
        {
            __m256 t[16];
            if (rowPad == PadNose1)
                WinogradKernel3x3Block2x2SetInputLoad8z(t + 0);
            else
                WinogradKernel3x3Block2x2SetInputLoad8n(src + 0 * srcStride, t + 0, colPad);
            WinogradKernel3x3Block2x2SetInputLoad8n(src + 1 * srcStride, t + 4, colPad);
            if (rowPad == PadTail2)
                WinogradKernel3x3Block2x2SetInputLoad8z(t + 8);
            else
                WinogradKernel3x3Block2x2SetInputLoad8n(src + 2 * srcStride, t + 8, colPad);
            if (rowPad >= PadTail1)
                WinogradKernel3x3Block2x2SetInputLoad8z(t + 12);
            else
                WinogradKernel3x3Block2x2SetInputLoad8n(src + 3 * srcStride, t + 12, colPad);
            WinogradKernel3x3Block2x2SetInput8Store(t, dst, dstStride);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput8t(const float * src, size_t srcS, size_t srcC, __m256 dst[16])
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

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput8t(const float * src, size_t srcW, size_t srcC, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
            {
                __m256 tmp[16];
                WinogradKernel3x3Block2x2SetInput8t(src + c, srcS, srcC, tmp);
                WinogradKernel3x3Block2x2SetInput8Store(tmp, dst + c, dstStride);
            }
            if (srcCF < srcC)
            {
                __m256 tmp[16];
                WinogradKernel3x3Block2x2SetInput8t(src + srcC - F, srcS, srcC, tmp);
                WinogradKernel3x3Block2x2SetInput8Store(tmp, dst + srcC - F, dstStride);
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput8t(const float * src, size_t srcS, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, __m256 dst[16])
        {
            for (size_t row = 0; row < rowB; ++row)
            {
                for (size_t col = 0; col < 4; ++col)
                    dst[col] = _mm256_setzero_ps();
                dst += 4;
            }
            for (size_t row = rowB; row < rowE; ++row)
            {
                for (size_t col = 0; col < colB; ++col)
                    dst[col] = _mm256_setzero_ps();
                for (size_t col = colB; col < colE; ++col)
                    dst[col] = _mm256_loadu_ps(src + row * srcS + col * srcC);
                for (size_t col = colE; col < 4; ++col)
                    dst[col] = _mm256_setzero_ps();
                dst += 4;
            }
            for (size_t row = rowE; row < 4; ++row)
            {
                for (size_t col = 0; col < 4; ++col)
                    dst[col] = _mm256_setzero_ps();
                dst += 4;
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput8t(const float * src, size_t srcW, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
            {
                __m256 tmp[16];
                WinogradKernel3x3Block2x2SetInput8t(src + c, srcS, srcC, rowB, rowE, colB, colE, tmp);
                WinogradKernel3x3Block2x2SetInput8Store(tmp, dst + c, dstStride);
            }
            if (srcCF < srcC)
            {
                __m256 tmp[16];
                WinogradKernel3x3Block2x2SetInput8t(src + srcC - F, srcS, srcC, rowB, rowE, colB, colE, tmp);
                WinogradKernel3x3Block2x2SetInput8Store(tmp, dst + srcC - F, dstStride);
            }
        }

        void WinogradKernel3x3Block2x2SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padY == padX && padY == padH && padY == padW && (padY == 0 || padY == 1));
            SimdBool pad = padY > 0 ? SimdTrue : SimdFalse;
            if (trans ? (srcChannels < 8) : (srcHeight < 4 || srcWidth < 18))
            {
                Sse2::WinogradKernel3x3Block2x2SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
                return;
            }
            size_t dstH = pad ? srcHeight : srcHeight - 2;
            size_t dstW = pad ? srcWidth : srcWidth - 2;
            size_t tileH = (dstH + 1) / 2;
            size_t tileW = (dstW + 1) / 2;
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
                        WinogradKernel3x3Block2x2SetInput8t(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstW2; col += 2)
                        WinogradKernel3x3Block2x2SetInput8t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 4, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        WinogradKernel3x3Block2x2SetInput8t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                for (row = start; row < dstH2; row += 2)
                {
                    if (pad)
                        WinogradKernel3x3Block2x2SetInput8t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 4, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstW2; col += 2)
                        WinogradKernel3x3Block2x2SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        WinogradKernel3x3Block2x2SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 4, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                if (row < dstH)
                {
                    if (pad)
                        WinogradKernel3x3Block2x2SetInput8t(src + row * srcWidth* srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstW2; col += 2)
                        WinogradKernel3x3Block2x2SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 4, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        WinogradKernel3x3Block2x2SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
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
                            WinogradKernel3x3Block2x2SetInput8n(s + col, srcWidth, PadNose1, PadNose1, d + tileX, dstStride), col += 16, tileX += 8;
                        for (; col < dstW16; col += 16, tileX += 8)
                            WinogradKernel3x3Block2x2SetInput8n(s + col, srcWidth, PadNose1, PadNone, d + tileX, dstStride);
                        if (specialColTail)
                            WinogradKernel3x3Block2x2SetInput8n(s + tailCol, srcWidth, PadNose1, colPad, d + tileW - 8, dstStride);
                        row += 2, tileY += 1;
                    }
                    for (; row < dstH2; row += 2, tileY += 1)
                    {
                        size_t col = 0, tileX = 0;
                        const float * s = src + row * srcWidth;
                        float * d = dst + tileY * tileW;
                        if (pad)
                            WinogradKernel3x3Block2x2SetInput8n(s + col, srcWidth, PadNone, PadNose1, d + tileX, dstStride), col += 16, tileX += 8;
                        for (; col < dstW16; col += 16, tileX += 8)
                            WinogradKernel3x3Block2x2SetInput8n(s + col, srcWidth, d + tileX, dstStride);
                        if (specialColTail)
                            WinogradKernel3x3Block2x2SetInput8n(s + tailCol, srcWidth, PadNone, colPad, d + tileW - 8, dstStride);
                    }
                    if (specialRowTail)
                    {
                        size_t col = 0, tileX = 0;
                        const float * s = src + tailRow * srcWidth;
                        float * d = dst + (tileH - 1) * tileW;
                        if (pad)
                            WinogradKernel3x3Block2x2SetInput8n(s + col, srcWidth, rowPad, PadNose1, d + tileX, dstStride), col += 16, tileX += 8;
                        for (; col < dstW16; col += 16, tileX += 8)
                            WinogradKernel3x3Block2x2SetInput8n(s + col, srcWidth, rowPad, PadNone, d + tileX, dstStride);
                        if (specialColTail)
                            WinogradKernel3x3Block2x2SetInput8n(s + tailCol, srcWidth, rowPad, colPad, d + tileW - 8, dstStride);
                    }
                    src += srcWidth * srcHeight;
                    dst += tileW * tileH;
                }
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutputLoad4(const float * src, size_t stride, __m256 * dst)
        {
            __m256 s0 = _mm256_loadu_ps(src + 0 * stride);
            __m256 s1 = _mm256_loadu_ps(src + 1 * stride);
            __m256 s2 = _mm256_loadu_ps(src + 2 * stride);
            __m256 s3 = _mm256_loadu_ps(src + 3 * stride);
            dst[0] = _mm256_add_ps(_mm256_add_ps(s0, s1), s2);
            dst[1] = _mm256_sub_ps(_mm256_sub_ps(s1, s2), s3);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutputLoad16(const float * src, size_t stride, __m256 * dst)
        {
            __m256 tmp[8];
            WinogradKernel3x3Block2x2SetOutputLoad4(src + 0 * stride, stride, tmp + 0);
            WinogradKernel3x3Block2x2SetOutputLoad4(src + 4 * stride, stride, tmp + 2);
            WinogradKernel3x3Block2x2SetOutputLoad4(src + 8 * stride, stride, tmp + 4);
            WinogradKernel3x3Block2x2SetOutputLoad4(src + 12 * stride, stride, tmp + 6);
            dst[0] = _mm256_add_ps(_mm256_add_ps(tmp[0], tmp[2]), tmp[4]);
            dst[1] = _mm256_add_ps(_mm256_add_ps(tmp[1], tmp[3]), tmp[5]);
            dst[2] = _mm256_sub_ps(_mm256_sub_ps(tmp[2], tmp[4]), tmp[6]);
            dst[3] = _mm256_sub_ps(_mm256_sub_ps(tmp[3], tmp[5]), tmp[7]);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutput8n(const float * src, size_t stride, __m256 * dst)
        {
            __m256 d[4], u[4];
            WinogradKernel3x3Block2x2SetOutputLoad16(src, stride, d);
            u[0] = _mm256_unpacklo_ps(d[0], d[1]);
            u[1] = _mm256_unpackhi_ps(d[0], d[1]);
            u[2] = _mm256_unpacklo_ps(d[2], d[3]);
            u[3] = _mm256_unpackhi_ps(d[2], d[3]);
            dst[0] = _mm256_permute2f128_ps(u[0], u[1], 0x20);
            dst[1] = _mm256_permute2f128_ps(u[0], u[1], 0x31);
            dst[2] = _mm256_permute2f128_ps(u[2], u[3], 0x20);
            dst[3] = _mm256_permute2f128_ps(u[2], u[3], 0x31);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutput8n(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m256 d[4];
            WinogradKernel3x3Block2x2SetOutput8n(src, srcStride, d);
            _mm256_storeu_ps(dst + 0 * dstStride + 0, d[0]);
            _mm256_storeu_ps(dst + 0 * dstStride + 8, d[1]);
            _mm256_storeu_ps(dst + 1 * dstStride + 0, d[2]);
            _mm256_storeu_ps(dst + 1 * dstStride + 8, d[3]);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutput8n(const float * src, size_t srcStride, float * dst, size_t dstStride, bool lastRow, bool lastCol, const __m256 & mask)
        {
            __m256 d[4];
            WinogradKernel3x3Block2x2SetOutput8n(src, srcStride, d);
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
                    _mm256_storeu_ps(dst + 8, d[3]);
                else
                    StoreMasked<false>(dst + 8, d[3], mask);
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutputStore4(const __m256 src[4], float * dst, size_t dstS, size_t dstC)
        {
            _mm256_storeu_ps(dst + 0 * dstS + 0 * dstC, src[0]);
            _mm256_storeu_ps(dst + 0 * dstS + 1 * dstC, src[1]);
            _mm256_storeu_ps(dst + 1 * dstS + 0 * dstC, src[2]);
            _mm256_storeu_ps(dst + 1 * dstS + 1 * dstC, src[3]);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutput8t(const float * src, size_t srcStride, float * dst, size_t dstW, size_t dstC)
        {
            size_t dstS = dstW * dstC, dstCF = AlignLo(dstC, F);
            for (size_t d = 0; d < dstCF; d += F)
            {
                __m256 tmp[4];
                WinogradKernel3x3Block2x2SetOutputLoad16(src + d, srcStride, tmp);
                WinogradKernel3x3Block2x2SetOutputStore4(tmp, dst + d, dstS, dstC);
            }
            if (dstCF < dstC)
            {
                __m256 tmp[4];
                WinogradKernel3x3Block2x2SetOutputLoad16(src + dstC - F, srcStride, tmp);
                WinogradKernel3x3Block2x2SetOutputStore4(tmp, dst + dstC - F, dstS, dstC);
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutputStore4(const __m256 src[4], float * dst, size_t dstS, size_t dstC, size_t rowE, size_t colE)
        {
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    _mm256_storeu_ps(dst + row * dstS + col * dstC, src[row * 2 + col]);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutput8t(const float * src, size_t srcStride, float * dst, size_t dstW, size_t dstC, size_t rowE, size_t colE)
        {
            size_t dstS = dstW * dstC, dstCF = AlignLo(dstC, F);
            for (size_t d = 0; d < dstCF; d += F)
            {
                __m256 tmp[4];
                WinogradKernel3x3Block2x2SetOutputLoad16(src + d, srcStride, tmp);
                WinogradKernel3x3Block2x2SetOutputStore4(tmp, dst + d, dstS, dstC, rowE, colE);
            }
            if (dstCF < dstC)
            {
                __m256 tmp[4];
                WinogradKernel3x3Block2x2SetOutputLoad16(src + dstC - F, srcStride, tmp);
                WinogradKernel3x3Block2x2SetOutputStore4(tmp, dst + dstC - F, dstS, dstC, rowE, colE);
            }
        }

        void WinogradKernel3x3Block2x2SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            if (trans ? (dstChannels < 8) : (dstHeight < 2 || dstWidth < 16))
            {
                Sse2::WinogradKernel3x3Block2x2SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
                return;
            }
            size_t tileH = (dstHeight + 1) / 2;
            size_t tileW = (dstWidth + 1) / 2;
            size_t dstH2 = AlignLo(dstHeight, 2);
            size_t dstW2 = AlignLo(dstWidth, 2);
            if (trans)
            {
                size_t row, col;
                for (row = 0; row < dstH2; row += 2)
                {
                    for (col = 0; col < dstW2; col += 2)
                        WinogradKernel3x3Block2x2SetOutput8t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block2x2SetOutput8t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, 2, dstWidth - col), src += dstChannels;
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstW2; col += 2)
                        WinogradKernel3x3Block2x2SetOutput8t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, dstHeight - row, 2), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block2x2SetOutput8t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, dstHeight - row, dstWidth - col), src += dstChannels;
                }
            }
            else
            {
                size_t dstW16 = AlignLo(dstWidth, 16);
                __m256 tailMask = LeftNotZero32f(8 + dstW2 - dstWidth);
                size_t tailCol = dstW2 < dstWidth ? dstWidth - 15 : dstWidth - 16;
                for (size_t c = 0; c < dstChannels; ++c)
                {
                    size_t row = 0, tileY = 0;
                    for (; row < dstH2; row += 2, tileY += 1)
                    {
                        size_t col = 0, tileX = 0;
                        const float * s = src + tileY * tileW;
                        float * d = dst + row * dstWidth;
                        for (; col < dstW16; col += 16, tileX += 8)
                            WinogradKernel3x3Block2x2SetOutput8n(s + tileX, srcStride, d + col, dstWidth);
                        if (col < dstWidth)
                            WinogradKernel3x3Block2x2SetOutput8n(s + tileW - 8, srcStride, d + tailCol, dstWidth, true, false, tailMask);
                    }
                    if (row < dstHeight)
                    {
                        size_t col = 0, tileX = 0;
                        const float * s = src + (tileH - 1) * tileW;
                        float * d = dst + (dstHeight - 1) * dstWidth;
                        for (; col < dstW16; col += 16, tileX += 8)
                            WinogradKernel3x3Block2x2SetOutput8n(s + tileX, srcStride, d + col, dstWidth, false, true, tailMask);
                        if (col < dstWidth)
                            WinogradKernel3x3Block2x2SetOutput8n(s + tileW - 8, srcStride, d + tailCol, dstWidth, false, false, tailMask);
                    }
                    src += tileW * tileH;
                    dst += dstHeight * dstWidth;
                }
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block3x3SetFilter8Row(const __m256 * t, float * dst, size_t stride)
        {
            const __m256 r6 = _mm256_set1_ps(1.0f / 6.0f);
            const __m256 r3 = _mm256_set1_ps(1.0f / 3.0f);
            const __m256 r2 = _mm256_set1_ps(1.0f / 2.0f);
            const __m256 f2_3 = _mm256_set1_ps(2.0f / 3.0f);
            const __m256 mr2 = _mm256_set1_ps(-1.0f / 2.0f);

            _mm256_storeu_ps(dst + 0 * stride, _mm256_mul_ps(r2, t[0]));
            __m256 t0 = _mm256_add_ps(t[0], t[2]);
            _mm256_storeu_ps(dst + 1 * stride, _mm256_mul_ps(mr2, _mm256_add_ps(t0, t[1])));
            _mm256_storeu_ps(dst + 2 * stride, _mm256_mul_ps(r6, _mm256_sub_ps(t[1], t0)));
            _mm256_storeu_ps(dst + 3 * stride, _mm256_add_ps(_mm256_mul_ps(r6, t[0]), _mm256_add_ps(_mm256_mul_ps(r3, t[1]), _mm256_mul_ps(f2_3, t[2]))));
            _mm256_storeu_ps(dst + 4 * stride, t[2]);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetFilter8All(const __m256 * s, float * dst, size_t stride)
        {
            const __m256 r6 = _mm256_set1_ps(1.0f / 6.0f);
            const __m256 r3 = _mm256_set1_ps(1.0f / 3.0f);
            const __m256 r2 = _mm256_set1_ps(1.0f / 2.0f);
            const __m256 f2_3 = _mm256_set1_ps(2.0f / 3.0f);
            const __m256 mr2 = _mm256_set1_ps(-1.0f / 2.0f);

            __m256 t[3];
            t[0] = _mm256_mul_ps(r2, s[0]);
            t[1] = _mm256_mul_ps(r2, s[1]);
            t[2] = _mm256_mul_ps(r2, s[2]);
            WinogradKernel3x3Block3x3SetFilter8Row(t, dst + 0 * stride, stride);

            t[0] = _mm256_mul_ps(mr2, _mm256_add_ps(_mm256_add_ps(s[0], s[6]), s[3]));
            t[1] = _mm256_mul_ps(mr2, _mm256_add_ps(_mm256_add_ps(s[1], s[7]), s[4]));
            t[2] = _mm256_mul_ps(mr2, _mm256_add_ps(_mm256_add_ps(s[2], s[8]), s[5]));
            WinogradKernel3x3Block3x3SetFilter8Row(t, dst + 5 * stride, stride);

            t[0] = _mm256_mul_ps(r6, _mm256_sub_ps(s[3], _mm256_add_ps(s[0], s[6])));
            t[1] = _mm256_mul_ps(r6, _mm256_sub_ps(s[4], _mm256_add_ps(s[1], s[7])));
            t[2] = _mm256_mul_ps(r6, _mm256_sub_ps(s[5], _mm256_add_ps(s[2], s[8])));
            WinogradKernel3x3Block3x3SetFilter8Row(t, dst + 10 * stride, stride);

            t[0] = _mm256_add_ps(_mm256_mul_ps(r6, s[0]), _mm256_add_ps(_mm256_mul_ps(r3, s[3]), _mm256_mul_ps(f2_3, s[6])));
            t[1] = _mm256_add_ps(_mm256_mul_ps(r6, s[1]), _mm256_add_ps(_mm256_mul_ps(r3, s[4]), _mm256_mul_ps(f2_3, s[7])));
            t[2] = _mm256_add_ps(_mm256_mul_ps(r6, s[2]), _mm256_add_ps(_mm256_mul_ps(r3, s[5]), _mm256_mul_ps(f2_3, s[8])));
            WinogradKernel3x3Block3x3SetFilter8Row(t, dst + 15 * stride, stride);

            WinogradKernel3x3Block3x3SetFilter8Row(s + 6, dst + 20 * stride, stride);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetFilter8t(const float * src, float * dst, size_t stride)
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
            WinogradKernel3x3Block3x3SetFilter8All(s, dst + 0 * stride, stride);
        }

        void WinogradKernel3x3Block3x3SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
        {
            size_t size8 = AlignLo(size, 8), i = 0;
            if (trans)
            {
                for (; i < size8; i += 8)
                    WinogradKernel3x3Block3x3SetFilter8t(src + i, dst + i, size);
                for (; i < size; i += 1)
                    Base::WinogradKernel3x3Block3x3SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                Sse2::WinogradKernel3x3Block3x3SetFilter(src, size, dst, trans);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInput8Store(const __m256 src[25], float * dst, size_t stride)
        {
            __m256 _2 = _mm256_set1_ps(2.0f);
            __m256 _3 = _mm256_set1_ps(3.0f);
            __m256 tmp[5];

            tmp[0] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[0], src[10])), _mm256_sub_ps(src[15], src[5]));
            tmp[1] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[1], src[11])), _mm256_sub_ps(src[16], src[6]));
            tmp[2] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[2], src[12])), _mm256_sub_ps(src[17], src[7]));
            tmp[3] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[3], src[13])), _mm256_sub_ps(src[18], src[8]));
            tmp[4] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[4], src[14])), _mm256_sub_ps(src[19], src[9]));
            _mm256_storeu_ps(dst + 0 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[0], tmp[2])), _mm256_sub_ps(tmp[3], tmp[1])));
            _mm256_storeu_ps(dst + 1 * stride, _mm256_sub_ps(_mm256_sub_ps(tmp[3], tmp[2]), _mm256_mul_ps(_2, tmp[1])));
            _mm256_storeu_ps(dst + 2 * stride, _mm256_add_ps(_mm256_mul_ps(_2, tmp[1]), _mm256_sub_ps(tmp[3], _mm256_mul_ps(_3, tmp[2]))));
            _mm256_storeu_ps(dst + 3 * stride, _mm256_sub_ps(tmp[3], tmp[1]));
            _mm256_storeu_ps(dst + 4 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[1], tmp[3])), _mm256_sub_ps(tmp[4], tmp[2])));

            tmp[0] = _mm256_sub_ps(_mm256_sub_ps(src[15], src[10]), _mm256_mul_ps(_2, src[5]));
            tmp[1] = _mm256_sub_ps(_mm256_sub_ps(src[16], src[11]), _mm256_mul_ps(_2, src[6]));
            tmp[2] = _mm256_sub_ps(_mm256_sub_ps(src[17], src[12]), _mm256_mul_ps(_2, src[7]));
            tmp[3] = _mm256_sub_ps(_mm256_sub_ps(src[18], src[13]), _mm256_mul_ps(_2, src[8]));
            tmp[4] = _mm256_sub_ps(_mm256_sub_ps(src[19], src[14]), _mm256_mul_ps(_2, src[9]));
            _mm256_storeu_ps(dst + 5 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[0], tmp[2])), _mm256_sub_ps(tmp[3], tmp[1])));
            _mm256_storeu_ps(dst + 6 * stride, _mm256_sub_ps(_mm256_sub_ps(tmp[3], tmp[2]), _mm256_mul_ps(_2, tmp[1])));
            _mm256_storeu_ps(dst + 7 * stride, _mm256_add_ps(_mm256_mul_ps(_2, tmp[1]), _mm256_sub_ps(tmp[3], _mm256_mul_ps(_3, tmp[2]))));
            _mm256_storeu_ps(dst + 8 * stride, _mm256_sub_ps(tmp[3], tmp[1]));
            _mm256_storeu_ps(dst + 9 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[1], tmp[3])), _mm256_sub_ps(tmp[4], tmp[2])));

            tmp[0] = _mm256_add_ps(_mm256_mul_ps(_2, src[5]), _mm256_sub_ps(src[15], _mm256_mul_ps(_3, src[10])));
            tmp[1] = _mm256_add_ps(_mm256_mul_ps(_2, src[6]), _mm256_sub_ps(src[16], _mm256_mul_ps(_3, src[11])));
            tmp[2] = _mm256_add_ps(_mm256_mul_ps(_2, src[7]), _mm256_sub_ps(src[17], _mm256_mul_ps(_3, src[12])));
            tmp[3] = _mm256_add_ps(_mm256_mul_ps(_2, src[8]), _mm256_sub_ps(src[18], _mm256_mul_ps(_3, src[13])));
            tmp[4] = _mm256_add_ps(_mm256_mul_ps(_2, src[9]), _mm256_sub_ps(src[19], _mm256_mul_ps(_3, src[14])));
            _mm256_storeu_ps(dst + 10 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[0], tmp[2])), _mm256_sub_ps(tmp[3], tmp[1])));
            _mm256_storeu_ps(dst + 11 * stride, _mm256_sub_ps(_mm256_sub_ps(tmp[3], tmp[2]), _mm256_mul_ps(_2, tmp[1])));
            _mm256_storeu_ps(dst + 12 * stride, _mm256_add_ps(_mm256_mul_ps(_2, tmp[1]), _mm256_sub_ps(tmp[3], _mm256_mul_ps(_3, tmp[2]))));
            _mm256_storeu_ps(dst + 13 * stride, _mm256_sub_ps(tmp[3], tmp[1]));
            _mm256_storeu_ps(dst + 14 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[1], tmp[3])), _mm256_sub_ps(tmp[4], tmp[2])));

            tmp[0] = _mm256_sub_ps(src[15], src[5]);
            tmp[1] = _mm256_sub_ps(src[16], src[6]);
            tmp[2] = _mm256_sub_ps(src[17], src[7]);
            tmp[3] = _mm256_sub_ps(src[18], src[8]);
            tmp[4] = _mm256_sub_ps(src[19], src[9]);
            _mm256_storeu_ps(dst + 15 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[0], tmp[2])), _mm256_sub_ps(tmp[3], tmp[1])));
            _mm256_storeu_ps(dst + 16 * stride, _mm256_sub_ps(_mm256_sub_ps(tmp[3], tmp[2]), _mm256_mul_ps(_2, tmp[1])));
            _mm256_storeu_ps(dst + 17 * stride, _mm256_add_ps(_mm256_mul_ps(_2, tmp[1]), _mm256_sub_ps(tmp[3], _mm256_mul_ps(_3, tmp[2]))));
            _mm256_storeu_ps(dst + 18 * stride, _mm256_sub_ps(tmp[3], tmp[1]));
            _mm256_storeu_ps(dst + 19 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[1], tmp[3])), _mm256_sub_ps(tmp[4], tmp[2])));

            tmp[0] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[5], src[15])), _mm256_sub_ps(src[20], src[10]));
            tmp[1] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[6], src[16])), _mm256_sub_ps(src[21], src[11]));
            tmp[2] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[7], src[17])), _mm256_sub_ps(src[22], src[12]));
            tmp[3] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[8], src[18])), _mm256_sub_ps(src[23], src[13]));
            tmp[4] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[9], src[19])), _mm256_sub_ps(src[24], src[14]));
            _mm256_storeu_ps(dst + 20 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[0], tmp[2])), _mm256_sub_ps(tmp[3], tmp[1])));
            _mm256_storeu_ps(dst + 21 * stride, _mm256_sub_ps(_mm256_sub_ps(tmp[3], tmp[2]), _mm256_mul_ps(_2, tmp[1])));
            _mm256_storeu_ps(dst + 22 * stride, _mm256_add_ps(_mm256_mul_ps(_2, tmp[1]), _mm256_sub_ps(tmp[3], _mm256_mul_ps(_3, tmp[2]))));
            _mm256_storeu_ps(dst + 23 * stride, _mm256_sub_ps(tmp[3], tmp[1]));
            _mm256_storeu_ps(dst + 24 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[1], tmp[3])), _mm256_sub_ps(tmp[4], tmp[2])));
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInput8t(const float * src, size_t srcS, size_t srcC, __m256 dst[25])
        {
            dst[0] = _mm256_loadu_ps(src + 0 * srcS + 0 * srcC);
            dst[1] = _mm256_loadu_ps(src + 0 * srcS + 1 * srcC);
            dst[2] = _mm256_loadu_ps(src + 0 * srcS + 2 * srcC);
            dst[3] = _mm256_loadu_ps(src + 0 * srcS + 3 * srcC);
            dst[4] = _mm256_loadu_ps(src + 0 * srcS + 4 * srcC);
            dst[5] = _mm256_loadu_ps(src + 1 * srcS + 0 * srcC);
            dst[6] = _mm256_loadu_ps(src + 1 * srcS + 1 * srcC);
            dst[7] = _mm256_loadu_ps(src + 1 * srcS + 2 * srcC);
            dst[8] = _mm256_loadu_ps(src + 1 * srcS + 3 * srcC);
            dst[9] = _mm256_loadu_ps(src + 1 * srcS + 4 * srcC);
            dst[10] = _mm256_loadu_ps(src + 2 * srcS + 0 * srcC);
            dst[11] = _mm256_loadu_ps(src + 2 * srcS + 1 * srcC);
            dst[12] = _mm256_loadu_ps(src + 2 * srcS + 2 * srcC);
            dst[13] = _mm256_loadu_ps(src + 2 * srcS + 3 * srcC);
            dst[14] = _mm256_loadu_ps(src + 2 * srcS + 4 * srcC);
            dst[15] = _mm256_loadu_ps(src + 3 * srcS + 0 * srcC);
            dst[16] = _mm256_loadu_ps(src + 3 * srcS + 1 * srcC);
            dst[17] = _mm256_loadu_ps(src + 3 * srcS + 2 * srcC);
            dst[18] = _mm256_loadu_ps(src + 3 * srcS + 3 * srcC);
            dst[19] = _mm256_loadu_ps(src + 3 * srcS + 4 * srcC);
            dst[20] = _mm256_loadu_ps(src + 4 * srcS + 0 * srcC);
            dst[21] = _mm256_loadu_ps(src + 4 * srcS + 1 * srcC);
            dst[22] = _mm256_loadu_ps(src + 4 * srcS + 2 * srcC);
            dst[23] = _mm256_loadu_ps(src + 4 * srcS + 3 * srcC);
            dst[24] = _mm256_loadu_ps(src + 4 * srcS + 4 * srcC);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInput8t(const float * src, size_t srcW, size_t srcC, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
            {
                __m256 tmp[25];
                WinogradKernel3x3Block3x3SetInput8t(src + c, srcS, srcC, tmp);
                WinogradKernel3x3Block3x3SetInput8Store(tmp, dst + c, dstStride);
            }
            if (srcCF < srcC)
            {
                __m256 tmp[25];
                WinogradKernel3x3Block3x3SetInput8t(src + srcC - F, srcS, srcC, tmp);
                WinogradKernel3x3Block3x3SetInput8Store(tmp, dst + srcC - F, dstStride);
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInput8t(const float * src, size_t srcS, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, __m256 dst[25])
        {
            for (size_t row = 0; row < rowB; ++row)
            {
                for (size_t col = 0; col < 5; ++col)
                    dst[col] = _mm256_setzero_ps();
                dst += 5;
            }
            for (size_t row = rowB; row < rowE; ++row)
            {
                for (size_t col = 0; col < colB; ++col)
                    dst[col] = _mm256_setzero_ps();
                for (size_t col = colB; col < colE; ++col)
                    dst[col] = _mm256_loadu_ps(src + row * srcS + col * srcC);
                for (size_t col = colE; col < 5; ++col)
                    dst[col] = _mm256_setzero_ps();
                dst += 5;
            }
            for (size_t row = rowE; row < 5; ++row)
            {
                for (size_t col = 0; col < 5; ++col)
                    dst[col] = _mm256_setzero_ps();
                dst += 5;
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInput8t(const float * src, size_t srcW, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
            {
                __m256 tmp[25];
                WinogradKernel3x3Block3x3SetInput8t(src + c, srcS, srcC, rowB, rowE, colB, colE, tmp);
                WinogradKernel3x3Block3x3SetInput8Store(tmp, dst + c, dstStride);
            }
            if (srcCF < srcC)
            {
                __m256 tmp[25];
                WinogradKernel3x3Block3x3SetInput8t(src + srcC - F, srcS, srcC, rowB, rowE, colB, colE, tmp);
                WinogradKernel3x3Block3x3SetInput8Store(tmp, dst + srcC - F, dstStride);
            }
        }

        void WinogradKernel3x3Block3x3SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padY == padX && padY == padH && padY == padW && (padY == 0 || padY == 1));
            SimdBool pad = padY > 0 ? SimdTrue : SimdFalse;
            if (trans ? (srcChannels < 8) : (srcHeight < 5 || srcWidth < 15))
            {
                Sse2::WinogradKernel3x3Block3x3SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
                return;
            }
            size_t dstH = pad ? srcHeight : srcHeight - 2;
            size_t dstW = pad ? srcWidth : srcWidth - 2;
            size_t tileH = (dstH + 2) / 3;
            size_t tileW = (dstW + 2) / 3;
            size_t dstH3 = AlignLoAny(dstH, 3);
            size_t dstW3 = AlignLoAny(dstW, 3);
            if (trans)
            {
                size_t noseW = Simd::Min<size_t>(5, dstW + 1);
                size_t noseH = Simd::Min<size_t>(5, dstH + 1);
                size_t start = pad ? 3 : 0;
                if (pad)
                {
                    if (dstH == dstH3)
                        dstH3 -= 3;
                    if (dstW == dstW3)
                        dstW3 -= 3;
                    src -= (srcWidth + 1)*srcChannels;
                }
                size_t tailW = dstW - dstW3 + (pad ? 1 : 2);
                size_t tailH = dstH - dstH3 + (pad ? 1 : 2);
                size_t row = 0, col = 0;
                if (pad)
                {
                    if (pad)
                        WinogradKernel3x3Block3x3SetInput8t(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstW3; col += 3)
                        WinogradKernel3x3Block3x3SetInput8t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 5, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        WinogradKernel3x3Block3x3SetInput8t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                for (row = start; row < dstH3; row += 3)
                {
                    if (pad)
                        WinogradKernel3x3Block3x3SetInput8t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 5, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstW3; col += 3)
                        WinogradKernel3x3Block3x3SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        WinogradKernel3x3Block3x3SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 5, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                if (row < dstH)
                {
                    if (pad)
                        WinogradKernel3x3Block3x3SetInput8t(src + row * srcWidth* srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstW3; col += 3)
                        WinogradKernel3x3Block3x3SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 5, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        WinogradKernel3x3Block3x3SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
            }
            else
            {
                Base::WinogradKernel3x3Block3x3SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block3x3SetOutputLoad25(const float * src, size_t stride, __m256 dst[9])
        {
            __m256 s[25];
            s[0] = _mm256_loadu_ps(src + 0 * stride);
            s[1] = _mm256_loadu_ps(src + 1 * stride);
            s[2] = _mm256_loadu_ps(src + 2 * stride);
            s[3] = _mm256_loadu_ps(src + 3 * stride);
            s[4] = _mm256_loadu_ps(src + 4 * stride);
            s[5] = _mm256_loadu_ps(src + 5 * stride);
            s[6] = _mm256_loadu_ps(src + 6 * stride);
            s[7] = _mm256_loadu_ps(src + 7 * stride);
            s[8] = _mm256_loadu_ps(src + 8 * stride);
            s[9] = _mm256_loadu_ps(src + 9 * stride);
            s[10] = _mm256_loadu_ps(src + 10 * stride);
            s[11] = _mm256_loadu_ps(src + 11 * stride);
            s[12] = _mm256_loadu_ps(src + 12 * stride);
            s[13] = _mm256_loadu_ps(src + 13 * stride);
            s[14] = _mm256_loadu_ps(src + 14 * stride);
            s[15] = _mm256_loadu_ps(src + 15 * stride);
            s[16] = _mm256_loadu_ps(src + 16 * stride);
            s[17] = _mm256_loadu_ps(src + 17 * stride);
            s[18] = _mm256_loadu_ps(src + 18 * stride);
            s[19] = _mm256_loadu_ps(src + 19 * stride);
            s[20] = _mm256_loadu_ps(src + 20 * stride);
            s[21] = _mm256_loadu_ps(src + 21 * stride);
            s[22] = _mm256_loadu_ps(src + 22 * stride);
            s[23] = _mm256_loadu_ps(src + 23 * stride);
            s[24] = _mm256_loadu_ps(src + 24 * stride);

            __m256 _2 = _mm256_set1_ps(2.0f);
            __m256 _4 = _mm256_set1_ps(4.0f);
            __m256 t[5];
            t[0] = _mm256_add_ps(_mm256_add_ps(s[0], s[5]), _mm256_add_ps(s[10], s[15]));
            t[1] = _mm256_add_ps(_mm256_add_ps(s[1], s[6]), _mm256_add_ps(s[11], s[16]));
            t[2] = _mm256_add_ps(_mm256_add_ps(s[2], s[7]), _mm256_add_ps(s[12], s[17]));
            t[3] = _mm256_add_ps(_mm256_add_ps(s[3], s[8]), _mm256_add_ps(s[13], s[18]));
            t[4] = _mm256_add_ps(_mm256_add_ps(s[4], s[9]), _mm256_add_ps(s[14], s[19]));
            dst[0] = _mm256_add_ps(_mm256_add_ps(t[0], t[1]), _mm256_add_ps(t[2], t[3]));
            dst[1] = _mm256_add_ps(_mm256_sub_ps(t[1], t[2]), _mm256_mul_ps(_2, t[3]));
            dst[2] = _mm256_add_ps(_mm256_add_ps(t[1], t[2]), _mm256_add_ps(_mm256_mul_ps(_4, t[3]), t[4]));

            t[0] = _mm256_add_ps(_mm256_sub_ps(s[5], s[10]), _mm256_mul_ps(_2, s[15]));
            t[1] = _mm256_add_ps(_mm256_sub_ps(s[6], s[11]), _mm256_mul_ps(_2, s[16]));
            t[2] = _mm256_add_ps(_mm256_sub_ps(s[7], s[12]), _mm256_mul_ps(_2, s[17]));
            t[3] = _mm256_add_ps(_mm256_sub_ps(s[8], s[13]), _mm256_mul_ps(_2, s[18]));
            t[4] = _mm256_add_ps(_mm256_sub_ps(s[9], s[14]), _mm256_mul_ps(_2, s[19]));
            dst[3] = _mm256_add_ps(_mm256_add_ps(t[0], t[1]), _mm256_add_ps(t[2], t[3]));
            dst[4] = _mm256_add_ps(_mm256_sub_ps(t[1], t[2]), _mm256_mul_ps(_2, t[3]));
            dst[5] = _mm256_add_ps(_mm256_add_ps(t[1], t[2]), _mm256_add_ps(_mm256_mul_ps(_4, t[3]), t[4]));

            t[0] = _mm256_add_ps(_mm256_add_ps(s[5], s[10]), _mm256_add_ps(_mm256_mul_ps(_4, s[15]), s[20]));
            t[1] = _mm256_add_ps(_mm256_add_ps(s[6], s[11]), _mm256_add_ps(_mm256_mul_ps(_4, s[16]), s[21]));
            t[2] = _mm256_add_ps(_mm256_add_ps(s[7], s[12]), _mm256_add_ps(_mm256_mul_ps(_4, s[17]), s[22]));
            t[3] = _mm256_add_ps(_mm256_add_ps(s[8], s[13]), _mm256_add_ps(_mm256_mul_ps(_4, s[18]), s[23]));
            t[4] = _mm256_add_ps(_mm256_add_ps(s[9], s[14]), _mm256_add_ps(_mm256_mul_ps(_4, s[19]), s[24]));
            dst[6] = _mm256_add_ps(_mm256_add_ps(t[0], t[1]), _mm256_add_ps(t[2], t[3]));
            dst[7] = _mm256_add_ps(_mm256_sub_ps(t[1], t[2]), _mm256_mul_ps(_2, t[3]));
            dst[8] = _mm256_add_ps(_mm256_add_ps(t[1], t[2]), _mm256_add_ps(_mm256_mul_ps(_4, t[3]), t[4]));
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetOutputStore9(const __m256 src[9], float * dst, size_t dstS, size_t dstC)
        {
            _mm256_storeu_ps(dst + 0 * dstS + 0 * dstC, src[0]);
            _mm256_storeu_ps(dst + 0 * dstS + 1 * dstC, src[1]);
            _mm256_storeu_ps(dst + 0 * dstS + 2 * dstC, src[2]);
            _mm256_storeu_ps(dst + 1 * dstS + 0 * dstC, src[3]);
            _mm256_storeu_ps(dst + 1 * dstS + 1 * dstC, src[4]);
            _mm256_storeu_ps(dst + 1 * dstS + 2 * dstC, src[5]);
            _mm256_storeu_ps(dst + 2 * dstS + 0 * dstC, src[6]);
            _mm256_storeu_ps(dst + 2 * dstS + 1 * dstC, src[7]);
            _mm256_storeu_ps(dst + 2 * dstS + 2 * dstC, src[8]);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetOutput8t(const float * src, size_t srcStride, float * dst, size_t dstW, size_t dstC)
        {
            size_t dstS = dstW * dstC, dstCF = AlignLo(dstC, F);
            for (size_t d = 0; d < dstCF; d += F)
            {
                __m256 tmp[9];
                WinogradKernel3x3Block3x3SetOutputLoad25(src + d, srcStride, tmp);
                WinogradKernel3x3Block3x3SetOutputStore9(tmp, dst + d, dstS, dstC);
            }
            if (dstCF < dstC)
            {
                __m256 tmp[9];
                WinogradKernel3x3Block3x3SetOutputLoad25(src + dstC - F, srcStride, tmp);
                WinogradKernel3x3Block3x3SetOutputStore9(tmp, dst + dstC - F, dstS, dstC);
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetOutputStore9(const __m256 src[16], float * dst, size_t dstS, size_t dstC, size_t rowE, size_t colE)
        {
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    _mm256_storeu_ps(dst + row * dstS + col * dstC, src[row * 3 + col]);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetOutput8t(const float * src, size_t srcStride, float * dst, size_t dstW, size_t dstC, size_t rowE, size_t colE)
        {
            size_t dstS = dstW * dstC, dstCF = AlignLo(dstC, F);
            for (size_t d = 0; d < dstCF; d += F)
            {
                __m256 tmp[9];
                WinogradKernel3x3Block3x3SetOutputLoad25(src + d, srcStride, tmp);
                WinogradKernel3x3Block3x3SetOutputStore9(tmp, dst + d, dstS, dstC, rowE, colE);
            }
            if (dstCF < dstC)
            {
                __m256 tmp[9];
                WinogradKernel3x3Block3x3SetOutputLoad25(src + dstC - F, srcStride, tmp);
                WinogradKernel3x3Block3x3SetOutputStore9(tmp, dst + dstC - F, dstS, dstC, rowE, colE);
            }
        }

        void WinogradKernel3x3Block3x3SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            if (trans ? (dstChannels < 8) : (dstHeight < 3 || dstWidth < 15))
            {
                Sse2::WinogradKernel3x3Block3x3SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
                return;
            }
            size_t tileH = (dstHeight + 2) / 3;
            size_t tileW = (dstWidth + 2) / 3;
            size_t dstH3 = AlignLoAny(dstHeight, 3);
            size_t dstW3 = AlignLoAny(dstWidth, 3);
            if (trans)
            {
                size_t row, col;
                for (row = 0; row < dstH3; row += 3)
                {
                    for (col = 0; col < dstW3; col += 3)
                        WinogradKernel3x3Block3x3SetOutput8t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block3x3SetOutput8t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, 3, dstWidth - col), src += dstChannels;
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstW3; col += 3)
                        WinogradKernel3x3Block3x3SetOutput8t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, dstHeight - row, 3), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block3x3SetOutput8t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, dstHeight - row, dstWidth - col), src += dstChannels;
                }
            }
            else
            {
                Base::WinogradKernel3x3Block3x3SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block4x4SetFilter8Row(const __m256 * t, float * dst, size_t stride)
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

        SIMD_INLINE void WinogradKernel3x3Block4x4SetFilter8All(const __m256 * s, float * dst, size_t stride)
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
            WinogradKernel3x3Block4x4SetFilter8Row(t, dst + 0 * stride, stride);

            t[0] = _mm256_mul_ps(mr6, _mm256_add_ps(_mm256_add_ps(s[0], s[3]), s[6]));
            t[1] = _mm256_mul_ps(mr6, _mm256_add_ps(_mm256_add_ps(s[1], s[4]), s[7]));
            t[2] = _mm256_mul_ps(mr6, _mm256_add_ps(_mm256_add_ps(s[2], s[5]), s[8]));
            WinogradKernel3x3Block4x4SetFilter8Row(t, dst + 6 * stride, stride);

            t[0] = _mm256_mul_ps(mr6, _mm256_add_ps(_mm256_sub_ps(s[0], s[3]), s[6]));
            t[1] = _mm256_mul_ps(mr6, _mm256_add_ps(_mm256_sub_ps(s[1], s[4]), s[7]));
            t[2] = _mm256_mul_ps(mr6, _mm256_add_ps(_mm256_sub_ps(s[2], s[5]), s[8]));
            WinogradKernel3x3Block4x4SetFilter8Row(t, dst + 12 * stride, stride);

            t[0] = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r24, s[0]), _mm256_mul_ps(r12, s[3])), _mm256_mul_ps(r6, s[6]));
            t[1] = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r24, s[1]), _mm256_mul_ps(r12, s[4])), _mm256_mul_ps(r6, s[7]));
            t[2] = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(r24, s[2]), _mm256_mul_ps(r12, s[5])), _mm256_mul_ps(r6, s[8]));
            WinogradKernel3x3Block4x4SetFilter8Row(t, dst + 18 * stride, stride);

            t[0] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(r24, s[0]), _mm256_mul_ps(r12, s[3])), _mm256_mul_ps(r6, s[6]));
            t[1] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(r24, s[1]), _mm256_mul_ps(r12, s[4])), _mm256_mul_ps(r6, s[7]));
            t[2] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(r24, s[2]), _mm256_mul_ps(r12, s[5])), _mm256_mul_ps(r6, s[8]));
            WinogradKernel3x3Block4x4SetFilter8Row(t, dst + 24 * stride, stride);

            WinogradKernel3x3Block4x4SetFilter8Row(s + 6, dst + 30 * stride, stride);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetFilter8t(const float * src, float * dst, size_t stride)
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
            WinogradKernel3x3Block4x4SetFilter8All(s, dst + 0 * stride, stride);
        }

        void WinogradKernel3x3Block4x4SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
        {
            if (trans)
            {
                size_t size8 = AlignLo(size, 8), i = 0;
                for (; i < size8; i += 8)
                    WinogradKernel3x3Block4x4SetFilter8t(src + i, dst + i, size);
                for (; i < size; i += 1)
                    Base::WinogradKernel3x3Block4x4SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                Sse2::WinogradKernel3x3Block4x4SetFilter(src, size, dst, trans);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block4x4SetInput8Store(const __m256 src[36], float * dst, size_t stride)
        {
            __m256 _2 = _mm256_set1_ps(2.0f);
            __m256 _4 = _mm256_set1_ps(4.0f);
            __m256 _5 = _mm256_set1_ps(5.0f);
            __m256 tmp[36];
            tmp[0] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, src[0]), _mm256_mul_ps(_5, src[12])), src[24]);
            tmp[1] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, src[1]), _mm256_mul_ps(_5, src[13])), src[25]);
            tmp[2] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, src[2]), _mm256_mul_ps(_5, src[14])), src[26]);
            tmp[3] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, src[3]), _mm256_mul_ps(_5, src[15])), src[27]);
            tmp[4] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, src[4]), _mm256_mul_ps(_5, src[16])), src[28]);
            tmp[5] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, src[5]), _mm256_mul_ps(_5, src[17])), src[29]);
            tmp[6] = _mm256_sub_ps(_mm256_add_ps(src[18], src[24]), _mm256_mul_ps(_4, _mm256_add_ps(src[6], src[12])));
            tmp[7] = _mm256_sub_ps(_mm256_add_ps(src[19], src[25]), _mm256_mul_ps(_4, _mm256_add_ps(src[7], src[13])));
            tmp[8] = _mm256_sub_ps(_mm256_add_ps(src[20], src[26]), _mm256_mul_ps(_4, _mm256_add_ps(src[8], src[14])));
            tmp[9] = _mm256_sub_ps(_mm256_add_ps(src[21], src[27]), _mm256_mul_ps(_4, _mm256_add_ps(src[9], src[15])));
            tmp[10] = _mm256_sub_ps(_mm256_add_ps(src[22], src[28]), _mm256_mul_ps(_4, _mm256_add_ps(src[10], src[16])));
            tmp[11] = _mm256_sub_ps(_mm256_add_ps(src[23], src[29]), _mm256_mul_ps(_4, _mm256_add_ps(src[11], src[17])));
            tmp[12] = _mm256_add_ps(_mm256_mul_ps(_4, _mm256_sub_ps(src[6], src[12])), _mm256_sub_ps(src[24], src[18]));
            tmp[13] = _mm256_add_ps(_mm256_mul_ps(_4, _mm256_sub_ps(src[7], src[13])), _mm256_sub_ps(src[25], src[19]));
            tmp[14] = _mm256_add_ps(_mm256_mul_ps(_4, _mm256_sub_ps(src[8], src[14])), _mm256_sub_ps(src[26], src[20]));
            tmp[15] = _mm256_add_ps(_mm256_mul_ps(_4, _mm256_sub_ps(src[9], src[15])), _mm256_sub_ps(src[27], src[21]));
            tmp[16] = _mm256_add_ps(_mm256_mul_ps(_4, _mm256_sub_ps(src[10], src[16])), _mm256_sub_ps(src[28], src[22]));
            tmp[17] = _mm256_add_ps(_mm256_mul_ps(_4, _mm256_sub_ps(src[11], src[17])), _mm256_sub_ps(src[29], src[23]));
            tmp[18] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[18], src[6])), _mm256_sub_ps(src[24], src[12]));
            tmp[19] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[19], src[7])), _mm256_sub_ps(src[25], src[13]));
            tmp[20] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[20], src[8])), _mm256_sub_ps(src[26], src[14]));
            tmp[21] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[21], src[9])), _mm256_sub_ps(src[27], src[15]));
            tmp[22] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[22], src[10])), _mm256_sub_ps(src[28], src[16]));
            tmp[23] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[23], src[11])), _mm256_sub_ps(src[29], src[17]));
            tmp[24] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[6], src[18])), _mm256_sub_ps(src[24], src[12]));
            tmp[25] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[7], src[19])), _mm256_sub_ps(src[25], src[13]));
            tmp[26] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[8], src[20])), _mm256_sub_ps(src[26], src[14]));
            tmp[27] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[9], src[21])), _mm256_sub_ps(src[27], src[15]));
            tmp[28] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[10], src[22])), _mm256_sub_ps(src[28], src[16]));
            tmp[29] = _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(src[11], src[23])), _mm256_sub_ps(src[29], src[17]));
            tmp[30] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, src[6]), _mm256_mul_ps(_5, src[18])), src[30]);
            tmp[31] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, src[7]), _mm256_mul_ps(_5, src[19])), src[31]);
            tmp[32] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, src[8]), _mm256_mul_ps(_5, src[20])), src[32]);
            tmp[33] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, src[9]), _mm256_mul_ps(_5, src[21])), src[33]);
            tmp[34] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, src[10]), _mm256_mul_ps(_5, src[22])), src[34]);
            tmp[35] = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, src[11]), _mm256_mul_ps(_5, src[23])), src[35]);

            _mm256_storeu_ps(dst + 0 * stride, _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, tmp[0]), _mm256_mul_ps(_5, tmp[2])), tmp[4]));
            _mm256_storeu_ps(dst + 1 * stride, _mm256_sub_ps(_mm256_add_ps(tmp[3], tmp[4]), _mm256_mul_ps(_4, _mm256_add_ps(tmp[1], tmp[2]))));
            _mm256_storeu_ps(dst + 2 * stride, _mm256_add_ps(_mm256_mul_ps(_4, _mm256_sub_ps(tmp[1], tmp[2])), _mm256_sub_ps(tmp[4], tmp[3])));
            _mm256_storeu_ps(dst + 3 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[3], tmp[1])), _mm256_sub_ps(tmp[4], tmp[2])));
            _mm256_storeu_ps(dst + 4 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[1], tmp[3])), _mm256_sub_ps(tmp[4], tmp[2])));
            _mm256_storeu_ps(dst + 5 * stride, _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, tmp[1]), _mm256_mul_ps(_5, tmp[3])), tmp[5]));
            _mm256_storeu_ps(dst + 6 * stride, _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, tmp[6]), _mm256_mul_ps(_5, tmp[8])), tmp[10]));
            _mm256_storeu_ps(dst + 7 * stride, _mm256_sub_ps(_mm256_add_ps(tmp[9], tmp[10]), _mm256_mul_ps(_4, _mm256_add_ps(tmp[7], tmp[8]))));
            _mm256_storeu_ps(dst + 8 * stride, _mm256_add_ps(_mm256_mul_ps(_4, _mm256_sub_ps(tmp[7], tmp[8])), _mm256_sub_ps(tmp[10], tmp[9])));
            _mm256_storeu_ps(dst + 9 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[9], tmp[7])), _mm256_sub_ps(tmp[10], tmp[8])));
            _mm256_storeu_ps(dst + 10 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[7], tmp[9])), _mm256_sub_ps(tmp[10], tmp[8])));
            _mm256_storeu_ps(dst + 11 * stride, _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, tmp[7]), _mm256_mul_ps(_5, tmp[9])), tmp[11]));
            _mm256_storeu_ps(dst + 12 * stride, _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, tmp[12]), _mm256_mul_ps(_5, tmp[14])), tmp[16]));
            _mm256_storeu_ps(dst + 13 * stride, _mm256_sub_ps(_mm256_add_ps(tmp[15], tmp[16]), _mm256_mul_ps(_4, _mm256_add_ps(tmp[13], tmp[14]))));
            _mm256_storeu_ps(dst + 14 * stride, _mm256_add_ps(_mm256_mul_ps(_4, _mm256_sub_ps(tmp[13], tmp[14])), _mm256_sub_ps(tmp[16], tmp[15])));
            _mm256_storeu_ps(dst + 15 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[15], tmp[13])), _mm256_sub_ps(tmp[16], tmp[14])));
            _mm256_storeu_ps(dst + 16 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[13], tmp[15])), _mm256_sub_ps(tmp[16], tmp[14])));
            _mm256_storeu_ps(dst + 17 * stride, _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, tmp[13]), _mm256_mul_ps(_5, tmp[15])), tmp[17]));
            _mm256_storeu_ps(dst + 18 * stride, _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, tmp[18]), _mm256_mul_ps(_5, tmp[20])), tmp[22]));
            _mm256_storeu_ps(dst + 19 * stride, _mm256_sub_ps(_mm256_add_ps(tmp[21], tmp[22]), _mm256_mul_ps(_4, _mm256_add_ps(tmp[19], tmp[20]))));
            _mm256_storeu_ps(dst + 20 * stride, _mm256_add_ps(_mm256_mul_ps(_4, _mm256_sub_ps(tmp[19], tmp[20])), _mm256_sub_ps(tmp[22], tmp[21])));
            _mm256_storeu_ps(dst + 21 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[21], tmp[19])), _mm256_sub_ps(tmp[22], tmp[20])));
            _mm256_storeu_ps(dst + 22 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[19], tmp[21])), _mm256_sub_ps(tmp[22], tmp[20])));
            _mm256_storeu_ps(dst + 23 * stride, _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, tmp[19]), _mm256_mul_ps(_5, tmp[21])), tmp[23]));
            _mm256_storeu_ps(dst + 24 * stride, _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, tmp[24]), _mm256_mul_ps(_5, tmp[26])), tmp[28]));
            _mm256_storeu_ps(dst + 25 * stride, _mm256_sub_ps(_mm256_add_ps(tmp[27], tmp[28]), _mm256_mul_ps(_4, _mm256_add_ps(tmp[25], tmp[26]))));
            _mm256_storeu_ps(dst + 26 * stride, _mm256_add_ps(_mm256_mul_ps(_4, _mm256_sub_ps(tmp[25], tmp[26])), _mm256_sub_ps(tmp[28], tmp[27])));
            _mm256_storeu_ps(dst + 27 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[27], tmp[25])), _mm256_sub_ps(tmp[28], tmp[26])));
            _mm256_storeu_ps(dst + 28 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[25], tmp[27])), _mm256_sub_ps(tmp[28], tmp[26])));
            _mm256_storeu_ps(dst + 29 * stride, _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, tmp[25]), _mm256_mul_ps(_5, tmp[27])), tmp[29]));
            _mm256_storeu_ps(dst + 30 * stride, _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, tmp[30]), _mm256_mul_ps(_5, tmp[32])), tmp[34]));
            _mm256_storeu_ps(dst + 31 * stride, _mm256_sub_ps(_mm256_add_ps(tmp[33], tmp[34]), _mm256_mul_ps(_4, _mm256_add_ps(tmp[31], tmp[32]))));
            _mm256_storeu_ps(dst + 32 * stride, _mm256_add_ps(_mm256_mul_ps(_4, _mm256_sub_ps(tmp[31], tmp[32])), _mm256_sub_ps(tmp[34], tmp[33])));
            _mm256_storeu_ps(dst + 33 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[33], tmp[31])), _mm256_sub_ps(tmp[34], tmp[32])));
            _mm256_storeu_ps(dst + 34 * stride, _mm256_add_ps(_mm256_mul_ps(_2, _mm256_sub_ps(tmp[31], tmp[33])), _mm256_sub_ps(tmp[34], tmp[32])));
            _mm256_storeu_ps(dst + 35 * stride, _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_4, tmp[31]), _mm256_mul_ps(_5, tmp[33])), tmp[35]));
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetInput8t(const float * src, size_t srcS, size_t srcC, __m256 dst[36])
        {
            dst[0] = _mm256_loadu_ps(src + 0 * srcS + 0 * srcC);
            dst[1] = _mm256_loadu_ps(src + 0 * srcS + 1 * srcC);
            dst[2] = _mm256_loadu_ps(src + 0 * srcS + 2 * srcC);
            dst[3] = _mm256_loadu_ps(src + 0 * srcS + 3 * srcC);
            dst[4] = _mm256_loadu_ps(src + 0 * srcS + 4 * srcC);
            dst[5] = _mm256_loadu_ps(src + 0 * srcS + 5 * srcC);
            dst[6] = _mm256_loadu_ps(src + 1 * srcS + 0 * srcC);
            dst[7] = _mm256_loadu_ps(src + 1 * srcS + 1 * srcC);
            dst[8] = _mm256_loadu_ps(src + 1 * srcS + 2 * srcC);
            dst[9] = _mm256_loadu_ps(src + 1 * srcS + 3 * srcC);
            dst[10] = _mm256_loadu_ps(src + 1 * srcS + 4 * srcC);
            dst[11] = _mm256_loadu_ps(src + 1 * srcS + 5 * srcC);
            dst[12] = _mm256_loadu_ps(src + 2 * srcS + 0 * srcC);
            dst[13] = _mm256_loadu_ps(src + 2 * srcS + 1 * srcC);
            dst[14] = _mm256_loadu_ps(src + 2 * srcS + 2 * srcC);
            dst[15] = _mm256_loadu_ps(src + 2 * srcS + 3 * srcC);
            dst[16] = _mm256_loadu_ps(src + 2 * srcS + 4 * srcC);
            dst[17] = _mm256_loadu_ps(src + 2 * srcS + 5 * srcC);
            dst[18] = _mm256_loadu_ps(src + 3 * srcS + 0 * srcC);
            dst[19] = _mm256_loadu_ps(src + 3 * srcS + 1 * srcC);
            dst[20] = _mm256_loadu_ps(src + 3 * srcS + 2 * srcC);
            dst[21] = _mm256_loadu_ps(src + 3 * srcS + 3 * srcC);
            dst[22] = _mm256_loadu_ps(src + 3 * srcS + 4 * srcC);
            dst[23] = _mm256_loadu_ps(src + 3 * srcS + 5 * srcC);
            dst[24] = _mm256_loadu_ps(src + 4 * srcS + 0 * srcC);
            dst[25] = _mm256_loadu_ps(src + 4 * srcS + 1 * srcC);
            dst[26] = _mm256_loadu_ps(src + 4 * srcS + 2 * srcC);
            dst[27] = _mm256_loadu_ps(src + 4 * srcS + 3 * srcC);
            dst[28] = _mm256_loadu_ps(src + 4 * srcS + 4 * srcC);
            dst[29] = _mm256_loadu_ps(src + 4 * srcS + 5 * srcC);
            dst[30] = _mm256_loadu_ps(src + 5 * srcS + 0 * srcC);
            dst[31] = _mm256_loadu_ps(src + 5 * srcS + 1 * srcC);
            dst[32] = _mm256_loadu_ps(src + 5 * srcS + 2 * srcC);
            dst[33] = _mm256_loadu_ps(src + 5 * srcS + 3 * srcC);
            dst[34] = _mm256_loadu_ps(src + 5 * srcS + 4 * srcC);
            dst[35] = _mm256_loadu_ps(src + 5 * srcS + 5 * srcC);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetInput8t(const float * src, size_t srcW, size_t srcC, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
            {
                __m256 tmp[36];
                WinogradKernel3x3Block4x4SetInput8t(src + c, srcS, srcC, tmp);
                WinogradKernel3x3Block4x4SetInput8Store(tmp, dst + c, dstStride);
            }
            if (srcCF < srcC)
            {
                __m256 tmp[36];
                WinogradKernel3x3Block4x4SetInput8t(src + srcC - F, srcS, srcC, tmp);
                WinogradKernel3x3Block4x4SetInput8Store(tmp, dst + srcC - F, dstStride);
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetInput8t(const float * src, size_t srcS, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, __m256 dst[36])
        {
            for (size_t row = 0; row < rowB; ++row)
            {
                dst[0] = _mm256_setzero_ps();
                dst[1] = _mm256_setzero_ps();
                dst[2] = _mm256_setzero_ps();
                dst[3] = _mm256_setzero_ps();
                dst[4] = _mm256_setzero_ps();
                dst[5] = _mm256_setzero_ps();
                dst += 6;
            }
            for (size_t row = rowB; row < rowE; ++row)
            {
                for (size_t col = 0; col < colB; ++col)
                    dst[col] = _mm256_setzero_ps();
                for (size_t col = colB; col < colE; ++col)
                    dst[col] = _mm256_loadu_ps(src + row * srcS + col * srcC);
                for (size_t col = colE; col < 6; ++col)
                    dst[col] = _mm256_setzero_ps();
                dst += 6;
            }
            for (size_t row = rowE; row < 6; ++row)
            {
                dst[0] = _mm256_setzero_ps();
                dst[1] = _mm256_setzero_ps();
                dst[2] = _mm256_setzero_ps();
                dst[3] = _mm256_setzero_ps();
                dst[4] = _mm256_setzero_ps();
                dst[5] = _mm256_setzero_ps();
                dst += 6;
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetInput8t(const float * src, size_t srcW, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
            {
                __m256 tmp[36];
                WinogradKernel3x3Block4x4SetInput8t(src + c, srcS, srcC, rowB, rowE, colB, colE, tmp);
                WinogradKernel3x3Block4x4SetInput8Store(tmp, dst + c, dstStride);
            }
            if (srcCF < srcC)
            {
                __m256 tmp[36];
                WinogradKernel3x3Block4x4SetInput8t(src + srcC - F, srcS, srcC, rowB, rowE, colB, colE, tmp);
                WinogradKernel3x3Block4x4SetInput8Store(tmp, dst + srcC - F, dstStride);
            }
        }

        void WinogradKernel3x3Block4x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            if (trans ? (srcChannels < 8) : (srcHeight < 6 || srcWidth < 14))
            {
                Sse2::WinogradKernel3x3Block4x4SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
                return;
            }
            if (trans)
            {
                assert(padY + padH <= 2 && padX + padW <= 2);
                size_t dstH = srcHeight - 2 + padY + padH;
                size_t dstW = srcWidth - 2 + padX + padW;
                size_t dstH4 = dstH / 4 * 4;
                size_t dstW4 = dstW / 4 * 4;
                size_t noseW = Simd::Min<size_t>(6, srcWidth + padX);
                size_t noseH = Simd::Min<size_t>(6, srcHeight + padY);
                size_t startY = padY ? 4 : 0;
                size_t startX = padX ? 4 : 0;
                if (padH && dstH == dstH4)
                    dstH4 -= 4;
                if (padY)
                    src -= srcWidth * srcChannels;
                if (padW && dstW == dstW4)
                    dstW4 -= 4;
                if (padX)
                    src -= srcChannels;
                size_t tailW = dstW - dstW4 + (padW ? 1 : 2);
                size_t tailH = dstH - dstH4 + (padH ? 1 : 2);
                size_t row = 0, col = 0;
                if (padY)
                {
                    if (padX)
                        WinogradKernel3x3Block4x4SetInput8t(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = startX; col < dstW4; col += 4)
                        WinogradKernel3x3Block4x4SetInput8t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 6, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        WinogradKernel3x3Block4x4SetInput8t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                for (row = startY; row < dstH4; row += 4)
                {
                    if (padX)
                        WinogradKernel3x3Block4x4SetInput8t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 6, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = startX; col < dstW4; col += 4)
                        WinogradKernel3x3Block4x4SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        WinogradKernel3x3Block4x4SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 6, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                if (row < dstH)
                {
                    if (padX)
                        WinogradKernel3x3Block4x4SetInput8t(src + row * srcWidth* srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = startX; col < dstW4; col += 4)
                        WinogradKernel3x3Block4x4SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 6, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        WinogradKernel3x3Block4x4SetInput8t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
            }
            else
            {
                Base::WinogradKernel3x3Block4x4SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block4x4SetOutputLoad36(const float * src, size_t stride, __m256 dst[16])
        {
            __m256 s[36];
            s[0] = _mm256_loadu_ps(src + 0 * stride);
            s[1] = _mm256_loadu_ps(src + 1 * stride);
            s[2] = _mm256_loadu_ps(src + 2 * stride);
            s[3] = _mm256_loadu_ps(src + 3 * stride);
            s[4] = _mm256_loadu_ps(src + 4 * stride);
            s[5] = _mm256_loadu_ps(src + 5 * stride);
            s[6] = _mm256_loadu_ps(src + 6 * stride);
            s[7] = _mm256_loadu_ps(src + 7 * stride);
            s[8] = _mm256_loadu_ps(src + 8 * stride);
            s[9] = _mm256_loadu_ps(src + 9 * stride);
            s[10] = _mm256_loadu_ps(src + 10 * stride);
            s[11] = _mm256_loadu_ps(src + 11 * stride);
            s[12] = _mm256_loadu_ps(src + 12 * stride);
            s[13] = _mm256_loadu_ps(src + 13 * stride);
            s[14] = _mm256_loadu_ps(src + 14 * stride);
            s[15] = _mm256_loadu_ps(src + 15 * stride);
            s[16] = _mm256_loadu_ps(src + 16 * stride);
            s[17] = _mm256_loadu_ps(src + 17 * stride);
            s[18] = _mm256_loadu_ps(src + 18 * stride);
            s[19] = _mm256_loadu_ps(src + 19 * stride);
            s[20] = _mm256_loadu_ps(src + 20 * stride);
            s[21] = _mm256_loadu_ps(src + 21 * stride);
            s[22] = _mm256_loadu_ps(src + 22 * stride);
            s[23] = _mm256_loadu_ps(src + 23 * stride);
            s[24] = _mm256_loadu_ps(src + 24 * stride);
            s[25] = _mm256_loadu_ps(src + 25 * stride);
            s[26] = _mm256_loadu_ps(src + 26 * stride);
            s[27] = _mm256_loadu_ps(src + 27 * stride);
            s[28] = _mm256_loadu_ps(src + 28 * stride);
            s[29] = _mm256_loadu_ps(src + 29 * stride);
            s[30] = _mm256_loadu_ps(src + 30 * stride);
            s[31] = _mm256_loadu_ps(src + 31 * stride);
            s[32] = _mm256_loadu_ps(src + 32 * stride);
            s[33] = _mm256_loadu_ps(src + 33 * stride);
            s[34] = _mm256_loadu_ps(src + 34 * stride);
            s[35] = _mm256_loadu_ps(src + 35 * stride);

            __m256 _2 = _mm256_set1_ps(2.0f);
            __m256 _4 = _mm256_set1_ps(4.0f);
            __m256 _8 = _mm256_set1_ps(8.0f);
            __m256 t[24];
            t[0] = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(s[0], s[6]), _mm256_add_ps(s[12], s[18])), s[24]);
            t[1] = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(s[1], s[7]), _mm256_add_ps(s[13], s[19])), s[25]);
            t[2] = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(s[2], s[8]), _mm256_add_ps(s[14], s[20])), s[26]);
            t[3] = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(s[3], s[9]), _mm256_add_ps(s[15], s[21])), s[27]);
            t[4] = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(s[4], s[10]), _mm256_add_ps(s[16], s[22])), s[28]);
            t[5] = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(s[5], s[11]), _mm256_add_ps(s[17], s[23])), s[29]);
            t[6] = _mm256_add_ps(_mm256_sub_ps(s[6], s[12]), _mm256_mul_ps(_2, _mm256_sub_ps(s[18], s[24])));
            t[7] = _mm256_add_ps(_mm256_sub_ps(s[7], s[13]), _mm256_mul_ps(_2, _mm256_sub_ps(s[19], s[25])));
            t[8] = _mm256_add_ps(_mm256_sub_ps(s[8], s[14]), _mm256_mul_ps(_2, _mm256_sub_ps(s[20], s[26])));
            t[9] = _mm256_add_ps(_mm256_sub_ps(s[9], s[15]), _mm256_mul_ps(_2, _mm256_sub_ps(s[21], s[27])));
            t[10] = _mm256_add_ps(_mm256_sub_ps(s[10], s[16]), _mm256_mul_ps(_2, _mm256_sub_ps(s[22], s[28])));
            t[11] = _mm256_add_ps(_mm256_sub_ps(s[11], s[17]), _mm256_mul_ps(_2, _mm256_sub_ps(s[23], s[29])));
            t[12] = _mm256_add_ps(_mm256_add_ps(s[6], s[12]), _mm256_mul_ps(_4, _mm256_add_ps(s[18], s[24])));
            t[13] = _mm256_add_ps(_mm256_add_ps(s[7], s[13]), _mm256_mul_ps(_4, _mm256_add_ps(s[19], s[25])));
            t[14] = _mm256_add_ps(_mm256_add_ps(s[8], s[14]), _mm256_mul_ps(_4, _mm256_add_ps(s[20], s[26])));
            t[15] = _mm256_add_ps(_mm256_add_ps(s[9], s[15]), _mm256_mul_ps(_4, _mm256_add_ps(s[21], s[27])));
            t[16] = _mm256_add_ps(_mm256_add_ps(s[10], s[16]), _mm256_mul_ps(_4, _mm256_add_ps(s[22], s[28])));
            t[17] = _mm256_add_ps(_mm256_add_ps(s[11], s[17]), _mm256_mul_ps(_4, _mm256_add_ps(s[23], s[29])));
            t[18] = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(s[6], s[12]), _mm256_mul_ps(_8, _mm256_sub_ps(s[18], s[24]))), s[30]);
            t[19] = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(s[7], s[13]), _mm256_mul_ps(_8, _mm256_sub_ps(s[19], s[25]))), s[31]);
            t[20] = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(s[8], s[14]), _mm256_mul_ps(_8, _mm256_sub_ps(s[20], s[26]))), s[32]);
            t[21] = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(s[9], s[15]), _mm256_mul_ps(_8, _mm256_sub_ps(s[21], s[27]))), s[33]);
            t[22] = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(s[10], s[16]), _mm256_mul_ps(_8, _mm256_sub_ps(s[22], s[28]))), s[34]);
            t[23] = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(s[11], s[17]), _mm256_mul_ps(_8, _mm256_sub_ps(s[23], s[29]))), s[35]);

            dst[0] = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(t[0], t[1]), _mm256_add_ps(t[2], t[3])), t[4]);
            dst[1] = _mm256_add_ps(_mm256_sub_ps(t[1], t[2]), _mm256_mul_ps(_2, _mm256_sub_ps(t[3], t[4])));
            dst[2] = _mm256_add_ps(_mm256_add_ps(t[1], t[2]), _mm256_mul_ps(_4, _mm256_add_ps(t[3], t[4])));
            dst[3] = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(t[1], t[2]), _mm256_mul_ps(_8, _mm256_sub_ps(t[3], t[4]))), t[5]);
            dst[4] = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(t[6], t[7]), _mm256_add_ps(t[8], t[9])), t[10]);
            dst[5] = _mm256_add_ps(_mm256_sub_ps(t[7], t[8]), _mm256_mul_ps(_2, _mm256_sub_ps(t[9], t[10])));
            dst[6] = _mm256_add_ps(_mm256_add_ps(t[7], t[8]), _mm256_mul_ps(_4, _mm256_add_ps(t[9], t[10])));
            dst[7] = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(t[7], t[8]), _mm256_mul_ps(_8, _mm256_sub_ps(t[9], t[10]))), t[11]);
            dst[8] = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(t[12], t[13]), _mm256_add_ps(t[14], t[15])), t[16]);
            dst[9] = _mm256_add_ps(_mm256_sub_ps(t[13], t[14]), _mm256_mul_ps(_2, _mm256_sub_ps(t[15], t[16])));
            dst[10] = _mm256_add_ps(_mm256_add_ps(t[13], t[14]), _mm256_mul_ps(_4, _mm256_add_ps(t[15], t[16])));
            dst[11] = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(t[13], t[14]), _mm256_mul_ps(_8, _mm256_sub_ps(t[15], t[16]))), t[17]);
            dst[12] = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(t[18], t[19]), _mm256_add_ps(t[20], t[21])), t[22]);
            dst[13] = _mm256_add_ps(_mm256_sub_ps(t[19], t[20]), _mm256_mul_ps(_2, _mm256_sub_ps(t[21], t[22])));
            dst[14] = _mm256_add_ps(_mm256_add_ps(t[19], t[20]), _mm256_mul_ps(_4, _mm256_add_ps(t[21], t[22])));
            dst[15] = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(t[19], t[20]), _mm256_mul_ps(_8, _mm256_sub_ps(t[21], t[22]))), t[23]);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetOutputStore16(const __m256 src[16], float * dst, size_t dstS, size_t dstC)
        {
            _mm256_storeu_ps(dst + 0 * dstS + 0 * dstC, src[0]);
            _mm256_storeu_ps(dst + 0 * dstS + 1 * dstC, src[1]);
            _mm256_storeu_ps(dst + 0 * dstS + 2 * dstC, src[2]);
            _mm256_storeu_ps(dst + 0 * dstS + 3 * dstC, src[3]);
            _mm256_storeu_ps(dst + 1 * dstS + 0 * dstC, src[4]);
            _mm256_storeu_ps(dst + 1 * dstS + 1 * dstC, src[5]);
            _mm256_storeu_ps(dst + 1 * dstS + 2 * dstC, src[6]);
            _mm256_storeu_ps(dst + 1 * dstS + 3 * dstC, src[7]);
            _mm256_storeu_ps(dst + 2 * dstS + 0 * dstC, src[8]);
            _mm256_storeu_ps(dst + 2 * dstS + 1 * dstC, src[9]);
            _mm256_storeu_ps(dst + 2 * dstS + 2 * dstC, src[10]);
            _mm256_storeu_ps(dst + 2 * dstS + 3 * dstC, src[11]);
            _mm256_storeu_ps(dst + 3 * dstS + 0 * dstC, src[12]);
            _mm256_storeu_ps(dst + 3 * dstS + 1 * dstC, src[13]);
            _mm256_storeu_ps(dst + 3 * dstS + 2 * dstC, src[14]);
            _mm256_storeu_ps(dst + 3 * dstS + 3 * dstC, src[15]);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetOutput8t(const float * src, size_t srcStride, float * dst, size_t dstW, size_t dstC)
        {
            size_t dstS = dstW * dstC, dstCF = AlignLo(dstC, F);
            for (size_t d = 0; d < dstCF; d += F)
            {
                __m256 tmp[16];
                WinogradKernel3x3Block4x4SetOutputLoad36(src + d, srcStride, tmp);
                WinogradKernel3x3Block4x4SetOutputStore16(tmp, dst + d, dstS, dstC);
            }
            if (dstCF < dstC)
            {
                __m256 tmp[16];
                WinogradKernel3x3Block4x4SetOutputLoad36(src + dstC - F, srcStride, tmp);
                WinogradKernel3x3Block4x4SetOutputStore16(tmp, dst + dstC - F, dstS, dstC);
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetOutputStore16(const __m256 src[16], float * dst, size_t dstS, size_t dstC, size_t rowE, size_t colE)
        {
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    _mm256_storeu_ps(dst + row * dstS + col * dstC, src[row * 4 + col]);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetOutput8t(const float * src, size_t srcStride, float * dst, size_t dstW, size_t dstC, size_t rowE, size_t colE)
        {
            size_t dstS = dstW * dstC, dstCF = AlignLo(dstC, F);
            for (size_t d = 0; d < dstCF; d += F)
            {
                __m256 tmp[16];
                WinogradKernel3x3Block4x4SetOutputLoad36(src + d, srcStride, tmp);
                WinogradKernel3x3Block4x4SetOutputStore16(tmp, dst + d, dstS, dstC, rowE, colE);
            }
            if (dstCF < dstC)
            {
                __m256 tmp[16];
                WinogradKernel3x3Block4x4SetOutputLoad36(src + dstC - F, srcStride, tmp);
                WinogradKernel3x3Block4x4SetOutputStore16(tmp, dst + dstC - F, dstS, dstC, rowE, colE);
            }
        }

        void WinogradKernel3x3Block4x4SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            if (trans ? (dstChannels < 8) : (dstHeight < 4 || dstWidth < 16))
            {
                Sse2::WinogradKernel3x3Block4x4SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
                return;
            }
            size_t tileH = (dstHeight + 3) / 4;
            size_t tileW = (dstWidth + 3) / 4;
            size_t dstH4 = AlignLo(dstHeight, 4);
            size_t dstW4 = AlignLo(dstWidth, 4);
            if (trans)
            {
                size_t row, col;
                for (row = 0; row < dstH4; row += 4)
                {
                    for (col = 0; col < dstW4; col += 4)
                        WinogradKernel3x3Block4x4SetOutput8t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block4x4SetOutput8t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, 4, dstWidth - col), src += dstChannels;
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstW4; col += 4)
                        WinogradKernel3x3Block4x4SetOutput8t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, dstHeight - row, 4), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block4x4SetOutput8t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, dstHeight - row, dstWidth - col), src += dstChannels;
                }
            }
            else
            {
                Base::WinogradKernel3x3Block4x4SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
            }
        }
    }
#endif// SIMD_AVX_ENABLE
}
