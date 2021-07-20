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
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdWinograd.h"

namespace Simd
{
#if defined(SIMD_AVX512F_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512f
    {
        SIMD_INLINE void WinogradKernel2x2Block2x2SetFilter(const __m512 src[4], float* dst, size_t stride, __mmask16 tail)
        {
            _mm512_mask_storeu_ps(dst + 0 * stride, tail, src[0]);
            _mm512_mask_storeu_ps(dst + 1 * stride, tail, _mm512_add_ps(src[0], src[1]));
            _mm512_mask_storeu_ps(dst + 2 * stride, tail, src[1]);

            _mm512_mask_storeu_ps(dst + 3 * stride, tail, _mm512_add_ps(src[0], src[2]));
            _mm512_mask_storeu_ps(dst + 4 * stride, tail, _mm512_add_ps(_mm512_add_ps(src[0], src[1]), _mm512_add_ps(src[2], src[3])));
            _mm512_mask_storeu_ps(dst + 5 * stride, tail, _mm512_add_ps(src[1], src[3]));

            _mm512_mask_storeu_ps(dst + 6 * stride, tail, src[2]);
            _mm512_mask_storeu_ps(dst + 7 * stride, tail, _mm512_add_ps(src[2], src[3]));
            _mm512_mask_storeu_ps(dst + 8 * stride, tail, src[3]);
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetFilter16t(const float* src, float* dst, size_t stride, __mmask16 tail = -1)
        {
            __m512 _src[4];
            _src[0] = _mm512_maskz_loadu_ps(tail, src + 0 * stride);
            _src[1] = _mm512_maskz_loadu_ps(tail, src + 1 * stride);
            _src[2] = _mm512_maskz_loadu_ps(tail, src + 2 * stride);
            _src[3] = _mm512_maskz_loadu_ps(tail, src + 3 * stride);
            WinogradKernel2x2Block2x2SetFilter(_src, dst, stride, tail);
        }

        void WinogradKernel2x2Block2x2SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            size_t sizeF = AlignLo(size, F), i = 0;
            if (trans)
            {
                for (; i < sizeF; i += F)
                    WinogradKernel2x2Block2x2SetFilter16t(src + i, dst + i, size);
                if (i < size)
                {
                    __mmask16 tail = TailMask16(size - sizeF);
                    WinogradKernel2x2Block2x2SetFilter16t(src + i, dst + i, size, tail);
                }
            }
            else
            {
                Sse2::WinogradKernel2x2Block2x2SetFilter(src, size, dst, trans);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel2x2Block2x2SetInput16Store(const __m512* src, float* dst, size_t stride, __mmask16 tail = -1)
        {
            _mm512_mask_storeu_ps(dst + 0 * stride, tail, _mm512_add_ps(_mm512_sub_ps(src[0], src[1]), _mm512_sub_ps(src[4], src[3])));
            _mm512_mask_storeu_ps(dst + 1 * stride, tail, _mm512_sub_ps(src[1], src[4]));
            _mm512_mask_storeu_ps(dst + 2 * stride, tail, _mm512_add_ps(_mm512_sub_ps(src[2], src[1]), _mm512_sub_ps(src[4], src[5])));
            _mm512_mask_storeu_ps(dst + 3 * stride, tail, _mm512_sub_ps(src[3], src[4]));
            _mm512_mask_storeu_ps(dst + 4 * stride, tail, src[4]);
            _mm512_mask_storeu_ps(dst + 5 * stride, tail, _mm512_sub_ps(src[5], src[4]));
            _mm512_mask_storeu_ps(dst + 6 * stride, tail, _mm512_add_ps(_mm512_sub_ps(src[4], src[3]), _mm512_sub_ps(src[6], src[7])));
            _mm512_mask_storeu_ps(dst + 7 * stride, tail, _mm512_sub_ps(src[7], src[4]));
            _mm512_mask_storeu_ps(dst + 8 * stride, tail, _mm512_add_ps(_mm512_sub_ps(src[4], src[5]), _mm512_sub_ps(src[8], src[7])));
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetInput16t(const float* src, size_t srcS, size_t srcC, __m512 dst[9], __mmask16 tail = -1)
        {
            dst[0] = _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 0 * srcC);
            dst[1] = _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 1 * srcC);
            dst[2] = _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 2 * srcC);
            dst[3] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 0 * srcC);
            dst[4] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 1 * srcC);
            dst[5] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 2 * srcC);
            dst[6] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 0 * srcC);
            dst[7] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 1 * srcC);
            dst[8] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 2 * srcC);
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetInput16t(const float* src, size_t srcW, size_t srcC, float* dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m512 tmp[9];
                WinogradKernel2x2Block2x2SetInput16t(src + c, srcS, srcC, tmp);
                WinogradKernel2x2Block2x2SetInput16Store(tmp, dst + c, dstStride);
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - srcCF);
                __m512 tmp[9];
                WinogradKernel2x2Block2x2SetInput16t(src + c, srcS, srcC, tmp, tail);
                WinogradKernel2x2Block2x2SetInput16Store(tmp, dst + c, dstStride, tail);
            }
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetInput16t(const float* src, size_t srcS, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, __m512 dst[9], __mmask16 tail = -1)
        {
            for (size_t row = 0; row < rowB; ++row)
            {
                for (size_t col = 0; col < 3; ++col)
                    dst[col] = _mm512_setzero_ps();
                dst += 3;
            }
            for (size_t row = rowB; row < rowE; ++row)
            {
                for (size_t col = 0; col < colB; ++col)
                    dst[col] = _mm512_setzero_ps();
                for (size_t col = colB; col < colE; ++col)
                    dst[col] = _mm512_maskz_loadu_ps(tail, src + row * srcS + col * srcC);
                for (size_t col = colE; col < 3; ++col)
                    dst[col] = _mm512_setzero_ps();
                dst += 3;
            }
            for (size_t row = rowE; row < 3; ++row)
            {
                for (size_t col = 0; col < 3; ++col)
                    dst[col] = _mm512_setzero_ps();
                dst += 3;
            }
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetInput16t(const float* src, size_t srcW, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m512 tmp[9];
                WinogradKernel2x2Block2x2SetInput16t(src + c, srcS, srcC, rowB, rowE, colB, colE, tmp);
                WinogradKernel2x2Block2x2SetInput16Store(tmp, dst + c, dstStride);
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - srcCF);
                __m512 tmp[9];
                WinogradKernel2x2Block2x2SetInput16t(src + c, srcS, srcC, rowB, rowE, colB, colE, tmp, tail);
                WinogradKernel2x2Block2x2SetInput16Store(tmp, dst + c, dstStride, tail);
            }
        }

        void WinogradKernel2x2Block2x2SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padY == padX && padW == padH && (padY + padH == 0 || padY + padH == 1));
            if (trans ? false : true)
            {
                Base::WinogradKernel2x2Block2x2SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
                return;
            }
            size_t dstH = srcHeight - 1 + padY + padH;
            size_t dstW = srcWidth - 1 + padX + padW;
            size_t dstH2 = AlignLo(dstH, 2);
            size_t dstW2 = AlignLo(dstW, 2);
            size_t noseW = Simd::Min<size_t>(3, dstW + 1);
            size_t noseH = Simd::Min<size_t>(3, dstH + 1);
            size_t startY = padY ? 2 : 0;
            size_t startX = padX ? 2 : 0;
            if (padY || padH)
            {
                if (dstH == dstH2)
                    dstH2 -= 2;
                if (dstW == dstW2)
                    dstW2 -= 2;
                if (padY)
                    src -= (srcWidth + 1) * (trans ? srcChannels : 1);
            }
            size_t tailW = dstW - dstW2 + (padW ? 0 : 1);
            size_t tailH = dstH - dstH2 + (padH ? 0 : 1);
            size_t row = 0, col = 0;
            if (padY)
            {
                if (padX)
                    WinogradKernel2x2Block2x2SetInput16t(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = startX; col < dstW2; col += 2)
                    WinogradKernel2x2Block2x2SetInput16t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 3, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel2x2Block2x2SetInput16t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
            }
            for (row = startY; row < dstH2; row += 2)
            {
                if (padX)
                    WinogradKernel2x2Block2x2SetInput16t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 3, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = startX; col < dstW2; col += 2)
                    WinogradKernel2x2Block2x2SetInput16t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel2x2Block2x2SetInput16t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 3, 0, tailW, dst, dstStride), dst += srcChannels;
            }
            if (row < dstH)
            {
                if (padX)
                    WinogradKernel2x2Block2x2SetInput16t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = startX; col < dstW2; col += 2)
                    WinogradKernel2x2Block2x2SetInput16t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 3, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel2x2Block2x2SetInput16t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel2x2Block2x2SetOutputLoad9(const float* src, size_t stride, __m512* dst, __mmask16 tail = -1)
        {
            __m512 s[9];
            s[0] = _mm512_maskz_loadu_ps(tail, src + 0 * stride);
            s[1] = _mm512_maskz_loadu_ps(tail, src + 1 * stride);
            s[2] = _mm512_maskz_loadu_ps(tail, src + 2 * stride);
            s[3] = _mm512_maskz_loadu_ps(tail, src + 3 * stride);
            s[4] = _mm512_maskz_loadu_ps(tail, src + 4 * stride);
            s[5] = _mm512_maskz_loadu_ps(tail, src + 5 * stride);
            s[6] = _mm512_maskz_loadu_ps(tail, src + 6 * stride);
            s[7] = _mm512_maskz_loadu_ps(tail, src + 7 * stride);
            s[8] = _mm512_maskz_loadu_ps(tail, src + 8 * stride);
            dst[0] = _mm512_add_ps(_mm512_add_ps(s[0], s[1]), _mm512_add_ps(s[3], s[4]));
            dst[1] = _mm512_add_ps(_mm512_add_ps(s[1], s[2]), _mm512_add_ps(s[4], s[5]));
            dst[2] = _mm512_add_ps(_mm512_add_ps(s[3], s[4]), _mm512_add_ps(s[6], s[7]));
            dst[3] = _mm512_add_ps(_mm512_add_ps(s[4], s[5]), _mm512_add_ps(s[7], s[8]));
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetOutputStore16(const __m512 src[4], float* dst, size_t dstS, size_t dstC, __mmask16 tail = -1)
        {
            _mm512_mask_storeu_ps(dst + 0 * dstS + 0 * dstC, tail, src[0]);
            _mm512_mask_storeu_ps(dst + 0 * dstS + 1 * dstC, tail, src[1]);
            _mm512_mask_storeu_ps(dst + 1 * dstS + 0 * dstC, tail, src[2]);
            _mm512_mask_storeu_ps(dst + 1 * dstS + 1 * dstC, tail, src[3]);
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetOutput16t(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC)
        {
            size_t dstS = dstW * dstC, dstCF = AlignLo(dstC, F);
            for (size_t d = 0; d < dstCF; d += F)
            {
                __m512 tmp[4];
                WinogradKernel2x2Block2x2SetOutputLoad9(src + d, srcStride, tmp);
                WinogradKernel2x2Block2x2SetOutputStore16(tmp, dst + d, dstS, dstC);
            }
            if (dstCF < dstC)
            {
                __mmask16 tail = TailMask16(dstC - dstCF);
                __m512 tmp[4];
                WinogradKernel2x2Block2x2SetOutputLoad9(src + dstCF, srcStride, tmp, tail);
                WinogradKernel2x2Block2x2SetOutputStore16(tmp, dst + dstCF, dstS, dstC, tail);
            }
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetOutputStore16(const __m512 src[4], float* dst, size_t dstS, size_t dstC, size_t rowE, size_t colE, __mmask16 tail = -1)
        {
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    _mm512_mask_storeu_ps(dst + row * dstS + col * dstC, tail, src[row * 2 + col]);
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetOutput16t(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC, size_t rowE, size_t colE)
        {
            size_t dstS = dstW * dstC, dstCF = AlignLo(dstC, F);
            for (size_t d = 0; d < dstCF; d += F)
            {
                __m512 tmp[4];
                WinogradKernel2x2Block2x2SetOutputLoad9(src + d, srcStride, tmp);
                WinogradKernel2x2Block2x2SetOutputStore16(tmp, dst + d, dstS, dstC, rowE, colE);
            }
            if (dstCF < dstC)
            {
                __mmask16 tail = TailMask16(dstC - dstCF);
                __m512 tmp[4];
                WinogradKernel2x2Block2x2SetOutputLoad9(src + dstCF, srcStride, tmp, tail);
                WinogradKernel2x2Block2x2SetOutputStore16(tmp, dst + dstCF, dstS, dstC, rowE, colE, tail);
            }
        }

        void WinogradKernel2x2Block2x2SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            if (trans ? false : true)
            {
                Base::WinogradKernel2x2Block2x2SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
                return;
            }
            size_t tileH = (dstHeight + 1) / 2;
            size_t tileW = (dstWidth + 1) / 2;
            size_t dstH2 = AlignLo(dstHeight, 2);
            size_t dstW2 = AlignLo(dstWidth, 2);
            size_t row, col;
            for (row = 0; row < dstH2; row += 2)
            {
                for (col = 0; col < dstW2; col += 2)
                    WinogradKernel2x2Block2x2SetOutput16t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels), src += dstChannels;
                if (col < dstWidth)
                    WinogradKernel2x2Block2x2SetOutput16t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, 2, dstWidth - col), src += dstChannels;
            }
            if (row < dstHeight)
            {
                for (col = 0; col < dstW2; col += 2)
                    WinogradKernel2x2Block2x2SetOutput16t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row, 2), src += dstChannels;
                if (col < dstWidth)
                    WinogradKernel2x2Block2x2SetOutput16t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row, dstWidth - col), src += dstChannels;
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel2x2Block4x4SetFilterRow(const __m512* t, float* dst, size_t stride, __mmask16 tail)
        {
            const __m512 r2 = _mm512_set1_ps(1.0f / 2.0f);
            const __m512 r3 = _mm512_set1_ps(1.0f / 3.0f);
            const __m512 r6 = _mm512_set1_ps(1.0f / 6.0f);
            const __m512 mr2 = _mm512_set1_ps(-1.0f / 2.0f);

            _mm512_mask_storeu_ps(dst + 0 * stride, tail, _mm512_mul_ps(r2, t[0]));
            _mm512_mask_storeu_ps(dst + 1 * stride, tail, _mm512_mul_ps(mr2, _mm512_add_ps(t[0], t[1])));
            _mm512_mask_storeu_ps(dst + 2 * stride, tail, _mm512_mul_ps(r6, _mm512_sub_ps(t[1], t[0])));
            _mm512_mask_storeu_ps(dst + 3 * stride, tail, _mm512_add_ps(_mm512_mul_ps(r6, t[0]), _mm512_mul_ps(r3, t[1])));
            _mm512_mask_storeu_ps(dst + 4 * stride, tail, t[1]);
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetFilter(const __m512 src[4], float* dst, size_t stride, __mmask16 tail)
        {
            const __m512 r2 = _mm512_set1_ps(1.0f / 2.0f);
            const __m512 r3 = _mm512_set1_ps(1.0f / 3.0f);
            const __m512 r6 = _mm512_set1_ps(1.0f / 6.0f);
            const __m512 mr2 = _mm512_set1_ps(-1.0f / 2.0f);

            __m512 t[2];
            t[0] = _mm512_mul_ps(r2, src[0]);
            t[1] = _mm512_mul_ps(r2, src[1]);
            WinogradKernel2x2Block4x4SetFilterRow(t, dst + 0 * stride, stride, tail);

            t[0] = _mm512_mul_ps(mr2, _mm512_add_ps(src[0], src[2]));
            t[1] = _mm512_mul_ps(mr2, _mm512_add_ps(src[1], src[3]));
            WinogradKernel2x2Block4x4SetFilterRow(t, dst + 5 * stride, stride, tail);

            t[0] = _mm512_mul_ps(r6, _mm512_sub_ps(src[2], src[0]));
            t[1] = _mm512_mul_ps(r6, _mm512_sub_ps(src[3], src[1]));
            WinogradKernel2x2Block4x4SetFilterRow(t, dst + 10 * stride, stride, tail);

            t[0] = _mm512_add_ps(_mm512_mul_ps(r6, src[0]), _mm512_mul_ps(r3, src[2]));
            t[1] = _mm512_add_ps(_mm512_mul_ps(r6, src[1]), _mm512_mul_ps(r3, src[3]));
            WinogradKernel2x2Block4x4SetFilterRow(t, dst + 15 * stride, stride, tail);

            t[0] = src[2];
            t[1] = src[3];
            WinogradKernel2x2Block4x4SetFilterRow(t, dst + 20 * stride, stride, tail);
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetFilter16t(const float* src, float* dst, size_t stride, __mmask16 tail = -1)
        {
            __m512 _src[4];
            _src[0] = _mm512_loadu_ps(src + 0 * stride);
            _src[1] = _mm512_loadu_ps(src + 1 * stride);
            _src[2] = _mm512_loadu_ps(src + 2 * stride);
            _src[3] = _mm512_loadu_ps(src + 3 * stride);
            WinogradKernel2x2Block4x4SetFilter(_src, dst, stride, tail);
        }

        void WinogradKernel2x2Block4x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            size_t sizeF = AlignLo(size, F), i = 0;
            if (trans)
            {
                for (; i < sizeF; i += F)
                    WinogradKernel2x2Block4x4SetFilter16t(src + i, dst + i, size);
                if (i < size)
                {
                    __mmask16 tail = TailMask16(size - sizeF);
                    WinogradKernel2x2Block4x4SetFilter16t(src + i, dst + i, size, tail);
                }
            }
            else
            {
                Sse2::WinogradKernel2x2Block4x4SetFilter(src, size, dst, trans);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel2x2Block4x4SetInputStoreRow(const __m512 tmp[5], __m512 _2, __m512 _3, float* dst, size_t stride, __mmask16 tail)
        {
            _mm512_mask_storeu_ps(dst + 0 * stride, tail, _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, tmp[0]), tmp[1]), _mm512_sub_ps(tmp[3], _mm512_mul_ps(_2, tmp[2]))));
            _mm512_mask_storeu_ps(dst + 1 * stride, tail, _mm512_sub_ps(tmp[3], _mm512_add_ps(_mm512_mul_ps(_2, tmp[1]), tmp[2])));
            _mm512_mask_storeu_ps(dst + 2 * stride, tail, _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, tmp[1]), _mm512_mul_ps(_3, tmp[2])), tmp[3]));
            _mm512_mask_storeu_ps(dst + 3 * stride, tail, _mm512_sub_ps(tmp[3], tmp[1]));
            _mm512_mask_storeu_ps(dst + 4 * stride, tail, _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, tmp[1]), tmp[2]), _mm512_sub_ps(tmp[4], _mm512_mul_ps(_2, tmp[3]))));
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetInputStore(const __m512* src, float* dst, size_t stride, __mmask16 tail = -1)
        {
            const __m512 _2 = _mm512_set1_ps(2.0f);
            const __m512 _3 = _mm512_set1_ps(3.0f);
            __m512 tmp[5];
            tmp[0] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, src[0]), src[5]), _mm512_sub_ps(src[15], _mm512_mul_ps(_2, src[10])));
            tmp[1] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, src[1]), src[6]), _mm512_sub_ps(src[16], _mm512_mul_ps(_2, src[11])));
            tmp[2] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, src[2]), src[7]), _mm512_sub_ps(src[17], _mm512_mul_ps(_2, src[12])));
            tmp[3] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, src[3]), src[8]), _mm512_sub_ps(src[18], _mm512_mul_ps(_2, src[13])));
            tmp[4] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, src[4]), src[9]), _mm512_sub_ps(src[19], _mm512_mul_ps(_2, src[14])));
            WinogradKernel2x2Block4x4SetInputStoreRow(tmp, _2, _3, dst + 0 * stride, stride, tail);

            tmp[0] = _mm512_sub_ps(src[15], _mm512_add_ps(_mm512_mul_ps(_2, src[5]), src[10]));
            tmp[1] = _mm512_sub_ps(src[16], _mm512_add_ps(_mm512_mul_ps(_2, src[6]), src[11]));
            tmp[2] = _mm512_sub_ps(src[17], _mm512_add_ps(_mm512_mul_ps(_2, src[7]), src[12]));
            tmp[3] = _mm512_sub_ps(src[18], _mm512_add_ps(_mm512_mul_ps(_2, src[8]), src[13]));
            tmp[4] = _mm512_sub_ps(src[19], _mm512_add_ps(_mm512_mul_ps(_2, src[9]), src[14]));
            WinogradKernel2x2Block4x4SetInputStoreRow(tmp, _2, _3, dst + 5 * stride, stride, tail);

            tmp[0] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, src[5]), _mm512_mul_ps(_3, src[10])), src[15]);
            tmp[1] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, src[6]), _mm512_mul_ps(_3, src[11])), src[16]);
            tmp[2] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, src[7]), _mm512_mul_ps(_3, src[12])), src[17]);
            tmp[3] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, src[8]), _mm512_mul_ps(_3, src[13])), src[18]);
            tmp[4] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, src[9]), _mm512_mul_ps(_3, src[14])), src[19]);
            WinogradKernel2x2Block4x4SetInputStoreRow(tmp, _2, _3, dst + 10 * stride, stride, tail);

            tmp[0] = _mm512_sub_ps(src[15], src[5]);
            tmp[1] = _mm512_sub_ps(src[16], src[6]);
            tmp[2] = _mm512_sub_ps(src[17], src[7]);
            tmp[3] = _mm512_sub_ps(src[18], src[8]);
            tmp[4] = _mm512_sub_ps(src[19], src[9]);
            WinogradKernel2x2Block4x4SetInputStoreRow(tmp, _2, _3, dst + 15 * stride, stride, tail);

            tmp[0] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, src[5]), src[10]), _mm512_sub_ps(src[20], _mm512_mul_ps(_2, src[15])));
            tmp[1] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, src[6]), src[11]), _mm512_sub_ps(src[21], _mm512_mul_ps(_2, src[16])));
            tmp[2] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, src[7]), src[12]), _mm512_sub_ps(src[22], _mm512_mul_ps(_2, src[17])));
            tmp[3] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, src[8]), src[13]), _mm512_sub_ps(src[23], _mm512_mul_ps(_2, src[18])));
            tmp[4] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, src[9]), src[14]), _mm512_sub_ps(src[24], _mm512_mul_ps(_2, src[19])));
            WinogradKernel2x2Block4x4SetInputStoreRow(tmp, _2, _3, dst + 20 * stride, stride, tail);
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetInput16t(const float* src, size_t srcS, size_t srcC, __m512 dst[25], __mmask16 tail = -1)
        {
            dst[0] = _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 0 * srcC);
            dst[1] = _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 1 * srcC);
            dst[2] = _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 2 * srcC);
            dst[3] = _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 3 * srcC);
            dst[4] = _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 4 * srcC);
            dst[5] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 0 * srcC);
            dst[6] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 1 * srcC);
            dst[7] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 2 * srcC);
            dst[8] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 3 * srcC);
            dst[9] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 4 * srcC);
            dst[10] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 0 * srcC);
            dst[11] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 1 * srcC);
            dst[12] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 2 * srcC);
            dst[13] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 3 * srcC);
            dst[14] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 4 * srcC);
            dst[15] = _mm512_maskz_loadu_ps(tail, src + 3 * srcS + 0 * srcC);
            dst[16] = _mm512_maskz_loadu_ps(tail, src + 3 * srcS + 1 * srcC);
            dst[17] = _mm512_maskz_loadu_ps(tail, src + 3 * srcS + 2 * srcC);
            dst[18] = _mm512_maskz_loadu_ps(tail, src + 3 * srcS + 3 * srcC);
            dst[19] = _mm512_maskz_loadu_ps(tail, src + 3 * srcS + 4 * srcC);
            dst[20] = _mm512_maskz_loadu_ps(tail, src + 4 * srcS + 0 * srcC);
            dst[21] = _mm512_maskz_loadu_ps(tail, src + 4 * srcS + 1 * srcC);
            dst[22] = _mm512_maskz_loadu_ps(tail, src + 4 * srcS + 2 * srcC);
            dst[23] = _mm512_maskz_loadu_ps(tail, src + 4 * srcS + 3 * srcC);
            dst[24] = _mm512_maskz_loadu_ps(tail, src + 4 * srcS + 4 * srcC);
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetInput16t(const float* src, size_t srcS, size_t srcC, float* dst, size_t stride, __mmask16 tail)
        {
            __m512 s[25], t[5];
            const __m512 _2 = _mm512_set1_ps(2.0f);
            const __m512 _3 = _mm512_set1_ps(3.0f);

            s[5] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 0 * srcC);
            s[6] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 1 * srcC);
            s[7] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 2 * srcC);
            s[8] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 3 * srcC);
            s[9] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 4 * srcC);
            s[10] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 0 * srcC);
            s[11] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 1 * srcC);
            s[12] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 2 * srcC);
            s[13] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 3 * srcC);
            s[14] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 4 * srcC);
            s[15] = _mm512_maskz_loadu_ps(tail, src + 3 * srcS + 0 * srcC);
            s[16] = _mm512_maskz_loadu_ps(tail, src + 3 * srcS + 1 * srcC);
            s[17] = _mm512_maskz_loadu_ps(tail, src + 3 * srcS + 2 * srcC);
            s[18] = _mm512_maskz_loadu_ps(tail, src + 3 * srcS + 3 * srcC);
            s[19] = _mm512_maskz_loadu_ps(tail, src + 3 * srcS + 4 * srcC);

            t[0] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 0 * srcC)), s[5]), _mm512_sub_ps(s[15], _mm512_mul_ps(_2, s[10])));
            t[1] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 1 * srcC)), s[6]), _mm512_sub_ps(s[16], _mm512_mul_ps(_2, s[11])));
            t[2] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 2 * srcC)), s[7]), _mm512_sub_ps(s[17], _mm512_mul_ps(_2, s[12])));
            t[3] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 3 * srcC)), s[8]), _mm512_sub_ps(s[18], _mm512_mul_ps(_2, s[13])));
            t[4] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 4 * srcC)), s[9]), _mm512_sub_ps(s[19], _mm512_mul_ps(_2, s[14])));
            WinogradKernel2x2Block4x4SetInputStoreRow(t, _2, _3, dst + 0 * stride, stride, tail);

            t[0] = _mm512_sub_ps(s[15], _mm512_add_ps(_mm512_mul_ps(_2, s[5]), s[10]));
            t[1] = _mm512_sub_ps(s[16], _mm512_add_ps(_mm512_mul_ps(_2, s[6]), s[11]));
            t[2] = _mm512_sub_ps(s[17], _mm512_add_ps(_mm512_mul_ps(_2, s[7]), s[12]));
            t[3] = _mm512_sub_ps(s[18], _mm512_add_ps(_mm512_mul_ps(_2, s[8]), s[13]));
            t[4] = _mm512_sub_ps(s[19], _mm512_add_ps(_mm512_mul_ps(_2, s[9]), s[14]));
            WinogradKernel2x2Block4x4SetInputStoreRow(t, _2, _3, dst + 5 * stride, stride, tail);

            t[0] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, s[5]), _mm512_mul_ps(_3, s[10])), s[15]);
            t[1] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, s[6]), _mm512_mul_ps(_3, s[11])), s[16]);
            t[2] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, s[7]), _mm512_mul_ps(_3, s[12])), s[17]);
            t[3] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, s[8]), _mm512_mul_ps(_3, s[13])), s[18]);
            t[4] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, s[9]), _mm512_mul_ps(_3, s[14])), s[19]);
            WinogradKernel2x2Block4x4SetInputStoreRow(t, _2, _3, dst + 10 * stride, stride, tail);

            t[0] = _mm512_sub_ps(s[15], s[5]);
            t[1] = _mm512_sub_ps(s[16], s[6]);
            t[2] = _mm512_sub_ps(s[17], s[7]);
            t[3] = _mm512_sub_ps(s[18], s[8]);
            t[4] = _mm512_sub_ps(s[19], s[9]);
            WinogradKernel2x2Block4x4SetInputStoreRow(t, _2, _3, dst + 15 * stride, stride, tail);

            t[0] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, s[5]), s[10]), _mm512_sub_ps(_mm512_maskz_loadu_ps(tail, src + 4 * srcS + 0 * srcC), _mm512_mul_ps(_2, s[15])));
            t[1] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, s[6]), s[11]), _mm512_sub_ps(_mm512_maskz_loadu_ps(tail, src + 4 * srcS + 1 * srcC), _mm512_mul_ps(_2, s[16])));
            t[2] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, s[7]), s[12]), _mm512_sub_ps(_mm512_maskz_loadu_ps(tail, src + 4 * srcS + 2 * srcC), _mm512_mul_ps(_2, s[17])));
            t[3] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, s[8]), s[13]), _mm512_sub_ps(_mm512_maskz_loadu_ps(tail, src + 4 * srcS + 3 * srcC), _mm512_mul_ps(_2, s[18])));
            t[4] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(_2, s[9]), s[14]), _mm512_sub_ps(_mm512_maskz_loadu_ps(tail, src + 4 * srcS + 4 * srcC), _mm512_mul_ps(_2, s[19])));
            WinogradKernel2x2Block4x4SetInputStoreRow(t, _2, _3, dst + 20 * stride, stride, tail);
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetInput16t(const float* src, size_t srcW, size_t srcC, float* dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
                WinogradKernel2x2Block4x4SetInput16t(src + c, srcS, srcC, dst + c, dstStride, __mmask16(-1));
            if (srcCF < srcC)
            {
                __mmask16 tail = TailMask16(srcC - srcCF);
                WinogradKernel2x2Block4x4SetInput16t(src + srcCF, srcS, srcC, dst + srcCF, dstStride, tail);
            }
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetInput16t(const float* src, size_t srcS, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, __m512 * dst, __mmask16 tail = -1)
        {
            for (size_t row = 0; row < rowB; ++row)
            {
                for (size_t col = 0; col < 5; ++col)
                    dst[col] = _mm512_setzero_ps();
                dst += 5;
            }
            for (size_t row = rowB; row < rowE; ++row)
            {
                for (size_t col = 0; col < colB; ++col)
                    dst[col] = _mm512_setzero_ps();
                for (size_t col = colB; col < colE; ++col)
                    dst[col] = _mm512_maskz_loadu_ps(tail, src + row * srcS + col * srcC);
                for (size_t col = colE; col < 5; ++col)
                    dst[col] = _mm512_setzero_ps();
                dst += 5;
            }
            for (size_t row = rowE; row < 5; ++row)
            {
                for (size_t col = 0; col < 5; ++col)
                    dst[col] = _mm512_setzero_ps();
                dst += 5;
            }
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetInput16t(const float* src, size_t srcW, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
            {
                __m512 tmp[25];
                WinogradKernel2x2Block4x4SetInput16t(src + c, srcS, srcC, rowB, rowE, colB, colE, tmp);
                WinogradKernel2x2Block4x4SetInputStore(tmp, dst + c, dstStride);
            }
            if (srcCF < srcC)
            {
                __mmask16 tail = TailMask16(srcC - srcCF);
                __m512 tmp[25];
                WinogradKernel2x2Block4x4SetInput16t(src + srcC - F, srcS, srcC, rowB, rowE, colB, colE, tmp, tail);
                WinogradKernel2x2Block4x4SetInputStore(tmp, dst + srcC - F, dstStride, tail);
            }
        }

        void WinogradKernel2x2Block4x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padY == padX && padW == padH && (padY + padH == 0 || padY + padH == 1));
            if (trans ? false : true)
            {
                Base::WinogradKernel2x2Block4x4SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
                return;
            }
            size_t dstH = srcHeight - 1 + padY + padH;
            size_t dstW = srcWidth - 1 + padX + padW;
            size_t dstH4 = AlignLo(dstH, 4);
            size_t dstW4 = AlignLo(dstW, 4);
            size_t noseW = Simd::Min<size_t>(5, dstW + 1);
            size_t noseH = Simd::Min<size_t>(5, dstH + 1);
            size_t startY = padY ? 4 : 0;
            size_t startX = padX ? 4 : 0;
            if (padY || padH)
            {
                if (dstH == dstH4)
                    dstH4 -= 4;
                if (dstW == dstW4)
                    dstW4 -= 4;
                if (padY)
                    src -= (srcWidth + 1) * (trans ? srcChannels : 1);
            }
            size_t tailW = dstW - dstW4 + (padW ? 0 : 1);
            size_t tailH = dstH - dstH4 + (padH ? 0 : 1);
            size_t row = 0, col = 0;
            if (padY)
            {
                if (padX)
                    WinogradKernel2x2Block4x4SetInput16t(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = startX; col < dstW4; col += 4)
                    WinogradKernel2x2Block4x4SetInput16t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 5, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel2x2Block4x4SetInput16t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
            }
            for (row = startY; row < dstH4; row += 4)
            {
                if (padX)
                    WinogradKernel2x2Block4x4SetInput16t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 5, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = startX; col < dstW4; col += 4)
                    WinogradKernel2x2Block4x4SetInput16t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel2x2Block4x4SetInput16t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 5, 0, tailW, dst, dstStride), dst += srcChannels;
            }
            if (row < dstH)
            {
                if (padX)
                    WinogradKernel2x2Block4x4SetInput16t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = startX; col < dstW4; col += 4)
                    WinogradKernel2x2Block4x4SetInput16t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 5, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel2x2Block4x4SetInput16t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel2x2Block4x4SetOutputGetRow(const __m512 t[5], __m512 _2, __m512 _4, __m512 _8, __m512* d)
        {
            d[0] = _mm512_add_ps(_mm512_add_ps(t[0], t[1]), _mm512_add_ps(t[2], t[3]));
            d[1] = _mm512_add_ps(_mm512_sub_ps(t[1], t[2]), _mm512_mul_ps(_2, t[3]));
            d[2] = _mm512_add_ps(_mm512_add_ps(t[1], t[2]), _mm512_mul_ps(_4, t[3]));
            d[3] = _mm512_add_ps(_mm512_sub_ps(t[1], t[2]), _mm512_add_ps(_mm512_mul_ps(_8, t[3]), t[4]));
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetOutputSaveRow(const __m512 t[5], __m512 _2, __m512 _4, __m512 _8, float* dst, size_t dstC, __mmask16 tail)
        {
            _mm512_mask_storeu_ps(dst + 0 * dstC, tail, _mm512_add_ps(_mm512_add_ps(t[0], t[1]), _mm512_add_ps(t[2], t[3])));
            _mm512_mask_storeu_ps(dst + 1 * dstC, tail, _mm512_add_ps(_mm512_sub_ps(t[1], t[2]), _mm512_mul_ps(_2, t[3])));
            _mm512_mask_storeu_ps(dst + 2 * dstC, tail, _mm512_add_ps(_mm512_add_ps(t[1], t[2]), _mm512_mul_ps(_4, t[3])));
            _mm512_mask_storeu_ps(dst + 3 * dstC, tail, _mm512_add_ps(_mm512_sub_ps(t[1], t[2]), _mm512_add_ps(_mm512_mul_ps(_8, t[3]), t[4])));
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetOutput(const float* src, size_t stride, float * dst, size_t dstS, size_t dstC, size_t rowE, __mmask16 tail)
        {
            __m512 s[25], t[5];
            s[5] = _mm512_maskz_loadu_ps(tail, src + 5 * stride);
            s[6] = _mm512_maskz_loadu_ps(tail, src + 6 * stride);
            s[7] = _mm512_maskz_loadu_ps(tail, src + 7 * stride);
            s[8] = _mm512_maskz_loadu_ps(tail, src + 8 * stride);
            s[9] = _mm512_maskz_loadu_ps(tail, src + 9 * stride);
            s[10] = _mm512_maskz_loadu_ps(tail, src + 10 * stride);
            s[11] = _mm512_maskz_loadu_ps(tail, src + 11 * stride);
            s[12] = _mm512_maskz_loadu_ps(tail, src + 12 * stride);
            s[13] = _mm512_maskz_loadu_ps(tail, src + 13 * stride);
            s[14] = _mm512_maskz_loadu_ps(tail, src + 14 * stride);
            s[15] = _mm512_maskz_loadu_ps(tail, src + 15 * stride);
            s[16] = _mm512_maskz_loadu_ps(tail, src + 16 * stride);
            s[17] = _mm512_maskz_loadu_ps(tail, src + 17 * stride);
            s[18] = _mm512_maskz_loadu_ps(tail, src + 18 * stride);
            s[19] = _mm512_maskz_loadu_ps(tail, src + 19 * stride);

            const __m512 _2 = _mm512_set1_ps(2.0f);
            const __m512 _4 = _mm512_set1_ps(4.0f);
            const __m512 _8 = _mm512_set1_ps(8.0f);
            t[0] = _mm512_add_ps(_mm512_add_ps(_mm512_maskz_loadu_ps(tail, src + 0 * stride), s[5]), _mm512_add_ps(s[10], s[15]));
            t[1] = _mm512_add_ps(_mm512_add_ps(_mm512_maskz_loadu_ps(tail, src + 1 * stride), s[6]), _mm512_add_ps(s[11], s[16]));
            t[2] = _mm512_add_ps(_mm512_add_ps(_mm512_maskz_loadu_ps(tail, src + 2 * stride), s[7]), _mm512_add_ps(s[12], s[17]));
            t[3] = _mm512_add_ps(_mm512_add_ps(_mm512_maskz_loadu_ps(tail, src + 3 * stride), s[8]), _mm512_add_ps(s[13], s[18]));
            t[4] = _mm512_add_ps(_mm512_add_ps(_mm512_maskz_loadu_ps(tail, src + 4 * stride), s[9]), _mm512_add_ps(s[14], s[19]));
            WinogradKernel2x2Block4x4SetOutputSaveRow(t, _2, _4, _8, dst + 0 * dstS, dstC, tail);
            if (rowE == 1) return;

            t[0] = _mm512_add_ps(_mm512_sub_ps(s[5], s[10]), _mm512_mul_ps(_2, s[15]));
            t[1] = _mm512_add_ps(_mm512_sub_ps(s[6], s[11]), _mm512_mul_ps(_2, s[16]));
            t[2] = _mm512_add_ps(_mm512_sub_ps(s[7], s[12]), _mm512_mul_ps(_2, s[17]));
            t[3] = _mm512_add_ps(_mm512_sub_ps(s[8], s[13]), _mm512_mul_ps(_2, s[18]));
            t[4] = _mm512_add_ps(_mm512_sub_ps(s[9], s[14]), _mm512_mul_ps(_2, s[19]));
            WinogradKernel2x2Block4x4SetOutputSaveRow(t, _2, _4, _8, dst + 1 * dstS, dstC, tail);
            if (rowE == 2) return;

            t[0] = _mm512_add_ps(_mm512_add_ps(s[5], s[10]), _mm512_mul_ps(_4, s[15]));
            t[1] = _mm512_add_ps(_mm512_add_ps(s[6], s[11]), _mm512_mul_ps(_4, s[16]));
            t[2] = _mm512_add_ps(_mm512_add_ps(s[7], s[12]), _mm512_mul_ps(_4, s[17]));
            t[3] = _mm512_add_ps(_mm512_add_ps(s[8], s[13]), _mm512_mul_ps(_4, s[18]));
            t[4] = _mm512_add_ps(_mm512_add_ps(s[9], s[14]), _mm512_mul_ps(_4, s[19]));
            WinogradKernel2x2Block4x4SetOutputSaveRow(t, _2, _4, _8, dst + 2 * dstS, dstC, tail);
            if (rowE == 3) return;

            t[0] = _mm512_add_ps(_mm512_sub_ps(s[5], s[10]), _mm512_add_ps(_mm512_mul_ps(_8, s[15]), _mm512_maskz_loadu_ps(tail, src + 20 * stride)));
            t[1] = _mm512_add_ps(_mm512_sub_ps(s[6], s[11]), _mm512_add_ps(_mm512_mul_ps(_8, s[16]), _mm512_maskz_loadu_ps(tail, src + 21 * stride)));
            t[2] = _mm512_add_ps(_mm512_sub_ps(s[7], s[12]), _mm512_add_ps(_mm512_mul_ps(_8, s[17]), _mm512_maskz_loadu_ps(tail, src + 22 * stride)));
            t[3] = _mm512_add_ps(_mm512_sub_ps(s[8], s[13]), _mm512_add_ps(_mm512_mul_ps(_8, s[18]), _mm512_maskz_loadu_ps(tail, src + 23 * stride)));
            t[4] = _mm512_add_ps(_mm512_sub_ps(s[9], s[14]), _mm512_add_ps(_mm512_mul_ps(_8, s[19]), _mm512_maskz_loadu_ps(tail, src + 24 * stride)));
            WinogradKernel2x2Block4x4SetOutputSaveRow(t, _2, _4, _8, dst + 3 * dstS, dstC, tail);
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetOutput16t(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC, size_t rowE)
        {
            size_t dstS = dstW * dstC, dstCF = AlignLo(dstC, F);
            for (size_t d = 0; d < dstCF; d += F)
                WinogradKernel2x2Block4x4SetOutput(src + d, srcStride, dst + d, dstS, dstC, rowE, __mmask16(-1));
            if (dstCF < dstC)
            {
                __mmask16 tail = TailMask16(dstC - dstCF);
                WinogradKernel2x2Block4x4SetOutput(src + dstCF, srcStride, dst + dstCF, dstS, dstC, rowE, tail);
            }
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetOutputLoad25(const float* src, size_t stride, __m512* dst, __mmask16 tail = -1)
        {
            __m512 s[25], t[5];
            s[5] = _mm512_maskz_loadu_ps(tail, src + 5 * stride);
            s[6] = _mm512_maskz_loadu_ps(tail, src + 6 * stride);
            s[7] = _mm512_maskz_loadu_ps(tail, src + 7 * stride);
            s[8] = _mm512_maskz_loadu_ps(tail, src + 8 * stride);
            s[9] = _mm512_maskz_loadu_ps(tail, src + 9 * stride);
            s[10] = _mm512_maskz_loadu_ps(tail, src + 10 * stride);
            s[11] = _mm512_maskz_loadu_ps(tail, src + 11 * stride);
            s[12] = _mm512_maskz_loadu_ps(tail, src + 12 * stride);
            s[13] = _mm512_maskz_loadu_ps(tail, src + 13 * stride);
            s[14] = _mm512_maskz_loadu_ps(tail, src + 14 * stride);
            s[15] = _mm512_maskz_loadu_ps(tail, src + 15 * stride);
            s[16] = _mm512_maskz_loadu_ps(tail, src + 16 * stride);
            s[17] = _mm512_maskz_loadu_ps(tail, src + 17 * stride);
            s[18] = _mm512_maskz_loadu_ps(tail, src + 18 * stride);
            s[19] = _mm512_maskz_loadu_ps(tail, src + 19 * stride);

            const __m512 _2 = _mm512_set1_ps(2.0f);
            const __m512 _4 = _mm512_set1_ps(4.0f);
            const __m512 _8 = _mm512_set1_ps(8.0f);
            t[0] = _mm512_add_ps(_mm512_add_ps(_mm512_maskz_loadu_ps(tail, src + 0 * stride), s[5]), _mm512_add_ps(s[10], s[15]));
            t[1] = _mm512_add_ps(_mm512_add_ps(_mm512_maskz_loadu_ps(tail, src + 1 * stride), s[6]), _mm512_add_ps(s[11], s[16]));
            t[2] = _mm512_add_ps(_mm512_add_ps(_mm512_maskz_loadu_ps(tail, src + 2 * stride), s[7]), _mm512_add_ps(s[12], s[17]));
            t[3] = _mm512_add_ps(_mm512_add_ps(_mm512_maskz_loadu_ps(tail, src + 3 * stride), s[8]), _mm512_add_ps(s[13], s[18]));
            t[4] = _mm512_add_ps(_mm512_add_ps(_mm512_maskz_loadu_ps(tail, src + 4 * stride), s[9]), _mm512_add_ps(s[14], s[19]));
            WinogradKernel2x2Block4x4SetOutputGetRow(t, _2, _4, _8, dst + 0);

            t[0] = _mm512_add_ps(_mm512_sub_ps(s[5], s[10]), _mm512_mul_ps(_2, s[15]));
            t[1] = _mm512_add_ps(_mm512_sub_ps(s[6], s[11]), _mm512_mul_ps(_2, s[16]));
            t[2] = _mm512_add_ps(_mm512_sub_ps(s[7], s[12]), _mm512_mul_ps(_2, s[17]));
            t[3] = _mm512_add_ps(_mm512_sub_ps(s[8], s[13]), _mm512_mul_ps(_2, s[18]));
            t[4] = _mm512_add_ps(_mm512_sub_ps(s[9], s[14]), _mm512_mul_ps(_2, s[19]));
            WinogradKernel2x2Block4x4SetOutputGetRow(t, _2, _4, _8, dst + 4);

            t[0] = _mm512_add_ps(_mm512_add_ps(s[5], s[10]), _mm512_mul_ps(_4, s[15]));
            t[1] = _mm512_add_ps(_mm512_add_ps(s[6], s[11]), _mm512_mul_ps(_4, s[16]));
            t[2] = _mm512_add_ps(_mm512_add_ps(s[7], s[12]), _mm512_mul_ps(_4, s[17]));
            t[3] = _mm512_add_ps(_mm512_add_ps(s[8], s[13]), _mm512_mul_ps(_4, s[18]));
            t[4] = _mm512_add_ps(_mm512_add_ps(s[9], s[14]), _mm512_mul_ps(_4, s[19]));
            WinogradKernel2x2Block4x4SetOutputGetRow(t, _2, _4, _8, dst + 8);

            t[0] = _mm512_add_ps(_mm512_sub_ps(s[5], s[10]), _mm512_add_ps(_mm512_mul_ps(_8, s[15]), _mm512_maskz_loadu_ps(tail, src + 20 * stride)));
            t[1] = _mm512_add_ps(_mm512_sub_ps(s[6], s[11]), _mm512_add_ps(_mm512_mul_ps(_8, s[16]), _mm512_maskz_loadu_ps(tail, src + 21 * stride)));
            t[2] = _mm512_add_ps(_mm512_sub_ps(s[7], s[12]), _mm512_add_ps(_mm512_mul_ps(_8, s[17]), _mm512_maskz_loadu_ps(tail, src + 22 * stride)));
            t[3] = _mm512_add_ps(_mm512_sub_ps(s[8], s[13]), _mm512_add_ps(_mm512_mul_ps(_8, s[18]), _mm512_maskz_loadu_ps(tail, src + 23 * stride)));
            t[4] = _mm512_add_ps(_mm512_sub_ps(s[9], s[14]), _mm512_add_ps(_mm512_mul_ps(_8, s[19]), _mm512_maskz_loadu_ps(tail, src + 24 * stride)));
            WinogradKernel2x2Block4x4SetOutputGetRow(t, _2, _4, _8, dst + 12);
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetOutputStore16(const __m512 src[16], float* dst, size_t dstS, size_t dstC, __mmask16 tail = -1)
        {
            _mm512_mask_storeu_ps(dst + 0 * dstS + 0 * dstC, tail, src[0]);
            _mm512_mask_storeu_ps(dst + 0 * dstS + 1 * dstC, tail, src[1]);
            _mm512_mask_storeu_ps(dst + 0 * dstS + 2 * dstC, tail, src[2]);
            _mm512_mask_storeu_ps(dst + 0 * dstS + 3 * dstC, tail, src[3]);
            _mm512_mask_storeu_ps(dst + 1 * dstS + 0 * dstC, tail, src[4]);
            _mm512_mask_storeu_ps(dst + 1 * dstS + 1 * dstC, tail, src[5]);
            _mm512_mask_storeu_ps(dst + 1 * dstS + 2 * dstC, tail, src[6]);
            _mm512_mask_storeu_ps(dst + 1 * dstS + 3 * dstC, tail, src[7]);
            _mm512_mask_storeu_ps(dst + 2 * dstS + 0 * dstC, tail, src[8]);
            _mm512_mask_storeu_ps(dst + 2 * dstS + 1 * dstC, tail, src[9]);
            _mm512_mask_storeu_ps(dst + 2 * dstS + 2 * dstC, tail, src[10]);
            _mm512_mask_storeu_ps(dst + 2 * dstS + 3 * dstC, tail, src[11]);
            _mm512_mask_storeu_ps(dst + 3 * dstS + 0 * dstC, tail, src[12]);
            _mm512_mask_storeu_ps(dst + 3 * dstS + 1 * dstC, tail, src[13]);
            _mm512_mask_storeu_ps(dst + 3 * dstS + 2 * dstC, tail, src[14]);
            _mm512_mask_storeu_ps(dst + 3 * dstS + 3 * dstC, tail, src[15]);
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetOutputStore16(const __m512 src[16], float* dst, size_t dstS, size_t dstC, size_t rowE, size_t colE, __mmask16 tail = -1)
        {
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    _mm512_mask_storeu_ps(dst + row * dstS + col * dstC, tail, src[row * 4 + col]);
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetOutput16t(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC, size_t rowE, size_t colE)
        {
            size_t dstS = dstW * dstC, dstCF = AlignLo(dstC, F);
            for (size_t d = 0; d < dstCF; d += F)
            {
                __m512 tmp[16];
                WinogradKernel2x2Block4x4SetOutputLoad25(src + d, srcStride, tmp);
                WinogradKernel2x2Block4x4SetOutputStore16(tmp, dst + d, dstS, dstC, rowE, colE);
            }
            if (dstCF < dstC)
            {
                __mmask16 tail = TailMask16(dstC - dstCF);
                __m512 tmp[16];
                WinogradKernel2x2Block4x4SetOutputLoad25(src + dstCF, srcStride, tmp, tail);
                WinogradKernel2x2Block4x4SetOutputStore16(tmp, dst + dstCF, dstS, dstC, rowE, colE, tail);
            }
        }

        void WinogradKernel2x2Block4x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            if (trans ? false : true)
            {
                Base::WinogradKernel2x2Block4x4SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
                return;
            }
            size_t tileH = (dstHeight + 3) / 4;
            size_t tileW = (dstWidth + 3) / 4;
            size_t dstH4 = AlignLo(dstHeight, 4);
            size_t dstW4 = AlignLo(dstWidth, 4);
            size_t row, col;
            for (row = 0; row < dstH4; row += 4)
            {
                for (col = 0; col < dstW4; col += 4)
                    WinogradKernel2x2Block4x4SetOutput16t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, 4), src += dstChannels;
                if (col < dstWidth)
                    WinogradKernel2x2Block4x4SetOutput16t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, 4, dstWidth - col), src += dstChannels;
            }
            if (row < dstHeight)
            {
                for (col = 0; col < dstW4; col += 4)
                    WinogradKernel2x2Block4x4SetOutput16t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row), src += dstChannels;
                if (col < dstWidth)
                    WinogradKernel2x2Block4x4SetOutput16t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row, dstWidth - col), src += dstChannels;
            }
        }
    }
#endif// SIMD_AVX512F_ENABLE
}
