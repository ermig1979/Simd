/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdWinograd.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE void WinogradKernel2x2Block2x2SetFilter(const svfloat32_t& s0, const svfloat32_t& s1,
            const svfloat32_t& s2, const svfloat32_t& s3, float* dst, size_t stride, const svbool_t& pg)
        {
            svst1_f32(pg, dst + 0 * stride, s0);
            svst1_f32(pg, dst + 1 * stride, svadd_f32_x(pg, s0, s1));
            svst1_f32(pg, dst + 2 * stride, s1);

            svst1_f32(pg, dst + 3 * stride, svadd_f32_x(pg, s0, s2));
            svst1_f32(pg, dst + 4 * stride, svadd_f32_x(pg, svadd_f32_x(pg, s0, s1), svadd_f32_x(pg, s2, s3)));
            svst1_f32(pg, dst + 5 * stride, svadd_f32_x(pg, s1, s3));

            svst1_f32(pg, dst + 6 * stride, s2);
            svst1_f32(pg, dst + 7 * stride, svadd_f32_x(pg, s2, s3));
            svst1_f32(pg, dst + 8 * stride, s3);
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetFilterVt(const float* src, size_t srcStride, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svfloat32_t s0 = svld1_f32(pg, src + 0 * srcStride);
            svfloat32_t s1 = svld1_f32(pg, src + 1 * srcStride);
            svfloat32_t s2 = svld1_f32(pg, src + 2 * srcStride);
            svfloat32_t s3 = svld1_f32(pg, src + 3 * srcStride);
            WinogradKernel2x2Block2x2SetFilter(s0, s1, s2, s3, dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetFilterVn(const float* src, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svuint32_t offsets = svindex_u32(0, 4);
            svfloat32_t s0 = svld1_gather_u32index_f32(pg, src + 0, offsets);
            svfloat32_t s1 = svld1_gather_u32index_f32(pg, src + 1, offsets);
            svfloat32_t s2 = svld1_gather_u32index_f32(pg, src + 2, offsets);
            svfloat32_t s3 = svld1_gather_u32index_f32(pg, src + 3, offsets);
            WinogradKernel2x2Block2x2SetFilter(s0, s1, s2, s3, dst, dstStride, pg);
        }

        void WinogradKernel2x2Block2x2SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            const size_t F = svcntw();
            const size_t sizeF = AlignLo(size, F);
            const svbool_t body = svptrue_b32();
            size_t i = 0;
            if (trans)
            {
                for (; i < sizeF; i += F)
                    WinogradKernel2x2Block2x2SetFilterVt(src + i, size, dst + i, size, body);
                if (i < size)
                    WinogradKernel2x2Block2x2SetFilterVt(src + i, size, dst + i, size, svwhilelt_b32(i, size));
            }
            else
            {
                for (; i < sizeF; i += F, src += 4 * F, dst += F)
                    WinogradKernel2x2Block2x2SetFilterVn(src, dst, size, body);
                if (i < size)
                    WinogradKernel2x2Block2x2SetFilterVn(src, dst, size, svwhilelt_b32(i, size));
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel2x2Block2x2SetInputStore(const svfloat32_t& s0, const svfloat32_t& s1, const svfloat32_t& s2,
            const svfloat32_t& s3, const svfloat32_t& s4, const svfloat32_t& s5, const svfloat32_t& s6, const svfloat32_t& s7,
            const svfloat32_t& s8, float* dst, size_t stride, const svbool_t& pg)
        {
            svst1_f32(pg, dst + 0 * stride, svadd_f32_x(pg, svsub_f32_x(pg, s0, s1), svsub_f32_x(pg, s4, s3)));
            svst1_f32(pg, dst + 1 * stride, svsub_f32_x(pg, s1, s4));
            svst1_f32(pg, dst + 2 * stride, svadd_f32_x(pg, svsub_f32_x(pg, s2, s1), svsub_f32_x(pg, s4, s5)));
            svst1_f32(pg, dst + 3 * stride, svsub_f32_x(pg, s3, s4));
            svst1_f32(pg, dst + 4 * stride, s4);
            svst1_f32(pg, dst + 5 * stride, svsub_f32_x(pg, s5, s4));
            svst1_f32(pg, dst + 6 * stride, svadd_f32_x(pg, svsub_f32_x(pg, s4, s3), svsub_f32_x(pg, s6, s7)));
            svst1_f32(pg, dst + 7 * stride, svsub_f32_x(pg, s7, s4));
            svst1_f32(pg, dst + 8 * stride, svadd_f32_x(pg, svsub_f32_x(pg, s4, s5), svsub_f32_x(pg, s8, s7)));
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetInput(const float* src, size_t srcS, size_t srcC, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svfloat32_t s0 = svld1_f32(pg, src + 0 * srcS + 0 * srcC);
            svfloat32_t s1 = svld1_f32(pg, src + 0 * srcS + 1 * srcC);
            svfloat32_t s2 = svld1_f32(pg, src + 0 * srcS + 2 * srcC);
            svfloat32_t s3 = svld1_f32(pg, src + 1 * srcS + 0 * srcC);
            svfloat32_t s4 = svld1_f32(pg, src + 1 * srcS + 1 * srcC);
            svfloat32_t s5 = svld1_f32(pg, src + 1 * srcS + 2 * srcC);
            svfloat32_t s6 = svld1_f32(pg, src + 2 * srcS + 0 * srcC);
            svfloat32_t s7 = svld1_f32(pg, src + 2 * srcS + 1 * srcC);
            svfloat32_t s8 = svld1_f32(pg, src + 2 * srcS + 2 * srcC);
            WinogradKernel2x2Block2x2SetInputStore(s0, s1, s2, s3, s4, s5, s6, s7, s8, dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetInput(const float* src, size_t srcW, size_t srcC, float* dst, size_t dstStride)
        {
            const size_t F = svcntw();
            const size_t srcS = srcW * srcC;
            const size_t srcCF = AlignLo(srcC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < srcCF; c += F)
                WinogradKernel2x2Block2x2SetInput(src + c, srcS, srcC, dst + c, dstStride, body);
            if (c < srcC)
                WinogradKernel2x2Block2x2SetInput(src + c, srcS, srcC, dst + c, dstStride, svwhilelt_b32(c, srcC));
        }

        SIMD_INLINE svfloat32_t WinogradKernel2x2Block2x2SetInputLoad(const float* src, size_t srcS, size_t srcC, size_t row,
            size_t rowB, size_t rowE, size_t col, size_t colB, size_t colE, const svbool_t& pg)
        {
            return row >= rowB && row < rowE && col >= colB && col < colE ? svld1_f32(pg, src + row * srcS + col * srcC) : svdup_n_f32(0.0f);
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetInput(const float* src, size_t srcS, size_t srcC, size_t rowB, size_t rowE,
            size_t colB, size_t colE, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svfloat32_t s0 = WinogradKernel2x2Block2x2SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 0, colB, colE, pg);
            svfloat32_t s1 = WinogradKernel2x2Block2x2SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 1, colB, colE, pg);
            svfloat32_t s2 = WinogradKernel2x2Block2x2SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 2, colB, colE, pg);
            svfloat32_t s3 = WinogradKernel2x2Block2x2SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 0, colB, colE, pg);
            svfloat32_t s4 = WinogradKernel2x2Block2x2SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 1, colB, colE, pg);
            svfloat32_t s5 = WinogradKernel2x2Block2x2SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 2, colB, colE, pg);
            svfloat32_t s6 = WinogradKernel2x2Block2x2SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 0, colB, colE, pg);
            svfloat32_t s7 = WinogradKernel2x2Block2x2SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 1, colB, colE, pg);
            svfloat32_t s8 = WinogradKernel2x2Block2x2SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 2, colB, colE, pg);
            WinogradKernel2x2Block2x2SetInputStore(s0, s1, s2, s3, s4, s5, s6, s7, s8, dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetInput(const float* src, size_t srcW, size_t srcC, size_t rowB, size_t rowE,
            size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            const size_t F = svcntw();
            const size_t srcS = srcW * srcC;
            const size_t srcCF = AlignLo(srcC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < srcCF; c += F)
                WinogradKernel2x2Block2x2SetInput(src + c, srcS, srcC, rowB, rowE, colB, colE, dst + c, dstStride, body);
            if (c < srcC)
                WinogradKernel2x2Block2x2SetInput(src + c, srcS, srcC, rowB, rowE, colB, colE, dst + c, dstStride, svwhilelt_b32(c, srcC));
        }

        void WinogradKernel2x2Block2x2SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padY == padX && padW == padH && (padY + padH == 0 || padY + padH == 1));
            if (!trans)
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
                    src -= (srcWidth + 1) * srcChannels;
            }
            size_t tailW = dstW - dstW2 + (padW ? 0 : 1);
            size_t tailH = dstH - dstH2 + (padH ? 0 : 1);
            size_t row = 0, col = 0;
            if (padY)
            {
                if (padX)
                    WinogradKernel2x2Block2x2SetInput(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = startX; col < dstW2; col += 2)
                    WinogradKernel2x2Block2x2SetInput(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 3, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel2x2Block2x2SetInput(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
            }
            for (row = startY; row < dstH2; row += 2)
            {
                if (padX)
                    WinogradKernel2x2Block2x2SetInput(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 3, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = startX; col < dstW2; col += 2)
                    WinogradKernel2x2Block2x2SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel2x2Block2x2SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 3, 0, tailW, dst, dstStride), dst += srcChannels;
            }
            if (row < dstH)
            {
                if (padX)
                    WinogradKernel2x2Block2x2SetInput(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = startX; col < dstW2; col += 2)
                    WinogradKernel2x2Block2x2SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 3, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel2x2Block2x2SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel2x2Block2x2SetOutputLoad9(const float* src, size_t stride, svfloat32_t& d0, svfloat32_t& d1, svfloat32_t& d2, svfloat32_t& d3, const svbool_t& pg)
        {
            svfloat32_t s0 = svld1_f32(pg, src + 0 * stride);
            svfloat32_t s1 = svld1_f32(pg, src + 1 * stride);
            svfloat32_t s2 = svld1_f32(pg, src + 2 * stride);
            svfloat32_t s3 = svld1_f32(pg, src + 3 * stride);
            svfloat32_t s4 = svld1_f32(pg, src + 4 * stride);
            svfloat32_t s5 = svld1_f32(pg, src + 5 * stride);
            svfloat32_t s6 = svld1_f32(pg, src + 6 * stride);
            svfloat32_t s7 = svld1_f32(pg, src + 7 * stride);
            svfloat32_t s8 = svld1_f32(pg, src + 8 * stride);
            d0 = svadd_f32_x(pg, svadd_f32_x(pg, s0, s1), svadd_f32_x(pg, s3, s4));
            d1 = svadd_f32_x(pg, svadd_f32_x(pg, s1, s2), svadd_f32_x(pg, s4, s5));
            d2 = svadd_f32_x(pg, svadd_f32_x(pg, s3, s4), svadd_f32_x(pg, s6, s7));
            d3 = svadd_f32_x(pg, svadd_f32_x(pg, s4, s5), svadd_f32_x(pg, s7, s8));
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetOutputStore(const svfloat32_t& d0, const svfloat32_t& d1, const svfloat32_t& d2, const svfloat32_t& d3, float* dst, size_t dstS, size_t dstC, const svbool_t& pg)
        {
            svst1_f32(pg, dst + 0 * dstS + 0 * dstC, d0);
            svst1_f32(pg, dst + 0 * dstS + 1 * dstC, d1);
            svst1_f32(pg, dst + 1 * dstS + 0 * dstC, d2);
            svst1_f32(pg, dst + 1 * dstS + 1 * dstC, d3);
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetOutput(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC)
        {
            const size_t F = svcntw();
            const size_t dstS = dstW * dstC;
            const size_t dstCF = AlignLo(dstC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < dstCF; c += F)
            {
                svfloat32_t d0, d1, d2, d3;
                WinogradKernel2x2Block2x2SetOutputLoad9(src + c, srcStride, d0, d1, d2, d3, body);
                WinogradKernel2x2Block2x2SetOutputStore(d0, d1, d2, d3, dst + c, dstS, dstC, body);
            }
            if (c < dstC)
            {
                svfloat32_t d0, d1, d2, d3;
                svbool_t tail = svwhilelt_b32(c, dstC);
                WinogradKernel2x2Block2x2SetOutputLoad9(src + c, srcStride, d0, d1, d2, d3, tail);
                WinogradKernel2x2Block2x2SetOutputStore(d0, d1, d2, d3, dst + c, dstS, dstC, tail);
            }
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetOutputStore(const svfloat32_t& d0, const svfloat32_t& d1, const svfloat32_t& d2, const svfloat32_t& d3, float* dst, size_t dstS, size_t dstC, size_t rowE, size_t colE, const svbool_t& pg)
        {
            if (0 < rowE && 0 < colE)
                svst1_f32(pg, dst + 0 * dstS + 0 * dstC, d0);
            if (0 < rowE && 1 < colE)
                svst1_f32(pg, dst + 0 * dstS + 1 * dstC, d1);
            if (1 < rowE && 0 < colE)
                svst1_f32(pg, dst + 1 * dstS + 0 * dstC, d2);
            if (1 < rowE && 1 < colE)
                svst1_f32(pg, dst + 1 * dstS + 1 * dstC, d3);
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetOutput(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC, size_t rowE, size_t colE)
        {
            const size_t F = svcntw();
            const size_t dstS = dstW * dstC;
            const size_t dstCF = AlignLo(dstC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < dstCF; c += F)
            {
                svfloat32_t d0, d1, d2, d3;
                WinogradKernel2x2Block2x2SetOutputLoad9(src + c, srcStride, d0, d1, d2, d3, body);
                WinogradKernel2x2Block2x2SetOutputStore(d0, d1, d2, d3, dst + c, dstS, dstC, rowE, colE, body);
            }
            if (c < dstC)
            {
                svfloat32_t d0, d1, d2, d3;
                svbool_t tail = svwhilelt_b32(c, dstC);
                WinogradKernel2x2Block2x2SetOutputLoad9(src + c, srcStride, d0, d1, d2, d3, tail);
                WinogradKernel2x2Block2x2SetOutputStore(d0, d1, d2, d3, dst + c, dstS, dstC, rowE, colE, tail);
            }
        }

        void WinogradKernel2x2Block2x2SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            if (!trans)
            {
                Base::WinogradKernel2x2Block2x2SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
                return;
            }
            size_t dstH2 = AlignLo(dstHeight, 2);
            size_t dstW2 = AlignLo(dstWidth, 2);
            size_t row, col;
            for (row = 0; row < dstH2; row += 2)
            {
                for (col = 0; col < dstW2; col += 2)
                    WinogradKernel2x2Block2x2SetOutput(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels), src += dstChannels;
                if (col < dstWidth)
                    WinogradKernel2x2Block2x2SetOutput(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, 2, dstWidth - col), src += dstChannels;
            }
            if (row < dstHeight)
            {
                for (col = 0; col < dstW2; col += 2)
                    WinogradKernel2x2Block2x2SetOutput(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row, 2), src += dstChannels;
                if (col < dstWidth)
                    WinogradKernel2x2Block2x2SetOutput(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row, dstWidth - col), src += dstChannels;
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel2x2Block4x4SetFilterRow(const svfloat32_t& t0, const svfloat32_t& t1, float* dst, size_t stride, const svbool_t& pg)
        {
            const svfloat32_t r2 = svdup_n_f32(1.0f / 2.0f);
            const svfloat32_t r3 = svdup_n_f32(1.0f / 3.0f);
            const svfloat32_t r6 = svdup_n_f32(1.0f / 6.0f);
            const svfloat32_t mr2 = svdup_n_f32(-1.0f / 2.0f);

            svst1_f32(pg, dst + 0 * stride, svmul_f32_x(pg, r2, t0));
            svst1_f32(pg, dst + 1 * stride, svmul_f32_x(pg, mr2, svadd_f32_x(pg, t0, t1)));
            svst1_f32(pg, dst + 2 * stride, svmul_f32_x(pg, r6, svsub_f32_x(pg, t1, t0)));
            svst1_f32(pg, dst + 3 * stride, svadd_f32_x(pg, svmul_f32_x(pg, r6, t0), svmul_f32_x(pg, r3, t1)));
            svst1_f32(pg, dst + 4 * stride, t1);
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetFilter(const svfloat32_t& src0, const svfloat32_t& src1,
            const svfloat32_t& src2, const svfloat32_t& src3, float* dst, size_t stride, const svbool_t& pg)
        {
            const svfloat32_t r2 = svdup_n_f32(1.0f / 2.0f);
            const svfloat32_t r3 = svdup_n_f32(1.0f / 3.0f);
            const svfloat32_t r6 = svdup_n_f32(1.0f / 6.0f);
            const svfloat32_t mr2 = svdup_n_f32(-1.0f / 2.0f);

            svfloat32_t t0 = svmul_f32_x(pg, r2, src0);
            svfloat32_t t1 = svmul_f32_x(pg, r2, src1);
            WinogradKernel2x2Block4x4SetFilterRow(t0, t1, dst + 0 * stride, stride, pg);

            t0 = svmul_f32_x(pg, mr2, svadd_f32_x(pg, src0, src2));
            t1 = svmul_f32_x(pg, mr2, svadd_f32_x(pg, src1, src3));
            WinogradKernel2x2Block4x4SetFilterRow(t0, t1, dst + 5 * stride, stride, pg);

            t0 = svmul_f32_x(pg, r6, svsub_f32_x(pg, src2, src0));
            t1 = svmul_f32_x(pg, r6, svsub_f32_x(pg, src3, src1));
            WinogradKernel2x2Block4x4SetFilterRow(t0, t1, dst + 10 * stride, stride, pg);

            t0 = svadd_f32_x(pg, svmul_f32_x(pg, r6, src0), svmul_f32_x(pg, r3, src2));
            t1 = svadd_f32_x(pg, svmul_f32_x(pg, r6, src1), svmul_f32_x(pg, r3, src3));
            WinogradKernel2x2Block4x4SetFilterRow(t0, t1, dst + 15 * stride, stride, pg);

            WinogradKernel2x2Block4x4SetFilterRow(src2, src3, dst + 20 * stride, stride, pg);
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetFilterVt(const float* src, size_t srcStride, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svfloat32_t src0 = svld1_f32(pg, src + 0 * srcStride);
            svfloat32_t src1 = svld1_f32(pg, src + 1 * srcStride);
            svfloat32_t src2 = svld1_f32(pg, src + 2 * srcStride);
            svfloat32_t src3 = svld1_f32(pg, src + 3 * srcStride);
            WinogradKernel2x2Block4x4SetFilter(src0, src1, src2, src3, dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetFilterVn(const float* src, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svuint32_t offsets = svindex_u32(0, 4);
            svfloat32_t src0 = svld1_gather_u32index_f32(pg, src + 0, offsets);
            svfloat32_t src1 = svld1_gather_u32index_f32(pg, src + 1, offsets);
            svfloat32_t src2 = svld1_gather_u32index_f32(pg, src + 2, offsets);
            svfloat32_t src3 = svld1_gather_u32index_f32(pg, src + 3, offsets);
            WinogradKernel2x2Block4x4SetFilter(src0, src1, src2, src3, dst, dstStride, pg);
        }

        void WinogradKernel2x2Block4x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            const size_t F = svcntw();
            const size_t sizeF = AlignLo(size, F);
            const svbool_t body = svptrue_b32();
            size_t i = 0;
            if (trans)
            {
                for (; i < sizeF; i += F)
                    WinogradKernel2x2Block4x4SetFilterVt(src + i, size, dst + i, size, body);
                if (i < size)
                    WinogradKernel2x2Block4x4SetFilterVt(src + i, size, dst + i, size, svwhilelt_b32(i, size));
            }
            else
            {
                for (; i < sizeF; i += F, src += 4 * F, dst += F)
                    WinogradKernel2x2Block4x4SetFilterVn(src, dst, size, body);
                if (i < size)
                    WinogradKernel2x2Block4x4SetFilterVn(src, dst, size, svwhilelt_b32(i, size));
            }
        }
    }
#endif
}
