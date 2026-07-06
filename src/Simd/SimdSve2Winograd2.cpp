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
    }
#endif
}
