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
        SIMD_INLINE void WinogradKernel3x3Block2x2SetFilter(const svfloat32_t& s0, const svfloat32_t& s1, const svfloat32_t& s2,
            const svfloat32_t& s3, const svfloat32_t& s4, const svfloat32_t& s5, const svfloat32_t& s6, const svfloat32_t& s7,
            const svfloat32_t& s8, float* dst, size_t stride, const svbool_t& pg)
        {
            const svfloat32_t r2 = svdup_n_f32(1.0f / 2.0f);
            const svfloat32_t r4 = svdup_n_f32(1.0f / 4.0f);

            svst1_f32(pg, dst + 0 * stride, s0);
            svfloat32_t a02 = svadd_f32_x(pg, s0, s2);
            svst1_f32(pg, dst + 1 * stride, svmul_f32_x(pg, svadd_f32_x(pg, a02, s1), r2));
            svst1_f32(pg, dst + 2 * stride, svmul_f32_x(pg, svsub_f32_x(pg, a02, s1), r2));
            svst1_f32(pg, dst + 3 * stride, s2);

            svfloat32_t a063 = svadd_f32_x(pg, svadd_f32_x(pg, s0, s6), s3);
            svst1_f32(pg, dst + 4 * stride, svmul_f32_x(pg, a063, r2));
            svfloat32_t a285 = svadd_f32_x(pg, svadd_f32_x(pg, s2, s8), s5);
            svfloat32_t a174 = svadd_f32_x(pg, svadd_f32_x(pg, s1, s7), s4);
            svst1_f32(pg, dst + 5 * stride, svmul_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, a063, a285), a174), r4));
            svst1_f32(pg, dst + 6 * stride, svmul_f32_x(pg, svsub_f32_x(pg, svadd_f32_x(pg, a063, a285), a174), r4));
            svst1_f32(pg, dst + 7 * stride, svmul_f32_x(pg, a285, r2));

            svfloat32_t a06m3 = svsub_f32_x(pg, svadd_f32_x(pg, s0, s6), s3);
            svst1_f32(pg, dst + 8 * stride, svmul_f32_x(pg, a06m3, r2));
            svfloat32_t a28m5 = svsub_f32_x(pg, svadd_f32_x(pg, s2, s8), s5);
            svfloat32_t a17m4 = svsub_f32_x(pg, svadd_f32_x(pg, s1, s7), s4);
            svst1_f32(pg, dst + 9 * stride, svmul_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, a06m3, a28m5), a17m4), r4));
            svst1_f32(pg, dst + 10 * stride, svmul_f32_x(pg, svsub_f32_x(pg, svadd_f32_x(pg, a06m3, a28m5), a17m4), r4));
            svst1_f32(pg, dst + 11 * stride, svmul_f32_x(pg, a28m5, r2));

            svst1_f32(pg, dst + 12 * stride, s6);
            svfloat32_t a68 = svadd_f32_x(pg, s6, s8);
            svst1_f32(pg, dst + 13 * stride, svmul_f32_x(pg, svadd_f32_x(pg, a68, s7), r2));
            svst1_f32(pg, dst + 14 * stride, svmul_f32_x(pg, svsub_f32_x(pg, a68, s7), r2));
            svst1_f32(pg, dst + 15 * stride, s8);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetFilterVt(const float* src, size_t srcStride, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svfloat32_t s0 = svld1_f32(pg, src + 0 * srcStride);
            svfloat32_t s1 = svld1_f32(pg, src + 1 * srcStride);
            svfloat32_t s2 = svld1_f32(pg, src + 2 * srcStride);
            svfloat32_t s3 = svld1_f32(pg, src + 3 * srcStride);
            svfloat32_t s4 = svld1_f32(pg, src + 4 * srcStride);
            svfloat32_t s5 = svld1_f32(pg, src + 5 * srcStride);
            svfloat32_t s6 = svld1_f32(pg, src + 6 * srcStride);
            svfloat32_t s7 = svld1_f32(pg, src + 7 * srcStride);
            svfloat32_t s8 = svld1_f32(pg, src + 8 * srcStride);
            WinogradKernel3x3Block2x2SetFilter(s0, s1, s2, s3, s4, s5, s6, s7, s8, dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetFilterVn(const float* src, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svuint32_t offsets = svindex_u32(0, 9);
            svfloat32_t s0 = svld1_gather_u32index_f32(pg, src + 0, offsets);
            svfloat32_t s1 = svld1_gather_u32index_f32(pg, src + 1, offsets);
            svfloat32_t s2 = svld1_gather_u32index_f32(pg, src + 2, offsets);
            svfloat32_t s3 = svld1_gather_u32index_f32(pg, src + 3, offsets);
            svfloat32_t s4 = svld1_gather_u32index_f32(pg, src + 4, offsets);
            svfloat32_t s5 = svld1_gather_u32index_f32(pg, src + 5, offsets);
            svfloat32_t s6 = svld1_gather_u32index_f32(pg, src + 6, offsets);
            svfloat32_t s7 = svld1_gather_u32index_f32(pg, src + 7, offsets);
            svfloat32_t s8 = svld1_gather_u32index_f32(pg, src + 8, offsets);
            WinogradKernel3x3Block2x2SetFilter(s0, s1, s2, s3, s4, s5, s6, s7, s8, dst, dstStride, pg);
        }

        void WinogradKernel3x3Block2x2SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            const size_t F = svcntw();
            const size_t sizeF = AlignLo(size, F);
            const svbool_t body = svptrue_b32();
            size_t i = 0;
            if (trans)
            {
                for (; i < sizeF; i += F)
                    WinogradKernel3x3Block2x2SetFilterVt(src + i, size, dst + i, size, body);
                if (i < size)
                    WinogradKernel3x3Block2x2SetFilterVt(src + i, size, dst + i, size, svwhilelt_b32(i, size));
            }
            else
            {
                for (; i < sizeF; i += F, src += 9 * F, dst += F)
                    WinogradKernel3x3Block2x2SetFilterVn(src, dst, size, body);
                if (i < size)
                    WinogradKernel3x3Block2x2SetFilterVn(src, dst, size, svwhilelt_b32(i, size));
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInputStore(
            const svfloat32_t& s0, const svfloat32_t& s1, const svfloat32_t& s2, const svfloat32_t& s3,
            const svfloat32_t& s4, const svfloat32_t& s5, const svfloat32_t& s6, const svfloat32_t& s7,
            const svfloat32_t& s8, const svfloat32_t& s9, const svfloat32_t& s10, const svfloat32_t& s11,
            const svfloat32_t& s12, const svfloat32_t& s13, const svfloat32_t& s14, const svfloat32_t& s15,
            float* dst, size_t stride, const svbool_t& pg)
        {
            svst1_f32(pg, dst + 0 * stride, svsub_f32_x(pg, svsub_f32_x(pg, s0, s8), svsub_f32_x(pg, s2, s10)));
            svst1_f32(pg, dst + 1 * stride, svadd_f32_x(pg, svsub_f32_x(pg, s1, s9), svsub_f32_x(pg, s2, s10)));
            svst1_f32(pg, dst + 2 * stride, svsub_f32_x(pg, svsub_f32_x(pg, s2, s10), svsub_f32_x(pg, s1, s9)));
            svst1_f32(pg, dst + 3 * stride, svsub_f32_x(pg, svsub_f32_x(pg, s1, s9), svsub_f32_x(pg, s3, s11)));
            svst1_f32(pg, dst + 4 * stride, svsub_f32_x(pg, svadd_f32_x(pg, s4, s8), svadd_f32_x(pg, s6, s10)));
            svst1_f32(pg, dst + 5 * stride, svadd_f32_x(pg, svadd_f32_x(pg, s5, s9), svadd_f32_x(pg, s6, s10)));
            svst1_f32(pg, dst + 6 * stride, svsub_f32_x(pg, svadd_f32_x(pg, s6, s10), svadd_f32_x(pg, s5, s9)));
            svst1_f32(pg, dst + 7 * stride, svsub_f32_x(pg, svadd_f32_x(pg, s5, s9), svadd_f32_x(pg, s7, s11)));
            svst1_f32(pg, dst + 8 * stride, svsub_f32_x(pg, svsub_f32_x(pg, s8, s4), svsub_f32_x(pg, s10, s6)));
            svst1_f32(pg, dst + 9 * stride, svadd_f32_x(pg, svsub_f32_x(pg, s9, s5), svsub_f32_x(pg, s10, s6)));
            svst1_f32(pg, dst + 10 * stride, svsub_f32_x(pg, svsub_f32_x(pg, s10, s6), svsub_f32_x(pg, s9, s5)));
            svst1_f32(pg, dst + 11 * stride, svsub_f32_x(pg, svsub_f32_x(pg, s9, s5), svsub_f32_x(pg, s11, s7)));
            svst1_f32(pg, dst + 12 * stride, svsub_f32_x(pg, svsub_f32_x(pg, s4, s12), svsub_f32_x(pg, s6, s14)));
            svst1_f32(pg, dst + 13 * stride, svadd_f32_x(pg, svsub_f32_x(pg, s5, s13), svsub_f32_x(pg, s6, s14)));
            svst1_f32(pg, dst + 14 * stride, svsub_f32_x(pg, svsub_f32_x(pg, s6, s14), svsub_f32_x(pg, s5, s13)));
            svst1_f32(pg, dst + 15 * stride, svsub_f32_x(pg, svsub_f32_x(pg, s5, s13), svsub_f32_x(pg, s7, s15)));
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput(const float* src, size_t srcS, size_t srcC, float* dst, size_t dstStride, const svbool_t& pg)
        {
            WinogradKernel3x3Block2x2SetInputStore(
                svld1_f32(pg, src + 0 * srcS + 0 * srcC), svld1_f32(pg, src + 0 * srcS + 1 * srcC), svld1_f32(pg, src + 0 * srcS + 2 * srcC), svld1_f32(pg, src + 0 * srcS + 3 * srcC),
                svld1_f32(pg, src + 1 * srcS + 0 * srcC), svld1_f32(pg, src + 1 * srcS + 1 * srcC), svld1_f32(pg, src + 1 * srcS + 2 * srcC), svld1_f32(pg, src + 1 * srcS + 3 * srcC),
                svld1_f32(pg, src + 2 * srcS + 0 * srcC), svld1_f32(pg, src + 2 * srcS + 1 * srcC), svld1_f32(pg, src + 2 * srcS + 2 * srcC), svld1_f32(pg, src + 2 * srcS + 3 * srcC),
                svld1_f32(pg, src + 3 * srcS + 0 * srcC), svld1_f32(pg, src + 3 * srcS + 1 * srcC), svld1_f32(pg, src + 3 * srcS + 2 * srcC), svld1_f32(pg, src + 3 * srcS + 3 * srcC),
                dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput(const float* src, size_t srcW, size_t srcC, float* dst, size_t dstStride)
        {
            const size_t F = svcntw();
            const size_t srcS = srcW * srcC;
            const size_t srcCF = AlignLo(srcC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < srcCF; c += F)
                WinogradKernel3x3Block2x2SetInput(src + c, srcS, srcC, dst + c, dstStride, body);
            if (c < srcC)
                WinogradKernel3x3Block2x2SetInput(src + c, srcS, srcC, dst + c, dstStride, svwhilelt_b32(c, srcC));
        }

        SIMD_INLINE svfloat32_t WinogradKernel3x3Block2x2SetInputLoad(const float* src, size_t srcS, size_t srcC, size_t row,
            size_t rowB, size_t rowE, size_t col, size_t colB, size_t colE, const svbool_t& pg)
        {
            return row >= rowB && row < rowE && col >= colB && col < colE ? svld1_f32(pg, src + row * srcS + col * srcC) : svdup_n_f32(0.0f);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput(const float* src, size_t srcS, size_t srcC, size_t rowB, size_t rowE,
            size_t colB, size_t colE, float* dst, size_t dstStride, const svbool_t& pg)
        {
            WinogradKernel3x3Block2x2SetInputStore(
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 0, colB, colE, pg),
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 1, colB, colE, pg),
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 2, colB, colE, pg),
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 3, colB, colE, pg),
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 0, colB, colE, pg),
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 1, colB, colE, pg),
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 2, colB, colE, pg),
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 3, colB, colE, pg),
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 0, colB, colE, pg),
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 1, colB, colE, pg),
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 2, colB, colE, pg),
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 3, colB, colE, pg),
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 3, rowB, rowE, 0, colB, colE, pg),
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 3, rowB, rowE, 1, colB, colE, pg),
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 3, rowB, rowE, 2, colB, colE, pg),
                WinogradKernel3x3Block2x2SetInputLoad(src, srcS, srcC, 3, rowB, rowE, 3, colB, colE, pg),
                dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput(const float* src, size_t srcW, size_t srcC, size_t rowB, size_t rowE,
            size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            const size_t F = svcntw();
            const size_t srcS = srcW * srcC;
            const size_t srcCF = AlignLo(srcC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < srcCF; c += F)
                WinogradKernel3x3Block2x2SetInput(src + c, srcS, srcC, rowB, rowE, colB, colE, dst + c, dstStride, body);
            if (c < srcC)
                WinogradKernel3x3Block2x2SetInput(src + c, srcS, srcC, rowB, rowE, colB, colE, dst + c, dstStride, svwhilelt_b32(c, srcC));
        }

        void WinogradKernel3x3Block2x2SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padY == padX && padY == padH && padY == padW && (padY == 0 || padY == 1));
            if (!trans)
            {
                Base::WinogradKernel3x3Block2x2SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
                return;
            }
            SimdBool pad = padY > 0 ? SimdTrue : SimdFalse;
            size_t dstH = pad ? srcHeight : srcHeight - 2;
            size_t dstW = pad ? srcWidth : srcWidth - 2;
            size_t dstH2 = AlignLo(dstH, 2);
            size_t dstW2 = AlignLo(dstW, 2);
            size_t noseW = Simd::Min<size_t>(4, dstW + 1);
            size_t noseH = Simd::Min<size_t>(4, dstH + 1);
            size_t start = pad ? 2 : 0;
            if (pad)
            {
                if (dstH == dstH2)
                    dstH2 -= 2;
                if (dstW == dstW2)
                    dstW2 -= 2;
                src -= (srcWidth + 1) * srcChannels;
            }
            size_t tailW = dstW - dstW2 + (pad ? 1 : 2);
            size_t tailH = dstH - dstH2 + (pad ? 1 : 2);
            size_t row = 0, col = 0;
            if (pad)
            {
                WinogradKernel3x3Block2x2SetInput(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = start; col < dstW2; col += 2)
                    WinogradKernel3x3Block2x2SetInput(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 4, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel3x3Block2x2SetInput(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
            }
            for (row = start; row < dstH2; row += 2)
            {
                if (pad)
                    WinogradKernel3x3Block2x2SetInput(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 4, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = start; col < dstW2; col += 2)
                    WinogradKernel3x3Block2x2SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel3x3Block2x2SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 4, 0, tailW, dst, dstStride), dst += srcChannels;
            }
            if (row < dstH)
            {
                if (pad)
                    WinogradKernel3x3Block2x2SetInput(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = start; col < dstW2; col += 2)
                    WinogradKernel3x3Block2x2SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 4, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel3x3Block2x2SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
            }
        }
    }
#endif
}
