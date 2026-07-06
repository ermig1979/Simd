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
        SIMD_INLINE void WinogradKernel1x3Block1x4SetFilter(svfloat32_t s0, svfloat32_t s1, svfloat32_t s2, float* dst, size_t stride, const svbool_t& pg)
        {
            const svfloat32_t r4 = svdup_n_f32(1.0f / 4.0f);
            const svfloat32_t mr6 = svdup_n_f32(-1.0f / 6.0f);
            const svfloat32_t r12 = svdup_n_f32(1.0f / 12.0f);
            const svfloat32_t r24 = svdup_n_f32(1.0f / 24.0f);

            svst1_f32(pg, dst + 0 * stride, svmul_f32_x(pg, r4, s0));
            svfloat32_t s02 = svadd_f32_x(pg, s0, s2);
            svst1_f32(pg, dst + 1 * stride, svmul_f32_x(pg, mr6, svadd_f32_x(pg, s02, s1)));
            svst1_f32(pg, dst + 2 * stride, svmul_f32_x(pg, mr6, svsub_f32_x(pg, s02, s1)));
            svfloat32_t b = svmul_f32_x(pg, svdup_n_f32(1.0f / 6.0f), s2);
            svfloat32_t d3 = svadd_f32_x(pg, svadd_f32_x(pg, svmul_f32_x(pg, r24, s0), svmul_f32_x(pg, r12, s1)), b);
            svfloat32_t d4 = svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, r24, s0), svmul_f32_x(pg, r12, s1)), b);
            svst1_f32(pg, dst + 3 * stride, d3);
            svst1_f32(pg, dst + 4 * stride, d4);
            svst1_f32(pg, dst + 5 * stride, s2);
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetFilterVt(const float* src, size_t srcStride, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svfloat32_t s0 = svld1_f32(pg, src + 0 * srcStride);
            svfloat32_t s1 = svld1_f32(pg, src + 1 * srcStride);
            svfloat32_t s2 = svld1_f32(pg, src + 2 * srcStride);
            WinogradKernel1x3Block1x4SetFilter(s0, s1, s2, dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetFilterVn(const float* src, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svuint32_t offsets = svindex_u32(0, 3);
            svfloat32_t s0 = svld1_gather_u32index_f32(pg, src + 0, offsets);
            svfloat32_t s1 = svld1_gather_u32index_f32(pg, src + 1, offsets);
            svfloat32_t s2 = svld1_gather_u32index_f32(pg, src + 2, offsets);
            WinogradKernel1x3Block1x4SetFilter(s0, s1, s2, dst, dstStride, pg);
        }

        void WinogradKernel1x3Block1x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            const size_t F = svcntw();
            const size_t sizeF = AlignLo(size, F);
            const svbool_t body = svptrue_b32();
            size_t i = 0;
            if (trans)
            {
                for (; i < sizeF; i += F)
                    WinogradKernel1x3Block1x4SetFilterVt(src + i, size, dst + i, size, body);
                if (i < size)
                    WinogradKernel1x3Block1x4SetFilterVt(src + i, size, dst + i, size, svwhilelt_b32(i, size));
            }
            else
            {
                for (; i < sizeF; i += F, src += 3 * F, dst += F)
                    WinogradKernel1x3Block1x4SetFilterVn(src, dst, size, body);
                if (i < size)
                    WinogradKernel1x3Block1x4SetFilterVn(src, dst, size, svwhilelt_b32(i, size));
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel1x3Block1x4SetInputStore(const svfloat32_t& s0, const svfloat32_t& s1, const svfloat32_t& s2,
            const svfloat32_t& s3, const svfloat32_t& s4, const svfloat32_t& s5, float* dst, size_t stride, const svbool_t& pg)
        {
            const svfloat32_t _2 = svdup_n_f32(2.0f);
            const svfloat32_t _4 = svdup_n_f32(4.0f);
            const svfloat32_t _5 = svdup_n_f32(5.0f);
            svst1_f32(pg, dst + 0 * stride, svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, _4, s0), svmul_f32_x(pg, _5, s2)), s4));
            svst1_f32(pg, dst + 1 * stride, svsub_f32_x(pg, svadd_f32_x(pg, s3, s4), svmul_f32_x(pg, _4, svadd_f32_x(pg, s1, s2))));
            svst1_f32(pg, dst + 2 * stride, svadd_f32_x(pg, svmul_f32_x(pg, _4, svsub_f32_x(pg, s1, s2)), svsub_f32_x(pg, s4, s3)));
            svst1_f32(pg, dst + 3 * stride, svadd_f32_x(pg, svmul_f32_x(pg, _2, svsub_f32_x(pg, s3, s1)), svsub_f32_x(pg, s4, s2)));
            svst1_f32(pg, dst + 4 * stride, svadd_f32_x(pg, svmul_f32_x(pg, _2, svsub_f32_x(pg, s1, s3)), svsub_f32_x(pg, s4, s2)));
            svst1_f32(pg, dst + 5 * stride, svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, _4, s1), svmul_f32_x(pg, _5, s3)), s5));
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetInput(const float* src, size_t srcC, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svfloat32_t s0 = svld1_f32(pg, src + 0 * srcC);
            svfloat32_t s1 = svld1_f32(pg, src + 1 * srcC);
            svfloat32_t s2 = svld1_f32(pg, src + 2 * srcC);
            svfloat32_t s3 = svld1_f32(pg, src + 3 * srcC);
            svfloat32_t s4 = svld1_f32(pg, src + 4 * srcC);
            svfloat32_t s5 = svld1_f32(pg, src + 5 * srcC);
            WinogradKernel1x3Block1x4SetInputStore(s0, s1, s2, s3, s4, s5, dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetInput(const float* src, size_t srcC, float* dst, size_t dstStride)
        {
            const size_t F = svcntw();
            const size_t srcCF = AlignLo(srcC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < srcCF; c += F)
                WinogradKernel1x3Block1x4SetInput(src + c, srcC, dst + c, dstStride, body);
            if (c < srcC)
                WinogradKernel1x3Block1x4SetInput(src + c, srcC, dst + c, dstStride, svwhilelt_b32(c, srcC));
        }

        SIMD_INLINE svfloat32_t WinogradKernel1x3Block1x4SetInputLoad(const float* src, size_t srcC, size_t col, size_t colB, size_t colE, const svbool_t& pg)
        {
            return col >= colB && col < colE ? svld1_f32(pg, src + col * srcC) : svdup_n_f32(0.0f);
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetInput(const float* src, size_t srcC, size_t colB, size_t colE, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svfloat32_t s0 = WinogradKernel1x3Block1x4SetInputLoad(src, srcC, 0, colB, colE, pg);
            svfloat32_t s1 = WinogradKernel1x3Block1x4SetInputLoad(src, srcC, 1, colB, colE, pg);
            svfloat32_t s2 = WinogradKernel1x3Block1x4SetInputLoad(src, srcC, 2, colB, colE, pg);
            svfloat32_t s3 = WinogradKernel1x3Block1x4SetInputLoad(src, srcC, 3, colB, colE, pg);
            svfloat32_t s4 = WinogradKernel1x3Block1x4SetInputLoad(src, srcC, 4, colB, colE, pg);
            svfloat32_t s5 = WinogradKernel1x3Block1x4SetInputLoad(src, srcC, 5, colB, colE, pg);
            WinogradKernel1x3Block1x4SetInputStore(s0, s1, s2, s3, s4, s5, dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetInput(const float* src, size_t srcC, size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            const size_t F = svcntw();
            const size_t srcCF = AlignLo(srcC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < srcCF; c += F)
                WinogradKernel1x3Block1x4SetInput(src + c, srcC, colB, colE, dst + c, dstStride, body);
            if (c < srcC)
                WinogradKernel1x3Block1x4SetInput(src + c, srcC, colB, colE, dst + c, dstStride, svwhilelt_b32(c, srcC));
        }

        void WinogradKernel1x3Block1x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padX == padW && padY == 0 && padH == 0 && (padX == 0 || padX == 1));
            if (!trans)
            {
                Base::WinogradKernel1x3Block1x4SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
                return;
            }
            size_t dstH = srcHeight;
            size_t dstW = padX ? srcWidth : srcWidth - 2;
            size_t dstW4 = AlignLo(dstW, 4);
            size_t noseW = Simd::Min<size_t>(6, dstW + 1);
            size_t startX = padX ? 4 : 0;
            if (padX)
            {
                if (dstW == dstW4)
                    dstW4 -= 4;
                src -= srcChannels;
            }
            size_t tailW = dstW - dstW4 + (padX ? 1 : 2);
            for (size_t row = 0; row < dstH; row += 1)
            {
                size_t col = 0;
                if (padX)
                    WinogradKernel1x3Block1x4SetInput(src, srcChannels, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = startX; col < dstW4; col += 4)
                    WinogradKernel1x3Block1x4SetInput(src + col * srcChannels, srcChannels, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel1x3Block1x4SetInput(src + col * srcChannels, srcChannels, 0, tailW, dst, dstStride), dst += srcChannels;
                src += srcWidth * srcChannels;
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel1x3Block1x4SetOutputStore(const svfloat32_t& s0, const svfloat32_t& s1, const svfloat32_t& s2,
            const svfloat32_t& s3, const svfloat32_t& s4, const svfloat32_t& s5, float* dst, size_t dstC, const svbool_t& pg)
        {
            const svfloat32_t _2 = svdup_n_f32(2.0f);
            const svfloat32_t _4 = svdup_n_f32(4.0f);
            const svfloat32_t _8 = svdup_n_f32(8.0f);
            svst1_f32(pg, dst + 0 * dstC, svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, s0, s1), svadd_f32_x(pg, s2, s3)), s4));
            svst1_f32(pg, dst + 1 * dstC, svadd_f32_x(pg, svsub_f32_x(pg, s1, s2), svmul_f32_x(pg, _2, svsub_f32_x(pg, s3, s4))));
            svst1_f32(pg, dst + 2 * dstC, svadd_f32_x(pg, svadd_f32_x(pg, s1, s2), svmul_f32_x(pg, _4, svadd_f32_x(pg, s3, s4))));
            svst1_f32(pg, dst + 3 * dstC, svadd_f32_x(pg, svadd_f32_x(pg, svsub_f32_x(pg, s1, s2), svmul_f32_x(pg, _8, svsub_f32_x(pg, s3, s4))), s5));
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstC, const svbool_t& pg)
        {
            svfloat32_t s0 = svld1_f32(pg, src + 0 * srcStride);
            svfloat32_t s1 = svld1_f32(pg, src + 1 * srcStride);
            svfloat32_t s2 = svld1_f32(pg, src + 2 * srcStride);
            svfloat32_t s3 = svld1_f32(pg, src + 3 * srcStride);
            svfloat32_t s4 = svld1_f32(pg, src + 4 * srcStride);
            svfloat32_t s5 = svld1_f32(pg, src + 5 * srcStride);
            WinogradKernel1x3Block1x4SetOutputStore(s0, s1, s2, s3, s4, s5, dst, dstC, pg);
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstC)
        {
            const size_t F = svcntw();
            const size_t dstCF = AlignLo(dstC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < dstCF; c += F)
                WinogradKernel1x3Block1x4SetOutput(src + c, srcStride, dst + c, dstC, body);
            if (c < dstC)
                WinogradKernel1x3Block1x4SetOutput(src + c, srcStride, dst + c, dstC, svwhilelt_b32(c, dstC));
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetOutputStore(const svfloat32_t& s0, const svfloat32_t& s1, const svfloat32_t& s2,
            const svfloat32_t& s3, const svfloat32_t& s4, const svfloat32_t& s5, float* dst, size_t dstC, size_t colE, const svbool_t& pg)
        {
            const svfloat32_t _2 = svdup_n_f32(2.0f);
            const svfloat32_t _4 = svdup_n_f32(4.0f);
            const svfloat32_t _8 = svdup_n_f32(8.0f);
            svfloat32_t d0 = svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, s0, s1), svadd_f32_x(pg, s2, s3)), s4);
            svfloat32_t d1 = svadd_f32_x(pg, svsub_f32_x(pg, s1, s2), svmul_f32_x(pg, _2, svsub_f32_x(pg, s3, s4)));
            svfloat32_t d2 = svadd_f32_x(pg, svadd_f32_x(pg, s1, s2), svmul_f32_x(pg, _4, svadd_f32_x(pg, s3, s4)));
            svfloat32_t d3 = svadd_f32_x(pg, svadd_f32_x(pg, svsub_f32_x(pg, s1, s2), svmul_f32_x(pg, _8, svsub_f32_x(pg, s3, s4))), s5);
            if (0 < colE)
                svst1_f32(pg, dst + 0 * dstC, d0);
            if (1 < colE)
                svst1_f32(pg, dst + 1 * dstC, d1);
            if (2 < colE)
                svst1_f32(pg, dst + 2 * dstC, d2);
            if (3 < colE)
                svst1_f32(pg, dst + 3 * dstC, d3);
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstC, size_t colE, const svbool_t& pg)
        {
            svfloat32_t s0 = svld1_f32(pg, src + 0 * srcStride);
            svfloat32_t s1 = svld1_f32(pg, src + 1 * srcStride);
            svfloat32_t s2 = svld1_f32(pg, src + 2 * srcStride);
            svfloat32_t s3 = svld1_f32(pg, src + 3 * srcStride);
            svfloat32_t s4 = svld1_f32(pg, src + 4 * srcStride);
            svfloat32_t s5 = svld1_f32(pg, src + 5 * srcStride);
            WinogradKernel1x3Block1x4SetOutputStore(s0, s1, s2, s3, s4, s5, dst, dstC, colE, pg);
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstC, size_t colE)
        {
            const size_t F = svcntw();
            const size_t dstCF = AlignLo(dstC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < dstCF; c += F)
                WinogradKernel1x3Block1x4SetOutput(src + c, srcStride, dst + c, dstC, colE, body);
            if (c < dstC)
                WinogradKernel1x3Block1x4SetOutput(src + c, srcStride, dst + c, dstC, colE, svwhilelt_b32(c, dstC));
        }

        void WinogradKernel1x3Block1x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            if (!trans)
            {
                Base::WinogradKernel1x3Block1x4SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
                return;
            }
            size_t dstW4 = AlignLo(dstWidth, 4);
            for (size_t row = 0; row < dstHeight; row += 1)
            {
                size_t col = 0;
                for (; col < dstW4; col += 4)
                    WinogradKernel1x3Block1x4SetOutput(src, srcStride, dst + col * dstChannels, dstChannels), src += dstChannels;
                if (col < dstWidth)
                    WinogradKernel1x3Block1x4SetOutput(src, srcStride, dst + col * dstChannels, dstChannels, dstWidth - col), src += dstChannels;
                dst += dstWidth * dstChannels;
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel1x5Block1x4SetFilter(const svfloat32_t& s0, const svfloat32_t& s1, const svfloat32_t& s2,
            const svfloat32_t& s3, const svfloat32_t& s4, float* dst, size_t stride, const svbool_t& pg)
        {
            const svfloat32_t r36 = svdup_n_f32(1.0f / 36.0f);
            const svfloat32_t r48 = svdup_n_f32(1.0f / 48.0f);
            const svfloat32_t mr120 = svdup_n_f32(-1.0f / 120.0f);
            const svfloat32_t r720 = svdup_n_f32(1.0f / 720.0f);
            const svfloat32_t _2 = svdup_n_f32(2.0f);
            const svfloat32_t _3 = svdup_n_f32(3.0f);
            const svfloat32_t _4 = svdup_n_f32(4.0f);
            const svfloat32_t _9 = svdup_n_f32(9.0f);

            svst1_f32(pg, dst + 0 * stride, svmul_f32_x(pg, r36, s0));
            svfloat32_t a0 = svadd_f32_x(pg, svadd_f32_x(pg, s0, s2), s4);
            svfloat32_t a1 = svadd_f32_x(pg, s1, s3);
            svst1_f32(pg, dst + 1 * stride, svmul_f32_x(pg, r48, svadd_f32_x(pg, a0, a1)));
            svst1_f32(pg, dst + 2 * stride, svmul_f32_x(pg, r48, svsub_f32_x(pg, a0, a1)));
            a0 = svadd_f32_x(pg, s0, svmul_f32_x(pg, _4, svadd_f32_x(pg, s2, svmul_f32_x(pg, _4, s4))));
            a1 = svmul_f32_x(pg, _2, svadd_f32_x(pg, s1, svmul_f32_x(pg, _4, s3)));
            svst1_f32(pg, dst + 3 * stride, svmul_f32_x(pg, mr120, svadd_f32_x(pg, a0, a1)));
            svst1_f32(pg, dst + 4 * stride, svmul_f32_x(pg, mr120, svsub_f32_x(pg, a0, a1)));
            a0 = svadd_f32_x(pg, s0, svmul_f32_x(pg, _9, svadd_f32_x(pg, s2, svmul_f32_x(pg, _9, s4))));
            a1 = svmul_f32_x(pg, _3, svadd_f32_x(pg, s1, svmul_f32_x(pg, _9, s3)));
            svst1_f32(pg, dst + 5 * stride, svmul_f32_x(pg, r720, svadd_f32_x(pg, a0, a1)));
            svst1_f32(pg, dst + 6 * stride, svmul_f32_x(pg, r720, svsub_f32_x(pg, a0, a1)));
            svst1_f32(pg, dst + 7 * stride, s4);
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetFilterVt(const float* src, size_t srcStride, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svfloat32_t s0 = svld1_f32(pg, src + 0 * srcStride);
            svfloat32_t s1 = svld1_f32(pg, src + 1 * srcStride);
            svfloat32_t s2 = svld1_f32(pg, src + 2 * srcStride);
            svfloat32_t s3 = svld1_f32(pg, src + 3 * srcStride);
            svfloat32_t s4 = svld1_f32(pg, src + 4 * srcStride);
            WinogradKernel1x5Block1x4SetFilter(s0, s1, s2, s3, s4, dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetFilterVn(const float* src, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svuint32_t offsets = svindex_u32(0, 5);
            svfloat32_t s0 = svld1_gather_u32index_f32(pg, src + 0, offsets);
            svfloat32_t s1 = svld1_gather_u32index_f32(pg, src + 1, offsets);
            svfloat32_t s2 = svld1_gather_u32index_f32(pg, src + 2, offsets);
            svfloat32_t s3 = svld1_gather_u32index_f32(pg, src + 3, offsets);
            svfloat32_t s4 = svld1_gather_u32index_f32(pg, src + 4, offsets);
            WinogradKernel1x5Block1x4SetFilter(s0, s1, s2, s3, s4, dst, dstStride, pg);
        }

        void WinogradKernel1x5Block1x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            const size_t F = svcntw();
            const size_t sizeF = AlignLo(size, F);
            const svbool_t body = svptrue_b32();
            size_t i = 0;
            if (trans)
            {
                for (; i < sizeF; i += F)
                    WinogradKernel1x5Block1x4SetFilterVt(src + i, size, dst + i, size, body);
                if (i < size)
                    WinogradKernel1x5Block1x4SetFilterVt(src + i, size, dst + i, size, svwhilelt_b32(i, size));
            }
            else
            {
                for (; i < sizeF; i += F, src += 5 * F, dst += F)
                    WinogradKernel1x5Block1x4SetFilterVn(src, dst, size, body);
                if (i < size)
                    WinogradKernel1x5Block1x4SetFilterVn(src, dst, size, svwhilelt_b32(i, size));
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel1x5Block1x4SetInputStore(const svfloat32_t& s0, const svfloat32_t& s1, const svfloat32_t& s2,
            const svfloat32_t& s3, const svfloat32_t& s4, const svfloat32_t& s5, const svfloat32_t& s6, const svfloat32_t& s7,
            float* dst, size_t stride, const svbool_t& pg)
        {
            const svfloat32_t _2 = svdup_n_f32(2.0f);
            const svfloat32_t _3 = svdup_n_f32(3.0f);
            const svfloat32_t _4 = svdup_n_f32(4.0f);
            const svfloat32_t _5 = svdup_n_f32(5.0f);
            const svfloat32_t _9 = svdup_n_f32(9.0f);
            const svfloat32_t _10 = svdup_n_f32(10.0f);
            const svfloat32_t _13 = svdup_n_f32(13.0f);
            const svfloat32_t _14 = svdup_n_f32(14.0f);
            const svfloat32_t _36 = svdup_n_f32(36.0f);
            const svfloat32_t _49 = svdup_n_f32(49.0f);

            svst1_f32(pg, dst + 0 * stride, svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, _36, s0), svmul_f32_x(pg, _49, s2)), svsub_f32_x(pg, svmul_f32_x(pg, _14, s4), s6)));
            svfloat32_t a0 = svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, _36, s2), svmul_f32_x(pg, _13, s4)), s6);
            svfloat32_t a1 = svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, _36, s1), svmul_f32_x(pg, _13, s3)), s5);
            svst1_f32(pg, dst + 1 * stride, svadd_f32_x(pg, a0, a1));
            svst1_f32(pg, dst + 2 * stride, svsub_f32_x(pg, a0, a1));
            a0 = svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, _9, s2), svmul_f32_x(pg, _10, s4)), s6);
            a1 = svmul_f32_x(pg, _2, svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, _9, s1), svmul_f32_x(pg, _10, s3)), s5));
            svst1_f32(pg, dst + 3 * stride, svadd_f32_x(pg, a0, a1));
            svst1_f32(pg, dst + 4 * stride, svsub_f32_x(pg, a0, a1));
            a0 = svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, _4, s2), svmul_f32_x(pg, _5, s4)), s6);
            a1 = svmul_f32_x(pg, _3, svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, _4, s1), svmul_f32_x(pg, _5, s3)), s5));
            svst1_f32(pg, dst + 5 * stride, svadd_f32_x(pg, a0, a1));
            svst1_f32(pg, dst + 6 * stride, svsub_f32_x(pg, a0, a1));
            svst1_f32(pg, dst + 7 * stride, svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, _49, s3), svmul_f32_x(pg, _36, s1)), svsub_f32_x(pg, s7, svmul_f32_x(pg, _14, s5))));
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetInput(const float* src, size_t srcC, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svfloat32_t s0 = svld1_f32(pg, src + 0 * srcC);
            svfloat32_t s1 = svld1_f32(pg, src + 1 * srcC);
            svfloat32_t s2 = svld1_f32(pg, src + 2 * srcC);
            svfloat32_t s3 = svld1_f32(pg, src + 3 * srcC);
            svfloat32_t s4 = svld1_f32(pg, src + 4 * srcC);
            svfloat32_t s5 = svld1_f32(pg, src + 5 * srcC);
            svfloat32_t s6 = svld1_f32(pg, src + 6 * srcC);
            svfloat32_t s7 = svld1_f32(pg, src + 7 * srcC);
            WinogradKernel1x5Block1x4SetInputStore(s0, s1, s2, s3, s4, s5, s6, s7, dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetInput(const float* src, size_t srcC, float* dst, size_t dstStride)
        {
            const size_t F = svcntw();
            const size_t srcCF = AlignLo(srcC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < srcCF; c += F)
                WinogradKernel1x5Block1x4SetInput(src + c, srcC, dst + c, dstStride, body);
            if (c < srcC)
                WinogradKernel1x5Block1x4SetInput(src + c, srcC, dst + c, dstStride, svwhilelt_b32(c, srcC));
        }

        SIMD_INLINE svfloat32_t WinogradKernel1x5Block1x4SetInputLoad(const float* src, size_t srcC, size_t col, size_t colB, size_t colE, const svbool_t& pg)
        {
            return col >= colB && col < colE ? svld1_f32(pg, src + col * srcC) : svdup_n_f32(0.0f);
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetInput(const float* src, size_t srcC, size_t colB, size_t colE, float* dst, size_t dstStride, const svbool_t& pg)
        {
            svfloat32_t s0 = WinogradKernel1x5Block1x4SetInputLoad(src, srcC, 0, colB, colE, pg);
            svfloat32_t s1 = WinogradKernel1x5Block1x4SetInputLoad(src, srcC, 1, colB, colE, pg);
            svfloat32_t s2 = WinogradKernel1x5Block1x4SetInputLoad(src, srcC, 2, colB, colE, pg);
            svfloat32_t s3 = WinogradKernel1x5Block1x4SetInputLoad(src, srcC, 3, colB, colE, pg);
            svfloat32_t s4 = WinogradKernel1x5Block1x4SetInputLoad(src, srcC, 4, colB, colE, pg);
            svfloat32_t s5 = WinogradKernel1x5Block1x4SetInputLoad(src, srcC, 5, colB, colE, pg);
            svfloat32_t s6 = WinogradKernel1x5Block1x4SetInputLoad(src, srcC, 6, colB, colE, pg);
            svfloat32_t s7 = WinogradKernel1x5Block1x4SetInputLoad(src, srcC, 7, colB, colE, pg);
            WinogradKernel1x5Block1x4SetInputStore(s0, s1, s2, s3, s4, s5, s6, s7, dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetInput(const float* src, size_t srcC, size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            const size_t F = svcntw();
            const size_t srcCF = AlignLo(srcC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < srcCF; c += F)
                WinogradKernel1x5Block1x4SetInput(src + c, srcC, colB, colE, dst + c, dstStride, body);
            if (c < srcC)
                WinogradKernel1x5Block1x4SetInput(src + c, srcC, colB, colE, dst + c, dstStride, svwhilelt_b32(c, srcC));
        }

        void WinogradKernel1x5Block1x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padX == padW && padY == 0 && padH == 0 && (padX == 0 || padX == 2));
            if (!trans)
            {
                Base::WinogradKernel1x5Block1x4SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
                return;
            }
            size_t dstH = srcHeight;
            size_t dstW = padX ? srcWidth : srcWidth - 4;
            size_t dstW4 = AlignLo(dstW, 4);
            size_t noseW = Simd::Min<size_t>(8, dstW + 2);
            size_t startX = padX ? 4 : 0;
            if (padX)
            {
                if (dstW == dstW4 || dstW == dstW4 + 1)
                    dstW4 -= 4;
                src -= 2 * srcChannels;
            }
            size_t tailW = dstW - dstW4 + (padX ? 2 : 4);
            for (size_t row = 0; row < dstH; row += 1)
            {
                size_t col = 0;
                if (padX)
                    WinogradKernel1x5Block1x4SetInput(src, srcChannels, 2, noseW, dst, dstStride), dst += srcChannels;
                for (col = startX; col < dstW4; col += 4)
                    WinogradKernel1x5Block1x4SetInput(src + col * srcChannels, srcChannels, dst, dstStride), dst += srcChannels;
                for (size_t tail = tailW; col < dstW; col += 4, tail -= 4)
                    WinogradKernel1x5Block1x4SetInput(src + col * srcChannels, srcChannels, 0, tail, dst, dstStride), dst += srcChannels;
                src += srcWidth * srcChannels;
            }
        }
    }
#endif
}
