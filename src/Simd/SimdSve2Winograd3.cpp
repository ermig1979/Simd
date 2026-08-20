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

        SIMD_INLINE void WinogradKernel3x3Block3x3SetFilterRow(const svfloat32_t& t0, const svfloat32_t& t1, const svfloat32_t& t2,
            float* dst, size_t stride, const svbool_t& pg)
        {
            const svfloat32_t r6 = svdup_n_f32(1.0f / 6.0f);
            const svfloat32_t r3 = svdup_n_f32(1.0f / 3.0f);
            const svfloat32_t r2 = svdup_n_f32(1.0f / 2.0f);
            const svfloat32_t f2_3 = svdup_n_f32(2.0f / 3.0f);
            const svfloat32_t mr2 = svdup_n_f32(-1.0f / 2.0f);

            svst1_f32(pg, dst + 0 * stride, svmul_f32_x(pg, r2, t0));
            svfloat32_t sum02 = svadd_f32_x(pg, t0, t2);
            svst1_f32(pg, dst + 1 * stride, svmul_f32_x(pg, mr2, svadd_f32_x(pg, sum02, t1)));
            svst1_f32(pg, dst + 2 * stride, svmul_f32_x(pg, r6, svsub_f32_x(pg, t1, sum02)));
            svst1_f32(pg, dst + 3 * stride, svmla_f32_x(pg, svmla_f32_x(pg, svmul_f32_x(pg, r3, t1), f2_3, t2), r6, t0));
            svst1_f32(pg, dst + 4 * stride, t2);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetFilterAll(
            const svfloat32_t& s0, const svfloat32_t& s1, const svfloat32_t& s2, const svfloat32_t& s3, const svfloat32_t& s4,
            const svfloat32_t& s5, const svfloat32_t& s6, const svfloat32_t& s7, const svfloat32_t& s8,
            float* dst, size_t stride, const svbool_t& pg)
        {
            const svfloat32_t r6 = svdup_n_f32(1.0f / 6.0f);
            const svfloat32_t r3 = svdup_n_f32(1.0f / 3.0f);
            const svfloat32_t r2 = svdup_n_f32(1.0f / 2.0f);
            const svfloat32_t f2_3 = svdup_n_f32(2.0f / 3.0f);
            const svfloat32_t mr2 = svdup_n_f32(-1.0f / 2.0f);

            svfloat32_t t0 = svmul_f32_x(pg, r2, s0);
            svfloat32_t t1 = svmul_f32_x(pg, r2, s1);
            svfloat32_t t2 = svmul_f32_x(pg, r2, s2);
            WinogradKernel3x3Block3x3SetFilterRow(t0, t1, t2, dst + 0 * stride, stride, pg);

            t0 = svmul_f32_x(pg, mr2, svadd_f32_x(pg, svadd_f32_x(pg, s0, s6), s3));
            t1 = svmul_f32_x(pg, mr2, svadd_f32_x(pg, svadd_f32_x(pg, s1, s7), s4));
            t2 = svmul_f32_x(pg, mr2, svadd_f32_x(pg, svadd_f32_x(pg, s2, s8), s5));
            WinogradKernel3x3Block3x3SetFilterRow(t0, t1, t2, dst + 5 * stride, stride, pg);

            t0 = svmul_f32_x(pg, r6, svsub_f32_x(pg, s3, svadd_f32_x(pg, s0, s6)));
            t1 = svmul_f32_x(pg, r6, svsub_f32_x(pg, s4, svadd_f32_x(pg, s1, s7)));
            t2 = svmul_f32_x(pg, r6, svsub_f32_x(pg, s5, svadd_f32_x(pg, s2, s8)));
            WinogradKernel3x3Block3x3SetFilterRow(t0, t1, t2, dst + 10 * stride, stride, pg);

            t0 = svmla_f32_x(pg, svmla_f32_x(pg, svmul_f32_x(pg, r6, s0), r3, s3), f2_3, s6);
            t1 = svmla_f32_x(pg, svmla_f32_x(pg, svmul_f32_x(pg, r6, s1), r3, s4), f2_3, s7);
            t2 = svmla_f32_x(pg, svmla_f32_x(pg, svmul_f32_x(pg, r6, s2), r3, s5), f2_3, s8);
            WinogradKernel3x3Block3x3SetFilterRow(t0, t1, t2, dst + 15 * stride, stride, pg);

            WinogradKernel3x3Block3x3SetFilterRow(s6, s7, s8, dst + 20 * stride, stride, pg);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetFilterVt(const float* src, float* dst, size_t stride, const svbool_t& pg)
        {
            WinogradKernel3x3Block3x3SetFilterAll(
                svld1_f32(pg, src + 0 * stride), svld1_f32(pg, src + 1 * stride), svld1_f32(pg, src + 2 * stride),
                svld1_f32(pg, src + 3 * stride), svld1_f32(pg, src + 4 * stride), svld1_f32(pg, src + 5 * stride),
                svld1_f32(pg, src + 6 * stride), svld1_f32(pg, src + 7 * stride), svld1_f32(pg, src + 8 * stride),
                dst + 0 * stride, stride, pg);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetFilterVn(const float* src, float* dst, size_t stride, const svbool_t& pg)
        {
            svuint32_t offsets = svindex_u32(0, 9);
            WinogradKernel3x3Block3x3SetFilterAll(
                svld1_gather_u32index_f32(pg, src + 0, offsets), svld1_gather_u32index_f32(pg, src + 1, offsets), svld1_gather_u32index_f32(pg, src + 2, offsets),
                svld1_gather_u32index_f32(pg, src + 3, offsets), svld1_gather_u32index_f32(pg, src + 4, offsets), svld1_gather_u32index_f32(pg, src + 5, offsets),
                svld1_gather_u32index_f32(pg, src + 6, offsets), svld1_gather_u32index_f32(pg, src + 7, offsets), svld1_gather_u32index_f32(pg, src + 8, offsets),
                dst + 0 * stride, stride, pg);
        }

        void WinogradKernel3x3Block3x3SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            const size_t F = svcntw();
            const size_t sizeF = AlignLo(size, F);
            const svbool_t body = svptrue_b32();
            size_t i = 0;
            if (trans)
            {
                for (; i < sizeF; i += F)
                    WinogradKernel3x3Block3x3SetFilterVt(src + i, dst + i, size, body);
                if (i < size)
                    WinogradKernel3x3Block3x3SetFilterVt(src + i, dst + i, size, svwhilelt_b32(i, size));
            }
            else
            {
                for (; i < sizeF; i += F, src += 9 * F, dst += F)
                    WinogradKernel3x3Block3x3SetFilterVn(src, dst, size, body);
                if (i < size)
                    WinogradKernel3x3Block3x3SetFilterVn(src, dst, size, svwhilelt_b32(i, size));
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block4x4SetFilterRow(const svfloat32_t& t0, const svfloat32_t& t1, const svfloat32_t& t2,
            float* dst, size_t stride, const svbool_t& pg)
        {
            const svfloat32_t r4 = svdup_n_f32(1.0f / 4.0f);
            const svfloat32_t mr6 = svdup_n_f32(-1.0f / 6.0f);
            const svfloat32_t r6 = svdup_n_f32(1.0f / 6.0f);
            const svfloat32_t r12 = svdup_n_f32(1.0f / 12.0f);
            const svfloat32_t r24 = svdup_n_f32(1.0f / 24.0f);

            svst1_f32(pg, dst + 0 * stride, svmul_f32_x(pg, r4, t0));
            svfloat32_t sum02 = svadd_f32_x(pg, t0, t2);
            svst1_f32(pg, dst + 1 * stride, svmul_f32_x(pg, mr6, svadd_f32_x(pg, sum02, t1)));
            svst1_f32(pg, dst + 2 * stride, svmul_f32_x(pg, mr6, svsub_f32_x(pg, sum02, t1)));
            svfloat32_t term02 = svadd_f32_x(pg, svmul_f32_x(pg, r24, t0), svmul_f32_x(pg, r6, t2));
            svfloat32_t term1 = svmul_f32_x(pg, r12, t1);
            svst1_f32(pg, dst + 3 * stride, svadd_f32_x(pg, term02, term1));
            svst1_f32(pg, dst + 4 * stride, svsub_f32_x(pg, term02, term1));
            svst1_f32(pg, dst + 5 * stride, t2);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetFilterAll(
            const svfloat32_t& s0, const svfloat32_t& s1, const svfloat32_t& s2, const svfloat32_t& s3, const svfloat32_t& s4,
            const svfloat32_t& s5, const svfloat32_t& s6, const svfloat32_t& s7, const svfloat32_t& s8,
            float* dst, size_t stride, const svbool_t& pg)
        {
            const svfloat32_t r4 = svdup_n_f32(1.0f / 4.0f);
            const svfloat32_t mr6 = svdup_n_f32(-1.0f / 6.0f);
            const svfloat32_t r6 = svdup_n_f32(1.0f / 6.0f);
            const svfloat32_t r12 = svdup_n_f32(1.0f / 12.0f);
            const svfloat32_t r24 = svdup_n_f32(1.0f / 24.0f);

            WinogradKernel3x3Block4x4SetFilterRow(
                svmul_f32_x(pg, r4, s0), svmul_f32_x(pg, r4, s1), svmul_f32_x(pg, r4, s2),
                dst + 0 * stride, stride, pg);

            WinogradKernel3x3Block4x4SetFilterRow(
                svmul_f32_x(pg, mr6, svadd_f32_x(pg, svadd_f32_x(pg, s0, s3), s6)),
                svmul_f32_x(pg, mr6, svadd_f32_x(pg, svadd_f32_x(pg, s1, s4), s7)),
                svmul_f32_x(pg, mr6, svadd_f32_x(pg, svadd_f32_x(pg, s2, s5), s8)),
                dst + 6 * stride, stride, pg);

            WinogradKernel3x3Block4x4SetFilterRow(
                svmul_f32_x(pg, mr6, svadd_f32_x(pg, svsub_f32_x(pg, s0, s3), s6)),
                svmul_f32_x(pg, mr6, svadd_f32_x(pg, svsub_f32_x(pg, s1, s4), s7)),
                svmul_f32_x(pg, mr6, svadd_f32_x(pg, svsub_f32_x(pg, s2, s5), s8)),
                dst + 12 * stride, stride, pg);

            WinogradKernel3x3Block4x4SetFilterRow(
                svadd_f32_x(pg, svadd_f32_x(pg, svmul_f32_x(pg, r24, s0), svmul_f32_x(pg, r12, s3)), svmul_f32_x(pg, r6, s6)),
                svadd_f32_x(pg, svadd_f32_x(pg, svmul_f32_x(pg, r24, s1), svmul_f32_x(pg, r12, s4)), svmul_f32_x(pg, r6, s7)),
                svadd_f32_x(pg, svadd_f32_x(pg, svmul_f32_x(pg, r24, s2), svmul_f32_x(pg, r12, s5)), svmul_f32_x(pg, r6, s8)),
                dst + 18 * stride, stride, pg);

            WinogradKernel3x3Block4x4SetFilterRow(
                svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, r24, s0), svmul_f32_x(pg, r12, s3)), svmul_f32_x(pg, r6, s6)),
                svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, r24, s1), svmul_f32_x(pg, r12, s4)), svmul_f32_x(pg, r6, s7)),
                svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, r24, s2), svmul_f32_x(pg, r12, s5)), svmul_f32_x(pg, r6, s8)),
                dst + 24 * stride, stride, pg);

            WinogradKernel3x3Block4x4SetFilterRow(s6, s7, s8, dst + 30 * stride, stride, pg);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetFilterVt(const float* src, float* dst, size_t stride, const svbool_t& pg)
        {
            WinogradKernel3x3Block4x4SetFilterAll(
                svld1_f32(pg, src + 0 * stride), svld1_f32(pg, src + 1 * stride), svld1_f32(pg, src + 2 * stride),
                svld1_f32(pg, src + 3 * stride), svld1_f32(pg, src + 4 * stride), svld1_f32(pg, src + 5 * stride),
                svld1_f32(pg, src + 6 * stride), svld1_f32(pg, src + 7 * stride), svld1_f32(pg, src + 8 * stride),
                dst + 0 * stride, stride, pg);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetFilterVn(const float* src, float* dst, size_t stride, const svbool_t& pg)
        {
            svuint32_t offsets = svindex_u32(0, 9);
            WinogradKernel3x3Block4x4SetFilterAll(
                svld1_gather_u32index_f32(pg, src + 0, offsets), svld1_gather_u32index_f32(pg, src + 1, offsets), svld1_gather_u32index_f32(pg, src + 2, offsets),
                svld1_gather_u32index_f32(pg, src + 3, offsets), svld1_gather_u32index_f32(pg, src + 4, offsets), svld1_gather_u32index_f32(pg, src + 5, offsets),
                svld1_gather_u32index_f32(pg, src + 6, offsets), svld1_gather_u32index_f32(pg, src + 7, offsets), svld1_gather_u32index_f32(pg, src + 8, offsets),
                dst + 0 * stride, stride, pg);
        }

        void WinogradKernel3x3Block4x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            const size_t F = svcntw();
            const size_t sizeF = AlignLo(size, F);
            const svbool_t body = svptrue_b32();
            size_t i = 0;
            if (trans)
            {
                for (; i < sizeF; i += F)
                    WinogradKernel3x3Block4x4SetFilterVt(src + i, dst + i, size, body);
                if (i < size)
                    WinogradKernel3x3Block4x4SetFilterVt(src + i, dst + i, size, svwhilelt_b32(i, size));
            }
            else
            {
                for (; i < sizeF; i += F, src += 9 * F, dst += F)
                    WinogradKernel3x3Block4x4SetFilterVn(src, dst, size, body);
                if (i < size)
                    WinogradKernel3x3Block4x4SetFilterVn(src, dst, size, svwhilelt_b32(i, size));
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

        SIMD_INLINE svbool_t WinogradKernel3x3Block2x2SetInputMaskN(ptrdiff_t count)
        {
            return count <= 0 ? svpfalse_b() : svwhilelt_b32((size_t)0, (size_t)count);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInputLoadN(const float* src,
            svfloat32_t& d0, svfloat32_t& d1, svfloat32_t& d2, svfloat32_t& d3,
            const svbool_t& pg0, const svbool_t& pg1, const svbool_t& pg2, const svbool_t& pg3)
        {
            const size_t F = svcntw();
            svfloat32_t a0 = svld1_f32(pg0, src + 0);
            svfloat32_t a1 = svld1_f32(pg1, src + 2);
            svfloat32_t a2 = svld1_f32(pg2, src + F);
            svfloat32_t a3 = svld1_f32(pg3, src + F + 2);
            d0 = svuzp1_f32(a0, a2);
            d1 = svuzp2_f32(a0, a2);
            d2 = svuzp1_f32(a1, a3);
            d3 = svuzp2_f32(a1, a3);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInputLoadNz(svfloat32_t& d0, svfloat32_t& d1, svfloat32_t& d2, svfloat32_t& d3)
        {
            d0 = svdup_n_f32(0.0f);
            d1 = svdup_n_f32(0.0f);
            d2 = svdup_n_f32(0.0f);
            d3 = svdup_n_f32(0.0f);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInputN(const float* src, size_t srcStride, float* dst, size_t dstStride,
            const svbool_t& pg0, const svbool_t& pg1, const svbool_t& pg2, const svbool_t& pg3, const svbool_t& pgDst)
        {
            svfloat32_t t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
            WinogradKernel3x3Block2x2SetInputLoadN(src + 0 * srcStride, t0, t1, t2, t3, pg0, pg1, pg2, pg3);
            WinogradKernel3x3Block2x2SetInputLoadN(src + 1 * srcStride, t4, t5, t6, t7, pg0, pg1, pg2, pg3);
            WinogradKernel3x3Block2x2SetInputLoadN(src + 2 * srcStride, t8, t9, t10, t11, pg0, pg1, pg2, pg3);
            WinogradKernel3x3Block2x2SetInputLoadN(src + 3 * srcStride, t12, t13, t14, t15, pg0, pg1, pg2, pg3);
            WinogradKernel3x3Block2x2SetInputStore(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, dst, dstStride, pgDst);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInputN(const float* src, size_t srcStride, PadType rowPad, float* dst, size_t dstStride,
            const svbool_t& pg0, const svbool_t& pg1, const svbool_t& pg2, const svbool_t& pg3, const svbool_t& pgDst)
        {
            svfloat32_t t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
            if (rowPad == PadNose1)
                WinogradKernel3x3Block2x2SetInputLoadNz(t0, t1, t2, t3);
            else
                WinogradKernel3x3Block2x2SetInputLoadN(src + 0 * srcStride, t0, t1, t2, t3, pg0, pg1, pg2, pg3);
            WinogradKernel3x3Block2x2SetInputLoadN(src + 1 * srcStride, t4, t5, t6, t7, pg0, pg1, pg2, pg3);
            if (rowPad == PadTail2)
                WinogradKernel3x3Block2x2SetInputLoadNz(t8, t9, t10, t11);
            else
                WinogradKernel3x3Block2x2SetInputLoadN(src + 2 * srcStride, t8, t9, t10, t11, pg0, pg1, pg2, pg3);
            if (rowPad >= PadTail1)
                WinogradKernel3x3Block2x2SetInputLoadNz(t12, t13, t14, t15);
            else
                WinogradKernel3x3Block2x2SetInputLoadN(src + 3 * srcStride, t12, t13, t14, t15, pg0, pg1, pg2, pg3);
            WinogradKernel3x3Block2x2SetInputStore(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, dst, dstStride, pgDst);
        }

        void WinogradKernel3x3Block2x2SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padY == padX && padY == padH && padY == padW && (padY == 0 || padY == 1));
            const size_t F = svcntw();
            const size_t DF = F * 2;
            SimdBool pad = padY > 0 ? SimdTrue : SimdFalse;
            if (!trans && (srcHeight < 4 || srcWidth < 4))
            {
                Base::WinogradKernel3x3Block2x2SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
                return;
            }
            size_t dstH = pad ? srcHeight : srcHeight - 2;
            size_t dstW = pad ? srcWidth : srcWidth - 2;
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
            else
            {
                size_t tileH = (dstH + 1) / 2;
                size_t tileW = (dstW + 1) / 2;
                size_t dstWDF = AlignLo(dstW, DF);
                if (pad && dstWDF == dstW)
                    dstWDF -= DF;
                PadType rowPad = dstH2 < dstH ? PadTail1 : PadNone;
                size_t tailRow = dstH2 < dstH ? dstH - 1 : dstH - 2;
                bool specialRowTail = dstH2 < dstH || (pad && dstH2);
                bool specialColTail = pad ? dstWDF != 0 : dstWDF < dstW;
                ptrdiff_t extra = pad ? 1 : 2;
                const svbool_t all = svptrue_b32();
                svbool_t nose0 = WinogradKernel3x3Block2x2SetInputMaskN((ptrdiff_t)dstW + extra);
                svbool_t nose1 = WinogradKernel3x3Block2x2SetInputMaskN((ptrdiff_t)dstW - 2 + extra);
                svbool_t nose2 = WinogradKernel3x3Block2x2SetInputMaskN((ptrdiff_t)dstW - (ptrdiff_t)F + extra);
                svbool_t nose3 = WinogradKernel3x3Block2x2SetInputMaskN((ptrdiff_t)dstW - (ptrdiff_t)F - 2 + extra);
                svbool_t noseD = WinogradKernel3x3Block2x2SetInputMaskN((ptrdiff_t)tileW);
                svbool_t tail0 = WinogradKernel3x3Block2x2SetInputMaskN((ptrdiff_t)dstW - (ptrdiff_t)dstWDF + extra);
                svbool_t tail1 = WinogradKernel3x3Block2x2SetInputMaskN((ptrdiff_t)dstW - (ptrdiff_t)dstWDF - 2 + extra);
                svbool_t tail2 = WinogradKernel3x3Block2x2SetInputMaskN((ptrdiff_t)dstW - (ptrdiff_t)dstWDF - (ptrdiff_t)F + extra);
                svbool_t tail3 = WinogradKernel3x3Block2x2SetInputMaskN((ptrdiff_t)dstW - (ptrdiff_t)dstWDF - (ptrdiff_t)F - 2 + extra);
                svbool_t tailD = WinogradKernel3x3Block2x2SetInputMaskN((ptrdiff_t)tileW - (ptrdiff_t)(dstWDF / 2));
                if (pad)
                {
                    src -= srcWidth + 1;
                    rowPad = dstH2 < dstH ? PadTail2 : PadTail1;
                    nose0 = svbic_b_z(all, nose0, svwhilelt_b32((size_t)0, (size_t)1));
                    if (dstH2 == dstH)
                        dstH2 -= 2;
                }
                for (size_t c = 0; c < srcChannels; ++c)
                {
                    size_t row = 0, tileY = 0;
                    if (pad)
                    {
                        size_t col = 0, tileX = 0;
                        const float* s = src + row * srcWidth;
                        float* d = dst + tileY * tileW;
                        WinogradKernel3x3Block2x2SetInputN(s + col, srcWidth, PadNose1, d + tileX, dstStride, nose0, nose1, nose2, nose3, noseD);
                        col += DF;
                        tileX += F;
                        for (; col < dstWDF; col += DF, tileX += F)
                            WinogradKernel3x3Block2x2SetInputN(s + col, srcWidth, PadNose1, d + tileX, dstStride, all, all, all, all, all);
                        if (specialColTail)
                            WinogradKernel3x3Block2x2SetInputN(s + col, srcWidth, PadNose1, d + tileX, dstStride, tail0, tail1, tail2, tail3, tailD);
                        row += 2;
                        tileY += 1;
                    }
                    for (; row < dstH2; row += 2, tileY += 1)
                    {
                        size_t col = 0, tileX = 0;
                        const float* s = src + row * srcWidth;
                        float* d = dst + tileY * tileW;
                        if (pad)
                        {
                            WinogradKernel3x3Block2x2SetInputN(s + col, srcWidth, PadNone, d + tileX, dstStride, nose0, nose1, nose2, nose3, noseD);
                            col += DF;
                            tileX += F;
                        }
                        for (; col < dstWDF; col += DF, tileX += F)
                            WinogradKernel3x3Block2x2SetInputN(s + col, srcWidth, d + tileX, dstStride, all, all, all, all, all);
                        if (specialColTail)
                            WinogradKernel3x3Block2x2SetInputN(s + col, srcWidth, PadNone, d + tileX, dstStride, tail0, tail1, tail2, tail3, tailD);
                    }
                    if (specialRowTail)
                    {
                        size_t col = 0, tileX = 0;
                        const float* s = src + tailRow * srcWidth;
                        float* d = dst + (tileH - 1) * tileW;
                        if (pad)
                        {
                            WinogradKernel3x3Block2x2SetInputN(s + col, srcWidth, rowPad, d + tileX, dstStride, nose0, nose1, nose2, nose3, noseD);
                            col += DF;
                            tileX += F;
                        }
                        for (; col < dstWDF; col += DF, tileX += F)
                            WinogradKernel3x3Block2x2SetInputN(s + col, srcWidth, rowPad, d + tileX, dstStride, all, all, all, all, all);
                        if (specialColTail)
                            WinogradKernel3x3Block2x2SetInputN(s + col, srcWidth, rowPad, d + tileX, dstStride, tail0, tail1, tail2, tail3, tailD);
                    }
                    src += srcWidth * srcHeight;
                    dst += tileW * tileH;
                }
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInputStoreRow(const svfloat32_t& s0, const svfloat32_t& s1,
            const svfloat32_t& s2, const svfloat32_t& s3, const svfloat32_t& s4, float* dst, size_t stride, const svbool_t& pg)
        {
            svfloat32_t _2 = svdup_n_f32(2.0f);
            svfloat32_t _3 = svdup_n_f32(3.0f);

            svst1_f32(pg, dst + 0 * stride, svadd_f32_x(pg, svmul_f32_x(pg, _2, svsub_f32_x(pg, s0, s2)), svsub_f32_x(pg, s3, s1)));
            svst1_f32(pg, dst + 1 * stride, svsub_f32_x(pg, svsub_f32_x(pg, s3, s2), svmul_f32_x(pg, _2, s1)));
            svst1_f32(pg, dst + 2 * stride, svadd_f32_x(pg, svmul_f32_x(pg, _2, s1), svsub_f32_x(pg, s3, svmul_f32_x(pg, _3, s2))));
            svst1_f32(pg, dst + 3 * stride, svsub_f32_x(pg, s3, s1));
            svst1_f32(pg, dst + 4 * stride, svadd_f32_x(pg, svmul_f32_x(pg, _2, svsub_f32_x(pg, s1, s3)), svsub_f32_x(pg, s4, s2)));
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInputStore(
            const svfloat32_t& s0, const svfloat32_t& s1, const svfloat32_t& s2, const svfloat32_t& s3, const svfloat32_t& s4,
            const svfloat32_t& s5, const svfloat32_t& s6, const svfloat32_t& s7, const svfloat32_t& s8, const svfloat32_t& s9,
            const svfloat32_t& s10, const svfloat32_t& s11, const svfloat32_t& s12, const svfloat32_t& s13, const svfloat32_t& s14,
            const svfloat32_t& s15, const svfloat32_t& s16, const svfloat32_t& s17, const svfloat32_t& s18, const svfloat32_t& s19,
            const svfloat32_t& s20, const svfloat32_t& s21, const svfloat32_t& s22, const svfloat32_t& s23, const svfloat32_t& s24,
            float* dst, size_t stride, const svbool_t& pg)
        {
            svfloat32_t _2 = svdup_n_f32(2.0f);
            svfloat32_t _3 = svdup_n_f32(3.0f);

            WinogradKernel3x3Block3x3SetInputStoreRow(
                svadd_f32_x(pg, svmul_f32_x(pg, _2, svsub_f32_x(pg, s0, s10)), svsub_f32_x(pg, s15, s5)),
                svadd_f32_x(pg, svmul_f32_x(pg, _2, svsub_f32_x(pg, s1, s11)), svsub_f32_x(pg, s16, s6)),
                svadd_f32_x(pg, svmul_f32_x(pg, _2, svsub_f32_x(pg, s2, s12)), svsub_f32_x(pg, s17, s7)),
                svadd_f32_x(pg, svmul_f32_x(pg, _2, svsub_f32_x(pg, s3, s13)), svsub_f32_x(pg, s18, s8)),
                svadd_f32_x(pg, svmul_f32_x(pg, _2, svsub_f32_x(pg, s4, s14)), svsub_f32_x(pg, s19, s9)),
                dst + 0 * stride, stride, pg);

            WinogradKernel3x3Block3x3SetInputStoreRow(
                svsub_f32_x(pg, svsub_f32_x(pg, s15, s10), svmul_f32_x(pg, _2, s5)),
                svsub_f32_x(pg, svsub_f32_x(pg, s16, s11), svmul_f32_x(pg, _2, s6)),
                svsub_f32_x(pg, svsub_f32_x(pg, s17, s12), svmul_f32_x(pg, _2, s7)),
                svsub_f32_x(pg, svsub_f32_x(pg, s18, s13), svmul_f32_x(pg, _2, s8)),
                svsub_f32_x(pg, svsub_f32_x(pg, s19, s14), svmul_f32_x(pg, _2, s9)),
                dst + 5 * stride, stride, pg);

            WinogradKernel3x3Block3x3SetInputStoreRow(
                svadd_f32_x(pg, svmul_f32_x(pg, _2, s5), svsub_f32_x(pg, s15, svmul_f32_x(pg, _3, s10))),
                svadd_f32_x(pg, svmul_f32_x(pg, _2, s6), svsub_f32_x(pg, s16, svmul_f32_x(pg, _3, s11))),
                svadd_f32_x(pg, svmul_f32_x(pg, _2, s7), svsub_f32_x(pg, s17, svmul_f32_x(pg, _3, s12))),
                svadd_f32_x(pg, svmul_f32_x(pg, _2, s8), svsub_f32_x(pg, s18, svmul_f32_x(pg, _3, s13))),
                svadd_f32_x(pg, svmul_f32_x(pg, _2, s9), svsub_f32_x(pg, s19, svmul_f32_x(pg, _3, s14))),
                dst + 10 * stride, stride, pg);

            WinogradKernel3x3Block3x3SetInputStoreRow(
                svsub_f32_x(pg, s15, s5), svsub_f32_x(pg, s16, s6), svsub_f32_x(pg, s17, s7),
                svsub_f32_x(pg, s18, s8), svsub_f32_x(pg, s19, s9), dst + 15 * stride, stride, pg);

            WinogradKernel3x3Block3x3SetInputStoreRow(
                svadd_f32_x(pg, svmul_f32_x(pg, _2, svsub_f32_x(pg, s5, s15)), svsub_f32_x(pg, s20, s10)),
                svadd_f32_x(pg, svmul_f32_x(pg, _2, svsub_f32_x(pg, s6, s16)), svsub_f32_x(pg, s21, s11)),
                svadd_f32_x(pg, svmul_f32_x(pg, _2, svsub_f32_x(pg, s7, s17)), svsub_f32_x(pg, s22, s12)),
                svadd_f32_x(pg, svmul_f32_x(pg, _2, svsub_f32_x(pg, s8, s18)), svsub_f32_x(pg, s23, s13)),
                svadd_f32_x(pg, svmul_f32_x(pg, _2, svsub_f32_x(pg, s9, s19)), svsub_f32_x(pg, s24, s14)),
                dst + 20 * stride, stride, pg);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInput(const float* src, size_t srcS, size_t srcC, float* dst, size_t dstStride, const svbool_t& pg)
        {
            WinogradKernel3x3Block3x3SetInputStore(
                svld1_f32(pg, src + 0 * srcS + 0 * srcC), svld1_f32(pg, src + 0 * srcS + 1 * srcC), svld1_f32(pg, src + 0 * srcS + 2 * srcC), svld1_f32(pg, src + 0 * srcS + 3 * srcC), svld1_f32(pg, src + 0 * srcS + 4 * srcC),
                svld1_f32(pg, src + 1 * srcS + 0 * srcC), svld1_f32(pg, src + 1 * srcS + 1 * srcC), svld1_f32(pg, src + 1 * srcS + 2 * srcC), svld1_f32(pg, src + 1 * srcS + 3 * srcC), svld1_f32(pg, src + 1 * srcS + 4 * srcC),
                svld1_f32(pg, src + 2 * srcS + 0 * srcC), svld1_f32(pg, src + 2 * srcS + 1 * srcC), svld1_f32(pg, src + 2 * srcS + 2 * srcC), svld1_f32(pg, src + 2 * srcS + 3 * srcC), svld1_f32(pg, src + 2 * srcS + 4 * srcC),
                svld1_f32(pg, src + 3 * srcS + 0 * srcC), svld1_f32(pg, src + 3 * srcS + 1 * srcC), svld1_f32(pg, src + 3 * srcS + 2 * srcC), svld1_f32(pg, src + 3 * srcS + 3 * srcC), svld1_f32(pg, src + 3 * srcS + 4 * srcC),
                svld1_f32(pg, src + 4 * srcS + 0 * srcC), svld1_f32(pg, src + 4 * srcS + 1 * srcC), svld1_f32(pg, src + 4 * srcS + 2 * srcC), svld1_f32(pg, src + 4 * srcS + 3 * srcC), svld1_f32(pg, src + 4 * srcS + 4 * srcC),
                dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInput(const float* src, size_t srcW, size_t srcC, float* dst, size_t dstStride)
        {
            const size_t F = svcntw();
            const size_t srcS = srcW * srcC;
            const size_t srcCF = AlignLo(srcC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < srcCF; c += F)
                WinogradKernel3x3Block3x3SetInput(src + c, srcS, srcC, dst + c, dstStride, body);
            if (c < srcC)
                WinogradKernel3x3Block3x3SetInput(src + c, srcS, srcC, dst + c, dstStride, svwhilelt_b32(c, srcC));
        }

        SIMD_INLINE svfloat32_t WinogradKernel3x3Block3x3SetInputLoad(const float* src, size_t srcS, size_t srcC, size_t row,
            size_t rowB, size_t rowE, size_t col, size_t colB, size_t colE, const svbool_t& pg)
        {
            return row >= rowB && row < rowE && col >= colB && col < colE ? svld1_f32(pg, src + row * srcS + col * srcC) : svdup_n_f32(0.0f);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInput(const float* src, size_t srcS, size_t srcC, size_t rowB, size_t rowE,
            size_t colB, size_t colE, float* dst, size_t dstStride, const svbool_t& pg)
        {
            WinogradKernel3x3Block3x3SetInputStore(
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 0, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 1, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 2, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 3, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 4, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 0, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 1, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 2, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 3, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 4, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 0, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 1, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 2, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 3, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 4, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 3, rowB, rowE, 0, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 3, rowB, rowE, 1, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 3, rowB, rowE, 2, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 3, rowB, rowE, 3, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 3, rowB, rowE, 4, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 4, rowB, rowE, 0, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 4, rowB, rowE, 1, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 4, rowB, rowE, 2, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 4, rowB, rowE, 3, colB, colE, pg),
                WinogradKernel3x3Block3x3SetInputLoad(src, srcS, srcC, 4, rowB, rowE, 4, colB, colE, pg),
                dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInput(const float* src, size_t srcW, size_t srcC, size_t rowB, size_t rowE,
            size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            const size_t F = svcntw();
            const size_t srcS = srcW * srcC;
            const size_t srcCF = AlignLo(srcC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < srcCF; c += F)
                WinogradKernel3x3Block3x3SetInput(src + c, srcS, srcC, rowB, rowE, colB, colE, dst + c, dstStride, body);
            if (c < srcC)
                WinogradKernel3x3Block3x3SetInput(src + c, srcS, srcC, rowB, rowE, colB, colE, dst + c, dstStride, svwhilelt_b32(c, srcC));
        }

        void WinogradKernel3x3Block3x3SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padY == padX && padY == padH && padY == padW && (padY == 0 || padY == 1));
            if (!trans)
            {
                Base::WinogradKernel3x3Block3x3SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
                return;
            }
            SimdBool pad = padY > 0 ? SimdTrue : SimdFalse;
            size_t dstH = pad ? srcHeight : srcHeight - 2;
            size_t dstW = pad ? srcWidth : srcWidth - 2;
            size_t dstH3 = AlignLoAny(dstH, 3);
            size_t dstW3 = AlignLoAny(dstW, 3);
            size_t noseW = Simd::Min<size_t>(5, dstW + 1);
            size_t noseH = Simd::Min<size_t>(5, dstH + 1);
            size_t start = pad ? 3 : 0;
            if (pad)
            {
                if (dstH == dstH3)
                    dstH3 -= 3;
                if (dstW == dstW3)
                    dstW3 -= 3;
                src -= (srcWidth + 1) * srcChannels;
            }
            size_t tailW = dstW - dstW3 + (pad ? 1 : 2);
            size_t tailH = dstH - dstH3 + (pad ? 1 : 2);
            size_t row = 0, col = 0;
            if (pad)
            {
                WinogradKernel3x3Block3x3SetInput(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = start; col < dstW3; col += 3)
                    WinogradKernel3x3Block3x3SetInput(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 5, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel3x3Block3x3SetInput(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
            }
            for (row = start; row < dstH3; row += 3)
            {
                if (pad)
                    WinogradKernel3x3Block3x3SetInput(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 5, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = start; col < dstW3; col += 3)
                    WinogradKernel3x3Block3x3SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel3x3Block3x3SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 5, 0, tailW, dst, dstStride), dst += srcChannels;
            }
            if (row < dstH)
            {
                if (pad)
                    WinogradKernel3x3Block3x3SetInput(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = start; col < dstW3; col += 3)
                    WinogradKernel3x3Block3x3SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 5, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel3x3Block3x3SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
            }
        }

        //-----------------------------------------------------------------------

#define SIMD_SVE2_WINOGRAD3X3_BLOCK4X4_SET_INPUT_ROW(s0, s1, s2, s3, s4, s5, d0, d1, d2, d3, d4, d5) \
            d0 = svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, _4, s0), svmul_f32_x(pg, _5, s2)), s4); \
            d1 = svsub_f32_x(pg, svadd_f32_x(pg, s3, s4), svmul_f32_x(pg, _4, svadd_f32_x(pg, s1, s2))); \
            d2 = svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, _4, svsub_f32_x(pg, s1, s2)), s3), s4); \
            d3 = svadd_f32_x(pg, svsub_f32_x(pg, s4, s2), svmul_f32_x(pg, _2, svsub_f32_x(pg, s3, s1))); \
            d4 = svadd_f32_x(pg, svsub_f32_x(pg, s4, s2), svmul_f32_x(pg, _2, svsub_f32_x(pg, s1, s3))); \
            d5 = svadd_f32_x(pg, svsub_f32_x(pg, svmul_f32_x(pg, _4, s1), svmul_f32_x(pg, _5, s3)), s5)

        SIMD_INLINE void WinogradKernel3x3Block4x4SetInputStore(
            const svfloat32_t& s0, const svfloat32_t& s1, const svfloat32_t& s2, const svfloat32_t& s3, const svfloat32_t& s4, const svfloat32_t& s5,
            const svfloat32_t& s6, const svfloat32_t& s7, const svfloat32_t& s8, const svfloat32_t& s9, const svfloat32_t& s10, const svfloat32_t& s11,
            const svfloat32_t& s12, const svfloat32_t& s13, const svfloat32_t& s14, const svfloat32_t& s15, const svfloat32_t& s16, const svfloat32_t& s17,
            const svfloat32_t& s18, const svfloat32_t& s19, const svfloat32_t& s20, const svfloat32_t& s21, const svfloat32_t& s22, const svfloat32_t& s23,
            const svfloat32_t& s24, const svfloat32_t& s25, const svfloat32_t& s26, const svfloat32_t& s27, const svfloat32_t& s28, const svfloat32_t& s29,
            const svfloat32_t& s30, const svfloat32_t& s31, const svfloat32_t& s32, const svfloat32_t& s33, const svfloat32_t& s34, const svfloat32_t& s35,
            float* dst, size_t dstStride, const svbool_t& pg)
        {
            svfloat32_t _2 = svdup_n_f32(2.0f);
            svfloat32_t _4 = svdup_n_f32(4.0f);
            svfloat32_t _5 = svdup_n_f32(5.0f);
            svfloat32_t t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
            svfloat32_t t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23;
            svfloat32_t t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35;
            svfloat32_t d0, d1, d2, d3, d4, d5;

            SIMD_SVE2_WINOGRAD3X3_BLOCK4X4_SET_INPUT_ROW(s0, s6, s12, s18, s24, s30, t0, t6, t12, t18, t24, t30);
            SIMD_SVE2_WINOGRAD3X3_BLOCK4X4_SET_INPUT_ROW(s1, s7, s13, s19, s25, s31, t1, t7, t13, t19, t25, t31);
            SIMD_SVE2_WINOGRAD3X3_BLOCK4X4_SET_INPUT_ROW(s2, s8, s14, s20, s26, s32, t2, t8, t14, t20, t26, t32);
            SIMD_SVE2_WINOGRAD3X3_BLOCK4X4_SET_INPUT_ROW(s3, s9, s15, s21, s27, s33, t3, t9, t15, t21, t27, t33);
            SIMD_SVE2_WINOGRAD3X3_BLOCK4X4_SET_INPUT_ROW(s4, s10, s16, s22, s28, s34, t4, t10, t16, t22, t28, t34);
            SIMD_SVE2_WINOGRAD3X3_BLOCK4X4_SET_INPUT_ROW(s5, s11, s17, s23, s29, s35, t5, t11, t17, t23, t29, t35);

            SIMD_SVE2_WINOGRAD3X3_BLOCK4X4_SET_INPUT_ROW(t0, t1, t2, t3, t4, t5, d0, d1, d2, d3, d4, d5);
            svst1_f32(pg, dst + 0 * dstStride, d0);
            svst1_f32(pg, dst + 1 * dstStride, d1);
            svst1_f32(pg, dst + 2 * dstStride, d2);
            svst1_f32(pg, dst + 3 * dstStride, d3);
            svst1_f32(pg, dst + 4 * dstStride, d4);
            svst1_f32(pg, dst + 5 * dstStride, d5);

            SIMD_SVE2_WINOGRAD3X3_BLOCK4X4_SET_INPUT_ROW(t6, t7, t8, t9, t10, t11, d0, d1, d2, d3, d4, d5);
            svst1_f32(pg, dst + 6 * dstStride, d0);
            svst1_f32(pg, dst + 7 * dstStride, d1);
            svst1_f32(pg, dst + 8 * dstStride, d2);
            svst1_f32(pg, dst + 9 * dstStride, d3);
            svst1_f32(pg, dst + 10 * dstStride, d4);
            svst1_f32(pg, dst + 11 * dstStride, d5);

            SIMD_SVE2_WINOGRAD3X3_BLOCK4X4_SET_INPUT_ROW(t12, t13, t14, t15, t16, t17, d0, d1, d2, d3, d4, d5);
            svst1_f32(pg, dst + 12 * dstStride, d0);
            svst1_f32(pg, dst + 13 * dstStride, d1);
            svst1_f32(pg, dst + 14 * dstStride, d2);
            svst1_f32(pg, dst + 15 * dstStride, d3);
            svst1_f32(pg, dst + 16 * dstStride, d4);
            svst1_f32(pg, dst + 17 * dstStride, d5);

            SIMD_SVE2_WINOGRAD3X3_BLOCK4X4_SET_INPUT_ROW(t18, t19, t20, t21, t22, t23, d0, d1, d2, d3, d4, d5);
            svst1_f32(pg, dst + 18 * dstStride, d0);
            svst1_f32(pg, dst + 19 * dstStride, d1);
            svst1_f32(pg, dst + 20 * dstStride, d2);
            svst1_f32(pg, dst + 21 * dstStride, d3);
            svst1_f32(pg, dst + 22 * dstStride, d4);
            svst1_f32(pg, dst + 23 * dstStride, d5);

            SIMD_SVE2_WINOGRAD3X3_BLOCK4X4_SET_INPUT_ROW(t24, t25, t26, t27, t28, t29, d0, d1, d2, d3, d4, d5);
            svst1_f32(pg, dst + 24 * dstStride, d0);
            svst1_f32(pg, dst + 25 * dstStride, d1);
            svst1_f32(pg, dst + 26 * dstStride, d2);
            svst1_f32(pg, dst + 27 * dstStride, d3);
            svst1_f32(pg, dst + 28 * dstStride, d4);
            svst1_f32(pg, dst + 29 * dstStride, d5);

            SIMD_SVE2_WINOGRAD3X3_BLOCK4X4_SET_INPUT_ROW(t30, t31, t32, t33, t34, t35, d0, d1, d2, d3, d4, d5);
            svst1_f32(pg, dst + 30 * dstStride, d0);
            svst1_f32(pg, dst + 31 * dstStride, d1);
            svst1_f32(pg, dst + 32 * dstStride, d2);
            svst1_f32(pg, dst + 33 * dstStride, d3);
            svst1_f32(pg, dst + 34 * dstStride, d4);
            svst1_f32(pg, dst + 35 * dstStride, d5);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetInput(const float* src, size_t srcS, size_t srcC, float* dst, size_t dstStride, const svbool_t& pg)
        {
            WinogradKernel3x3Block4x4SetInputStore(
                svld1_f32(pg, src + 0 * srcS + 0 * srcC), svld1_f32(pg, src + 0 * srcS + 1 * srcC), svld1_f32(pg, src + 0 * srcS + 2 * srcC), svld1_f32(pg, src + 0 * srcS + 3 * srcC), svld1_f32(pg, src + 0 * srcS + 4 * srcC), svld1_f32(pg, src + 0 * srcS + 5 * srcC),
                svld1_f32(pg, src + 1 * srcS + 0 * srcC), svld1_f32(pg, src + 1 * srcS + 1 * srcC), svld1_f32(pg, src + 1 * srcS + 2 * srcC), svld1_f32(pg, src + 1 * srcS + 3 * srcC), svld1_f32(pg, src + 1 * srcS + 4 * srcC), svld1_f32(pg, src + 1 * srcS + 5 * srcC),
                svld1_f32(pg, src + 2 * srcS + 0 * srcC), svld1_f32(pg, src + 2 * srcS + 1 * srcC), svld1_f32(pg, src + 2 * srcS + 2 * srcC), svld1_f32(pg, src + 2 * srcS + 3 * srcC), svld1_f32(pg, src + 2 * srcS + 4 * srcC), svld1_f32(pg, src + 2 * srcS + 5 * srcC),
                svld1_f32(pg, src + 3 * srcS + 0 * srcC), svld1_f32(pg, src + 3 * srcS + 1 * srcC), svld1_f32(pg, src + 3 * srcS + 2 * srcC), svld1_f32(pg, src + 3 * srcS + 3 * srcC), svld1_f32(pg, src + 3 * srcS + 4 * srcC), svld1_f32(pg, src + 3 * srcS + 5 * srcC),
                svld1_f32(pg, src + 4 * srcS + 0 * srcC), svld1_f32(pg, src + 4 * srcS + 1 * srcC), svld1_f32(pg, src + 4 * srcS + 2 * srcC), svld1_f32(pg, src + 4 * srcS + 3 * srcC), svld1_f32(pg, src + 4 * srcS + 4 * srcC), svld1_f32(pg, src + 4 * srcS + 5 * srcC),
                svld1_f32(pg, src + 5 * srcS + 0 * srcC), svld1_f32(pg, src + 5 * srcS + 1 * srcC), svld1_f32(pg, src + 5 * srcS + 2 * srcC), svld1_f32(pg, src + 5 * srcS + 3 * srcC), svld1_f32(pg, src + 5 * srcS + 4 * srcC), svld1_f32(pg, src + 5 * srcS + 5 * srcC),
                dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetInput(const float* src, size_t srcW, size_t srcC, float* dst, size_t dstStride)
        {
            const size_t F = svcntw();
            const size_t srcS = srcW * srcC;
            const size_t srcCF = AlignLo(srcC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < srcCF; c += F)
                WinogradKernel3x3Block4x4SetInput(src + c, srcS, srcC, dst + c, dstStride, body);
            if (c < srcC)
                WinogradKernel3x3Block4x4SetInput(src + c, srcS, srcC, dst + c, dstStride, svwhilelt_b32(c, srcC));
        }

        SIMD_INLINE svfloat32_t WinogradKernel3x3Block4x4SetInputLoad(const float* src, size_t srcS, size_t srcC, size_t row,
            size_t rowB, size_t rowE, size_t col, size_t colB, size_t colE, const svbool_t& pg)
        {
            return row >= rowB && row < rowE && col >= colB && col < colE ? svld1_f32(pg, src + row * srcS + col * srcC) : svdup_n_f32(0.0f);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetInput(const float* src, size_t srcS, size_t srcC, size_t rowB, size_t rowE,
            size_t colB, size_t colE, float* dst, size_t dstStride, const svbool_t& pg)
        {
            WinogradKernel3x3Block4x4SetInputStore(
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 0, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 1, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 2, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 3, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 4, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 0, rowB, rowE, 5, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 0, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 1, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 2, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 3, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 4, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 1, rowB, rowE, 5, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 0, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 1, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 2, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 3, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 4, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 2, rowB, rowE, 5, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 3, rowB, rowE, 0, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 3, rowB, rowE, 1, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 3, rowB, rowE, 2, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 3, rowB, rowE, 3, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 3, rowB, rowE, 4, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 3, rowB, rowE, 5, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 4, rowB, rowE, 0, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 4, rowB, rowE, 1, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 4, rowB, rowE, 2, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 4, rowB, rowE, 3, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 4, rowB, rowE, 4, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 4, rowB, rowE, 5, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 5, rowB, rowE, 0, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 5, rowB, rowE, 1, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 5, rowB, rowE, 2, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 5, rowB, rowE, 3, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 5, rowB, rowE, 4, colB, colE, pg),
                WinogradKernel3x3Block4x4SetInputLoad(src, srcS, srcC, 5, rowB, rowE, 5, colB, colE, pg),
                dst, dstStride, pg);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetInput(const float* src, size_t srcW, size_t srcC, size_t rowB, size_t rowE,
            size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            const size_t F = svcntw();
            const size_t srcS = srcW * srcC;
            const size_t srcCF = AlignLo(srcC, F);
            const svbool_t body = svptrue_b32();
            size_t c = 0;
            for (; c < srcCF; c += F)
                WinogradKernel3x3Block4x4SetInput(src + c, srcS, srcC, rowB, rowE, colB, colE, dst + c, dstStride, body);
            if (c < srcC)
                WinogradKernel3x3Block4x4SetInput(src + c, srcS, srcC, rowB, rowE, colB, colE, dst + c, dstStride, svwhilelt_b32(c, srcC));
        }

        void WinogradKernel3x3Block4x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padY + padH <= 2 && padX + padW <= 2);
            if (!trans)
            {
                Base::WinogradKernel3x3Block4x4SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
                return;
            }
            size_t dstH = srcHeight - 2 + padY + padH;
            size_t dstW = srcWidth - 2 + padX + padW;
            size_t dstH4 = AlignLo(dstH, 4);
            size_t dstW4 = AlignLo(dstW, 4);
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
                    WinogradKernel3x3Block4x4SetInput(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = startX; col < dstW4; col += 4)
                    WinogradKernel3x3Block4x4SetInput(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 6, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel3x3Block4x4SetInput(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
            }
            for (row = startY; row < dstH4; row += 4)
            {
                if (padX)
                    WinogradKernel3x3Block4x4SetInput(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 6, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = startX; col < dstW4; col += 4)
                    WinogradKernel3x3Block4x4SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel3x3Block4x4SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 6, 0, tailW, dst, dstStride), dst += srcChannels;
            }
            if (row < dstH)
            {
                if (padX)
                    WinogradKernel3x3Block4x4SetInput(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                for (col = startX; col < dstW4; col += 4)
                    WinogradKernel3x3Block4x4SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 6, dst, dstStride), dst += srcChannels;
                if (col < dstW)
                    WinogradKernel3x3Block4x4SetInput(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
            }
        }

#undef SIMD_SVE2_WINOGRAD3X3_BLOCK4X4_SET_INPUT_ROW

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutputLoad4(const float* src, size_t stride, svfloat32_t& dst0, svfloat32_t& dst1, const svbool_t& pg)
        {
            svfloat32_t s0 = svld1_f32(pg, src + 0 * stride);
            svfloat32_t s1 = svld1_f32(pg, src + 1 * stride);
            svfloat32_t s2 = svld1_f32(pg, src + 2 * stride);
            svfloat32_t s3 = svld1_f32(pg, src + 3 * stride);
            dst0 = svadd_f32_x(pg, svadd_f32_x(pg, s0, s1), s2);
            dst1 = svsub_f32_x(pg, svsub_f32_x(pg, s1, s2), s3);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutputLoad16(const float* src, size_t stride, svfloat32_t& dst0, svfloat32_t& dst1, svfloat32_t& dst2, svfloat32_t& dst3, const svbool_t& pg)
        {
            svfloat32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
            WinogradKernel3x3Block2x2SetOutputLoad4(src + 0 * stride, stride, tmp0, tmp1, pg);
            WinogradKernel3x3Block2x2SetOutputLoad4(src + 4 * stride, stride, tmp2, tmp3, pg);
            WinogradKernel3x3Block2x2SetOutputLoad4(src + 8 * stride, stride, tmp4, tmp5, pg);
            WinogradKernel3x3Block2x2SetOutputLoad4(src + 12 * stride, stride, tmp6, tmp7, pg);
            dst0 = svadd_f32_x(pg, svadd_f32_x(pg, tmp0, tmp2), tmp4);
            dst1 = svadd_f32_x(pg, svadd_f32_x(pg, tmp1, tmp3), tmp5);
            dst2 = svsub_f32_x(pg, svsub_f32_x(pg, tmp2, tmp4), tmp6);
            dst3 = svsub_f32_x(pg, svsub_f32_x(pg, tmp3, tmp5), tmp7);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutputN(const float* src, size_t srcStride, float* dst, size_t dstStride, const svbool_t& pg)
        {
            const size_t F = svcntw();
            svfloat32_t tmp0, tmp1, tmp2, tmp3;
            WinogradKernel3x3Block2x2SetOutputLoad16(src, srcStride, tmp0, tmp1, tmp2, tmp3, pg);
            svst1_f32(pg, dst + 0 * dstStride + 0, svzip1_f32(tmp0, tmp1));
            svst1_f32(pg, dst + 0 * dstStride + F, svzip2_f32(tmp0, tmp1));
            svst1_f32(pg, dst + 1 * dstStride + 0, svzip1_f32(tmp2, tmp3));
            svst1_f32(pg, dst + 1 * dstStride + F, svzip2_f32(tmp2, tmp3));
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutputN(const float* src, size_t srcStride, float* dst, size_t dstStride, size_t rowE, size_t colE)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svbool_t lo = svwhilelt_b32((size_t)0, colE);
            const svbool_t hi = svwhilelt_b32(F, colE);
            svfloat32_t tmp0, tmp1, tmp2, tmp3;
            WinogradKernel3x3Block2x2SetOutputLoad16(src, srcStride, tmp0, tmp1, tmp2, tmp3, body);
            svst1_f32(lo, dst + 0, svzip1_f32(tmp0, tmp1));
            svst1_f32(hi, dst + F, svzip2_f32(tmp0, tmp1));
            if (rowE > 1)
            {
                dst += dstStride;
                svst1_f32(lo, dst + 0, svzip1_f32(tmp2, tmp3));
                svst1_f32(hi, dst + F, svzip2_f32(tmp2, tmp3));
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutputStore(const svfloat32_t& src0, const svfloat32_t& src1, const svfloat32_t& src2, const svfloat32_t& src3, float* dst, size_t dstS, size_t dstC, const svbool_t& pg)
        {
            svst1_f32(pg, dst + 0 * dstS + 0 * dstC, src0);
            svst1_f32(pg, dst + 0 * dstS + 1 * dstC, src1);
            svst1_f32(pg, dst + 1 * dstS + 0 * dstC, src2);
            svst1_f32(pg, dst + 1 * dstS + 1 * dstC, src3);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutputStore(const svfloat32_t& src0, const svfloat32_t& src1, const svfloat32_t& src2, const svfloat32_t& src3, float* dst, size_t dstS, size_t dstC, size_t rowE, size_t colE, const svbool_t& pg)
        {
            if (rowE > 0 && colE > 0)
                svst1_f32(pg, dst + 0 * dstS + 0 * dstC, src0);
            if (rowE > 0 && colE > 1)
                svst1_f32(pg, dst + 0 * dstS + 1 * dstC, src1);
            if (rowE > 1 && colE > 0)
                svst1_f32(pg, dst + 1 * dstS + 0 * dstC, src2);
            if (rowE > 1 && colE > 1)
                svst1_f32(pg, dst + 1 * dstS + 1 * dstC, src3);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutputT(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC)
        {
            const size_t F = svcntw();
            const size_t dstCF = AlignLo(dstC, F);
            const svbool_t body = svptrue_b32();
            const size_t dstS = dstW * dstC;
            size_t d = 0;
            for (; d < dstCF; d += F)
            {
                svfloat32_t tmp0, tmp1, tmp2, tmp3;
                WinogradKernel3x3Block2x2SetOutputLoad16(src + d, srcStride, tmp0, tmp1, tmp2, tmp3, body);
                WinogradKernel3x3Block2x2SetOutputStore(tmp0, tmp1, tmp2, tmp3, dst + d, dstS, dstC, body);
            }
            if (d < dstC)
            {
                svbool_t tail = svwhilelt_b32(d, dstC);
                svfloat32_t tmp0, tmp1, tmp2, tmp3;
                WinogradKernel3x3Block2x2SetOutputLoad16(src + d, srcStride, tmp0, tmp1, tmp2, tmp3, tail);
                WinogradKernel3x3Block2x2SetOutputStore(tmp0, tmp1, tmp2, tmp3, dst + d, dstS, dstC, tail);
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutputT(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC, size_t rowE, size_t colE)
        {
            const size_t F = svcntw();
            const size_t dstCF = AlignLo(dstC, F);
            const svbool_t body = svptrue_b32();
            const size_t dstS = dstW * dstC;
            size_t d = 0;
            for (; d < dstCF; d += F)
            {
                svfloat32_t tmp0, tmp1, tmp2, tmp3;
                WinogradKernel3x3Block2x2SetOutputLoad16(src + d, srcStride, tmp0, tmp1, tmp2, tmp3, body);
                WinogradKernel3x3Block2x2SetOutputStore(tmp0, tmp1, tmp2, tmp3, dst + d, dstS, dstC, rowE, colE, body);
            }
            if (d < dstC)
            {
                svbool_t tail = svwhilelt_b32(d, dstC);
                svfloat32_t tmp0, tmp1, tmp2, tmp3;
                WinogradKernel3x3Block2x2SetOutputLoad16(src + d, srcStride, tmp0, tmp1, tmp2, tmp3, tail);
                WinogradKernel3x3Block2x2SetOutputStore(tmp0, tmp1, tmp2, tmp3, dst + d, dstS, dstC, rowE, colE, tail);
            }
        }

        void WinogradKernel3x3Block2x2SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            const size_t F = svcntw();
            const size_t DF = 2 * F;
            if (!trans && (dstHeight < 2 || dstWidth < DF))
            {
                Base::WinogradKernel3x3Block2x2SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
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
                        WinogradKernel3x3Block2x2SetOutputT(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block2x2SetOutputT(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, 2, dstWidth - col), src += dstChannels;
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstW2; col += 2)
                        WinogradKernel3x3Block2x2SetOutputT(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row, 2), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block2x2SetOutputT(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row, dstWidth - col), src += dstChannels;
                }
            }
            else
            {
                const size_t dstWF = AlignLo(dstWidth, DF);
                const svbool_t body = svptrue_b32();
                const size_t tailCol = dstW2 < dstWidth ? dstWidth - DF + 1 : dstWidth - DF;
                const size_t tailE = dstWidth - tailCol;
                for (size_t c = 0; c < dstChannels; ++c)
                {
                    size_t row = 0, tileY = 0;
                    for (; row < dstH2; row += 2, tileY += 1)
                    {
                        size_t col = 0, tileX = 0;
                        const float* s = src + tileY * tileW;
                        float* d = dst + row * dstWidth;
                        for (; col < dstWF; col += DF, tileX += F)
                            WinogradKernel3x3Block2x2SetOutputN(s + tileX, srcStride, d + col, dstWidth, body);
                        if (col < dstWidth)
                            WinogradKernel3x3Block2x2SetOutputN(s + tileW - F, srcStride, d + tailCol, dstWidth, 2, tailE);
                    }
                    if (row < dstHeight)
                    {
                        size_t col = 0, tileX = 0;
                        const float* s = src + (tileH - 1) * tileW;
                        float* d = dst + (dstHeight - 1) * dstWidth;
                        for (; col < dstWF; col += DF, tileX += F)
                            WinogradKernel3x3Block2x2SetOutputN(s + tileX, srcStride, d + col, dstWidth, 1, DF);
                        if (col < dstWidth)
                            WinogradKernel3x3Block2x2SetOutputN(s + tileW - F, srcStride, d + tailCol, dstWidth, 1, tailE);
                    }
                    src += tileW * tileH;
                    dst += dstHeight * dstWidth;
                }
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block3x3SetOutputLoad25(const float* src, size_t stride,
            svfloat32_t& dst0, svfloat32_t& dst1, svfloat32_t& dst2, svfloat32_t& dst3, svfloat32_t& dst4,
            svfloat32_t& dst5, svfloat32_t& dst6, svfloat32_t& dst7, svfloat32_t& dst8, const svbool_t& pg)
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
            svfloat32_t s9 = svld1_f32(pg, src + 9 * stride);
            svfloat32_t s10 = svld1_f32(pg, src + 10 * stride);
            svfloat32_t s11 = svld1_f32(pg, src + 11 * stride);
            svfloat32_t s12 = svld1_f32(pg, src + 12 * stride);
            svfloat32_t s13 = svld1_f32(pg, src + 13 * stride);
            svfloat32_t s14 = svld1_f32(pg, src + 14 * stride);
            svfloat32_t s15 = svld1_f32(pg, src + 15 * stride);
            svfloat32_t s16 = svld1_f32(pg, src + 16 * stride);
            svfloat32_t s17 = svld1_f32(pg, src + 17 * stride);
            svfloat32_t s18 = svld1_f32(pg, src + 18 * stride);
            svfloat32_t s19 = svld1_f32(pg, src + 19 * stride);
            svfloat32_t s20 = svld1_f32(pg, src + 20 * stride);
            svfloat32_t s21 = svld1_f32(pg, src + 21 * stride);
            svfloat32_t s22 = svld1_f32(pg, src + 22 * stride);
            svfloat32_t s23 = svld1_f32(pg, src + 23 * stride);
            svfloat32_t s24 = svld1_f32(pg, src + 24 * stride);

            svfloat32_t _2 = svdup_n_f32(2.0f);
            svfloat32_t _4 = svdup_n_f32(4.0f);
            svfloat32_t t0, t1, t2, t3, t4;

            t0 = svadd_f32_x(pg, svadd_f32_x(pg, s0, s5), svadd_f32_x(pg, s10, s15));
            t1 = svadd_f32_x(pg, svadd_f32_x(pg, s1, s6), svadd_f32_x(pg, s11, s16));
            t2 = svadd_f32_x(pg, svadd_f32_x(pg, s2, s7), svadd_f32_x(pg, s12, s17));
            t3 = svadd_f32_x(pg, svadd_f32_x(pg, s3, s8), svadd_f32_x(pg, s13, s18));
            t4 = svadd_f32_x(pg, svadd_f32_x(pg, s4, s9), svadd_f32_x(pg, s14, s19));
            dst0 = svadd_f32_x(pg, svadd_f32_x(pg, t0, t1), svadd_f32_x(pg, t2, t3));
            dst1 = svadd_f32_x(pg, svsub_f32_x(pg, t1, t2), svmul_f32_x(pg, _2, t3));
            dst2 = svadd_f32_x(pg, svadd_f32_x(pg, t1, t2), svadd_f32_x(pg, svmul_f32_x(pg, _4, t3), t4));

            t0 = svadd_f32_x(pg, svsub_f32_x(pg, s5, s10), svmul_f32_x(pg, _2, s15));
            t1 = svadd_f32_x(pg, svsub_f32_x(pg, s6, s11), svmul_f32_x(pg, _2, s16));
            t2 = svadd_f32_x(pg, svsub_f32_x(pg, s7, s12), svmul_f32_x(pg, _2, s17));
            t3 = svadd_f32_x(pg, svsub_f32_x(pg, s8, s13), svmul_f32_x(pg, _2, s18));
            t4 = svadd_f32_x(pg, svsub_f32_x(pg, s9, s14), svmul_f32_x(pg, _2, s19));
            dst3 = svadd_f32_x(pg, svadd_f32_x(pg, t0, t1), svadd_f32_x(pg, t2, t3));
            dst4 = svadd_f32_x(pg, svsub_f32_x(pg, t1, t2), svmul_f32_x(pg, _2, t3));
            dst5 = svadd_f32_x(pg, svadd_f32_x(pg, t1, t2), svadd_f32_x(pg, svmul_f32_x(pg, _4, t3), t4));

            t0 = svadd_f32_x(pg, svadd_f32_x(pg, s5, s10), svadd_f32_x(pg, svmul_f32_x(pg, _4, s15), s20));
            t1 = svadd_f32_x(pg, svadd_f32_x(pg, s6, s11), svadd_f32_x(pg, svmul_f32_x(pg, _4, s16), s21));
            t2 = svadd_f32_x(pg, svadd_f32_x(pg, s7, s12), svadd_f32_x(pg, svmul_f32_x(pg, _4, s17), s22));
            t3 = svadd_f32_x(pg, svadd_f32_x(pg, s8, s13), svadd_f32_x(pg, svmul_f32_x(pg, _4, s18), s23));
            t4 = svadd_f32_x(pg, svadd_f32_x(pg, s9, s14), svadd_f32_x(pg, svmul_f32_x(pg, _4, s19), s24));
            dst6 = svadd_f32_x(pg, svadd_f32_x(pg, t0, t1), svadd_f32_x(pg, t2, t3));
            dst7 = svadd_f32_x(pg, svsub_f32_x(pg, t1, t2), svmul_f32_x(pg, _2, t3));
            dst8 = svadd_f32_x(pg, svadd_f32_x(pg, t1, t2), svadd_f32_x(pg, svmul_f32_x(pg, _4, t3), t4));
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetOutputStore9(
            const svfloat32_t& src0, const svfloat32_t& src1, const svfloat32_t& src2, const svfloat32_t& src3, const svfloat32_t& src4,
            const svfloat32_t& src5, const svfloat32_t& src6, const svfloat32_t& src7, const svfloat32_t& src8,
            float* dst, size_t dstS, size_t dstC, const svbool_t& pg)
        {
            svst1_f32(pg, dst + 0 * dstS + 0 * dstC, src0);
            svst1_f32(pg, dst + 0 * dstS + 1 * dstC, src1);
            svst1_f32(pg, dst + 0 * dstS + 2 * dstC, src2);
            svst1_f32(pg, dst + 1 * dstS + 0 * dstC, src3);
            svst1_f32(pg, dst + 1 * dstS + 1 * dstC, src4);
            svst1_f32(pg, dst + 1 * dstS + 2 * dstC, src5);
            svst1_f32(pg, dst + 2 * dstS + 0 * dstC, src6);
            svst1_f32(pg, dst + 2 * dstS + 1 * dstC, src7);
            svst1_f32(pg, dst + 2 * dstS + 2 * dstC, src8);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetOutputStore9(
            const svfloat32_t& src0, const svfloat32_t& src1, const svfloat32_t& src2, const svfloat32_t& src3, const svfloat32_t& src4,
            const svfloat32_t& src5, const svfloat32_t& src6, const svfloat32_t& src7, const svfloat32_t& src8,
            float* dst, size_t dstS, size_t dstC, size_t rowE, size_t colE, const svbool_t& pg)
        {
            if (rowE > 0 && colE > 0)
                svst1_f32(pg, dst + 0 * dstS + 0 * dstC, src0);
            if (rowE > 0 && colE > 1)
                svst1_f32(pg, dst + 0 * dstS + 1 * dstC, src1);
            if (rowE > 0 && colE > 2)
                svst1_f32(pg, dst + 0 * dstS + 2 * dstC, src2);
            if (rowE > 1 && colE > 0)
                svst1_f32(pg, dst + 1 * dstS + 0 * dstC, src3);
            if (rowE > 1 && colE > 1)
                svst1_f32(pg, dst + 1 * dstS + 1 * dstC, src4);
            if (rowE > 1 && colE > 2)
                svst1_f32(pg, dst + 1 * dstS + 2 * dstC, src5);
            if (rowE > 2 && colE > 0)
                svst1_f32(pg, dst + 2 * dstS + 0 * dstC, src6);
            if (rowE > 2 && colE > 1)
                svst1_f32(pg, dst + 2 * dstS + 1 * dstC, src7);
            if (rowE > 2 && colE > 2)
                svst1_f32(pg, dst + 2 * dstS + 2 * dstC, src8);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetOutputT(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC)
        {
            const size_t F = svcntw();
            const size_t dstCF = AlignLo(dstC, F);
            const svbool_t body = svptrue_b32();
            const size_t dstS = dstW * dstC;
            size_t d = 0;
            for (; d < dstCF; d += F)
            {
                svfloat32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
                WinogradKernel3x3Block3x3SetOutputLoad25(src + d, srcStride, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, body);
                WinogradKernel3x3Block3x3SetOutputStore9(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, dst + d, dstS, dstC, body);
            }
            if (d < dstC)
            {
                svbool_t tail = svwhilelt_b32(d, dstC);
                svfloat32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
                WinogradKernel3x3Block3x3SetOutputLoad25(src + d, srcStride, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tail);
                WinogradKernel3x3Block3x3SetOutputStore9(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, dst + d, dstS, dstC, tail);
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetOutputT(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC, size_t rowE, size_t colE)
        {
            const size_t F = svcntw();
            const size_t dstCF = AlignLo(dstC, F);
            const svbool_t body = svptrue_b32();
            const size_t dstS = dstW * dstC;
            size_t d = 0;
            for (; d < dstCF; d += F)
            {
                svfloat32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
                WinogradKernel3x3Block3x3SetOutputLoad25(src + d, srcStride, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, body);
                WinogradKernel3x3Block3x3SetOutputStore9(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, dst + d, dstS, dstC, rowE, colE, body);
            }
            if (d < dstC)
            {
                svbool_t tail = svwhilelt_b32(d, dstC);
                svfloat32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
                WinogradKernel3x3Block3x3SetOutputLoad25(src + d, srcStride, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tail);
                WinogradKernel3x3Block3x3SetOutputStore9(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, dst + d, dstS, dstC, rowE, colE, tail);
            }
        }

        void WinogradKernel3x3Block3x3SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            if (!trans)
            {
                Base::WinogradKernel3x3Block3x3SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
                return;
            }
            size_t dstH3 = AlignLoAny(dstHeight, 3);
            size_t dstW3 = AlignLoAny(dstWidth, 3);
            size_t row, col;
            for (row = 0; row < dstH3; row += 3)
            {
                for (col = 0; col < dstW3; col += 3)
                    WinogradKernel3x3Block3x3SetOutputT(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels), src += dstChannels;
                if (col < dstWidth)
                    WinogradKernel3x3Block3x3SetOutputT(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, 3, dstWidth - col), src += dstChannels;
            }
            if (row < dstHeight)
            {
                for (col = 0; col < dstW3; col += 3)
                    WinogradKernel3x3Block3x3SetOutputT(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row, 3), src += dstChannels;
                if (col < dstWidth)
                    WinogradKernel3x3Block3x3SetOutputT(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row, dstWidth - col), src += dstChannels;
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block4x4SetOutputLoad36(const float* src, size_t stride,
            svfloat32_t& dst0, svfloat32_t& dst1, svfloat32_t& dst2, svfloat32_t& dst3,
            svfloat32_t& dst4, svfloat32_t& dst5, svfloat32_t& dst6, svfloat32_t& dst7,
            svfloat32_t& dst8, svfloat32_t& dst9, svfloat32_t& dst10, svfloat32_t& dst11,
            svfloat32_t& dst12, svfloat32_t& dst13, svfloat32_t& dst14, svfloat32_t& dst15, const svbool_t& pg)
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
            svfloat32_t s9 = svld1_f32(pg, src + 9 * stride);
            svfloat32_t s10 = svld1_f32(pg, src + 10 * stride);
            svfloat32_t s11 = svld1_f32(pg, src + 11 * stride);
            svfloat32_t s12 = svld1_f32(pg, src + 12 * stride);
            svfloat32_t s13 = svld1_f32(pg, src + 13 * stride);
            svfloat32_t s14 = svld1_f32(pg, src + 14 * stride);
            svfloat32_t s15 = svld1_f32(pg, src + 15 * stride);
            svfloat32_t s16 = svld1_f32(pg, src + 16 * stride);
            svfloat32_t s17 = svld1_f32(pg, src + 17 * stride);
            svfloat32_t s18 = svld1_f32(pg, src + 18 * stride);
            svfloat32_t s19 = svld1_f32(pg, src + 19 * stride);
            svfloat32_t s20 = svld1_f32(pg, src + 20 * stride);
            svfloat32_t s21 = svld1_f32(pg, src + 21 * stride);
            svfloat32_t s22 = svld1_f32(pg, src + 22 * stride);
            svfloat32_t s23 = svld1_f32(pg, src + 23 * stride);
            svfloat32_t s24 = svld1_f32(pg, src + 24 * stride);
            svfloat32_t s25 = svld1_f32(pg, src + 25 * stride);
            svfloat32_t s26 = svld1_f32(pg, src + 26 * stride);
            svfloat32_t s27 = svld1_f32(pg, src + 27 * stride);
            svfloat32_t s28 = svld1_f32(pg, src + 28 * stride);
            svfloat32_t s29 = svld1_f32(pg, src + 29 * stride);
            svfloat32_t s30 = svld1_f32(pg, src + 30 * stride);
            svfloat32_t s31 = svld1_f32(pg, src + 31 * stride);
            svfloat32_t s32 = svld1_f32(pg, src + 32 * stride);
            svfloat32_t s33 = svld1_f32(pg, src + 33 * stride);
            svfloat32_t s34 = svld1_f32(pg, src + 34 * stride);
            svfloat32_t s35 = svld1_f32(pg, src + 35 * stride);

            svfloat32_t _2 = svdup_n_f32(2.0f);
            svfloat32_t _4 = svdup_n_f32(4.0f);
            svfloat32_t _8 = svdup_n_f32(8.0f);
            svfloat32_t t0, t1, t2, t3, t4, t5;
            svfloat32_t t6, t7, t8, t9, t10, t11;
            svfloat32_t t12, t13, t14, t15, t16, t17;
            svfloat32_t t18, t19, t20, t21, t22, t23;

            t0 = svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, s0, s6), s12), s18), s24);
            t1 = svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, s1, s7), s13), s19), s25);
            t2 = svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, s2, s8), s14), s20), s26);
            t3 = svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, s3, s9), s15), s21), s27);
            t4 = svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, s4, s10), s16), s22), s28);
            t5 = svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, s5, s11), s17), s23), s29);
            t6 = svmla_f32_x(pg, svsub_f32_x(pg, s6, s12), _2, svsub_f32_x(pg, s18, s24));
            t7 = svmla_f32_x(pg, svsub_f32_x(pg, s7, s13), _2, svsub_f32_x(pg, s19, s25));
            t8 = svmla_f32_x(pg, svsub_f32_x(pg, s8, s14), _2, svsub_f32_x(pg, s20, s26));
            t9 = svmla_f32_x(pg, svsub_f32_x(pg, s9, s15), _2, svsub_f32_x(pg, s21, s27));
            t10 = svmla_f32_x(pg, svsub_f32_x(pg, s10, s16), _2, svsub_f32_x(pg, s22, s28));
            t11 = svmla_f32_x(pg, svsub_f32_x(pg, s11, s17), _2, svsub_f32_x(pg, s23, s29));
            t12 = svmla_f32_x(pg, svadd_f32_x(pg, s6, s12), _4, svadd_f32_x(pg, s18, s24));
            t13 = svmla_f32_x(pg, svadd_f32_x(pg, s7, s13), _4, svadd_f32_x(pg, s19, s25));
            t14 = svmla_f32_x(pg, svadd_f32_x(pg, s8, s14), _4, svadd_f32_x(pg, s20, s26));
            t15 = svmla_f32_x(pg, svadd_f32_x(pg, s9, s15), _4, svadd_f32_x(pg, s21, s27));
            t16 = svmla_f32_x(pg, svadd_f32_x(pg, s10, s16), _4, svadd_f32_x(pg, s22, s28));
            t17 = svmla_f32_x(pg, svadd_f32_x(pg, s11, s17), _4, svadd_f32_x(pg, s23, s29));
            t18 = svadd_f32_x(pg, svmla_f32_x(pg, svsub_f32_x(pg, s6, s12), _8, svsub_f32_x(pg, s18, s24)), s30);
            t19 = svadd_f32_x(pg, svmla_f32_x(pg, svsub_f32_x(pg, s7, s13), _8, svsub_f32_x(pg, s19, s25)), s31);
            t20 = svadd_f32_x(pg, svmla_f32_x(pg, svsub_f32_x(pg, s8, s14), _8, svsub_f32_x(pg, s20, s26)), s32);
            t21 = svadd_f32_x(pg, svmla_f32_x(pg, svsub_f32_x(pg, s9, s15), _8, svsub_f32_x(pg, s21, s27)), s33);
            t22 = svadd_f32_x(pg, svmla_f32_x(pg, svsub_f32_x(pg, s10, s16), _8, svsub_f32_x(pg, s22, s28)), s34);
            t23 = svadd_f32_x(pg, svmla_f32_x(pg, svsub_f32_x(pg, s11, s17), _8, svsub_f32_x(pg, s23, s29)), s35);

            dst0 = svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, t0, t1), t2), t3), t4);
            dst1 = svmla_f32_x(pg, svsub_f32_x(pg, t1, t2), _2, svsub_f32_x(pg, t3, t4));
            dst2 = svmla_f32_x(pg, svadd_f32_x(pg, t1, t2), _4, svadd_f32_x(pg, t3, t4));
            dst3 = svadd_f32_x(pg, svmla_f32_x(pg, svsub_f32_x(pg, t1, t2), _8, svsub_f32_x(pg, t3, t4)), t5);
            dst4 = svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, t6, t7), t8), t9), t10);
            dst5 = svmla_f32_x(pg, svsub_f32_x(pg, t7, t8), _2, svsub_f32_x(pg, t9, t10));
            dst6 = svmla_f32_x(pg, svadd_f32_x(pg, t7, t8), _4, svadd_f32_x(pg, t9, t10));
            dst7 = svadd_f32_x(pg, svmla_f32_x(pg, svsub_f32_x(pg, t7, t8), _8, svsub_f32_x(pg, t9, t10)), t11);
            dst8 = svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, t12, t13), t14), t15), t16);
            dst9 = svmla_f32_x(pg, svsub_f32_x(pg, t13, t14), _2, svsub_f32_x(pg, t15, t16));
            dst10 = svmla_f32_x(pg, svadd_f32_x(pg, t13, t14), _4, svadd_f32_x(pg, t15, t16));
            dst11 = svadd_f32_x(pg, svmla_f32_x(pg, svsub_f32_x(pg, t13, t14), _8, svsub_f32_x(pg, t15, t16)), t17);
            dst12 = svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, svadd_f32_x(pg, t18, t19), t20), t21), t22);
            dst13 = svmla_f32_x(pg, svsub_f32_x(pg, t19, t20), _2, svsub_f32_x(pg, t21, t22));
            dst14 = svmla_f32_x(pg, svadd_f32_x(pg, t19, t20), _4, svadd_f32_x(pg, t21, t22));
            dst15 = svadd_f32_x(pg, svmla_f32_x(pg, svsub_f32_x(pg, t19, t20), _8, svsub_f32_x(pg, t21, t22)), t23);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetOutputStore16(
            const svfloat32_t& src0, const svfloat32_t& src1, const svfloat32_t& src2, const svfloat32_t& src3,
            const svfloat32_t& src4, const svfloat32_t& src5, const svfloat32_t& src6, const svfloat32_t& src7,
            const svfloat32_t& src8, const svfloat32_t& src9, const svfloat32_t& src10, const svfloat32_t& src11,
            const svfloat32_t& src12, const svfloat32_t& src13, const svfloat32_t& src14, const svfloat32_t& src15,
            float* dst, size_t dstS, size_t dstC, const svbool_t& pg)
        {
            svst1_f32(pg, dst + 0 * dstS + 0 * dstC, src0);
            svst1_f32(pg, dst + 0 * dstS + 1 * dstC, src1);
            svst1_f32(pg, dst + 0 * dstS + 2 * dstC, src2);
            svst1_f32(pg, dst + 0 * dstS + 3 * dstC, src3);
            svst1_f32(pg, dst + 1 * dstS + 0 * dstC, src4);
            svst1_f32(pg, dst + 1 * dstS + 1 * dstC, src5);
            svst1_f32(pg, dst + 1 * dstS + 2 * dstC, src6);
            svst1_f32(pg, dst + 1 * dstS + 3 * dstC, src7);
            svst1_f32(pg, dst + 2 * dstS + 0 * dstC, src8);
            svst1_f32(pg, dst + 2 * dstS + 1 * dstC, src9);
            svst1_f32(pg, dst + 2 * dstS + 2 * dstC, src10);
            svst1_f32(pg, dst + 2 * dstS + 3 * dstC, src11);
            svst1_f32(pg, dst + 3 * dstS + 0 * dstC, src12);
            svst1_f32(pg, dst + 3 * dstS + 1 * dstC, src13);
            svst1_f32(pg, dst + 3 * dstS + 2 * dstC, src14);
            svst1_f32(pg, dst + 3 * dstS + 3 * dstC, src15);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetOutputStore16(
            const svfloat32_t& src0, const svfloat32_t& src1, const svfloat32_t& src2, const svfloat32_t& src3,
            const svfloat32_t& src4, const svfloat32_t& src5, const svfloat32_t& src6, const svfloat32_t& src7,
            const svfloat32_t& src8, const svfloat32_t& src9, const svfloat32_t& src10, const svfloat32_t& src11,
            const svfloat32_t& src12, const svfloat32_t& src13, const svfloat32_t& src14, const svfloat32_t& src15,
            float* dst, size_t dstS, size_t dstC, size_t rowE, size_t colE, const svbool_t& pg)
        {
            if (rowE > 0 && colE > 0)
                svst1_f32(pg, dst + 0 * dstS + 0 * dstC, src0);
            if (rowE > 0 && colE > 1)
                svst1_f32(pg, dst + 0 * dstS + 1 * dstC, src1);
            if (rowE > 0 && colE > 2)
                svst1_f32(pg, dst + 0 * dstS + 2 * dstC, src2);
            if (rowE > 0 && colE > 3)
                svst1_f32(pg, dst + 0 * dstS + 3 * dstC, src3);
            if (rowE > 1 && colE > 0)
                svst1_f32(pg, dst + 1 * dstS + 0 * dstC, src4);
            if (rowE > 1 && colE > 1)
                svst1_f32(pg, dst + 1 * dstS + 1 * dstC, src5);
            if (rowE > 1 && colE > 2)
                svst1_f32(pg, dst + 1 * dstS + 2 * dstC, src6);
            if (rowE > 1 && colE > 3)
                svst1_f32(pg, dst + 1 * dstS + 3 * dstC, src7);
            if (rowE > 2 && colE > 0)
                svst1_f32(pg, dst + 2 * dstS + 0 * dstC, src8);
            if (rowE > 2 && colE > 1)
                svst1_f32(pg, dst + 2 * dstS + 1 * dstC, src9);
            if (rowE > 2 && colE > 2)
                svst1_f32(pg, dst + 2 * dstS + 2 * dstC, src10);
            if (rowE > 2 && colE > 3)
                svst1_f32(pg, dst + 2 * dstS + 3 * dstC, src11);
            if (rowE > 3 && colE > 0)
                svst1_f32(pg, dst + 3 * dstS + 0 * dstC, src12);
            if (rowE > 3 && colE > 1)
                svst1_f32(pg, dst + 3 * dstS + 1 * dstC, src13);
            if (rowE > 3 && colE > 2)
                svst1_f32(pg, dst + 3 * dstS + 2 * dstC, src14);
            if (rowE > 3 && colE > 3)
                svst1_f32(pg, dst + 3 * dstS + 3 * dstC, src15);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetOutputT(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC)
        {
            const size_t F = svcntw();
            const size_t dstCF = AlignLo(dstC, F);
            const svbool_t body = svptrue_b32();
            const size_t dstS = dstW * dstC;
            size_t d = 0;
            for (; d < dstCF; d += F)
            {
                svfloat32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
                WinogradKernel3x3Block4x4SetOutputLoad36(src + d, srcStride, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, body);
                WinogradKernel3x3Block4x4SetOutputStore16(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, dst + d, dstS, dstC, body);
            }
            if (d < dstC)
            {
                svbool_t tail = svwhilelt_b32(d, dstC);
                svfloat32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
                WinogradKernel3x3Block4x4SetOutputLoad36(src + d, srcStride, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tail);
                WinogradKernel3x3Block4x4SetOutputStore16(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, dst + d, dstS, dstC, tail);
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetOutputT(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC, size_t rowE, size_t colE)
        {
            const size_t F = svcntw();
            const size_t dstCF = AlignLo(dstC, F);
            const svbool_t body = svptrue_b32();
            const size_t dstS = dstW * dstC;
            size_t d = 0;
            for (; d < dstCF; d += F)
            {
                svfloat32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
                WinogradKernel3x3Block4x4SetOutputLoad36(src + d, srcStride, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, body);
                WinogradKernel3x3Block4x4SetOutputStore16(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, dst + d, dstS, dstC, rowE, colE, body);
            }
            if (d < dstC)
            {
                svbool_t tail = svwhilelt_b32(d, dstC);
                svfloat32_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
                WinogradKernel3x3Block4x4SetOutputLoad36(src + d, srcStride, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, tail);
                WinogradKernel3x3Block4x4SetOutputStore16(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15, dst + d, dstS, dstC, rowE, colE, tail);
            }
        }

        void WinogradKernel3x3Block4x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            if (!trans)
            {
                Base::WinogradKernel3x3Block4x4SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
                return;
            }
            size_t dstH4 = AlignLo(dstHeight, 4);
            size_t dstW4 = AlignLo(dstWidth, 4);
            size_t row, col;
            for (row = 0; row < dstH4; row += 4)
            {
                for (col = 0; col < dstW4; col += 4)
                    WinogradKernel3x3Block4x4SetOutputT(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels), src += dstChannels;
                if (col < dstWidth)
                    WinogradKernel3x3Block4x4SetOutputT(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, 4, dstWidth - col), src += dstChannels;
            }
            if (row < dstHeight)
            {
                for (col = 0; col < dstW4; col += 4)
                    WinogradKernel3x3Block4x4SetOutputT(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row, 4), src += dstChannels;
                if (col < dstWidth)
                    WinogradKernel3x3Block4x4SetOutputT(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row, dstWidth - col), src += dstChannels;
            }
        }
    }
#endif
}
