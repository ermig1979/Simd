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
    }
#endif
}
