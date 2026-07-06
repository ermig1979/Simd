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
    }
#endif
}
