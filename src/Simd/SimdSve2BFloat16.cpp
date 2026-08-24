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
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svbfloat16_t Float32ToBFloat16(svfloat32_t even, svfloat32_t odd, const svbool_t& mask)
        {
            return svcvtnt_bf16_f32_x(svcvt_bf16_f32_x(mask, even), mask, odd);
        }

        SIMD_INLINE void Float32ToBFloat16(const float* src, const svbool_t& lo, const svbool_t& hi, const svbool_t& store, uint16_t* dst)
        {
            size_t F = svcntw();
            svfloat32_t s0 = svld1_f32(lo, src + 0);
            svfloat32_t s1 = svld1_f32(hi, src + F);
            svst1_u16(store, dst, svreinterpret_u16_bf16(Float32ToBFloat16(svuzp1_f32(s0, s1), svuzp2_f32(s0, s1), svptrue_b32())));
        }

        void Float32ToBFloat16(const float* src, size_t size, uint16_t* dst)
        {
            size_t A = svcnth(), F = svcntw(), QA = A * 4, sizeQA = AlignLo(size, QA), sizeA = AlignLo(size, A), i = 0;
            const svbool_t body16 = svptrue_b16();
            const svbool_t body32 = svptrue_b32();
            for (; i < sizeQA; i += QA)
            {
                Float32ToBFloat16(src + i + 0 * A, body32, body32, body16, dst + i + 0 * A);
                Float32ToBFloat16(src + i + 1 * A, body32, body32, body16, dst + i + 1 * A);
                Float32ToBFloat16(src + i + 2 * A, body32, body32, body16, dst + i + 2 * A);
                Float32ToBFloat16(src + i + 3 * A, body32, body32, body16, dst + i + 3 * A);
            }
            for (; i < sizeA; i += A)
                Float32ToBFloat16(src + i, body32, body32, body16, dst + i);
            if (i < size)
            {
                size_t tail = size - i;
                Float32ToBFloat16(src + i, svwhilelt_b32(size_t(0), Simd::Min(tail, F)),
                    svwhilelt_b32(F, tail), svwhilelt_b16(size_t(0), tail), dst + i);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void BFloat16ToFloat32(const uint16_t* src, const svbool_t& load, const svuint16_t &zero, const svbool_t& lo, const svbool_t& hi, float* dst)
        {
            svuint16_t _src = svld1_u16(load, src);
            svst1_vnum_u16(lo, (uint16_t*)dst, 0, svzip1_u16(zero, _src));
            svst1_vnum_u16(hi, (uint16_t*)dst, 1, svzip2_u16(zero, _src));
        }

        void BFloat16ToFloat32(const uint16_t* src, size_t size, float* dst)
        {
            size_t A = svlen(svuint16_t()), sizeA = AlignLo(size, A);
            const svbool_t body = svptrue_b16();
            svuint16_t zero = svdup_n_u16(0);
            size_t i = 0;
            for (; i < sizeA; i += A)
                BFloat16ToFloat32(src + i, body, zero, body, body, dst + i);
            if (i < size)
            {
                size_t tail = size - i, half = 2 * tail;
                BFloat16ToFloat32(src + i, svwhilelt_b16(size_t(0), tail), zero, 
                    svwhilelt_b16(size_t(0), Simd::Min(half, A)), svwhilelt_b16(A, half), dst + i);
            }
        }
    }
#endif
}
