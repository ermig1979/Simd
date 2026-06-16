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

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE void CosineDistance16f(const uint16_t* a, const uint16_t* b, const svbool_t& load, const svbool_t& lo, const svbool_t& hi,
            svfloat32_t& aa0, svfloat32_t& aa1, svfloat32_t& ab0, svfloat32_t& ab1, svfloat32_t& bb0, svfloat32_t& bb1)
        {
            svfloat16_t _a = svreinterpret_f16_u16(svld1_u16(load, a));
            svfloat16_t _b = svreinterpret_f16_u16(svld1_u16(load, b));
            svfloat32_t a0 = svcvt_f32_f16_x(lo, _a);
            svfloat32_t b0 = svcvt_f32_f16_x(lo, _b);
            aa0 = svmla_f32_x(lo, aa0, a0, a0);
            ab0 = svmla_f32_x(lo, ab0, a0, b0);
            bb0 = svmla_f32_x(lo, bb0, b0, b0);
            svfloat32_t a1 = svcvtlt_f32_f16_x(hi, _a);
            svfloat32_t b1 = svcvtlt_f32_f16_x(hi, _b);
            aa1 = svmla_f32_x(hi, aa1, a1, a1);
            ab1 = svmla_f32_x(hi, ab1, a1, b1);
            bb1 = svmla_f32_x(hi, bb1, b1, b1);
        }

        void CosineDistance16f(const uint16_t* a, const uint16_t* b, size_t size, float* distance)
        {
            size_t A = svlen(svuint16_t()), sizeA = AlignLo(size, A), i = 0;
            const svbool_t body16 = svptrue_b16();
            const svbool_t body32 = svptrue_b32();
            svfloat32_t aa0 = svdup_n_f32(0.0f), aa1 = svdup_n_f32(0.0f);
            svfloat32_t ab0 = svdup_n_f32(0.0f), ab1 = svdup_n_f32(0.0f);
            svfloat32_t bb0 = svdup_n_f32(0.0f), bb1 = svdup_n_f32(0.0f);
            for (; i < sizeA; i += A)
                CosineDistance16f(a + i, b + i, body16, body32, body32, aa0, aa1, ab0, ab1, bb0, bb1);
            if (i < size)
            {
                size_t tail = size - i;
                CosineDistance16f(a + i, b + i, svwhilelt_b16(size_t(0), tail),
                    svwhilelt_b32(size_t(0), (tail + 1) / 2), svwhilelt_b32(size_t(0), tail / 2), aa0, aa1, ab0, ab1, bb0, bb1);
            }
            float _aa = svaddv_f32(body32, svadd_f32_x(body32, aa0, aa1));
            float _ab = svaddv_f32(body32, svadd_f32_x(body32, ab0, ab1));
            float _bb = svaddv_f32(body32, svadd_f32_x(body32, bb0, bb1));
            *distance = 1.0f - _ab / ::sqrt(_aa * _bb);
        }
    }
#endif
}
