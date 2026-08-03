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
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svuint32_t Float32ToUint8(const float* src, const svbool_t& mask, const svfloat32_t& lower, const svfloat32_t& upper, const svfloat32_t& boost)
        {
            svfloat32_t value = svld1_f32(mask, src);
            value = svmul_f32_x(mask, svsub_f32_x(mask, svmin_f32_x(mask, svmax_f32_x(mask, value, lower), upper), lower), boost);
            return svcvt_u32_f32_x(mask, value);
        }

        void Float32ToUint8(const float* src, size_t size, const float* lower, const float* upper, uint8_t* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _lower = svdup_n_f32(lower[0]);
            const svfloat32_t _upper = svdup_n_f32(upper[0]);
            const svfloat32_t boost = svdup_n_f32(255.0f / (upper[0] - lower[0]));

            for (; i + QF <= size; i += QF)
            {
                svst1b_u32(body, dst + i + 0 * F, Float32ToUint8(src + i + 0 * F, body, _lower, _upper, boost));
                svst1b_u32(body, dst + i + 1 * F, Float32ToUint8(src + i + 1 * F, body, _lower, _upper, boost));
                svst1b_u32(body, dst + i + 2 * F, Float32ToUint8(src + i + 2 * F, body, _lower, _upper, boost));
                svst1b_u32(body, dst + i + 3 * F, Float32ToUint8(src + i + 3 * F, body, _lower, _upper, boost));
            }
            for (; i + F <= size; i += F)
                svst1b_u32(body, dst + i, Float32ToUint8(src + i, body, _lower, _upper, boost));
            if (i < size)
            {
                svbool_t tail = svwhilelt_b32(i, size);
                svst1b_u32(tail, dst + i, Float32ToUint8(src + i, tail, _lower, _upper, boost));
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t Uint8ToFloat32(const uint8_t* src, const svbool_t& mask, const svfloat32_t& lower, const svfloat32_t& boost)
        {
            svfloat32_t value = svcvt_f32_u32_x(mask, svld1ub_u32(mask, src));
            return svmla_f32_x(mask, lower, value, boost);
        }

        void Uint8ToFloat32(const uint8_t* src, size_t size, const float* lower, const float* upper, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _lower = svdup_n_f32(lower[0]);
            const svfloat32_t boost = svdup_n_f32((upper[0] - lower[0]) / 255.0f);

            for (; i + QF <= size; i += QF)
            {
                svst1_f32(body, dst + i + 0 * F, Uint8ToFloat32(src + i + 0 * F, body, _lower, boost));
                svst1_f32(body, dst + i + 1 * F, Uint8ToFloat32(src + i + 1 * F, body, _lower, boost));
                svst1_f32(body, dst + i + 2 * F, Uint8ToFloat32(src + i + 2 * F, body, _lower, boost));
                svst1_f32(body, dst + i + 3 * F, Uint8ToFloat32(src + i + 3 * F, body, _lower, boost));
            }
            for (; i + F <= size; i += F)
                svst1_f32(body, dst + i, Uint8ToFloat32(src + i, body, _lower, boost));
            if (i < size)
            {
                svbool_t tail = svwhilelt_b32(i, size);
                svst1_f32(tail, dst + i, Uint8ToFloat32(src + i, tail, _lower, boost));
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void CosineDistance32f(const float* a, const float* b, const svbool_t& mask,
            svfloat32_t& aa, svfloat32_t& ab, svfloat32_t& bb)
        {
            svfloat32_t _a = svld1_f32(mask, a);
            svfloat32_t _b = svld1_f32(mask, b);
            aa = svmla_f32_m(mask, aa, _a, _a);
            ab = svmla_f32_m(mask, ab, _a, _b);
            bb = svmla_f32_m(mask, bb, _b, _b);
        }

        void CosineDistance32f(const float* a, const float* b, size_t size, float* distance)
        {
            size_t F = svcntw(), DF = 2 * F, sizeDF = AlignLo(size, DF), sizeF = AlignLo(size, F), i = 0;
            const svbool_t body = svptrue_b32();
            svfloat32_t aa0 = svdup_n_f32(0.0f), aa1 = svdup_n_f32(0.0f);
            svfloat32_t ab0 = svdup_n_f32(0.0f), ab1 = svdup_n_f32(0.0f);
            svfloat32_t bb0 = svdup_n_f32(0.0f), bb1 = svdup_n_f32(0.0f);
            for (; i < sizeDF; i += DF)
            {
                CosineDistance32f(a + i, b + i, body, aa0, ab0, bb0);
                CosineDistance32f(a + i + F, b + i + F, body, aa1, ab1, bb1);
            }
            aa0 = svadd_f32_x(body, aa0, aa1);
            ab0 = svadd_f32_x(body, ab0, ab1);
            bb0 = svadd_f32_x(body, bb0, bb1);
            for (; i < sizeF; i += F)
                CosineDistance32f(a + i, b + i, body, aa0, ab0, bb0);
            if (i < size)
                CosineDistance32f(a + i, b + i, svwhilelt_b32(i, size), aa0, ab0, bb0);
            float _aa = svaddv_f32(body, aa0);
            float _ab = svaddv_f32(body, ab0);
            float _bb = svaddv_f32(body, bb0);
            *distance = 1.0f - _ab / ::sqrt(_aa * _bb);
        }
    }
#endif
}
