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
        SIMD_INLINE void AddMultiplied(const svbool_t& mask, const svfloat32_t& value, const float* src, float* dst)
        {
            svst1_f32(mask, dst, svmla_f32_m(mask, svld1_f32(mask, dst), svld1_f32(mask, src), value));
        }

        void NeuralAddVectorMultipliedByValue(const float* src, size_t size, const float* value, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _value = svdup_n_f32(value[0]);

            for (; i + QF <= size; i += QF)
            {
                AddMultiplied(body, _value, src + i + 0 * F, dst + i + 0 * F);
                AddMultiplied(body, _value, src + i + 1 * F, dst + i + 1 * F);
                AddMultiplied(body, _value, src + i + 2 * F, dst + i + 2 * F);
                AddMultiplied(body, _value, src + i + 3 * F, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                AddMultiplied(body, _value, src + i, dst + i);
            if (i < size)
                AddMultiplied(svwhilelt_b32(i, size), _value, src + i, dst + i);
        }

        SIMD_INLINE void AddVector(const svbool_t& mask, const float* src, float* dst)
        {
            svst1_f32(mask, dst, svadd_f32_x(mask, svld1_f32(mask, dst), svld1_f32(mask, src)));
        }

        void NeuralAddVector(const float* src, size_t size, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();

            for (; i + QF <= size; i += QF)
            {
                AddVector(body, src + i + 0 * F, dst + i + 0 * F);
                AddVector(body, src + i + 1 * F, dst + i + 1 * F);
                AddVector(body, src + i + 2 * F, dst + i + 2 * F);
                AddVector(body, src + i + 3 * F, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                AddVector(body, src + i, dst + i);
            if (i < size)
                AddVector(svwhilelt_b32(i, size), src + i, dst + i);
        }

        SIMD_INLINE void AddValue(const svbool_t& mask, const svfloat32_t& value, float* dst)
        {
            svst1_f32(mask, dst, svadd_f32_x(mask, svld1_f32(mask, dst), value));
        }

        void NeuralAddValue(const float* value, float* dst, size_t size)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _value = svdup_n_f32(value[0]);

            for (; i + QF <= size; i += QF)
            {
                AddValue(body, _value, dst + i + 0 * F);
                AddValue(body, _value, dst + i + 1 * F);
                AddValue(body, _value, dst + i + 2 * F);
                AddValue(body, _value, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                AddValue(body, _value, dst + i);
            if (i < size)
                AddValue(svwhilelt_b32(i, size), _value, dst + i);
        }
    }
#endif
}
