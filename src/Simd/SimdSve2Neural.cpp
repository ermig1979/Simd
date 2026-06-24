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
