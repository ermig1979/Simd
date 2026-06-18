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
        void Fill32f(float* dst, size_t size, const float* value)
        {
            if (value == 0 || value[0] == 0)
                memset(dst, 0, size * sizeof(float));
            else
            {
                size_t F = svcntw(), QF = 4 * F, i = 0;
                const svbool_t body = svptrue_b32();
                const svfloat32_t _value = svdup_n_f32(value[0]);

                for (; i + QF <= size; i += QF)
                {
                    svst1_f32(body, dst + i + 0 * F, _value);
                    svst1_f32(body, dst + i + 1 * F, _value);
                    svst1_f32(body, dst + i + 2 * F, _value);
                    svst1_f32(body, dst + i + 3 * F, _value);
                }
                for (; i + F <= size; i += F)
                    svst1_f32(body, dst + i, _value);
                if (i < size)
                    svst1_f32(svwhilelt_b32(i, size), dst + i, _value);
            }
        }
    }
#endif
}
