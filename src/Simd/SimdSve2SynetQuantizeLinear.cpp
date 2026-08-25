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
#include "Simd/SimdBase.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdSynetQuantizeLinear.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        void SynetDequantizeLinear(const uint8_t* src, size_t size, int32_t bias, const float* norm, float* dst)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            svint32_t _bias = svdup_n_s32(bias);
            svfloat32_t _norm = svdup_n_f32(norm[0]);
            size_t i = 0;
            for (; i + QF <= size; i += QF)
            {
                DequantizeLinear(src + i + 0 * F, _bias, _norm, dst + i + 0 * F, body);
                DequantizeLinear(src + i + 1 * F, _bias, _norm, dst + i + 1 * F, body);
                DequantizeLinear(src + i + 2 * F, _bias, _norm, dst + i + 2 * F, body);
                DequantizeLinear(src + i + 3 * F, _bias, _norm, dst + i + 3 * F, body);
            }
            for (; i + F <= size; i += F)
                DequantizeLinear(src + i, _bias, _norm, dst + i, body);
            if (i < size)
                DequantizeLinear(src + i, _bias, _norm, dst + i, svwhilelt_b32(i, size));
        }

        //-------------------------------------------------------------------------------------------------

        void SynetQuantizeLinear(const float* src, size_t size, const float* norm, int32_t zero, uint8_t* dst)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            svfloat32_t _norm = svdup_n_f32(norm[0]);
            svint32_t _zero = svdup_n_s32(zero);
            size_t i = 0;
            for (; i + QF <= size; i += QF)
            {
                QuantizeLinear(src + i + 0 * F, _norm, _zero, dst + i + 0 * F, body);
                QuantizeLinear(src + i + 1 * F, _norm, _zero, dst + i + 1 * F, body);
                QuantizeLinear(src + i + 2 * F, _norm, _zero, dst + i + 2 * F, body);
                QuantizeLinear(src + i + 3 * F, _norm, _zero, dst + i + 3 * F, body);
            }
            for (; i + F <= size; i += F)
                QuantizeLinear(src + i, _norm, _zero, dst + i, body);
            if (i < size)
                QuantizeLinear(src + i, _norm, _zero, dst + i, svwhilelt_b32(i, size));
        }
    }
#endif
}
