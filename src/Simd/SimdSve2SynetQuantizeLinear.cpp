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

#include "Simd/SimdSve2.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE void SynetDequantizeLinear(const uint8_t* src, const svint32_t& bias, const svfloat32_t& norm, float* dst, const svbool_t& mask)
        {
            svint32_t value = svadd_s32_x(mask, svreinterpret_s32_u32(svld1ub_u32(mask, src)), bias);
            svst1_f32(mask, dst, svmul_f32_x(mask, svcvt_f32_s32_x(mask, value), norm));
        }

        void SynetDequantizeLinear(const uint8_t* src, size_t size, int32_t bias, const float* norm, float* dst)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            svint32_t _bias = svdup_n_s32(bias);
            svfloat32_t _norm = svdup_n_f32(norm[0]);
            size_t i = 0;
            for (; i + QF <= size; i += QF)
            {
                SynetDequantizeLinear(src + i + 0 * F, _bias, _norm, dst + i + 0 * F, body);
                SynetDequantizeLinear(src + i + 1 * F, _bias, _norm, dst + i + 1 * F, body);
                SynetDequantizeLinear(src + i + 2 * F, _bias, _norm, dst + i + 2 * F, body);
                SynetDequantizeLinear(src + i + 3 * F, _bias, _norm, dst + i + 3 * F, body);
            }
            for (; i < size; i += F)
                SynetDequantizeLinear(src + i, _bias, _norm, dst + i, svwhilelt_b32(i, size));
        }
    }
#endif
}
