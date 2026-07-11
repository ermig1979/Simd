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
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE svfloat32_t Poly5(const svbool_t& mask, svfloat32_t x)
        {
            svfloat32_t p = svdup_n_f32(1.8775767e-3f);
            p = svmla_f32_x(mask, svdup_n_f32(8.9893397e-3f), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(5.5826318e-2f), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(2.4015361e-1f), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(6.9315308e-1f), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(9.9999994e-1f), x, p);
            return p;
        }

        SIMD_INLINE svfloat32_t Exp2(const svbool_t& mask, svfloat32_t x)
        {
            x = svmax_f32_x(mask, svmin_f32_x(mask, x, svdup_n_f32(126.99999f)), svdup_n_f32(-126.99999f));
            svint32_t ipart = svcvt_s32_f32_x(mask, svsub_n_f32_x(mask, x, 0.5f));
            svfloat32_t fpart = svsub_f32_x(mask, x, svcvt_f32_s32_x(mask, ipart));
            svfloat32_t expipart = svreinterpret_f32_s32(svlsl_n_s32_x(mask, svadd_n_s32_x(mask, ipart, 127), 23));
            svfloat32_t expfpart = Poly5(mask, fpart);
            return svmul_f32_x(mask, expipart, expfpart);
        }

        SIMD_INLINE svfloat32_t Exponent(const svbool_t& mask, svfloat32_t value)
        {
            return Exp2(mask, svmul_n_f32_x(mask, value, 1.44269504f));
        }

        SIMD_INLINE svfloat32_t Elu(const svbool_t& mask, svfloat32_t value, svfloat32_t alpha)
        {
            svfloat32_t exp = Exponent(mask, value);
            svfloat32_t neg = svmul_f32_x(mask, alpha, svsub_n_f32_x(mask, exp, 1.0f));
            return svsel_f32(svcmplt_n_f32(mask, value, 0.0f), neg, value);
        }

        SIMD_INLINE void SynetElu32f(const float* src, const svbool_t& mask, svfloat32_t alpha, float* dst)
        {
            svst1_f32(mask, dst, Elu(mask, svld1_f32(mask, src), alpha));
        }

        void SynetElu32f(const float* src, size_t size, const float* alpha, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _alpha = svdup_n_f32(alpha[0]);
            for (; i + QF <= size; i += QF)
            {
                SynetElu32f(src + i + 0 * F, body, _alpha, dst + i + 0 * F);
                SynetElu32f(src + i + 1 * F, body, _alpha, dst + i + 1 * F);
                SynetElu32f(src + i + 2 * F, body, _alpha, dst + i + 2 * F);
                SynetElu32f(src + i + 3 * F, body, _alpha, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                SynetElu32f(src + i, body, _alpha, dst + i);
            if (i < size)
                SynetElu32f(src + i, svwhilelt_b32(i, size), _alpha, dst + i);
        }
    }
#endif
}
