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
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE svfloat32_t UnaryPoly5(const svbool_t& mask, svfloat32_t x, float a0, float a1, float a2, float a3, float a4, float a5)
        {
            svfloat32_t p = svdup_n_f32(a5);
            p = svmla_f32_x(mask, svdup_n_f32(a4), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(a3), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(a2), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(a1), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(a0), x, p);
            return p;
        }

        SIMD_INLINE svfloat32_t UnaryErf(const svbool_t& mask, svfloat32_t x)
        {
            const svfloat32_t _1 = svdup_n_f32(1.0f);
            svfloat32_t a = svmin_f32_x(mask, svabs_f32_x(mask, x), svdup_n_f32(9.0f));
            svfloat32_t p = svdup_n_f32(0.0000430638f);
            p = svmla_f32_x(mask, svdup_n_f32(0.0002765672f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0001520143f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0092705272f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0422820123f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0705230784f), a, p);
            p = svmla_f32_x(mask, _1, a, p);
            p = svmul_f32_x(mask, p, p);
            p = svmul_f32_x(mask, p, p);
            p = svmul_f32_x(mask, p, p);
            p = svmul_f32_x(mask, p, p);
            svfloat32_t r = svsub_f32_x(mask, _1, svdiv_f32_x(mask, _1, p));
            return svsel_f32(svcmplt_n_f32(mask, x, 0.0f), svneg_f32_x(mask, r), r);
        }

        namespace UnaryDetail
        {
            SIMD_INLINE svfloat32_t NormSin(const svbool_t& mask, svfloat32_t x)
            {
                const svfloat32_t _1 = svdup_n_f32(1.0f);
                svfloat32_t p = UnaryPoly5(mask, svmul_f32_x(mask, x, x), -3.1415926444234477f, 2.0261194642649887f, -0.5240361513980939f, 0.0751872634325299f, -0.006860187425683514f, 0.000385937753182769f);
                return svmul_f32_x(mask, svmul_f32_x(mask, svsub_f32_x(mask, x, _1), svadd_f32_x(mask, x, _1)), svmul_f32_x(mask, p, x));
            }
        }

        SIMD_INLINE svfloat32_t UnarySin(const svbool_t& mask, svfloat32_t x)
        {
            x = svmul_n_f32_x(mask, x, float(M_1_PI));
            svfloat32_t f = svrintm_f32_x(mask, x);
            svfloat32_t s = UnaryDetail::NormSin(mask, svsub_f32_x(mask, x, f));
            svuint32_t n = svlsl_n_u32_x(mask, svreinterpret_u32_s32(svcvt_s32_f32_x(mask, f)), 31);
            return svreinterpret_f32_u32(svorr_u32_x(mask, svreinterpret_u32_f32(s), n));
        }

        SIMD_INLINE svfloat32_t UnaryCos(const svbool_t& mask, svfloat32_t x)
        {
            return UnarySin(mask, svsub_f32_x(mask, svdup_n_f32(float(M_PI_2)), x));
        }

        template<SimdSynetUnaryOperation32fType type> svfloat32_t SynetUnaryOperation32f(const svbool_t& mask, svfloat32_t value);

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fAbs>(const svbool_t& mask, svfloat32_t value)
        {
            return svabs_f32_x(mask, value);
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fCeil>(const svbool_t& mask, svfloat32_t value)
        {
            return svrintp_f32_x(mask, value);
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fCos>(const svbool_t& mask, svfloat32_t value)
        {
            return UnaryCos(mask, value);
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fErf>(const svbool_t& mask, svfloat32_t value)
        {
            return UnaryErf(mask, value);
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fExp>(const svbool_t& mask, svfloat32_t value)
        {
            return Exponent(mask, value);
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fFloor>(const svbool_t& mask, svfloat32_t value)
        {
            return svrintm_f32_x(mask, value);
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fLog>(const svbool_t& mask, svfloat32_t value)
        {
            return Logarithm(mask, value);
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fNeg>(const svbool_t& mask, svfloat32_t value)
        {
            return svneg_f32_x(mask, value);
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fNot>(const svbool_t& mask, svfloat32_t value)
        {
            return svreinterpret_f32_u32(svnot_u32_x(mask, svreinterpret_u32_f32(value)));
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fRcp>(const svbool_t& mask, svfloat32_t value)
        {
            return svdiv_f32_x(mask, svdup_n_f32(1.0f), value);
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fRound>(const svbool_t& mask, svfloat32_t value)
        {
            svbool_t positive = svcmpge_n_f32(mask, value, 0.0f);
            svfloat32_t round = svsel_f32(positive, svdup_n_f32(0.5f), svdup_n_f32(-0.5f));
            return svcvt_f32_s32_x(mask, svcvt_s32_f32_x(mask, svadd_f32_x(mask, value, round)));
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fRsqrt>(const svbool_t& mask, svfloat32_t value)
        {
            return svdiv_f32_x(mask, svdup_n_f32(1.0f), svsqrt_f32_x(mask, value));
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fSign>(const svbool_t& mask, svfloat32_t value)
        {
            svfloat32_t zero = svdup_n_f32(0.0f), positive = svdup_n_f32(1.0f), negative = svdup_n_f32(-1.0f);
            return svsel_f32(svcmplt_n_f32(mask, value, 0.0f), negative, svsel_f32(svcmpeq_n_f32(mask, value, 0.0f), zero, positive));
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fSin>(const svbool_t& mask, svfloat32_t value)
        {
            return UnarySin(mask, value);
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fSqrt>(const svbool_t& mask, svfloat32_t value)
        {
            return svsqrt_f32_x(mask, value);
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fTanh>(const svbool_t& mask, svfloat32_t value)
        {
            return Tanh(mask, value);
        }

        template<> SIMD_INLINE svfloat32_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fZero>(const svbool_t& mask, svfloat32_t value)
        {
            return svdup_n_f32(0.0f);
        }

        template<SimdSynetUnaryOperation32fType type> SIMD_INLINE void SynetUnaryOperation32f(const float* src, const svbool_t& mask, float* dst)
        {
            svst1_f32(mask, dst, SynetUnaryOperation32f<type>(mask, svld1_f32(mask, src)));
        }

        template<SimdSynetUnaryOperation32fType type> void SynetUnaryOperation32f(const float* src, size_t size, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            for (; i + QF <= size; i += QF)
            {
                SynetUnaryOperation32f<type>(src + i + 0 * F, body, dst + i + 0 * F);
                SynetUnaryOperation32f<type>(src + i + 1 * F, body, dst + i + 1 * F);
                SynetUnaryOperation32f<type>(src + i + 2 * F, body, dst + i + 2 * F);
                SynetUnaryOperation32f<type>(src + i + 3 * F, body, dst + i + 3 * F);
            }
            for (; i < size; i += F)
                SynetUnaryOperation32f<type>(src + i, svwhilelt_b32(i, size), dst + i);
        }

        void SynetUnaryOperation32f(const float* src, size_t size, SimdSynetUnaryOperation32fType type, float* dst)
        {
            switch (type)
            {
            case SimdSynetUnaryOperation32fAbs: SynetUnaryOperation32f<SimdSynetUnaryOperation32fAbs>(src, size, dst); break;
            case SimdSynetUnaryOperation32fCeil: SynetUnaryOperation32f<SimdSynetUnaryOperation32fCeil>(src, size, dst); break;
            case SimdSynetUnaryOperation32fCos: SynetUnaryOperation32f<SimdSynetUnaryOperation32fCos>(src, size, dst); break;
            case SimdSynetUnaryOperation32fExp: SynetUnaryOperation32f<SimdSynetUnaryOperation32fExp>(src, size, dst); break;
            case SimdSynetUnaryOperation32fErf: SynetUnaryOperation32f<SimdSynetUnaryOperation32fErf>(src, size, dst); break;
            case SimdSynetUnaryOperation32fFloor: SynetUnaryOperation32f<SimdSynetUnaryOperation32fFloor>(src, size, dst); break;
            case SimdSynetUnaryOperation32fLog: SynetUnaryOperation32f<SimdSynetUnaryOperation32fLog>(src, size, dst); break;
            case SimdSynetUnaryOperation32fNeg: SynetUnaryOperation32f<SimdSynetUnaryOperation32fNeg>(src, size, dst); break;
            case SimdSynetUnaryOperation32fNot: SynetUnaryOperation32f<SimdSynetUnaryOperation32fNot>(src, size, dst); break;
            case SimdSynetUnaryOperation32fRcp: SynetUnaryOperation32f<SimdSynetUnaryOperation32fRcp>(src, size, dst); break;
            case SimdSynetUnaryOperation32fRound: SynetUnaryOperation32f<SimdSynetUnaryOperation32fRound>(src, size, dst); break;
            case SimdSynetUnaryOperation32fRsqrt: SynetUnaryOperation32f<SimdSynetUnaryOperation32fRsqrt>(src, size, dst); break;
            case SimdSynetUnaryOperation32fSign: SynetUnaryOperation32f<SimdSynetUnaryOperation32fSign>(src, size, dst); break;
            case SimdSynetUnaryOperation32fSin: SynetUnaryOperation32f<SimdSynetUnaryOperation32fSin>(src, size, dst); break;
            case SimdSynetUnaryOperation32fSqrt: SynetUnaryOperation32f<SimdSynetUnaryOperation32fSqrt>(src, size, dst); break;
            case SimdSynetUnaryOperation32fTanh: SynetUnaryOperation32f<SimdSynetUnaryOperation32fTanh>(src, size, dst); break;
            case SimdSynetUnaryOperation32fZero: SynetUnaryOperation32f<SimdSynetUnaryOperation32fZero>(src, size, dst); break;
            default:
                Base::SynetUnaryOperation32f(src, size, type, dst);
            }
        }
    }
#endif
}
