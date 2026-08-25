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
#include "Simd/SimdSynetQuantizedAdd.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdSynetQuantizeLinear.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE void UnpackU8(const svuint8_t& src, svint32_t& d0, svint32_t& d1, svint32_t& d2, svint32_t& d3)
        {
            svuint16_t lo = svunpklo_u16(src);
            svuint16_t hi = svunpkhi_u16(src);
            d0 = svreinterpret_s32_u32(svunpklo_u32(lo));
            d1 = svreinterpret_s32_u32(svunpkhi_u32(lo));
            d2 = svreinterpret_s32_u32(svunpklo_u32(hi));
            d3 = svreinterpret_s32_u32(svunpkhi_u32(hi));
        }

        SIMD_INLINE svuint8_t PackU8(const svint32_t& d0, const svint32_t& d1, const svint32_t& d2, const svint32_t& d3)
        {
            svuint16_t lo = svuzp1_u16(svreinterpret_u16_u32(svreinterpret_u32_s32(d0)), svreinterpret_u16_u32(svreinterpret_u32_s32(d1)));
            svuint16_t hi = svuzp1_u16(svreinterpret_u16_u32(svreinterpret_u32_s32(d2)), svreinterpret_u16_u32(svreinterpret_u32_s32(d3)));
            return svuzp1_u8(svreinterpret_u8_u16(lo), svreinterpret_u8_u16(hi));
        }

        SIMD_INLINE svint32_t QuantizedAdd(const svint32_t& a, const svfloat32_t& adScale, const svint32_t& b, const svfloat32_t& bdScale,
            const svfloat32_t& term, const svfloat32_t& one, const svint32_t& zero, const svbool_t& mask)
        {
            svfloat32_t value = svmla_f32_x(mask, term, svcvt_f32_s32_x(mask, a), adScale);
            value = svmla_f32_x(mask, value, svcvt_f32_s32_x(mask, b), bdScale);
            svint32_t dst = QuantizeLinear(value, one, zero, mask);
            return svmin_n_s32_x(mask, svmax_n_s32_x(mask, dst, 0), 255);
        }

        SIMD_INLINE void QuantizedAdd8u8u8u(const uint8_t* a, const svfloat32_t& adScale, const uint8_t* b, const svfloat32_t& bdScale,
            const svfloat32_t& term, const svfloat32_t& one, const svint32_t& zero, uint8_t* dst, const svbool_t& mask)
        {
            svint32_t va = svreinterpret_s32_u32(svld1ub_u32(mask, a));
            svint32_t vb = svreinterpret_s32_u32(svld1ub_u32(mask, b));
            svst1b_u32(mask, dst, svreinterpret_u32_s32(QuantizedAdd(va, adScale, vb, bdScale, term, one, zero, mask)));
        }

        SIMD_INLINE void QuantizedAdd8u8u8uA(const uint8_t* a, const svfloat32_t& adScale, const uint8_t* b, const svfloat32_t& bdScale,
            const svfloat32_t& term, const svfloat32_t& one, const svint32_t& zero, uint8_t* dst)
        {
            const svbool_t mask32 = svptrue_b32();
            const svbool_t mask8 = svptrue_b8();
            svint32_t a0, a1, a2, a3, b0, b1, b2, b3;
            UnpackU8(svld1_u8(mask8, a), a0, a1, a2, a3);
            UnpackU8(svld1_u8(mask8, b), b0, b1, b2, b3);
            svint32_t d0 = QuantizedAdd(a0, adScale, b0, bdScale, term, one, zero, mask32);
            svint32_t d1 = QuantizedAdd(a1, adScale, b1, bdScale, term, one, zero, mask32);
            svint32_t d2 = QuantizedAdd(a2, adScale, b2, bdScale, term, one, zero, mask32);
            svint32_t d3 = QuantizedAdd(a3, adScale, b3, bdScale, term, one, zero, mask32);
            svst1_u8(mask8, dst, PackU8(d0, d1, d2, d3));
        }

        static void QuantizedAddUniform8u8u8u(const uint8_t* a, float aScale, int aZero, const uint8_t* b, float bScale, int bZero, size_t size, const float*, float dScale, int dZero, uint8_t* dst)
        {
            float adScale = aScale / dScale;
            float bdScale = bScale / dScale;
            float term = float(dZero) - (adScale * float(aZero) + bdScale * float(bZero));
            const size_t F = svcntw(), A = svcntb(), QA = 4 * A;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _adScale = svdup_n_f32(adScale), _bdScale = svdup_n_f32(bdScale), _term = svdup_n_f32(term);
            const svfloat32_t _one = svdup_n_f32(1.0f);
            const svint32_t _zero = svdup_n_s32(0);
            size_t i = 0;
            for (; i + QA <= size; i += QA)
            {
                QuantizedAdd8u8u8uA(a + i + 0 * A, _adScale, b + i + 0 * A, _bdScale, _term, _one, _zero, dst + i + 0 * A);
                QuantizedAdd8u8u8uA(a + i + 1 * A, _adScale, b + i + 1 * A, _bdScale, _term, _one, _zero, dst + i + 1 * A);
                QuantizedAdd8u8u8uA(a + i + 2 * A, _adScale, b + i + 2 * A, _bdScale, _term, _one, _zero, dst + i + 2 * A);
                QuantizedAdd8u8u8uA(a + i + 3 * A, _adScale, b + i + 3 * A, _bdScale, _term, _one, _zero, dst + i + 3 * A);
            }
            for (; i + A <= size; i += A)
                QuantizedAdd8u8u8uA(a + i, _adScale, b + i, _bdScale, _term, _one, _zero, dst + i);
            for (; i + F <= size; i += F)
                QuantizedAdd8u8u8u(a + i, _adScale, b + i, _bdScale, _term, _one, _zero, dst + i, body);
            if (i < size)
                QuantizedAdd8u8u8u(a + i, _adScale, b + i, _bdScale, _term, _one, _zero, dst + i, svwhilelt_b32(i, size));
        }

        static SynetQuantizedAddUniform::UniformPtr GetQuantizedAddUniform8u8u8u(SimdConvolutionActivationType type)
        {
            switch (type)
            {
            case SimdConvolutionActivationIdentity:
            case SimdConvolutionActivationRelu: return QuantizedAddUniform8u8u8u;
            default:
                return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetQuantizedAddUniform::SynetQuantizedAddUniform(const QuantizedAddParam& p)
            : Base::SynetQuantizedAddUniform(p)
        {
            if (p.aType == SimdTensorData8u && p.bType == SimdTensorData8u && p.dType == SimdTensorData8u)
                _uniform = GetQuantizedAddUniform8u8u8u(p.actType);
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetQuantizedAddInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const float* aScale, int32_t aZero,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, const float* bScale, int32_t bZero,
            SimdConvolutionActivationType actType, const float* actParams, SimdTensorDataType dstType, const float* dstScale, int32_t dstZero)
        {
            QuantizedAddParam param(aShape, aCount, aType, aScale, aZero, bShape, bCount, bType, bScale, bZero, actType, actParams, dstType, dstScale, dstZero);
            if (!param.Valid())
                return NULL;
            if (SynetQuantizedAddUniform::Preferable(param))
                return new SynetQuantizedAddUniform(param);
            return NULL;
        }
    }
#endif
}
