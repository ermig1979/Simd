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
        SIMD_INLINE svint32_t QuantizedAdd(const svint32_t& a, const svfloat32_t& adScale, const svint32_t& b, const svfloat32_t& bdScale,
            const svfloat32_t& term, const svfloat32_t& one, const svint32_t& zero, const svbool_t& mask)
        {
            svfloat32_t value = svmla_f32_x(mask, term, svcvt_f32_s32_x(mask, a), adScale);
            value = svmla_f32_x(mask, value, svcvt_f32_s32_x(mask, b), bdScale);
            return QuantizeLinear(value, one, zero, mask);
        }

        SIMD_INLINE void QuantizedAdd8u8u8u(const uint8_t* a, const svfloat32_t& adScale, const uint8_t* b, const svfloat32_t& bdScale,
            const svfloat32_t& term, const svfloat32_t& one, const svint32_t& zero, uint8_t* dst, const svbool_t& mask)
        {
            svint32_t va = svreinterpret_s32_u32(svld1ub_u32(mask, a));
            svint32_t vb = svreinterpret_s32_u32(svld1ub_u32(mask, b));
            svint32_t dst32 = QuantizedAdd(va, adScale, vb, bdScale, term, one, zero, mask);
            dst32 = svmin_n_s32_x(mask, svmax_n_s32_x(mask, dst32, 0), 255);
            svst1b_u32(mask, dst, svreinterpret_u32_s32(dst32));
        }

        static void QuantizedAddUniform8u8u8u(const uint8_t* a, float aScale, int aZero, const uint8_t* b, float bScale, int bZero, size_t size, const float*, float dScale, int dZero, uint8_t* dst)
        {
            float adScale = aScale / dScale;
            float bdScale = bScale / dScale;
            float term = float(dZero) - (adScale * float(aZero) + bdScale * float(bZero));
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _adScale = svdup_n_f32(adScale), _bdScale = svdup_n_f32(bdScale), _term = svdup_n_f32(term);
            const svfloat32_t _one = svdup_n_f32(1.0f);
            const svint32_t _zero = svdup_n_s32(0);
            size_t i = 0;
            for (; i + QF <= size; i += QF)
            {
                QuantizedAdd8u8u8u(a + i + 0 * F, _adScale, b + i + 0 * F, _bdScale, _term, _one, _zero, dst + i + 0 * F, body);
                QuantizedAdd8u8u8u(a + i + 1 * F, _adScale, b + i + 1 * F, _bdScale, _term, _one, _zero, dst + i + 1 * F, body);
                QuantizedAdd8u8u8u(a + i + 2 * F, _adScale, b + i + 2 * F, _bdScale, _term, _one, _zero, dst + i + 2 * F, body);
                QuantizedAdd8u8u8u(a + i + 3 * F, _adScale, b + i + 3 * F, _bdScale, _term, _one, _zero, dst + i + 3 * F, body);
            }
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
