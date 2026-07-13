/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdSynetQuantizedMul.h"
#include "Simd/SimdSynetQuantizedMulCommon.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetActivation.h"
#include "Simd/SimdFmadd.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)

    SynetQuantizedMul::SynetQuantizedMul(const QuantizedMulParam& p)
        : _param(p)
    {

    }

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        template <class S, class D> SIMD_INLINE D Convert8u(const S& src, float norm, int bias)
        {
            return (D)src;
        }

        template <> SIMD_INLINE float Convert8u(const uint8_t& src, float norm, int bias)
        {
            return DequantizeLinear(src, bias, norm);
        }

        template <> SIMD_INLINE uint8_t Convert8u(const float& src, float norm, int bias)
        {
            return QuantizeLinear(src, norm, bias, 0, 255);
        }

        //-------------------------------------------------------------------------------------------------

        template <typename A, typename B, typename D> SIMD_INLINE void QuantizedMul(const A& a, int aBias, float aNorm, const B& b, int bBias, float bNorm, D& dst, float dNorm, int dZero)
        {
            float _a = Convert8u<A, float>(a, aNorm, aBias);
            float _b = Convert8u<B, float>(b, bNorm, bBias);
            dst = Convert8u<float, D>(_a * _b, dNorm, dZero);
        }

        template <typename A, typename B, typename D> static void QuantizedMulUniform(const uint8_t* a8, float aScale, int aZero, const uint8_t* b8, float bScale, int bZero, size_t size, float dScale, int dZero, uint8_t* dst8)
        {
            const A* a = (const A*)a8;
            const B* b = (const B*)b8;
            D* dst = (D*)dst8;
            int aBias = -aZero, bBias = -bZero;
            float dNorm = 1.0f / (dScale);
            for (size_t i = 0; i < size; ++i)
                QuantizedMul<A, B, D>(a[i], aBias, aScale, b[i], bBias, bScale, dst[i], dNorm, dZero);
        }

        //-------------------------------------------------------------------------------------------------

        template<class A, class B> static SynetQuantizedMulUniform::UniformPtr GetQuantizedMulUniform(SimdTensorDataType dType)
        {
            switch (dType)
            {
            case SimdTensorData32f: return QuantizedMulUniform<A, B, float>;
            case SimdTensorData8u: return QuantizedMulUniform<A, B, uint8_t>;
            default:
                return NULL;
            }
        }

        template<class A> static SynetQuantizedMulUniform::UniformPtr GetQuantizedMulUniform(SimdTensorDataType bType, SimdTensorDataType dType)
        {
            switch (bType)
            {
            case SimdTensorData32f: return GetQuantizedMulUniform<A, float>(dType);
            case SimdTensorData8u: return GetQuantizedMulUniform<A, uint8_t>(dType);
            default:
                return NULL;
            }
        }

        static SynetQuantizedMulUniform::UniformPtr GetQuantizedMulUniform(SimdTensorDataType aType, SimdTensorDataType bType, SimdTensorDataType dType)
        {
            switch (aType)
            {
            case SimdTensorData32f: return GetQuantizedMulUniform<float>(bType, dType);
            case SimdTensorData8u: return GetQuantizedMulUniform<uint8_t>(bType, dType);
            default:
                return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetQuantizedMulUniform::SynetQuantizedMulUniform(const QuantizedMulParam& p)
            : SynetQuantizedMul(p)
            , _size(0)
            , _uniform(0)
        {
            assert(p.aShape == p.bShape);
            _size = 1;
            for(size_t i = 0; i < p.aShape.size(); ++i)
                _size *= p.aShape[i];
            _uniform = GetQuantizedMulUniform(p.aType, p.bType, p.dType);
        }

        bool SynetQuantizedMulUniform::Preferable(const QuantizedMulParam& p)
        {
            if (p.aShape == p.bShape)
                return true;
            return false;
        }

        void SynetQuantizedMulUniform::Forward(const uint8_t* a, const uint8_t* b, uint8_t* dst)
        {
            const QuantizedMulParam& p = _param;
            _uniform(a, p.aScale, (int)p.aZero, b, p.bScale, (int)p.bZero, _size, p.dScale, (int)p.dZero, dst);
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetQuantizedMulInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const float* aScale, int32_t aZero,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, const float* bScale, int32_t bZero, SimdTensorDataType dstType, const float* dstScale, int32_t dstZero)
        {
            QuantizedMulParam param(aShape, aCount, aType, aScale, aZero, bShape, bCount, bType, bScale, bZero, dstType, dstScale, dstZero);
            if (!param.Valid())
                return NULL;
            if (SynetQuantizedMulUniform::Preferable(param))
                return new SynetQuantizedMulUniform(param);
            return NULL;
        }
    }
#endif
}
