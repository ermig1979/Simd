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
#include "Simd/SimdSynetQuantizedAdd.h"
#include "Simd/SimdSynetQuantizedAddCommon.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetActivation.h"
#include "Simd/SimdFmadd.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)

    SynetQuantizedAdd::SynetQuantizedAdd(const QuantizedAddParam& p)
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

        template <typename A, typename B, SimdConvolutionActivationType type, typename D> SIMD_INLINE void QuantizedAdd(const A& a, int aBias, float aNorm, const B& b, int bBias, float bNorm, const float* params, D& dst, float dNorm, int dZero)
        {
            float _a = Convert8u<A, float>(a, aNorm, aBias);
            float _b = Convert8u<B, float>(b, bNorm, bBias);
            dst = Convert8u<float, D>(Activate<type>(_a + _b, params, 0), dNorm, dZero);
        }

        template <typename A, typename B, SimdConvolutionActivationType type, typename D> static void QuantizedAddUniform(const uint8_t* a8, float aScale, int aZero, const uint8_t* b8, float bScale, int bZero, size_t size, const float* params, float dScale, int dZero, uint8_t* dst8)
        {
            const A* a = (const A*)a8;
            const B* b = (const B*)b8;
            D* dst = (D*)dst8;
            int aBias = -aZero, bBias = -bZero;
            float dNorm = 1.0f / (dScale);
            for (size_t i = 0; i < size; ++i)
                QuantizedAdd<A, B, type, D>(a[i], aBias, aScale, b[i], bBias, bScale, params, dst[i], dNorm, dZero);
        }

        //-------------------------------------------------------------------------------------------------

        template<class A, class B, SimdConvolutionActivationType type> static SynetQuantizedAddUniform::UniformPtr GetQuantizedAddUniform(SimdTensorDataType dType)
        {
            switch (dType)
            {
            case SimdTensorData32f: return QuantizedAddUniform<A, B, type, float>;
            case SimdTensorData8u: return QuantizedAddUniform<A, B, type, uint8_t>;
            default:
                return NULL;
            }
        }

        template<class A, class B> static SynetQuantizedAddUniform::UniformPtr GetQuantizedAddUniform(SimdConvolutionActivationType type, SimdTensorDataType dType)
        {
            switch (type)
            {
            case SimdConvolutionActivationIdentity: return GetQuantizedAddUniform<A, B, SimdConvolutionActivationIdentity>(dType);
            case SimdConvolutionActivationRelu: return GetQuantizedAddUniform<A, B, SimdConvolutionActivationRelu>(dType);
            default:
                return NULL;
            }
        }

        template<class A> static SynetQuantizedAddUniform::UniformPtr GetQuantizedAddUniform(SimdTensorDataType bType, SimdConvolutionActivationType type, SimdTensorDataType dType)
        {
            switch (bType)
            {
            case SimdTensorData32f: return GetQuantizedAddUniform<A, float>(type, dType);
            case SimdTensorData8u: return GetQuantizedAddUniform<A, uint8_t>(type, dType);
            default:
                return NULL;
            }
        }

        static SynetQuantizedAddUniform::UniformPtr GetQuantizedAddUniform(SimdTensorDataType aType, SimdTensorDataType bType, SimdConvolutionActivationType type, SimdTensorDataType dType)
        {
            switch (aType)
            {
            case SimdTensorData32f: return GetQuantizedAddUniform<float>(bType, type, dType);
            case SimdTensorData8u: return GetQuantizedAddUniform<uint8_t>(bType, type, dType);
            default:
                return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        static void QuantizedAddUniform(const uint8_t* a, float aScale, int aZero, const uint8_t* b, float bScale, int bZero, size_t size, const float*, float dScale, int dZero, uint8_t* dst)
        {
            float adScale = aScale / dScale;
            float bdScale = bScale / dScale;
            float term = float(dZero) - (adScale * float(aZero) + bdScale * float(bZero));
            for (size_t i = 0; i < size; ++i)
                QuantizedAdd(a[i], adScale, b[i], bdScale, term, dst[i]);
        }

        //-------------------------------------------------------------------------------------------------

        SynetQuantizedAddUniform::SynetQuantizedAddUniform(const QuantizedAddParam& p)
            : SynetQuantizedAdd(p)
            , _size(0)
            , _uniform(0)
        {
            assert(p.aShape == p.bShape);
            _size = 1;
            for(size_t i = 0; i < p.aShape.size(); ++i)
                _size *= p.aShape[i];
            if (p.aType == SimdTensorData8u && p.bType == SimdTensorData8u && p.dType == SimdTensorData8u)
                _uniform = QuantizedAddUniform;
            else
                _uniform = GetQuantizedAddUniform(p.aType, p.bType, p.actType, p.dType);
        }

        bool SynetQuantizedAddUniform::Preferable(const QuantizedAddParam& p)
        {
            if (p.aShape == p.bShape)
                return true;
            return false;
        }

        void SynetQuantizedAddUniform::Forward(const uint8_t* a, const uint8_t* b, uint8_t* dst)
        {
            const QuantizedAddParam& p = _param;
            _uniform(a, p.aScale, (int)p.aZero, b, p.bScale, (int)p.bZero, _size, p.actParams, p.dScale, (int)p.dZero, dst);
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
