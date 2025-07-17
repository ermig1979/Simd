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
        template <typename A, typename B, SimdConvolutionActivationType type, typename D> static void QuantizedAddUniform(const uint8_t* a8, int aBias, float aNorm, const uint8_t* b8, int bBias, float bNorm, size_t size, const float* params, float dNorm, int dZero, uint8_t* dst8)
        {
            const A* a = (const A*)a8;
            const B* b = (const B*)b8;
            D* dst = (D*)dst8;
            for (size_t i = 0; i < size; ++i)
                QuantizedAdd<A, B, type, D>(a[i], aBias, aNorm, b[i], bBias, bNorm, params, dst[i], dNorm, dZero);
        }

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

        SynetQuantizedAddUniform::SynetQuantizedAddUniform(const QuantizedAddParam& p)
            : SynetQuantizedAdd(p)
            , _size(0)
            , _uniform(0)
        {
            assert(p.aShape == p.bShape);
            _size = 1;
            for(size_t i = 0; i < p.aShape.size(); ++i)
                _size *= p.aShape[i];
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
            _uniform(a, p.aBias, p.aNorm, b, p.bBias, p.bNorm, _size, p.actParams, p.dNorm, p.dZero, dst);
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetQuantizedAddInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, int32_t aBias, const float* aNorm,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, int32_t bBias, const float* bNorm,
            SimdConvolutionActivationType actType, const float* actParams, SimdTensorDataType dstType, const float* dstNorm, int32_t dstZero)
        {
            QuantizedAddParam param(aShape, aCount, aType, aBias, aNorm, bShape, bCount, bType, bBias, bNorm, actType, actParams, dstType, dstNorm, dstZero);
            if (!param.Valid())
                return NULL;
            if (SynetQuantizedAddUniform::Preferable(param))
                return new SynetQuantizedAddUniform(param);
            return NULL;
        }
    }
#endif
}
