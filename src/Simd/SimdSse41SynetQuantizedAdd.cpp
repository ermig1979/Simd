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
#include "Simd/SimdFmadd.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        static void QuantizedAddUniform8u8u8u(const uint8_t* a, float aScale, int aZero, const uint8_t* b, float bScale, int bZero, size_t size, const float*, float dScale, int dZero, uint8_t* dst)
        {
            float adScale = aScale / dScale;
            float bdScale = bScale / dScale;
            float term = float(dZero) - (adScale * float(aZero) + bdScale * float(bZero));
            __m128 _adScale = _mm_set1_ps(adScale), _bdScale = _mm_set1_ps(bdScale), _term = _mm_set1_ps(term);
            size_t i = 0, size4 = AlignLo(size, 4), size16 = AlignLo(size, 16);
            for (; i < size16; i += 16)
                QuantizedAdd8u8u8u16(a + i, _adScale, b + i, _bdScale, _term, dst + i);
            for (; i < size4; i += 4)
                QuantizedAdd8u8u8u4(a + i, _adScale, b + i, _bdScale, _term, dst + i);
            for (; i < size; i += 1)
                QuantizedAdd8u8u8u1(a + i, _adScale, b + i, _bdScale, _term, dst + i);
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
            if(p.aType == SimdTensorData8u && p.bType == SimdTensorData8u && p.dType == SimdTensorData8u)
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
