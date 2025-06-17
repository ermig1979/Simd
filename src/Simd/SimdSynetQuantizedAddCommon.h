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
#ifndef __SimdSynetQuantizedAddCommon_h__
#define __SimdSynetQuantizedAddCommon_h__

#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetActivation.h"

namespace Simd
{
    namespace Base
    {
        template <class S, class D> SIMD_INLINE D Convert8u(const S& src, float norm, int bias)
        {
            return (D)src;
        }

        template <> SIMD_INLINE float Convert8u(const uint8_t& src, float norm, int bias)
        {
            return QuantizeLinear(src, norm, bias, 0, 255);
        }

        template <> SIMD_INLINE uint8_t Convert8u(const float& src, float norm, int bias)
        {
            return DequantizeLinear(src, bias, norm);
        }

        //-------------------------------------------------------------------------------------------------

        template <typename A, typename B, SimdConvolutionActivationType type, typename D> void QuantizedAdd(const A& a, int aBias, float aNorm, const B& b, int bBias, float bNorm, const float *params, D& dst, float dNorm, int dZero)
        {
            float _a = Convert8u<A, float>(a, aNorm, aBias);
            float _b = Convert8u<B, float>(b, bNorm, bBias);
            dst = Convert8u<float, D>(Activate<type>(_a + _b, params, 0), dNorm, dZero);
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
    }
#endif
}

#endif
