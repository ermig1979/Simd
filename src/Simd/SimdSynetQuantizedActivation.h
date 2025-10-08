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
#ifndef __SimdSynetQuantizedActivation_h__
#define __SimdSynetQuantizedActivation_h__

#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetActivation.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void QuantizedPrelu(const uint8_t& src, int sBias, float sNorm, float slope, uint8_t& dst, float dNorm, int dZero)
        {
            float _src = DequantizeLinear(src, sBias, sNorm);
            float _dst = Simd::Max(0.0f, _src) + slope * Simd::Min(_src, 0.0f);
            dst = (uint8_t)QuantizeLinear(_dst, dNorm, dZero, 0, 255);
        }

        //--------------------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> SIMD_INLINE int QuantizeActivateSum(int sum, int sBias, float sNorm,
            int iZero, float iScale, const float* params, size_t offset, float dNorm, int dZero, int min, int max)
        {
            int iInt = QuantizeSumLinear(sum, sBias, sNorm, iZero, min, max);
            float fInt = float(iInt - iZero) * iScale;
            float fDst = Activate<type>(fInt, params, offset);
            return QuantizeLinear(fDst, dNorm, dZero, min, max);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE int QuantizeActivateSum(int sum, int sBias, float sNorm,
            float iScale, const float* params, size_t offset, float dNorm, int dZero, int min, int max)
        {
            int iInt = NearByInt(float(sum + sBias) * sNorm);
            float fInt = float(iInt) * iScale;
            float fDst = Activate<type>(fInt, params, offset);
            return QuantizeLinear(fDst, dNorm, dZero, min, max);
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
