/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template<bool overflow, int norm, Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2x1(const uint8_t * src0, const ConvParam8i& p, 
            const SynetConvolution8iNhwcDirect::AlgParam & a, size_t kY, size_t kH, size_t kX, size_t kW, size_t srcC, size_t dstC, const int8_t * weight, 
            const int32_t* bias, const __m128 * scale, const __m128* shift, int32_t * buf, uint8_t* dst)
        {

        }

        SynetConvolution8iNhwcDirect::SynetConvolution8iNhwcDirect(const ConvParam8i& p)
            : Base::SynetConvolution8iNhwcDirect(p)
        {
        }

        //---------------------------------------------------------------------

        void * SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility)
        {
            ConvParam8i param(batch, conv, compatibility);
            if (!param.Valid())
                return NULL;
            else if (SynetConvolution8iNhwcDirect::Preferable(param))
                return new SynetConvolution8iNhwcDirect(param);
            else
                return new Base::SynetConvolution8iGemmNN(param);
        }
    }
#endif
}
