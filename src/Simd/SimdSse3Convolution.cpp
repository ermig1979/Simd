/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdConvolution.h"
#include "Simd/SimdSse1.h"
#include "Simd/SimdSse3.h"

namespace Simd
{
#ifdef SIMD_SSE3_ENABLE    
    namespace Sse3
    {
        ConvolutionGemmNT::ConvolutionGemmNT(const ConvParam & p)
            : Base::ConvolutionGemmNT(p)
        {
        }

        bool ConvolutionGemmNT::Preferable(const ConvParam & p)
        {
            return p.srcH < 6 && p.srcW < 6 && p.group == 1 && p.trans == 0;
        }

        void ConvolutionGemmNT::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
                Sse3::Gemm32fNT(_M, _N, _K, &_1, _weight + _weightStep * g, _K, src + _srcStep * g, _K, &_0, dst + _dstStep * g, _N);
            Sse::ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, p.activation, _params, ::SimdFalse, dst);
        }

        //---------------------------------------------------------------------

        void * ConvolutionInit(SimdBool trans, size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm)
        {
            ConvParam param(trans, batch, conv, gemm);
            if (!param.Valid())
                return NULL;
            else if (Sse::ConvolutionDepthwiseDotProduct::Preferable(param))
                return new Sse::ConvolutionDepthwiseDotProduct(param);
            else if (ConvolutionWinograd::Preferable(param))
                return new Sse::ConvolutionWinograd(param);
            else if (ConvolutionGemmNT::Preferable(param))
                return new ConvolutionGemmNT(param);
            else if (ConvolutionDirectNchw::Preferable(param))
                return new Sse::ConvolutionDirectNchw(param);
            else if (ConvolutionDirectNhwc::Preferable(param))
                return new Sse::ConvolutionDirectNhwc(param);
            else
                return new Sse::ConvolutionGemmNN(param);
        }
    }
#endif//SIMD_SSE3_ENABLE
}
