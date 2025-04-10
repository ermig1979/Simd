/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdSynetConvolution16b.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Avx2
    {
        void* SynetConvolution16bInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility)
        {
            ConvParam param(batch, conv, compatibility);
            if (!param.Valid(SimdTensorData32f, SimdTensorData16b))
                return NULL;
            if (SynetConvolution16bNhwcSpecV1::Preferable(param))
                return new Avx2::SynetConvolution16bNhwcSpecV1(param);
            if (SynetConvolution16bNhwcSpecV0::Preferable(param))
                return new Avx2::SynetConvolution16bNhwcSpecV0(param);
            if (SynetConvolution16bNhwcGemm::Preferable(param))
                return new Avx2::SynetConvolution16bNhwcGemm(param);
            if (Base::SynetConvolution16bNchwGemm::Preferable(param))
                return new Avx2::SynetConvolution16bNchwGemm(param);
            if (Base::SynetConvolution16bNhwcDepthwise::Preferable(param))
                return new Avx2::SynetConvolution16bNhwcDepthwise(param);
            return new Base::SynetConvolution16bGemm(param);
        }
    }
#endif
}
