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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynetInnerProduct32f.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdNeon.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)      
    namespace Sve2
    {
        SynetInnerProduct32fGemm::SynetInnerProduct32fGemm(const InnerProductParam32f& p)
            : Base::SynetInnerProduct32fGemm(p)
        {
            _biasAndActivation = Neon::ConvolutionBiasAndActivation;
            if (_param.transB)
            {
                _gemm = Sve2::Gemm32fNT;
                if (_M == 1 && _param.activation == SimdConvolutionActivationIdentity && _param.constB)
                    _prod = Sve2::SynetInnerProductLayerForward;
                else
                    _prod = NULL;
            }
            else
            {
                _gemm = Sve2::Gemm32fNN;
            }
            if (_param.N > Neon::F && _prod == NULL && _param.constB)
            {
                _cbRun = Neon::Gemm32fNNcbRun;
                _cbPack = Neon::Gemm32fNNcbReorderB;
                _cbWeight.Resize(Neon::Gemm32fNNcbBufferSize(_M, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE));
            }
        }

        //-----------------------------------------------------------------------------------------

        void* SynetInnerProduct32fInit(size_t M, size_t N, size_t K, SimdBool transB, SimdBool constB, SimdBool bias, SimdConvolutionActivationType activation)
        {
            InnerProductParam32f param(M, N, K, transB, constB, bias, activation);
            if (!param.Valid())
                return NULL;
            return new SynetInnerProduct32fGemm(param);
        }
    }
#endif
}
