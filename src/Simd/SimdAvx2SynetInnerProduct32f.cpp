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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynetInnerProduct32f.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx2.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SynetInnerProduct32fGemm::SynetInnerProduct32fGemm(const InnerProductParam32f& p)
            : Avx::SynetInnerProduct32fGemm(p)
        {
            _biasAndActivation = Avx2::ConvolutionBiasAndActivation;
            if (_param.transpose)
            {
                if (_param.input > Sse::F)
                    _gemm = Avx2::Gemm32fNT;
            }
            else
            {
                if (_param.output > Sse::F)
                    _gemm = Avx2::Gemm32fNN;
            }
        }

        //---------------------------------------------------------------------

        void* SynetInnerProduct32fInit(size_t batch, size_t input, size_t output, SimdBool transpose, SimdConvolutionActivationType activation)
        {
            InnerProductParam32f param(batch, input, output, transpose, activation);
            if (!param.Valid())
                return NULL;
            return new SynetInnerProduct32fGemm(param);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
