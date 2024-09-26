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
#include "Simd/SimdSynetMergedConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE) 
	namespace Avx512bw
    {
        void* SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add)
        {
            MergConvParam param(batch, convs, count, add, SimdSynetCompatibilityDefault);
            if (!param.Valid(SimdTensorData32f))
                return NULL;
            if (SynetMergedConvolution32fCdc::Preferable(param))
            {
                if (param.conv[1].dstC <= HF && param.conv[2].dstC <= HF)
                    return new Avx2::SynetMergedConvolution32fCdc(param);
                else
                    return new Avx512bw::SynetMergedConvolution32fCdc(param);
            }
            else if (SynetMergedConvolution32fCd::Preferable(param))
            {
                if (param.conv[1].dstC <= HF)
                    return new Avx2::SynetMergedConvolution32fCd(param);
                else
                    return new Avx512bw::SynetMergedConvolution32fCd(param);
            }
            else if (SynetMergedConvolution32fDc::Preferable(param))
            {
                if (param.conv[0].dstC <= HF || param.conv[1].dstC <= HF)
                    return new Avx2::SynetMergedConvolution32fDc(param);
                else
                    return new Avx512bw::SynetMergedConvolution32fDc(param);
            }
            else
                return new Base::SynetMergedConvolution32f(param);
        }
	}
#endif
}
