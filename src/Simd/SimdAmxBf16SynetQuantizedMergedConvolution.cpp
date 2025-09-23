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
#include "Simd/SimdSynetQuantizedMergedConvolution.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace AmxBf16
    {
        typedef Base::SynetQuantizedMergedConvolution::AlgParam AlgParam;

        //------------------------------------------------------------------------------------------------

        SynetQuantizedMergedConvolutionCdc::SynetQuantizedMergedConvolutionCdc(const MergConvParam& p)
            : Avx512vnni::SynetQuantizedMergedConvolutionCdc(p)
        {
            if (p.conv[0].srcC >= 32 || p.conv[2].srcC >= 32)
            {
                SetSize(F, 64, 1);
                if (p.conv[0].srcC >= 32)
                {
                    SetInputPreprocess(p.conv[0], _alg, _inputPreprocess);
                    SetInputConvolution(p.conv[0], _alg, _inputConvolution);
                }
                else
                    _alg.isB = 0;
                SetOutputConvolution(p.conv[2], _alg, _outputConvolution);
            }
        }

        //------------------------------------------------------------------------------------------------

        SynetQuantizedMergedConvolutionCd::SynetQuantizedMergedConvolutionCd(const MergConvParam& p)
            : Avx512vnni::SynetQuantizedMergedConvolutionCd(p)
        {
            if (p.conv[0].srcC >= 32)
            {
                SetSize(F, 64, 1);
                SetInputPreprocess(p.conv[0], _alg, _inputPreprocess);
                SetInputConvolution(p.conv[0], _alg, _inputConvolution);
            }
        }

        //------------------------------------------------------------------------------------------------

        SynetQuantizedMergedConvolutionDc::SynetQuantizedMergedConvolutionDc(const MergConvParam& p)
            : Avx512vnni::SynetQuantizedMergedConvolutionDc(p)
        {
            if (p.conv[1].srcC >= 32)
            {
                SetSize(F, 64, 1);
                SetOutputConvolution(p.conv[1], _alg, _outputConvolution);
            }
        }

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedMergedConvolutionInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, int add)
        {
            MergConvParam param(batch, convs, count, add);
            if (!param.Valid(SimdTensorData8u, SimdTensorData8u))
                return NULL;
            else if (SynetQuantizedMergedConvolutionCdc::Preferable(param))
                return new SynetQuantizedMergedConvolutionCdc(param);
            else if (SynetQuantizedMergedConvolutionCd::Preferable(param))
                return new SynetQuantizedMergedConvolutionCd(param);
            else if (SynetQuantizedMergedConvolutionDc::Preferable(param))
                return new SynetQuantizedMergedConvolutionDc(param);
            return new Base::SynetQuantizedMergedConvolutionRef(param);
        }
    }
#endif
}
