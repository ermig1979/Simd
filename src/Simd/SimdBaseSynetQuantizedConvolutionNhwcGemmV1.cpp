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
#include "Simd/SimdSynetQuantizedConvolution.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetQuantizedConvolutionNhwcGemmV1::SynetQuantizedConvolutionNhwcGemmV1(const ConvParam& p)
            : SynetQuantizedConvolution(p)
        {
            _convAny = 0;
            _conv1x1 = 0;
            _gemm[0] = 0;
            _gemm[1] = 0;
        }

        String SynetQuantizedConvolutionNhwcGemmV1::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NhwcGemmV1";
            if (_alg.batch > 1)
                desc << "-" << _alg.batch;
            if (_alg.reorderType)
                desc << "-r";
            return desc.str();
        }

        void SynetQuantizedConvolutionNhwcGemmV1::SetAlgParam(size_t F, size_t microD, size_t microM, size_t microK, size_t L1, size_t L2, size_t L3)
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;

        }

        size_t SynetQuantizedConvolutionNhwcGemmV1::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = 0;
            return size;
        }

        void SynetQuantizedConvolutionNhwcGemmV1::SetWeight(const int8_t* weight)
        {
        }

        void SynetQuantizedConvolutionNhwcGemmV1::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
        }

        void SynetQuantizedConvolutionNhwcGemmV1::Forward(const uint8_t* src, uint8_t* buf, int32_t* sum, uint8_t* dst)
        {
        }

        bool SynetQuantizedConvolutionNhwcGemmV1::Preferable(const ConvParam& p)
        {
            return p.trans != 0 && p.group == 1;
        }
    }
#endif
}
