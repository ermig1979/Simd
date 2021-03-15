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
#include "Simd/SimdSynetInnerProduct32f.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_PERFORMANCE_STATISTIC)
    Base::PerformanceMeasurer * SynetInnerProduct32f::Perf(const String& func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

    namespace Base
    {
        SynetInnerProduct32fGemm::SynetInnerProduct32fGemm(const InnerProductParam32f & p)
            : SynetInnerProduct32f(p)
            , _0(0.0f)
            , _1(1.0f)
        {
            _gemm = _param.transpose ? Base::Gemm32fNT : Base::Gemm32fNN;
            _biasAndActivation = Base::ConvolutionBiasAndActivation;
            _M = _param.batch;
            _N = _param.output;
            _K = _param.input;
            _ldW = _param.transpose ? _K : _N;
            _ldS = _K;
            _ldD = _N;
        }

        void SynetInnerProduct32fGemm::SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params)
        {
            Simd::SynetInnerProduct32f::SetParams(weight, internal, bias, params);
        }

        void SynetInnerProduct32fGemm::Forward(const float * src, float * dst)
        {
            _gemm(_M, _N, _K, &_1, src, _ldS, _weight, _ldW, &_0, dst, _ldD);
            _biasAndActivation(_bias, 1, _N, _param.activation, _params, SimdTrue, dst);
        }

        //---------------------------------------------------------------------

        void * SynetInnerProduct32fInit(size_t batch, size_t input, size_t output, SimdBool transpose, SimdConvolutionActivationType activation)
        {
            InnerProductParam32f param(batch, input, output, transpose, activation);
            if (!param.Valid())
                return NULL;
            return new SynetInnerProduct32fGemm(param);
        }
    }
}
