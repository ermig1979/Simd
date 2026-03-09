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
#include "Simd/SimdSynetInnerProduct32f.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
    Base::PerformanceMeasurer * SynetInnerProduct32f::Perf(const char* func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

    namespace Base
    {
        void SynetInnerProductLayerForward(const float* src, const float* weight, const float* bias, size_t count, size_t size, float* dst)
        {
            size_t aligned = Simd::AlignLo(size, 4);
            for (size_t i = 0; i < count; ++i)
            {
                size_t j = 0;
                float sums[4] = { 0, 0, 0, 0 };
                for (; j < aligned; j += 4)
                {
                    sums[0] += src[j + 0] * weight[j + 0];
                    sums[1] += src[j + 1] * weight[j + 1];
                    sums[2] += src[j + 2] * weight[j + 2];
                    sums[3] += src[j + 3] * weight[j + 3];
                }
                for (; j < size; ++j)
                    sums[0] += src[j] * weight[j];
                dst[i] = sums[0] + sums[1] + sums[2] + sums[3] + (bias ? bias[i] : 0);
                weight += size;
            }
        }

        //--------------------------------------------------------------------------------------------------

        SynetInnerProduct32fGemm::SynetInnerProduct32fGemm(const InnerProductParam32f & p)
            : SynetInnerProduct32f(p)
            , _0(0.0f)
            , _1(1.0f)
        {
            _M = _param.M;
            _N = _param.N;
            _K = _param.K;
            _ldS = _K;
            _ldD = _N;
            _biasAndActivation = Base::ConvolutionBiasAndActivation;
            _prod = NULL;
            if (_param.transB)
            {
                _gemm = Base::Gemm32fNT;
                _ldW = _K;
                if (_M == 1 && _param.activation == SimdConvolutionActivationIdentity)
                    _prod = Base::SynetInnerProductLayerForward;
            }
            else
            {
                _gemm = Base::Gemm32fNN;
                _ldW = _N;
            }
        }

        String SynetInnerProduct32fGemm::Desc() const 
        { 
            return Ext() + "::Gemm" + (_prod ? "Prod" : 
                String("N") + (_cbWeight.size ? "Ncb" : (_param.transB == SimdTrue ? "T" : "N")));
        }

        void SynetInnerProduct32fGemm::SetParams(const float* weight, SimdBool* internal, const float* bias, const float* params)
        {
            Simd::SynetInnerProduct32f::SetParams(weight, internal, bias, params);
            if (_cbWeight.data && _param.constB)
            {
                Array32f buffer;
                if (_param.transB)
                {
                    buffer.Resize(_N * _K);
                    for (size_t k = 0; k < _K; ++k)
                        for (size_t j = 0; j < _N; ++j)
                            buffer[k*_N + j] = weight[j * _K + k];
                    weight = buffer.data;
                }
                _cbPack(_M, _N, _K, weight, _cbWeight.data, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                if (internal)
                    *internal = SimdTrue;
            }
        }

        void SynetInnerProduct32fGemm::Forward(const float* A, const float* B, float* buf, float* C)
        {
            if (_prod)
                _prod(A, _weight, _bias, _N, _K, C);
            else
            {
                if (_cbWeight.data)
                    _cbRun(_M, _N, _K, A, _cbWeight.data, C, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                else
                    _gemm(_M, _N, _K, &_1, A, _ldS, _param.constB ? _weight : B, _ldW, &_0, C, _ldD);
                _biasAndActivation(_bias, _N, _M, _param.activation, _params, SimdTrue, C);
            }
        }

        //--------------------------------------------------------------------------------------------------

        SynetInnerProduct32fProd::SynetInnerProduct32fProd(const InnerProductParam32f& p)
            : SynetInnerProduct32f(p)
        {
            _N = _param.N;
            _K = _param.K;
        }

        void SynetInnerProduct32fProd::SetParams(const float* weight, SimdBool* internal, const float* bias, const float* params)
        {
            SynetInnerProduct32f::SetParams(weight, internal, bias, params);
            ReorderWeight(_weight, _rWeight.data);
            if (internal)
                *internal = SimdTrue;
            if (bias)
                memcpy(_rBias.data, bias, _param.N * sizeof(float));
        }

        void SynetInnerProduct32fProd::Forward(const float* A, const float* B, float* buf, float* C)
        {
            _prod(A, _rWeight.data, _rBias.data, _K, _N, C);
        }

        bool SynetInnerProduct32fProd::Preferable(const InnerProductParam32f& p)
        {
            return
                p.activation == SimdConvolutionActivationIdentity && p.constB &&
                p.M == 1 && p.N >= 4 &&
                Base::AlgCacheL3() > p.K * p.N * sizeof(float);
        }

        void SynetInnerProduct32fProd::SetSize(size_t F)
        {
            _F = F;
            _rWeight.Resize(AlignHi(_N, _F) * _K);
            _rBias.Resize(AlignHi(_N, _F), true);
        }

        void SynetInnerProduct32fProd::ReorderWeight(const float* src, float* dst)
        {
            if (_param.transB)
            {
                for (size_t n = 0; n < _N; n += _F)
                {
                    size_t F = Simd::Min(_N, n + _F) - n;
                    const float* psrc = src + n * _K;
                    for (size_t k = 0; k < _K; ++k)
                    {
                        size_t f = 0;
                        for (; f < F; ++f)
                            *(dst++) = psrc[f * _K];
                        for (; f < _F; ++f)
                            *(dst++) = 0.0f;
                        psrc++;
                    }
                }            
            }
            else
            {
                for (size_t n = 0; n < _N; n += _F)
                {
                    size_t F = Simd::Min(_N, n + _F) - n;
                    const float* psrc = src + n;
                    for (size_t k = 0; k < _K; ++k)
                    {
                        size_t f = 0;
                        for (; f < F; ++f)
                            *(dst++) = psrc[f];
                        for (; f < _F; ++f)
                            *(dst++) = 0.0f;
                        psrc += _N;
                    }
                }
            }
        }

        //--------------------------------------------------------------------------------------------------

        void * SynetInnerProduct32fInit(size_t M, size_t N, size_t K, SimdBool transB, SimdBool constB, SimdBool bias, SimdConvolutionActivationType activation)
        {
            InnerProductParam32f param(M, N, K, transB, constB, bias, activation);
            if (!param.Valid())
                return NULL;
            return new SynetInnerProduct32fGemm(param);
        }
    }
#endif
}
