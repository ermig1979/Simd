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
#include "Simd/SimdSynetDynamicQuantizedInnerProduct.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetDynamicQuantizedInnerProduct::SynetDynamicQuantizedInnerProduct(const DynamicQuantizedInnerProductParam& p)
            : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            , _perf(NULL)
#endif
        {
            _sizeA = p.M * p.K;
            _sizeB = 0;
            _sizeC = p.M * p.N;
            _sizeS = 0;
            _aN = p.N;
        }

        size_t SynetDynamicQuantizedInnerProduct::ExternalBufferSize() const
        {
            size_t size = SIMD_ALIGN;
            return size;
        }

        size_t SynetDynamicQuantizedInnerProduct::InternalBufferSize() const
        {
            return _buffer.RawSize() + _weight.RawSize() + _sums.RawSize() + _scale.RawSize() + _bias.RawSize() + _params.RawSize();
        }

        void SynetDynamicQuantizedInnerProduct::SetParams(const int8_t* weight, const float* scale, const float* bias, const float* params)
        {
            const DynamicQuantizedInnerProductParam& p = _param;

            SetWeight(weight);

            _scale.Resize(_aN, true);
            for (size_t j = 0; j < p.N; ++j)
                _scale[j] = scale[j];

            _sums.Resize(_aN, true);
            for (size_t j = 0; j < p.N; ++j)
                for (size_t k = 0; k < p.K; ++k)
                    _sums[j] -= weight[k * p.N + j];

            _bias.Resize(_aN, true);
            if (bias)
            {
                for (size_t j = 0; j < p.N; ++j)
                    _bias[j] = bias[j];
            }

            _params.Resize(_aN, true);
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity:
                _params.data[0] = -FLT_MAX;
                _params.data[1] = FLT_MAX;
                break;
            case SimdConvolutionActivationRelu:
                _params.data[0] = 0;
                _params.data[1] = FLT_MAX;
                break;
            case SimdConvolutionActivationLeakyRelu:
                for (size_t j = 0; j < p.N; ++j)
                    _params.data[j] = params[0];
                break;
            case SimdConvolutionActivationRestrictRange:
                _params.data[0] = params[0];
                _params.data[1] = params[1];
                break;
            case SimdConvolutionActivationPrelu:
                for (size_t j = 0; j < p.N; ++j)
                    _params.data[j] = params[j];
                break;
            case SimdConvolutionActivationElu:
                _params.data[0] = params[0];
                break;
            case SimdConvolutionActivationHswish:
                _params.data[0] = params[0];
                _params.data[1] = params[1];
                break;
            case SimdConvolutionActivationMish:
                _params.data[0] = params[0];
                break;
            case SimdConvolutionActivationHardSigmoid:
                _params.data[0] = params[0];
                _params.data[1] = params[1];
                break;
            case SimdConvolutionActivationSwish:
                _params.data[0] = params[0];
                break;
            case SimdConvolutionActivationGelu:
                break;
            default:
                assert(0);
            }
        }

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer* SynetDynamicQuantizedInnerProduct::Perf(const char* func)
        {
            if (_perf == NULL)
                _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
            return _perf;
        }
#endif

        //-------------------------------------------------------------------------------------------------

        void SynetDynamicQuantizedInnerProductRef_MinMax(const float* src, size_t size, float& min, float& max)
        {
            min = FLT_MAX;
            max = -FLT_MAX;
            for (size_t i = 0; i < size; ++i)
            {
                float val = src[i];
                min = Simd::Min(val, min);
                max = Simd::Max(val, max);
            }
        }

        //-------------------------------------------------------------------------------------------------

        void SynetDynamicQuantizedInnerProductRef_Quantize(const float* src, size_t size, uint8_t* dst, float& scale, uint8_t& zero)
        {
            float min, max;
            SynetDynamicQuantizedInnerProductRef_MinMax(src, size, min, max);
            min = Simd::Min(min, 0.0f);
            max = Simd::Max(max, 0.0f);
            const int qmin = std::numeric_limits<uint8_t>::min(), qmax = std::numeric_limits<uint8_t>::max();
            scale = max == min ? 1.0f : (max - min) / float(qmax - qmin);
            float initialZeroPoint = qmin - min / scale;
            zero = (uint8_t)NearByInt(Max(float(qmin), Min(float(qmax), initialZeroPoint)));
            float norm = 1.0f / scale;
            int _zero = zero;
            for (size_t i = 0; i < size; ++i)
                dst[i] = QuantizeLinear(src[i], norm, _zero, qmin, qmax);
        }

        //-------------------------------------------------------------------------------------------------

        void SynetDynamicQuantizedInnerProductRef_Gemm(size_t M, size_t N, size_t K, const uint8_t* src, const int8_t* weight, const int32_t* bias, int32_t* dst, bool overflow)
        {
            const size_t K2 = overflow ? K / 2 * 2 : 0;
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                    dst[j] = bias[j];
                size_t k = 0;
                for (; k < K2; k += 2)
                {
                    int32_t s0 = src[k + 0];
                    int32_t s1 = src[k + 1];
                    const int8_t* w0 = weight + (k + 0) * N;
                    const int8_t* w1 = weight + (k + 1) * N;
                    for (size_t j = 0; j < N; ++j)
                        dst[j] += RestrictRange(s0 * w0[j] + s1 * w1[j], SHRT_MIN, SHRT_MAX);
                }
                for (; k < K; k += 1)
                {
                    int32_t s0 = src[k];
                    const int8_t* w0 = weight + k * N;
                    for (size_t j = 0; j < N; ++j)
                        dst[j] += s0 * w0[j];
                }
                src += K;
                dst += N;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void SynetDynamicQuantizedInnerProductRef_Norm(const int32_t* src, size_t M, size_t N, const float *norm, float* dst)
        {
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                    dst[j] = (float)src[j] * norm[j];
                src += N;
                dst += N;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetDynamicQuantizedInnerProductRef::SynetDynamicQuantizedInnerProductRef(const DynamicQuantizedInnerProductParam& p)
            : SynetDynamicQuantizedInnerProduct(p)
        {
        }

        String SynetDynamicQuantizedInnerProductRef::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::Ref";
            return desc.str();
        }

        size_t SynetDynamicQuantizedInnerProductRef::ExternalBufferSize() const
        {
            size_t size = SynetDynamicQuantizedInnerProduct::ExternalBufferSize();
            size += _sizeA;
            size += _aN * sizeof(int32_t);
            size += _aN * sizeof(float);
            return size;
        }

        void SynetDynamicQuantizedInnerProductRef::Forward(const float* A, uint8_t* buf, float* C)
        {
            const DynamicQuantizedInnerProductParam& p = _param;
            buf = Buffer(buf);
            uint8_t* bufA = Allocate<uint8_t>(buf, _sizeA);
            int32_t* bias = Allocate<int32_t>(buf, _aN);
            float* norm = Allocate<float>(buf, _aN);
            float scale;
            uint8_t zero;
            SynetDynamicQuantizedInnerProductRef_Quantize(A, _sizeA, bufA, scale, zero);
            for (size_t j = 0; j < p.N; ++j)
            {
                bias[j] = _sums[j] * zero;
                norm[j] = _scale[j] * scale;
            }
#if defined(__MINGW32__) || defined(__MINGW64__)
            bool overflow = true;
#else
            bool overflow = SimdCpuInfo(SimdCpuInfoAvx512vnni) == 0;
#endif
            SynetDynamicQuantizedInnerProductRef_Gemm(p.M, p.N, p.K, bufA, _weight.data, bias, (int32_t*)C, overflow);

            SynetDynamicQuantizedInnerProductRef_Norm((int32_t*)C, p.M, p.N, norm, C);

            ConvolutionBiasAndActivation(_bias.data, p.N, p.M, p.activation, _params.data, SimdTrue, C);
        }

        void SynetDynamicQuantizedInnerProductRef::SetWeight(const int8_t* weight)
        {
            const DynamicQuantizedInnerProductParam& p = _param;
            _weight.Resize(p.N * p.K);
            _weight.Assign(weight, _weight.size);
        }
        
        //-------------------------------------------------------------------------------------------------

        void* SynetDynamicQuantizedInnerProductInit(size_t M, size_t N, size_t K, SimdBool bias, SimdConvolutionActivationType activation)
        {
            DynamicQuantizedInnerProductParam param(M, N, K, bias, activation);
            if (!param.Valid())
                return NULL;
            return new SynetDynamicQuantizedInnerProductRef(param);
        }
    }
#endif
}
