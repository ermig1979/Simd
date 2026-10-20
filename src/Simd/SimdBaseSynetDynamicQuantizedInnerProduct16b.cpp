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
#include "Simd/SimdSynetDynamicQuantizedInnerProduct16b.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetDynamicQuantizedInnerProduct16b::SynetDynamicQuantizedInnerProduct16b(const DynamicQuantizedInnerProduct16bParam& p)
            : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            , _perf(NULL)
#endif
        {
            _sizeA = p.M * p.K;
            _sizeS = 0;
            _sizeB = 0;
            _aM = p.M;
            _aN = p.N;
        }

        size_t SynetDynamicQuantizedInnerProduct16b::ExternalBufferSize() const
        {
            size_t size = SIMD_ALIGN;
            return size;
        }

        size_t SynetDynamicQuantizedInnerProduct16b::InternalBufferSize() const
        {
            return _buffer.RawSize() + _weight.RawSize() + _aZero.RawSize() + _bSums.RawSize() + 
                _aScale.RawSize() + _bScale.RawSize() + _bias.RawSize() + _params.RawSize();
        }

        void SynetDynamicQuantizedInnerProduct16b::SetParams(const uint8_t* weight, const float* scale, const float* bias, const float* params)
        {
            const DynamicQuantizedInnerProduct16bParam& p = _param;

            _aScale.Resize(_aM, true);
            _bScale.Resize(_aN, true);
            _bSums.Resize(_aN, true);
            if (p.constB)
            {
                if (p.typeB == SimdTensorData8i)
                {
                    const int8_t* B = (int8_t*)weight;
                    SetWeight((int8_t*)B);
                    for (size_t j = 0; j < p.N; ++j)
                        _bScale[j] = scale[j];
                    for (size_t j = 0; j < p.N; ++j)
                        for (size_t k = 0; k < p.K; ++k)
                            _bSums[j] -= B[k * p.N + j];
                }
                else
                    QuantizeB((float*)weight);
            }

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
        Base::PerformanceMeasurer* SynetDynamicQuantizedInnerProduct16b::Perf(const char* func)
        {
            if (_perf == NULL)
                _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
            return _perf;
        }
#endif

        //-------------------------------------------------------------------------------------------------

        void SynetDynamicQuantizedInnerProduct32fRef_QuantizeA(const float* src, size_t size, uint8_t* dst, float& scale, int32_t& zero)
        {
            float min = FLT_MAX;
            float max = -FLT_MAX;
            for (size_t i = 0; i < size; ++i)
            {
                float val = src[i];
                min = Simd::Min(val, min);
                max = Simd::Max(val, max);
            }
            min = Simd::Min(min, 0.0f);
            max = Simd::Max(max, 0.0f);
            const int qmin = std::numeric_limits<uint8_t>::min(), qmax = std::numeric_limits<uint8_t>::max();
            scale = max == min ? 1.0f : (max - min) / float(qmax - qmin);
            float initialZeroPoint = qmin - min / scale;
            zero = NearByInt(Max(float(qmin), Min(float(qmax), initialZeroPoint)));
            float norm = 1.0f / scale;
            for (size_t i = 0; i < size; ++i)
                dst[i] = QuantizeLinear(src[i], norm, zero, qmin, qmax);
        }

        void SynetDynamicQuantizedInnerProduct32fRef_QuantizeA(const uint16_t* src, size_t size, uint8_t* dst, float& scale, int32_t& zero)
        {
            float min = FLT_MAX;
            float max = -FLT_MAX;
            for (size_t i = 0; i < size; ++i)
            {
                float val = BFloat16ToFloat32(src[i]);
                min = Simd::Min(val, min);
                max = Simd::Max(val, max);
            }
            min = Simd::Min(min, 0.0f);
            max = Simd::Max(max, 0.0f);
            const int qmin = std::numeric_limits<uint8_t>::min(), qmax = std::numeric_limits<uint8_t>::max();
            scale = max == min ? 1.0f : (max - min) / float(qmax - qmin);
            float initialZeroPoint = qmin - min / scale;
            zero = NearByInt(Max(float(qmin), Min(float(qmax), initialZeroPoint)));
            float norm = 1.0f / scale;
            for (size_t i = 0; i < size; ++i)
                dst[i] = QuantizeLinear(BFloat16ToFloat32(src[i]), norm, zero, 0, 255);
        }

        //-------------------------------------------------------------------------------------------------

        void SynetDynamicQuantizedInnerProduct16bRef_Gemm(size_t N, size_t K, const uint8_t* src, const int8_t* weight, const int32_t* bias, int32_t* dst, bool overflow)
        {
            const size_t K2 = overflow ? K / 2 * 2 : 0;
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

        //-------------------------------------------------------------------------------------------------

        void SynetDynamicQuantizedInnerProduct32fRef_Norm(const int32_t* src, size_t M, size_t N, const float *norm, float* dst)
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

        SynetDynamicQuantizedInnerProduct16bRef::SynetDynamicQuantizedInnerProduct16bRef(const DynamicQuantizedInnerProduct16bParam& p)
            : SynetDynamicQuantizedInnerProduct16b(p)
        {
            _sizeA = p.K;
        }

        String SynetDynamicQuantizedInnerProduct16bRef::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::Ref";
            return desc.str();
        }

        size_t SynetDynamicQuantizedInnerProduct16bRef::ExternalBufferSize() const
        {
            size_t size = SynetDynamicQuantizedInnerProduct16b::ExternalBufferSize();
            size += _sizeA * sizeof(uint8_t);
            size += _sizeB * sizeof(int8_t);
            size += _aN * sizeof(int32_t);
            size += _aN * sizeof(float);
            return size;
        }

        void SynetDynamicQuantizedInnerProduct16bRef::Forward(const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C)
        {
#if defined(__MINGW32__) || defined(__MINGW64__)
            bool overflow = true;
#else
            bool overflow = SimdCpuInfo(SimdCpuInfoAvx512vnni) == 0;
#endif            
            const DynamicQuantizedInnerProduct16bParam& p = _param;
            buf = Buffer(buf);
            uint8_t* bufA = Allocate<uint8_t>(buf, _sizeA);
            int32_t* corr = Allocate<int32_t>(buf, _aN);
            float* norm = Allocate<float>(buf, _aN);
            for (size_t i = 0; i < p.M; ++i)
            {
                //AnalyseInput(A, p.M * p.K, corr, norm);
                //float aNorm = 1.0f / _aScale;
                //SynetQuantizeLinear(A, _sizeA, &aNorm, _aZero, bufA);

                //SynetDynamicQuantizedInnerProduct32fRef_Gemm(p.M, p.N, p.K, bufA, _weight.data, corr, (int32_t*)C, overflow);

                //SynetDynamicQuantizedInnerProduct32fRef_Norm((int32_t*)C, p.M, p.N, norm, C);

                //ConvolutionBiasAndActivation(_bias.data, p.N, p.M, p.activation, _params.data, SimdTrue, C);
            }
        }

        void SynetDynamicQuantizedInnerProduct16bRef::SetWeight(const int8_t* weight)
        {
            const DynamicQuantizedInnerProduct16bParam& p = _param;
            _weight.Resize(p.N * p.K);
            _weight.Assign(weight, _weight.size);
        }

        void SynetDynamicQuantizedInnerProduct16bRef::QuantizeB(const float* weight)
        {
            const DynamicQuantizedInnerProduct16bParam& p = _param;
            _weight.Resize(p.N * p.K);
            int lo = p.overflow ? -64 : -128, hi = p.overflow ? 63 : 127;
            for (size_t j = 0; j < p.N; ++j)
            {
                float max = 0;
                for (size_t k = 0; k < p.K; ++k)
                {
                    size_t offset = k * p.N + j;
                    max = Simd::Max(max, Simd::Abs(weight[offset]));
                }
                float range = std::max(0.000001f, max);
                float scale = range / (p.overflow ? 63.0f : 127.0f), invScale = (p.overflow ? 63.0f : 127.0f) / range;
                _bScale[j] = scale;
                for (size_t k = 0; k < p.K; ++k)
                {
                    size_t offset = k * p.N + j;
                    _weight[offset] = Simd::RestrictRange((int)NearByInt(weight[offset] * invScale), lo, hi);
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetDynamicQuantizedInnerProduct16bInit(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB,
            SimdTensorDataType typeC, SimdBool constB, SimdBool bias, SimdConvolutionActivationType activation, SimdBool avoidOverflow)
        {
            DynamicQuantizedInnerProduct16bParam param(M, N, K, typeA, typeB, typeC, constB, bias, activation, avoidOverflow);
            if (!param.Valid())
                return NULL;
            return new SynetDynamicQuantizedInnerProduct16bRef(param);
        }
    }
#endif
}
