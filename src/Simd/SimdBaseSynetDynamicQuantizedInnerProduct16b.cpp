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
            _sizeA = 0;
            _sizeB = 0;
            _sizeC = 0;
            _sizeT = 0;
            _aN = 0;
        }

        size_t SynetDynamicQuantizedInnerProduct16b::ExternalBufferSize() const
        {
            return 0;
        }

        size_t SynetDynamicQuantizedInnerProduct16b::InternalBufferSize() const
        {
            return _buffer.RawSize() + _weight.RawSize() + _sums.RawSize() + 
                _scale.RawSize() + _bias.RawSize() + _params.RawSize();
        }

        void SynetDynamicQuantizedInnerProduct16b::SetParams(const uint8_t* weight, const float* scale, const float* bias, const float* params)
        {
            const DynamicQuantizedInnerProduct16bParam& p = _param;

            if (p.constB)
            {
                _scale.Resize(_aN, true);
                _sums.Resize(_aN, true);
                if (p.typeB == SimdTensorData8i)
                {
                    const int8_t* B = (int8_t*)weight;
                    SetWeight(B);
                    for (size_t j = 0; j < p.N; ++j)
                        _scale[j] = scale[j];
                    for (size_t j = 0; j < p.N; ++j)
                        for (size_t k = 0; k < p.K; ++k)
                            _sums[j] -= B[k * p.N + j];
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

        void SynetDynamicQuantizedInnerProduct16bRef_QuantizeAf(const float* src, size_t size, uint8_t* dst, float* scale, int32_t* zero)
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
            const int qmin = 0, qmax = 255;
            scale[0] = max == min ? 1.0f : (max - min) / float(qmax - qmin);
            float initialZeroPoint = qmin - min / scale[0];
            zero[0] = NearByInt(Simd::Max(float(qmin), Simd::Min(float(qmax), initialZeroPoint)));
            float norm = 1.0f / scale[0];
            for (size_t i = 0; i < size; ++i)
                dst[i] = QuantizeLinear(src[i], norm, zero[0], qmin, qmax);
        }

        void SynetDynamicQuantizedInnerProduct16bRef_QuantizeAb(const uint16_t* src, size_t size, uint8_t* dst, float* scale, int32_t* zero)
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
            const int qmin = 0, qmax = 255;
            scale[0] = max == min ? 1.0f : (max - min) / float(qmax - qmin);
            float initialZeroPoint = qmin - min / scale[0];
            zero[0] = NearByInt(Simd::Max(float(qmin), Simd::Min(float(qmax), initialZeroPoint)));
            float norm = 1.0f / scale[0];
            for (size_t i = 0; i < size; ++i)
                dst[i] = QuantizeLinear(BFloat16ToFloat32(src[i]), norm, zero[0], qmin, qmax);
        }

        //-------------------------------------------------------------------------------------------------

        void SynetDynamicQuantizedInnerProduct16bRef_QuantizeBf(const float* src, size_t N, size_t K, bool overflow, int8_t * dst, float *scale, int32_t *sums)
        {
            int lo = overflow ? -64 : -128, hi = overflow ? 63 : 127;
            float q = overflow ? 63.0f : 127.0f;
            for (size_t j = 0; j < N; ++j)
            {
                float max = 0;
                for (size_t k = 0; k < K; ++k)
                {
                    size_t offset = k * N + j;
                    max = Simd::Max(max, Simd::Abs(src[offset]));
                }
                float range = Simd::Max(0.000001f, max), invScale = q / range;
                scale[j] = range / q;
                int sum = 0;
                for (size_t k = 0; k < K; ++k)
                {
                    size_t offset = k * N + j;
                    int val = Simd::RestrictRange(NearByInt(src[offset] * invScale), lo, hi);
                    dst[offset] = val;
                    sum += val;
                }
                sums[j] = -sum;
            }
        }

        void SynetDynamicQuantizedInnerProduct16bRef_QuantizeBb(const uint16_t* src, size_t N, size_t K, bool overflow, int8_t* dst, float* scale, int32_t* sums)
        {
            int lo = overflow ? -64 : -128, hi = overflow ? 63 : 127;
            float q = overflow ? 63.0f : 127.0f;
            for (size_t j = 0; j < N; ++j)
            {
                float max = 0;
                for (size_t k = 0; k < K; ++k)
                {
                    size_t offset = k * N + j;
                    max = Simd::Max(max, Simd::Abs(BFloat16ToFloat32(src[offset])));
                }
                float range = Simd::Max(0.000001f, max), invScale = q / range;
                scale[j] = range / q;
                int sum = 0;
                for (size_t k = 0; k < K; ++k)
                {
                    size_t offset = k * N + j;
                    int val = Simd::RestrictRange(NearByInt(BFloat16ToFloat32(src[offset]) * invScale), lo, hi);
                    dst[offset] = val;
                    sum += val;
                }
                sums[j] = -sum;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void SynetDynamicQuantizedInnerProduct16bRef_Gemm(size_t N, size_t K, const uint8_t* A, const int8_t* B, int32_t* C, bool overflow)
        {
            const size_t K2 = overflow ? K / 2 * 2 : 0;
            for (size_t j = 0; j < N; ++j)
                C[j] = 0;
            size_t k = 0;
            for (; k < K2; k += 2)
            {
                int32_t a0 = A[k + 0];
                int32_t a1 = A[k + 1];
                const int8_t* b0 = B + (k + 0) * N;
                const int8_t* b1 = B + (k + 1) * N;
                for (size_t j = 0; j < N; ++j)
                    C[j] += RestrictRange(a0 * b0[j] + a1 * b1[j], SHRT_MIN, SHRT_MAX);
            }
            for (; k < K; k += 1)
            {
                int32_t a0 = A[k];
                const int8_t* b0 = B + k * N;
                for (size_t j = 0; j < N; ++j)
                    C[j] += a0 * b0[j];
            }
        }

        //-------------------------------------------------------------------------------------------------

        void SynetDynamicQuantizedInnerProduct16bRef_Norm(const int32_t* src, size_t N, int32_t aZero, const int32_t* bSums, float aScale, const float *bScale, float* dst)
        {
            for (size_t j = 0; j < N; ++j)
                dst[j] = float(src[j] + aZero * bSums[j]) * aScale * bScale[j];
        }

        //-------------------------------------------------------------------------------------------------

        SynetDynamicQuantizedInnerProduct16bRef::SynetDynamicQuantizedInnerProduct16bRef(const DynamicQuantizedInnerProduct16bParam& p)
            : SynetDynamicQuantizedInnerProduct16b(p)
        {
            _sizeA = p.K;
            if (_param.constB == SimdFalse)
                _sizeB = p.K * p.N;
            _sizeC = (p.typeC == SimdTensorData16b) ? p.N : 0;
            _aN = p.N;
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
            if (_sizeB)
            {
                size += _sizeB * sizeof(int8_t);
                size += _aN * sizeof(int32_t);
                size += _aN * sizeof(float);
            }
            size += _sizeC * sizeof(int32_t);
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
            uint8_t* bufA = Allocate<uint8_t>(buf, _sizeA, 1);
            int8_t* bufB = _sizeB ? Allocate<int8_t>(buf, _sizeB, 1) : _weight.data;
            float* bScale = _sizeB ? Allocate<float>(buf, _aN, 1) : (float*)_scale.data;
            int32_t* bSums = _sizeB ? Allocate<int32_t>(buf, _aN, 1) : (int32_t*)_sums.data;
            int32_t* bufC = _sizeC ? Allocate<int32_t>(buf, _sizeC, 1) : (int32_t*)C;
            if (p.constB == SimdFalse)
            {
                if(p.typeB == SimdTensorData16b)
                    SynetDynamicQuantizedInnerProduct16bRef_QuantizeBb((uint16_t*)B, p.N, p.K, p.overflow, bufB, bScale, bSums);
                else
                    SynetDynamicQuantizedInnerProduct16bRef_QuantizeBf((float*)B, p.N, p.K, p.overflow, bufB, bScale, bSums);
            }
            for (size_t i = 0; i < p.M; ++i)
            {
                int32_t aZero;
                float aScale;
                if (p.typeA == SimdTensorData16b)
                    SynetDynamicQuantizedInnerProduct16bRef_QuantizeAb((uint16_t*)A + i * p.K, p.K, bufA, &aScale, &aZero);
                else
                    SynetDynamicQuantizedInnerProduct16bRef_QuantizeAf((float*)A + i * p.K, p.K, bufA, &aScale, &aZero);
                
                SynetDynamicQuantizedInnerProduct16bRef_Gemm(p.N, p.K, bufA, bufB, bufC, overflow);

                SynetDynamicQuantizedInnerProduct16bRef_Norm(bufC, p.N, aZero, bSums, aScale, bScale, (float*)bufC);

                ConvolutionBiasAndActivation(_bias.data, p.N, 1, p.activation, _params.data, SimdTrue, (float*)bufC);

                if (_sizeC == 0)
                    bufC += p.N;
                else
                    Float32ToBFloat16((float*)bufC, p.N, (uint16_t*)C + i * p.N);
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
            SynetDynamicQuantizedInnerProduct16bRef_QuantizeBf(weight, p.N, p.K, p.overflow, _weight.data, _scale.data, _sums.data);
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
