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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetInnerProduct16b.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdAlignment.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)

    SynetInnerProduct16b::SynetInnerProduct16b(const InnerProductParam16b& p)
        : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        , _perf(NULL)
#endif
        , _sizeA(0)
        , _sizeB(0)
        , _sizeC(0)
        , _sizeS(0)
    {
    }

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
    Base::PerformanceMeasurer* SynetInnerProduct16b::Perf(const char* func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

    size_t SynetInnerProduct16b::InternalBufferSize() const
    {
        return _buffer.RawSize() + _weight.RawSize() + _bias.RawSize() + _params.RawSize();
    }

    size_t SynetInnerProduct16b::ExternalBufferSize() const
    {
        return _sizeA * 2 + _sizeB * 2 + _sizeC * 4 + _sizeS * 4;
    }

    uint8_t* SynetInnerProduct16b::Buffer(uint8_t* buffer)
    {
        if (buffer)
            return buffer;
        else
        {
            _buffer.Resize(ExternalBufferSize());
            return _buffer.data;
        }
    }

    void SynetInnerProduct16b::SetBias(const float* bias, size_t align)
    {
        const InnerProductParam16b& p = _param;
        _bias.Resize(AlignHi(p.N, align), true);
        if (bias && p.bias)
            memcpy(_bias.data, bias, p.N * sizeof(float));
    }

    void SynetInnerProduct16b::SetParams(const float* params, size_t align)
    {
        const InnerProductParam16b& p = _param;
        if (p.activation == SimdConvolutionActivationLeakyRelu || p.activation == SimdConvolutionActivationPrelu)
            _params.Resize(AlignHi(p.N, align), true);
        else
            _params.Resize(2, true);
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
            for (size_t d = 0; d < p.N; ++d)
                _params.data[d] = params[0];
            break;
        case SimdConvolutionActivationRestrictRange:
            _params.data[0] = params[0];
            _params.data[1] = params[1];
            break;
        case SimdConvolutionActivationPrelu:
            for (size_t d = 0; d < p.N; ++d)
                _params.data[d] = params[d];
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

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        SynetInnerProduct16bRef::SynetInnerProduct16bRef(const InnerProductParam16b& p)
            :SynetInnerProduct16b(p)
        {
            _sizeA = p.typeA == SimdTensorData32f ? p.M * p.K : 0;
            _sizeB = (p.typeB == SimdTensorData32f && !p.constB) ? p.K * p.N : 0;
            _sizeC = p.typeC == SimdTensorData16b ? p.M * p.N : 0;
        }

        String SynetInnerProduct16bRef::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::Ref";
            return desc.str();
        }
        
        void SynetInnerProduct16bRef::SetParams(const float* weight, const float* bias, const float* params)
        {
            const InnerProductParam16b& p = _param;
            if (p.constB)
            {
                assert(weight);
                _weight.Resize(p.K * p.N);
                Float32ToBFloat16(weight, p.K * p.N, _weight.data);
            }
            SynetInnerProduct16b::SetBias(bias, Alignment());
            SynetInnerProduct16b::SetParams(params, Alignment());
        }

        void SynetInnerProduct16bRef::Forward(const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C)
        {
            const InnerProductParam16b& p = _param;
            buf = Buffer(buf);
            uint16_t* bufA = (uint16_t*)A;
            if (_sizeA)
            {
                bufA = Allocate<uint16_t>(buf, _sizeA);
                Float32ToBFloat16((float*)A, _sizeA, bufA);
            }
            uint16_t* bufB = (uint16_t*)B;
            if (_sizeB)
            {
                bufB = Allocate<uint16_t>(buf, _sizeB);
                Float32ToBFloat16((float*)B, _sizeB, bufB);
            }
            else if (p.constB)
                bufB = _weight.data;
            float* bufC = (float*)C;
            if (_sizeC)
                bufC = Allocate<float>(buf, _sizeC);
            GemmAndBias(bufA, bufB, bufC);
            if (_sizeC)
                Float32ToBFloat16(bufC, _sizeC, (uint16_t*)C);
        }

        void SynetInnerProduct16bRef::GemmAndBias(const uint16_t* A, const uint16_t* B, float* C)
        {
            const InnerProductParam16b& p = _param;
            Array32f Af(p.K);
            for (size_t i = 0; i < p.M; ++i)
            {            
                for (size_t k = 0; k < p.K; ++k)
                    Af[k] = BFloat16ToFloat32(A[k]);
                if (p.transB)
                {
                    for (size_t j = 0; j < p.N; ++j)
                    {
                        const uint16_t* pB = B + j * p.K;
                        C[j] = 0;
                        for (size_t k = 0; k < p.K; ++k)
                            C[j] += Af[k] * BFloat16ToFloat32(pB[k]);
                    }
                }
                else
                {
                    for (size_t j = 0; j < p.N; ++j)
                        C[j] = 0.0;
                    for (size_t k = 0; k < p.K; ++k)
                    {
                        const uint16_t* pB = B + k * p.N;
                        for (size_t j = 0; j < p.N; ++j)
                            C[j] += Af[k] * BFloat16ToFloat32(pB[j]);
                    }
                }
                ConvolutionBiasAndActivation(_bias.data, p.N, 1, p.activation, _params.data, SimdTrue, C);
                A += p.K;
                C += p.N;
            }  
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetInnerProduct16bInit(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB, SimdTensorDataType typeC, SimdBool transB, SimdBool constB, SimdBool bias, SimdConvolutionActivationType activation)
        {
            InnerProductParam16b param(M, N, K, typeA, typeB, typeC, transB, constB, bias, activation);
            if (!param.Valid())
                return NULL;
            return new SynetInnerProduct16bRef(param);
        }
    }
#endif
}
