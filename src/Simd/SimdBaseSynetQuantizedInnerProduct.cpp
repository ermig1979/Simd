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
#include "Simd/SimdSynetQuantizedInnerProduct.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)

    SynetQuantizedInnerProduct::SynetQuantizedInnerProduct(const QuantizedInnerProductParam& p)
        : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        , _perf(NULL)
#endif
    {
        _a8u = p.typeA == SimdTensorData8u;
        _c8u = p.typeC == SimdTensorData8u;
        _elemA = _a8u ? 1 : 4;
        _elemC = _c8u ? 1 : 4;
        _sizeA = p.M * p.K;
        _sizeC = p.M * p.N;
        _aN = p.N;
    }

    size_t SynetQuantizedInnerProduct::ExternalBufferSize() const
    {
        size_t size = SIMD_ALIGN;
        return size;
    }

    size_t SynetQuantizedInnerProduct::InternalBufferSize() const
    {
        return _buffer.RawSize() + _b.RawSize() + _aZero.RawSize() + _cZero.RawSize() + _norm.RawSize() +
            _bias.RawSize() + _bScale.RawSize() + _norm.RawSize();
    }

    void SynetQuantizedInnerProduct::SetParams(const float* aScale, const uint8_t* aZero, const int8_t* b, const float* bScale, const int32_t* bias, const float* cScale, const uint8_t* cZero)
    {
        const QuantizedInnerProductParam& p = _param;

        _aScale = aScale ? aScale[0] : 0.0f;

        _aZero.Resize(p.K, true);
        if (aZero)
            memset(_aZero.data, aZero[0], p.K);

        _bScale.Resize(_aN, true);
        for (size_t j = 0; j < p.N; ++j)
            _bScale[j] = bScale[j];

        SetB(b);

        SetBias(b, bias);

        _cScale = cScale ? cScale[0] : 0.0f;

        _cZero.Resize(_aN, true);
        if (cZero)
        {
            for (size_t j = 0; j < p.N; ++j)
                _cZero[j] = cZero[0];
        }

        SetOther();
    }

    void SynetQuantizedInnerProduct::SetBias(const int8_t* b, const int32_t* bias)
    {
        const QuantizedInnerProductParam& p = _param;
        _bias.Resize(_aN, true);
        if (bias)
        {
            for (size_t j = 0; j < p.N; ++j)
                _bias[j] = bias[j];
        }
        int aZero = _aZero[0];
        int32_t* pb = _bias.data;
        if (p.transB)
        {
            for (size_t j = 0; j < p.N; ++j)
                for (size_t k = 0; k < p.K; ++k)
                    pb[j] -= b[j * p.K + k] * aZero;
        }
        else
        {
            for (size_t j = 0; j < p.N; ++j)
                for (size_t k = 0; k < p.K; ++k)
                    pb[j] -= b[k * p.N + j] * aZero;
        }
    }

    void SynetQuantizedInnerProduct::SetOther()
    {
        const QuantizedInnerProductParam& p = _param;
        _norm.Resize(_aN, true);
        const float* psb = _bScale.data;
        float* pn = _norm.data;
        for (size_t j = 0; j < p.N; ++j)
            pn[j] = _aScale * psb[j] / _cScale;
    }

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
    Base::PerformanceMeasurer* SynetQuantizedInnerProduct::Perf(const char* func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        SynetQuantizedInnerProductRef::SynetQuantizedInnerProductRef(const QuantizedInnerProductParam& p)
            :SynetQuantizedInnerProduct(p)
        {
        }

        String SynetQuantizedInnerProductRef::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::Ref";
            return desc.str();
        }

        size_t SynetQuantizedInnerProductRef::ExternalBufferSize() const
        {
            size_t size = SynetQuantizedInnerProduct::ExternalBufferSize();
            if (!_a8u)
                size += _sizeA;
            size += _sizeC * sizeof(int32_t);
            return size;
        }

        void SynetQuantizedInnerProductRef::Forward(const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C)
        {
            const QuantizedInnerProductParam& p = _param;
            buf = Buffer(buf);
            uint8_t* bufA = (uint8_t*)A;
            if (!_a8u)
            {
                bufA = Allocate<uint8_t>(buf, _sizeA);
                SynetQuantizeLinear((float*)A, _sizeA, &_aScale, _aZero[0], bufA);
            }
            const int8_t* bufB = (int8_t*)B;
            if (p.constB)
                bufB = _b.data;
            int32_t* bufC = Allocate<int32_t>(buf, _sizeC);
            Gemm(bufA, bufB, bufC);
            if (_c8u)
                QuantizeSumLinear(bufC, 1, p.N, 1, p.M, SimdTensorFormatNhwc, _bias.data, _norm.data, _cZero.data, C);
            else
                assert(0);
        }

        void SynetQuantizedInnerProductRef::Gemm(const uint8_t* A, const int8_t* B, int32_t* C)
        {
            const QuantizedInnerProductParam& p = _param;
            const bool overflow = true;
            size_t K2 = overflow ? p.K / 2 * 2 : 0, k;
            for (size_t i = 0; i < p.M; ++i)
            {
                if (p.transB)
                {
                    for (size_t j = 0; j < p.N; ++j)
                    {
                        const int8_t* b = B + j * p.K;
                        C[j] = 0;
                        for (k = 0; k < K2; k += 2)
                            C[j] += RestrictRange(int(A[k + 0]) * int(b[k + 0]) + int(A[k + 1]) * int(b[k + 1]), SHRT_MIN, SHRT_MAX);
                        for (; k < p.K; ++k)
                            C[j] += int(A[k]) * int(b[k]);
                    }
                }
                else
                {
                    for (size_t j = 0; j < p.N; ++j)
                        C[j] = 0;
                    for (k = 0; k < K2; k += 2)
                    {
                        const int8_t* b = B + k * p.N;
                        for (size_t j = 0; j < p.N; ++j)
                            C[j] += RestrictRange(int(A[k + 0]) * int(b[j + 0]) + int(A[k + 1]) * int(b[j + p.N]), SHRT_MIN, SHRT_MAX);
                    }
                    for (; k < p.K; ++k)
                    {
                        const int8_t* b = B + k * p.N;
                        for (size_t j = 0; j < p.N; ++j)
                            C[j] += int(A[k]) * int(b[j]);
                    }
                }
                A += p.K;
                C += p.N;
            }
        }

        void SynetQuantizedInnerProductRef::SetB(const int8_t* b)
        {
            const QuantizedInnerProductParam& p = _param;
            _b.Resize(p.N * p.K);
            _b.Assign(b, _b.size);
        }
        
        //-------------------------------------------------------------------------------------------------

        void* SynetQuantizedInnerProductInit(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB, SimdTensorDataType typeC, SimdBool transB, SimdBool constB, SimdBool bias)
        {
            QuantizedInnerProductParam param(M, N, K, typeA, typeB, typeC, transB, constB, bias);
            if (!param.Valid())
                return NULL;
            return new SynetQuantizedInnerProductRef(param);
        }
    }
#endif
}
