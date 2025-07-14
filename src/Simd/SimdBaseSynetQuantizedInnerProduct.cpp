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

        SetB(b);

        _bScale.Assign(bScale, p.N);

        SetBias(b, bias);

        _cScale = cScale ? cScale[0] : 0.0f;

        _cZero.Resize(p.N, true);
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
        if (bias)
            _bias.Assign(bias, p.N);
        else
            _bias.Resize(p.N, true);
        int aZero = _aZero[0];
        int32_t* pb = _bias.data;
        if (p.transB)
        {
            for (size_t j = 0; j < p.N; ++j)
                for (size_t k = 0; k < p.K; ++k)
                    pb[j] -= b[k * p.N + j] * aZero;
        }
        else
        {
            for (size_t j = 0; j < p.N; ++j)
                for (size_t k = 0; k < p.K; ++k)
                    pb[j] -= b[j * p.K + k] * aZero;
        }
    }

    void SynetQuantizedInnerProduct::SetOther()
    {
        const QuantizedInnerProductParam& p = _param;
        _norm.Resize(p.N);
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

        void SynetQuantizedInnerProductRef::Forward(const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C)
        {

        }

        void SynetQuantizedInnerProductRef::SetB(const int8_t* b)
        {

        }
        
        //-------------------------------------------------------------------------------------------------

        void* SynetQuantizedInnerProductInit(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB, SimdTensorDataType typeC, SimdBool transB, SimdBool constB, SimdBool bias)
        {
            QuantizedInnerProductParam param(M, N, K, typeA, typeB, typeC, transB, constB, bias);
            if (!param.Valid())
                return NULL;
            return NULL;// new SynetQuantizedInnerProductRef(param);
        }
    }
#endif
}
