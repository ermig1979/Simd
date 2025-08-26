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
#if defined(SIMD_SYNET_ENABLE)
    SynetQuantizedMergedConvolution::SynetQuantizedMergedConvolution(const MergConvParam& p)
        : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        , _perf(NULL)
#endif
    {
        _merge = 1;
    }

    uint8_t* SynetQuantizedMergedConvolution::Buffer(uint8_t* buffer)
    {
        if (buffer)
            return buffer;
        else
        {
            _buffer.Resize(ExternalBufferSize());
            return _buffer.data;
        }
    }

    const char* SynetQuantizedMergedConvolution::Info() const
    {
        _info = Desc();
        return _info.c_str();
    }

    size_t SynetQuantizedMergedConvolution::ExternalBufferSize() const
    {
        size_t size = SIMD_ALIGN;
        return size;
    }

    size_t SynetQuantizedMergedConvolution::InternalBufferSize() const
    {
        size_t size = _buffer.RawSize() + _dwSrcZero.RawSize();
        for(size_t c = 0; c < _count; ++c)
            size += _weight[c].RawSize() + _bias[c].RawSize() + _weightScale[3].RawSize() + _norm[3].RawSize();
        return size;
    }

    void SynetQuantizedMergedConvolution::SetParams(const float* srcScale, const uint8_t* srcZero, const int8_t* const* weight, const float* const* weightScale, const int32_t* const* bias, const float* dstScale, const uint8_t* dstZero)
    {
        const MergConvParam& p = _param;

/*        _srcScale = srcScale ? srcScale[0] : 0.0f;

        _srcZero.Resize(p.srcC, true);
        if(srcZero)
            memset(_srcZero.data, srcZero[0], p.srcC);

        SetWeight(weight);

        _weightScale.Assign(weightScale, p.dstC);

        SetBias(weight, bias);

        if (params)
            _params.Assign(params, p.activation == SimdConvolutionActivationPrelu ? p.dstC : 2);
        else
            _params.Resize(p.dstC, true);

        _dstScale = dstScale ? dstScale[0] : 0.0f;

        _dstZero.Resize(p.dstC, true);
        if (dstZero)
        {
            for (size_t d = 0; d < p.dstC; ++d)
                _dstZero[d] = dstZero[0];
        }*/

        SetOther();
    }

    void SynetQuantizedMergedConvolution::SetBias(const int8_t* const* weight, const int32_t* const* bias)
    {
        const MergConvParam& p = _param;
        //if (bias)
        //    _bias.Assign(bias, p.dstC);
        //else
        //    _bias.Resize(p.dstC, true);
        //size_t K = p.kernelY * p.kernelX * p.srcC / p.group, D = p.dstC;
        //int srcZero = _srcZero[0];
        //int32_t* pb = _bias.data;
        //if (p.trans)
        //{
        //    for (size_t d = 0; d < D; ++d)
        //        for (size_t k = 0; k < K; ++k)
        //            pb[d] -= weight[k * D + d] * srcZero;
        //}
        //else
        //{
        //    for (size_t d = 0; d < D; ++d)
        //        for (size_t k = 0; k < K; ++k)
        //            pb[d] -= weight[d * K + k] * srcZero;
        //}
    }

    void SynetQuantizedMergedConvolution::SetOther()
    {
        const MergConvParam& p = _param;
        //size_t D = p.dstC;
        //_norm.Resize(D);
        //const float* psw = _weightScale.data;
        //float* pn = _norm.data;
        //for (size_t d = 0; d < D; ++d)
        //    pn[d] = _srcScale * psw[d] / _dstScale;
    }

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
    Base::PerformanceMeasurer * SynetQuantizedMergedConvolution::Perf(const char* func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

    //------------------------------------------------------------------------------------------------

    namespace Base
    {

        //-------------------------------------------------------------------------------------------------

        void* SynetQuantizedMergedConvolutionInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add)
        {
            MergConvParam param(batch, convs, count, add);
            if (!param.Valid(SimdTensorData8u, SimdTensorData8u))
                return NULL;
            return NULL;
        }
    }
#endif
}
