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
#include "Simd/SimdSynetQuantizedConvolution.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    SynetQuantizedConvolution::SynetQuantizedConvolution(const ConvParam& p)
        : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        , _perf(NULL)
#endif
    {
        _sizeS = p.srcC * p.srcH * p.srcW;
        _sizeD = p.dstC * p.dstH * p.dstW;
        _merge = 1;
        _weight.Resize(p.kernelY * p.kernelX * p.srcC / p.group * p.dstC);
        _norm.Resize(p.dstC);
        _bias.Resize(p.dstC);
    }

    size_t SynetQuantizedConvolution::ExternalBufferSize() const
    {
        size_t size = SIMD_ALIGN;
        return size;
    }

    size_t SynetQuantizedConvolution::InternalBufferSize() const
    {
        return _buffer.RawSize() + _weight.RawSize() + _srcZero.RawSize() + _dstZero.RawSize() + _norm.RawSize() + _bias.RawSize();
    }

    void SynetQuantizedConvolution::SetParams(const int8_t* weight, const int32_t* bias, const float* norm, const uint8_t* srcZero, const uint8_t* dstZero)
    {
        const ConvParam& p = _param;
    }

    void SynetQuantizedConvolution::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
    {
        if (buf == NULL)
        {
            _buffer.Resize(ExternalBufferSize());
            buf = _buffer.data;
        }
        const ConvParam& p = _param;
        for (size_t b = 0; b < p.batch; b += _merge)
        {
            Forward8u(src + b * _sizeS, buf, dst + b * _sizeD);
        }
    }

    void SynetQuantizedConvolution::Forward8u(const uint8_t* src, uint8_t* buf, uint8_t* dst)
    {
        const ConvParam& p = _param;
    }

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
    Base::PerformanceMeasurer * SynetQuantizedConvolution::Perf(const char* func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

    namespace Base
    {

        //-------------------------------------------------------------------------------------------------

        void* SynetQuantizedConvolutionInit(size_t batch, const SimdConvolutionParameters* conv)
        {
            return NULL;
        }
    }
#endif
}
