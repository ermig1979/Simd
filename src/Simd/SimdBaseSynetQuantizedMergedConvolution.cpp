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
            size += _weight[c].RawSize() + _bias[c].RawSize() + _norm[c].RawSize();
        return size;
    }

    void SynetQuantizedMergedConvolution::SetParams(const float* imgScale, const uint8_t* imgZero, const int8_t* const* weight, const float* const* weightScale, const int32_t* const* bias)
    {
        const MergConvParam& p = _param;
        for (size_t i = 0, n = p.count + (p.add ? 1 : 0); i <= n; ++i)
        {
            _imgScale[i] = imgScale[i];
            _imgZero[i] = imgZero[i];
        }
        for (size_t c = 0; c < p.count; ++c)
        {
            if (p.conv[c].IsDepthwise())
            {
                _dwSrcZero.Resize(p.conv[c].group);
                memset(_dwSrcZero.data, imgZero[c], p.conv[c].group);
                SetDepthwise(weight[c], p.conv[c], _weight[c]);
            }
            else
            {
                if(c == 0)
                    SetInput(weight[c], p.conv[c], _weight[c]);
                else
                    SetOutput(weight[c], p.conv[c], _weight[c]);
            }

            SetBias(weight[c], bias[c], imgZero[c], p.conv[c], _bias[c]);

            SetNorm(weightScale[c], imgScale[c], imgScale[c + 1], p.conv[c], _norm[c]);
        }
        if (p.add)
        {
            assert(p.count == 3 && p.conv[0].srcC == p.conv[2].dstC && p.conv[0].srcH == p.conv[2].dstH && p.conv[0].srcW == p.conv[2].dstW);
            _srcBias = -imgZero[0];
            _srcNorm = imgScale[0];
            _dstBias = -imgZero[3];
            _dstNorm = imgScale[3];
            _addZero = imgZero[4];
            _addScale = 1.0f / imgScale[4];
        }
    }

    void SynetQuantizedMergedConvolution::SetBias(const int8_t* weight, const int32_t* bias, int32_t zero, const ConvParam& p, Array32i& dst)
    {
        if (bias)
            dst.Assign(bias, p.dstC);
        else
            dst.Resize(p.dstC, true);
        size_t K = p.kernelY * p.kernelX * p.srcC / p.group, D = p.dstC;
        for (size_t d = 0; d < D; ++d)
            for (size_t k = 0; k < K; ++k)
                dst[d] -= weight[k * D + d] * zero;
    }

    void SynetQuantizedMergedConvolution::SetNorm(const float* weightScale, float srcScale, float dstScale, const ConvParam& p, Array32f& dst)
    {
        size_t D = p.dstC;
        dst.Resize(D);
        for (size_t d = 0; d < D; ++d)
            dst[d] = srcScale * weightScale[d] / dstScale;
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
