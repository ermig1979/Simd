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
    SynetQuantizedConvolution::SynetQuantizedConvolution(const ConvParam& p)
        : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        , _perf(NULL)
#endif
    {
        _src8u = p.srcT == SimdTensorData8u;
        _dst8u = p.dstT == SimdTensorData8u;
        _elemS = _src8u ? 1 : 4;
        _elemD = _dst8u ? 1 : 4;
        _is1x1 = p.Is1x1();
        _sizeS = p.srcC * p.srcH * p.srcW;
        _sizeD = p.dstC * p.dstH * p.dstW;
        _merge = 1;
    }

    size_t SynetQuantizedConvolution::ExternalBufferSize() const
    {
        size_t size = SIMD_ALIGN;
        return size;
    }

    size_t SynetQuantizedConvolution::InternalBufferSize() const
    {
        return _buffer.RawSize() + _weight.RawSize() + _srcZero.RawSize() + _dstZero.RawSize() + _norm.RawSize() + 
            _bias.RawSize() + _weightScale.RawSize() + _norm.RawSize();
    }

    void SynetQuantizedConvolution::SetParams(const float* srcScale, const uint8_t* srcZero, const int8_t* weight, const float* weightScale, const int32_t* bias, const float* params, const float* dstScale, const uint8_t* dstZero)
    {
        const ConvParam& p = _param;

        _srcScale = srcScale ? srcScale[0] : 0.0f;

        _srcZero.Resize(p.srcC, true);
        if(srcZero)
            memset(_srcZero.data, srcZero[0], p.srcC);

        SetWeight(weight);

        _weightScale.Assign(weightScale, p.dstC);

        SetBias(bias);

        if (params)
            _params.Assign(params, p.activation == SimdConvolutionActivationPrelu ? p.dstC : 2);
        else
            _params.Resize(p.dstC, true);

        _dstScale = dstScale ? dstScale[0] : 0.0f;

        _dstZero.Resize(p.dstC, true);
        if(dstZero)
            memset(_dstZero.data, dstZero[0], p.dstC);

        SetOther();
    }

    void SynetQuantizedConvolution::SetBias(const int32_t* bias)
    {
        const ConvParam& p = _param;
        if (bias)
            _bias.Assign(bias, p.dstC);
        else
            _bias.Resize(p.dstC, true);
        size_t K = p.kernelY * p.kernelX * p.srcC / p.group, D = p.dstC;
        int srcZero = _srcZero[0];
        const int8_t* pw = _weight.data;
        int32_t* pb = _bias.data;
        if (p.trans)
        {
            for (size_t d = 0; d < D; ++d)
                for (size_t k = 0; k < K; ++k)
                    pb[d] -= pw[k * D + d] * srcZero;
        }
        else
        {
            for (size_t d = 0; d < D; ++d)
                for (size_t k = 0; k < K; ++k)
                    pb[d] -= pw[d * K + k] * srcZero;
        }
    }

    void SynetQuantizedConvolution::SetOther()
    {
        const ConvParam& p = _param;
        size_t D = p.dstC;
        _norm.Resize(D);
        const float* psw = _weightScale.data;
        float* pn = _norm.data;
        for (size_t d = 0; d < D; ++d)
            pn[d] = _srcScale * psw[d] / _dstScale;
    }

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
    Base::PerformanceMeasurer * SynetQuantizedConvolution::Perf(const char* func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

    //------------------------------------------------------------------------------------------------

    namespace Base
    {
        SynetQuantizedConvolutionGemm::SynetQuantizedConvolutionGemm(const ConvParam& p)
            : SynetQuantizedConvolution(p)
        {
            if (p.IsDilation(1) && p.IsStride(1) && p.IsPad(0))
            {
                _skipConv = p.IsKernel(1) || (p.srcH == p.kernelY && p.srcW == p.kernelX);
            }
            else
                _skipConv = false;
            _sizeB = p.srcC * p.kernelY * p.kernelX * p.dstH * p.dstW;
            if (p.trans)
            {
                _ldS = p.srcC * p.kernelY * p.kernelX / p.group * (_skipConv ? p.group : 1);
                _ldW = p.dstC;
                _ldD = p.dstC;
                _grW = p.dstC / p.group;
                _grS = p.srcC * p.kernelY * p.kernelX / p.group * (_skipConv ? 1 : p.dstH * p.dstW);
                _grD = p.dstC / p.group;
            }
            else
            {
                _ldW = p.srcC * p.kernelY * p.kernelX / p.group;
                _ldS = p.dstH * p.dstW;
                _ldD = p.dstH * p.dstW;
                _grW = p.dstC / p.group * p.srcC * p.kernelY * p.kernelX / p.group;
                _grS = p.srcC * p.kernelY * p.kernelX / p.group * p.dstH * p.dstW;
                _grD = p.dstH * p.dstW * p.dstC / p.group;
            }
            _siK = p.kernelY * p.kernelX;
            _siC = p.srcC / p.group;
            _siD = p.dstC / p.group;
            _siS = p.dstH * p.dstW;
        }

        size_t SynetQuantizedConvolutionGemm::ExternalBufferSize() const
        {
            size_t size = SynetQuantizedConvolution::ExternalBufferSize();
            if (!_skipConv)
                size += AlignHi(_sizeB * _merge * sizeof(uint8_t), SIMD_ALIGN);
            size += AlignHi(_sizeD * _merge * sizeof(int32_t), SIMD_ALIGN);
            return size;
        }

        void SynetQuantizedConvolutionGemm::SetWeight(const int8_t* weight)
        {
            const ConvParam& p = _param;
            _weight.Resize(p.kernelY * p.kernelX * p.srcC / p.group * p.dstC);
            _weight.Assign(weight, _weight.size);
        }

        void SynetQuantizedConvolutionGemm::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const ConvParam& p = _param;
            buf = Buffer(buf);
            int32_t* sum =  Allocate<int32_t>(buf, _sizeD * _merge);
            for (size_t b = 0; b < p.batch; b += _merge)
            {
                Forward(src + b * _sizeS * _elemS, buf, sum, dst + b * _sizeD * _elemD);
            }
        }

        void SynetQuantizedConvolutionGemm::Forward(const uint8_t* src, uint8_t* buf, int32_t* sum, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const int8_t* weight = _weight.data;
            if (!_skipConv)
            {
                const uint8_t* zero = _srcZero.data;
                if (p.trans)
                    for (size_t m = 0; m < _merge; ++m)
                        ImgToRow(src + m * _sizeS, p, zero, buf + m * _sizeB);
                else
                    for (size_t m = 0; m < _merge; ++m)
                        ImgToCol(src + m * _sizeS, p, zero, buf + m * _sizeB);
                src = buf;
            }
            if (_merge > 1)
            {
                assert(0);
            }
            else
            {
                bool overflow = true;// Overflow(p.compatibility);
                for (size_t g = 0; g < p.group; ++g)
                {
                    if (p.trans)
                        GemmNhwc(_siS, _siD, _siK, _siC, src + _grS * g, _ldS, weight + _grW * g, _ldW, sum + _grD * g, _ldD, overflow);
                    else
                        GemmNchw(_siD, _siS, _siC, _siK, weight + _grW * g, _ldW, src + _grS * g, _ldS, sum + _grD * g, _ldD, overflow);
                }
            }
            if (p.activation == SimdConvolutionActivationIdentity || p.activation == SimdConvolutionActivationRelu || p.activation == SimdConvolutionActivationRestrictRange)
            {
                const float* norm = _norm.data;
                const int32_t* bias = _bias.data;
                const uint8_t* zero = _dstZero.data;
                QuantizeSumLinear(sum, 1, p.dstC, p.dstH, p.dstW, p.dstF, bias, norm, zero, dst);
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetQuantizedConvolutionInit(size_t batch, const SimdConvolutionParameters* conv)
        {
            ConvParam param(batch, conv);
            if (!ValidQuantized(param))
                return NULL;
            else
                return new SynetQuantizedConvolutionGemm(param);
        }
    }
#endif
}
