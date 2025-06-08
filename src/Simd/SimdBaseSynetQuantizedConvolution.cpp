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
        _weight.Assign(weight, _weight.size);
        _bias.Assign(bias, _bias.size);
        _norm.Assign(norm, _norm.size);
        _srcZero.Assign(srcZero, _bias.size);
        _dstZero.Assign(dstZero, _bias.size);
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
        SIMD_INLINE int NearByInt(float value)
        {
            return (int)std::nearbyint(value);
        }

        SIMD_INLINE int QuantizeSumLinear(int sum, int bias, float norm, int zero, int min, int max)
        {
            return RestrictRange(NearByInt(float(sum + bias) * norm) + zero, min, max);
        }

        SIMD_INLINE void QuantizeSumLinear(const int32_t* sum, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const int32_t* bias, const float* norm, const uint8_t* zero, uint8_t* dst)
        {
            int min = std::numeric_limits<uint8_t>::min();
            int max = std::numeric_limits<uint8_t>::max();
            for (size_t b = 0; b < batch; ++b)
            {
                if (format == SimdTensorFormatNchw)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        int32_t _bias = bias[c];
                        float _norm = norm[c];
                        int32_t _zero = zero[c];
                        for (size_t h = 0; h < height; ++h)
                        {
                            for (size_t w = 0; w < width; ++w)
                                dst[w] = (uint8_t)QuantizeSumLinear(sum[w], _bias, _norm, _zero, min, max);
                            sum += width;
                            dst += width;
                        }
                    }
                }
                else if (format == SimdTensorFormatNhwc)
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                        {
                            for (size_t c = 0; c < channels; ++c)
                                dst[c] = (uint8_t)QuantizeSumLinear(sum[c], bias[c], norm[c], zero[c], min, max);
                            sum += channels;
                            dst += channels;
                        }
                    }
                }
                else
                    assert(0);
            }
        }

        //------------------------------------------------------------------------------------------------

        SynetQuantizedConvolutionGemmNN::SynetQuantizedConvolutionGemmNN(const ConvParam& p)
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

        size_t SynetQuantizedConvolutionGemmNN::ExternalBufferSize() const
        {
            size_t size = SynetQuantizedConvolution::ExternalBufferSize();
            if (!_skipConv)
                size += AlignHi(_sizeB * _merge * sizeof(uint8_t), SIMD_ALIGN);
            size += AlignHi(_sizeD * _merge * sizeof(int32_t), SIMD_ALIGN);
            return size;
        }

        void SynetQuantizedConvolutionGemmNN::Forward8u(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const int8_t* weight = _weight.data;
            int32_t* sum = Allocate<int32_t>(buf, _sizeD * _merge);
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
                bool overflow = Overflow(p.compatibility);
                for (size_t g = 0; g < p.group; ++g)
                {
                    if (p.trans)
                        GemmNhwc(_siS, _siD, _siK, _siC, src + _grS * g, _ldS, weight + _grW * g, _ldW, sum + _grD * g, _ldD, overflow);
                    else
                        GemmNchw(_siD, _siS, _siC, _siK, weight + _grW * g, _ldW, src + _grS * g, _ldS, sum + _grD * g, _ldD, overflow);
                }
            }
            if (p.activation == SimdConvolutionActivationIdentity || p.activation == SimdConvolutionActivationRelu)
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
                return new SynetQuantizedConvolutionGemmNN(param);
        }
    }
#endif
}
