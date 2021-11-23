/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdSynetMergedConvolution32f.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        template<SimdConvolutionActivationType type, UpdateType update> void DirectConvolution(const float* src, const SimdConvolutionParameters& p,
            size_t maC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW, dstC = p.dstC;
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            Array32f buf(dstC);
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    if (bias)
                        memcpy(buf.data, bias, dstC * sizeof(float));
                    else
                        memset(buf.data, 0, dstC * sizeof(float));
                    for (size_t ky = 0; ky < kernelY; ++ky)
                    {
                        size_t sy = dy * strideY + ky - padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < kernelX; ++kx)
                            {
                                size_t sx = dx * strideX + kx - padX;
                                if (sx < p.srcW)
                                {
                                    const float* pw = weight + (ky * kernelX + kx) * srcC * dstC;
                                    const float* ps = src + (sy * srcW + sx) * srcC;
                                    for (size_t sc = 0; sc < srcC; ++sc)
                                    {
                                        for (size_t dc = 0; dc < dstC; ++dc)
                                            buf[dc] += ps[sc] * pw[dc];
                                        pw += dstC;
                                    }
                                }
                            }
                        }
                    }
                    if (update == UpdateAdd)
                    {
                        for (size_t dc = 0; dc < dstC; ++dc)
                            dst[dc] = Activate<type>(dst[dc] + buf[dc], params, dc);
                    }
                    else
                    {
                        for (size_t dc = 0; dc < dstC; ++dc)
                            dst[dc] = Activate<type>(buf[dc], params, dc);
                    }
                    dst += p.dstC;
                }
            }
        }

        template <SimdConvolutionActivationType type> void Set(const MergConvParam32f& p, size_t index, SynetMergedConvolution32fCdc::ConvolutionPtr * convolution)
        {
            switch (index)
            {
            case 0:
                if(p.conv[0].group == 1)
                    convolution[0] = DirectConvolution<type, UpdateSet>;
                else
                    convolution[0] = DepthwiseConvolution<type>;
                break;
            case 1:
                if (p.conv[1].group == 1)
                    convolution[1] = DirectConvolution<type, UpdateSet>;
                else
                    convolution[1] = DepthwiseConvolution<type>;
                break;
            case 2:
                if (p.add)
                    convolution[2] = DirectConvolution<type, UpdateAdd>;
                else
                    convolution[2] = DirectConvolution<type, UpdateSet>;
                break;
            default:
                assert(0);
            }
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution32f::SynetMergedConvolution32f(const MergConvParam32f& p)
           :  _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
           , _perf(NULL)
#endif        
        {
            for (size_t i = 0; i < 4; ++i)
                _convolution[i] = NULL;
            const SimdConvolutionParameters& beg = p.conv[0];
            const SimdConvolutionParameters& end = p.conv[p.count - 1];
            _sizeS = beg.srcH * beg.srcW * beg.srcC;
            _sizeD = end.dstH * end.dstW * end.dstC;
            _sizeB[0] = p.conv[1].srcH * p.conv[1].srcW * p.conv[1].srcC;
            _sizeB[1] = p.count == 3 ? p.conv[1].dstH * p.conv[1].dstW * p.conv[1].dstC : 0;
            for (size_t i = 0; i < p.count; ++i)
            {
                switch (p.conv[i].activation)
                {
                case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationIdentity>(_param, i, _convolution); break;
                case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRelu>(_param, i, _convolution); break;
                case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationLeakyRelu>(_param, i, _convolution); break;
                case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(_param, i, _convolution); break;
                case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(_param, i, _convolution); break;
                case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(_param, i, _convolution); break;
                case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(_param, i, _convolution); break;
                case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(_param, i, _convolution); break;
                case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(_param, i, _convolution); break;
                case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(_param, i, _convolution); break;
                default: assert(0);
                }
            }
        }

        size_t SynetMergedConvolution32f::ExternalBufferSize() const
        {
            return _sizeB[0] + _sizeB[1];
        }

        size_t SynetMergedConvolution32f::InternalBufferSize() const
        {
            size_t size = _buffer.size;
            for (size_t i = 0; i < _param.count; ++i)
                size += _rWeight[i].size + _rBias[i].size + _rParams[i].size;
            return size;
        }

        void SynetMergedConvolution32f::SetParams(const float* const* weight, SimdBool* internal, const float* const* bias, const float* const* params)
        {
            const MergConvParam32f& p = _param;
            for (size_t i = 0; i < p.count; ++i)
            {
                if (_rWeight[i].data)
                {
                    switch (i)
                    {
                    case 0: ReorderFirstWeight(weight[i], _rWeight[i].data); break;
                    case 1: ReorderSecondWeight(weight[i], _rWeight[i].data); break;
                    case 2: ReorderThirdWeight(weight[i], _rWeight[i].data); break;
                    default: assert(0);
                    }
                    _weight[i] = _rWeight[i].data;
                    if (internal)
                        internal[i] = SimdTrue;
                }
                else
                {
                    _weight[i] = weight[i];
                    if (internal)
                        internal[i] = SimdFalse;
                }
                if (_rBias[i].data)
                {
                    if (bias[i])
                        memcpy(_rBias[i].data, bias[i], p.conv[i].dstC * sizeof(float));
                    _bias[i] = _rBias[i].data;
                }
                else
                    _bias[i] = bias[i];
                if (_rParams[i].data)
                {
                    switch (p.conv[i].activation)
                    {
                    case SimdConvolutionActivationIdentity:
                        _rParams[i].data[0] = -FLT_MAX;
                        _rParams[i].data[1] = FLT_MAX;
                        break;
                    case SimdConvolutionActivationRelu:
                        _rParams[i].data[0] = 0;
                        _rParams[i].data[1] = FLT_MAX;
                        break;
                    case SimdConvolutionActivationLeakyRelu:
                        for (size_t d = 0; d < p.conv[i].dstC; ++d)
                            _rParams[i].data[d] = params[i][0];
                        break;
                    case SimdConvolutionActivationRestrictRange:
                        _rParams[i].data[0] = params[i][0];
                        _rParams[i].data[1] = params[i][1];
                        break;
                    case SimdConvolutionActivationPrelu:
                        for (size_t d = 0; d < p.conv[i].dstC; ++d)
                            _rParams[i].data[d] = params[i][d];
                        break;
                    case SimdConvolutionActivationElu:
                        _rParams[i].data[0] = params[i][0];
                        break;
                    case SimdConvolutionActivationHswish:
                        _rParams[i].data[0] = params[i][0];
                        _rParams[i].data[1] = params[i][1];
                        break;
                    case SimdConvolutionActivationMish:
                        _rParams[i].data[0] = params[i][0];
                        break;
                    case SimdConvolutionActivationHardSigmoid:
                        _rParams[i].data[0] = params[i][0];
                        _rParams[i].data[1] = params[i][1];
                        break;                    
                    case SimdConvolutionActivationSwish:
                        _rParams[i].data[0] = params[i][0];
                        break;
                    default:
                        assert(0);
                    }
                    _params[i] = _rParams[i].data;
                }
                else
                    _params[i] = params[i];
            }
        }

        void SynetMergedConvolution32f::Forward(const float* src, float* buf, float* dst)
        {
            const MergConvParam32f& p = _param;
            float* buf0 = GetBuffer(buf);
            float* buf1 = buf0 + _sizeB[0];
            for (size_t b = 0; b < p.batch; ++b)
            {
                _convolution[0](src, p.conv[0], 0, 0, p.conv[0].dstH, NULL, _weight[0], _bias[0], _params[0], buf0, 1);
                _convolution[1](buf0, p.conv[1], 0, 0, p.conv[1].dstH, NULL, _weight[1], _bias[1], _params[1], (p.count == 3 ? buf1 : dst), 1);
                if (p.count > 2)
                {
                    if (p.add)
                        memcpy(dst, src, sizeof(float) * _sizeS);
                    _convolution[2](buf1, p.conv[2], 0, 0, p.conv[2].dstH, NULL, _weight[2], _bias[2], _params[2], dst, 1);
                }
                src += _sizeS;
                dst += _sizeD;
            }
        }

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer* SynetMergedConvolution32f::Perf(const char* func)
        {
            if (_perf == NULL)
                _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
            return _perf;
        }
#endif

        float* SynetMergedConvolution32f::GetBuffer(float* buffer)
        {
            if (buffer)
                return buffer;
            else
            {
                _buffer.Resize(ExternalBufferSize());
                return _buffer.data;
            }
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution32fCdc::SynetMergedConvolution32fCdc(const MergConvParam32f & p)
            : SynetMergedConvolution32f(p)
        {
        }

        void SynetMergedConvolution32fCdc::SetSize(size_t L1, size_t L2, size_t L3, size_t F)
        {
            const MergConvParam32f & p = _param;
            _miC = F;
            size_t size = 0;
            for (size_t i = 0; i < 3; ++i)
                size += p.conv[i].kernelY*p.conv[i].kernelX *p.conv[i].srcC * p.conv[i].dstC / p.conv[i].group;
            size_t count = size * sizeof(float) / (L3/2) + 1;
            _maC = AlignHiAny(p.conv[0].dstC / count, 2 * _miC);
            for (size_t yStep = p.conv[1].dstH; yStep >= 1; yStep--)
            {
                _yStep[1] = Simd::Max<size_t>(1, yStep);
                for (_bufH[1] = 1; _bufH[1] < _yStep[1]; _bufH[1] *= 2);
                _yStep[0] = _yStep[1] * p.conv[1].strideY;
                for (_bufH[0] = 1; _bufH[0] < (_yStep[1] - 1) * p.conv[1].strideY + p.conv[1].kernelY; _bufH[0] *= 2);
                _sizeB[0] = _bufH[0] * p.conv[0].dstW * _maC;
                _sizeB[1] = _bufH[1] * p.conv[1].dstW * _maC;
                if ((_sizeB[0] + _sizeB[1]) * sizeof(float) <= L2)
                    break;
            }
            for (size_t i = 0; i < 3; ++i)
            {
                size_t dstC = AlignHiAny(p.conv[i].dstC, i == 1 ? _miC : 2 * _miC);
                _rWeight[i].Resize(dstC*p.conv[i].kernelY*p.conv[i].kernelX*p.conv[i].srcC / p.conv[i].group);
                _rBias[i].Resize(dstC, true);
                if (p.conv[i].activation == SimdConvolutionActivationLeakyRelu || p.conv[i].activation == SimdConvolutionActivationPrelu)
                    _rParams[i].Resize(dstC, true);
                else
                    _rParams[i].Resize(2);
            }
            _dp[0] = p.conv[0].activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            _dp[1] = p.conv[1].activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            _dw[0] = p.conv[0].kernelY*p.conv[0].kernelX*p.conv[0].srcC;
            _dw[1] = p.conv[1].kernelY*p.conv[1].kernelX;
            _dw[2] = AlignHiAny(p.conv[2].dstC, 2 * _miC);
        }

        void SynetMergedConvolution32fCdc::ReorderFirstWeight(const float * src, float * dst) const
        {
            const SimdConvolutionParameters & p = _param.conv[0];
            size_t size = p.kernelY*p.kernelX*p.srcC, dstC = p.dstC, micD = _miC*2;
            for (size_t c = 0; c < dstC; c += micD)
            {
                size_t n = Simd::Min(micD, dstC - c);
                for (size_t s = 0; s < size; s++)
                {
                    size_t i = 0;
                    for (; i < n; ++i)
                        dst[i] = src[s*dstC + c + i];
                    for (; i < micD; ++i)
                        dst[i] = 0;
                    dst += micD;
                }
            }
        }

        void SynetMergedConvolution32fCdc::ReorderSecondWeight(const float * src, float * dst) const
        {
            const SimdConvolutionParameters & p = _param.conv[1];
            size_t dstC = p.dstC, size = p.kernelY*p.kernelX, micD = _miC;
            for (size_t c = 0; c < dstC; c += micD)
            {
                size_t n = Simd::Min(micD, dstC - c);
                for (size_t s = 0; s < size; s++)
                {
                    size_t i = 0;
                    for (; i < n; ++i)
                        dst[i] = src[s*dstC + c + i];
                    for (; i < micD; ++i)
                        dst[i] = 0;
                    dst += micD;
                }
            }
        }

        void SynetMergedConvolution32fCdc::ReorderThirdWeight(const float * src, float * dst) const
        {
            const SimdConvolutionParameters & p = _param.conv[2];
            size_t srcC = p.srcC, dstC = p.dstC, micD = _miC * 2;
            for (size_t m = 0; m < srcC; m += _maC)
            {
                size_t maC = Simd::Min(srcC, m + _maC) - m;
                for (size_t d = 0; d < dstC; d += micD)
                {
                    size_t n = Simd::Min(micD, dstC - d);
                    for (size_t s = 0; s < maC; s++)
                    {
                        size_t i = 0;
                        for (; i < n; ++i)
                            dst[i] = src[s*dstC + d + i];
                        for (; i < micD; ++i)
                            dst[i] = 0;
                        dst += micD;
                    }
                }
                src += dstC*maC;
            }
        }

        void SynetMergedConvolution32fCdc::Forward(const float * src, float * buf, float * dst)
        {
            if (_rWeight[0].data == NULL)
            {
                SynetMergedConvolution32f::Forward(src, buf, dst);
                return;
            }
            const MergConvParam32f & p = _param;
            float * buf0 = GetBuffer(buf);
            float * buf1 = buf0 + _sizeB[0];
            for (size_t b = 0; b < p.batch; ++b)
            {
                for (size_t c = 0, C = p.conv[1].dstC; c < C; c += _maC)
                {
                    size_t maC = Simd::Min(C, c + _maC) - c;
                    for (size_t yBeg1 = 0, yBeg0 = 0; yBeg1 < p.conv[1].dstH;)
                    {
                        size_t yEnd1 = Simd::Min(yBeg1 + _yStep[1], p.conv[1].dstH);
                        size_t yEnd0 = Simd::Min(Simd::Max(yBeg0 + _yStep[0], (_yStep[1] - 1)*p.conv[1].strideY + p.conv[1].kernelY - p.conv[1].padY), p.conv[0].dstH);
                        _convolution[0](src, p.conv[0], maC, yBeg0, yEnd0, _bufH, _weight[0] + c * _dw[0], _bias[0] + c, _params[0] + c * _dp[0], buf0, 1);
                        _convolution[1](buf0, p.conv[1], maC, yBeg1, yEnd1, _bufH, _weight[1] + c * _dw[1], _bias[1] + c, _params[1] + c * _dp[1], buf1, 1);
                        if (p.add && c == 0)
                        {
                            size_t offset = yBeg1 * p.conv[2].dstW * p.conv[2].dstC, size = (yEnd1 - yBeg1)*p.conv[2].dstW * p.conv[2].dstC;
                            memcpy(dst + offset, src + offset, sizeof(float)*size);
                        }
                        if(c + maC == C)
                            _convolution[2](buf1, p.conv[2], maC, yBeg1, yEnd1, _bufH, _weight[2] + c * _dw[2], _bias[2], _params[2], dst, (maC != C || p.add) ? 0 : 1);
                        else
                            _convolution[3](buf1, p.conv[2], maC, yBeg1, yEnd1, _bufH, _weight[2] + c * _dw[2], _bias[2], _params[2], dst, (c != 0 || p.add) ? 0 : 1);
                        yBeg1 = yEnd1;
                        yBeg0 = yEnd0;
                    }
                }
                src += _sizeS;
                dst += _sizeD;
            }
        }

        bool SynetMergedConvolution32fCdc::Preferable(const MergConvParam32f& p)
        {
            return p.count == 3;
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution32fCd::SynetMergedConvolution32fCd(const MergConvParam32f& p)
            : SynetMergedConvolution32f(p)
        {
        }

        void SynetMergedConvolution32fCd::SetSize(size_t L1, size_t L2, size_t L3, size_t F)
        {
            const MergConvParam32f& p = _param;
            _miC = F;
            size_t size = 0;
            for (size_t i = 0; i < 2; ++i)
                size += p.conv[i].kernelY * p.conv[i].kernelX * p.conv[i].srcC * p.conv[i].dstC / p.conv[i].group;
            size_t count = size * sizeof(float) / (L3 / 2) + 1;
            _maC = AlignHiAny(p.conv[0].dstC / count, 2 * _miC);
            for (size_t yStep = p.conv[1].dstH; yStep >= 1; yStep--)
            {
                _yStep[1] = Simd::Max<size_t>(1, yStep);
                _yStep[0] = _yStep[1] * p.conv[1].strideY;
                for (_bufH[0] = 1; _bufH[0] < (_yStep[1] - 1) * p.conv[1].strideY + p.conv[1].kernelY; _bufH[0] *= 2);
                _sizeB[0] = _bufH[0] * p.conv[0].dstW * _maC;
                if (_sizeB[0] * sizeof(float) <= L2)
                    break;
            }
            _sizeB[1] = 0;
            for (size_t i = 0; i < 2; ++i)
            {
                size_t dstC = AlignHiAny(p.conv[i].dstC, i == 1 ? _miC : 2 * _miC);
                _rWeight[i].Resize(dstC * p.conv[i].kernelY * p.conv[i].kernelX * p.conv[i].srcC / p.conv[i].group);
                _rBias[i].Resize(dstC, true);
                if (p.conv[i].activation == SimdConvolutionActivationLeakyRelu || p.conv[i].activation == SimdConvolutionActivationPrelu)
                    _rParams[i].Resize(dstC, true);
                else
                    _rParams[i].Resize(2);
            }
            _dp[0] = p.conv[0].activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            _dp[1] = p.conv[1].activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            _dw[0] = p.conv[0].kernelY * p.conv[0].kernelX * p.conv[0].srcC;
            _dw[1] = p.conv[1].kernelY * p.conv[1].kernelX;
        }

        void SynetMergedConvolution32fCd::ReorderFirstWeight(const float* src, float* dst) const
        {
            const SimdConvolutionParameters& p = _param.conv[0];
            size_t size = p.kernelY * p.kernelX * p.srcC, dstC = p.dstC, micD = _miC * 2;
            for (size_t c = 0; c < dstC; c += micD)
            {
                size_t n = Simd::Min(micD, dstC - c);
                for (size_t s = 0; s < size; s++)
                {
                    size_t i = 0;
                    for (; i < n; ++i)
                        dst[i] = src[s * dstC + c + i];
                    for (; i < micD; ++i)
                        dst[i] = 0;
                    dst += micD;
                }
            }
        }

        void SynetMergedConvolution32fCd::ReorderSecondWeight(const float* src, float* dst) const
        {
            const SimdConvolutionParameters& p = _param.conv[1];
            size_t dstC = p.dstC, size = p.kernelY * p.kernelX, micD = _miC;
            for (size_t c = 0; c < dstC; c += micD)
            {
                size_t n = Simd::Min(micD, dstC - c);
                for (size_t s = 0; s < size; s++)
                {
                    size_t i = 0;
                    for (; i < n; ++i)
                        dst[i] = src[s * dstC + c + i];
                    for (; i < micD; ++i)
                        dst[i] = 0;
                    dst += micD;
                }
            }
        }

        void SynetMergedConvolution32fCd::Forward(const float* src, float* buf, float* dst)
        {
            if (_rWeight[0].data == NULL)
            {
                SynetMergedConvolution32f::Forward(src, buf, dst);
                return;
            }
            const MergConvParam32f& p = _param;
            float* buf0 = GetBuffer(buf);
            for (size_t b = 0; b < p.batch; ++b)
            {
                for (size_t c = 0, C = p.conv[1].dstC; c < C; c += _maC)
                {
                    size_t maC = Simd::Min(C, c + _maC) - c;
                    for (size_t yBeg1 = 0, yBeg0 = 0; yBeg1 < p.conv[1].dstH;)
                    {
                        size_t yEnd1 = Simd::Min(yBeg1 + _yStep[1], p.conv[1].dstH);
                        size_t yEnd0 = Simd::Min(Simd::Max(yBeg0 + _yStep[0], (_yStep[1] - 1) * p.conv[1].strideY + p.conv[1].kernelY - p.conv[1].padY), p.conv[0].dstH);
                        _convolution[0](src, p.conv[0], maC, yBeg0, yEnd0, _bufH, _weight[0] + c * _dw[0], _bias[0] + c, _params[0] + c * _dp[0], buf0, 1);
                        _convolution[1](buf0, p.conv[1], maC, yBeg1, yEnd1, _bufH, _weight[1] + c * _dw[1], _bias[1] + c, _params[1] + c * _dp[1], dst + c, 1);
                        yBeg1 = yEnd1;
                        yBeg0 = yEnd0;
                    }
                }
                src += _sizeS;
                dst += _sizeD;
            }
        }

        bool SynetMergedConvolution32fCd::Preferable(const MergConvParam32f& p)
        {
            return p.count == 2 && p.conv[0].group == 1;
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution32fDc::SynetMergedConvolution32fDc(const MergConvParam32f& p)
            : SynetMergedConvolution32f(p)
        {
        }

        void SynetMergedConvolution32fDc::SetSize(size_t L1, size_t L2, size_t L3, size_t F)
        {
            const MergConvParam32f& p = _param;
            _miC = F;
            size_t size = 0;
            for (size_t i = 0; i < 2; ++i)
                size += p.conv[i].kernelY * p.conv[i].kernelX * p.conv[i].srcC * p.conv[i].dstC / p.conv[i].group;
            size_t count = size * sizeof(float) / (L3 / 2) + 1;
            _maC = AlignHiAny(p.conv[0].dstC / count, 2 * _miC);
            for (size_t yStep = p.conv[0].dstH; yStep >= 1; yStep--)
            {
                _yStep[0] = Simd::Max<size_t>(1, yStep);
                for (_bufH[0] = 1; _bufH[0] < _yStep[0]; _bufH[0] *= 2);
                _sizeB[0] = _bufH[0] * p.conv[0].dstW * _maC;
                if (_sizeB[0]* sizeof(float) <= L2)
                    break;
            }
            _bufH[1] = _bufH[0];
            _sizeB[1] = 0;
            for (size_t i = 0; i < 2; ++i)
            {
                size_t dstC = AlignHiAny(p.conv[i].dstC, i == 0 ? _miC : 2 * _miC);
                _rWeight[i].Resize(dstC * p.conv[i].kernelY * p.conv[i].kernelX * p.conv[i].srcC / p.conv[i].group);
                _rBias[i].Resize(dstC, true);
                if (p.conv[i].activation == SimdConvolutionActivationLeakyRelu || p.conv[i].activation == SimdConvolutionActivationPrelu)
                    _rParams[i].Resize(dstC, true);
                else
                    _rParams[i].Resize(2);
            }
            _dp[0] = p.conv[0].activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            _dw[0] = p.conv[0].kernelY * p.conv[0].kernelX;
            _dw[1] = AlignHiAny(p.conv[1].dstC, 2 * _miC);
        }

        void SynetMergedConvolution32fDc::ReorderFirstWeight(const float* src, float* dst) const
        {
            const SimdConvolutionParameters& p = _param.conv[0];
            size_t dstC = p.dstC, size = p.kernelY * p.kernelX, micD = _miC;
            for (size_t c = 0; c < dstC; c += micD)
            {
                size_t n = Simd::Min(micD, dstC - c);
                for (size_t s = 0; s < size; s++)
                {
                    size_t i = 0;
                    for (; i < n; ++i)
                        dst[i] = src[s * dstC + c + i];
                    for (; i < micD; ++i)
                        dst[i] = 0;
                    dst += micD;
                }
            }
        }

        void SynetMergedConvolution32fDc::ReorderSecondWeight(const float* src, float* dst) const
        {
            const SimdConvolutionParameters& p = _param.conv[1];
            size_t srcC = p.srcC, dstC = p.dstC, micD = _miC * 2;
            for (size_t m = 0; m < srcC; m += _maC)
            {
                size_t maC = Simd::Min(srcC, m + _maC) - m;
                for (size_t d = 0; d < dstC; d += micD)
                {
                    size_t n = Simd::Min(micD, dstC - d);
                    for (size_t s = 0; s < maC; s++)
                    {
                        size_t i = 0;
                        for (; i < n; ++i)
                            dst[i] = src[s * dstC + d + i];
                        for (; i < micD; ++i)
                            dst[i] = 0;
                        dst += micD;
                    }
                }
                src += dstC * maC;
            }
        }

        void SynetMergedConvolution32fDc::Forward(const float* src, float* buf, float* dst)
        {
            if (_rWeight[0].data == NULL)
            {
                SynetMergedConvolution32f::Forward(src, buf, dst);
                return;
            }
            const MergConvParam32f& p = _param;
            float* buf0 = GetBuffer(buf);
            for (size_t b = 0; b < p.batch; ++b)
            {
                for (size_t c = 0, C = p.conv[0].dstC; c < C; c += _maC)
                {
                    size_t maC = Simd::Min(C, c + _maC) - c;
                    for (size_t yBeg0 = 0; yBeg0 < p.conv[0].dstH;)
                    {
                        size_t yEnd0 = Simd::Min(yBeg0 + _yStep[0], p.conv[0].dstH);
                        _convolution[0](src + c, p.conv[0], maC, yBeg0, yEnd0, _bufH, _weight[0] + c * _dw[0], _bias[0] + c, _params[0] + c * _dp[0], buf0, 1);
                        if (c + maC == C)
                            _convolution[1](buf0, p.conv[1], maC, yBeg0, yEnd0, _bufH, _weight[1] + c * _dw[1], _bias[1], _params[1], dst, maC == C ? 1 : 0);
                        else
                            _convolution[2](buf0, p.conv[1], maC, yBeg0, yEnd0, _bufH, _weight[1] + c * _dw[1], _bias[1], _params[1], dst, c == 0 ? 1 : 0);
                        yBeg0 = yEnd0;
                    }
                }
                src += _sizeS;
                dst += _sizeD;
            }
        }

        bool SynetMergedConvolution32fDc::Preferable(const MergConvParam32f& p)
        {
            return p.count == 2 && p.conv[1].group == 1;
        }

        //---------------------------------------------------------------------

        void * SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add)
        {
            MergConvParam32f param(batch, convs, count, add);
            if (!param.Valid())
                return NULL;
            return new Base::SynetMergedConvolution32f(param);
        }
    }
#endif
}
