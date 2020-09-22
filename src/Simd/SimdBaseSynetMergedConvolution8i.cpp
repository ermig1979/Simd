/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdSynetMergedConvolution8i.h"
#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void ExtendSize(size_t value, size_t& size)
        {
            size = Simd::Max(size, AlignHi(value, SIMD_ALIGN));
        }

        SynetMergedConvolution8i::SynetMergedConvolution8i(const MergConvParam8i& p)
           :  _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC)
           , _perf(NULL)
#endif        
        {
            const SimdConvolutionParameters& beg = p.conv[0];
            const SimdConvolutionParameters& end = p.conv[p.count - 1];
            _sizeS = beg.srcH * beg.srcW * beg.srcC;
            _sizeD = end.dstH * end.dstW * end.dstC;
            _sizeI[0] = p.conv[1].srcH * p.conv[1].srcW * p.conv[1].srcC;
            _sizeI[1] = p.count == 3 ? p.conv[1].dstH * p.conv[1].dstW * p.conv[1].dstC : 0;
            _s8u = beg.srcT == SimdTensorData8u;
            _d8u = end.dstT == SimdTensorData8u;
            _dw0 = beg.group != 1;
            _1x1 = beg.kernelY == 1 && beg.strideY == 1;
            switch (p.conv[_dw0 ? 0 : 1].activation)
            {
            case SimdConvolutionActivationIdentity: _depthwise = DepthwiseConvolution<SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: _depthwise = DepthwiseConvolution<SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: _depthwise = DepthwiseConvolution<SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: _depthwise = DepthwiseConvolution<SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: _depthwise = DepthwiseConvolution<SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: _depthwise = DepthwiseConvolution<SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: _depthwise = DepthwiseConvolution<SimdConvolutionActivationHswish>; break;
            default: assert(0);
            }
            for (size_t i = 0; i < 5; ++i)
                _sizeB[i] = 0;
            if (_dw0)
            {
                if (_s8u)
                    ExtendSize(_sizeS, _sizeB[0]);
                ExtendSize(_sizeI[0], _sizeB[1]);
                ExtendSize(_sizeI[0], _sizeB[2]);
                ExtendSize(_sizeD, _sizeB[4]);
            }
            else
            {
                if (!_s8u)
                    ExtendSize(_sizeS, _sizeB[2]);
                if (!_1x1)
                    ExtendSize(beg.kernelY * beg.kernelX * beg.srcC * beg.dstH * beg.dstW, _sizeB[3]);
                ExtendSize(_sizeI[0], _sizeB[4]);
                ExtendSize(_sizeI[0], _sizeB[0]);
                if (p.count == 3)
                {
                    ExtendSize(_sizeI[1], _sizeB[1]);
                    ExtendSize(_sizeI[1], _sizeB[3]);
                    ExtendSize(_sizeD, _sizeB[4]);
                }
            }
            if (_d8u)
                ExtendSize(_sizeD, _sizeB[1]);
        }

        size_t SynetMergedConvolution8i::ExternalBufferSize() const
        {
            return (_sizeB[0] + _sizeB[1] + _sizeB[4]) * 4 + _sizeB[2] + _sizeB[3] + SIMD_ALIGN;
        }

        size_t SynetMergedConvolution8i::InternalBufferSize() const
        {
            size_t size = _buffer.RawSize() + _weight32f.RawSize();
            for (size_t i = 0; i < 3; ++i)
            {
                if (i < 2) 
                    size += _norm[i].RawSize() + _weight8i[i].RawSize();
                size += _bias[i].RawSize() + _params[i].RawSize() + _cvt[i].Size();
            }
            return size;
        }

        void SynetMergedConvolution8i::SetParams(const float* const* weight, SimdBool* internal, const float* const* bias, const float* const* params, const float* const* stats)
        {
            const MergConvParam8i& p = _param;
            const SimdConvolutionParameters& beg = p.conv[0];
            const SimdConvolutionParameters& end = p.conv[p.count - 1];
            _cvt[0].Init(stats[0], stats[1], beg.srcC, p.compatibility);
            _cvt[1].Init(stats[2], stats[3], end.srcC, p.compatibility);
            _cvt[2].Init(stats[4], stats[5], end.dstC, p.compatibility);
            for (size_t i = 0, q = 0; i < p.count; ++i)
            {
                const SimdConvolutionParameters& c = p.conv[i];
                if (p.conv[i].group == 1)
                {
                    _weight8i[q].Resize(c.dstC * c.kernelY * c.kernelX * c.srcC);
                    _norm[q].Resize(c.dstC);
                    _bias[i].Resize(c.dstC);
                    Quantize(weight[i], bias[i], i, q++);
                }
                else
                {
                    _weight32f.Assign(weight[i], c.dstC * c.kernelY * c.kernelX);
                    _bias[i].Assign(bias[i], c.dstC);
                }
                if (c.activation == SimdConvolutionActivationLeakyRelu || c.activation == SimdConvolutionActivationPrelu)
                    _params[i].Resize(c.dstC);
                else
                    _params[i].Resize(2);
                switch (c.activation)
                {
                case SimdConvolutionActivationIdentity:
                    _params[i][0] = -FLT_MAX;
                    _params[i][1] = FLT_MAX;
                    break;
                case SimdConvolutionActivationRelu:
                    _params[i][0] = 0;
                    _params[i][1] = FLT_MAX;
                    break;
                case SimdConvolutionActivationLeakyRelu:
                    for (size_t d = 0; d < c.dstC; ++d)
                        _params[i][d] = params[i][0];
                    break;
                case SimdConvolutionActivationRestrictRange:
                    _params[i][0] = params[i][0];
                    _params[i][1] = params[i][1];
                    break;
                case SimdConvolutionActivationPrelu:
                    for (size_t d = 0; d < c.dstC; ++d)
                        _params[i][d] = params[i][d];
                    break;
                case SimdConvolutionActivationElu:
                    _params[i][0] = params[i][0];
                    break;
                case SimdConvolutionActivationHswish:
                    _params[i][0] = params[i][0];
                    _params[i][1] = params[i][1];
                    break;
                default:
                    assert(0);
                }
                if (internal)
                    internal[i] = SimdTrue;
            }
        }

        void SynetMergedConvolution8i::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const MergConvParam8i& p = _param;
            float* buf0 = Allocate<float>(buf, _sizeB[0]);
            float* buf1 = Allocate<float>(buf, _sizeB[1]);
            uint8_t* buf2 = Allocate<uint8_t>(buf, _sizeB[2]);
            uint8_t* buf3 = Allocate<uint8_t>(buf, _sizeB[3]);
            int32_t* buf4 = Allocate<int32_t>(buf, _sizeB[4]);

            float* src32f = _s8u ? (_dw0 ? buf0 : NULL) : (float*)src;
            uint8_t* src8u = _s8u ? (uint8_t*)src : (_dw0 ? NULL : buf2);
            float* dst32f = _d8u ? buf1 : (float*)dst;
            uint8_t* dst8u = _d8u ? dst : NULL;

            for (size_t b = 0; b < p.batch; ++b)
            {
                if (_dw0)
                {
                    if (_s8u)
                    {
                        Convert<uint8_t, float, float>(src8u, 1, p.conv[0].srcC, p.conv[0].srcH, p.conv[0].srcW, p.conv[0].srcF, _cvt[0].iScale.data, _cvt[0].iShift.data, 0, 0, src32f);
                        src8u += _sizeS;
                    }
                    _depthwise(src32f, p.conv[0], 0, 0, p.conv[0].dstH, NULL, _weight32f.data, _bias[0].data, _params[0].data, buf1);
                    if (!_s8u)
                        src32f += _sizeS;
                    Convert<float, uint8_t, float>(buf1, 1, p.conv[1].srcC, p.conv[1].srcH, p.conv[1].srcW, p.conv[1].srcF, _cvt[1].scale.data, _cvt[1].shift.data, _cvt[1].uMin, _cvt[1].uMax, buf2);
                    DirectConvolution8i(buf2, 1, 0, NULL, buf4, dst32f);
                }
                else
                {
                    if (!_s8u)
                    {
                        Convert<float, uint8_t, float>(src32f, 1, p.conv[0].srcC, p.conv[0].srcH, p.conv[0].srcW, p.conv[0].srcF, _cvt[0].scale.data, _cvt[0].shift.data, _cvt[0].uMin, _cvt[0].uMax, src8u);
                        src32f += _sizeS;
                    }                    
                    DirectConvolution8i(src8u, 0, 0, buf3, buf4, buf0);
                    if (_s8u)
                        src8u += _sizeS;
                    _depthwise(buf0, p.conv[1], 0, 0, p.conv[1].dstH, NULL, _weight32f.data, _bias[1].data, _params[1].data, p.count == 3 ? buf1 : dst32f);
                    if (p.count == 3)
                    {
                        Convert<float, uint8_t, float>(buf1, 1, p.conv[2].srcC, p.conv[2].srcH, p.conv[2].srcW, p.conv[2].srcF, _cvt[1].scale.data, _cvt[1].shift.data, _cvt[1].uMin, _cvt[1].uMax, buf2);
                        DirectConvolution8i(buf2, 2, 1, NULL, buf4, dst32f);
                    }
                }
                if (_d8u)
                {
                    const SimdConvolutionParameters& end = p.conv[p.count - 1];
                    Convert<float, uint8_t, float>(dst32f, 1, end.dstC, end.dstH, end.dstW, end.dstF, _cvt[2].scale.data, _cvt[2].shift.data, _cvt[2].uMin, _cvt[2].uMax, dst8u);
                    dst8u += _sizeD;
                }
                else
                    dst32f += _sizeD;
            }
        }

#if defined(SIMD_PERFORMANCE_STATISTIC)
        Base::PerformanceMeasurer* SynetMergedConvolution8i::Perf(const String& func)
        {
            if (_perf == NULL)
                _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
            return _perf;
        }
#endif

        uint8_t* SynetMergedConvolution8i::GetBuffer(uint8_t* buffer)
        {
            if (buffer)
                return buffer;
            else
            {
                _buffer.Resize(ExternalBufferSize());
                return _buffer.data;
            }
        }

        void SynetMergedConvolution8i::Quantize(const float* weight, const float* bias, size_t i, size_t q)
        {
            const SimdConvolutionParameters & conv = _param.conv[i];
            const CvtParam& cvt = _cvt[i ? 1 : 0];
            size_t D = conv.dstC, C = conv.srcC, K = conv.kernelY * conv.kernelX;
            Array32f  normW(C * K);
            bool avoidOverflow16i = cvt.neg && Base::Overflow(_param.compatibility);
            for (size_t d = 0; d < D; ++d)
            {
                float normB = 0, minW = FLT_MAX, maxW = -FLT_MAX, scale = 1.0f;
                for (size_t k = 0, kc = 0; k < K; ++k)
                {
                    for (size_t c = 0; c < C; ++c, ++kc)
                    {
                        normW[kc] = weight[kc * D + d] / cvt.scale[c];
                        minW = Simd::Min(minW, normW[kc]);
                        maxW = Simd::Max(maxW, normW[kc]);
                    }
                }
                scale = cvt.iMax / Simd::Max(Simd::Abs(maxW), Simd::Abs(minW));
                for (size_t k = 0, kc = 0; k < K; ++k)
                {
                    for (size_t c = 0; c < C; ++c, ++kc)
                    {
                        int w = Base::SynetConvert32fTo8i(normW[kc], scale, 0.0f, cvt.iMin, cvt.iMax);
                        if (avoidOverflow16i)
                        {
                            if (w & 1)
                                w = Round(w * 0.25f) * 4;
                            w = w / 2;
                        }
                        _weight8i[q][kc * D + d] = w;
                        normB -= w * cvt.shift[c];
                    }
                }
                _norm[q][d] = (avoidOverflow16i ? 2.0f : 1.0f) / scale;
                _bias[i][d] = (bias ? bias[d] : 0.0f) + normB / scale;
            }
        }

        void SynetMergedConvolution8i::DirectConvolution8i(const uint8_t* src, size_t i, size_t q, uint8_t* buf, int32_t* sum, float* dst)
        {
            const SimdConvolutionParameters& conv = _param.conv[i];
            const float* params = _params[i].data;
            const uint8_t* tmp = src;
            if (!_1x1 && i == 0)
            {
                Base::ImgToRow(tmp, conv, _cvt[0].zero.data, buf);
                tmp = buf;
            }
            size_t K = conv.srcC * conv.kernelY * conv.kernelX, N = conv.dstH * conv.dstW, M = conv.dstC;
            GemmNhwc(N, M, conv.kernelY * conv.kernelX, conv.srcC, tmp, K, _weight8i[q].data, M, sum, M, true);
            Convert<int32_t, float, float>(sum, 1, conv.dstC, conv.dstH, conv.dstW, conv.dstF, _norm[q].data, _bias[i].data, 0, 0, dst);
            size_t sizeD = conv.dstC * conv.dstH * conv.dstW;
            switch (conv.activation)
            {
            case SimdConvolutionActivationIdentity:
                break;
            case SimdConvolutionActivationRelu:
            {
                float slope = 0;
                SynetRelu32f(dst, sizeD, &slope, dst);
                break;
            }
            case SimdConvolutionActivationLeakyRelu:
                SynetRelu32f(dst, sizeD, params, dst);
                break;
            case SimdConvolutionActivationRestrictRange:
                SynetRestrictRange32f(dst, sizeD, params + 0, params + 1, dst);
                break;
            case SimdConvolutionActivationPrelu:
                SynetPreluLayerForward(dst, params, M, N, dst, conv.dstF);
                break;
            case SimdConvolutionActivationElu:
                SynetElu32f(dst, sizeD, params, dst);
                break;
            case SimdConvolutionActivationHswish:
                SynetHswish32f(dst, sizeD, params + 0, params + 1, dst);
                break;
            default:
                assert(0);
            }
        }

        //---------------------------------------------------------------------

        void * SynetMergedConvolution8iInit(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdSynetCompatibilityType compatibility)
        {
            MergConvParam8i param(batch, convs, count, compatibility);
            if (!param.Valid())
                return NULL;
            return new Base::SynetMergedConvolution8i(param);
        }
    }
}
