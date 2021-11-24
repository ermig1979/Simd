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
#include "Simd/SimdSynetMergedConvolution8i.h"
#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SIMD_INLINE void ExtendSize(size_t value, size_t& size)
        {
            size = Simd::Max(size, AlignHi(value, SIMD_ALIGN));
        }

        void Convert8uTo32f(const uint8_t* src, size_t maC, size_t yBeg, size_t yEnd, size_t width, size_t channels,
            const float* scale, const float* shift, float* dst, size_t bufH, SimdSynetCompatibilityType compatibility)
        {
            Base::SynetConvert8uTo32f(src, 1, channels, yEnd, width, SimdTensorFormatNhwc, scale, shift, dst, compatibility);
        }

        void Convert32fTo8u(const float* src, size_t yBeg, size_t yEnd, size_t width, size_t channels,
            const float* scale, const float* shift, uint8_t* dst, size_t bufH, SimdSynetCompatibilityType compatibility)
        {
            Base::SynetConvert32fTo8u(src, 1, channels, yEnd, width, SimdTensorFormatNhwc, scale, shift, dst, compatibility);
        }

        template<SimdConvolutionActivationType type> void DepthwiseConvolution(const float* src, const ConvParam8i& p, const SynetMergedConvolution8i::AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            DepthwiseConvolution<type>(src, p, 0, 0, p.dstH, NULL, weight, bias, params, (float*)dst, 1);
        }

        SynetMergedConvolution8i::SynetMergedConvolution8i(const MergConvParam8i& p)
           :  _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
           , _perf(NULL)
#endif        
        {
            _alg.miC = 0;
            const ConvParam8i& beg = p.conv[0];
            const ConvParam8i& end = p.conv[p.count - 1];
            _sizeS = beg.srcH * beg.srcW * beg.srcC;
            _sizeD = end.dstH * end.dstW * end.dstC;
            _sizeI[0] = p.conv[1].srcH * p.conv[1].srcW * p.conv[1].srcC;
            _sizeI[1] = p.count == 3 ? p.conv[1].dstH * p.conv[1].dstW * p.conv[1].dstC : 0;
            _s8u = beg.srcT == SimdTensorData8u;
            _d8u = end.dstT == SimdTensorData8u;
            _dw0 = beg.group != 1;
            _1x1 = beg.kernelY == 1 && beg.strideY == 1;

            _cvt8uTo32f = Convert8uTo32f;
            _cvt32fTo8u = Convert32fTo8u;
            switch (p.conv[_dw0 ? 0 : 1].activation)
            {
            case SimdConvolutionActivationIdentity: _depthwise = DepthwiseConvolution<SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: _depthwise = DepthwiseConvolution<SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: _depthwise = DepthwiseConvolution<SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: _depthwise = DepthwiseConvolution<SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: _depthwise = DepthwiseConvolution<SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: _depthwise = DepthwiseConvolution<SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: _depthwise = DepthwiseConvolution<SimdConvolutionActivationHswish>; break;
            case SimdConvolutionActivationMish: _depthwise = DepthwiseConvolution<SimdConvolutionActivationMish>; break;
            case SimdConvolutionActivationHardSigmoid: _depthwise = DepthwiseConvolution<SimdConvolutionActivationHardSigmoid>; break;
            case SimdConvolutionActivationSwish: _depthwise = DepthwiseConvolution<SimdConvolutionActivationSwish>; break;
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
            const ConvParam8i& beg = p.conv[0];
            const ConvParam8i& end = p.conv[p.count - 1];
            _cvt[0].Init(stats[0], stats[1], beg.srcC, beg.compatibility);
            _cvt[1].Init(stats[2], stats[3], end.srcC, beg.compatibility);
            _cvt[2].Init(stats[4], stats[5], end.dstC, beg.compatibility);
            for (size_t i = 0, q = 0; i < p.count; ++i)
            {
                const ConvParam8i& c = p.conv[i];
                if (p.conv[i].group == 1)
                {
                    _weight8i[q].Resize(c.dstC * c.kernelY * c.kernelX * c.srcC);
                    _norm[q].Resize(c.dstC);
                    _bias[i].Resize(c.dstC);
                    Quantize(weight[i], bias[i], i, q);
                    if (i)
                        ReorderOutputWeight(c, _weight8i[q]);
                    else
                        ReorderInputWeight(c, _weight8i[q]);
                    q++;
                }
                else
                {
                    _weight32f.Assign(weight[i], c.dstC * c.kernelY * c.kernelX);
                    _bias[i].Assign(bias[i], c.dstC);
                    ReorderDepthwiseWeight(c, _weight32f);
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
                case SimdConvolutionActivationMish:
                    _params[i][0] = params[i][0];
                    break;
                case SimdConvolutionActivationHardSigmoid:
                    _params[i][0] = params[i][0];
                    _params[i][1] = params[i][1];
                    break;
                case SimdConvolutionActivationSwish:
                    _params[i][0] = params[i][0];
                    break;
                default:
                    assert(0);
                }
                if (internal)
                    internal[i] = SimdTrue;
            }
            _alg.zero = Set4(_cvt[0].zero[0]);
            _alg.upper = Set4(_cvt[0].uMax);
        }

        void SynetMergedConvolution8i::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const MergConvParam8i& p = _param;
            const ConvParam8i& c0 = p.conv[0];
            const ConvParam8i& c1 = p.conv[1];
            const ConvParam8i& c2 = p.conv[2];

            buf = GetBuffer(buf);
            float* buf0 = Allocate<float>(buf, _sizeB[0]);
            float* buf1 = Allocate<float>(buf, _sizeB[1]);
            uint8_t* buf2 = Allocate<uint8_t>(buf, _sizeB[2]);
            uint8_t* buf3 = Allocate<uint8_t>(buf, _sizeB[3]);
            int32_t* buf4 = Allocate<int32_t>(buf, _sizeB[4]);

            float* src32f = _s8u ? (_dw0 ? buf0 : NULL) : (float*)src;
            uint8_t* src8u = _s8u ? (uint8_t*)src : (_dw0 ? NULL : buf2);
            float* dst32f = _d8u ? buf1 : (float*)dst;
            uint8_t* dst8u = _d8u ? dst : NULL;

            for (size_t b = 0; b < c0.batch; ++b)
            {
                if (_dw0)
                {
                    if (_s8u)
                    {
                        _cvt8uTo32f(src8u, c0.srcC, 0, c0.srcH, c0.srcW, c0.srcC, _cvt[0].iScale.data, _cvt[0].iShift.data, src32f, 0, c0.compatibility);
                        src8u += _sizeS;
                    }
                    _depthwise(src32f, c0, _alg, 0, 0, c0.dstH, _weight32f.data, _bias[0].data, _params[0].data, NULL, NULL, (uint8_t*)buf1);
                    if (!_s8u)
                        src32f += _sizeS;
                    _cvt32fTo8u(buf1, 0, c1.srcH, c1.srcW, c1.srcC, _cvt[1].scale.data, _cvt[1].shift.data, buf2, 0, c0.compatibility);
                    DirectConvolution8i(buf2, 1, 0, NULL, buf4, dst32f);
                }
                else
                {
                    if (!_s8u)
                    {
                        _cvt32fTo8u(src32f, 0, c0.srcH, c0.srcW, c0.srcC, _cvt[0].scale.data, _cvt[0].shift.data, src8u, 0, c0.compatibility);
                        src32f += _sizeS;
                    }                    
                    DirectConvolution8i(src8u, 0, 0, buf3, buf4, buf0);
                    if (_s8u)
                        src8u += _sizeS;
                    _depthwise(buf0, c1, _alg, 0, 0, c1.dstH, _weight32f.data, _bias[1].data, _params[1].data, NULL, NULL, (uint8_t*)(p.count == 3 ? buf1 : dst32f));
                    if (p.count == 3)
                    {
                        _cvt32fTo8u(buf1, 0, c2.srcH, c2.srcW, c2.srcC, _cvt[1].scale.data, _cvt[1].shift.data, buf2, 0, c0.compatibility);
                        DirectConvolution8i(buf2, 2, 1, NULL, buf4, dst32f);
                    }
                }
                if (_d8u)
                {
                    const ConvParam8i& e = p.conv[p.count - 1];
                    _cvt32fTo8u(dst32f, 0, e.dstH, e.dstW, e.dstC, _cvt[2].scale.data, _cvt[2].shift.data, dst8u, 0, c0.compatibility);
                    dst8u += _sizeD;
                }
                else
                    dst32f += _sizeD;
            }
        }

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer* SynetMergedConvolution8i::Perf(const char* func)
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
            const ConvParam8i& conv = _param.conv[i];
            const CvtParam& cvt = _cvt[i ? 1 : 0];
            size_t D = conv.dstC, C = conv.srcC, K = conv.kernelY * conv.kernelX;
            Array32f  normW(C * K);
            bool avoidOverflow16i = cvt.neg && Base::Overflow(conv.compatibility);
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
                            _weight8i[q][kc * D + d] = w / 2;
                        }
                        else
                            _weight8i[q][kc * D + d] = w;
                        normB -= w * cvt.shift[c];
                    }
                }
                _norm[q][d] = (avoidOverflow16i ? 2.0f : 1.0f) / scale;
                _bias[i][d] = (bias ? bias[d] : 0.0f) + normB / scale;
            }
        }

        void SynetMergedConvolution8i::ReorderInputWeight(const ConvParam8i & p, Array8i& weight)
        {
            if (_alg.miC == 0)
                return;
            size_t F = _alg.miC * 2, C = DivHi(p.srcC, 4), D = DivHi(p.dstC, F), K = p.kernelY*p.kernelX;
            Array8i buf(K * C * D * F * 4);
            int8_t* dst = buf.data;
            for (size_t d = 0; d < D; d++)
            {
                for (size_t k = 0; k < K; ++k)
                {
                    for (size_t c = 0; c < C; ++c)
                    {
                        const int8_t* src = weight.data + (k* p.srcC + c * 4) * p.dstC + d * F;
                        for (size_t f = 0; f < F; ++f)
                        {
                            for (size_t i = 0; i < 4; ++i)
                            {
                                if (d * F + f < p.dstC && c * 4 + i < p.srcC)
                                    *(dst++) = src[i * p.dstC];
                                else
                                    *(dst++) = 0;
                            }
                            src++;
                        }
                    }
                }
            }
            weight.Swap(buf);
        }

        void SynetMergedConvolution8i::ReorderDepthwiseWeight(const ConvParam8i& p, Array32f& weight)
        {
            if (_alg.miC == 0)
                return;
            size_t D = p.dstC, K = p.kernelY * p.kernelX, F = _alg.miC;
            Array32f buf(AlignHiAny(D, F)* K);
            const float* src = weight.data;
            float* dst = buf.data;
            for (size_t d = 0; d < D; d += F)
            {
                size_t n = Simd::Min(F, D - d);
                for (size_t k = 0; k < K; k++)
                {
                    size_t i = 0;
                    for (; i < n; ++i)
                        dst[i] = src[k * D + d + i];
                    for (; i < F; ++i)
                        dst[i] = 0;
                    dst += F;
                }
            }
            weight.Swap(buf);
        }

        void SynetMergedConvolution8i::ReorderOutputWeight(const ConvParam8i& p, Array8i& weight)
        {
            if (_alg.miC == 0)
                return;
            size_t F = _alg.miC * 2, C = DivHi(p.srcC, 4), D = DivHi(p.dstC, F), M = DivHi(_alg.maC, 4);
            Array8i buf(C * D * F * 4);
            int8_t* dst = buf.data;
            for (size_t cB = 0; cB < C; cB += M)
            {
                size_t cE = Simd::Min(C, cB + M);
                for (size_t d = 0; d < D; d++)
                {
                    for (size_t c = cB; c < cE; ++c)
                    {
                        const int8_t* src = weight.data + c * 4 * p.dstC + d * F;
                        for (size_t f = 0; f < F; ++f)
                        {
                            for (size_t i = 0; i < 4; ++i)
                            {
                                if (d * F + f < p.dstC && c * 4 + i < p.srcC)
                                    *(dst++) = src[i * p.dstC];
                                else
                                    *(dst++) = 0;
                            }
                            src++;
                        }
                    }
                }
            }
            weight.Swap(buf);
        }

        void SynetMergedConvolution8i::DirectConvolution8i(const uint8_t* src, size_t i, size_t q, uint8_t* buf, int32_t* sum, float* dst)
        {
            const ConvParam8i& conv = _param.conv[i];
            const float* params = _params[i].data;
            const uint8_t* tmp = src;
            if (!_1x1 && i == 0)
            {
                Base::ImgToRow(tmp, conv, _cvt[0].zero.data, buf);
                tmp = buf;
            }
            size_t K = conv.srcC * conv.kernelY * conv.kernelX, N = conv.dstH * conv.dstW, M = conv.dstC;
            GemmNhwc(N, M, conv.kernelY * conv.kernelX, conv.srcC, tmp, K, _weight8i[q].data, M, sum, M, Overflow(conv.compatibility));
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
            case SimdConvolutionActivationMish:
                SynetMish32f(dst, sizeD, params, dst);
                break;
            case SimdConvolutionActivationHardSigmoid:
                SynetHardSigmoid32f(dst, sizeD, params + 0, params + 1, dst);
                break;
            case SimdConvolutionActivationSwish:
                SynetSwish32f(dst, sizeD, params, dst);
                break;
            default:
                assert(0);
            }
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution8iCdc::SynetMergedConvolution8iCdc(const MergConvParam8i& p)
            : SynetMergedConvolution8i(p)
        {
        }

        void SynetMergedConvolution8iCdc::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const MergConvParam8i& p = _param;
            const ConvParam8i& c0 = p.conv[0];
            const ConvParam8i& c1 = p.conv[1];
            const ConvParam8i& c2 = p.conv[2];
            const AlgParam& a = _alg;

            buf = GetBuffer(buf);
            float* buf0 = Allocate<float>(buf, _sizeB[0]);
            uint8_t* buf2 = Allocate<uint8_t>(buf, _sizeB[2]);
            uint8_t* buf3 = Allocate<uint8_t>(buf, _sizeB[3]);
            int32_t* buf4 = Allocate<int32_t>(buf, _sizeB[4]);

            for (size_t b = 0; b < c0.batch; ++b)
            {
                for (size_t c = 0, C = c1.dstC; c < C; c += a.maC)
                {
                    size_t maC = Simd::Min(C, c + a.maC) - c;
                    for (size_t yBeg2 = 0, yBeg1 = 0, yBeg0 = 0; yBeg2 < c1.dstH;)
                    {
                        size_t yEnd2 = Simd::RestrictRange(yBeg2 + a.yStep[2], a.yStart[2], c1.dstH);
                        size_t yEnd1 = Simd::RestrictRange(yBeg1 + a.yStep[1], a.yStart[1], c1.srcH);
                        size_t yEnd0 = Simd::RestrictRange(yBeg0 + a.yStep[0], a.yStart[0], c0.srcH);
                        if (!_s8u)
                            _cvt32fTo8u((float*)src, yBeg0, yEnd0, c0.srcW, c0.srcC, _cvt[0].scale.data, _cvt[0].shift.data, buf2, a.bufH[0], c0.compatibility);
                        _input(_s8u ? src : buf2, c0, a, maC, yBeg1, yEnd1, _weight8i[0].data + c * a.dw[0], _norm[0].data + c, 
                            _bias[0].data + c, _params[0].data + c * a.dp[0], buf0);
                        _depthwise(buf0, c1, a, maC, yBeg2, yEnd2, _weight32f.data + c * a.dw[1], _bias[1].data + c, 
                            _params[1].data + c * a.dp[1], _cvt[1].scale.data + c, _cvt[1].shift.data + c, buf3);
                        if (c + maC == C)
                            _output[0](buf3, c2, a, maC, yBeg2, yEnd2, _weight8i[1].data + c * a.dw[2], _norm[1].data, _bias[2].data, 
                                _params[2].data, _cvt[2].scale.data, _cvt[2].shift.data, maC == C ? NULL : buf4, dst, maC == C ? 1 : 0);
                        else
                            _output[1](buf3, c2, a, maC, yBeg2, yEnd2, _weight8i[1].data + c * a.dw[2], _norm[1].data, _bias[2].data,
                                _params[2].data, _cvt[2].scale.data, _cvt[2].shift.data, buf4, dst, c == 0 ? 1 : 0);
                        yBeg2 = yEnd2;
                        yBeg1 = yEnd1;
                        yBeg0 = yEnd0;
                    }
                }
                src += _sizeS * (_s8u ? 1 : 4);
                dst += _sizeD * (_d8u ? 1 : 4);
            }
        }

        bool SynetMergedConvolution8iCdc::Preferable(const MergConvParam8i& p)
        {
            return p.count == 3;
        }

        void SynetMergedConvolution8iCdc::SetSize(size_t F)
        {
            const size_t L1 = Base::AlgCacheL1(), L2 = Base::AlgCacheL2(), L3 = Base::AlgCacheL3();
            const MergConvParam8i& p = _param;
            const ConvParam8i& c0 = p.conv[0];
            const ConvParam8i& c1 = p.conv[1];
            const ConvParam8i& c2 = p.conv[2];
            AlgParam & a = _alg;
            a.miC = F;
            size_t size = 0;
            for (size_t i = 0; i < 3; ++i)
            {
                const ConvParam8i & c = p.conv[i];
                size += c.kernelY * c.kernelX * c.srcC * (c.group == 1 ? c.dstC : 4);
            }
            size_t count = size / (L3 / 2) + 1;
            a.maC = AlignHiAny(c0.dstC / count, 2 * a.miC);
            for (size_t yStep = c1.dstH; yStep >= 1; yStep--)
            {
                a.yStep[2] = Simd::Max<size_t>(1, yStep);
                a.yStart[2] = a.yStep[2];
                a.bufH[2] = Pow2Hi(a.yStep[2]);

                a.yStep[1] = a.yStep[2] * c1.strideY;
                a.yStart[1] = Simd::Min((a.yStart[2] - 1) * c1.strideY + c1.kernelY - c1.padY, c1.srcH);
                a.bufH[1] = Pow2Hi(Simd::Max((a.yStep[2] - 1) * c1.strideY + c1.kernelY, a.yStart[1]));

                a.yStep[0] = a.yStep[1] * c0.strideY;
                a.yStart[0] = Simd::Min((a.yStart[1] - 1) * c0.strideY + c0.kernelY - c0.padY, c0.srcH);
                a.bufH[0] = Pow2Hi(Simd::Max((a.yStep[1] - 1) * c0.strideY + c0.kernelY, a.yStart[0])) * (_s8u ? 0 : 1);

                _sizeB[2] = a.bufH[0] * p.conv[0].srcW * p.conv[0].srcC;
                _sizeB[0] = a.bufH[1] * p.conv[1].srcW * a.maC;
                _sizeB[3] = a.bufH[2] * p.conv[1].dstW * a.maC;
                if (_sizeB[0]*4 + _sizeB[2] + _sizeB[3] <= L2)
                    break;
            }
            _sizeB[1] = 0;
            _sizeB[4] = count > 1 ? _sizeD : 0;
            a.dp[0] = c0.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dp[1] = c1.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dw[0] = c0.kernelY * c0.kernelX * c0.srcC;
            a.dw[1] = c1.kernelY * c1.kernelX;
            a.dw[2] = AlignHiAny(c2.dstC, 2 * a.miC);
            a.size = _d8u ? 1 : 4;
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution8iCd::SynetMergedConvolution8iCd(const MergConvParam8i& p)
            : SynetMergedConvolution8i(p)
        {
        }

        void SynetMergedConvolution8iCd::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const MergConvParam8i& p = _param;
            const ConvParam8i& c0 = p.conv[0];
            const ConvParam8i& c1 = p.conv[1];
            const AlgParam& a = _alg;

            buf = GetBuffer(buf);
            float* buf0 = Allocate<float>(buf, _sizeB[0]);
            uint8_t* buf2 = Allocate<uint8_t>(buf, _sizeB[2]);

            for (size_t b = 0; b < c0.batch; ++b)
            {
                for (size_t c = 0, C = c1.dstC; c < C; c += a.maC)
                {
                    size_t maC = Simd::Min(C, c + a.maC) - c;
                    for (size_t yBeg2 = 0, yBeg1 = 0, yBeg0 = 0; yBeg2 < c1.dstH;)
                    {
                        size_t yEnd2 = Simd::RestrictRange(yBeg2 + a.yStep[2], a.yStart[2], c1.dstH);
                        size_t yEnd1 = Simd::RestrictRange(yBeg1 + a.yStep[1], a.yStart[1], c1.srcH);
                        size_t yEnd0 = Simd::RestrictRange(yBeg0 + a.yStep[0], a.yStart[0], c0.srcH);
                        if (!_s8u)
                            _cvt32fTo8u((float*)src, yBeg0, yEnd0, c0.srcW, c0.srcC, _cvt[0].scale.data, _cvt[0].shift.data, buf2, a.bufH[0], c0.compatibility);
                        _input(_s8u ? src : buf2, c0, a, maC, yBeg1, yEnd1, _weight8i[0].data + c * a.dw[0], _norm[0].data + c,
                            _bias[0].data + c, _params[0].data + c * a.dp[0], buf0);
                        _depthwise(buf0, c1, a, maC, yBeg2, yEnd2, _weight32f.data + c * a.dw[1], _bias[1].data + c,
                            _params[1].data + c * a.dp[1], _cvt[2].scale.data + c, _cvt[2].shift.data + c, dst + c);
                        yBeg2 = yEnd2;
                        yBeg1 = yEnd1;
                        yBeg0 = yEnd0;
                    }
                }
                src += _sizeS * (_s8u ? 1 : 4);
                dst += _sizeD * (_d8u ? 1 : 4);
            }
        }

        bool SynetMergedConvolution8iCd::Preferable(const MergConvParam8i& p)
        {
            return p.count == 2 && p.conv[0].group == 1;
        }

        void SynetMergedConvolution8iCd::SetSize(size_t F)
        {
            const size_t L1 = Base::AlgCacheL1(), L2 = Base::AlgCacheL2(), L3 = Base::AlgCacheL3();
            const MergConvParam8i& p = _param;
            const ConvParam8i& c0 = p.conv[0];
            const ConvParam8i& c1 = p.conv[1];
            AlgParam& a = _alg;
            a.miC = F;
            size_t size = 0;
            for (size_t i = 0; i < 2; ++i)
            {
                const ConvParam8i& c = p.conv[i];
                size += c.kernelY * c.kernelX * c.srcC * (c.group == 1 ? c.dstC : 4);
            }
            size_t count = size / (L3 / 2) + 1;
            a.maC = AlignHiAny(c0.dstC / count, 2 * a.miC);
            for (size_t yStep = c1.dstH; yStep >= 1; yStep--)
            {
                a.yStep[2] = Simd::Max<size_t>(1, yStep);
                a.yStart[2] = a.yStep[2];
                a.bufH[2] = Pow2Hi(a.yStep[2]);

                a.yStep[1] = a.yStep[2] * c1.strideY;
                a.yStart[1] = Simd::Min((a.yStart[2] - 1) * c1.strideY + c1.kernelY - c1.padY, c1.srcH);
                a.bufH[1] = Pow2Hi(Simd::Max((a.yStep[2] - 1) * c1.strideY + c1.kernelY, a.yStart[1]));

                a.yStep[0] = a.yStep[1] * c0.strideY;
                a.yStart[0] = Simd::Min((a.yStart[1] - 1) * c0.strideY + c0.kernelY - c0.padY, c0.srcH);
                a.bufH[0] = Pow2Hi(Simd::Max((a.yStep[1] - 1) * c0.strideY + c0.kernelY, a.yStart[0])) * (_s8u ? 0 : 1);

                _sizeB[2] = a.bufH[0] * p.conv[0].srcW * p.conv[0].srcC;
                _sizeB[0] = a.bufH[1] * p.conv[1].srcW * a.maC;
                if (_sizeB[0] * 4 + _sizeB[2] <= L2)
                    break;
            }
            a.bufH[2] = 0;
            _sizeB[1] = 0;
            _sizeB[3] = 0;
            _sizeB[4] = 0;
            a.dp[0] = c0.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dp[1] = c1.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dw[0] = c0.kernelY * c0.kernelX * c0.srcC;
            a.dw[1] = c1.kernelY * c1.kernelX;
            a.size = _d8u ? 1 : 4;
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution8iDc::SynetMergedConvolution8iDc(const MergConvParam8i& p)
            : SynetMergedConvolution8i(p)
        {
        }

        void SynetMergedConvolution8iDc::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const MergConvParam8i& p = _param;
            const ConvParam8i& c0 = p.conv[0];
            const ConvParam8i& c1 = p.conv[1];
            const AlgParam& a = _alg;

            buf = GetBuffer(buf);
            float* buf0 = Allocate<float>(buf, _sizeB[0]);
            uint8_t* buf2 = Allocate<uint8_t>(buf, _sizeB[2]);
            int32_t* buf4 = Allocate<int32_t>(buf, _sizeB[4]);

            for (size_t b = 0; b < c0.batch; ++b)
            {
                for (size_t c = 0, C = c0.dstC; c < C; c += a.maC)
                {
                    size_t maC = Simd::Min(C, c + a.maC) - c;
                    for (size_t yBeg2 = 0, yBeg1 = 0, yBeg0 = 0; yBeg2 < c0.dstH;)
                    {
                        size_t yEnd2 = Simd::RestrictRange(yBeg2 + a.yStep[2], a.yStart[2], c0.dstH);
                        size_t yEnd1 = Simd::RestrictRange(yBeg1 + a.yStep[1], a.yStart[1], c0.srcH);
                        if (_s8u)
                            _cvt8uTo32f(src + c, maC, yBeg1, yEnd1, c0.srcW, c0.srcC, _cvt[0].iScale.data + c, 
                                _cvt[0].iShift.data + c, buf0, a.bufH[1], c0.compatibility);
                        _depthwise(_s8u ? buf0 : (float*)src + c, c0, a, maC, yBeg2, yEnd2, _weight32f.data + c * a.dw[0], _bias[0].data + c,
                            _params[0].data + c * a.dp[0], _cvt[1].scale.data + c, _cvt[1].shift.data + c, buf2);
                        if (c + maC == C)
                            _output[0](buf2, c1, a, maC, yBeg2, yEnd2, _weight8i[0].data + c * a.dw[1], _norm[0].data, _bias[1].data,
                                _params[1].data, _cvt[2].scale.data, _cvt[2].shift.data, maC == C ? NULL : buf4, dst, maC == C ? 1 : 0);
                        else
                            _output[1](buf2, c1, a, maC, yBeg2, yEnd2, _weight8i[0].data + c * a.dw[1], _norm[0].data, _bias[1].data,
                                _params[1].data, _cvt[2].scale.data, _cvt[2].shift.data, buf4, dst, c == 0 ? 1 : 0);
                        yBeg2 = yEnd2;
                        yBeg1 = yEnd1;
                    }
                }
                src += _sizeS * (_s8u ? 1 : 4);
                dst += _sizeD * (_d8u ? 1 : 4);
            }
        }

        bool SynetMergedConvolution8iDc::Preferable(const MergConvParam8i& p)
        {
            return p.count == 2 && p.conv[1].group == 1;
        }

        void SynetMergedConvolution8iDc::SetSize(size_t F)
        {
            const size_t L1 = Base::AlgCacheL1(), L2 = Base::AlgCacheL2(), L3 = Base::AlgCacheL3();
            const MergConvParam8i& p = _param;
            const ConvParam8i& c0 = p.conv[0];
            const ConvParam8i& c1 = p.conv[1];
            AlgParam& a = _alg;
            a.miC = F;
            size_t size = 0;
            for (size_t i = 0; i < 2; ++i)
            {
                const ConvParam8i& c = p.conv[i];
                size += c.kernelY * c.kernelX * c.srcC * (c.group == 1 ? c.dstC : 4);
            }
            size_t count = size / (L3 / 2) + 1;
            a.maC = AlignHiAny(c0.srcC / count, 2 * a.miC);
            for (size_t yStep = c0.dstH; yStep >= 1; yStep--)
            {
                a.yStep[2] = Simd::Max<size_t>(1, yStep);
                a.yStart[2] = a.yStep[2];
                a.bufH[2] = Pow2Hi(a.yStep[2]);

                a.yStep[1] = a.yStep[2] * c0.strideY;
                a.yStart[1] = Simd::Min((a.yStart[2] - 1) * c0.strideY + c0.kernelY - c0.padY, c0.srcH);
                a.bufH[1] = Pow2Hi(Simd::Max((a.yStep[2] - 1) * c0.strideY + c0.kernelY, a.yStart[1])) * (_s8u ? 1 : 0);

                _sizeB[0] = a.bufH[1] * p.conv[0].srcW * a.maC;
                _sizeB[2] = a.bufH[2] * p.conv[1].srcW * a.maC;
                if (_sizeB[0] * 4 + _sizeB[2] <= L2)
                    break;
            }
            a.bufH[0] = 0;
            _sizeB[1] = 0;
            _sizeB[3] = 0;
            _sizeB[4] = count > 1 ? _sizeD : 0;
            a.dp[0] = c0.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dp[1] = c1.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dw[0] = c0.kernelY * c0.kernelX;
            a.dw[1] = AlignHiAny(c1.dstC, 2 * a.miC);
            a.size = _d8u ? 1 : 4;
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
#endif
}
