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
#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    void CvtParam::Init(const float* min, const float* max, size_t size, SimdSynetCompatibilityType compatibility)
    {
        zero.Resize(size);
        scale.Resize(size);
        shift.Resize(size);
        iScale.Resize(size);
        iShift.Resize(size);
        for (size_t i = 0; i < size; ++i)
        {
            assert(min[i] <= max[i]);
            if (min[i] < 0.0f)
                neg = true;
        }
        uMin = Base::Narrowed(compatibility) ? Base::U8_NARROWED_MIN : Base::U8_PRECISE_MIN;
        uMax = Base::Narrowed(compatibility) ? Base::U8_NARROWED_MAX : Base::U8_PRECISE_MAX;
        iMin = Base::Narrowed(compatibility) ? Base::I8_NARROWED_MIN : Base::I8_PRECISE_MIN;
        iMax = Base::Narrowed(compatibility) ? Base::I8_NARROWED_MAX : Base::I8_PRECISE_MAX;
        for (size_t i = 0; i < size; ++i)
        {
            float abs = ::fmax(::fabs(min[i]), ::fabs(max[i]));
            float inv = abs / (neg ? iMax : uMax);
            if (::fabs(inv) < 1e-7)
                inv = 1.0f;
            zero[i] = uint8_t(neg ? -iMin : uMin);
            scale[i] = float(1.0 / inv);
            shift[i] = float(zero[i]);
            iScale[i] = inv;
            iShift[i] = -float(zero[i]) * inv;
        }
    }

    //-------------------------------------------------------------------------

    SynetConvolution8i::SynetConvolution8i(const ConvParam8i& p)
        : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        , _perf(NULL)
#endif
    {
        _sizeS = p.srcC * p.srcH * p.srcW;
        _sizeD = p.dstC * p.dstH * p.dstW;
        _merge = 1;
        _src8u = p.srcT == SimdTensorData8u;
        _dst8u = p.dstT == SimdTensorData8u;
        _weight.Resize(p.kernelY * p.kernelX * p.srcC / p.group * p.dstC);
        _norm.Resize(p.dstC);
        _bias.Resize(p.dstC);
        _convertSrc = Base::SynetConvert32fTo8u;
    }

    size_t SynetConvolution8i::ExternalBufferSize() const
    {
        size_t size = SIMD_ALIGN;
        if (!_src8u)
            size += AlignHi(_sizeS * _merge * sizeof(uint8_t), SIMD_ALIGN);
        return size;
    }

    size_t SynetConvolution8i::InternalBufferSize() const
    {
        return (_buffer.size + _weight.size) * sizeof(uint8_t) + _srcCvt.Size() + 
            _dstCvt.Size() + (_norm.size + _bias.size + _params.size) * sizeof(float);
    }

    void SynetConvolution8i::SetParams(const float* weight, const float* bias, const float* params, const float* const* stats)
    {
        const ConvParam8i& p = _param;
        _srcCvt.Init(stats[0], stats[1], p.srcC, p.compatibility);
        _dstCvt.Init(stats[2], stats[3], p.dstC, p.compatibility);
        size_t G = p.group, D = p.dstC / G, C = p.srcC / G, K = p.kernelY * p.kernelX, CK = C * K, GD = G * D;
        Array32f normW(CK);
        const float* pSrcW = weight;
        const float* pSrcB = bias;
        const float* pScale = _srcCvt.scale.data;
        const float* pShift = _srcCvt.shift.data;
        float* pNormW = normW.data;
        int8_t* pDstW = _weight.data;
        float* pNorm = _norm.data;
        float* pBias = _bias.data;
        bool avoidOverflow = _srcCvt.neg && Base::Overflow(p.compatibility);
        for (size_t g = 0; g < G; ++g)
        {
            for (size_t d = 0; d < D; ++d)
            {
                float normB = 0, minW = FLT_MAX, maxW = -FLT_MAX, scale = 1.0f;
                if (p.trans)
                {
                    for (size_t k = 0, kc = 0; k < K; ++k)
                        for (size_t c = 0; c < C; ++c, ++kc)
                        {
                            pNormW[kc] = pSrcW[kc * GD + d] / pScale[c];
                            minW = Simd::Min(minW, pNormW[kc]);
                            maxW = Simd::Max(maxW, pNormW[kc]);
                        }
                    scale = _srcCvt.iMax / Max(Simd::Abs(maxW), Simd::Abs(minW));
                    for (size_t k = 0, kc = 0; k < K; ++k)
                        for (size_t c = 0; c < C; ++c, ++kc)
                            if (avoidOverflow)
                            {
                                int w = Base::SynetConvert32fTo8i(pNormW[kc], scale, 0.0f, _srcCvt.iMin, _srcCvt.iMax);
                                if (w & 1)
                                    w = Round(w * 0.25f) * 4;
                                pDstW[kc * GD + d] = w / 2;
                                normB -= w * pShift[c];
                            }
                            else
                            {
                                pDstW[kc * GD + d] = Base::SynetConvert32fTo8i(pNormW[kc], scale, 0.0f, _srcCvt.iMin, _srcCvt.iMax);
                                normB -= pDstW[kc * GD + d] * pShift[c];
                            }
                }
                else
                {
                    for (size_t c = 0, ck = 0; c < C; ++c)
                        for (size_t k = 0; k < K; ++k, ++ck)
                        {
                            pNormW[ck] = pSrcW[d * CK + ck] / pScale[c];
                            minW = Simd::Min(minW, pNormW[ck]);
                            maxW = Simd::Max(maxW, pNormW[ck]);
                        }
                    scale = _srcCvt.iMax / Max(Simd::Abs(maxW), Simd::Abs(minW));
                    for (size_t c = 0, ck = 0; c < C; ++c)
                        for (size_t k = 0; k < K; ++k, ++ck)
                            if (avoidOverflow)
                            {
                                int w = Base::SynetConvert32fTo8i(pNormW[ck], scale, 0.0f, _srcCvt.iMin, _srcCvt.iMax);
                                if (w & 1)
                                    w = Round(w * 0.25f) * 4;
                                pDstW[d * CK + ck] = w / 2;
                                normB -= w * pShift[c];
                            }
                            else
                            {
                                pDstW[d * CK + ck] = Base::SynetConvert32fTo8i(pNormW[ck], scale, 0.0f, _srcCvt.iMin, _srcCvt.iMax);
                                normB -= pDstW[d * CK + ck] * pShift[c];
                            }
                }
                pNorm[d] = (avoidOverflow ? 2.0f : 1.0f) / scale;
                pBias[d] = (pSrcB ? pSrcB[d] : 0.0f) + normB / scale;
            }
            if (p.trans)
            {
                pSrcW += D;
                pDstW += D;
            }
            else
            {
                pSrcW += CK * D;
                pDstW += CK * D;
            }
            if (pSrcB)
                pSrcB += D;
            pScale += C;
            pShift += C;
            pNorm += D;
            pBias += D;
        }
        if (p.activation == SimdConvolutionActivationLeakyRelu || p.activation == SimdConvolutionActivationPrelu)
            _params.Resize(p.dstC);
        else
            _params.Resize(2);
        switch (p.activation)
        {
        case SimdConvolutionActivationIdentity:
            _params[0] = -FLT_MAX;
            _params[1] = FLT_MAX;
            break;
        case SimdConvolutionActivationRelu:
            _params[0] = 0;
            _params[1] = FLT_MAX;
            break;
        case SimdConvolutionActivationLeakyRelu:
            for (size_t d = 0; d < p.dstC; ++d)
                _params[d] = params[0];
            break;
        case SimdConvolutionActivationRestrictRange:
            _params[0] = params[0];
            _params[1] = params[1];
            break;
        case SimdConvolutionActivationPrelu:
            for (size_t d = 0; d < p.dstC; ++d)
                _params[d] = params[d];
            break;
        case SimdConvolutionActivationElu:
            _params[0] = params[0];
            break;
        case SimdConvolutionActivationHswish:
            _params[0] = params[0];
            _params[1] = params[1];
            break;
        case SimdConvolutionActivationMish:
            _params[0] = params[0];
            break;
        case SimdConvolutionActivationHardSigmoid:
            _params[0] = params[0];
            _params[1] = params[1];
            break;
        case SimdConvolutionActivationSwish:
            _params[0] = params[0];
            break;
        default:
            assert(0);
        }
    }

    void SynetConvolution8i::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
    {
        if (buf == NULL)
        {
            _buffer.Resize(ExternalBufferSize());
            buf = _buffer.data;
        }
        const ConvParam8i& p = _param;
        uint8_t* src8u = _src8u ? NULL : Allocate<uint8_t>(buf, _sizeS * _merge);
        for (size_t b = 0; b < p.batch; b += _merge)
        {
            if (!_src8u)
                _convertSrc((float*)src + b * _sizeS, _merge, p.srcC, p.srcH, p.srcW, p.srcF, _srcCvt.scale.data, _srcCvt.shift.data, src8u, p.compatibility);
            Forward8u(_src8u ? src + b * _sizeS : src8u, buf, dst + b * _sizeD * (_dst8u ? sizeof(uint8_t) : sizeof(float)));
        }
    }

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
    Base::PerformanceMeasurer * SynetConvolution8i::Perf(const char* func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

    //-------------------------------------------------------------------------

    namespace Base
    {
        SynetConvolution8iGemmNN::SynetConvolution8iGemmNN(const ConvParam8i& p)
            : SynetConvolution8i(p)
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
                _grD = p.dstH * p.dstW *p.dstC / p.group;
            }
            _siK = p.kernelY * p.kernelX;
            _siC = p.srcC / p.group;
            _siD = p.dstC / p.group;
            _siS = p.dstH * p.dstW;
        }

        size_t SynetConvolution8iGemmNN::ExternalBufferSize() const
        {
            size_t size = SynetConvolution8i::ExternalBufferSize();
            if(!_skipConv)
                size += AlignHi(_sizeB * _merge * sizeof(uint8_t), SIMD_ALIGN);
            size += AlignHi(_sizeD * _merge * sizeof(int32_t), SIMD_ALIGN);
            if(_dst8u)
                size += AlignHi(_sizeD * _merge * sizeof(float), SIMD_ALIGN);
            return size;
        }

        void SynetConvolution8iGemmNN::Forward8u(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const ConvParam8i& p = _param;
            const int8_t * weight = _weight.data;
            int32_t * sum = Allocate<int32_t>(buf, _sizeD * _merge);
            float * dst32f = _dst8u ? Allocate<float>(buf, _sizeD * _merge) : (float*)dst;
            if (!_skipConv)
            {
                const uint8_t* zero = _srcCvt.zero.data;
                if(p.trans)
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
            Convert<int32_t, float, float>(sum, _merge, p.dstC, p.dstH, p.dstW, p.dstF, _norm.data, _bias.data, 0, 0, dst32f);
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity:
                break;
            case SimdConvolutionActivationRelu:
            {
                float slope = 0;
                SynetRelu32f(dst32f, _merge * _sizeD, &slope, dst32f);
                break;
            }
            case SimdConvolutionActivationLeakyRelu:
                SynetRelu32f(dst32f, _merge * _sizeD, _params.data, dst32f);
                break;
            case SimdConvolutionActivationRestrictRange:
                SynetRestrictRange32f(dst32f, _merge * _sizeD, _params.data, _params.data + 1, dst32f);
                break;
            case SimdConvolutionActivationPrelu:
                for (size_t m = 0; m < _merge; ++m)
                    SynetPreluLayerForward(dst32f + m * _sizeD, _params.data, p.dstC, p.dstH*p.dstW, dst32f + m * _sizeD, p.dstF);
                break;
            case SimdConvolutionActivationElu:
                SynetElu32f(dst32f, _merge * _sizeD, _params.data, dst32f);
                break;
            case SimdConvolutionActivationHswish:
                SynetHswish32f(dst32f, _merge * _sizeD, _params.data, _params.data + 1, dst32f);
                break;
            case SimdConvolutionActivationMish:
                SynetMish32f(dst32f, _merge * _sizeD, _params.data, dst32f);
                break;
            case SimdConvolutionActivationHardSigmoid:
                SynetHardSigmoid32f(dst32f, _merge * _sizeD, _params.data, _params.data + 1, dst32f);
                break;
            case SimdConvolutionActivationSwish:
                SynetSwish32f(dst32f, _merge * _sizeD, _params.data, dst32f);
                break;
            default:
                assert(0);
            }
            if (_dst8u)
                Convert<float, uint8_t, float>(dst32f, _merge, p.dstC, p.dstH, p.dstW, p.dstF, _dstCvt.scale.data, _dstCvt.shift.data, _dstCvt.uMin, _dstCvt.uMax, dst);
        }

        //---------------------------------------------------------------------

        SynetConvolution8iNhwcDirect::SynetConvolution8iNhwcDirect(const ConvParam8i& p)
            : SynetConvolution8i(p)
        {
            for (size_t i = 0; i < Term8iSize; ++i)
                _convolutions[i] = NULL;
        }

        String SynetConvolution8iNhwcDirect::Desc() const
        {
            const ConvParam8i& p = _param;
            return Ext() + "::NhwcDirect" + (Overflow(p.compatibility) ? "-o" : (Narrowed(p.compatibility) ? "-n" : "-p"));
        }

        size_t SynetConvolution8iNhwcDirect::InternalBufferSize() const
        {
            size_t size = SynetConvolution8i::InternalBufferSize();
            return size;
        }

        size_t SynetConvolution8iNhwcDirect::ExternalBufferSize() const
        {
            const ConvParam8i& p = _param;
            size_t size = SynetConvolution8i::ExternalBufferSize();
            if (_alg.macroC < p.srcC)
                size += AlignHi(_sizeD * sizeof(int32_t), SIMD_ALIGN);
            if(_sizeP)
                size += AlignHi(_sizeP * sizeof(uint8_t), SIMD_ALIGN);
            return size;
        }

        void SynetConvolution8iNhwcDirect::SetParams(const float* weight, const float* bias, const float* params, const float* const* stats)
        {
            SynetConvolution8i::SetParams(weight, bias, params, stats);
            ReorderWeight();
            _alg.zero = Set4(_srcCvt.zero[0]);
            _alg.upper = Set4(_dstCvt.uMax);
        }

        bool SynetConvolution8iNhwcDirect::Preferable(const ConvParam8i& p)
        {
            return false;
        }

        bool SynetConvolution8iNhwcDirect::PadEnable(size_t microC)
        {
            const ConvParam8i& p = _param;
            if (p.padX == 0 && p.padW == 0)
                return false;
            if (_alg.macroH < p.dstH)
                return false;
            size_t nose = p.NoseW(), body = p.BodyW() - nose, tail = DivHi(p.padW, p.strideX);
            size_t srcSteps = nose + DivHi(body, microC) + tail;
            size_t padSteps = DivHi(nose + body + tail, microC);
            return srcSteps >= padSteps*1.3;
        }

        void SynetConvolution8iNhwcDirect::SetAlgParam(size_t F, size_t microD, size_t L1, size_t L2, size_t L3, size_t microC)
        {
            const ConvParam8i& p = _param;
            _alg.F = F;
            _alg.microD = microD;
            _alg.macroC = Simd::Min(AlignLoAny(L1 / p.kernelY / p.kernelX / microD, 4), p.srcC);
            for (size_t macroH = p.dstH; macroH >= 1; macroH--)
            {
                _alg.macroH = macroH;
                if (_alg.macroC * p.srcW * (_alg.macroH * p.strideY + p.kernelY * p.dilationY - 1) <= L2)
                    break;
            }
            _alg.macroD = Simd::Min(AlignLoAny(L3 / p.kernelY / p.kernelX / _alg.macroC, _alg.microD), AlignHiAny(p.dstC, _alg.microD));
            _alg.size = (p.dstT == SimdTensorData32f ? 4 : 1);
            if (PadEnable(microC))
            {
                _paramP = p;
                _paramP.srcW = p.srcW + p.padX + p.padW;
                _paramP.padX = 0;
                _paramP.padW = 0;
                _sizeP = p.srcH * _paramP.srcW * p.srcC;
            }
            else
                _sizeP = 0;
        }

        void SynetConvolution8iNhwcDirect::ReorderWeight()
        {
            const ConvParam8i& p = _param;
            size_t C = DivHi(p.srcC, 4), D = DivHi(p.dstC, _alg.F);
            Array8i weight(p.kernelY * p.kernelX * C * D * _alg.F * 4);
            int8_t* dst = weight.data;
            for (size_t d = 0; d < D; d++)
            {
                for (size_t ky = 0; ky < p.kernelY; ++ky)
                {
                    for (size_t kx = 0; kx < p.kernelX; ++kx)
                    {
                        for (size_t c = 0; c < C; ++c)
                        {
                            const int8_t* src = _weight.data + ((ky * p.kernelX + kx) * p.srcC + c * 4) * p.dstC + d * _alg.F;
                            for (size_t f = 0; f < _alg.F; ++f)
                            {
                                for (size_t i = 0; i < 4; ++i)
                                {
                                    if (d * _alg.F + f < p.dstC && c * 4 + i < p.srcC)
                                        *(dst++) = src[i * p.dstC];
                                    else
                                        *(dst++) = 0;
                                }
                                src++;
                            }
                        }
                    }
                }
            }
            _weight.Swap(weight);
        }

        void SynetConvolution8iNhwcDirect::Forward8u(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            int32_t * sum = _alg.macroC < _param.srcC ? Allocate<int32_t>(buf, _sizeD) : NULL;
            uint8_t * pad = _sizeP ? Allocate<uint8_t>(buf, _sizeP) : NULL;
            for (size_t m = 0; m < _merge; ++m)
            {
                if (_sizeP)
                {
                    PadInput(src, pad);
                    Forward8u(pad, _paramP, sum, dst);
                }
                else
                    Forward8u(src, _param, sum, dst);
                src += _sizeS;
                dst += _sizeD * (_dst8u ? sizeof(uint8_t) : sizeof(float));
            }
        }

        void SynetConvolution8iNhwcDirect::PadInput(const uint8_t* src, uint8_t* dst)
        {
            const ConvParam8i& p = _param;
            size_t nose = p.padX * p.srcC * sizeof(uint8_t);
            size_t body = p.srcW * p.srcC * sizeof(uint8_t);
            size_t tail = p.padW * p.srcC * sizeof(uint8_t);
            for (size_t y = 0; y < p.srcH; ++y)
            {
                memset(dst, _srcCvt.zero[0], nose), dst += nose;
                memcpy(dst, src, body), src += body, dst += body;
                memset(dst, _srcCvt.zero[0], tail), dst += tail;
            }
        }

        void SynetConvolution8iNhwcDirect::Forward8u(const uint8_t* src, const ConvParam8i& p, int32_t* buf, uint8_t* dst)
        {
            const int8_t* weight = _weight.data;
            const float* norm = _norm.data;
            const float* bias = _bias.data;
            const float* params = _params.data;
            const float* scale = _dstCvt.scale.data;
            const float* shift = _dstCvt.shift.data;
            for (size_t dc = 0; dc < p.dstC; dc += _alg.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + _alg.macroD) - dc;
                for (size_t sc = 0; sc < p.srcC; sc += _alg.macroC)
                {
                    size_t macroC = Simd::Min(p.srcC, sc + _alg.macroC) - sc;
                    for (size_t yBeg = 0; yBeg < p.dstH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + _alg.macroH, p.dstH);
                        if (sc + macroC == p.srcC)
                        {
                            int first = macroC == p.srcC ? 1 : 0;
                            if (_alg.size == 1)
                                _convolutions[Term8iLast8u](src + sc, p, _alg, macroD, yBeg, yEnd, macroC, weight, norm, bias, params, scale, shift, buf, dst, first);
                            else
                                _convolutions[Term8iLast32f](src + sc, p, _alg, macroD, yBeg, yEnd, macroC, weight, norm, bias, params, scale, shift, buf, dst, first);
                        }
                        else
                            _convolutions[Term8iInterim](src + sc, p, _alg, macroD, yBeg, yEnd, macroC, weight, norm, bias, params, scale, shift, buf, dst, sc == 0 ? 1 : 0);
                        yBeg = yEnd;
                    }
                    weight += DivHi(macroC, 4) * _alg.F * 4;
                }
                weight += p.kernelY * p.kernelX * DivHi(p.srcC, 4) * macroD * 4 - DivHi(p.srcC, 4) * _alg.F * 4;
                norm += macroD;
                bias += macroD;
                if (p.activation == ::SimdConvolutionActivationLeakyRelu || p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroD;
                shift += macroD;
                scale += macroD;
                if (buf)
                    buf += _alg.macroD;
                dst += _alg.macroD * _alg.size;
            }
        }

        //---------------------------------------------------------------------

        SynetConvolution8iNhwcDepthwise::SynetConvolution8iNhwcDepthwise(const ConvParam8i& p)
            : SynetConvolution8i(p)
        {
            _convolution = NULL;
        }

        String SynetConvolution8iNhwcDepthwise::Desc() const
        {
            const ConvParam8i& p = _param;
            return Ext() + "::NhwcDepthwise" + (Overflow(p.compatibility) ? "-o" : (Narrowed(p.compatibility) ? "-n" : "-p"));
        }

        void SynetConvolution8iNhwcDepthwise::SetParams(const float* weight, const float* bias, const float* params, const float* const* stats)
        {
            SynetConvolution8i::SetParams(weight, bias, params, stats);
            _alg.zero = _srcCvt.zero[0];
            _alg.upper = Set4(_dstCvt.uMax);
            _alg.size = (_param.dstT == SimdTensorData32f ? 4 : 1);
        }

        bool SynetConvolution8iNhwcDepthwise::Preferable(const ConvParam8i& p)
        {
            return false;
        }

        void SynetConvolution8iNhwcDepthwise::Forward8u(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const int8_t* weight = _weight.data;
            const float* norm = _norm.data;
            const float* bias = _bias.data;
            const float* params = _params.data;
            const float* scale = _dstCvt.scale.data;
            const float* shift = _dstCvt.shift.data;
            for (size_t m = 0; m < _merge; ++m)
            {
                _convolution(src, _param, _alg, weight, norm, bias, params, scale, shift, dst);
                src += _sizeS;
                dst += _sizeD * _alg.size;
            }
        }

        //---------------------------------------------------------------------

//#define SIMD_BASE_ONLY_GEMM_NN

        void * SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility)
        {
            ConvParam8i param(batch, conv, compatibility);
            if (!param.Valid())
                return NULL;
#if !defined(SIMD_BASE_ONLY_GEMM_NN)
            else if (SynetConvolution8iNhwcDepthwise::Preferable(param))
                return new SynetConvolution8iNhwcDepthwise(param);
            else if (SynetConvolution8iNhwcDirect::Preferable(param))
                return new SynetConvolution8iNhwcDirect(param);
#endif
            else
                return new SynetConvolution8iGemmNN(param);
        }
    }
#endif
}
