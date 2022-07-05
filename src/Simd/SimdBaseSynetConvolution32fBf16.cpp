/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetConvolution32fBf16Gemm::SynetConvolution32fBf16Gemm(const ConvParam32f & p)
            : SynetConvolution32f(p)
        {
            if (p.trans)
            {
                _M = p.dstH * p.dstW;
                _N = p.dstC / p.group;
                _K = p.srcC * p.kernelY * p.kernelX / p.group;
                _ldS = _K;
                _ldW = p.dstC;
                _ldD = p.dstC;
                _grW = _N;
                _grS = _K * _M;
                _grD = _N;
                _weight.Resize(_K * _N);
            }
            else
            {
                _M = p.dstC / p.group;
                _N = p.dstH * p.dstW;
                _K = p.srcC * p.kernelY * p.kernelX / p.group;
                _ldW = _K;
                _ldS = _N;
                _ldD = _N;
                _grW = _M * _K;
                _grS = _K * _N;
                _grD = _M * _N;
                _weight.Resize(_K * _M);
            }
            _batch = p.batch;
            _sizeS = p.srcC * p.srcH * p.srcW;
            _sizeB = p.srcC * p.kernelY * p.kernelX * p.dstH * p.dstW;
            _sizeD = p.dstC * p.dstH * p.dstW;
        }

        size_t SynetConvolution32fBf16Gemm::ExternalBufferSize() const
        {
            return _sizeB;
        };

        void SynetConvolution32fBf16Gemm::SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params)
        {
            Simd::SynetConvolution32f::SetParams(weight, internal, bias, params);
            Float32ToBFloat16(weight, _weight.size, _weight.data);
            if (internal)
                *internal = SimdTrue;
        }

        void SynetConvolution32fBf16Gemm::Forward(const float * src, float * buf_, float * dst)
        {
            const ConvParam32f & p = _param;
            uint16_t * buf = (uint16_t*)Buffer(buf_);
            const uint16_t* wgt = _weight.data;
            for (size_t b = 0; b < _batch; ++b)
            {
                if (_param.trans)
                {
                    ImgToRow(src, buf);
                    for (size_t g = 0; g < p.group; ++g)
                        GemmNN(_M, _N, _K, buf + _grS * g, _ldS, wgt + _grW * g, _ldW, dst + _grD * g, _ldD);
                }
                else
                {
                    ImgToCol(src, buf);
                    for (size_t g = 0; g < p.group; ++g)
                        GemmNN(_M, _N, _K, wgt + _grW * g, _ldW, buf + _grS * g, _ldS, dst + _grD * g, _ldD);
                }
                ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH * p.dstW, p.activation, _params, p.trans, dst);
                src += _sizeS;
                dst += _sizeD;
            }
        }

        void SynetConvolution32fBf16Gemm::ImgToCol(const float* src, uint16_t* dst)
        {
            const ConvParam32f& p = _param;
            assert(!p.trans);
            size_t srcSize = p.srcW * p.srcH;
            for (size_t c = 0; c < p.srcC; ++c)
            {
                for (size_t ky = 0; ky < p.kernelY; ky++)
                {
                    for (size_t kx = 0; kx < p.kernelX; kx++)
                    {
                        size_t sy = ky * p.dilationY - p.padY;
                        for (size_t dy = 0; dy < p.dstH; ++dy)
                        {
                            if (sy < p.srcH)
                            {
                                size_t sx = kx * p.dilationX - p.padX;
                                for (size_t dx = 0; dx < p.dstW; ++dx)
                                {
                                    if (sx < p.srcW)
                                        *(dst++) = Float32ToBFloat16(src[sy * p.srcW + sx]);
                                    else
                                        *(dst++) = 0;
                                    sx += p.strideX;
                                }
                            }
                            else
                            {
                                for (size_t dx = 0; dx < p.dstW; ++dx)
                                    *(dst++) = 0;
                            }
                            sy += p.strideY;
                        }
                    }
                }
                src += srcSize;
            }
        }

        void SynetConvolution32fBf16Gemm::ImgToRow(const float* src, uint16_t* dst)
        {
            const ConvParam32f& p = _param;
            assert(p.trans);
            size_t size = p.srcC / p.group;
            for (size_t g = 0; g < p.group; ++g)
            {
                for (size_t dy = 0; dy < p.dstH; ++dy)
                {
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                    {
                        for (size_t ky = 0; ky < p.kernelY; ky++)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; kx++)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        Float32ToBFloat16(src + (sy * p.srcW + sx) * p.srcC, size, dst);
                                        dst += size;
                                    }
                                    else
                                    {
                                        memset(dst, 0, size * sizeof(uint16_t));
                                        dst += size;
                                    }
                                }
                            }
                            else
                            {
                                memset(dst, 0, p.kernelX * size * sizeof(uint16_t));
                                dst += p.kernelX * size;
                            }
                        }
                    }
                }
                src += size;
            }
        }

        void SynetConvolution32fBf16Gemm::GemmNN(size_t M, size_t N, size_t K, const uint16_t* A, size_t lda, const uint16_t* B, size_t ldb, float* C, size_t ldc)
        {
            for (size_t i = 0; i < M; ++i)
            {
                float* pC = C + i * ldc;
                for (size_t j = 0; j < N; ++j)
                    pC[j] = 0.0f;
                for (size_t k = 0; k < K; ++k)
                {
                    const uint16_t* pB = B + k * ldb;
                    float a = BFloat16ToFloat32(A[i * lda + k]);
                    for (size_t j = 0; j < N; ++j)
                        pC[j] += a * BFloat16ToFloat32(pB[j]);
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        SynetConvolution32fBf16Nhwc::SynetConvolution32fBf16Nhwc(const ConvParam32f& p)
            : SynetConvolution32f(p)
        {
        }

        String SynetConvolution32fBf16Nhwc::Desc() const 
        {
            std::stringstream desc;
            desc << Ext() << "::Bf16Nhwc-";
            if (_alg.mode)
            {
                desc << "gemm";
                if (_alg.batch > 1)
                    desc << "-" << _alg.batch;
            }
            else
                desc << "conv";
            return desc.str();
        }


        void SynetConvolution32fBf16Nhwc::SetAlgParam(size_t microD, size_t microHW, size_t microC, size_t L1, size_t L2, size_t L3)
        {
            const ConvParam32f& p = _param;
            AlgParam& a = _alg;

            a.mode = PreferableMode(microD, microHW, microC, L1, L2, L3);
            a.batch = 1;
            if (a.mode)
            {
                a.srcH = p.dstH;
                a.srcW = p.dstW;
                a.microD = microD;
                a.macroC = Simd::RestrictRange(AlignHiAny(L1 / p.kernelY / p.kernelX / microD / 2, 2), size_t(1), p.srcC);
                size_t matSize = p.srcC * p.dstW * p.dstH * p.kernelY * p.kernelX * 2;
                if (matSize * 2 <= L2 && p.batch > 1)
                {
                    for (size_t batch = 1; batch <= p.batch; ++batch)
                        if (p.batch % batch == 0 && batch * matSize <= L2)
                            a.batch = batch;
                }
                a.macroH = Simd::RestrictRange(L2 / 2 / a.macroC / p.kernelY / p.kernelX / p.dstW, size_t(1), p.dstH * a.batch);
                a.macroD = Simd::RestrictRange(AlignLoAny(L3 / a.macroC / 2, a.microD), a.microD, AlignHiAny(p.dstC, a.microD));
            }
            else
            {
                a.srcH = p.srcH + p.padY + p.padH;
                a.srcW = p.srcW + p.padX + p.padW;
                a.microD = microD;
                a.macroC = Simd::RestrictRange<size_t>(AlignHiAny(L1 / p.kernelY / p.kernelX / microD / 2, 2), size_t(1), p.srcC);
                for (size_t macroH = p.dstH; macroH >= 1; macroH--)
                {
                    a.macroH = macroH;
                    if (a.macroC * a.srcW * (a.macroH * p.strideY + p.kernelY * p.dilationY - 1) * 2 <= L2)
                        break;
                }
                a.macroD = Simd::RestrictRange(AlignLoAny(L3 / p.kernelY / p.kernelX / a.macroC / 2, a.microD), a.microD, AlignHiAny(p.dstC, a.microD));
            }
        }

        int SynetConvolution32fBf16Nhwc::PreferableMode(size_t microD, size_t microHW, size_t microC, size_t L1, size_t L2, size_t L3)
        {
            const ConvParam32f& p = _param;
            if (p.Is1x1())
                return 1;
            float kernelArea = float(p.kernelY * p.kernelX);
            float convArea = kernelArea / float(p.strideY * p.strideX);
            if (convArea <= 1.5f)
                return 1;
            if (p.dstW * p.dstH < 25)
                return 1;
            if (L1 / kernelArea / microD / 2 < microC)
                return 1;
            return 0;
        }

        size_t SynetConvolution32fBf16Nhwc::ExternalBufferSize() const
        {
            const ConvParam32f& p = _param;
            const AlgParam& a = _alg;
            if (p.dstC <= a.macroD)
                return Base::AlgCacheL2() / 4;
            else
            {
                if (a.mode)
                    return a.batch * AlignHi(p.srcC, 2) * p.dstW * p.dstH * p.kernelY * p.kernelX / 2;
                else
                    return a.srcH * a.srcW * AlignHi(p.srcC, 2) / 2;
            }
        }

        size_t SynetConvolution32fBf16Nhwc::InternalBufferSize() const
        {
            return _buffer.size + _weight.size / 2 + _bias.size + _params.size;
        }

        void SynetConvolution32fBf16Nhwc::SetParams(const float* weight, SimdBool* internal, const float* bias, const float* params)
        {
            SetWeight(weight);
            if (internal)
                *internal = SimdTrue;
            SetBias(bias);
            SetParams(params);
        }

        void SynetConvolution32fBf16Nhwc::SetWeight(const float * weight)
        {
            const ConvParam32f& p = _param;
            const AlgParam& a = _alg;
            _weight.Resize(p.kernelY * p.kernelX * AlignHi(p.srcC, 2) * AlignHiAny(p.dstC, a.microD));
            uint16_t * dst = _weight.data;
            for (size_t mad = 0; mad < p.dstC; mad += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, mad + a.macroD) - mad;
                for (size_t mac = 0; mac < p.srcC; mac += a.macroC)
                {
                    size_t macroC = Simd::Min(p.srcC, mac + a.macroC) - mac;
                    for (size_t mid = 0; mid < macroD; mid += a.microD)
                    {
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                for (size_t c = 0; c < macroC; c += 2)
                                {
                                    const float* src = weight + ((ky * p.kernelX + kx) * p.srcC + mac + c) * p.dstC + mad + mid;
                                    for (size_t d = 0; d < a.microD; ++d)
                                    {
                                        if (mad + mid + d < p.dstC)
                                        {
                                            *(dst++) = Float32ToBFloat16(src[d]);
                                            *(dst++) = mac + c + 1 < p.srcC ? Float32ToBFloat16(src[p.dstC + d]) : 0;
                                        }
                                        else
                                        {
                                            *(dst++) = 0;
                                            *(dst++) = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        void SynetConvolution32fBf16Nhwc::SetBias(const float* bias)
        {
            const ConvParam32f& p = _param;
            const AlgParam& a = _alg;
            _bias.Resize(AlignHiAny(p.dstC, a.microD), true);
            if (bias)
                memcpy(_bias.data, bias, p.dstC * sizeof(float));
        }

        void SynetConvolution32fBf16Nhwc::SetParams(const float* params)
        {
            const ConvParam32f& p = _param;
            const AlgParam& a = _alg;
            if (p.activation == SimdConvolutionActivationLeakyRelu || p.activation == SimdConvolutionActivationPrelu)
                _params.Resize(AlignHiAny(p.dstC, a.microD), true);
            else
                _params.Resize(2, true);
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity:
                _params.data[0] = -FLT_MAX;
                _params.data[1] = FLT_MAX;
                break;
            case SimdConvolutionActivationRelu:
                _params.data[0] = 0;
                _params.data[1] = FLT_MAX;
                break;
            case SimdConvolutionActivationLeakyRelu:
                for (size_t d = 0; d < p.dstC; ++d)
                    _params.data[d] = params[0];
                break;
            case SimdConvolutionActivationRestrictRange:
                _params.data[0] = params[0];
                _params.data[1] = params[1];
                break;
            case SimdConvolutionActivationPrelu:
                for (size_t d = 0; d < p.dstC; ++d)
                    _params.data[d] = params[d];
                break;
            case SimdConvolutionActivationElu:
                _params.data[0] = params[0];
                break;
            case SimdConvolutionActivationHswish:
                _params.data[0] = params[0];
                _params.data[1] = params[1];
                break;
            case SimdConvolutionActivationMish:
                _params.data[0] = params[0];
                break;
            case SimdConvolutionActivationHardSigmoid:
                _params.data[0] = params[0];
                _params.data[1] = params[1];
                break;
            case SimdConvolutionActivationSwish:
                _params.data[0] = params[0];
                break;
            default:
                assert(0);
            }
        }

        void SynetConvolution32fBf16Nhwc::Forward(const float* src, float* buf, float* dst)
        {
            const ConvParam32f& p = _param;
            const AlgParam& a = _alg;
            buf = Buffer(buf);
            for (size_t b = 0; b < p.batch; b += a.batch)
            {
                if(a.mode)
                    ForwardGemm(src, (uint16_t*)buf, dst);
                else
                    ForwardConv(src, (uint16_t*)buf, dst);
                src += p.srcH * p.srcW * p.srcC * a.batch;
                dst += p.dstH * p.dstW * p.dstC * a.batch;
            }
        }

        bool SynetConvolution32fBf16Nhwc::Preferable(const ConvParam32f& p)
        {
            return p.trans != 0 && p.group == 1;
        }

        void SynetConvolution32fBf16Nhwc::ForwardConv(const float* src, uint16_t* buf, float* dst)
        {
            const ConvParam32f& p = _param;
            const AlgParam& a = _alg;
            const uint16_t* weight = _weight.data;
            const float* bias = _bias.data, * params = _params.data;
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                for (size_t sc = 0; sc < p.srcC; sc += a.macroC)
                {
                    size_t macroC = Simd::Min(p.srcC, sc + a.macroC) - sc;
                    for (size_t yBeg = 0; yBeg < p.dstH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + a.macroH, p.dstH);
                        size_t offs = Offset(yBeg, sc, sc + macroC);
                        if (dc == 0)
                            _convert(src + sc, p, yBeg, yEnd, macroC, buf + offs);
                        if (sc + macroC == p.srcC)
                            _convolutions[TermLast](buf + offs, p, macroD, yEnd - yBeg, macroC, macroC == p.srcC ? 1 : 0, 
                                weight, bias, params, dst + yBeg * p.dstW * p.dstC);
                        else
                            _convolutions[TermInterim](buf + offs, p, macroD, yEnd - yBeg, macroC, sc == 0 ? 1 : 0,
                                weight, bias, params, dst + yBeg * p.dstW * p.dstC);
                        yBeg = yEnd;
                    }
                    weight += p.kernelY * p.kernelY * AlignHi(macroC, 2) * AlignHiAny(macroD, a.microD);
                }
                bias += macroD;
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroD;
                dst += macroD;
            }
        }

        void SynetConvolution32fBf16Nhwc::ForwardGemm(const float* src, uint16_t* buf, float* dst)
        {
            const ConvParam32f& p = _param;
            const AlgParam& a = _alg;
            const uint16_t* weight = _weight.data;
            const float* bias = _bias.data, * params = _params.data;
            size_t dstH = p.dstH * a.batch;
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                for (size_t sc = 0; sc < p.srcC; sc += a.macroC)
                {
                    size_t macroC = Simd::Min(p.srcC, sc + a.macroC) - sc;
                    size_t macroK = AlignHi(macroC, 2) * p.kernelY * p.kernelX;
                    for (size_t yBeg = 0; yBeg < dstH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + a.macroH, dstH);
                        size_t offs = Offset(yBeg, sc, sc + AlignHi(macroC, 2));
                        if (dc == 0)
                        {
                            if (a.batch > 1)
                            {
                                size_t dS = p.srcH * p.srcW * p.srcC;
                                size_t dB = p.dstH * p.dstW * macroK;
                                for (size_t b = 0; b < a.batch; ++b)
                                    _convert(src + sc + b * dS, p, 0, p.dstH, macroC, buf + offs + b * dB);
                            }
                            else
                                _convert(src + sc, p, yBeg, yEnd, macroC, buf + offs);
                        }
                        if (sc + macroC == p.srcC)
                            _convolutions[TermLast](buf + offs, p, macroD, yEnd - yBeg, macroK, macroC == p.srcC ? 1 : 0,
                                weight, bias, params, dst + yBeg * p.dstW * p.dstC);
                        else
                            _convolutions[TermInterim](buf + offs, p, macroD, yEnd - yBeg, macroK, sc == 0 ? 1 : 0,
                                weight, bias, params, dst + yBeg * p.dstW * p.dstC);
                        yBeg = yEnd;
                    }
                    weight += AlignHi(macroC, 2) * p.kernelY * p.kernelX * AlignHiAny(macroD, a.microD);
                }
                bias += macroD;
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroD;
                dst += macroD;
            }
        }

        size_t SynetConvolution32fBf16Nhwc::Offset(size_t yBeg, size_t cBeg, size_t cEnd)
        {
            const ConvParam32f& p = _param;
            const AlgParam& a = _alg;
            if (p.dstC <= a.macroD)
                return 0;
            else
            {
                if (a.mode)
                    return p.dstW * (p.dstH * a.batch * cBeg + (cEnd - cBeg) * yBeg);
                else
                    return a.srcW * (a.srcH * cBeg + (cEnd - cBeg) * p.strideY * yBeg);
            }
        }
    }
#endif
}
