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
#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdAlignment.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)

    SynetConvolution16b::SynetConvolution16b(const ConvParam& p)
        : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        , _perf(NULL)
#endif
    {
        _src16b = p.srcT == SimdTensorData16b;
        _dst16b = p.dstT == SimdTensorData16b;
        _elemS = _src16b ? 2 : 4;
        _elemD = _dst16b ? 2 : 4;
        _is1x1 = p.Is1x1();
    }

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
    Base::PerformanceMeasurer * SynetConvolution16b::Perf(const char* func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

    void SynetConvolution16b::SetBias(const float* bias, size_t align)
    {
        const ConvParam& p = _param;
        _bias.Resize(AlignHi(p.dstC, align), true);
        if (bias)
            memcpy(_bias.data, bias, p.dstC * sizeof(float));
    }

    void SynetConvolution16b::SetParams(const float* params, size_t align)
    {
        const ConvParam& p = _param;
        if (p.activation == SimdConvolutionActivationLeakyRelu || p.activation == SimdConvolutionActivationPrelu)
            _params.Resize(AlignHi(p.dstC, align), true);
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
        case SimdConvolutionActivationGelu:
            break;
        default:
            assert(0);
        }
    }

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        SynetConvolution16bGemm::SynetConvolution16bGemm(const ConvParam& p)
            : SynetConvolution16b(p)
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
            _stepS = _sizeS * _elemS;
            _stepD = _sizeD * _elemD;
        }

        size_t SynetConvolution16bGemm::ExternalBufferSize() const
        {
            size_t size = 0;
            if (!_src16b)
                size += _sizeS * sizeof(uint16_t);
            if (!_is1x1)
                size += _sizeB * sizeof(uint16_t);
            if (_dst16b)
                size += _sizeD * sizeof(float);
            return size;
        }

        void SynetConvolution16bGemm::SetParams(const float* weight, const float* bias, const float* params)
        {
            const ConvParam& p = _param;
            Float32ToBFloat16(weight, _weight.size, _weight.data);
            SynetConvolution16b::SetBias(bias, Alignment());
            SynetConvolution16b::SetParams(params, Alignment());
        }

        void SynetConvolution16bGemm::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const ConvParam& p = _param;
            buf = Buffer(buf);
            uint16_t* bufS = _src16b ? (uint16_t*)src : Allocate<uint16_t>(buf, _sizeS);
            uint16_t* bufB = _is1x1 ? bufS : Allocate<uint16_t>(buf, _sizeB);
            float* bufD = _dst16b ? Allocate<float>(buf, _sizeD) : (float*)dst;
            const uint16_t* wgt = _weight.data;
            for (size_t b = 0; b < _batch; ++b)
            {
                if (!_src16b)
                    Float32ToBFloat16((float*)src, _sizeS, bufS);
                if (_param.trans)
                {
                    if(!_is1x1)
                        ImgToRow(bufS, bufB);
                    for (size_t g = 0; g < p.group; ++g)
                        GemmNN(_M, _N, _K, bufB + _grS * g, _ldS, wgt + _grW * g, _ldW, bufD + _grD * g, _ldD);
                }
                else
                {
                    if (!_is1x1)
                        ImgToCol(bufS, bufB);
                    for (size_t g = 0; g < p.group; ++g)
                        GemmNN(_M, _N, _K, wgt + _grW * g, _ldW, bufB + _grS * g, _ldS, bufD + _grD * g, _ldD);
                }
                ConvolutionBiasAndActivation(_bias.data, p.dstC, p.dstH * p.dstW, p.activation, _params.data, p.trans, bufD);
                if(_dst16b)
                    Float32ToBFloat16(bufD, _sizeD, (uint16_t*)dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        void SynetConvolution16bGemm::ImgToCol(const uint16_t* src, uint16_t* dst)
        {
            const ConvParam& p = _param;
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
                                        *(dst++) = src[sy * p.srcW + sx];
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

        void SynetConvolution16bGemm::ImgToRow(const uint16_t* src, uint16_t* dst)
        {
            const ConvParam& p = _param;
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
                                        memcpy(dst, src + (sy * p.srcW + sx) * p.srcC, size * sizeof(uint16_t));
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

        void SynetConvolution16bGemm::GemmNN(size_t M, size_t N, size_t K, const uint16_t* A, size_t lda, const uint16_t* B, size_t ldb, float* C, size_t ldc)
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

        //-------------------------------------------------------------------------------------------------

        SynetConvolution16bNhwcGemm::SynetConvolution16bNhwcGemm(const ConvParam& p)
            : SynetConvolution16b(p)
        {
            _convert = 0;
            _convolutions[0] = 0;
            _convolutions[1] = 0;
        }

        String SynetConvolution16bNhwcGemm::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::NhwcGemm";
            if (_alg.batch > 1)
                desc << "-" << _alg.batch;
            return desc.str();
        }

        void SynetConvolution16bNhwcGemm::SetAlgParam(size_t microD, size_t microM, size_t microK, size_t L1, size_t L2, size_t L3)
        {
            const ConvParam& p = _param;
            AlgParam& a = _alg;

            a.M = p.dstW * p.dstH;
            a.K = p.srcC * p.kernelY * p.kernelX;
            a.microD = microD;
            a.microM = microM;
            a.microK = microK;
            a.bufD = AlignHiAny(p.dstC, a.microD);
            a.bufK = AlignHi(a.K, a.microK);
            a.macroK = Simd::RestrictRange(AlignLo(L1 / a.microD / 2, a.microK), a.microK, a.bufK);
            a.batch = 1;
            size_t bufSize = a.M * a.bufK * 2;
            if (bufSize * 2 <= L2 && p.batch > 1)
            {
                for (size_t batch = 1; batch <= p.batch; ++batch)
                    if (p.batch % batch == 0 && batch * bufSize <= L2)
                        a.batch = batch;
            }
            a.bufM = a.batch * a.M;
            a.macroH = Simd::RestrictRange(L2 / a.macroK / p.dstW / 2, size_t(1), p.dstH * a.batch);
            a.macroD = Simd::RestrictRange(AlignLoAny(L3 / a.macroK / 2, a.microD), a.microD, a.bufD);
            _stepS = p.srcH * p.srcW * p.srcC * a.batch * _elemS;
            _stepD = p.dstH * p.dstW * p.dstC * a.batch * _elemD;
        }

        size_t SynetConvolution16bNhwcGemm::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = (a.bufM + 1) * a.bufK * sizeof(uint16_t);
            if (_dst16b && a.macroK < a.K)
                size += a.macroD * a.bufM * sizeof(float);
            return size;
        }

        void SynetConvolution16bNhwcGemm::SetParams(const float* weight, const float* bias, const float* params)
        {
            SetWeight(weight);
            SynetConvolution16b::SetBias(bias, _alg.microD);
            SynetConvolution16b::SetParams(params, _alg.microD);
        }

        void SynetConvolution16bNhwcGemm::SetWeight(const float* weight)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            Array16u buffer(a.bufD * a.bufK, true);
            uint16_t* buf = buffer.data;
            for (size_t k = 0; k < a.K; k += 2)
            {
                for (size_t d = 0; d < p.dstC; ++d)
                {
                    *(buf++) = Float32ToBFloat16(weight[d]);
                    *(buf++) = k + 1 < a.K ? Float32ToBFloat16(weight[d + p.dstC]) : 0;
                }
                buf += 2 * (a.bufD - p.dstC);
                weight += 2 * p.dstC;
            }
            _weight.Resize(a.bufK * a.bufD, true);
            size_t bufK = a.bufK / 2, macK = a.macroK / 2, bufD = a.bufD * 2, macD = a.macroD * 2, micD = a.microD * 2;
            const uint16_t* src = buffer.data;
            uint16_t* dst = _weight.data;
            for (size_t mad = 0; mad < bufD; mad += macD)
            {
                size_t macroD = Simd::Min(bufD, mad + macD) - mad;
                for (size_t mak = 0; mak < bufK; mak += macK)
                {
                    size_t macroK = Simd::Min(bufK, mak + macK) - mak;
                    for (size_t mid = 0; mid < macroD; mid += micD)
                    {
                        for (size_t k = 0; k < macroK; ++k)
                        {
                            memcpy(dst, src + (mak + k) * bufD + mad + mid, micD * 2);
                            dst += micD;
                        }
                    }
                }
            }
        }

        void SynetConvolution16bNhwcGemm::Forward(const uint8_t* src, uint8_t* buf8, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            buf8 = Buffer(buf8);
            uint16_t* buf = Allocate<uint16_t>(buf8, (a.bufM + 1) * a.bufK);
            float* sum = _dst16b && a.macroK < a.K ? Allocate<float>(buf8, a.macroD * a.bufM) : NULL;
            for (size_t b = 0; b < p.batch; b += a.batch)
            {
                Forward(src, buf, sum, dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        void SynetConvolution16bNhwcGemm::Forward(const uint8_t* src, uint16_t* buf, float* sum, uint8_t* dst)
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            const uint16_t* weight = _weight.data;
            const float* bias = _bias.data, * params = _params.data;
            size_t dstH = p.dstH * a.batch;
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                for (size_t mak = 0; mak < a.K; mak += a.macroK)
                {
                    size_t macroK = Simd::Min(a.bufK, mak + a.macroK) - mak;
                    for (size_t yBeg = 0; yBeg < dstH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + a.macroH, dstH);
                        size_t bufOffs = a.macroK < a.bufK ? mak * a.bufM + yBeg * p.dstW * macroK : 0;
                        size_t sumOffs = yBeg * p.dstW * a.macroD;
                        size_t dstOffs = yBeg * p.dstW * p.dstC * _elemD;
                        if (dc == 0 && mak == 0)
                        {
                            if (a.batch > 1)
                            {
                                size_t dS = p.srcH * p.srcW * p.srcC * _elemS;
                                for (size_t b = 0; b < a.batch; ++b)
                                    _convert(src + b * dS, p, a, b, 0, p.dstH, buf);
                            }
                            else
                                _convert(src, p, a, 0, yBeg, yEnd, buf);
                        }
                        if (mak + macroK == a.bufK)
                            _convolutions[TermLast](buf + bufOffs, p, macroD, yEnd - yBeg, macroK, macroK == a.bufK ? 1 : 0,
                                weight, bias, params, sum + sumOffs, dst + dstOffs);
                        else
                            _convolutions[TermInterim](buf + bufOffs, p, macroD, yEnd - yBeg, macroK, mak == 0 ? 1 : 0,
                                weight, bias, params, sum + sumOffs, dst + dstOffs);
                        yBeg = yEnd;
                    }
                    weight += AlignHi(macroK, a.microK) * AlignHiAny(macroD, a.microD);
                }
                bias += macroD;
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroD;
                dst += macroD * _elemD;
            }
        }

        bool SynetConvolution16bNhwcGemm::Preferable(const ConvParam& p)
        {
            return p.trans != 0 && p.group == 1;
        }

        //-------------------------------------------------------------------------------------------------

        void * SynetConvolution16bInit(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility)
        {
            ConvParam param(batch, conv, compatibility);
            if (!param.Valid(SimdTensorData32f, SimdTensorData16b))
                return NULL;
            return new SynetConvolution16bGemm(param);
        }
    }
#endif
}
