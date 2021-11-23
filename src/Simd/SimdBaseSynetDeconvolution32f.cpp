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
#include "Simd/SimdSynetDeconvolution32f.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
    Base::PerformanceMeasurer * SynetDeconvolution32f::Perf(const String& func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

    namespace Base
    {
        SynetDeconvolution32fGemmNN::SynetDeconvolution32fGemmNN(const DeconvParam32f & p)
            : SynetDeconvolution32f(p)
        {
            _is1x1 = p.Is1x1();
            if (p.trans)
            {
                assert(p.group == 1);

                _M = p.srcH * p.srcW;
                _N = p.kernelY * p.kernelX * p.dstC;
                _K = p.srcC;
                _ldS = _K;
                _ldW = _N;
                _ldD = _N;
                _grW = 0;
                _grS = 0;
                _grD = 0;
            }
            else
            {
                _M = p.kernelY * p.kernelX * p.dstC / p.group;
                _N = p.srcH * p.srcW;
                _K = p.srcC / p.group;
                _ldW = _K;
                _ldS = _N;
                _ldD = _N;
                _grW = _M * _K;
                _grS = _K * _N;
                _grD = _M * _N;
            }
            _batch = p.batch;
            _sizeS = p.srcC*p.srcH*p.srcW;
            _sizeB = p.dstC*p.kernelY*p.kernelX*p.srcH*p.srcW;
            _sizeD = p.dstC*p.dstH*p.dstW;
            _merge = 1;
            if (p.trans)
            {
                if (p.group == 1 && _batch > 1)
                {
                    for (size_t merge = 1; merge <= _batch; ++merge)
                        if (_batch%merge == 0 && _M*merge <= 256)
                            _merge = merge;
                }
            }
            else
                _weightT.Resize(p.srcC * p.kernelY * p.kernelX * p.dstC / p.group);
            _gemm.Init(InitGemmFuncs(Base::Gemm32fNN, "Base", p.gemm, "Ext"));
            _biasAndActivation = Base::ConvolutionBiasAndActivation;
        }

        size_t SynetDeconvolution32fGemmNN::ExternalBufferSize() const
        {
            if (_is1x1)
                return 1;
            else
                return _sizeB*_merge;
        };

        void SynetDeconvolution32fGemmNN::SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params)
        {
            Simd::SynetDeconvolution32f::SetParams(weight, internal, bias, params);
            if (_nhwcWeight.data)
            {
                if (_gemmCb.Size())
                    _gemmCb.At(0).ReorderB(_M*_merge, _N, _K, weight, _nhwcWeight.data);
                else
                    _nhwcReorderB(_M*_merge, _N, _K, weight, _nhwcWeight.data, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                if (internal)
                    *internal = SimdTrue;
            }
            if (_weightT.data)
            {
                const float * src = weight;
                float * dst = _weightT.data;
                for (size_t g = 0; g < _param.group; ++g)
                {
                    for (size_t i = 0; i < _M; ++i)
                        for (size_t k = 0; k < _K; ++k)
                            dst[i * _K + k] = src[k * _M + i];
                    src += _grW;
                    dst += _grW;
                }
                if (internal)
                    *internal = SimdTrue;
            }
        }

        void SynetDeconvolution32fGemmNN::Forward(const float * src, float * buf, float * dst)
        {
            const DeconvParam32f & p = _param;
            if (!_is1x1)
                buf = Buffer(buf);
            if (_merge > 1)
            {
                for (size_t b = 0; b < _batch; b += _merge)
                {
                    float * tmp = _is1x1 ? dst : buf;
                    if (_nhwcWeight.data)
                    {
                        if (_gemmCb.Size())
                            _gemmCb.Run(GemmCbArgs(_M*_merge, _N, _K, src, _nhwcWeight.data, tmp));
                        else
                            _nhwcRun(_M*_merge, _N, _K, src, _nhwcWeight.data, tmp, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                    }
                    else
                        _gemm.Run(GemmArgs(_M*_merge, _N, _K, &_1, src, _ldS, _weight, _ldW, &_0, tmp, _ldD));
                    if (!_is1x1)
                    {
                        for (size_t m = 0; m < _merge; ++m)
                            RowToImg(tmp + m * _sizeS, dst + m * _sizeB);
                    }                    
                    for (size_t m = 0; m < _merge; ++m)
                        _biasAndActivation(_bias, p.dstC, p.dstH*p.dstW, p.activation, _params, p.trans, dst + m * _sizeD);
                    src += _sizeS * _merge;
                    dst += _sizeD * _merge;
                }
            }
            else
            {
                for (size_t b = 0; b < _batch; ++b)
                {
                    float * tmp = _is1x1 ? dst : buf;
                    for (size_t g = 0; g < p.group; ++g)
                    {
                        if (p.trans)
                        {
                            if (_nhwcWeight.data)
                            {
                                if (_gemmCb.Size())
                                    _gemmCb.Run(GemmCbArgs(_M, _N, _K, src, _nhwcWeight.data, tmp));
                                else
                                    _nhwcRun(_M, _N, _K, src, _nhwcWeight.data, tmp, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                            }
                            else
                                _gemm.Run(GemmArgs(_M, _N, _K, &_1, src + _grS * g, _ldS, _weight + _grW * g, _ldW, &_0, tmp + _grD * g, _ldD));
                        }
                        else
                            _gemm.Run(GemmArgs(_M, _N, _K, &_1, _weightT.data + _grW * g, _ldW, src + _grS * g, _ldS, &_0, tmp + _grD * g, _ldD));
                    }
                    if (!_is1x1)
                    {
                        if (_param.trans)
                            RowToImg(tmp, dst);
                        else
                            ColToImg(tmp, dst);
                    }                    
                    _biasAndActivation(_bias, p.dstC, p.dstH*p.dstW, p.activation, _params, p.trans, dst);
                    src += _sizeS;
                    dst += _sizeD;
                }
            }
        }

        void SynetDeconvolution32fGemmNN::ColToImg(const float * src, float * dst)
        {
            const DeconvParam32f & p = _param;
            assert(!p.trans);
            size_t dstSize = p.dstW * p.dstH;
            for (size_t cd = 0; cd < p.dstC; ++cd)
            {
                memset(dst, 0, dstSize * sizeof(float));
                for (size_t ky = 0; ky < p.kernelY; ++ky)
                {
                    for (size_t kx = 0; kx < p.kernelX; ++kx)
                    {
                        size_t dy = ky * p.dilationY - p.padY;
                        for (size_t sy = 0; sy < p.srcH; ++sy, dy += p.strideY)
                        {
                            if (dy < p.dstH)
                            {
                                size_t dx = kx * p.dilationX - p.padX;
                                for (size_t sx = 0; sx < p.srcW; ++sx, dx += p.strideX)
                                {
                                    if (dx < p.dstW)
                                        dst[dy * p.dstW + dx] += *src;
                                    src++;
                                }
                            }
                            else
                                src += p.srcW;
                        }
                    }
                }
                dst += dstSize;
            }
        }

        void SynetDeconvolution32fGemmNN::RowToImg(const float * src, float * dst)
        {
            const DeconvParam32f & p = _param;
            assert(p.trans && p.group == 1);
            if (p.IsPad(0) && p.IsDilation(1) && p.kernelY == p.strideX && p.kernelX == p.strideX)
            {
                for (size_t sy = 0; sy < p.srcH; ++sy)
                {
                    for (size_t sx = 0; sx < p.srcW; ++sx)
                    {
                        size_t dy = sy * p.strideY;
                        for (size_t ky = 0; ky < p.kernelY; ky++, dy += 1)
                        {
                            size_t dx = sx * p.strideX;
                            for (size_t kx = 0; kx < p.kernelX; kx++, dx += 1)
                            {
                                memcpy(dst + (dy * p.dstW + dx)*p.dstC, src, p.dstC * sizeof(float));
                                src += p.dstC;
                            }
                        }
                    }
                }
            }
            else
            {
                for (size_t dy = 0; dy < p.dstH; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        memset(dst + (dy*p.dstW + dx)*p.dstC, 0, p.dstC * sizeof(float));
                for (size_t sy = 0; sy < p.srcH; ++sy)
                {
                    for (size_t sx = 0; sx < p.srcW; ++sx)
                    {
                        size_t dy = sy * p.strideY - p.padY;
                        for (size_t ky = 0; ky < p.kernelY; ky++, dy += p.dilationY)
                        {
                            if (dy < p.dstH)
                            {
                                size_t dx = sx * p.strideX - p.padX;
                                for (size_t kx = 0; kx < p.kernelX; kx++, dx += p.dilationX)
                                {
                                    if (dx < p.dstW)
                                    {
                                        float * d = dst + (dy * p.dstW + dx)*p.dstC;
                                        for (size_t dc = 0; dc < p.dstC; ++dc)
                                            d[dc] += src[dc];
                                    }
                                    src += p.dstC;
                                }
                            }
                            else
                                src += p.kernelX * p.dstC;
                        }
                    }
                }
            }
        }

        //---------------------------------------------------------------------

        SynetDeconvolution32fNhwcDirect2x2::SynetDeconvolution32fNhwcDirect2x2(const DeconvParam32f & p)
            : SynetDeconvolution32f(p)
        {
            _sizeS = p.srcC*p.srcH*p.srcW;
            _sizeD = p.dstC*p.dstH*p.dstW;
            _deconvolution = NULL;
        }

        void SynetDeconvolution32fNhwcDirect2x2::SetAlgParam(size_t F, size_t L1, size_t L2, size_t L3)
        {
            const DeconvParam32f & p = _param;
            _alg.microD = F;
            _alg.macroC = Simd::Min(L1 / sizeof(float) / p.kernelX / _alg.microD, p.srcC);
            _alg.macroH = Simd::Min(L2 / sizeof(float) / _alg.macroC / p.srcW, p.srcH);
            _alg.macroD = Simd::Min(AlignLoAny(L3 / sizeof(float) / p.kernelY / _alg.macroC, _alg.microD), AlignHiAny(p.dstC, _alg.microD));
            _rWeight.Resize(AlignHiAny(p.dstC, _alg.microD) * p.kernelY * p.kernelX * p.srcC);
            _rBias.Resize(AlignHiAny(p.dstC, _alg.microD), true);
            if (p.activation == SimdConvolutionActivationLeakyRelu || p.activation == SimdConvolutionActivationPrelu)
                _rParams.Resize(AlignHiAny(p.dstC, _alg.microD), true);
            else
                _rParams.Resize(2, true);
        }

        void SynetDeconvolution32fNhwcDirect2x2::ReorderWeight(const float * src, float * dst)
        {
            const DeconvParam32f & p = _param;
            const AlgParam & a = _alg;
            for (size_t da = 0; da < p.dstC; da += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, da + a.macroD) - da;
                for (size_t sa = 0; sa < p.srcC; sa += a.macroC)
                {
                    size_t macroC = Simd::Min(p.srcC, sa + a.macroC) - sa;
                    for (size_t di = 0; di < macroD; di += a.microD)
                    {
                        size_t microD = Simd::Min(macroD, di + a.microD) - di;
                        for (size_t ky = 0; ky < p.kernelY; ky++)
                        {
                            for (size_t kx = 0; kx < p.kernelX; kx++)
                            {
                                for (size_t si = 0; si < macroC; si++)
                                {
                                    const float * s = src + (((sa + si)*p.kernelY + ky) * p.kernelX + kx) * p.dstC + da + di;
                                    size_t i = 0;
                                    for (; i < microD; i++)
                                        dst[i] = s[i];
                                    for (; i < a.microD; i++)
                                        dst[i] = 0;
                                    dst += a.microD;
                                }
                            }
                        }
                    }
                }
            }
        }

        size_t SynetDeconvolution32fNhwcDirect2x2::InternalBufferSize() const
        {
            return _buffer.size + _rWeight.size + _rBias.size + _rParams.size;
        }

        void SynetDeconvolution32fNhwcDirect2x2::SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params)
        {
            SynetDeconvolution32f::SetParams(weight, internal, bias, params);
            if (_rWeight.data)
            {
                const DeconvParam32f & p = _param;
                ReorderWeight(weight, _rWeight.data);
                _weight = _rWeight.data;
                if (internal)
                    *internal = SimdTrue;
            }
            if (_rBias.data)
            {
                if (bias)
                    memcpy(_rBias.data, bias, _param.dstC * sizeof(float));
                _bias = _rBias.data;
            }
            if (_rParams.data)
            {
                const DeconvParam32f& p = _param;
                switch (p.activation)
                {
                case SimdConvolutionActivationIdentity:
                    _rParams.data[0] = -FLT_MAX;
                    _rParams.data[1] = FLT_MAX;
                    break;
                case SimdConvolutionActivationRelu:
                    _rParams.data[0] = 0;
                    _rParams.data[1] = FLT_MAX;
                    break;
                case SimdConvolutionActivationLeakyRelu:
                    for (size_t d = 0; d < p.dstC; ++d)
                        _rParams.data[d] = params[0];
                    break;
                case SimdConvolutionActivationRestrictRange:
                    _rParams.data[0] = params[0];
                    _rParams.data[1] = params[1];
                    break;
                case SimdConvolutionActivationPrelu:
                    for (size_t d = 0; d < p.dstC; ++d)
                        _rParams.data[d] = params[d];
                    break;
                case SimdConvolutionActivationElu:
                    _rParams.data[0] = params[0];
                    break;
                case SimdConvolutionActivationHswish:
                    _rParams.data[0] = params[0];
                    _rParams.data[1] = params[1];
                    break;
                case SimdConvolutionActivationMish:
                    _rParams.data[0] = params[0];
                    break;
                case SimdConvolutionActivationHardSigmoid:
                    _rParams.data[0] = params[0];
                    _rParams.data[1] = params[1];
                    break;
                case SimdConvolutionActivationSwish:
                    _rParams.data[0] = params[0];
                    break;
                default:
                    assert(0);
                }
                _params = _rParams.data;
            }
        }

        void SynetDeconvolution32fNhwcDirect2x2::Forward(const float * src, float * buf, float * dst)
        {
            const DeconvParam32f & p = _param;
            for (size_t b = 0; b < p.batch; ++b)
            {
                _deconvolution(src, _param, _alg, _weight, _bias, _params, dst);
                src += _sizeS;
                dst += _sizeD;
            }
        }

        bool SynetDeconvolution32fNhwcDirect2x2::Preferable(const DeconvParam32f & p)
        {
            return false;
        }

        //---------------------------------------------------------------------

        void * SynetDeconvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm)
        {
            DeconvParam32f param(batch, conv, gemm);
            if (!param.Valid())
                return NULL;
            if (SynetDeconvolution32fNhwcDirect2x2::Preferable(param))
                return new SynetDeconvolution32fNhwcDirect2x2(param);
            else
                return new SynetDeconvolution32fGemmNN(param);
        }
    }
#endif
}
