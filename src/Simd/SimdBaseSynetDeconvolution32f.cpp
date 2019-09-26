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
#include "Simd/SimdSynetDeconvolution32f.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"

namespace Simd
{
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
                            dst[k * _M + i] = src[i * _K + k];
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

        //---------------------------------------------------------------------

        void * SynetDeconvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm)
        {
            DeconvParam32f param(batch, conv, gemm);
            if (!param.Valid())
                return NULL;
            //if (SynetConvolution32fDirectNhwc::Preferable(param))
            //    return new SynetConvolution32fDirectNhwc(param);
            //else
                return new SynetDeconvolution32fGemmNN(param);
        }
    }
}
