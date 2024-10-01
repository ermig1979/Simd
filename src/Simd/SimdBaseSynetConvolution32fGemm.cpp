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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, SimdBool trans, float * dst)
        {
            if (activation == ::SimdConvolutionActivationIdentity)
            {
                if(bias)
                    SynetAddBias(bias, count, size, dst, (SimdTensorFormatType)trans);
            }
            else if (activation == ::SimdConvolutionActivationRelu)
            {
                if (bias)
                {
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            for (size_t i = 0; i < count; ++i)
                                dst[i] = Simd::Max(0.0f, dst[i] + bias[i]);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            for (size_t j = 0; j < size; ++j)
                                dst[j] = Simd::Max(0.0f, dst[j] + bias[i]);
                            dst += size;
                        }
                    }
                }
                else
                {
                    float slope = 0;
                    SynetRelu32f(dst, size*count, &slope, dst);
                }
            }
            else if (activation == ::SimdConvolutionActivationLeakyRelu)
            {
                float slope = params[0];
                if (bias)
                {
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            for (size_t i = 0; i < count; ++i)
                                dst[i] = SynetRelu32f(dst[i] + bias[i], slope);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            for (size_t j = 0; j < size; ++j)
                                dst[j] = SynetRelu32f(dst[j] + bias[i], slope);
                            dst += size;
                        }
                    }
                }
                else
                    SynetRelu32f(dst, size*count, &slope, dst);
            }
            else if (activation == ::SimdConvolutionActivationRestrictRange)
            {
                float lower = params[0];
                float upper = params[1];
                if (bias)
                {
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            for (size_t i = 0; i < count; ++i)
                                dst[i] = Simd::RestrictRange(dst[i] + bias[i], lower, upper);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            for (size_t j = 0; j < size; ++j)
                                dst[j] = Simd::RestrictRange(dst[j] + bias[i], lower, upper);
                            dst += size;
                        }
                    }
                }
                else
                    SynetRestrictRange32f(dst, size*count, &lower, &upper, dst);
            }
            else if (activation == ::SimdConvolutionActivationPrelu)
            {
                if (bias)
                {
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            for (size_t i = 0; i < count; ++i)
                                dst[i] = SynetRelu32f(dst[i] + bias[i], params[i]);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            for (size_t j = 0; j < size; ++j)
                                dst[j] = SynetRelu32f(dst[j] + bias[i], params[i]);
                            dst += size;
                        }
                    }
                }
                else
                    Base::SynetPreluLayerForward(dst, params, count, size, dst, (SimdTensorFormatType)trans);
            }
            else if (activation == ::SimdConvolutionActivationElu)
            {
                float alpha = params[0];
                if (bias)
                {
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            for (size_t i = 0; i < count; ++i)
                                dst[i] = SynetElu32f(dst[i] + bias[i], alpha);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            for (size_t j = 0; j < size; ++j)
                                dst[j] = SynetElu32f(dst[j] + bias[i], alpha);
                            dst += size;
                        }
                    }
                }
                else
                    SynetElu32f(dst, size*count, &alpha, dst);
            }
            else if (activation == ::SimdConvolutionActivationHswish)
            {
                float shift = params[0];
                float scale = params[1];
                if (bias)
                {
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            for (size_t i = 0; i < count; ++i)
                                dst[i] = SynetHswish32f(dst[i] + bias[i], shift, scale);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            for (size_t j = 0; j < size; ++j)
                                dst[j] = SynetHswish32f(dst[j] + bias[i], shift, scale);
                            dst += size;
                        }
                    }
                }
                else
                    SynetHswish32f(dst, size*count, &shift, &scale, dst);
            }
            else if (activation == ::SimdConvolutionActivationMish)
            {
                float threshold = params[0];
                if (bias)
                {
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            for (size_t i = 0; i < count; ++i)
                                dst[i] = SynetMish32f(dst[i] + bias[i], threshold);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            for (size_t j = 0; j < size; ++j)
                                dst[j] = SynetMish32f(dst[j] + bias[i], threshold);
                            dst += size;
                        }
                    }
                }
                else
                    SynetMish32f(dst, size * count, &threshold, dst);
            }
            else if (activation == ::SimdConvolutionActivationHardSigmoid)
            {
                float scale = params[0];
                float shift = params[1];
                if (bias)
                {
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            for (size_t i = 0; i < count; ++i)
                                dst[i] = SynetHardSigmoid32f(dst[i] + bias[i], scale, shift);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            for (size_t j = 0; j < size; ++j)
                                dst[j] = SynetHardSigmoid32f(dst[j] + bias[i], scale, shift);
                            dst += size;
                        }
                    }
                }
                else
                    SynetHardSigmoid32f(dst, size * count, &scale, &shift, dst);
            }
            else if(activation == ::SimdConvolutionActivationSwish)
            {
                float slope = params[0];
                if (bias)
                {
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            for (size_t i = 0; i < count; ++i)
                                dst[i] = SynetSwish32f(dst[i] + bias[i], slope);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            for (size_t j = 0; j < size; ++j)
                                dst[j] = SynetSwish32f(dst[j] + bias[i], slope);
                            dst += size;
                        }
                    }
                }
                else
                    SynetSwish32f(dst, size * count, &slope, dst);
            }
            else if (activation == ::SimdConvolutionActivationGelu)
            {
                if (bias)
                {
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            for (size_t i = 0; i < count; ++i)
                                dst[i] = Gelu(dst[i] + bias[i]);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            for (size_t j = 0; j < size; ++j)
                                dst[j] = Gelu(dst[j] + bias[i]);
                            dst += size;
                        }
                    }
                }
                else
                    SynetGelu32f(dst, size * count, dst);
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fGemmNN::SynetConvolution32fGemmNN(const ConvParam & p)
            : SynetConvolution32f(p)
        {
            if (p.IsDilation(1) && p.IsStride(1) && p.IsPad(0))
            {
                _skipConv = p.IsKernel(1) || (p.srcH == p.kernelY && p.srcW == p.kernelX);
            }
            else
                _skipConv = false;
            if (p.trans)
            {
                _M = p.dstH * p.dstW;
                _N = p.dstC / p.group;
                _K = p.srcC * p.kernelY * p.kernelX / p.group;
                _ldS = _K * (p.Is1x1() ? p.group : 1);
                _ldW = p.dstC;
                _ldD = p.dstC;
                _grW = _N;
                _grS = _K * (p.Is1x1() ? 1 : _M);
                _grD = _N;
            }
            else
            {
                _M = p.dstC / p.group;
                _N = p.dstH * p.dstW;
                _K = p.srcC *  p.kernelY *  p.kernelX / p.group;
                _ldW = _K;
                _ldS = _N;
                _ldD = _N;
                _grW = _M * _K;
                _grS = _K * _N;
                _grD = _M * _N;
            }
            _batch = p.batch;
            _sizeS = p.srcC*p.srcH*p.srcW;
            _sizeB = p.srcC*p.kernelY*p.kernelX*p.dstH*p.dstW;
            _sizeD = p.dstC*p.dstH*p.dstW;
            _merge = 1;
            if (p.trans && p.group == 1 && _batch > 1)
            {
                for (size_t merge = 1; merge <= _batch; ++merge)
                    if (_batch%merge == 0 && _M*merge*_K*sizeof(float) <= Base::AlgCacheL2())
                        _merge = merge;
            }
            _gemm.Init(InitGemmFuncs(Base::Gemm32fNN, "Base"));
            _biasAndActivation = Base::ConvolutionBiasAndActivation;
        }

        size_t SynetConvolution32fGemmNN::ExternalBufferSize() const
        {
            if (_skipConv)
                return 1;
            else
                return _sizeB*_merge;
        };

        void SynetConvolution32fGemmNN::SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params)
        {
            Simd::SynetConvolution32f::SetParams(weight, internal, bias, params);
            if (_nhwcWeight.data)
            {
                if (_gemmCb.Size())
                    _gemmCb.At(0).ReorderB(_M*_merge, _N, _K, weight, _nhwcWeight.data);
                else
                    _nhwcReorderB(_M*_merge, _N, _K, weight, _nhwcWeight.data, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                if (internal)
                    *internal = SimdTrue;
            }
        }

        void SynetConvolution32fGemmNN::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam & p = _param;
            if (!_skipConv)
                buf = Buffer(buf);
            if (_merge > 1)
            {
                for (size_t b = 0; b < _batch; b += _merge)
                {
                    const float * tmp = src;
                    if (!_skipConv)
                    {
                        for (size_t m = 0; m < _merge; ++m)
                            ImgToRow(src + m * _sizeS, buf + m * _sizeB);
                        tmp = buf;
                    }
                    if (_nhwcWeight.data)
                    {
                        if (_gemmCb.Size())
                            _gemmCb.Run(GemmCbArgs(_M*_merge, _N, _K, tmp, _nhwcWeight.data, dst));
                        else
                            _nhwcRun(_M*_merge, _N, _K, tmp, _nhwcWeight.data, dst, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                    }
                    else
                        _gemm.Run(GemmArgs(_M*_merge, _N, _K, &_1, tmp, _ldS, _weight, _ldW, &_0, dst, _ldD));
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
                    const float * tmp = src;
                    if (!_skipConv)
                    {
                        if (_param.trans)
                            ImgToRow(src, buf);
                        else
                            ImgToCol(src, buf);
                        tmp = buf;
                    }
                    for (size_t g = 0; g < p.group; ++g)
                    {
                        if (p.trans)
                        {
                            if (_nhwcWeight.data)
                            {
                                if (_gemmCb.Size())
                                    _gemmCb.Run(GemmCbArgs(_M, _N, _K, tmp, _nhwcWeight.data, dst));
                                else
                                    _nhwcRun(_M, _N, _K, tmp, _nhwcWeight.data, dst, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                            }
                            else
                                _gemm.Run(GemmArgs(_M, _N, _K, &_1, tmp + _grS * g, _ldS, _weight + _grW * g, _ldW, &_0, dst + _grD * g, _ldD));
                        }
                        else
                            _gemm.Run(GemmArgs(_M, _N, _K, &_1, _weight + _grW * g, _ldW, tmp + _grS * g, _ldS, &_0, dst + _grD * g, _ldD));
                    }
                    _biasAndActivation(_bias, p.dstC, p.dstH*p.dstW, p.activation, _params, p.trans, dst);
                    src += _sizeS;
                    dst += _sizeD;
                }
            }
        }

        void SynetConvolution32fGemmNN::ImgToCol(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            assert(!p.trans);
            size_t srcSize = p.srcW * p.srcH;
            if (p.IsDilation(1) && p.IsStride(2) && p.IsPad(0) && p.IsKernel(1))
            {
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t dy = 0; dy < p.dstH; ++dy)
                    {
                        const float * psrc = src + 2 * dy*p.srcW;
                        for (size_t dx = 0, sx = 0; dx < p.dstW; ++dx, sx += 2)
                            *(dst++) = psrc[sx];
                    }
                    src += srcSize;
                }
            }
            else if (p.IsDilation(1) && p.IsStride(1))
            {
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t ky = 0; ky < p.kernelY; ++ky)
                    {
                        for (size_t kx = 0; kx < p.kernelX; ++kx)
                        {
                            size_t sy = ky - p.padY;
                            for (size_t dy = 0; dy < p.dstH; ++dy, ++sy)
                            {
                                if (sy < p.srcH)
                                {
                                    size_t sx = kx - p.padX, dx = 0;
                                    for (size_t dx = 0; dx < p.dstW; ++dx, ++sx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = src[sy * p.srcW + sx];
                                        else
                                            *(dst++) = 0;
                                    }
                                }
                                else
                                {
                                    for (size_t dx = 0; dx < p.dstW; ++dx)
                                        *(dst++) = 0;
                                }
                            }
                        }
                    }
                    src += srcSize;
                }
            }
            else
            {
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
        }

        void SynetConvolution32fGemmNN::ImgToRow(const float * src, float * dst)
        {
            const ConvParam & p = _param;
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
                                        memcpy(dst, src + (sy * p.srcW + sx)*p.srcC, size * sizeof(float));
                                        dst += size;
                                    }
                                    else
                                    {
                                        memset(dst, 0, size * sizeof(float));
                                        dst += size;
                                    }
                                }
                            }
                            else
                            {
                                memset(dst, 0, p.kernelX * size * sizeof(float));
                                dst += p.kernelX * size;
                            }
                        }
                    }
                }
                src += size;
            }
        }

        bool SynetConvolution32fGemmNN::GemmRuntime() const
        {
            return NHWC_GEMM_RUNTIME && _param.SizeW() * sizeof(float) < Base::AlgCacheL3() * 1.0f;
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fGemmNT::SynetConvolution32fGemmNT(const ConvParam & p)
            : SynetConvolution32f(p)
        {
            assert(p.group == 1);
            if (p.trans)
                assert(p.dstC == 1 && p.Is1x1());
            _M = p.dstC;
            _N = p.dstH * p.dstW;
            _K = p.srcC * p.kernelY * p.kernelX; 
            _batch = p.batch;
            _sizeS = p.srcC*p.srcH*p.srcW;
            _sizeB = p.srcC*p.kernelY*p.kernelX*p.dstH*p.dstW;
            _sizeD = p.dstC*p.dstH*p.dstW;
            _gemm.Init(InitGemmFuncs(Base::Gemm32fNT, "Base"));
            _biasAndActivation = Base::ConvolutionBiasAndActivation;
        }

        size_t SynetConvolution32fGemmNT::ExternalBufferSize() const
        {
            return _param.trans ? 1 : _sizeB;
        };

        void SynetConvolution32fGemmNT::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam& p = _param;
            if (p.trans == 0)
                buf = Buffer(buf);
            for (size_t b = 0; b < _batch; ++b)
            {
                if (p.trans)
                {
                    _gemm.Run(GemmArgs(_M, _N, _K, &_1, _weight, _K, src, _K, &_0, dst, _N));
                    _biasAndActivation(_bias, 1, p.dstH * p.dstW, p.activation, _params, SimdFalse, dst);
                }
                else
                {
                    ImgToRow(src, _param, buf);
                    _gemm.Run(GemmArgs(_M, _N, _K, &_1, _weight, _K, buf, _K, &_0, dst, _N));
                    _biasAndActivation(_bias, p.dstC, p.dstH * p.dstW, p.activation, _params, SimdFalse, dst);
                }
                src += _sizeS;
                dst += _sizeD;
            }
        }

        bool SynetConvolution32fGemmNT::Preferable(const ConvParam & p)
        {
            if (p.group != 1)
                return false;
            if (p.trans)
                return p.Is1x1() && p.dstC == 1;
            else
                return p.srcH < 6 && p.srcW < 6;
        }

        void SynetConvolution32fGemmNT::ImgToRow(const float * src, const ConvParam & p, float * dst)
        {
            const size_t K = p.kernelX * p.kernelY*p.srcC, N = p.dstH * p.dstW;
            if (p.IsDilation(1) && p.IsStride(1))
            {
                if (p.IsKernel(1))
                {
                    for (size_t i = 0; i < N; ++i)
                    {
                        for (size_t k = 0; k < K; ++k)
                            *(dst++) = src[k*N + i];
                    }
                }
                else
                {
                    for (size_t dstRow = 0; dstRow < p.dstH; ++dstRow)
                    {
                        size_t srcRow0 = dstRow - p.padY;
                        for (size_t dstCol = 0; dstCol < p.dstW; ++dstCol)
                        {
                            size_t srcCol0 = dstCol - p.padX;
                            for (size_t channel = 0; channel < p.srcC; ++channel)
                            {
                                for (size_t kernelRow = 0; kernelRow < p.kernelY; ++kernelRow)
                                {
                                    size_t srcRow = srcRow0 + kernelRow;
                                    if (srcRow < p.srcH)
                                    {
                                        const float * psrc = src + (channel*p.srcH + srcRow)*p.srcW;
                                        for (size_t kernelCol = 0; kernelCol < p.kernelX; ++kernelCol)
                                        {
                                            size_t srcCol = srcCol0 + kernelCol;
                                            if (srcCol < p.srcW)
                                                *(dst++) = psrc[srcCol];
                                            else
                                                *(dst++) = 0;
                                        }
                                    }
                                    else
                                    {
                                        for (size_t kernelCol = 0; kernelCol < p.kernelX; ++kernelCol)
                                            *(dst++) = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                for (size_t dstRow = 0; dstRow < p.dstH; ++dstRow)
                {
                    size_t srcRow0 = dstRow * p.strideY - p.padY;
                    for (size_t dstCol = 0; dstCol < p.dstW; ++dstCol)
                    {
                        size_t srcCol0 = dstCol * p.strideX - p.padX;
                        for (size_t channel = 0; channel < p.srcC; ++channel)
                        {
                            for (size_t kernelRow = 0; kernelRow < p.kernelY; ++kernelRow)
                            {
                                size_t srcRow = srcRow0 + kernelRow * p.dilationY;
                                if (srcRow < p.srcH)
                                {
                                    const float * psrc = src + (channel*p.srcH + srcRow)*p.srcW;
                                    for (size_t kernelCol = 0; kernelCol < p.kernelX; ++kernelCol)
                                    {
                                        size_t srcCol = srcCol0 + kernelCol * p.dilationX;
                                        if (srcCol < p.srcW)
                                            *(dst++) = psrc[srcCol];
                                        else
                                            *(dst++) = 0;
                                    }
                                }
                                else
                                {
                                    for (size_t kernelCol = 0; kernelCol < p.kernelX; ++kernelCol)
                                        *(dst++) = 0;
                                }
                            }
                        }
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fWinograd::SynetConvolution32fWinograd(const ConvParam& p)
            : SynetConvolution32f(p)
        {
            if (p.kernelY == 1 && p.kernelX == 3)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Base::WinogradKernel1x3Block1x4SetFilter;
                    _setInput = Base::WinogradKernel1x3Block1x4SetInput;
                    _setOutput = Base::WinogradKernel1x3Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 1 && p.kernelX == 5)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Base::WinogradKernel1x5Block1x4SetFilter;
                    _setInput = Base::WinogradKernel1x5Block1x4SetInput;
                    _setOutput = Base::WinogradKernel1x5Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 2 && p.kernelX == 2)
            {
                if (p.trans && p.srcH >= 8 && p.srcW >= 8 && p.srcH * p.srcW * p.batch >= 144)
                {
                    SetBlock(4, 4);
                    _setFilter = Base::WinogradKernel2x2Block4x4SetFilter;
                    _setInput = Base::WinogradKernel2x2Block4x4SetInput;
                    _setOutput = Base::WinogradKernel2x2Block4x4SetOutput;
                }
                else
                {
                    SetBlock(2, 2);
                    _setFilter = Base::WinogradKernel2x2Block2x2SetFilter;
                    _setInput = Base::WinogradKernel2x2Block2x2SetInput;
                    _setOutput = Base::WinogradKernel2x2Block2x2SetOutput;
                }
            }
            else if (p.kernelY == 3 && p.kernelX == 3)
            {
                if (p.trans && p.srcH >= 8 && p.srcW >= 8 && p.srcH * p.srcW * p.batch >= 144)
                {
                    SetBlock(4, 4);
                    _setFilter = Base::WinogradKernel3x3Block4x4SetFilter;
                    _setInput = Base::WinogradKernel3x3Block4x4SetInput;
                    _setOutput = Base::WinogradKernel3x3Block4x4SetOutput;
                }
                else if (p.trans && p.srcH >= 6 && p.srcW >= 6 && p.srcH * p.srcW * p.batch >= 81 && p.dstH % 3 == 0 && p.dstW % 3 == 0)
                {
                    SetBlock(3, 3);
                    _setFilter = Base::WinogradKernel3x3Block3x3SetFilter;
                    _setInput = Base::WinogradKernel3x3Block3x3SetInput;
                    _setOutput = Base::WinogradKernel3x3Block3x3SetOutput;
                }
                else
                {
                    SetBlock(2, 2);
                    _setFilter = Base::WinogradKernel3x3Block2x2SetFilter;
                    _setInput = Base::WinogradKernel3x3Block2x2SetInput;
                    _setOutput = Base::WinogradKernel3x3Block2x2SetOutput;
                }
            }
            else
                assert(0);
            _gemm.Init(InitGemmFuncs(Base::Gemm32fNN, "Base"));
            _biasAndActivation = Base::ConvolutionBiasAndActivation;
        }

        String SynetConvolution32fWinograd::Desc() const 
        { 
            const ConvParam& p = this->Param();
            return Ext() + "::Winograd F(" + ToStr(_blockY) + "x" + ToStr(_blockX) + "," + ToStr(p.kernelY) + "x" + ToStr(p.kernelX) + ")" 
                + (_merge > 1 ? "*" + ToStr(_merge) : "") + (_split > 1 ? "/" + ToStr(_split) : "");
        }
        
        size_t SynetConvolution32fWinograd::ExternalBufferSize() const
        {
            return (_strideS + _strideD)*_count*_merge;
        }

        size_t SynetConvolution32fWinograd::InternalBufferSize() const
        {
            return Simd::SynetConvolution32f::InternalBufferSize() + _winogradWeight.size;
        }

        void SynetConvolution32fWinograd::SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params)
        {
            Simd::SynetConvolution32f::SetParams(weight, internal, bias, params);
            _winogradWeight.Resize(_strideW*_count);
            _setFilter(weight, _param.srcC*_param.dstC, _winogradWeight.data, _param.trans);
            if (_nhwcWeight.data)
            {
                for (size_t i = 0; i < _count; ++i)
                {
                    if (_gemmCb.Size())
                        _gemmCb.At(0).ReorderB(_M * _merge, _N, _K, _winogradWeight.data + i * _strideW, _nhwcWeight.data + i * _nhwcStrideW);
                    else
                        _nhwcReorderB(_M * _merge, _N, _K, _winogradWeight.data + i * _strideW, _nhwcWeight.data + i * _nhwcStrideW, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                }
                _winogradWeight.Resize(0);
            }
            if (internal)
                *internal = SimdTrue;
        }
        
        void SynetConvolution32fWinograd::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam & p = _param;
            float * bufS = Buffer(buf);
            float * bufD = bufS + _strideS * _count * _merge;
            if (p.trans)
            {
                if (_split > 1)
                    ForwardSplitted(src, bufS, bufD, dst);
                else
                    ForwardMerged(src, bufS, bufD, dst);
            }
            else
            {
                for (size_t b = 0; b < _batch; ++b)
                {
                    _setInput(src, p.srcC, p.srcH, p.srcW, p.padY, p.padX, p.padH, p.padW, bufS, _strideS, p.trans);
                    for (size_t i = 0; i < _count; ++i)
                        _gemm.Run(GemmArgs(_M, _N, _K, &_1, _winogradWeight.data + i * _strideW, _K, bufS + i * _strideS, _N, &_0, bufD + i * _strideD, _N));
                    _setOutput(bufD, _strideD, dst, p.dstC, p.dstH, p.dstW, p.trans);
                    _biasAndActivation(_bias, p.dstC, p.dstH*p.dstW, p.activation, _params, p.trans, dst);
                    src += _sizeS;
                    dst += _sizeD;
                }
            }
        }

        bool SynetConvolution32fWinograd::Preferable(const ConvParam & p)
        {
            if (!p.IsDilation(1) || !p.IsStride(1) || p.group != 1 || p.srcC <= 16)
                return false;

            if (p.IsKernel(1, 3))
            {
                if (!(p.IsPad(0) || (p.padX == 1 && p.padW == 1)) )
                    return false;
                if (p.srcC <= 32)
                    return false;
                return p.trans && p.srcW >= 8 && p.srcH * p.srcW * p.batch >= 36;
            }
            else if (p.IsKernel(1, 5))
            {
                if (!(p.IsPad(0) || (p.padX == 2 && p.padW == 2)))
                    return false;
                return p.trans && p.srcW >= 8 && p.srcH * p.srcW * p.batch >= 36;
            }           
            else if (p.IsKernel(2))
            {
                if (!(p.IsPad(0) || (p.padY + p.padH == 1 && p.padX + p.padW == 1)))
                    return false;
                return p.trans && p.srcH >= 4 && p.srcW >= 4 && p.srcH * p.srcW * p.batch >= 36;
            }
            else if (p.IsKernel(3))
            {
                if (!(p.IsPad(0) || p.IsPad(1)))
                    return false;
                if (p.trans)
                    return p.srcH >= 4 && p.srcW >= 4 && p.srcH * p.srcW * p.batch >= 36;
                else
                    return p.srcH >= 6 && p.srcW >= 6;
            }
            return false;
        }

        void SynetConvolution32fWinograd::SetBlock(size_t blockY, size_t blockX)
        {
            const ConvParam & p = _param;
            _blockY = blockY;
            _blockX = blockX;
            _count = (_blockY + p.kernelY - 1) * (_blockX + p.kernelX - 1);
            _tileH = (p.dstH + _blockY - 1) / _blockY;
            _tileW = (p.dstW + _blockX - 1) / _blockX;
            _strideW = p.srcC * p.dstC;
            _M = p.trans ? _tileW * _tileH : p.dstC;
            _N = p.trans ? p.dstC : _tileW * _tileH;
            _K = p.srcC;
            _batch = p.batch;
            _sizeS = p.srcC*p.srcH*p.srcW;
            _sizeD = p.dstC*p.dstH*p.dstW;
            _merge = 1;
            _split = 1;
            _tileHs = _tileH;
            if (p.trans)
            {
                if (_batch > 1)
                {
                    for (size_t merge = 1; merge <= _batch; ++merge)
                        if (_batch % merge == 0 && _M * merge <= 128)
                            _merge = merge;
                }
                if (_merge == 1 && _blockY == 4)
                {
                    size_t cacheL2 = Base::AlgCacheL2() / sizeof(float);
                    size_t cacheL3 = Base::AlgCacheL3() / sizeof(float);
                    size_t bufferSize = _count * (p.srcC + p.dstC) * _tileW * _tileH;
                    size_t weightSize = _count * p.srcC * p.dstC;
                    if (bufferSize > cacheL2)
                    {
                        _tileHs = Simd::RestrictRange<size_t>(size_t(cacheL2*0.5) * _tileH / bufferSize, 1, _tileH);
                        _split = DivHi(_tileH, _tileHs);
                        while (_split * _tileHs >= _tileH + _split)
                            _tileHs--;
                        if (_split > 1 && weightSize > cacheL3)
                        {
                            _split = DivHi(bufferSize, weightSize);
                            _tileHs = DivHi(_tileH, _split);
                            while (_split * _tileHs >= _tileH + _tileHs)
                                _split--;
                        }
                    }
                }
            }
            _strideS = p.srcC * _tileHs * _tileW;
            _strideD = p.dstC * _tileHs * _tileW;
        }

        void SynetConvolution32fWinograd::ForwardMerged(const float * src, float * bufS, float * bufD, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t b = 0; b < _batch; b += _merge)
            {
                for (size_t m = 0; m < _merge; ++m)
                    _setInput(src + m * _sizeS, p.srcC, p.srcH, p.srcW, p.padY, p.padX, p.padH, p.padW, bufS + m * _strideS, _strideS * _merge, p.trans);
                for (size_t i = 0; i < _count; ++i)
                {
                    if (_nhwcWeight.data)
                    {
                        if (_gemmCb.Size())
                            _gemmCb.Run(GemmCbArgs(_M * _merge, _N, _K, bufS + i * _strideS * _merge, _nhwcWeight.data + i * _nhwcStrideW, bufD + i * _strideD * _merge));
                        else
                            _nhwcRun(_M * _merge, _N, _K, bufS + i * _strideS * _merge, _nhwcWeight.data + i * _nhwcStrideW, bufD + i * _strideD * _merge, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                    }
                    else
                        _gemm.Run(GemmArgs(_M * _merge, _N, _K, &_1, bufS + i * _strideS * _merge, _K, _winogradWeight.data + i * _strideW, _N, &_0, bufD + i * _strideD * _merge, _N));
                }
                for (size_t m = 0; m < _merge; ++m)
                {
                    _setOutput(bufD + m * _strideD, _strideD * _merge, dst + m * _sizeD, p.dstC, p.dstH, p.dstW, p.trans);
                    _biasAndActivation(_bias, p.dstC, p.dstH * p.dstW, p.activation, _params, p.trans, dst + m * _sizeD);
                }
                src += _sizeS * _merge;
                dst += _sizeD * _merge;
            }
        }

        void SynetConvolution32fWinograd::ForwardSplitted(const float* src, float* bufS, float* bufD, float* dst)
        {
            const ConvParam& p = _param;
            for (size_t b = 0; b < _batch; ++b)
            {
                for (size_t s = 0; s < _split; ++s)
                {
                    size_t padY = s ? 0 : p.padY;
                    size_t padH = s == _split - 1 ? p.padH : 0;
                    size_t srcY = s * _tileHs * _blockY + padY - p.padY;
                    size_t srcH = Simd::Min(_tileHs * _blockY + p.kernelY - 1 - padY - padH, p.srcH - srcY);
                    size_t M = _tileW * Simd::Min(_tileHs, _tileH - s * _tileHs);
                    size_t dstY = s * _tileHs * _blockY;
                    size_t dstH = Simd::Min(_tileHs * _blockY, p.dstH - dstY);
                    _setInput(src + srcY * p.srcC * p.srcW, p.srcC, srcH, p.srcW, padY, p.padX, padH, p.padW, bufS, _strideS, p.trans);
                    for (size_t i = 0; i < _count; ++i)
                    {
                        if (_nhwcWeight.data)
                        {
                            if (_gemmCb.Size())
                                _gemmCb.Run(GemmCbArgs(M, _N, _K, bufS + i * _strideS, _nhwcWeight.data + i * _nhwcStrideW, bufD + i * _strideD));
                            else
                                _nhwcRun(M, _N, _K, bufS + i * _strideS, _nhwcWeight.data + i * _nhwcStrideW, bufD + i * _strideD, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                        }
                        else
                            _gemm.Run(GemmArgs(M, _N, _K, &_1, bufS + i * _strideS, _K, _winogradWeight.data + i * _strideW, _N, &_0, bufD + i * _strideD, _N));
                    }
                    _setOutput(bufD, _strideD, dst + dstY * p.dstC * p.dstW, p.dstC, dstH, p.dstW, p.trans);
                    _biasAndActivation(_bias, p.dstC, dstH * p.dstW, p.activation, _params, p.trans, dst + dstY * p.dstC * p.dstW);
                }
                src += _sizeS;
                dst += _sizeD;
            }
        }
    }
#endif
}
