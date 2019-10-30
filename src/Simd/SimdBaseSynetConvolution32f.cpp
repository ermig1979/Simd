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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
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
                    NeuralRelu(dst, size*count, &slope, dst);
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
                                dst[i] = SynetPreluLayerForward(dst[i] + bias[i], slope);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            for (size_t j = 0; j < size; ++j)
                                dst[j] = SynetPreluLayerForward(dst[j] + bias[i], slope);
                            dst += size;
                        }
                    }
                }
                else
                    NeuralRelu(dst, size*count, &slope, dst);
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
                                dst[i] = SynetPreluLayerForward(dst[i] + bias[i], params[i]);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            for (size_t j = 0; j < size; ++j)
                                dst[j] = SynetPreluLayerForward(dst[j] + bias[i], params[i]);
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
            else
                assert(0);
        }

        SynetConvolution32fGemmNN::SynetConvolution32fGemmNN(const ConvParam32f & p)
            : SynetConvolution32f(p)
        {
            _is1x1 = p.Is1x1();
            if (p.trans)
            {
                _M = p.dstH * p.dstW;
                _N = p.dstC / p.group;
                _K = p.srcC * p.kernelY * p.kernelX / p.group;
                _ldS = _K * (_is1x1 ? p.group : 1);
                _ldW = p.dstC;
                _ldD = p.dstC;
                _grW = _N;
                _grS = _K * (_is1x1 ? 1 : _M);
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
            _gemm.Init(InitGemmFuncs(Base::Gemm32fNN, "Base", p.gemm, "Ext"));
            _biasAndActivation = Base::ConvolutionBiasAndActivation;
        }

        size_t SynetConvolution32fGemmNN::ExternalBufferSize() const
        {
            if (_is1x1)
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
            const ConvParam32f & p = _param;
            if (!_is1x1)
                buf = Buffer(buf);
            if (_merge > 1)
            {
                for (size_t b = 0; b < _batch; b += _merge)
                {
                    const float * tmp = src;
                    if (!_is1x1)
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
                    if (!_is1x1)
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
            const ConvParam32f & p = _param;
            assert(!p.trans);
            size_t srcSize = p.srcW * p.srcH;
            if (p.dilationX == 1 && p.dilationY == 1 && p.strideX == 2 && p.strideY == 2 && p.padX == 0 && p.padY == 0 && p.padW == 0 && p.padH == 0 && p.kernelX == 1 && p.kernelY == 1)
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
            else if (p.dilationX*p.dilationY*p.strideX*p.strideY != 1)
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
            else
            {
                const ptrdiff_t bodySize = p.dstW - p.padX - p.padW;
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
                                    const float * psrc = src + sy * p.srcW;
                                    for (; dx < p.padX; ++dx, ++sx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = psrc[sx];
                                        else
                                            *(dst++) = 0;
                                    }
                                    if (bodySize > 0)
                                    {
                                        memcpy(dst, psrc + sx, bodySize * sizeof(float));
                                        dst += bodySize;
                                        dx += bodySize;
                                        sx += bodySize;
                                    }
                                    for (; dx < p.dstW; ++dx, ++sx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = psrc[sx];
                                        else
                                            *(dst++) = 0;
                                    }
                                }
                                else
                                {
                                    memset(dst, 0, p.dstW * sizeof(float));
                                    dst += p.dstW;
                                }
                            }
                        }
                    }
                    src += srcSize;
                }
            }
        }

        void SynetConvolution32fGemmNN::ImgToRow(const float * src, float * dst)
        {
            const ConvParam32f & p = _param;
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

        //---------------------------------------------------------------------

        SynetConvolution32fGemmNT::SynetConvolution32fGemmNT(const ConvParam32f & p)
            : SynetConvolution32f(p)
        {
            _is1x1 = p.IsKernel(1) && p.IsDilation(1) && p.IsStride(1) && p.IsPad(0);
            _M = p.dstC / p.group;
            _N = p.dstH  * p.dstW;
            _K = p.srcC * p.kernelY * p.kernelX / p.group;
            _weightStep = p.dstC * _K / p.group;
            _batch = p.batch;
            _sizeS = p.srcC*p.srcH*p.srcW;
            _sizeB = p.srcC*p.kernelY*p.kernelX*p.dstH*p.dstW;
            _sizeD = p.dstC*p.dstH*p.dstW;
        }

        size_t SynetConvolution32fGemmNT::ExternalBufferSize() const
        {
            return _sizeB;
        };

        void SynetConvolution32fGemmNT::Forward(const float * src, float * buf, float * dst)
        {
            buf = Buffer(buf);
            for (size_t b = 0; b < _batch; ++b)
            {
                ImgToRow(src, _param, buf);
                GemmAndBias(buf, dst);
                src += _sizeS;
                dst += _sizeD;
            }
        }

        bool SynetConvolution32fGemmNT::Preferable(const ConvParam32f & p)
        {
            return p.trans == 0 && p.srcH < 6 && p.srcW < 6 && p.group == 1;
        }

        void SynetConvolution32fGemmNT::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam32f & p = _param;
            for (size_t g = 0; g < p.group; ++g)
                Base::Gemm32fNT(_M, _N, _K, &_1, _weight + _weightStep * g, _K, src + _srcStep * g, _K, &_0, dst + _dstStep * g, _N);
            ConvolutionBiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, p.activation, _params, ::SimdFalse, dst);
        }

        void SynetConvolution32fGemmNT::ImgToRow(const float * src, const ConvParam32f & p, float * dst)
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

        //---------------------------------------------------------------------

        SynetConvolution32fWinograd::SynetConvolution32fWinograd(const ConvParam32f & p)
            : SynetConvolution32f(p)
        {
            if (p.trans && p.srcH >= 8 && p.srcW >= 8 && p.srcH*p.srcW*p.batch >= 144)
                SetBlock(4);
            else if (p.trans && p.srcH >= 6 && p.srcW >= 6 && p.srcH*p.srcW*p.batch >= 81 && p.dstH % 3 == 0 && p.dstW % 3 == 0)
                SetBlock(3);
            else
                SetBlock(2);
            switch (_block)
            {
            case 2:
                _setFilter = Base::Winograd2x3SetFilter;
                _setInput = Base::Winograd2x3SetInput;
                _setOutput = Base::Winograd2x3SetOutput;
                break;
            case 3:
                _setFilter = Base::Winograd3x3SetFilter;
                _setInput = Base::Winograd3x3SetInput;
                _setOutput = Base::Winograd3x3SetOutput;
                break;
            case 4:
                _setFilter = Base::Winograd4x3SetFilter;
                _setInput = Base::Winograd4x3SetInput;
                _setOutput = Base::Winograd4x3SetOutput;
                break;
            default:
                assert(0);
            }
            _gemm.Init(InitGemmFuncs(Base::Gemm32fNN, "Base", p.gemm, "Ext"));
            _biasAndActivation = Base::ConvolutionBiasAndActivation;
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
            const ConvParam32f & p = _param;
            float * bufS = Buffer(buf);
            float * bufD = bufS + _strideS * _count * _merge;
            if (p.trans)
            {
                ForwardMerged(src, bufS, bufD, dst, _merge);
            }
            else
            {
                for (size_t b = 0; b < _batch; ++b)
                {
                    _setInput(src, p.srcC, p.srcH, p.srcW, bufS, _strideS, _pad, p.trans);
                    for (size_t i = 0; i < _count; ++i)
                        _gemm.Run(GemmArgs(_M, _N, _K, &_1, _winogradWeight.data + i * _strideW, _K, bufS + i * _strideS, _N, &_0, bufD + i * _strideD, _N));
                    _setOutput(bufD, _strideD, dst, p.dstC, p.dstH, p.dstW, p.trans);
                    _biasAndActivation(_bias, p.dstC, p.dstH*p.dstW, p.activation, _params, p.trans, dst);
                    src += _sizeS;
                    dst += _sizeD;
                }
            }
        }

        bool SynetConvolution32fWinograd::Preferable(const ConvParam32f & p)
        {
            return p.IsKernel(3) && p.IsDilation(1) && p.IsStride(1) && (p.IsPad(0) || p.IsPad(1)) && p.group == 1 && p.srcC > 16 && 
                (p.trans ? (p.srcH >= 4 && p.srcW >= 4 && p.srcH*p.srcW*p.batch >= 36) : (p.srcH >= 6 && p.srcW >= 6));
        }

        void SynetConvolution32fWinograd::SetBlock(size_t block)
        {
            const ConvParam32f & p = _param;
            _block = block;
            _count = Simd::Square(_block + p.kernelX - 1);
            _tileH = (p.dstH + _block - 1) / _block;
            _tileW = (p.dstW + _block - 1) / _block;
            _strideW = p.srcC * p.dstC;
            _strideS = p.srcC * _tileH * _tileW;
            _strideD = p.dstC * _tileH * _tileW;
            _M = p.trans ? _tileW * _tileH : p.dstC;
            _N = p.trans ? p.dstC : _tileW * _tileH;
            _K = p.srcC;
            _pad = (SimdBool)p.padX;
            _batch = p.batch;
            _sizeS = p.srcC*p.srcH*p.srcW;
            _sizeD = p.dstC*p.dstH*p.dstW;
            _merge = 1;
            if (p.trans && _batch > 1)
            {
                for (size_t merge = 1; merge <= _batch; ++merge)
                    if (_batch%merge == 0 && _M*merge <= 128)
                        _merge = merge;
            }
        }

        void SynetConvolution32fWinograd::ForwardMerged(const float * src, float * bufS, float * bufD, float * dst, size_t merge)
        {
            const ConvParam32f & p = _param;
            for (size_t b = 0; b < _batch; b += merge)
            {
                for (size_t m = 0; m < merge; ++m)
                    _setInput(src + m * _sizeS, p.srcC, p.srcH, p.srcW, bufS + m * _strideS, _strideS * merge, _pad, p.trans);
                for (size_t i = 0; i < _count; ++i)
                {
                    if (_nhwcWeight.data)
                    {
                        if (_gemmCb.Size())
                            _gemmCb.Run(GemmCbArgs(_M * merge, _N, _K, bufS + i * _strideS * merge, _nhwcWeight.data + i * _nhwcStrideW, bufD + i * _strideD * merge));
                        else
                            _nhwcRun(_M * merge, _N, _K, bufS + i * _strideS * merge, _nhwcWeight.data + i * _nhwcStrideW, bufD + i * _strideD * merge, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                    }
                    else
                        _gemm.Run(GemmArgs(_M * merge, _N, _K, &_1, bufS + i * _strideS * merge, _K, _winogradWeight.data + i * _strideW, _N, &_0, bufD + i * _strideD * merge, _N));
                }
                for (size_t m = 0; m < merge; ++m)
                {
                    _setOutput(bufD + m * _strideD, _strideD * merge, dst + m * _sizeD, p.dstC, p.dstH, p.dstW, p.trans);
                    _biasAndActivation(_bias, p.dstC, p.dstH*p.dstW, p.activation, _params, p.trans, dst + m * _sizeD);
                }
                src += _sizeS * merge;
                dst += _sizeD * merge;
            }
        }

        //---------------------------------------------------------------------

        SynetConvolution32fDirectNchw::SynetConvolution32fDirectNchw(const ConvParam32f & p)
            : SynetConvolution32f(p)
        {
            _srcC = p.srcC / p.group;
            _srcH = p.padY + p.srcH + p.padH;
            _srcW = p.padX + p.srcW + p.padW;
            _dstC = p.dstC / p.group;
            _grW = _srcC * _dstC * p.kernelY * p.kernelX;
            _grS = _srcC * p.srcH * p.srcW;
            _grD = _dstC * p.dstH  * p.dstW;
            _pad = p.IsPad(0) ? 0 : 1;
            _convolutionBiasActivation = SetConvolutionBiasActivation();
        }

        size_t SynetConvolution32fDirectNchw::ExternalBufferSize() const
        {
            if (_pad)
                return _srcC*_srcH*_srcW;
            else
                return 1;
        }

        void SynetConvolution32fDirectNchw::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam32f & p = _param;
            if(_pad)
                buf = Buffer(buf);
            for (size_t b = 0; b < p.batch; ++b)
            {
                const float * weight = _weight;
                const float * bias = _bias;
                const float * params = _params;
                for (size_t g = 0; g < p.group; ++g)
                {
                    if (_pad)
                    {
                        Pad(src, buf);
                        _convolutionBiasActivation(buf, _srcC, _srcH, _srcW, weight, bias, params, dst, _dstC, p.dstH, p.dstW);
                    }
                    else
                        _convolutionBiasActivation(src, _srcC, _srcH, _srcW, weight, bias, params, dst, _dstC, p.dstH, p.dstW);
                    weight += _grW;
                    if (bias)
                        bias += _dstC;
                    if (p.activation == ::SimdConvolutionActivationPrelu)
                        params += _dstC;
                    src += _grS;
                    dst += _grD;
                }
            }
        }

        bool SynetConvolution32fDirectNchw::Preferable(const ConvParam32f & p)
        {
            if (!p.IsDilation(1))
                return false;
            if (!(p.IsStride(1) || p.IsStride(2) || p.IsStride(3)))
                return false;
            double k = double(p.srcC) / p.group * p.strideX * p.strideY / p.kernelX / p.kernelY;
            return k < 2.0 && (p.IsKernel(2) || p.IsKernel(3)) && p.trans == 0;
        }

        void SynetConvolution32fDirectNchw::Pad(const float * src, float * dst) const
        {
            const ConvParam32f & p = _param;
            for (size_t c = 0; c < _srcC; ++c)
            {
                if (p.padY)
                {
                    memset(dst, 0, p.padY*_srcW * sizeof(float));
                    dst += p.padY*_srcW;
                }
                for (size_t row = 0; row < p.srcH; ++row)
                {
                    for (size_t col = 0; col < p.padX; ++col)
                        *dst++ = 0;
                    memcpy(dst, src, p.srcW * sizeof(float));
                    dst += p.srcW;
                    src += p.srcW;
                    for (size_t col = 0; col < p.padW; ++col)
                        *dst++ = 0;
                }
                if (p.padH)
                {
                    memset(dst, 0, p.padH*_srcW * sizeof(float));
                    dst += p.padH*_srcW;
                }
            }
        }

        SIMD_INLINE void AddConvolutionKernel1x1(const float * src, size_t srcW, size_t strideY, size_t strideX, const float * weight, float * dst, size_t dstH, size_t dstW)
        {
            for (size_t dy = 0; dy < dstH; ++dy)
            {
                for (size_t dx = 0, sx = 0; dx < dstW; ++dx, sx += strideX)
                    dst[dx] += src[sx]*weight[0];
                src += srcW * strideY;
                dst += dstW;
            }
        }

        SIMD_INLINE float ConvolutionKernel2(const float * src, const float * weight)
        {
            return src[0] * weight[0] + src[1] * weight[1];
        }

        SIMD_INLINE float ConvolutionKernel2x2(const float * src, size_t srcW, const float * weight)
        {
            return
                ConvolutionKernel2(src, weight) +
                ConvolutionKernel2(src + srcW, weight + 2);
        }

        SIMD_INLINE void AddConvolutionKernel2x2(const float * src, size_t srcW, size_t strideY, size_t strideX, const float * weight, float * dst, size_t dstH, size_t dstW)
        {
            for (size_t dy = 0; dy < dstH; ++dy)
            {
                for (size_t dx = 0, sx = 0; dx < dstW; ++dx, sx += strideX)
                    dst[dx] += ConvolutionKernel2x2(src + sx, srcW, weight);
                src += srcW * strideY;
                dst += dstW;
            }
        }

        SIMD_INLINE float ConvolutionKernel3(const float * src, const float * weight)
        {
            return src[0] * weight[0] + src[1] * weight[1] + src[2] * weight[2];
        }

        SIMD_INLINE float ConvolutionKernel3x3(const float * src, size_t srcW, const float * weight)
        {
            return
                ConvolutionKernel3(src, weight) +
                ConvolutionKernel3(src + srcW, weight + 3) +
                ConvolutionKernel3(src + 2 * srcW, weight + 6);
        }

        SIMD_INLINE void AddConvolutionKernel3x3(const float * src, size_t srcW, size_t strideY, size_t strideX, const float * weight, float * dst, size_t dstH, size_t dstW)
        {
            for (size_t dy = 0; dy < dstH; ++dy)
            {
                for (size_t dx = 0, sx = 0; dx < dstW; ++dx, sx += strideX)
                    dst[dx] += ConvolutionKernel3x3(src + sx, srcW, weight);
                src += srcW * strideY;
                dst += dstW;
            }
        }

        template<int kernel, int stride, ::SimdConvolutionActivationType type> 
        void ConvolutionBiasActivation(const float * src, size_t srcC, size_t srcH, size_t srcW, const float * weight, 
            const float * bias, const float * params, float * dst, size_t dstC, size_t dstH, size_t dstW)
        {
            for (size_t dc = 0; dc < dstC; ++dc)
            {
                Fill32f(dst, dstW * dstH, bias ? bias + dc : NULL);
                for (size_t sc = 0; sc < srcC; ++sc)
                {
                    const float * ps = src + sc * srcW * srcH;
                    const float * pw = weight + (dc*srcC + sc)*kernel*kernel;
                    float * pd = dst;
                    if (kernel == 1)
                        AddConvolutionKernel1x1(ps, srcW, stride, stride, pw, pd, dstH, dstW);
                    else if (kernel == 2)
                        AddConvolutionKernel2x2(ps, srcW, stride, stride, pw, pd, dstH, dstW);
                    else if (kernel == 3)
                        AddConvolutionKernel3x3(ps, srcW, stride, stride, pw, pd, dstH, dstW);
                    else
                    {
                        for (size_t dy = 0; dy < dstH; ++dy)
                        {
                            for (size_t dx = 0, sx = 0; dx < dstW; ++dx, sx += stride)
                            {
                                float sum = 0;
                                for (size_t ky = 0; ky < kernel; ++ky)
                                {
                                    const float * s = ps + ky * srcW + sx;
                                    const float * w = pw + kernel*ky;
                                    for (size_t kx = 0; kx < kernel; ++kx)
                                        sum += s[kx] * w[kx];
                                }
                                pd[dx] += sum;
                            }
                            ps += srcW * stride;
                            pd += dstW;
                        }
                    }
                }
                ConvolutionBiasAndActivation(NULL, 1, dstH*dstW, type, params, ::SimdFalse, dst);
                if (type == ::SimdConvolutionActivationPrelu)
                    params++;
                dst += dstW * dstH;
            }
        }

        template <int kernel, int stride> SynetConvolution32fDirectNchw::ConvolutionBiasActivationPtr SetConvolutionBiasActivation(::SimdConvolutionActivationType type)
        {
            switch (type)
            {
            case ::SimdConvolutionActivationIdentity: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationIdentity>;
            case ::SimdConvolutionActivationRelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationRelu>;
            case ::SimdConvolutionActivationLeakyRelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationLeakyRelu>;
            case ::SimdConvolutionActivationRestrictRange: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationRestrictRange>;
            case ::SimdConvolutionActivationPrelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationPrelu>;
            case ::SimdConvolutionActivationElu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationElu>;
            case ::SimdConvolutionActivationHswish: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationHswish>;
            default:
                assert(0);
                return NULL;
            }
        }

        SynetConvolution32fDirectNchw::ConvolutionBiasActivationPtr SynetConvolution32fDirectNchw::SetConvolutionBiasActivation()
        {
            const ConvParam32f & p = _param;
            switch (p.strideX)
            {
            case 1:
                if (p.kernelX == 1)
                    return Base::SetConvolutionBiasActivation<1, 1>(p.activation);
                if (p.kernelX == 2)
                    return Base::SetConvolutionBiasActivation<2, 1>(p.activation);
                if (p.kernelX == 3)
                    return Base::SetConvolutionBiasActivation<3, 1>(p.activation);
                break;
            case 2: 
                if (p.kernelX == 2)
                    return Base::SetConvolutionBiasActivation<2, 2>(p.activation);
                if (p.kernelX == 3)
                    return Base::SetConvolutionBiasActivation<3, 2>(p.activation);
                break;
            case 3: 
                if (p.kernelX == 3)
                    return Base::SetConvolutionBiasActivation<3, 3>(p.activation);
                break;
            }
            return NULL;
        }

        //---------------------------------------------------------------------

        SynetConvolution32fDirectNhwc::SynetConvolution32fDirectNhwc(const ConvParam32f & p)
            : SynetConvolution32f(p)
        {
            _batch = p.batch;
            _sizeS = p.srcC*p.srcH*p.srcW;
            _sizeD = p.dstC*p.dstH*p.dstW;
            _convolutionBiasActivation = SetConvolutionBiasActivation();
        }

        void SynetConvolution32fDirectNhwc::Forward(const float * src, float * buf, float * dst)
        {
            for (size_t b = 0; b < _batch; ++b)
            {
                _convolutionBiasActivation(src, _param, _weight, _bias, _params, dst);
                src += _sizeS;
                dst += _sizeD;
            }
        }

        bool SynetConvolution32fDirectNhwc::Preferable(const ConvParam32f & p)
        {
            if (p.trans == 0)
                return false;
            if (p.group == 1)
            {
                double k = double(p.srcC) / p.group * p.strideX * p.strideY / p.kernelX / p.kernelY;
                return k < 2.0;
            }
            return p.IsDepthwise();
        }

        static void ConvolutionDirectNhwcConvolutionBiasActivationDefault(const float * src, const ConvParam32f & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t group = p.group;
            size_t srcC = p.srcC / group;
            size_t dstC = p.dstC / group;
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    memset(dst, 0, p.dstC * sizeof(float));
                    for (size_t ky = 0; ky < p.kernelY; ++ky)
                    {
                        size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                if (sx < p.srcW)
                                {

                                    const float * pw = weight + (ky*p.kernelX + kx)*srcC*p.dstC;
                                    const float * ps = src + (sy*p.srcW + sx)*p.srcC;
                                    if (group == 1)
                                    {
                                        for (size_t sc = 0; sc < srcC; ++sc)
                                        {
                                            for (size_t dc = 0; dc < dstC; ++dc)
                                                dst[dc] += ps[0] * pw[dc];
                                            ps += 1;
                                            pw += dstC;
                                        }
                                    }
                                    else
                                    {
                                        for (size_t g = 0; g < group; ++g)
                                            dst[g] += ps[g] * pw[g];
                                    }
                                }
                            }
                        }
                    }
                    ConvolutionBiasAndActivation(bias, p.dstC, 1, p.activation, params, ::SimdTrue, dst);
                    dst += p.dstC;
                }
            }
        }

        SynetConvolution32fDirectNhwc::ConvolutionBiasActivationPtr SynetConvolution32fDirectNhwc::SetConvolutionBiasActivation()
        {
            return ConvolutionDirectNhwcConvolutionBiasActivationDefault;
        }

        //---------------------------------------------------------------------

        SynetConvolution32fDepthwiseDotProduct::SynetConvolution32fDepthwiseDotProduct(const ConvParam32f & p)
            : SynetConvolution32f(p)
        {
            _count = p.srcC;
            _size = p.srcH*p.srcW;
            _batch = p.batch;
            _sizeS = p.srcC*p.srcH*p.srcW;
            _sizeD = p.dstC*p.dstH*p.dstW;
        }

        SIMD_INLINE float DotProduct(const float * a, const float * b, size_t size)
        {
            size_t i = 0, aligned = size&(~3);
            float sums[4] = { 0, 0, 0, 0 };
            for (; i < aligned; i += 4)
            {
                sums[0] += a[i + 0] * b[i + 0];
                sums[1] += a[i + 1] * b[i + 1];
                sums[2] += a[i + 2] * b[i + 2];
                sums[3] += a[i + 3] * b[i + 3];
            }
            for (; i < size; ++i)
                sums[0] += a[i] * b[i];
            return sums[0] + sums[1] + sums[2] + sums[3];
        }
       
        void SynetConvolution32fDepthwiseDotProduct::Forward(const float * src, float * buf, float * dst)
        {
            for (size_t b = 0; b < _batch; ++b)
            {
                if (_bias)
                {
                    for (size_t i = 0; i < _count; ++i)
                        dst[i] = DotProduct(src + i * _size, _weight + i * _size, _size) + _bias[i];
                }
                else
                {
                    for (size_t i = 0; i < _count; ++i)
                        dst[i] = DotProduct(src + i * _size, _weight + i * _size, _size);
                }
                if (_param.activation)
                    ConvolutionBiasAndActivation(NULL, _count, 1, _param.activation, _params, ::SimdFalse, dst);                
                src += _sizeS;
                dst += _sizeD;
            }
        }

        bool SynetConvolution32fDepthwiseDotProduct::Preferable(const ConvParam32f & p)
        {
            if (!(p.IsPad(0) && p.IsDilation(1) && p.IsStride(1)))
                return false;
            if (!(p.dstC == p.srcC && p.dstC == p.group && p.srcW == p.kernelX && p.srcH == p.kernelY))
                return false;
            return p.trans == 0;
        }

        //---------------------------------------------------------------------

        void ConvolutionNhwcDirectAny(const float * src, const ConvParam32f & p, const SynetConvolution32fNhwcDirect::AlgParam & a, 
            const float * weight, const float * bias, const float * params, float * dst)
        {
            ConvolutionDirectNhwcConvolutionBiasActivationDefault(src, p, weight, bias, params, dst);
        }

        SynetConvolution32fNhwcDirect::SynetConvolution32fNhwcDirect(const ConvParam32f & p)
            : SynetConvolution32f(p)
        {
            _sizeS = p.srcC*p.srcH*p.srcW;
            _sizeD = p.dstC*p.dstH*p.dstW;
            _convolution = ConvolutionNhwcDirectAny;
        }

        void SynetConvolution32fNhwcDirect::SetAlgParam(size_t microD, size_t L1, size_t L2, size_t L3)
        {
            const ConvParam32f & p = _param;
            _alg.microD = microD;
            _alg.macroC = Simd::Min(L1 / sizeof(float) / p.kernelY / p.kernelX / microD, p.srcC);
            for (size_t macroH = p.dstH; macroH >= 1; macroH--)
            {
                _alg.macroH = macroH;
                if (_alg.macroC * p.srcW * (_alg.macroH * p.strideY + p.kernelY * p.dilationY - 1) <= L2)
                    break;
            }
            _alg.macroD = Simd::Min(AlignLoAny(L3 / sizeof(float) / p.kernelY / p.kernelX / _alg.macroC, _alg.microD), AlignHiAny(p.dstC, _alg.microD));
            _rWeight.Resize(AlignHiAny(p.dstC, _alg.microD) * p.kernelY * p.kernelX * p.srcC);
            _rBias.Resize(AlignHiAny(p.dstC, _alg.microD), true);
            if(p.activation == ::SimdConvolutionActivationPrelu)
                _rParams.Resize(AlignHiAny(p.dstC, _alg.microD));
        }

        void SynetConvolution32fNhwcDirect::ReorderWeight(const float * src, float * dst)
        {
            const ConvParam32f & p = _param;
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
                                    const float * s = src + ((ky * p.kernelX + kx) * p.srcC + sa + si) * p.dstC + da + di;
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

        size_t SynetConvolution32fNhwcDirect::InternalBufferSize() const
        {
            return _buffer.size + _rWeight.size + _rBias.size + _rParams.size;
        }

        void SynetConvolution32fNhwcDirect::SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params)
        {
            SynetConvolution32f::SetParams(weight, internal, bias, params);
            if (_rWeight.data)
            {
                const ConvParam32f & p = _param;
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
            if (_rParams.data && _param.activation == ::SimdConvolutionActivationPrelu)
            {
                memcpy(_rParams.data, params, _param.dstC * sizeof(float));
                _params = _rParams.data;
            }
        }

        void SynetConvolution32fNhwcDirect::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam32f & p = _param;
            for (size_t b = 0; b < p.batch; ++b)
            {
                _convolution(src, _param, _alg, _weight, _bias, _params, dst);
                src += _sizeS;
                dst += _sizeD;
            }
        }

        bool SynetConvolution32fNhwcDirect::Preferable(const ConvParam32f & p)
        {
            return false;
        }

        //---------------------------------------------------------------------

        void * SynetConvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm)
        {
            ConvParam32f param(batch, conv, gemm);
            if (!param.Valid())
                return NULL;
            else if (SynetConvolution32fDepthwiseDotProduct::Preferable(param))
                return new SynetConvolution32fDepthwiseDotProduct(param);
            else if(SynetConvolution32fWinograd::Preferable(param))
                return new SynetConvolution32fWinograd(param);
            else if (SynetConvolution32fGemmNT::Preferable(param))
                return new SynetConvolution32fGemmNT(param);
            else if (SynetConvolution32fDirectNchw::Preferable(param))
                return new SynetConvolution32fDirectNchw(param);
            else if (SynetConvolution32fNhwcDirect::Preferable(param))
                return new SynetConvolution32fNhwcDirect(param);
            else if (SynetConvolution32fDirectNhwc::Preferable(param))
                return new SynetConvolution32fDirectNhwc(param);
            else
                return new SynetConvolution32fGemmNN(param);
        }
    }
}
