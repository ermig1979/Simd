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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
    Base::PerformanceMeasurer * SynetConvolution32f::Perf(const char* func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

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
            else
                assert(0);
        }

        SynetConvolution32fGemmNN::SynetConvolution32fGemmNN(const ConvParam32f & p)
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
            _gemm.Init(InitGemmFuncs(Base::Gemm32fNN, "Base", p.gemm, "Ext"));
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
            const ConvParam32f & p = _param;
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
            const ConvParam32f & p = _param;
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
                                    const float* psrc = src + sy * p.srcW;
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

        bool SynetConvolution32fGemmNN::GemmRuntime() const
        {
            return NHWC_GEMM_RUNTIME && _param.SizeW() * sizeof(float) < Base::AlgCacheL3() * 1.0f;
        }

        //---------------------------------------------------------------------

        SynetConvolution32fGemmNT::SynetConvolution32fGemmNT(const ConvParam32f & p)
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
            const ConvParam32f& p = _param;
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

        bool SynetConvolution32fGemmNT::Preferable(const ConvParam32f & p)
        {
            if (p.group != 1)
                return false;
            if (p.trans)
                return p.Is1x1() && p.dstC == 1;
            else
                return p.srcH < 6 && p.srcW < 6;
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

        SynetConvolution32fWinograd::SynetConvolution32fWinograd(const ConvParam32f& p)
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
            _gemm.Init(InitGemmFuncs(Base::Gemm32fNN, "Base", p.gemm, "Ext"));
            _biasAndActivation = Base::ConvolutionBiasAndActivation;
        }

        String SynetConvolution32fWinograd::Desc() const 
        { 
            const ConvParam32f& p = this->Param();
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
            const ConvParam32f & p = _param;
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

        bool SynetConvolution32fWinograd::Preferable(const ConvParam32f & p)
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
            const ConvParam32f & p = _param;
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
            const ConvParam32f & p = _param;
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
            const ConvParam32f& p = _param;
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
            case ::SimdConvolutionActivationMish: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationMish>;
            case ::SimdConvolutionActivationHardSigmoid: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationHardSigmoid>;
            case ::SimdConvolutionActivationSwish: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationSwish>;
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

        SynetConvolution32fNhwcDirect::SynetConvolution32fNhwcDirect(const ConvParam32f & p)
            : SynetConvolution32f(p)
        {
            _sizeS = p.srcC*p.srcH*p.srcW;
            _sizeD = p.dstC*p.dstH*p.dstW;
            _old.enable = false;
            if (p.IsDilation(1))
            {
                if (p.srcC <= 3)
                    _old.enable = true;
                if (p.SizeW()*sizeof(float) > Base::AlgCacheL3() * 1.0)
                    _old.enable = true;
            }
            _old.convolution = NULL;
        }

        size_t SynetConvolution32fNhwcDirect::InternalBufferSize() const
        {
            size_t size = _buffer.size + _rWeight.size + _rBias.size + _rParams.size;
            size += _old.weight.size;
            return size;
        }

        void SynetConvolution32fNhwcDirect::SetParams(const float * weight, SimdBool * internal, const float * bias, const float * params)
        {
            SynetConvolution32f::SetParams(weight, internal, bias, params);
            if (_old.enable && _old.weight.data)
            {
                OldReorderWeight(weight, _old.weight.data);
                _weight = _old.weight.data;
                if (internal)
                    *internal = SimdTrue;
            }
            else
            if (_rWeight.data)
            {
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
                const ConvParam32f& p = _param;
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

        void SynetConvolution32fNhwcDirect::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam32f & p = _param;
            for (size_t b = 0; b < p.batch; ++b)
            {
                if(_old.enable)
                    _old.convolution(src, _param, _old.alg, _weight, _bias, _params, dst);
                else
                _run.Run(RunArgs(src, _param, _weight, _bias, _params, dst));
                src += _sizeS;
                dst += _sizeD;
            }
        }

        void SynetConvolution32fNhwcDirect::Forward(const float* src, const ConvParam32f& p, const AlgParam& a, const float* weight, const float* bias, const float* params, float* dst)
        {
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                for (size_t sc = 0; sc < p.srcC; sc += a.macroC)
                {
                    size_t macroC = Simd::Min(p.srcC, sc + a.macroC) - sc;
                    for (size_t yBeg = 0; yBeg < p.dstH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + a.macroH, p.dstH);
                        if (sc + macroC == p.srcC)
                            a.convolutions[TermLast](src + sc, p, a, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, macroC == p.srcC ? 1 : 0);
                        else
                            a.convolutions[TermInterim](src + sc, p, a, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, sc == 0 ? 1 : 0);
                        yBeg = yEnd;
                    }
                    weight += a.F * macroC;
                }
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroD;
                weight += p.kernelY * p.kernelY * p.srcC * macroD - p.srcC * a.F;
            }
        }

        void SynetConvolution32fNhwcDirect::SetAlgParam(size_t F, size_t N, AlgParam & alg)
        {
            const ConvParam32f& p = _param;
            alg.F = F;
            alg.microD = F*N;
            alg.macroC = Simd::Min(Base::AlgCacheL1() / sizeof(float) / p.kernelY / p.kernelX / alg.microD, p.srcC);
            for (size_t macroH = p.dstH; macroH >= 1; macroH--)
            {
                alg.macroH = macroH;
                if (alg.macroC * p.srcW * (alg.macroH * p.strideY + p.kernelY * p.dilationY - 1) * sizeof(float) <= Base::AlgCacheL2())
                    break;
            }
            alg.macroD = Simd::RestrictRange(AlignLoAny(Base::AlgCacheL3() / sizeof(float) / p.kernelY / p.kernelX / alg.macroC, alg.microD), 
                alg.microD, AlignHiAny(p.dstC, alg.microD));
            alg.stepW = p.kernelY * p.kernelX * p.srcC * alg.F;
            _rWeight.Resize(DivHi(p.dstC, alg.F)*alg.stepW);
            _rBias.Resize(AlignHiAny(p.dstC, alg.F), true);
            if (p.activation == SimdConvolutionActivationLeakyRelu || p.activation == SimdConvolutionActivationPrelu)
                _rParams.Resize(AlignHiAny(p.dstC, alg.F), true);
            else
                _rParams.Resize(2, true);
        }

        void SynetConvolution32fNhwcDirect::ReorderWeight(const float* src, float* dst)
        {
            const ConvParam32f& p = _param;
            const AlgParam & a = _run.At(0).alg;
            for (size_t dc = 0; dc < p.dstC; dc += a.F)
            {
                size_t F = Simd::Min(p.dstC, dc + a.F) - dc;
                const float* psrc = src;
                for (size_t ky = 0; ky < p.kernelY; ++ky)
                {
                    for (size_t kx = 0; kx < p.kernelX; ++kx)
                    {
                        for (size_t sc = 0; sc < p.srcC; ++sc)
                        {
                            size_t f = 0;
                            for (; f < F; ++f)
                                *(dst++) = psrc[f];
                            for (; f < a.F; ++f)
                                *(dst++) = 0.0f;
                            psrc += p.dstC;
                        }
                    }
                }
                src += F;
            }
        }

        void SynetConvolution32fNhwcDirect::OldSetAlgParam(size_t F)
        {
            const ConvParam32f& p = _param;
            AlgParam & a = _old.alg;
            a.F = F;
            a.microD = a.F*2;
            a.macroC = Simd::Min(Base::AlgCacheL1() / sizeof(float) / p.kernelY / p.kernelX / a.microD, p.srcC);
            for (size_t macroH = p.dstH; macroH >= 1; macroH--)
            {
                a.macroH = macroH;
                if (a.macroC * p.srcW * (a.macroH * p.strideY + p.kernelY * p.dilationY - 1) * sizeof(float) <= Base::AlgCacheL2())
                    break;
            }
            a.macroD = Simd::RestrictRange(AlignLoAny(Base::AlgCacheL3() / sizeof(float) / p.kernelY / p.kernelX / a.macroC, a.microD), 
                a.microD, AlignHiAny(p.dstC, a.microD));
            _old.weight.Resize(AlignHiAny(p.dstC, a.microD) * p.kernelY * p.kernelX * p.srcC);
            _rBias.Resize(AlignHiAny(p.dstC, a.microD), true);
            if (p.activation == SimdConvolutionActivationLeakyRelu || p.activation == SimdConvolutionActivationPrelu)
                _rParams.Resize(AlignHiAny(p.dstC, a.microD), true);
            else
                _rParams.Resize(2, true);
        }

        void SynetConvolution32fNhwcDirect::OldReorderWeight(const float* src, float* dst)
        {
            const ConvParam32f& p = _param;
            const AlgParam& a = _old.alg;
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
                                    const float* s = src + ((ky * p.kernelX + kx) * p.srcC + sa + si) * p.dstC + da + di;
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

        bool SynetConvolution32fNhwcDirect::Preferable(const ConvParam32f & p)
        {
            return false;
        }

        //---------------------------------------------------------------------

//#define SIMD_BASE_ONLY_GEMM_NN

        void * SynetConvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm)
        {
            ConvParam32f param(batch, conv, gemm);
            if (!param.Valid())
                return NULL;
#if !defined(SIMD_BASE_ONLY_GEMM_NN)
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
#endif
            else
                return new SynetConvolution32fGemmNN(param);
        }
    }
#endif
}
