/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#include "Simd/SimdConvolution.h"
#include "Simd/SimdBase.h"

namespace Simd
{
    namespace Base
    {
        static void BiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType type, const float * params, float * dst)
        {
            if (type == ::SimdConvolutionActivationIdentity)
            {
                if(bias)
                    SynetAddBias(bias, count, size, dst);
            }
            else if (type == ::SimdConvolutionActivationRelu)
            {
                if (bias)
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        float shift = bias[i];
                        for (size_t j = 0; j < size; ++j)
                            dst[j] = Simd::Max(0.0f, dst[j] + shift);
                        dst += size;
                    }
                }
                else
                {
                    float slope = 0;
                    NeuralRelu(dst, size*count, &slope, dst);
                }
            }
            else if (type == ::SimdConvolutionActivationLeakyRelu)
            {
                float slope = params[0];
                if (bias)
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        float shift = bias[i];
                        for (size_t j = 0; j < size; ++j)
                        {
                            float value = dst[j] + shift;
                            dst[i] = Simd::Max(0.0f, value) + slope*Simd::Min(value, 0.0f);
                        }
                        dst += size;
                    }
                }
                else
                    NeuralRelu(dst, size*count, &slope, dst);
            }
            else if (type == ::SimdConvolutionActivationRestrictRange)
            {
                float lower = params[0];
                float upper = params[1];
                if (bias)
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        float shift = bias[i];
                        for (size_t j = 0; j < size; ++j)
                            dst[j] = Simd::RestrictRange(dst[j] + shift, lower, upper);
                        dst += size;
                    }
                }
                else
                    SynetRestrictRange(dst, size*count, &lower, &upper, dst);
            }
            else if (type == ::SimdConvolutionActivationPrelu)
            {
                if (bias)
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        float shift = bias[i];
                        float slope = params[i];
                        for (size_t j = 0; j < size; ++j)
                        {
                            float value = dst[j] + shift;
                            dst[i] = Simd::Max(0.0f, value) + slope*Simd::Min(value, 0.0f);
                        }
                        dst += size;
                    }
                }
                else
                {
                    for (size_t i = 0; i < count; ++i)
                        NeuralRelu(dst + i*size, size, params + i, dst + i*size);
                }
            }
        }

        ConvolutionImgToCol::ConvolutionImgToCol(const ConvParam & p)
            : Convolution(p)
        {
            _is1x1 = p.IsKernel(1) && p.IsDilation(1) && p.IsStride(1) && p.IsPad(0);
            _M = p.dstC / p.group;
            _N = p.dstH  * p.dstW;
            _K = p.srcC * p.kernelY * p.kernelX / p.group;
            _weightStep = p.dstC * _K / p.group;
            _srcStep = _K * _N;
            _dstStep = p.dstC * _N / p.group;
        }

        size_t ConvolutionImgToCol::BufferSize() const
        {
            if (_is1x1)
                return 1;
            else
            {
                const ConvParam & p = _param;
                return p.srcC*p.kernelY*p.kernelX*p.dstH*p.dstW;
            }
        };

        void ConvolutionImgToCol::SetWeight(const float * weight, const float * bias, SimdBool * internal)
        {
            _weight = weight;
            _bias = bias;
            if (internal)
                *internal = SimdFalse;
        }

        void ConvolutionImgToCol::Forward(const float * src, float * buf, float * dst)
        {
            if (!_is1x1)
            {
                buf = Buffer(buf);
                ImgToCol(src, _param, buf);
                src = buf;
            }
            GemmAndBias(src, dst);
        }

        void ConvolutionImgToCol::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
                Base::Gemm32fNN(_M, _N, _K, &_1, _weight + _weightStep * g, _K, src + _srcStep * g, _N, &_0, dst + _dstStep * g, _N);
            BiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, _activationType, _activationParams, dst);
        }

        void ConvolutionImgToCol::ImgToCol(const float * src, const ConvParam & p, float * dst)
        {
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

        //---------------------------------------------------------------------

        ConvolutionImgToRow::ConvolutionImgToRow(const ConvParam & p)
            : Convolution(p)
        {
            _M = p.dstC / p.group;
            _N = p.dstH  * p.dstW;
            _K = p.srcC * p.kernelY * p.kernelX / p.group;
            _weightStep = p.dstC * _K / p.group;
            _srcStep = _K * _N;
            _dstStep = p.dstC * _N / p.group;
        }

        size_t ConvolutionImgToRow::BufferSize() const
        {
            const ConvParam & p = _param;
            return p.srcC*p.kernelY*p.kernelX*p.dstH*p.dstW;
        };

        void ConvolutionImgToRow::SetWeight(const float * weight, const float * bias, SimdBool * internal)
        {
            _weight = weight;
            _bias = bias;
            if (internal)
                *internal = SimdFalse;
        }

        void ConvolutionImgToRow::Forward(const float * src, float * buf, float * dst)
        {
            buf = Buffer(buf);
            ImgToRow(src, _param, buf);
            GemmAndBias(buf, dst);
        }

        bool ConvolutionImgToRow::Preferable(const ConvParam & p)
        {
            return p.srcH < 6 && p.srcW < 6 && p.group == 1;
        }

        void ConvolutionImgToRow::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
                Base::Gemm32fNT(_M, _N, _K, &_1, _weight + _weightStep * g, _K, src + _srcStep * g, _K, &_0, dst + _dstStep * g, _N);
            BiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, _activationType, _activationParams, dst);
        }

        void ConvolutionImgToRow::ImgToRow(const float * src, const ConvParam & p, float * dst)
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

        ConvolutionWinograd2x3p::ConvolutionWinograd2x3p(const ConvParam & p)
            : Convolution(p)
            , _block(2)
        {
            _count = Simd::Square(_block + p.kernelX - 1);
            _tileH = (p.dstH + _block - 1) / _block;
            _tileW = (p.dstW + _block - 1) / _block;
            _strideW = p.srcC * p.dstC;
            _strideS = p.srcC * _tileH * _tileW;
            _strideD = p.dstC * _tileH * _tileW;
            _M = p.dstC;
            _N = _tileW * _tileH;
            _K = p.srcC;
            _pad = (int)p.padX;
        }
        
        size_t ConvolutionWinograd2x3p::BufferSize() const
        {
            return (_strideS + _strideD)*_count;
        }
        
        void ConvolutionWinograd2x3p::SetWeight(const float * weight, const float * bias, SimdBool * internal)
        {
            const ConvParam & p = _param;
            _weight.Resize(_strideW*_count);
            Base::Winograd2x3pSetFilter(weight, p.srcC*p.dstC, _weight.data);
            _bias = bias;
            if (internal)
                *internal = SimdTrue;
        }
        
        void ConvolutionWinograd2x3p::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam & p = _param;
            float * bufS = Buffer(buf);
            float * bufD = bufS + _strideS * _count;
            Base::Winograd2x3pSetInput(src, p.srcC, p.srcH, p.srcW, buf, _pad);
            for (size_t i = 0; i < _count; ++i)
                Base::Gemm32fNN(_M, _N, _K, &_1, _weight.data + i * _strideW, _K, bufS + i * _strideS, _N, &_0, bufD + i * _strideD, _N);
            Base::Winograd2x3pSetOutput(bufD, dst, p.dstC, p.dstH, p.dstW);
            BiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, _activationType, _activationParams, dst);
        }

        bool ConvolutionWinograd2x3p::Preferable(const ConvParam & p)
        {
            return p.IsKernel(3) && p.IsDilation(1) && p.IsStride(1) && (p.IsPad(0) || p.IsPad(1)) && p.group == 1 && p.srcC > 16 && p.srcH >= 6 && p.srcW >= 6;
        }

        //---------------------------------------------------------------------

        ConvolutionDirect::ConvolutionDirect(const ConvParam & p)
            : Convolution(p)
        {
            _srcC = p.srcC / p.group;
            _srcH = p.padY + p.srcH + p.padH;
            _srcW = p.padX + p.srcW + p.padW;
            _dstC = p.dstC / p.group;
            _weightStep = _srcC * _dstC * p.kernelY * p.kernelX;
            _srcStep = _srcC * p.srcH * p.srcW;
            _dstStep = _dstC * p.dstH  * p.dstW;
            _pad = p.IsPad(0) ? 0 : 1;
        }

        size_t ConvolutionDirect::BufferSize() const
        {
            if (_pad)
                return _srcC*_srcH*_srcW;
            else
                return 1;
        }

        void ConvolutionDirect::SetWeight(const float * weight, const float * bias, SimdBool * internal)
        {
            _weight = weight;
            _bias = bias;
            if (internal)
                *internal = SimdFalse;
        }

        void ConvolutionDirect::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam & p = _param;
            const float * weight = _weight;
            const float * bias = _bias;
            const float * params = _activationParams;
            if(_pad)
                buf = Buffer(buf);
            for (size_t g = 0; g < p.group; ++g)
            {
                if (_pad)
                {
                    Pad(src, buf);
                    ConvolutionAndBias(buf, weight, bias, params, dst);
                }
                else
                    ConvolutionAndBias(src, weight, bias, params, dst);
                weight += _weightStep;
                if(bias)
                    bias += _dstC;
                if (_activationType == ::SimdConvolutionActivationPrelu)
                    params += _dstC;
                src += _srcStep;
                dst += _dstStep;
            }
        }

        bool ConvolutionDirect::Preferable(const ConvParam & p)
        {
            if (!p.IsDilation(1))
                return false;
            if (!(p.IsStride(1) || p.IsStride(2)))
                return false;
            double k = double(p.srcC) / p.group * p.strideX * p.strideY / p.kernelX / p.kernelY;
            return k < 2.0 && (p.IsKernel(2) || p.IsKernel(3));
        }

        void ConvolutionDirect::Pad(const float * src, float * dst) const
        {
            const ConvParam & p = _param;
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

        void ConvolutionDirect::ConvolutionAndBias(const float * src, const float * weight, const float * bias, const float * params, float * dst) const
        {
            const ConvParam & p = _param;
            for (size_t dc = 0; dc < _dstC; ++dc)
            {
                Fill32f(dst, p.dstW * p.dstH, bias ? bias + dc : NULL);
                for (size_t sc = 0; sc < _srcC; ++sc)
                {
                    const float * ps = src + sc * _srcW * _srcH;
                    const float * pw = weight + (dc*_srcC + sc)*p.kernelX*p.kernelY;
                    float * pd = dst;
                    if (p.IsKernel(1))
                        AddConvolutionKernel1x1(ps, _srcW, p.strideY, p.strideX, pw, pd, p.dstH, p.dstW);
                    else if (p.IsKernel(2))
                        AddConvolutionKernel2x2(ps, _srcW, p.strideY, p.strideX, pw, pd, p.dstH, p.dstW);
                    else if (p.IsKernel(3))
                        AddConvolutionKernel3x3(ps, _srcW, p.strideY, p.strideX, pw, pd, p.dstH, p.dstW);
                    else
                    {
                        for (size_t dy = 0; dy < p.dstH; ++dy)
                        {
                            for (size_t dx = 0, sx = 0; dx < p.dstW; ++dx, sx += p.strideX)
                            {
                                float sum = 0;
                                for (size_t ky = 0; ky < p.kernelY; ++ky)
                                {
                                    const float * s = ps + ky * _srcW + sx;
                                    const float * w = pw + p.kernelX*ky;
                                    for (size_t kx = 0; kx < p.kernelX; ++kx)
                                        sum += s[kx] * w[kx];
                                }
                                pd[dx] += sum;
                            }
                            ps += _srcW * p.strideY;
                            pd += p.dstW;
                        }
                    }
                }
                BiasAndActivation(NULL, 1, p.dstH*p.dstW, _activationType, params, dst);
                dst += p.dstW * p.dstH;
            }
        }
        //---------------------------------------------------------------------

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group)
        {
            ConvParam param(srcC, srcH, srcW, dstC, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group);
            if(ConvolutionWinograd2x3p::Preferable(param))
                return new ConvolutionWinograd2x3p(param);
            else if (ConvolutionImgToRow::Preferable(param))
                return new ConvolutionImgToRow(param);
            else if (ConvolutionDirect::Preferable(param))
                return new ConvolutionDirect(param);
            else
                return new ConvolutionImgToCol(param);
        }
    }
}
