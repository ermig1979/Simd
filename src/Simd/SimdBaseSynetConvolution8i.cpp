/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_PERFORMANCE_STATISTIC)
    Base::PerformanceMeasurer * SynetConvolution8i::Perf(const String& func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

    namespace Base
    {
        SIMD_INLINE int Quantize(float value)
        {
            return (int)(value + (value >= 0 ? 0.5f : -0.5f));
        }

        SIMD_INLINE uint8_t Convert32fTo8u(float value, float scale, float shift)
        {
            return (uint8_t)Simd::RestrictRange(Quantize(value * scale + shift), 0, 255);
        }

        void Convert32fTo8u(const float * src, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, uint8_t * dst)
        {
            if (format == SimdTensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float _scale = scale[c];
                    float _shift = shift[c];
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                            dst[w] = Convert32fTo8u(src[w], _scale, _shift);
                        src += width;
                        dst += width;
                    }
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                for (size_t h = 0; h < height; ++h)
                {
                    for (size_t w = 0; w < width; ++w)
                    {
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = Convert32fTo8u(src[c], scale[c], shift[c]);
                        src += channels;
                        dst += channels;
                    }
                }
            }
            else
                assert(0);
        }

        template<class S, class D> SIMD_INLINE D Convert(S value, float scale, float shift)
        {
            return (D)(float(value) * scale + shift);
        }

        template<> SIMD_INLINE uint8_t Convert<float, uint8_t>(float value, float scale, float shift)
        {
            return (uint8_t)Simd::RestrictRange(Quantize(value * scale + shift), 0, 255);
        }

        template<> SIMD_INLINE int8_t Convert<float, int8_t>(float value, float scale, float shift)
        {
            return (int8_t)Simd::RestrictRange(Quantize(value * scale + shift), -128, 127);
        }

        template<class S, class D> void Convert(const S * src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, D * dst)
        {
            for (size_t b = 0; b < batch; ++b)
            {
                if (format == SimdTensorFormatNchw)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        float _scale = scale[c];
                        float _shift = shift[c];
                        for (size_t h = 0; h < height; ++h)
                        {
                            for (size_t w = 0; w < width; ++w)
                                dst[w] = Convert<S, D>(src[w], _scale, _shift);
                            src += width;
                            dst += width;
                        }
                    }
                }
                else if (format == SimdTensorFormatNhwc)
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                        {
                            for (size_t c = 0; c < channels; ++c)
                                dst[c] = Convert<S, D>(src[c], scale[c], shift[c]);
                            src += channels;
                            dst += channels;
                        }
                    }
                }
                else
                    assert(0);
            }
        }

        SynetConvolution8iGemmNN::SynetConvolution8iGemmNN(const ConvParam8i& p)
            : SynetConvolution8i(p)
        {
            if (p.IsDilation(1) && p.IsStride(1) && p.IsPad(0))
            {
                _skipConv = p.IsKernel(1) || (p.srcH == p.kernelY && p.srcW == p.kernelX);
            }
            else
                _skipConv = false;
            _sizeS = p.srcC * p.srcH * p.srcW;
            _sizeB = p.srcC * p.kernelY * p.kernelX * p.dstH * p.dstW;
            _sizeD = p.dstC * p.dstH * p.dstW;
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
            _siD = p.dstC;
            _siS = p.dstH * p.dstW;
            _merge = 1;
            _batch = p.batch;
            _src8u = p.srcT == SimdTensorData8u;
            _dst8u = p.dstT == SimdTensorData8u;
            _zero8u.Resize(p.srcC);
            _weight8i.Resize(p.srcC * p.kernelY * p.kernelX / p.group * p.dstC);
            _norm32i.Resize(2 * p.dstC);
            _norm32f.Resize(2 * p.dstC);
        }

        size_t SynetConvolution8iGemmNN::InternalBufferSize() const
        {
            return SynetConvolution8i::InternalBufferSize() + 
                _weight8i.size * sizeof(int8_t) +
                _zero8u.size * sizeof(uint8_t) + 
                _norm32i.size * sizeof(int32_t) + 
                (_norm32f.size + _srcScale.size + _srcShift.size + _dstScale.size + _dstShift.size)* sizeof(float);
        }

        size_t SynetConvolution8iGemmNN::ExternalBufferSize() const
        {
            size_t size = _sizeD * _merge * sizeof(int32_t);
            if (!_src8u)
                size += _sizeS * _merge * sizeof(uint8_t);
            if(!_skipConv)
                size += _sizeB * _merge * sizeof(uint8_t);
            return size;
        }

        void SynetConvolution8iGemmNN::SetParams(const float* weight, const float* bias, const float* params, const float* const* stats)
        {

        }

        void SynetConvolution8iGemmNN::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const ConvParam8i& p = _param;
            const int8_t * weight = _weight8i.data;
            const int32_t * scale = _norm32i.data;
            const int32_t * shift = scale + p.dstC;
            buf = Buffer(buf);
            uint8_t * src8u = _src8u ? NULL : Allocate<uint8_t>(buf, _sizeS*_merge);
            int32_t * sum32i = Allocate<int32_t>(buf, _sizeD * _merge);
            for (size_t b = 0; b < _batch; b += _merge)
            {
                if (!_src8u)
                    Convert<float, uint8_t>((float*)src + b * _sizeS, _merge, p.srcC, p.srcH, p.srcW, p.srcF, _srcScale.data, _srcShift.data, src8u);
                const uint8_t * pSrc = _src8u ? src + b * _sizeS : src8u;
                int32_t* pSum = sum32i;
                if (!_skipConv)
                {
                    if(p.trans)
                        for (size_t m = 0; m < _merge; ++m)
                            ImgToRow(pSrc + m * _sizeS, buf + m * _sizeB);
                    else
                        for (size_t m = 0; m < _merge; ++m)
                            ImgToCol(pSrc + m * _sizeS, buf + m * _sizeB);
                    pSrc = buf;
                }
                if (_merge > 1)
                {
                    assert(0);
                }
                else
                {
                    for (size_t g = 0; g < p.group; ++g)
                    {
                        if (p.trans)
                            GemmNN(_siS, _siD, _siK, _siC, pSrc + _grS * g, _ldS, weight + _grW * g, _ldW, pSum + _grD * g, _ldD);
                        else
                            GemmNN(_siD, _siS, _siC, _siK, weight + _grW * g, _ldW, pSrc + _grS * g, _ldS, pSum + _grD * g, _ldD);
                    }
                }
                if (_dst8u)
                    Convert<int32_t, uint8_t>(sum32i, _merge, p.dstC, p.dstH, p.dstW, p.dstF, _dstScale.data, _dstShift.data, dst + b * _sizeD);
                else
                    Convert<int32_t, float>(sum32i, _merge, p.dstC, p.dstH, p.dstW, p.dstF, _dstScale.data, _dstShift.data, (float*)dst + b * _sizeD);
            }
        }

        void SynetConvolution8iGemmNN::ImgToCol(const uint8_t* src, uint8_t* dst)
        {
            const ConvParam8i& p = _param;
            assert(!p.trans);
            size_t srcSize = p.srcW * p.srcH;
            const uint8_t* zero = _zero8u.data;
            if (p.IsDilation(1) && p.IsStride(2) && p.IsPad(0) && p.IsKernel(1))
            {
                for (size_t channel = 0; channel < p.srcC; ++channel)
                {
                    for (size_t dy = 0; dy < p.dstH; ++dy)
                    {
                        const uint8_t * psrc = src + 2 * dy * p.srcW;
                        for (size_t dx = 0, sx = 0; dx < p.dstW; ++dx, sx += 2)
                            *(dst++) = psrc[sx];
                    }
                    src += srcSize;
                }
            }
            else if (p.IsDilation(1) && p.IsStride(1))
            {
                const ptrdiff_t bodySize = p.dstW - p.padX - p.padW;
                for (size_t channel = 0; channel < p.srcC; ++channel)
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
                                    const uint8_t * psrc = src + sy * p.srcW;
                                    for (; dx < p.padX; ++dx, ++sx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = psrc[sx];
                                        else
                                            *(dst++) = zero[channel];
                                    }
                                    if (bodySize > 0)
                                    {
                                        memcpy(dst, psrc + sx, bodySize * sizeof(uint8_t));
                                        dst += bodySize;
                                        dx += bodySize;
                                        sx += bodySize;
                                    }
                                    for (; dx < p.dstW; ++dx, ++sx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = psrc[sx];
                                        else
                                            *(dst++) = zero[channel];
                                    }
                                }
                                else
                                {
                                    for (size_t dx = 0; dx < p.dstW; ++dx)
                                        *(dst++) = zero[channel];
                                }
                            }
                        }
                    }
                    src += srcSize;
                }
            }
            else
            {
                for (size_t channel = 0; channel < p.srcC; ++channel)
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
                                            *(dst++) = zero[channel];
                                        sx += p.strideX;
                                    }
                                }
                                else
                                {
                                    for (size_t dx = 0; dx < p.dstW; ++dx)
                                        *(dst++) = zero[channel];
                                }
                                sy += p.strideY;
                            }
                        }
                    }
                    src += srcSize;
                }
            }
        }

        void SynetConvolution8iGemmNN::ImgToRow(const uint8_t* src, uint8_t* dst)
        {
            const ConvParam8i& p = _param;
            assert(p.trans);
            size_t size = p.srcC / p.group;
            const uint8_t* zero = _zero8u.data;
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
                                        memcpy(dst, src + (sy * p.srcW + sx) * p.srcC, size * sizeof(uint8_t));
                                        dst += size;
                                    }
                                    else
                                    {
                                        memcpy(dst, zero, size * sizeof(uint8_t));
                                        dst += size;
                                    }
                                }
                            }
                            else
                            {
                                for (size_t kx = 0; kx < p.kernelX; kx++)
                                {
                                    memcpy(dst, zero, size * sizeof(uint8_t));
                                    dst += size;
                                }
                            }
                        }
                    }
                }
                src += size;
                zero += size;
            }
        }

        void SynetConvolution8iGemmNN::GemmNN(size_t S, size_t D, size_t K, size_t C,
            const uint8_t* src, size_t lda, const int8_t* weight, size_t ldb, int32_t* dst, size_t ldc)
        {

        }

        void SynetConvolution8iGemmNN::GemmNN(size_t D, size_t S, size_t C, size_t K,
            const int8_t* weight, size_t lda, const uint8_t* src, size_t ldb, int32_t* dst, size_t ldc)
        {

        }

        //---------------------------------------------------------------------

//#define SIMD_BASE_ONLY_GEMM_NN

        void * SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters * conv)
        {
            ConvParam8i param(batch, conv);
            if (!param.Valid())
                return NULL;
#if !defined(SIMD_BASE_ONLY_GEMM_NN)
#endif
            else
                return new SynetConvolution8iGemmNN(param);
        }
    }
}
