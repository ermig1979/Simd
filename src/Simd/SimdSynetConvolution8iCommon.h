/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#ifndef __SimdSynetConvolution8iCommon_h__
#define __SimdSynetConvolution8iCommon_h__

#include "Simd/SimdMath.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSynetActivation.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynetConvParam.h"
#include "Simd/SimdSynetConvolution32fCommon.h"

namespace Simd
{
    enum Term8iType
    {
        Term8iLast8u,
        Term8iLast32f,
        Term8iInterim,
        Term8iSize
    };

    namespace Base
    {
        template<class S, class D, class F> SIMD_INLINE D Convert(S value, F scale, F shift, int lower, int upper)
        {
            return (D)(F(value) * scale + shift);
        }

        template<> SIMD_INLINE uint8_t Convert<float, uint8_t, float>(float value, float scale, float shift, int lower, int upper)
        {
            return (uint8_t)Simd::RestrictRange(Round(value * scale + shift), lower, upper);
        }

        template<> SIMD_INLINE int8_t Convert<float, int8_t, float>(float value, float scale, float shift, int lower, int upper)
        {
            return (int8_t)Simd::RestrictRange(Round(value * scale + shift), lower, upper);
        }

        template<class S, class D, class F> void Convert(const S* src, size_t batch, size_t channels, size_t height, size_t width,
            SimdTensorFormatType format, const F* scale, const F* shift, int lower, int upper, D* dst)
        {
            for (size_t b = 0; b < batch; ++b)
            {
                if (format == SimdTensorFormatNchw)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        F _scale = scale[c];
                        F _shift = shift[c];
                        for (size_t h = 0; h < height; ++h)
                        {
                            for (size_t w = 0; w < width; ++w)
                                dst[w] = Convert<S, D, F>(src[w], _scale, _shift, lower, upper);
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
                                dst[c] = Convert<S, D, F>(src[c], scale[c], shift[c], lower, upper);
                            src += channels;
                            dst += channels;
                        }
                    }
                }
                else
                    assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        inline void ImgToCol(const uint8_t* src, const ConvParam& p, const uint8_t* zero, uint8_t* dst)
        {
            assert(!p.trans);
            size_t srcSize = p.srcW * p.srcH;
            if (p.IsDilation(1) && p.IsStride(2) && p.IsPad(0) && p.IsKernel(1))
            {
                for (size_t channel = 0; channel < p.srcC; ++channel)
                {
                    for (size_t dy = 0; dy < p.dstH; ++dy)
                    {
                        const uint8_t* psrc = src + 2 * dy * p.srcW;
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
                                    const uint8_t* psrc = src + sy * p.srcW;
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

        inline void ImgToRow(const uint8_t* src, const SimdConvolutionParameters & p, const uint8_t* zero, uint8_t* dst)
        {
            assert(p.srcF == SimdTensorFormatNhwc);
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

        inline void GemmNchw(size_t D, size_t S, size_t C, size_t K, const int8_t* wgt, size_t ldw, const uint8_t* src, size_t lds, int32_t* dst, size_t ldd, bool overflow)
        {
            const size_t C2 = overflow ? AlignLo(C, 2) : 0;
            for (size_t i = 0; i < D; ++i)
            {
                for (size_t j = 0; j < S; ++j)
                    dst[j] = 0;
                size_t c = 0;
                for (; c < C2; c += 2)
                {
                    for (size_t k = 0; k < K; k++)
                    {
                        int32_t w0 = wgt[(c + 0) * K + k];
                        int32_t w1 = wgt[(c + 1) * K + k];
                        const uint8_t* s0 = src + ((c + 0) * K + k) * lds;
                        const uint8_t* s1 = src + ((c + 1) * K + k) * lds;
                        for (size_t j = 0; j < S; ++j)
                            dst[j] += Simd::RestrictRange(s0[j] * w0 + s1[j] * w1, SHRT_MIN, SHRT_MAX);
                    }
                }
                for (; c < C; ++c)
                {
                    for (size_t k = 0; k < K; k++)
                    {
                        int32_t w0 = wgt[(c + 0) * K + k];
                        const uint8_t* s0 = src + ((c + 0) * K + k) * lds;
                        for (size_t j = 0; j < S; ++j)
                            dst[j] += s0[j] * w0;
                    }
                }
                wgt += ldw;
                dst += ldd;
            }
        }

        inline void GemmNhwc(size_t S, size_t D, size_t K, size_t C, const uint8_t* src, size_t lds, const int8_t* wgt, size_t ldw, int32_t* dst, size_t ldd, bool overflow)
        {
            const size_t C2 = overflow ? AlignLo(C, 2) : 0;
            for (size_t i = 0; i < S; ++i)
            {
                for (size_t j = 0; j < D; ++j)
                    dst[j] = 0;
                for (size_t k = 0, o = 0; k < K; k++)
                {
                    size_t c = 0;
                    for (; c < C2; c += 2, o += 2)
                    {
                        int32_t s0 = src[o + 0];
                        int32_t s1 = src[o + 1];
                        const int8_t* w0 = wgt + (o + 0) * ldw;
                        const int8_t* w1 = wgt + (o + 1) * ldw;
                        for (size_t j = 0; j < D; ++j)
                            dst[j] += Simd::RestrictRange(s0 * w0[j] + s1 * w1[j], SHRT_MIN, SHRT_MAX);
                    }
                    for (; c < C; ++c, ++o)
                    {
                        int32_t s0 = src[o];
                        const int8_t* w0 = wgt + o * ldw;
                        for (size_t j = 0; j < D; ++j)
                            dst[j] += s0 * w0[j];
                    }
                }
                src += lds;
                dst += ldd;
            }
        }
    }
}
#endif//__SimdSynetConvolution8iCommon_h__
