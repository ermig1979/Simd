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
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
    namespace Base
    {
        void SynetAddBiasNchw(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(spatial, 4);
            for (size_t c = 0; c < channels; ++c)
            {
                float value = bias[c];
                size_t s = 0;
                for (; s < aligned; s += 4)
                {
                    dst[s + 0] += value;
                    dst[s + 1] += value;
                    dst[s + 2] += value;
                    dst[s + 3] += value;
                }
                for (; s < spatial; ++s)
                    dst[s] += value;
                dst += spatial;
            }
        }

        void SynetAddBiasNhwc(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(channels, 4);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < aligned; c += 4)
                {
                    dst[c + 0] += bias[c + 0];
                    dst[c + 1] += bias[c + 1];
                    dst[c + 2] += bias[c + 2];
                    dst[c + 3] += bias[c + 3];
                }
                for (; c < channels; ++c)
                    dst[c] += bias[c];
                dst += channels;
            }
        }

        template<int N> void SynetAddBiasNchwXc(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            for (size_t c = 0; c < channels; c += N)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t i = 0; i < N; ++i)
                        dst[i] += bias[i];
                    dst += N;
                }
                bias += N;
            }
        }

        void SynetAddBias(const float * bias, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetAddBiasNchw(bias, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetAddBiasNhwc(bias, channels, spatial, dst);
            else if(format == SimdTensorFormatNchw4c)
                SynetAddBiasNchwXc<4>(bias, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw8c)
                SynetAddBiasNchwXc<8>(bias, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw16c)
                SynetAddBiasNchwXc<16>(bias, channels, spatial, dst);
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        template <SimdSynetEltwiseOperationType type> void SynetEltwiseLayerForward(float const * const * src, size_t count, size_t size, float * dst)
        {
            size_t aligned = Simd::AlignLo(size, 4);
            const float * src0 = src[0];
            const float * src1 = src[1];
            size_t j = 0;
            for (; j < aligned; j += 4)
            {
                dst[j + 0] = SynetEltwiseLayerForward<type>(src0[j + 0], src1[j + 0]);
                dst[j + 1] = SynetEltwiseLayerForward<type>(src0[j + 1], src1[j + 1]);
                dst[j + 2] = SynetEltwiseLayerForward<type>(src0[j + 2], src1[j + 2]);
                dst[j + 3] = SynetEltwiseLayerForward<type>(src0[j + 3], src1[j + 3]);
            }
            for (; j < size; ++j)
                dst[j] = SynetEltwiseLayerForward<type>(src0[j], src1[j]);
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                for (j = 0; j < aligned; j += 4)
                {
                    dst[j + 0] = SynetEltwiseLayerForward<type>(dst[j + 0], srci[j + 0]);
                    dst[j + 1] = SynetEltwiseLayerForward<type>(dst[j + 1], srci[j + 1]);
                    dst[j + 2] = SynetEltwiseLayerForward<type>(dst[j + 2], srci[j + 2]);
                    dst[j + 3] = SynetEltwiseLayerForward<type>(dst[j + 3], srci[j + 3]);
                }
                for (; j < size; ++j)
                    dst[j] = SynetEltwiseLayerForward<type>(dst[j], srci[j]);
            }
        }

        void SynetEltwiseLayerForwardSum(float const * const * src, const float * weight, size_t count, size_t size, float * dst)
        {
            size_t aligned = Simd::AlignLo(size, 4);
            const float * src0 = src[0];
            const float * src1 = src[1];
            float weight0 = weight[0], weight1 = weight[1];
            size_t j = 0;
            for (; j < aligned; j += 4)
            {
                dst[j + 0] = src0[j + 0] * weight0 + src1[j + 0] * weight1;
                dst[j + 1] = src0[j + 1] * weight0 + src1[j + 1] * weight1;
                dst[j + 2] = src0[j + 2] * weight0 + src1[j + 2] * weight1;
                dst[j + 3] = src0[j + 3] * weight0 + src1[j + 3] * weight1;
            }
            for (; j < size; ++j)
                dst[j] = src0[j] * weight0 + src1[j] * weight1;
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                float weighti = weight[i];
                for (j = 0; j < aligned; j += 4)
                {
                    dst[j + 0] += srci[j + 0] * weighti;
                    dst[j + 1] += srci[j + 1] * weighti;
                    dst[j + 2] += srci[j + 2] * weighti;
                    dst[j + 3] += srci[j + 3] * weighti;
                }
                for (; j < size; ++j)
                    dst[j] += srci[j] * weighti;
            }
        }

        void SynetEltwiseLayerForward(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst)
        {
            switch (type)
            {
            case SimdSynetEltwiseOperationProduct:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationSum:
                SynetEltwiseLayerForwardSum(src, weight, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMax:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMin:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin>(src, count, size, dst);
                break;
            default:
                assert(0);
            }
        }

        //---------------------------------------------------------------------

        void SynetInnerProductLayerForward(const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst)
        {
            size_t aligned = Simd::AlignLo(size, 4);
            for (size_t i = 0; i < count; ++i)
            {
                size_t j = 0;
                float sums[4] = { 0, 0, 0, 0 };
                for (; j < aligned; j += 4)
                {
                    sums[0] += src[j + 0] * weight[j + 0];
                    sums[1] += src[j + 1] * weight[j + 1];
                    sums[2] += src[j + 2] * weight[j + 2];
                    sums[3] += src[j + 3] * weight[j + 3];
                }
                for (; j < size; ++j)
                    sums[0] += src[j] * weight[j];
                dst[i] = sums[0] + sums[1] + sums[2] + sums[3] + (bias ? bias[i] : 0);
                weight += size;
            }
        }

        void SynetLrnLayerCrossChannels(const float * src, size_t half, size_t count, size_t size, const float * k, float * dst, SimdBool trans)
        {
            float k0 = k[0], k1 = k[1], k2 = k[2];
            if (trans)
            {
                size_t beg = half + 1;
                size_t end = count - half;
                for (size_t j = 0; j < size; ++j)
                {
                    float sum = 0;
                    for (size_t i = 0; i < half; ++i)
                        sum += Simd::Square(src[i]);
                    for (size_t i = 0; i < beg; ++i)
                    {
                        sum += Simd::Square(src[i + half]);
                        dst[i] = src[i] * Pow(k0 + k1 * sum, k2);
                    }
                    for (size_t i = beg; i < end; ++i)
                    {
                        sum += Simd::Square(src[i + half]);
                        sum -= Simd::Square(src[i - half - 1]);
                        dst[i] = src[i] * Pow(k0 + k1 * sum, k2);
                    }
                    for (size_t i = end; i < count; ++i)
                    {
                        sum -= Simd::Square(src[i - half - 1]);
                        dst[i] = src[i] * Pow(k0 + k1 * sum, k2);
                    }
                    src += count;
                    dst += count;
                }
            }
            else
            {
                Array32f sum(size, true), zero(size, true);
                for (size_t i = 0; i < half; ++i)
                {
                    const float * pos = src + i * size;
                    for (size_t j = 0; j < size; ++j)
                        sum[j] += Simd::Square(pos[j]);
                }
                for (size_t i = 0; i < count; ++i)
                {
                    const float * pos = (i < count - half) ? src + half * size : zero.data;
                    const float * neg = (i > half) ? src - (half + 1) * size : zero.data;
                    for (size_t j = 0; j < size; ++j)
                    {
                        sum[j] += Simd::Square(pos[j]);
                        sum[j] -= Simd::Square(neg[j]);
                        dst[j] = src[j] * Pow(k0 + k1 * sum[j], k2);
                    }
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetPoolingForwardMax(const float * src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, float * dst, size_t dstH, size_t dstW, SimdBool trans)
        {
            if (trans)
            {
                for (size_t ph = 0; ph < dstH; ++ph)
                {
                    size_t hStart = ph * strideY - padY;
                    size_t hEnd = Simd::Min(hStart + kernelY, srcH);
                    hStart = Simd::Max<ptrdiff_t>(0, hStart);
                    for (size_t pw = 0; pw < dstW; ++pw)
                    {
                        size_t wStart = pw * strideX - padX;
                        size_t wEnd = Simd::Min(wStart + kernelX, srcW);
                        wStart = Simd::Max<ptrdiff_t>(0, wStart);
                        for (size_t c = 0; c < srcC; ++c)
                            dst[c] = -FLT_MAX;
                        for (size_t h = hStart; h < hEnd; ++h)
                        {
                            for (size_t w = wStart; w < wEnd; ++w)
                            {
                                const float * pc = src + (h * srcW + w)*srcC;
                                for (size_t c = 0; c < srcC; ++c)
                                    dst[c] = Simd::Max(dst[c], pc[c]);
                            }
                        }
                        dst += srcC;
                    }
                }
            }
            else
            {
                for (size_t c = 0; c < srcC; ++c)
                {
                    for (size_t ph = 0; ph < dstH; ++ph)
                    {
                        size_t hStart = ph * strideY - padY;
                        size_t hEnd = Simd::Min(hStart + kernelY, srcH);
                        hStart = Simd::Max<ptrdiff_t>(0, hStart);
                        for (size_t pw = 0; pw < dstW; ++pw)
                        {
                            size_t wStart = pw * strideX - padX;
                            size_t wEnd = Simd::Min(wStart + kernelX, srcW);
                            wStart = Simd::Max<ptrdiff_t>(0, wStart);
                            float max = -FLT_MAX;
                            for (size_t h = hStart; h < hEnd; ++h)
                                for (size_t w = wStart; w < wEnd; ++w)
                                    max = Simd::Max(max, src[h * srcW + w]);
                            dst[ph*dstW + pw] = max;
                        }
                    }
                    src += srcW * srcH;
                    dst += dstW * dstH;
                }
            }
        }

        void SynetPreluLayerForward(const float * src, const float * slope, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if ((trans || size == 1) && count != 1)
            {
                size_t aligned = Simd::AlignLo(count, 4);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    for (; i < aligned; i += 4)
                    {
                        dst[i + 0] = SynetPreluLayerForward(src[i + 0], slope[i + 0]);
                        dst[i + 1] = SynetPreluLayerForward(src[i + 1], slope[i + 1]);
                        dst[i + 2] = SynetPreluLayerForward(src[i + 2], slope[i + 2]);
                        dst[i + 3] = SynetPreluLayerForward(src[i + 3], slope[i + 3]);
                    }
                    for (; i < count; ++i)
                        dst[i] = SynetPreluLayerForward(src[i], slope[i]);
                    src += count;
                    dst += count;
                }
            }
            else
            {
                size_t aligned = Simd::AlignLo(size, 4);
                for (size_t i = 0; i < count; ++i)
                {
                    float s = slope[i];
                    size_t j = 0;
                    for (; j < aligned; j += 4)
                    {
                        dst[j + 0] = SynetPreluLayerForward(src[j + 0], s);
                        dst[j + 1] = SynetPreluLayerForward(src[j + 1], s);
                        dst[j + 2] = SynetPreluLayerForward(src[j + 2], s);
                        dst[j + 3] = SynetPreluLayerForward(src[j + 3], s);
                    }
                    for (; j < size; ++j)
                        dst[j] = SynetPreluLayerForward(src[j], s);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetRestrictRange(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            float min = *lower;
            float max = *upper;
            for (size_t i = 0; i < size; ++i)
                 *dst++ = Simd::RestrictRange(*src++, min, max);
        }

        //---------------------------------------------------------------------

        void SynetScaleLayerForwardNchw(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(spatial, 4);
            if (bias)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float _scale = scale[c];
                    float _bias = bias[c];
                    size_t s = 0;
                    for (; s < aligned; s += 4)
                    {
                        dst[s + 0] = src[s + 0] * _scale + _bias;
                        dst[s + 1] = src[s + 1] * _scale + _bias;
                        dst[s + 2] = src[s + 2] * _scale + _bias;
                        dst[s + 3] = src[s + 3] * _scale + _bias;
                    }
                    for (; s < spatial; ++s)
                        dst[s] = src[s] * _scale + _bias;
                    src += spatial;
                    dst += spatial;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float _scale = scale[c];
                    size_t s = 0;
                    for (; s < aligned; s += 4)
                    {
                        dst[s + 0] = src[s + 0] * _scale;
                        dst[s + 1] = src[s + 1] * _scale;
                        dst[s + 2] = src[s + 2] * _scale;
                        dst[s + 3] = src[s + 3] * _scale;
                    }
                    for (; s < spatial; ++s)
                        dst[s] = src[s] * _scale;
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        void SynetScaleLayerForwardNhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(channels, 4);
            if (bias)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < aligned; c += 4)
                    {
                        dst[c + 0] = src[c + 0] * scale[c + 0] + bias[c + 0];
                        dst[c + 1] = src[c + 1] * scale[c + 1] + bias[c + 1];
                        dst[c + 2] = src[c + 2] * scale[c + 2] + bias[c + 2];
                        dst[c + 3] = src[c + 3] * scale[c + 3] + bias[c + 3];
                    }
                    for (; c < channels; ++c)
                        dst[c] = src[c] * scale[c] + bias[c];
                    src += channels;
                    dst += channels;

                }
            }
            else
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < aligned; c += 4)
                    {
                        dst[c + 0] = src[c + 0] * scale[c + 0];
                        dst[c + 1] = src[c + 1] * scale[c + 1];
                        dst[c + 2] = src[c + 2] * scale[c + 2];
                        dst[c + 3] = src[c + 3] * scale[c + 3];
                    }
                    for (; c < channels; ++c)
                        dst[c] = src[c] * scale[c];
                    src += channels;
                    dst += channels;
                }
            }
        }

        template<int N> void SynetScaleLayerForwardNchwXc(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (bias)
            {
                for (size_t c = 0; c < channels; c += N)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        for (size_t i = 0; i < N; ++i)
                            dst[i] = src[i]*scale[i] + bias[i];
                        src += N;
                        dst += N;
                    }
                    scale += N;
                    bias += N;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; c += N)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        for (size_t i = 0; i < N; ++i)
                            dst[i] = src[i] * scale[i];
                        src += N;
                        dst += N;
                    }
                    scale += N;
                }
            }
        }

        void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetScaleLayerForwardNchw(src, scale, bias, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetScaleLayerForwardNhwc(src, scale, bias, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetScaleLayerForwardNchwXc<4>(src, scale, bias, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw8c)
                SynetScaleLayerForwardNchwXc<8>(src, scale, bias, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw16c)
                SynetScaleLayerForwardNchwXc<16>(src, scale, bias, channels, spatial, dst);
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        void SynetSoftmaxLayerForward(const float * src, size_t outer, size_t count, size_t inner, float * dst)
        {
            if (inner == 1 && count == 2)
            {
                for (size_t o = 0; o < outer; ++o)
                {
                    float max = Simd::Max(src[0], src[1]);
                    float exp0 = ::exp(src[0] - max);
                    float exp1 = ::exp(src[1] - max);
                    float sum = exp0 + exp1;
                    dst[0] = exp0 / sum;
                    dst[1] = exp1 / sum;
                    src += 2;
                    dst += 2;
                }
            }
            else
            {
                Array32f tmp(inner * 2);
                const float * s;
                float * max = tmp.data, *sum = tmp.data + inner, *d;
                for (size_t o = 0; o < outer; ++o)
                {
                    for (size_t i = 0; i < inner; ++i)
                        max[i] = src[i];
                    s = src + inner;
                    for (size_t c = 1; c < count; ++c)
                    {
                        for (size_t i = 0; i < inner; ++i)
                            max[i] = Simd::Max(max[i], s[i]);
                        s += inner;
                    }

                    s = src;
                    d = dst;
                    for (size_t i = 0; i < inner; ++i)
                        sum[i] = 0;
                    for (size_t c = 0; c < count; ++c)
                    {
                        for (size_t i = 0; i < inner; ++i)
                        {
                            d[i] = ::exp(s[i] - max[i]);
                            sum[i] += d[i];
                        }
                        s += inner;
                        d += inner;
                    }

                    d = dst;
                    for (size_t c = 0; c < count; ++c)
                    {
                        for (size_t i = 0; i < inner; ++i)
                            d[i] /= sum[i];
                        d += inner;
                    }
                    src += count * inner;
                    dst += count * inner;
                }
            }
        }
    }
}
