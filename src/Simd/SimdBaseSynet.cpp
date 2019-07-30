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
        void SynetAddBias(const float * bias, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if ((trans || size == 1) && count != 1)
            {
                size_t aligned = Simd::AlignLo(count, 4);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    for (; i < aligned; i += 4)
                    {
                        dst[i + 0] += bias[i + 0];
                        dst[i + 1] += bias[i + 1];
                        dst[i + 2] += bias[i + 2];
                        dst[i + 3] += bias[i + 3];
                    }
                    for (; i < count; ++i)
                        dst[i] += bias[i];
                    dst += count;
                }
            }
            else
            {
                size_t aligned = Simd::AlignLo(size, 4);
                for (size_t i = 0; i < count; ++i)
                {
                    float value = bias[i];
                    size_t j = 0;
                    for (; j < aligned; j += 4)
                    {
                        dst[j + 0] += value;
                        dst[j + 1] += value;
                        dst[j + 2] += value;
                        dst[j + 3] += value;
                    }
                    for (; j < size; ++j)
                        dst[j] += value;
                    dst += size;
                }
            }
        }

        template<size_t N> SIMD_INLINE void Copy(const float * src, float * dst)
        {
            for (size_t i = 0; i < N; ++i)
                dst[i] = src[i];
        }

        void SynetConvertImage_Chw_Hwc(size_t channels, size_t spatial, const float * src, float * dst)
        {
            for (size_t s = 0; s < spatial; ++s, src += 1, dst += channels)
                for (size_t c = 0; c < channels; ++c)
                    dst[c] = src[c*spatial];
        }

        template<size_t N> void SynetConvertImage_Chw_ChwXc(size_t channels, size_t spatial, const float * src, float * dst)
        {
            for (size_t c = 0; c < channels; c += N, src += N*spatial)
            {
                size_t n = Simd::Min(channels, c + N) - c;
                const float * ps = src;
                for (size_t s = 0; s < spatial; ++s, dst += N, ps += 1)
                {
                    size_t i = 0;
                    for (; i < n; ++i)
                        dst[i] = ps[i*spatial];
                    for (; i < N; ++i)
                        dst[i] = 0;
                }
            }
        }
        
        void SynetConvertImage_Hwc_Chw(size_t channels, size_t spatial, const float * src, float * dst)
        {
            SynetConvertImage_Chw_Hwc(spatial, channels, src, dst);
        }

        template<size_t N> void SynetConvertImage_Hwc_ChwXc(size_t channels, size_t spatial, const float * src, float * dst)
        {
            size_t channelsN = AlignLo(channels, N);
            size_t tail = channels - channelsN;
            for (size_t c = 0; c < channelsN; c += N, src += N)
            {
                const float * psrc = src;
                for (size_t s = 0; s < spatial; ++s, psrc += channels, dst += N)
                    Copy<N>(psrc, dst);
            }
            if(tail)
            {
                const float * psrc = src;
                for (size_t s = 0; s < spatial; ++s, psrc += channels, dst += N)
                {
                    size_t i = 0;
                    for (; i < tail; ++i)
                        dst[i] = psrc[i];
                    for (; i < N; ++i)
                        dst[i] = 0;
                }
            }
        }

        template<size_t N> void SynetConvertImage_ChwXc_Chw(size_t channels, size_t spatial, const float * src, float * dst)
        {
            for (size_t c = 0; c < channels; c += N, src += N * spatial)
            {
                const float * ps = src;
                for (size_t i = 0, n = Simd::Min(channels, c + N) - c; i < n; ++i, ps += 1, dst += spatial)
                {
                    for (size_t s = 0; s < spatial; ++s)
                        dst[s] = ps[s*N];
                }
            }
        }

        template<size_t N> void SynetConvertImage_ChwXc_Hwc(size_t channels, size_t spatial, const float * src, float * dst)
        {
            size_t stride = N * spatial;
            size_t channelsN = AlignLo(channels, N);
            size_t tail = channels - channelsN;
            for (size_t s = 0; s < spatial; ++s, src += N)
            {
                const float * psrc = src;
                for (size_t c = 0; c < channelsN; c += N, psrc += stride, dst += N)
                    Copy<N>(psrc, dst);
                if (tail)
                {
                    for (size_t i = 0; i < tail; ++i)
                        *(dst++) = psrc[i];
                }
            }
        }

        typedef void(*SynetImageConverterPtr)(size_t channels, size_t spatial, const float * src, float * dst);
        SynetImageConverterPtr GetImageConverter(SimdTensorFormatType src, SimdTensorFormatType dst)
        {
            if (src == SimdTensorFormatNchw)
            {
                if(dst == SimdTensorFormatNhwc)
                    return SynetConvertImage_Chw_Hwc;
                if (dst == SimdTensorFormatNchw4c)
                    return SynetConvertImage_Chw_ChwXc<4>;
                if (dst == SimdTensorFormatNchw8c)
                    return SynetConvertImage_Chw_ChwXc<8>;
                if (dst == SimdTensorFormatNchw16c)
                    return SynetConvertImage_Chw_ChwXc<16>;
            }
            if (src == SimdTensorFormatNhwc)
            {
                if(dst == SimdTensorFormatNchw)
                    return SynetConvertImage_Hwc_Chw;
                if (dst == SimdTensorFormatNchw4c)
                    return SynetConvertImage_Hwc_ChwXc<4>;
                if (dst == SimdTensorFormatNchw8c)
                    return SynetConvertImage_Hwc_ChwXc<8>;
                if (dst == SimdTensorFormatNchw16c)
                    return SynetConvertImage_Hwc_ChwXc<16>;
            }
            if (src == SimdTensorFormatNchw4c)
            {
                if (dst == SimdTensorFormatNchw)
                    return SynetConvertImage_ChwXc_Chw<4>;
                if (dst == SimdTensorFormatNhwc)
                    return SynetConvertImage_ChwXc_Hwc<4>;
            }
            if (src == SimdTensorFormatNchw8c)
            {
                if (dst == SimdTensorFormatNchw)
                    return SynetConvertImage_ChwXc_Chw<8>;
                if (dst == SimdTensorFormatNhwc)
                    return SynetConvertImage_ChwXc_Hwc<8>;
            }
            if (src == SimdTensorFormatNchw16c)
            {
                if (dst == SimdTensorFormatNchw)
                    return SynetConvertImage_ChwXc_Chw<16>;
                if (dst == SimdTensorFormatNhwc)
                    return SynetConvertImage_ChwXc_Hwc<16>;
            }
            return NULL;
        }

        void SynetConvertImage(size_t batch, size_t channels, size_t spatial, const float * src, SimdTensorFormatType srcFormat, float * dst, SimdTensorFormatType dstFormat)
        {
            SynetImageConverterPtr imageConverter = GetImageConverter(srcFormat, dstFormat);
            size_t srcStride = AlignHi(channels, SynetTensorAlignment(srcFormat))*spatial;
            size_t dstStride = AlignHi(channels, SynetTensorAlignment(dstFormat))*spatial;
            for (size_t n = 0; n < batch; ++n)
            {
                if (srcFormat == dstFormat)
                    memcpy(dst, src, srcStride*sizeof(float));
                else
                {
                    assert(imageConverter);
                    imageConverter(channels, spatial, src, dst);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        void SynetConvertFilter_Oiyx_Yxio(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            size_t stride = input * kernel;
            for (size_t k = 0; k < kernel; ++k, src += 1)
            {
                const float * ps = src;
                for (size_t i = 0; i < input; ++i, ps += kernel)
                {
                    for (size_t o = 0; o < output; ++o)
                        *(dst++) = ps[o * stride];
                }
            }
        }

        template<size_t N> void SynetConvertFilter_Oiyx_OyxiXo(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            for (size_t o = 0; o < output; o += N)
            {
                size_t n = Simd::Min(output, o + N) - o;
                for (size_t k = 0; k < kernel; ++k)
                {
                    for (size_t i = 0; i < input; ++i)
                    {
                        size_t j = 0;
                        for (; j < n; ++j)
                            *(dst++) = src[((o + j) * input + i)*kernel + k];
                        for (; j < N; ++j)
                            *(dst++) = 0;
                    }
                }
            }
        }

        void SynetConvertFilter_Yxio_Oiyx(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            SynetConvertFilter_Oiyx_Yxio(kernel, input, output, src, dst);
        }

        template<size_t N> void SynetConvertFilter_Yxio_OyxiXo(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            size_t outputN = AlignLo(output, N);
            for (size_t o = 0; o < outputN; o += N, src += N)
            {
                const float * psrc = src;
                for (size_t k = 0; k < kernel; ++k)
                    for (size_t i = 0; i < input; ++i, dst += N, psrc += output)
                        Copy<N>(psrc, dst);
            }
            if(outputN < output)
            {
                size_t tail = output - outputN;
                for (size_t k = 0; k < kernel; ++k)
                {
                    for (size_t i = 0; i < input; ++i, src += output)
                    {
                        size_t j = 0;
                        for (; j < tail; ++j)
                            *(dst++) = src[j];
                        for (; j < N; ++j)
                            *(dst++) = 0;
                    }
                }
            }
        }

        template<size_t N> void SynetConvertFilter_OyxiXo_Oiyx(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            for (size_t o = 0; o < output; o += N, src += N*kernel*input)
            {
                for (size_t j = 0, n = Simd::Min(output, o + N) - o; j < n; ++j)
                {
                    for (size_t i = 0; i < input; ++i)
                    {                
                        for (size_t k = 0; k < kernel; ++k)
                            *(dst++) = src[ (k*input + i)*N + j];
                    }
                }
            }
        }

        template<size_t N> void SynetConvertFilter_OyxiXo_Yxio(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            size_t outputN = AlignLo(output, N);
            size_t tail = output - outputN;
            size_t stride = kernel * input * N;
            for (size_t k = 0; k < kernel; ++k)
            {
                for (size_t i = 0; i < input; ++i, src += N)
                {
                    const float * psrc = src;
                    for (size_t o = 0; o < outputN; o += N, psrc += stride, dst += N)
                        Copy<N>(psrc, dst);
                    if(outputN < output)
                    {
                        for (size_t j = 0; j < tail; ++j)
                            *(dst++) = psrc[j];
                    }
                }
            }
        }

        typedef void(*SynetFilterConverterPtr)(size_t output, size_t input, size_t kernel, const float * src, float * dst);
        SynetFilterConverterPtr GetFilterConverter(SimdTensorFormatType src, SimdTensorFormatType dst)
        {
            if (src == SimdTensorFormatOiyx)
            {
                if (dst == SimdTensorFormatYxio)
                    return SynetConvertFilter_Oiyx_Yxio;
                if (dst == SimdTensorFormatOyxi4o)
                    return SynetConvertFilter_Oiyx_OyxiXo<4>;
                if (dst == SimdTensorFormatOyxi8o)
                    return SynetConvertFilter_Oiyx_OyxiXo<8>;
                if (dst == SimdTensorFormatOyxi16o)
                    return SynetConvertFilter_Oiyx_OyxiXo<16>;
            }
            if (src == SimdTensorFormatYxio)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetConvertFilter_Yxio_Oiyx;
                if (dst == SimdTensorFormatOyxi4o)
                    return SynetConvertFilter_Yxio_OyxiXo<4>;
                if (dst == SimdTensorFormatOyxi8o)
                    return SynetConvertFilter_Yxio_OyxiXo<8>;
                if (dst == SimdTensorFormatOyxi16o)
                    return SynetConvertFilter_Yxio_OyxiXo<16>;
            }
            if (src == SimdTensorFormatOyxi4o)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetConvertFilter_OyxiXo_Oiyx<4>;
                if (dst == SimdTensorFormatYxio)
                    return SynetConvertFilter_OyxiXo_Yxio<4>;
            }
            if (src == SimdTensorFormatOyxi8o)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetConvertFilter_OyxiXo_Oiyx<8>;
                if (dst == SimdTensorFormatYxio)
                    return SynetConvertFilter_OyxiXo_Yxio<8>;
            }
            if (src == SimdTensorFormatOyxi16o)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetConvertFilter_OyxiXo_Oiyx<16>;
                if (dst == SimdTensorFormatYxio)
                    return SynetConvertFilter_OyxiXo_Yxio<16>;
            }
            return NULL;
        }

        void SynetConvertFilter(size_t output, size_t input, size_t kernel, const float * src, SimdTensorFormatType srcFormat, float * dst, SimdTensorFormatType dstFormat)
        {
            if (srcFormat == dstFormat)
            {
                size_t aligned = AlignHi(output, SynetTensorAlignment(srcFormat));
                memcpy(dst, src, aligned * input * kernel * sizeof(float));
                return;
            }
            SynetFilterConverterPtr filterConverter = GetFilterConverter(srcFormat, dstFormat);
            assert(filterConverter);
            filterConverter(output, input, kernel, src, dst);

        }

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

        void SynetFusedLayerForward0(const float * src, const float * bias, const float * scale, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if ((trans || size == 1) && count != 1)
            {
                size_t aligned = Simd::AlignLo(count, 4);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    for (; i < aligned; i += 4)
                    {
                        dst[i + 0] = SynetFusedLayerForward0(src[i + 0] + bias[i + 0], scale[i + 0]);
                        dst[i + 1] = SynetFusedLayerForward0(src[i + 1] + bias[i + 1], scale[i + 1]);
                        dst[i + 2] = SynetFusedLayerForward0(src[i + 2] + bias[i + 2], scale[i + 2]);
                        dst[i + 3] = SynetFusedLayerForward0(src[i + 3] + bias[i + 3], scale[i + 3]);
                    }
                    for (; i < count; ++i)
                        dst[i] = SynetFusedLayerForward0(src[i] + bias[i], scale[i]);
                    src += count;
                    dst += count;
                }
            }
            else
            {
                size_t aligned = Simd::AlignLo(size, 4);
                for (size_t i = 0; i < count; ++i)
                {
                    float b = bias[i];
                    float s = scale[i];
                    size_t j = 0;
                    for (; j < aligned; j += 4)
                    {
                        dst[j + 0] = SynetFusedLayerForward0(src[j + 0] + b, s);
                        dst[j + 1] = SynetFusedLayerForward0(src[j + 1] + b, s);
                        dst[j + 2] = SynetFusedLayerForward0(src[j + 2] + b, s);
                        dst[j + 3] = SynetFusedLayerForward0(src[j + 3] + b, s);
                    }
                    for (; j < size; ++j)
                        dst[j] = SynetFusedLayerForward0(src[j] + b, s);
                    src += size;
                    dst += size;
                } 
            }
        }

        void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if ((trans || size == 1) && count != 1)
            {
                size_t aligned = Simd::AlignLo(count, 4);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    for (; i < aligned; i += 4)
                    {
                        dst[i + 0] = SynetFusedLayerForward1(src[i + 0] + bias0[i + 0], scale1[i + 0], bias1[i + 0]);
                        dst[i + 1] = SynetFusedLayerForward1(src[i + 1] + bias0[i + 1], scale1[i + 1], bias1[i + 1]);
                        dst[i + 2] = SynetFusedLayerForward1(src[i + 2] + bias0[i + 2], scale1[i + 2], bias1[i + 2]);
                        dst[i + 3] = SynetFusedLayerForward1(src[i + 3] + bias0[i + 3], scale1[i + 3], bias1[i + 3]);
                    }
                    for (; i < count; ++i)
                        dst[i] = SynetFusedLayerForward1(src[i] + bias0[i], scale1[i], bias1[i]);
                    src += count;
                    dst += count;
                }
            }
            else
            {
                size_t aligned = Simd::AlignLo(size, 4);
                for (size_t i = 0; i < count; ++i)
                {
                    float b0 = bias0[i];
                    float s1 = scale1[i];
                    float b1 = bias1[i];
                    size_t j = 0;
                    for (; j < aligned; j += 4)
                    {
                        dst[j + 0] = SynetFusedLayerForward1(src[j + 0] + b0, s1, b1);
                        dst[j + 1] = SynetFusedLayerForward1(src[j + 1] + b0, s1, b1);
                        dst[j + 2] = SynetFusedLayerForward1(src[j + 2] + b0, s1, b1);
                        dst[j + 3] = SynetFusedLayerForward1(src[j + 3] + b0, s1, b1);
                    }
                    for (; j < size; ++j)
                        dst[j] = SynetFusedLayerForward1(src[j] + b0, s1, b1);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetFusedLayerForward2(const float * src, const float * scale, const float * bias, size_t count, size_t size, const float * slope, float * dst, SimdBool trans)
        {
            float _slope = slope[0];
            if ((trans || size == 1) && count != 1)
            {
                size_t aligned = Simd::AlignLo(count, 4);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    for (; i < aligned; i += 4)
                    {
                        dst[i + 0] = SynetFusedLayerForward2(src[i + 0], scale[i + 0], bias[i + 0], _slope);
                        dst[i + 1] = SynetFusedLayerForward2(src[i + 1], scale[i + 1], bias[i + 1], _slope);
                        dst[i + 2] = SynetFusedLayerForward2(src[i + 2], scale[i + 2], bias[i + 2], _slope);
                        dst[i + 3] = SynetFusedLayerForward2(src[i + 3], scale[i + 3], bias[i + 3], _slope);
                    }
                    for (; i < count; ++i)
                        dst[i] = SynetFusedLayerForward2(src[i], scale[i], bias[i], _slope);
                    src += count;
                    dst += count;
                }
            }
            else
            {
                size_t aligned = Simd::AlignLo(size, 4);
                for (size_t i = 0; i < count; ++i)
                {
                    float _scale = scale[i];
                    float _bias = bias[i];
                    size_t j = 0;
                    for (; j < aligned; j += 4)
                    {
                        dst[j + 0] = SynetFusedLayerForward2(src[j + 0], _scale, _bias, _slope);
                        dst[j + 1] = SynetFusedLayerForward2(src[j + 1], _scale, _bias, _slope);
                        dst[j + 2] = SynetFusedLayerForward2(src[j + 2], _scale, _bias, _slope);
                        dst[j + 3] = SynetFusedLayerForward2(src[j + 3], _scale, _bias, _slope);
                    }
                    for (; j < size; ++j)
                        dst[j] = SynetFusedLayerForward2(src[j], _scale, _bias, _slope);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetFusedLayerForward3(const float * src, const float * bias, const float * scale, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if ((trans || size == 1) && count != 1)
            {
                size_t aligned = Simd::AlignLo(count, 4);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    for (; i < aligned; i += 4)
                    {
                        dst[i + 0] = SynetFusedLayerForward3(src[i + 0] + bias[i + 0], scale[i + 0]);
                        dst[i + 1] = SynetFusedLayerForward3(src[i + 1] + bias[i + 1], scale[i + 1]);
                        dst[i + 2] = SynetFusedLayerForward3(src[i + 2] + bias[i + 2], scale[i + 2]);
                        dst[i + 3] = SynetFusedLayerForward3(src[i + 3] + bias[i + 3], scale[i + 3]);
                    }
                    for (; i < count; ++i)
                        dst[i] = SynetFusedLayerForward3(src[i] + bias[i], scale[i]);
                    src += count;
                    dst += count;
                }
            }
            else
            {
                size_t aligned = Simd::AlignLo(size, 4);
                for (size_t i = 0; i < count; ++i)
                {
                    float b = bias[i];
                    float s = scale[i];
                    size_t j = 0;
                    for (; j < aligned; j += 4)
                    {
                        dst[j + 0] = SynetFusedLayerForward3(src[j + 0] + b, s);
                        dst[j + 1] = SynetFusedLayerForward3(src[j + 1] + b, s);
                        dst[j + 2] = SynetFusedLayerForward3(src[j + 2] + b, s);
                        dst[j + 3] = SynetFusedLayerForward3(src[j + 3] + b, s);
                    }
                    for (; j < size; ++j)
                        dst[j] = SynetFusedLayerForward3(src[j] + b, s);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetFusedLayerForward4(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst, SimdBool trans)
        {
            float s1 = scale1[0], b1 = bias1[0];
            if ((trans || size == 1) && count != 1)
            {
                size_t aligned = Simd::AlignLo(count, 4);
                float * dst0 = dst, * dst1 = dst + count;
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    for (; i < aligned; i += 4)
                    {
                        SynetFusedLayerForward4(src[i + 0], bias0[i + 0], s1, b1, dst0 + i + 0, dst1 + i + 0);
                        SynetFusedLayerForward4(src[i + 1], bias0[i + 1], s1, b1, dst0 + i + 1, dst1 + i + 1);
                        SynetFusedLayerForward4(src[i + 2], bias0[i + 2], s1, b1, dst0 + i + 2, dst1 + i + 2);
                        SynetFusedLayerForward4(src[i + 3], bias0[i + 3], s1, b1, dst0 + i + 3, dst1 + i + 3);
                    }
                    for (; i < count; ++i)
                        SynetFusedLayerForward4(src[i], bias0[i], s1, b1, dst0 + i, dst1 + i);
                    src += count;
                    dst0 += 2*count;
                    dst1 += 2 * count;
                }
            }
            else
            {
                size_t aligned = Simd::AlignLo(size, 4);
                float * dst0 = dst, * dst1 = dst + count * size;
                for (size_t i = 0; i < count; ++i)
                {
                    float b0 = bias0[i];
                    size_t j = 0;
                    for (; j < aligned; j += 4)
                    {
                        SynetFusedLayerForward4(src[j + 0], b0, s1, b1, dst0 + j + 0, dst1 + j + 0);
                        SynetFusedLayerForward4(src[j + 1], b0, s1, b1, dst0 + j + 1, dst1 + j + 1);
                        SynetFusedLayerForward4(src[j + 2], b0, s1, b1, dst0 + j + 2, dst1 + j + 2);
                        SynetFusedLayerForward4(src[j + 3], b0, s1, b1, dst0 + j + 3, dst1 + j + 3);
                    }
                    for (; j < size; ++j)
                        SynetFusedLayerForward4(src[j], b0, s1, b1, dst0 + j, dst1 + j);
                    src += size;
                    dst0 += size;
                    dst1 += size;
                }
            }
        }

        void SynetFusedLayerForward8(const float * src0, const float * src1, const float * src2, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if ((trans || size == 1) && count != 1)
            {
                size_t aligned = Simd::AlignLo(count, 4);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    for (; i < aligned; i += 4)
                    {
                        dst[i + 0] = SynetFusedLayerForward8(src0[i + 0], src1[i + 0], src2[i + 0]);
                        dst[i + 1] = SynetFusedLayerForward8(src0[i + 1], src1[i + 1], src2[i + 1]);
                        dst[i + 2] = SynetFusedLayerForward8(src0[i + 2], src1[i + 2], src2[i + 2]);
                        dst[i + 3] = SynetFusedLayerForward8(src0[i + 3], src1[i + 3], src2[i + 3]);
                    }
                    for (; i < count; ++i)
                        dst[i] = SynetFusedLayerForward8(src0[i], src1[i], src2[i]);
                    src0 += count;
                    src1 += count;
                    dst += count;
                }
            }
            else
            {
                size_t aligned = Simd::AlignLo(size, 4);
                for (size_t i = 0; i < count; ++i)
                {
                    float s2 = src2[i];
                    size_t j = 0;
                    for (; j < aligned; j += 4)
                    {
                        dst[j + 0] = SynetFusedLayerForward8(src0[j + 0], src1[j + 0], s2);
                        dst[j + 1] = SynetFusedLayerForward8(src0[j + 1], src1[j + 1], s2);
                        dst[j + 2] = SynetFusedLayerForward8(src0[j + 2], src1[j + 2], s2);
                        dst[j + 3] = SynetFusedLayerForward8(src0[j + 3], src1[j + 3], s2);
                    }
                    for (; j < size; ++j)
                        dst[j] = SynetFusedLayerForward8(src0[j], src1[j], s2);
                    src0 += size;
                    src1 += size;
                    dst += size;
                }
            }
        }

        void SynetFusedLayerForward9(const float * src0, const float * src1, const float * scale0, const float * bias0, size_t count0, size_t count1, size_t size, float * dst0, float * dst1, SimdBool trans)
        {
            const float * scale1 = scale0 + count0;
            const float * bias1 = bias0 + count0;
            if (trans || size == 1)
            {
                if (dst1)
                {
                    for (size_t j = 0; j < size; ++j)
                    {
                        for (size_t i = 0; i < count0; ++i)
                            dst0[i] = SynetFusedLayerForward9(src0[i], scale0[i], bias0[i]), dst1[i] = src0[i];
                        src0 += count0, dst0 += count0, dst1 += count0;
                        for (size_t i = 0; i < count1; ++i)
                            dst0[i] = SynetFusedLayerForward9(src1[i], scale1[i], bias1[i]), dst1[i] = src1[i];
                        src1 += count1, dst0 += count1, dst1 += count1;
                    }
                }
                else
                {
                    for (size_t j = 0; j < size; ++j)
                    {
                        for (size_t i = 0; i < count0; ++i)
                            dst0[i] = SynetFusedLayerForward9(src0[i], scale0[i], bias0[i]);
                        src0 += count0, dst0 += count0;
                        for (size_t i = 0; i < count1; ++i)
                            dst0[i] = SynetFusedLayerForward9(src1[i], scale1[i], bias1[i]);
                        src1 += count1, dst0 += count1;
                    }
                }
            }
            else
            {
                if (dst1)
                {
                    for (size_t i = 0; i < count0; ++i, src0 += size, dst0 += size, dst1 += size)
                        for (size_t j = 0; j < size; ++j)
                            dst0[j] = SynetFusedLayerForward9(src0[j], scale0[i], bias0[i]), dst1[j] = src0[j];
                    for (size_t i = 0; i < count1; ++i, src1 += size, dst0 += size, dst1 += size)
                        for (size_t j = 0; j < size; ++j)
                            dst0[j] = SynetFusedLayerForward9(src1[j], scale1[i], bias1[i]), dst1[j] = src1[j];
                }
                else
                {
                    for (size_t i = 0; i < count0; ++i, src0 += size, dst0 += size)
                        for (size_t j = 0; j < size; ++j)
                            dst0[j] = SynetFusedLayerForward9(src0[j], scale0[i], bias0[i]);
                    for (size_t i = 0; i < count1; ++i, src1 += size, dst0 += size)
                        for (size_t j = 0; j < size; ++j)
                            dst0[j] = SynetFusedLayerForward9(src1[j], scale1[i], bias1[i]);
                }
            }
        }

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

        void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if ((trans || size == 1) && count != 1)
            {
                size_t aligned = Simd::AlignLo(count, 4);
                if (bias)
                {
                    for (size_t j = 0; j < size; ++j)
                    {
                        size_t i = 0;
                        for (; i < aligned; i += 4)
                        {
                            dst[i + 0] = src[i + 0] * scale[i + 0] + bias[i + 0];
                            dst[i + 1] = src[i + 1] * scale[i + 1] + bias[i + 1];
                            dst[i + 2] = src[i + 2] * scale[i + 2] + bias[i + 2];
                            dst[i + 3] = src[i + 3] * scale[i + 3] + bias[i + 3];
                        }
                        for (; i < count; ++i)
                            dst[i] = src[i] * scale[i] + bias[i];
                        src += count;
                        dst += count;

                    }
                }
                else
                {
                    for (size_t j = 0; j < size; ++j)
                    {
                        size_t i = 0;
                        for (; i < aligned; i += 4)
                        {
                            dst[i + 0] = src[i + 0] * scale[i + 0];
                            dst[i + 1] = src[i + 1] * scale[i + 1];
                            dst[i + 2] = src[i + 2] * scale[i + 2];
                            dst[i + 3] = src[i + 3] * scale[i + 3];
                        }
                        for (; i < count; ++i)
                            dst[i] = src[i] * scale[i];
                        src += count;
                        dst += count;
                    }
                }
            }
            else
            {
                size_t aligned = Simd::AlignLo(size, 4);
                if (bias)
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        float s = scale[i];
                        float b = bias[i];
                        size_t j = 0;
                        for (; j < aligned; j += 4)
                        {
                            dst[j + 0] = src[j + 0] * s + b;
                            dst[j + 1] = src[j + 1] * s + b;
                            dst[j + 2] = src[j + 2] * s + b;
                            dst[j + 3] = src[j + 3] * s + b;
                        }
                        for (; j < size; ++j)
                            dst[j] = src[j] * s + b;
                        src += size;
                        dst += size;
                    }
                }
                else
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        float s = scale[i];
                        size_t j = 0;
                        for (; j < aligned; j += 4)
                        {
                            dst[j + 0] = src[j + 0] * s;
                            dst[j + 1] = src[j + 1] * s;
                            dst[j + 2] = src[j + 2] * s;
                            dst[j + 3] = src[j + 3] * s;
                        }
                        for (; j < size; ++j)
                            dst[j] = src[j] * s;
                        src += size;
                        dst += size;
                    }
                }
            }
        }

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
