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
#include "Simd/SimdMemory.h"
#include "Simd/SimdTranspose.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"

namespace Simd
{
#if defined(SIMD_AVX_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx
    {
        template<bool align> void SynetReorderImage_Chw_Hwc(size_t channels, size_t spatial, const float * src, float * dst)
        {
            size_t spatial4 = AlignLo(spatial, 4);
            size_t channels8 = AlignLo(channels, 8);
            size_t spatial8 = AlignLo(spatial, 8);
            size_t s = 0;
            for (; s < spatial8; s += 8, src += 8, dst += 8 * channels)
            {
                size_t c = 0;
                const float * ps = src;
                float * pd = dst;
                for (; c < channels8; c += 8, ps += 8 * spatial, pd += 8)
                    Transpose8x8<align>(ps, spatial, pd, channels);
                for (; c < channels; ++c, ps += spatial, pd += 1)
                {
                    pd[0 * channels] = ps[0];
                    pd[1 * channels] = ps[1];
                    pd[2 * channels] = ps[2];
                    pd[3 * channels] = ps[3];
                    pd[4 * channels] = ps[4];
                    pd[5 * channels] = ps[5];
                    pd[6 * channels] = ps[6];
                    pd[7 * channels] = ps[7];
                }
            }
            for (; s < spatial4; s += 4, src += 4, dst += 4 * channels)
            {
                size_t c = 0;
                const float * ps = src;
                float * pd = dst;
                for (; c < channels8; c += 8, ps += 8 * spatial, pd += 8)
                    Transpose4x8<align>(ps, spatial, pd, channels);
                for (; c < channels; ++c, ps += spatial, pd += 1)
                {
                    pd[0 * channels] = ps[0];
                    pd[1 * channels] = ps[1];
                    pd[2 * channels] = ps[2];
                    pd[3 * channels] = ps[3];
                }
            }
            for (; s < spatial; ++s, src += 1, dst += channels)
                for (size_t c = 0; c < channels; ++c)
                    dst[c] = src[c*spatial];
        }

        template<bool align> void SynetReorderImage_Chw_Chw8c(size_t channels, size_t spatial, const float * src, float * dst)
        {
            size_t channels8 = AlignLo(channels, 8);
            size_t spatial8 = AlignLo(spatial, 8);
            size_t tail = channels - channels8;
            size_t c = 0;
            for (; c < channels8; c += 8, src += 8 * spatial)
            {
                size_t s = 0;
                const float * ps = src;
                for (; s < spatial8; s += 8, dst += 8 * F, ps += 8)
                    Transpose8x8<align>(ps, spatial, dst, 8);
                for (; s < spatial; ++s, dst += F, ps += 1)
                {
                    dst[0] = ps[0 * spatial];
                    dst[1] = ps[1 * spatial];
                    dst[2] = ps[2 * spatial];
                    dst[3] = ps[3 * spatial];
                    dst[4] = ps[4 * spatial];
                    dst[5] = ps[5 * spatial];
                    dst[6] = ps[6 * spatial];
                    dst[7] = ps[7 * spatial];
                }
            }
            if (tail)
            {
                const float * ps = src;
                for (size_t s = 0; s < spatial; ++s, dst += F, ps += 1)
                {
                    size_t i = 0;
                    for (; i < tail; ++i)
                        dst[i] = ps[i*spatial];
                    for (; i < F; ++i)
                        dst[i] = 0;
                }
            }
        }

        template<bool align> void SynetReorderImage_Hwc_Chw(size_t channels, size_t spatial, const float * src, float * dst)
        {
            SynetReorderImage_Chw_Hwc<align>(spatial, channels, src, dst);
        }

        template<bool align> void SynetReorderImage_Hwc_Chw8c(size_t channels, size_t spatial, const float * src, float * dst)
        {
            size_t channelsF = AlignLo(channels, F);
            size_t channelsF4 = AlignLo(channels, 4 * F);
            size_t tail = channels - channelsF;
            size_t spatial4 = AlignLo(spatial, 4);
            size_t stride = spatial * F;
            size_t c = 0;
            for (; c < channelsF4; c += 4 * F, src += 4 * F)
            {
                const float * ps = src;
                float * pd = dst;
                size_t i = 0;
                for (; i < spatial4; i += 4, pd += 4 * F, ps += 4 * channels)
                    Transpose4x4xF<align>(ps, channels, pd, stride);
                for (; i < spatial; ++i, pd += F, ps += channels)
                {
                    Copy<align>(ps + 0 * F, pd + 0 * stride);
                    Copy<align>(ps + 1 * F, pd + 1 * stride);
                    Copy<align>(ps + 2 * F, pd + 2 * stride);
                    Copy<align>(ps + 3 * F, pd + 3 * stride);
                }
                dst += 4 * stride;
            }
            for (; c < channelsF; c += F, src += F)
            {
                const float * ps = src;
                for (size_t s = 0; s < spatial; ++s, ps += channels, dst += F)
                    Copy<align>(ps, dst);
            }
            if (tail)
            {
                const float * psrc = src;
                for (size_t s = 0; s < spatial; ++s, psrc += channels, dst += F)
                {
                    size_t i = 0;
                    for (; i < tail; ++i)
                        dst[i] = psrc[i];
                    for (; i < F; ++i)
                        dst[i] = 0;
                }
            }
        }

        template<bool align> void SynetReorderImage_Chw8c_Chw(size_t channels, size_t spatial, const float * src, float * dst)
        {
            size_t channels8 = AlignLo(channels, 8);
            size_t spatial8 = AlignLo(spatial, 8);
            size_t tail = channels - channels8;
            size_t c = 0;
            for (; c < channels8; c += 8, dst += 8 * spatial, src += 8 * spatial)
            {
                const float * ps = src;
                size_t s = 0;
                for (; s < spatial8; s += 8, ps += 8 * F)
                    Transpose8x8<align>(ps, 8, dst + s, spatial);
                for (; s < spatial; ++s, ps += 8)
                {
                    dst[s + 0 * spatial] = ps[0];
                    dst[s + 1 * spatial] = ps[1];
                    dst[s + 2 * spatial] = ps[2];
                    dst[s + 3 * spatial] = ps[3];
                    dst[s + 4 * spatial] = ps[4];
                    dst[s + 5 * spatial] = ps[5];
                    dst[s + 6 * spatial] = ps[6];
                    dst[s + 7 * spatial] = ps[7];
                }
            }
            if (tail)
            {
                const float * ps = src;
                for (size_t i = 0; i < tail; ++i, ps += 1, dst += spatial)
                {
                    for (size_t s = 0; s < spatial; ++s)
                        dst[s] = ps[s*F];
                }
            }
        }

        template<bool align> void SynetReorderImage_Chw8c_Hwc(size_t channels, size_t spatial, const float * src, float * dst)
        {
            size_t stride = F * spatial;
            size_t channelsF = AlignLo(channels, F);
            size_t channelsF4 = AlignLo(channels, 4 * F);
            size_t tail = channels - channelsF;
            size_t spatial4 = AlignLo(spatial, 4);
            size_t s = 0;
            for (; s < spatial4; s += 4, src += 4 * F, dst += 4 * channels)
            {
                const float * ps = src;
                float * pd = dst;
                size_t c = 0;
                for (; c < channelsF4; c += 4 * F, ps += 4 * stride, pd += 4 * F)
                    Transpose4x4xF<align>(ps, stride, pd, channels);
                for (; c < channelsF; c += F, ps += stride, pd += F)
                {
                    Copy<align>(ps + 0 * F, pd + 0 * channels);
                    Copy<align>(ps + 1 * F, pd + 1 * channels);
                    Copy<align>(ps + 2 * F, pd + 2 * channels);
                    Copy<align>(ps + 3 * F, pd + 3 * channels);
                }
                if (tail)
                {
                    for (size_t i = 0; i < tail; ++i)
                    {
                        pd[i + 0 * channels] = ps[i + 0 * F];
                        pd[i + 1 * channels] = ps[i + 1 * F];
                        pd[i + 2 * channels] = ps[i + 2 * F];
                        pd[i + 3 * channels] = ps[i + 3 * F];
                    }
                }
            }
            for (; s < spatial; ++s, src += F)
            {
                const float * ps = src;
                for (size_t c = 0; c < channelsF; c += F, ps += stride, dst += F)
                    Copy<align>(ps, dst);
                if (tail)
                {
                    for (size_t i = 0; i < tail; ++i)
                        *(dst++) = ps[i];
                }
            }
        }

        typedef void(*SynetImageConverterPtr)(size_t channels, size_t spatial, const float * src, float * dst);
        SynetImageConverterPtr GetImageConverter(SimdTensorFormatType src, SimdTensorFormatType dst)
        {
            if (src == SimdTensorFormatNchw)
            {
                if (dst == SimdTensorFormatNhwc)
                    return SynetReorderImage_Chw_Hwc<false>;
                if (dst == SimdTensorFormatNchw8c)
                    return SynetReorderImage_Chw_Chw8c<false>;
            }
            if (src == SimdTensorFormatNhwc)
            {
                if (dst == SimdTensorFormatNchw)
                    return SynetReorderImage_Hwc_Chw<false>;
                if (dst == SimdTensorFormatNchw8c)
                    return SynetReorderImage_Hwc_Chw8c<false>;
            }
            if (src == SimdTensorFormatNchw8c)
            {
                if (dst == SimdTensorFormatNchw)
                    return SynetReorderImage_Chw8c_Chw<false>;
                if (dst == SimdTensorFormatNhwc)
                    return SynetReorderImage_Chw8c_Hwc<false>;
            }
            return NULL;
        }

        void SynetReorderImage(size_t batch, size_t channels, size_t spatial, const float * src, SimdTensorFormatType srcFormat, float * dst, SimdTensorFormatType dstFormat)
        {
            SynetImageConverterPtr imageConverter = GetImageConverter(srcFormat, dstFormat);
            if (imageConverter)
            {
                size_t srcStride = AlignHi(channels, Base::SynetTensorAlignment(srcFormat))*spatial;
                size_t dstStride = AlignHi(channels, Base::SynetTensorAlignment(dstFormat))*spatial;
                for (size_t n = 0; n < batch; ++n)
                {
                    imageConverter(channels, spatial, src, dst);
                    src += srcStride;
                    dst += dstStride;
                }
            }
            else
                return Sse2::SynetReorderImage(batch, channels, spatial, src, srcFormat, dst, dstFormat);
        }

        template<bool align> void SynetReorderFilter_Oiyx_Yxio(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            if (kernel == 1)
            {
                SynetReorderImage_Chw_Hwc<align>(output, input, src, dst);
                return;
            }
            size_t output8 = AlignLo(output, 8);
            size_t kernel8 = AlignLo(kernel, 8);
            size_t ik = input * kernel, oi = output * input;
            for (size_t i = 0; i < input; ++i, src += kernel, dst += output)
            {
                const float * ps = src;
                float * pd = dst;
                size_t k = 0;
                for (; k < kernel8; k += 8, ps += 8, pd += 8 * oi)
                {
                    size_t o = 0;
                    for (; o < output8; o += 8)
                        Transpose8x8<align>(ps + o * ik, ik, pd + o, oi);
                    for (; o < output; ++o)
                    {
                        pd[0 * oi + o] = ps[o * ik + 0];
                        pd[1 * oi + o] = ps[o * ik + 1];
                        pd[2 * oi + o] = ps[o * ik + 2];
                        pd[3 * oi + o] = ps[o * ik + 3];
                        pd[4 * oi + o] = ps[o * ik + 4];
                        pd[5 * oi + o] = ps[o * ik + 5];
                        pd[6 * oi + o] = ps[o * ik + 6];
                        pd[7 * oi + o] = ps[o * ik + 7];
                    }
                }
                for (; k < kernel; ++k, ps += 1, pd += oi)
                    for (size_t o = 0; o < output; ++o)
                        pd[o] = ps[o*ik];
            }
        }

        template<bool align> void SynetReorderFilter_Oiyx_Oyxi8o(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            if (kernel == 1)
            {
                SynetReorderImage_Chw_Chw8c<align>(output, input, src, dst);
                return;
            }
            size_t outputF = AlignLo(output, F);
            size_t kernelF = AlignLo(kernel, F);
            size_t tail = output - outputF;
            size_t ik = input * kernel;
            size_t stride = input * F;
            for (size_t o = 0; o < outputF; o += F)
            {
                for (size_t i = 0; i < input; ++i)
                {
                    const float * ps = src + o * ik + i * kernel;
                    float * pd = dst + o * ik + i * F;
                    size_t k = 0;
                    for (; k < kernelF; k += F, ps += F, pd += F * stride)
                        Transpose8x8<align>(ps, ik, pd, stride);
                    for (; k < kernel; ++k, ps += 1, pd += stride)
                        for (size_t j = 0; j < F; ++j)
                            pd[j] = ps[j*ik];
                }
            }
            if (tail)
            {
                for (size_t i = 0; i < input; ++i)
                {
                    const float * ps = src + outputF * ik + i * kernel;
                    float * pd = dst + outputF * ik + i * F;
                    for (size_t k = 0; k < kernel; ++k, ps += 1, pd += stride)
                    {
                        size_t j = 0;
                        for (; j < tail; ++j)
                            pd[j] = ps[j*ik];
                        for (; j < F; ++j)
                            pd[j] = 0;
                    }
                }
            }
        }

        template<bool align> void SynetReorderFilter_Yxio_Oiyx(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            if (kernel == 1)
            {
                SynetReorderImage_Chw_Hwc<align>(input, output, src, dst);
                return;
            }
            SynetReorderFilter_Oiyx_Yxio<align>(kernel, input, output, src, dst);
        }

        template<bool align> void SynetReorderFilter_Yxio_Oyxi8o(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            size_t outputF = AlignLo(output, F);
            size_t outputF4 = AlignLo(output, F * 4);
            size_t ki = kernel * input;
            size_t stride = ki * F;
            size_t ki4 = AlignLo(ki, 4);
            size_t o = 0;
            for (; o < outputF4; o += 4 * F, src += 4 * F)
            {
                const float * ps = src;
                float * pd = dst;
                size_t i = 0;
                for (; i < ki4; i += 4, pd += 4 * F, ps += 4 * output)
                    Transpose4x4xF<align>(ps, output, pd, stride);
                for (; i < ki; ++i, pd += F, ps += output)
                {
                    Copy<align>(ps + 0 * F, pd + 0 * stride);
                    Copy<align>(ps + 1 * F, pd + 1 * stride);
                    Copy<align>(ps + 2 * F, pd + 2 * stride);
                    Copy<align>(ps + 3 * F, pd + 3 * stride);
                }
                dst += 4 * stride;
            }
            for (; o < outputF; o += F, src += F)
            {
                const float * ps = src;
                float * pd = dst;
                size_t i = 0;
                for (; i < ki; ++i, pd += F, ps += output)
                    Copy<align>(ps, pd);
                dst += stride;
            }
            if (outputF < output)
            {
                size_t tail = output - outputF;
                for (size_t k = 0; k < kernel; ++k)
                {
                    for (size_t i = 0; i < input; ++i, src += output)
                    {
                        size_t j = 0;
                        for (; j < tail; ++j)
                            *(dst++) = src[j];
                        for (; j < F; ++j)
                            *(dst++) = 0;
                    }
                }
            }
        }

        template<bool align> void SynetReorderFilter_Oyxi8o_Oiyx(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            if (kernel == 1)
            {
                SynetReorderImage_Chw8c_Chw<align>(output, input, src, dst);
                return;
            }
            size_t outputF = AlignLo(output, F);
            size_t tail = output - outputF;
            size_t kernelF = AlignLo(kernel, F);
            size_t ik = input * kernel;
            size_t stride = F * input;
            size_t o = 0;
            for (; o < outputF; o += F, src += F * ik)
            {
                const float * ps = src;
                float * pd = dst;
                for (size_t i = 0; i < input; ++i, ps += F)
                {
                    size_t k = 0;
                    for (; k < kernelF; k += F, pd += F)
                        Transpose8x8<align>(ps + k * stride, stride, pd, ik);
                    for (; k < kernel; ++k, pd++)
                    {
                        pd[0 * ik] = ps[k*stride + 0];
                        pd[1 * ik] = ps[k*stride + 1];
                        pd[2 * ik] = ps[k*stride + 2];
                        pd[3 * ik] = ps[k*stride + 3];
                        pd[4 * ik] = ps[k*stride + 4];
                        pd[5 * ik] = ps[k*stride + 5];
                        pd[6 * ik] = ps[k*stride + 6];
                        pd[7 * ik] = ps[k*stride + 7];
                    }
                }
                dst += F * ik;
            }
            if (tail)
            {
                for (size_t j = 0; j < tail; ++j)
                {
                    const float * ps = src + j;
                    for (size_t i = 0; i < input; ++i, ps += F)
                        for (size_t k = 0; k < kernel; ++k)
                            *(dst++) = ps[k*stride];
                }
            }
        }

        template<bool align> void SynetReorderFilter_Oyxi8o_Yxio(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            size_t outputF = AlignLo(output, F);
            size_t outputF4 = AlignLo(output, 4 * F);
            size_t tail = output - outputF;
            size_t ki = kernel * input;
            size_t ki4 = AlignLo(ki, 4);
            size_t stride = ki * F;
            size_t i = 0;
            for (; i < ki4; i += 4, src += 4 * F)
            {
                const float * ps = src;
                float * pd = dst;
                size_t o = 0;
                for (; o < outputF4; o += 4 * F, ps += 4 * stride, pd += 4 * F)
                    Transpose4x4xF<align>(ps, stride, pd, output);
                for (; o < outputF; o += F, ps += stride, pd += F)
                {
                    Copy<align>(ps + 0 * F, pd + 0 * output);
                    Copy<align>(ps + 1 * F, pd + 1 * output);
                    Copy<align>(ps + 2 * F, pd + 2 * output);
                    Copy<align>(ps + 3 * F, pd + 3 * output);
                }
                if (tail)
                {
                    for (size_t j = 0; j < tail; ++j)
                    {
                        pd[j + 0 * output] = ps[j + 0 * F];
                        pd[j + 1 * output] = ps[j + 1 * F];
                        pd[j + 2 * output] = ps[j + 2 * F];
                        pd[j + 3 * output] = ps[j + 3 * F];
                    }
                }
                dst += 4 * output;
            }
            for (; i < ki; ++i, src += F)
            {
                const float * ps = src;
                for (size_t o = 0; o < outputF; o += F, ps += stride, dst += F)
                    Copy<align>(ps, dst);
                if (tail)
                {
                    for (size_t j = 0; j < tail; ++j)
                        *(dst++) = ps[j];
                }
            }
        }

        typedef void(*SynetFilterConverterPtr)(size_t output, size_t input, size_t kernel, const float * src, float * dst);
        SynetFilterConverterPtr GetFilterConverter(SimdTensorFormatType src, SimdTensorFormatType dst)
        {
            if (src == SimdTensorFormatOiyx)
            {
                if (dst == SimdTensorFormatYxio)
                    return SynetReorderFilter_Oiyx_Yxio<false>;
                if (dst == SimdTensorFormatOyxi8o)
                    return SynetReorderFilter_Oiyx_Oyxi8o<false>;
            }
            if (src == SimdTensorFormatYxio)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetReorderFilter_Yxio_Oiyx<false>;
                if (dst == SimdTensorFormatOyxi8o)
                    return SynetReorderFilter_Yxio_Oyxi8o<false>;
            }
            if (src == SimdTensorFormatOyxi8o)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetReorderFilter_Oyxi8o_Oiyx<false>;
                if (dst == SimdTensorFormatYxio)
                    return SynetReorderFilter_Oyxi8o_Yxio<false>;
            }
            return NULL;
        }

        void SynetReorderFilter(size_t output, size_t input, size_t kernel, const float * src, SimdTensorFormatType srcFormat, float * dst, SimdTensorFormatType dstFormat)
        {
            SynetFilterConverterPtr filterConverter = GetFilterConverter(srcFormat, dstFormat);
            if (filterConverter)
                filterConverter(output, input, kernel, src, dst);
            else
                Sse2::SynetReorderFilter(output, input, kernel, src, srcFormat, dst, dstFormat);
        }
    }
#endif// SIMD_AVX_ENABLE
}
