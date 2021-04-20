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
#include "Simd/SimdAvx1.h"

namespace Simd
{
#if defined(SIMD_AVX512F_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512f
    {
        template<bool align> void SynetReorderImage_Chw_Hwc(size_t channels, size_t spatial, const float * src, float * dst)
        {
            size_t channels8 = AlignLo(channels, 8);
            size_t spatial8 = AlignLo(spatial, 8);
            size_t channels16 = AlignLo(channels, 16);
            size_t spatial16 = AlignLo(spatial, 16);
            size_t s = 0;
            for (; s < spatial16; s += 16, src += 16, dst += 16 * channels)
            {
                size_t c = 0;
                const float * ps = src;
                float * pd = dst;
                for (; c < channels16; c += 16, ps += 16 * spatial, pd += 16)
                    Transpose16x16<align>(ps, spatial, pd, channels);
                for (; c < channels8; c += 8, ps += 8 * spatial, pd += 8)
                    Transpose16x8<align>(ps, spatial, pd, channels);
                for (; c < channels; ++c, ps += spatial, pd += 1)
                {
                    pd[0x0 * channels] = ps[0x0];
                    pd[0x1 * channels] = ps[0x1];
                    pd[0x2 * channels] = ps[0x2];
                    pd[0x3 * channels] = ps[0x3];
                    pd[0x4 * channels] = ps[0x4];
                    pd[0x5 * channels] = ps[0x5];
                    pd[0x6 * channels] = ps[0x6];
                    pd[0x7 * channels] = ps[0x7];
                    pd[0x8 * channels] = ps[0x8];
                    pd[0x9 * channels] = ps[0x9];
                    pd[0xA * channels] = ps[0xA];
                    pd[0xB * channels] = ps[0xB];
                    pd[0xC * channels] = ps[0xC];
                    pd[0xD * channels] = ps[0xD];
                    pd[0xE * channels] = ps[0xE];
                    pd[0xF * channels] = ps[0xF];
                }
            }
            for (; s < spatial8; s += 8, src += 8, dst += 8 * channels)
            {
                size_t c = 0;
                const float * ps = src;
                float * pd = dst;
                for (; c < channels16; c += 16, ps += 16 * spatial, pd += 16)
                    Transpose8x16<align>(ps, spatial, pd, channels);
                for (; c < channels8; c += 8, ps += 8 * spatial, pd += 8)
                    Avx::Transpose8x8<align>(ps, spatial, pd, channels);
                for (; c < channels; ++c, ps += spatial, pd += 1)
                {
                    pd[0x0 * channels] = ps[0x0];
                    pd[0x1 * channels] = ps[0x1];
                    pd[0x2 * channels] = ps[0x2];
                    pd[0x3 * channels] = ps[0x3];
                    pd[0x4 * channels] = ps[0x4];
                    pd[0x5 * channels] = ps[0x5];
                    pd[0x6 * channels] = ps[0x6];
                    pd[0x7 * channels] = ps[0x7];
                }
            }
            for (; s < spatial; ++s, src += 1, dst += channels)
                for (size_t c = 0; c < channels; ++c)
                    dst[c] = src[c*spatial];
        }

        template<bool align> void SynetReorderImage_Chw_Chw16c(size_t channels, size_t spatial, const float * src, float * dst)
        {
            size_t spatial8 = AlignLo(spatial, 8);
            size_t channels16 = AlignLo(channels, 16);
            size_t spatial16 = AlignLo(spatial, 16);
            size_t tail = channels - channels16;
            size_t c = 0;
            for (; c < channels16; c += 16, src += 16 * spatial)
            {
                size_t s = 0;
                const float * ps = src;
                for (; s < spatial16; s += 16, dst += 16 * F, ps += 16)
                    Transpose16x16<align>(ps, spatial, dst, 16);
                for (; s < spatial8; s += 8, dst += 8 * F, ps += 8)
                    Transpose8x16<align>(ps, spatial, dst, 16);
                for (; s < spatial; ++s, dst += F, ps += 1)
                {
                    dst[0x0] = ps[0x0 * spatial];
                    dst[0x1] = ps[0x1 * spatial];
                    dst[0x2] = ps[0x2 * spatial];
                    dst[0x3] = ps[0x3 * spatial];
                    dst[0x4] = ps[0x4 * spatial];
                    dst[0x5] = ps[0x5 * spatial];
                    dst[0x6] = ps[0x6 * spatial];
                    dst[0x7] = ps[0x7 * spatial];
                    dst[0x8] = ps[0x8 * spatial];
                    dst[0x9] = ps[0x9 * spatial];
                    dst[0xA] = ps[0xA * spatial];
                    dst[0xB] = ps[0xB * spatial];
                    dst[0xC] = ps[0xC * spatial];
                    dst[0xD] = ps[0xD * spatial];
                    dst[0xE] = ps[0xE * spatial];
                    dst[0xF] = ps[0xF * spatial];
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

        template<bool align> void SynetReorderImage_Hwc_Chw16c(size_t channels, size_t spatial, const float * src, float * dst)
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
                __mmask16 mask = TailMask16(tail);
                const float * ps = src;
                for (size_t s = 0; s < spatial; ++s, ps += channels, dst += F)
                    CopyZP<align>(ps, dst, mask);
            }
        }

        template<bool align> void SynetReorderImage_Chw16c_Chw(size_t channels, size_t spatial, const float * src, float * dst)
        {
            size_t spatial8 = AlignLo(spatial, 8);
            size_t channels16 = AlignLo(channels, 16);
            size_t spatial16 = AlignLo(spatial, 16);
            size_t tail = channels - channels16;
            size_t c = 0;
            for (; c < channels16; c += 16, dst += 16 * spatial, src += 16 * spatial)
            {
                const float * ps = src;
                size_t s = 0;
                for (; s < spatial16; s += 16, ps += 16 * F)
                    Transpose16x16<align>(ps, 16, dst + s, spatial);
                for (; s < spatial8; s += 8, ps += 8 * F)
                    Transpose16x8<align>(ps, 16, dst + s, spatial);
                for (; s < spatial; ++s, ps += 16)
                {
                    dst[s + 0x0 * spatial] = ps[0x0];
                    dst[s + 0x1 * spatial] = ps[0x1];
                    dst[s + 0x2 * spatial] = ps[0x2];
                    dst[s + 0x3 * spatial] = ps[0x3];
                    dst[s + 0x4 * spatial] = ps[0x4];
                    dst[s + 0x5 * spatial] = ps[0x5];
                    dst[s + 0x6 * spatial] = ps[0x6];
                    dst[s + 0x7 * spatial] = ps[0x7];
                    dst[s + 0x8 * spatial] = ps[0x8];
                    dst[s + 0x9 * spatial] = ps[0x9];
                    dst[s + 0xA * spatial] = ps[0xA];
                    dst[s + 0xB * spatial] = ps[0xB];
                    dst[s + 0xC * spatial] = ps[0xC];
                    dst[s + 0xD * spatial] = ps[0xD];
                    dst[s + 0xE * spatial] = ps[0xE];
                    dst[s + 0xF * spatial] = ps[0xF];
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

        template<bool align> void SynetReorderImage_Chw16c_Hwc(size_t channels, size_t spatial, const float * src, float * dst)
        {
            size_t stride = F * spatial;
            size_t channelsF = AlignLo(channels, F);
            size_t channelsF4 = AlignLo(channels, 4 * F);
            size_t tail = channels - channelsF;
            __mmask16 mask = TailMask16(tail);
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
                    Copy<align, true>(ps + 0 * F, pd + 0 * channels, mask);
                    Copy<align, true>(ps + 1 * F, pd + 1 * channels, mask);
                    Copy<align, true>(ps + 2 * F, pd + 2 * channels, mask);
                    Copy<align, true>(ps + 3 * F, pd + 3 * channels, mask);
                }
            }
            for (; s < spatial; ++s, src += F)
            {
                const float * ps = src;
                for (size_t c = 0; c < channelsF; c += F, ps += stride, dst += F)
                    Copy<align>(ps, dst);
                if (tail)
                    Copy<align, true>(ps, dst, mask), dst += tail;
            }
        }

        typedef void(*SynetImageConverterPtr)(size_t channels, size_t spatial, const float * src, float * dst);
        SynetImageConverterPtr GetImageConverter(SimdTensorFormatType src, SimdTensorFormatType dst)
        {
            if (src == SimdTensorFormatNchw)
            {
                if (dst == SimdTensorFormatNhwc)
                    return SynetReorderImage_Chw_Hwc<false>;
                if (dst == SimdTensorFormatNchw16c)
                    return SynetReorderImage_Chw_Chw16c<false>;
            }
            if (src == SimdTensorFormatNhwc)
            {
                if (dst == SimdTensorFormatNchw)
                    return SynetReorderImage_Hwc_Chw<false>;
                if (dst == SimdTensorFormatNchw16c)
                    return SynetReorderImage_Hwc_Chw16c<false>;
            }
            if (src == SimdTensorFormatNchw16c)
            {
                if (dst == SimdTensorFormatNchw)
                    return SynetReorderImage_Chw16c_Chw<false>;
                if (dst == SimdTensorFormatNhwc)
                    return SynetReorderImage_Chw16c_Hwc<false>;
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
                return Avx::SynetReorderImage(batch, channels, spatial, src, srcFormat, dst, dstFormat);
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
            size_t output16 = AlignLo(output, 16);
            size_t kernel16 = AlignLo(kernel, 16);
            size_t ik = input * kernel, oi = output * input;
            for (size_t i = 0; i < input; ++i, src += kernel, dst += output)
            {
                const float * ps = src;
                float * pd = dst;
                size_t k = 0;
                for (; k < kernel16; k += 16, ps += 16, pd += 16 * oi)
                {
                    size_t o = 0;
                    for (; o < output16; o += 16)
                        Transpose16x16<align>(ps + o * ik, ik, pd + o, oi);
                    for (; o < output8; o += 8)
                        Transpose16x8<align>(ps + o * ik, ik, pd + o, oi);
                    for (; o < output; ++o)
                    {
                        pd[0x0 * oi + o] = ps[o * ik + 0x0];
                        pd[0x1 * oi + o] = ps[o * ik + 0x1];
                        pd[0x2 * oi + o] = ps[o * ik + 0x2];
                        pd[0x3 * oi + o] = ps[o * ik + 0x3];
                        pd[0x4 * oi + o] = ps[o * ik + 0x4];
                        pd[0x5 * oi + o] = ps[o * ik + 0x5];
                        pd[0x6 * oi + o] = ps[o * ik + 0x6];
                        pd[0x7 * oi + o] = ps[o * ik + 0x7];
                        pd[0x8 * oi + o] = ps[o * ik + 0x8];
                        pd[0x9 * oi + o] = ps[o * ik + 0x9];
                        pd[0xA * oi + o] = ps[o * ik + 0xA];
                        pd[0xB * oi + o] = ps[o * ik + 0xB];
                        pd[0xC * oi + o] = ps[o * ik + 0xC];
                        pd[0xD * oi + o] = ps[o * ik + 0xD];
                        pd[0xE * oi + o] = ps[o * ik + 0xE];
                        pd[0xF * oi + o] = ps[o * ik + 0xF];
                    }
                }
                for (; k < kernel8; k += 8, ps += 8, pd += 8 * oi)
                {
                    size_t o = 0;
                    for (; o < output16; o += 16)
                        Transpose8x16<align>(ps + o * ik, ik, pd + o, oi);
                    for (; o < output8; o += 8)
                        Avx::Transpose8x8<align>(ps + o * ik, ik, pd + o, oi);
                    for (; o < output; ++o)
                    {
                        pd[0x0 * oi + o] = ps[o * ik + 0x0];
                        pd[0x1 * oi + o] = ps[o * ik + 0x1];
                        pd[0x2 * oi + o] = ps[o * ik + 0x2];
                        pd[0x3 * oi + o] = ps[o * ik + 0x3];
                        pd[0x4 * oi + o] = ps[o * ik + 0x4];
                        pd[0x5 * oi + o] = ps[o * ik + 0x5];
                        pd[0x6 * oi + o] = ps[o * ik + 0x6];
                        pd[0x7 * oi + o] = ps[o * ik + 0x7];
                    }
                }
                for (; k < kernel; ++k, ps += 1, pd += oi)
                    for (size_t o = 0; o < output; ++o)
                        pd[o] = ps[o*ik];
            }
        }

        template<bool align> void SynetReorderFilter_Oiyx_Oyxi16o(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            if (kernel == 1)
            {
                SynetReorderImage_Chw_Chw16c<align>(output, input, src, dst);
                return;
            }
            size_t output16 = AlignLo(output, 16);
            size_t kernel8 = AlignLo(kernel, 8);
            size_t tail = output - output16;
            size_t ik = input * kernel;
            size_t stride = input * 16;
            for (size_t o = 0; o < output16; o += F)
            {
                for (size_t i = 0; i < input; ++i)
                {
                    const float * ps = src + o * ik + i * kernel;
                    float * pd = dst + o * ik + i * 16;
                    size_t k = 0;
                    for (; k < kernel8; k += 8, ps += 8, pd += 8 * stride)
                        Transpose8x16<align>(ps, ik, pd, stride);
                    for (; k < kernel; ++k, ps += 1, pd += stride)
                        for (size_t j = 0; j < 16; ++j)
                            pd[j] = ps[j*ik];
                }
            }
            if (tail)
            {

                __mmask16 mask = TailMask16(tail);
                for (size_t i = 0; i < input; ++i)
                {
                    const float * ps = src + output16 * ik + i * kernel;
                    float * pd = dst + output16 * ik + i * 16;
                    for (size_t k = 0; k < kernel; ++k, ps += 1, pd += stride)
                    {
                        size_t j = 0;
                        for (; j < tail; ++j)
                            pd[j] = ps[j*ik];
                        for (; j < 16; ++j)
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

        template<bool align> void SynetReorderFilter_Yxio_Oyxi16o(size_t output, size_t input, size_t kernel, const float * src, float * dst)
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
                __mmask16 mask = TailMask16(tail);
                for (size_t k = 0; k < kernel; ++k)
                    for (size_t i = 0; i < input; ++i, src += output, dst += F)
                        CopyZP<align>(src, dst, mask);
            }
        }

        template<bool align> void SynetReorderFilter_Oyxi16o_Oiyx(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            if (kernel == 1)
            {
                SynetReorderImage_Chw16c_Chw<align>(output, input, src, dst);
                return;
            }
            size_t output16 = AlignLo(output, 16);
            size_t tail = output - output16;
            size_t kernel8 = AlignLo(kernel, 8);
            size_t ik = input * kernel;
            size_t stride = 16 * input;
            size_t o = 0;
            for (; o < output16; o += 16, src += 16 * ik)
            {
                const float * ps = src;
                float * pd = dst;
                for (size_t i = 0; i < input; ++i, ps += 16)
                {
                    size_t k = 0;
                    for (; k < kernel8; k += 8, pd += 8)
                        Transpose16x8<align>(ps + k * stride, stride, pd, ik);
                    for (; k < kernel; ++k, pd++)
                    {
                        pd[0x0 * ik] = ps[k*stride + 0x0];
                        pd[0x1 * ik] = ps[k*stride + 0x1];
                        pd[0x2 * ik] = ps[k*stride + 0x2];
                        pd[0x3 * ik] = ps[k*stride + 0x3];
                        pd[0x4 * ik] = ps[k*stride + 0x4];
                        pd[0x5 * ik] = ps[k*stride + 0x5];
                        pd[0x6 * ik] = ps[k*stride + 0x6];
                        pd[0x7 * ik] = ps[k*stride + 0x7];
                        pd[0x8 * ik] = ps[k*stride + 0x8];
                        pd[0x9 * ik] = ps[k*stride + 0x9];
                        pd[0xA * ik] = ps[k*stride + 0xA];
                        pd[0xB * ik] = ps[k*stride + 0xB];
                        pd[0xC * ik] = ps[k*stride + 0xC];
                        pd[0xD * ik] = ps[k*stride + 0xD];
                        pd[0xE * ik] = ps[k*stride + 0xE];
                        pd[0xF * ik] = ps[k*stride + 0xF];
                    }
                }
                dst += 16 * ik;
            }
            if (tail)
            {
                for (size_t j = 0; j < tail; ++j)
                {
                    const float * ps = src + j;
                    for (size_t i = 0; i < input; ++i, ps += 16)
                        for (size_t k = 0; k < kernel; ++k)
                            *(dst++) = ps[k*stride];
                }
            }
        }

        template<bool align> void SynetReorderFilter_Oyxi16o_Yxio(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            size_t outputF = AlignLo(output, F);
            size_t outputF4 = AlignLo(output, 4 * F);
            size_t tail = output - outputF;
            __mmask16 mask = TailMask16(tail);
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
                    Copy<align, true>(ps + 0 * F, pd + 0 * output, mask);
                    Copy<align, true>(ps + 1 * F, pd + 1 * output, mask);
                    Copy<align, true>(ps + 2 * F, pd + 2 * output, mask);
                    Copy<align, true>(ps + 3 * F, pd + 3 * output, mask);
                }
                dst += 4 * output;
            }
            for (; i < ki; ++i, src += F)
            {
                const float * ps = src;
                for (size_t o = 0; o < outputF; o += F, ps += stride, dst += F)
                    Copy<align>(ps, dst);
                if (tail)
                    Copy<align, true>(ps, dst, mask), dst += tail;
            }
        }

        typedef void(*SynetFilterConverterPtr)(size_t output, size_t input, size_t kernel, const float * src, float * dst);
        SynetFilterConverterPtr GetFilterConverter(SimdTensorFormatType src, SimdTensorFormatType dst)
        {
            if (src == SimdTensorFormatOiyx)
            {
                if (dst == SimdTensorFormatYxio)
                    return SynetReorderFilter_Oiyx_Yxio<false>;
                if (dst == SimdTensorFormatOyxi16o)
                    return SynetReorderFilter_Oiyx_Oyxi16o<false>;
            }
            if (src == SimdTensorFormatYxio)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetReorderFilter_Yxio_Oiyx<false>;
                if (dst == SimdTensorFormatOyxi16o)
                    return SynetReorderFilter_Yxio_Oyxi16o<false>;
            }
            if (src == SimdTensorFormatOyxi16o)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetReorderFilter_Oyxi16o_Oiyx<false>;
                if (dst == SimdTensorFormatYxio)
                    return SynetReorderFilter_Oyxi16o_Yxio<false>;
            }
            return NULL;
        }

        void SynetReorderFilter(size_t output, size_t input, size_t kernel, const float * src, SimdTensorFormatType srcFormat, float * dst, SimdTensorFormatType dstFormat)
        {
            SynetFilterConverterPtr filterConverter = GetFilterConverter(srcFormat, dstFormat);
            if (filterConverter)
                filterConverter(output, input, kernel, src, dst);
            else
                Avx::SynetReorderFilter(output, input, kernel, src, srcFormat, dst, dstFormat);
        }
    }
#endif// SIMD_AVX512F_ENABLE
}
