/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdSse41.h"

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
                return Sse41::SynetReorderImage(batch, channels, spatial, src, srcFormat, dst, dstFormat);
        }
    }
#endif// SIMD_AVX_ENABLE
}
