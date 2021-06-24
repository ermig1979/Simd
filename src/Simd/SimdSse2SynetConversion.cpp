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
#include "Simd/SimdMath.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdTranspose.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_SYNET_ENABLE)  
    namespace Sse2
    {
        template <bool align> SIMD_INLINE void SynetConvert32fTo8u(const float * src, __m128 scale, __m128 shift, __m128i upper, uint8_t* dst)
        {
            __m128i i32 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Load<align>(src), scale), shift));
            *((int32_t*)dst) = _mm_cvtsi128_si32(_mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(i32, K_ZERO), K_ZERO), upper));
        }

        template <bool align> SIMD_INLINE void SynetConvert32fTo8uNchw(const float* src, __m128 scale, __m128 shift, __m128i upper, uint8_t* dst)
        {
            __m128i i32_0 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Load<align>(src + 0 * F), scale), shift));
            __m128i i32_1 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Load<align>(src + 1 * F), scale), shift));
            __m128i i32_2 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Load<align>(src + 2 * F), scale), shift));
            __m128i i32_3 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Load<align>(src + 3 * F), scale), shift));
            Store<align>((__m128i*)dst, _mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(i32_0, i32_1), _mm_packs_epi32(i32_2, i32_3)), upper));
        }        
        
        template <bool align> void SynetConvert32fTo8uNchw(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(spatial, A));

            size_t spatialF = AlignLo(spatial, F);
            size_t spatialA = AlignLo(spatial, A);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m128 _scale = _mm_set1_ps(scale[c]);
                    __m128 _shift = _mm_set1_ps(shift[c]);
                    size_t s = 0;
                    for (; s < spatialA; s += A)
                        SynetConvert32fTo8uNchw<align>(src + s, _scale, _shift, _upper, dst + s);
                    for (; s < spatialF; s += F)
                        SynetConvert32fTo8u<align>(src + s, _scale, _shift, _upper, dst + s);
                    for (; s < spatial; s += 1)
                            dst[s] = Base::SynetConvert32fTo8u(src[s], scale[c], shift[c], 0, upper);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template <bool align> void SynetConvert32fTo8uNhwc(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(channels, A) && Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        SynetConvert32fTo8u<align>(src + c, Load<align>(scale + c), Load<align>(shift + c), _upper, dst + c);
                    for (; c < channels; ++c)
                        dst[c] = Base::SynetConvert32fTo8u(src[c], scale[c], shift[c], 0, upper);
                    src += channels;
                    dst += channels;
                }
            }
        }

        template <bool align> void SynetConvert32fTo8uNhwc3(const float* src, size_t batch, size_t spatial, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(spatial, A));

            size_t spatial3 = spatial * 3;
            size_t spatial3F = AlignLo(spatial, F) * 3;
            __m128i _upper = _mm_set1_epi8(upper);
            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];

            __m128 _scale0 = Load<false>(_scale + 0 * F);
            __m128 _scale1 = Load<false>(_scale + 1 * F);
            __m128 _scale2 = Load<false>(_scale + 2 * F);
            __m128 _shift0 = Load<false>(_shift + 0 * F);
            __m128 _shift1 = Load<false>(_shift + 1 * F);
            __m128 _shift2 = Load<false>(_shift + 2 * F);

            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatial3F; s += 3 * F)
                {
                    SynetConvert32fTo8u<align>(src + 0 * F, _scale0, _shift0, _upper, dst + 0 * F);
                    SynetConvert32fTo8u<align>(src + 1 * F, _scale1, _shift1, _upper, dst + 1 * F);
                    SynetConvert32fTo8u<align>(src + 2 * F, _scale2, _shift2, _upper, dst + 2 * F);
                    src += 3 * F;
                    dst += 3 * F;
                }
                for (; s < spatial3; s += 3)
                {
                    dst[0] = Base::SynetConvert32fTo8u(src[0], scale[0], shift[0], 0, upper);
                    dst[1] = Base::SynetConvert32fTo8u(src[1], scale[1], shift[1], 0, upper);
                    dst[2] = Base::SynetConvert32fTo8u(src[2], scale[2], shift[2], 0, upper);
                    src += 3;
                    dst += 3;
                }
            }
        }

        void SynetConvert32fTo8u(const float* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, uint8_t* dst, SimdSynetCompatibilityType compatibility)
        {
            int upper = Base::Narrowed(compatibility) ? Base::U8_NARROWED_MAX : Base::U8_PRECISE_MAX;
            size_t spatial = height * width;
            if (Base::NchwCompatible(channels, spatial, format))
            {
                if(Aligned(src) && Aligned(dst) && Aligned(spatial, A))
                    SynetConvert32fTo8uNchw<true>(src, batch, channels, spatial, scale, shift, upper, dst);
                else
                    SynetConvert32fTo8uNchw<false>(src, batch, channels, spatial, scale, shift, upper, dst);
            }
            else if (Base::NhwcCompatible(channels, spatial, format))
            {
                if (channels == 3)
                {
                    if (Aligned(src) && Aligned(dst) && Aligned(spatial, A))
                        SynetConvert32fTo8uNhwc3<true>(src, batch, spatial, scale, shift, upper, dst);
                    else
                        SynetConvert32fTo8uNhwc3<false>(src, batch, spatial, scale, shift, upper, dst);
                }
                else
                {
                    if (Aligned(src) && Aligned(dst) && Aligned(channels, A) && Aligned(scale) && Aligned(shift))
                        SynetConvert32fTo8uNhwc<true>(src, batch, channels, spatial, scale, shift, upper, dst);
                    else
                        SynetConvert32fTo8uNhwc<false>(src, batch, channels, spatial, scale, shift, upper, dst);
                }
            }
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        template<bool align> void SynetReorderImage_Chw_Hwc(size_t channels, size_t spatial, const float* src, float* dst)
        {
            size_t channels4 = AlignLo(channels, 4);
            size_t spatial4 = AlignLo(spatial, 4);
            size_t s = 0;
            for (; s < spatial4; s += 4, src += 4, dst += 4 * channels)
            {
                size_t c = 0;
                const float* ps = src;
                float* pd = dst;
                for (; c < channels4; c += 4, ps += 4 * spatial, pd += 4)
                    Transpose4x4<align>(ps, spatial, pd, channels);
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
                    dst[c] = src[c * spatial];
        }

        template<bool align> void SynetReorderImage_Chw_Chw4c(size_t channels, size_t spatial, const float* src, float* dst)
        {
            size_t channels4 = AlignLo(channels, 4);
            size_t spatial4 = AlignLo(spatial, 4);
            size_t tail = channels - channels4;
            size_t c = 0;
            for (; c < channels4; c += 4, src += 4 * spatial)
            {
                size_t s = 0;
                const float* ps = src;
                for (; s < spatial4; s += 4, dst += 4 * F, ps += 4)
                    Transpose4x4<align>(ps, spatial, dst, 4);
                for (; s < spatial; ++s, dst += F, ps += 1)
                {
                    dst[0] = ps[0 * spatial];
                    dst[1] = ps[1 * spatial];
                    dst[2] = ps[2 * spatial];
                    dst[3] = ps[3 * spatial];
                }
            }
            if (tail)
            {
                const float* ps = src;
                for (size_t s = 0; s < spatial; ++s, dst += F, ps += 1)
                {
                    size_t i = 0;
                    for (; i < tail; ++i)
                        dst[i] = ps[i * spatial];
                    for (; i < F; ++i)
                        dst[i] = 0;
                }
            }
        }

        template<bool align> void SynetReorderImage_Hwc_Chw(size_t channels, size_t spatial, const float* src, float* dst)
        {
            SynetReorderImage_Chw_Hwc<align>(spatial, channels, src, dst);
        }

        template<bool align> void SynetReorderImage_Hwc_Chw4c(size_t channels, size_t spatial, const float* src, float* dst)
        {
            size_t channelsF = AlignLo(channels, F);
            size_t channelsF4 = AlignLo(channels, 4 * F);
            size_t tail = channels - channelsF;
            size_t spatial4 = AlignLo(spatial, 4);
            size_t stride = spatial * F;
            size_t c = 0;
            for (; c < channelsF4; c += 4 * F, src += 4 * F)
            {
                const float* ps = src;
                float* pd = dst;
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
                const float* ps = src;
                for (size_t s = 0; s < spatial; ++s, ps += channels, dst += F)
                    Copy<align>(ps, dst);
            }
            if (tail)
            {
                const float* psrc = src;
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

        template<bool align> void SynetReorderImage_Chw4c_Chw(size_t channels, size_t spatial, const float* src, float* dst)
        {
            size_t channels4 = AlignLo(channels, 4);
            size_t spatial4 = AlignLo(spatial, 4);
            size_t tail = channels - channels4;
            size_t c = 0;
            for (; c < channels4; c += 4, dst += 4 * spatial, src += 4 * spatial)
            {
                const float* ps = src;
                size_t s = 0;
                for (; s < spatial4; s += 4, ps += 4 * F)
                    Transpose4x4<align>(ps, 4, dst + s, spatial);
                for (; s < spatial; ++s, ps += 4)
                {
                    dst[s + 0 * spatial] = ps[0];
                    dst[s + 1 * spatial] = ps[1];
                    dst[s + 2 * spatial] = ps[2];
                    dst[s + 3 * spatial] = ps[3];
                }
            }
            if (tail)
            {
                const float* ps = src;
                for (size_t i = 0; i < tail; ++i, ps += 1, dst += spatial)
                {
                    for (size_t s = 0; s < spatial; ++s)
                        dst[s] = ps[s * F];
                }
            }
        }

        template<bool align> void SynetReorderImage_Chw4c_Hwc(size_t channels, size_t spatial, const float* src, float* dst)
        {
            size_t stride = F * spatial;
            size_t channelsF = AlignLo(channels, F);
            size_t channelsF4 = AlignLo(channels, 4 * F);
            size_t tail = channels - channelsF;
            size_t spatial4 = AlignLo(spatial, 4);
            size_t s = 0;
            for (; s < spatial4; s += 4, src += 4 * F, dst += 4 * channels)
            {
                const float* ps = src;
                float* pd = dst;
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
                const float* ps = src;
                for (size_t c = 0; c < channelsF; c += F, ps += stride, dst += F)
                    Copy<align>(ps, dst);
                if (tail)
                {
                    for (size_t i = 0; i < tail; ++i)
                        *(dst++) = ps[i];
                }
            }
        }

        typedef void(*SynetImageConverterPtr)(size_t channels, size_t spatial, const float* src, float* dst);
        SynetImageConverterPtr GetImageConverter(SimdTensorFormatType src, SimdTensorFormatType dst)
        {
            if (src == SimdTensorFormatNchw)
            {
                if (dst == SimdTensorFormatNhwc)
                    return SynetReorderImage_Chw_Hwc<false>;
                if (dst == SimdTensorFormatNchw4c)
                    return SynetReorderImage_Chw_Chw4c<false>;
            }
            if (src == SimdTensorFormatNhwc)
            {
                if (dst == SimdTensorFormatNchw)
                    return SynetReorderImage_Hwc_Chw<false>;
                if (dst == SimdTensorFormatNchw4c)
                    return SynetReorderImage_Hwc_Chw4c<false>;
            }
            if (src == SimdTensorFormatNchw4c)
            {
                if (dst == SimdTensorFormatNchw)
                    return SynetReorderImage_Chw4c_Chw<false>;
                if (dst == SimdTensorFormatNhwc)
                    return SynetReorderImage_Chw4c_Hwc<false>;
            }
            return NULL;
        }

        void SynetReorderImage(size_t batch, size_t channels, size_t spatial, const float* src, SimdTensorFormatType srcFormat, float* dst, SimdTensorFormatType dstFormat)
        {
            SynetImageConverterPtr imageConverter = GetImageConverter(srcFormat, dstFormat);
            if (imageConverter)
            {
                size_t srcStride = AlignHi(channels, Base::SynetTensorAlignment(srcFormat)) * spatial;
                size_t dstStride = AlignHi(channels, Base::SynetTensorAlignment(dstFormat)) * spatial;
                for (size_t n = 0; n < batch; ++n)
                {
                    imageConverter(channels, spatial, src, dst);
                    src += srcStride;
                    dst += dstStride;
                }
            }
            else
                return Base::SynetReorderImage(batch, channels, spatial, src, srcFormat, dst, dstFormat);
        }

        //---------------------------------------------------------------------

        template<bool align> void SynetReorderFilter_Oiyx_Yxio(size_t output, size_t input, size_t kernel, const float* src, float* dst)
        {
            if (kernel == 1)
            {
                SynetReorderImage_Chw_Hwc<align>(output, input, src, dst);
                return;
            }
            size_t output4 = AlignLo(output, 4);
            size_t kernel4 = AlignLo(kernel, 4);
            size_t ik = input * kernel, oi = output * input;
            for (size_t i = 0; i < input; ++i, src += kernel, dst += output)
            {
                const float* ps = src;
                float* pd = dst;
                size_t k = 0;
                for (; k < kernel4; k += 4, ps += 4, pd += 4 * oi)
                {
                    size_t o = 0;
                    for (; o < output4; o += 4)
                        Transpose4x4<align>(ps + o * ik, ik, pd + o, oi);
                    for (; o < output; ++o)
                    {
                        pd[0 * oi + o] = ps[o * ik + 0];
                        pd[1 * oi + o] = ps[o * ik + 1];
                        pd[2 * oi + o] = ps[o * ik + 2];
                        pd[3 * oi + o] = ps[o * ik + 3];
                    }
                }
                for (; k < kernel; ++k, ps += 1, pd += oi)
                    for (size_t o = 0; o < output; ++o)
                        pd[o] = ps[o * ik];
            }
        }

        template<bool align> void SynetReorderFilter_Oiyx_Oyxi4o(size_t output, size_t input, size_t kernel, const float* src, float* dst)
        {
            if (kernel == 1)
            {
                SynetReorderImage_Chw_Chw4c<align>(output, input, src, dst);
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
                    const float* ps = src + o * ik + i * kernel;
                    float* pd = dst + o * ik + i * F;
                    size_t k = 0;
                    for (; k < kernelF; k += F, ps += F, pd += F * stride)
                        Transpose4x4<align>(ps, ik, pd, stride);
                    for (; k < kernel; ++k, ps += 1, pd += stride)
                        for (size_t j = 0; j < F; ++j)
                            pd[j] = ps[j * ik];
                }
            }
            if (tail)
            {
                for (size_t i = 0; i < input; ++i)
                {
                    const float* ps = src + outputF * ik + i * kernel;
                    float* pd = dst + outputF * ik + i * F;
                    for (size_t k = 0; k < kernel; ++k, ps += 1, pd += stride)
                    {
                        size_t j = 0;
                        for (; j < tail; ++j)
                            pd[j] = ps[j * ik];
                        for (; j < F; ++j)
                            pd[j] = 0;
                    }
                }
            }
        }

        template<bool align> void SynetReorderFilter_Yxio_Oiyx(size_t output, size_t input, size_t kernel, const float* src, float* dst)
        {
            if (kernel == 1)
            {
                SynetReorderImage_Chw_Hwc<align>(input, output, src, dst);
                return;
            }
            SynetReorderFilter_Oiyx_Yxio<align>(kernel, input, output, src, dst);
        }

        template<bool align> void SynetReorderFilter_Yxio_Oyxi4o(size_t output, size_t input, size_t kernel, const float* src, float* dst)
        {
            size_t outputF = AlignLo(output, F);
            size_t outputF4 = AlignLo(output, F * 4);
            size_t ki = kernel * input;
            size_t stride = ki * F;
            size_t ki4 = AlignLo(ki, 4);
            size_t o = 0;
            for (; o < outputF4; o += 4 * F, src += 4 * F)
            {
                const float* ps = src;
                float* pd = dst;
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
                const float* ps = src;
                float* pd = dst;
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

        template<bool align> void SynetReorderFilter_Oyxi4o_Oiyx(size_t output, size_t input, size_t kernel, const float* src, float* dst)
        {
            if (kernel == 1)
            {
                SynetReorderImage_Chw4c_Chw<align>(output, input, src, dst);
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
                const float* ps = src;
                float* pd = dst;
                for (size_t i = 0; i < input; ++i, ps += F)
                {
                    size_t k = 0;
                    for (; k < kernelF; k += F, pd += F)
                        Transpose4x4<align>(ps + k * stride, stride, pd, ik);
                    for (; k < kernel; ++k, pd++)
                    {
                        pd[0 * ik] = ps[k * stride + 0];
                        pd[1 * ik] = ps[k * stride + 1];
                        pd[2 * ik] = ps[k * stride + 2];
                        pd[3 * ik] = ps[k * stride + 3];
                    }
                }
                dst += F * ik;
            }
            if (tail)
            {
                for (size_t j = 0; j < tail; ++j)
                {
                    const float* ps = src + j;
                    for (size_t i = 0; i < input; ++i, ps += F)
                        for (size_t k = 0; k < kernel; ++k)
                            *(dst++) = ps[k * stride];
                }
            }
        }

        template<bool align> void SynetReorderFilter_Oyxi4o_Yxio(size_t output, size_t input, size_t kernel, const float* src, float* dst)
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
                const float* ps = src;
                float* pd = dst;
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
                const float* ps = src;
                for (size_t o = 0; o < outputF; o += F, ps += stride, dst += F)
                    Copy<align>(ps, dst);
                if (tail)
                {
                    for (size_t j = 0; j < tail; ++j)
                        *(dst++) = ps[j];
                }
            }
        }

        typedef void(*SynetFilterConverterPtr)(size_t output, size_t input, size_t kernel, const float* src, float* dst);
        SynetFilterConverterPtr GetFilterConverter(SimdTensorFormatType src, SimdTensorFormatType dst)
        {
            if (src == SimdTensorFormatOiyx)
            {
                if (dst == SimdTensorFormatYxio)
                    return SynetReorderFilter_Oiyx_Yxio<false>;
                if (dst == SimdTensorFormatOyxi4o)
                    return SynetReorderFilter_Oiyx_Oyxi4o<false>;
            }
            if (src == SimdTensorFormatYxio)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetReorderFilter_Yxio_Oiyx<false>;
                if (dst == SimdTensorFormatOyxi4o)
                    return SynetReorderFilter_Yxio_Oyxi4o<false>;
            }
            if (src == SimdTensorFormatOyxi4o)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetReorderFilter_Oyxi4o_Oiyx<false>;
                if (dst == SimdTensorFormatYxio)
                    return SynetReorderFilter_Oyxi4o_Yxio<false>;
            }
            return NULL;
        }

        void SynetReorderFilter(size_t output, size_t input, size_t kernel, const float* src, SimdTensorFormatType srcFormat, float* dst, SimdTensorFormatType dstFormat)
        {
            SynetFilterConverterPtr filterConverter = GetFilterConverter(srcFormat, dstFormat);
            if (filterConverter)
                filterConverter(output, input, kernel, src, dst);
            else
                Base::SynetReorderFilter(output, input, kernel, src, srcFormat, dst, dstFormat);
        }
    }
#endif// SIMD_SSE2_ENABLE
}
