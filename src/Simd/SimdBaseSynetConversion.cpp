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
#include "Simd/SimdBase.h"
#include "Simd/SimdConversion.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        void SynetConvert32fTo8u(const float* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, uint8_t* dst, SimdSynetCompatibilityType compatibility)
        {
            int upper = Base::Narrowed(compatibility) ? Base::U8_NARROWED_MAX : Base::U8_PRECISE_MAX;
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
                                dst[w] = SynetConvert32fTo8u(src[w], _scale, _shift, 0, upper);
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
                                dst[c] = SynetConvert32fTo8u(src[c], scale[c], shift[c], 0, upper);
                            src += channels;
                            dst += channels;
                        }
                    }
                }
                else
                    assert(0);
            }
        }

        //---------------------------------------------------------------------

        void SynetConvert8uTo32f(const uint8_t * src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, float* dst, SimdSynetCompatibilityType compatibility)
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
                                dst[w] = SynetConvert8uTo32f(src[w], _scale, _shift);
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
                                dst[c] = SynetConvert8uTo32f(src[c], scale[c], shift[c]);
                            src += channels;
                            dst += channels;
                        }
                    }
                }
                else
                    assert(0);
            }
        }

        //---------------------------------------------------------------------

        template<SimdPixelFormatType format> SIMD_INLINE int ToGray(const uint8_t* src);

        template<> SIMD_INLINE int ToGray<SimdPixelFormatGray8>(const uint8_t* src)
        {
            return src[0];
        }

        template<> SIMD_INLINE int ToGray<SimdPixelFormatBgr24>(const uint8_t* src)
        {
            return BgrToGray(src[0], src[1], src[2]);
        }

        template<> SIMD_INLINE int ToGray<SimdPixelFormatRgb24>(const uint8_t* src)
        {
            return BgrToGray(src[2], src[1], src[0]);
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInput1(const uint8_t* src, size_t width, size_t height, size_t stride, const float* scale, const float* shift, float* dst)
        {
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; ++x, src += step)
                    *dst++ = SynetConvert8uTo32f(ToGray<format>(src), scale[0], shift[0]);
                src += (stride - width * step);
            }
        }

        template<SimdPixelFormatType format> SIMD_INLINE int ToBgr(const uint8_t* src, size_t channel);

        template<> SIMD_INLINE int ToBgr<SimdPixelFormatGray8>(const uint8_t* src, size_t channel)
        {
            return src[0];
        }

        template<> SIMD_INLINE int ToBgr<SimdPixelFormatBgr24>(const uint8_t* src, size_t channel)
        {
            return src[channel];
        }

        template<> SIMD_INLINE int ToBgr<SimdPixelFormatRgb24>(const uint8_t* src, size_t channel)
        {
            return src[2 - channel];
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInputNchw3(const uint8_t* src, size_t width, size_t height, size_t stride, const float* scale, const float* shift, float* dst0)
        {
            float* dst1 = dst0 + width * height;
            float* dst2 = dst1 + width * height;
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; ++x, src += step)
                {
                    *dst0++ = SynetConvert8uTo32f(ToBgr<format>(src, 0), scale[0], shift[0]);
                    *dst1++ = SynetConvert8uTo32f(ToBgr<format>(src, 1), scale[1], shift[1]);
                    *dst2++ = SynetConvert8uTo32f(ToBgr<format>(src, 2), scale[2], shift[2]);
                }
                src += (stride - width * step);
            }
        }

        template<SimdPixelFormatType format, size_t step> void SynetSetInputNhwc3(const uint8_t* src, size_t width, size_t height, size_t stride, const float* scale, const float* shift, float* dst)
        {
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; ++x, src += step)
                {
                    *dst++ = SynetConvert8uTo32f(ToBgr<format>(src, 0), scale[0], shift[0]);
                    *dst++ = SynetConvert8uTo32f(ToBgr<format>(src, 1), scale[1], shift[1]);
                    *dst++ = SynetConvert8uTo32f(ToBgr<format>(src, 2), scale[2], shift[2]);
                }
                src += (stride - width * step);
            }
        }

        void SynetSetInput(const uint8_t* src, size_t width, size_t height, size_t stride, SimdPixelFormatType srcFormat,
            const float* lower, const float* upper, float* dst, size_t channels, SimdTensorFormatType dstFormat)
        {
            float scale[3];
            for (size_t i = 0; i < channels; ++i)
                scale[i] = (upper[i] - lower[i]) / 255.0f;
            switch (channels)
            {
            case 1:
                switch (srcFormat)
                {
                case SimdPixelFormatGray8: SynetSetInput1<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatBgr24: SynetSetInput1<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatBgra32: SynetSetInput1<SimdPixelFormatBgr24, 4>(src, width, height, stride, scale, lower, dst); return;
                case SimdPixelFormatRgb24: SynetSetInput1<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                default: assert(0);
                }
                break;
            case 3:
                switch (dstFormat)
                {
                case SimdTensorFormatNchw:
                    switch (srcFormat)
                    {
                    case SimdPixelFormatGray8: SynetSetInputNchw3<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgr24: SynetSetInputNchw3<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgra32: SynetSetInputNchw3<SimdPixelFormatBgr24, 4>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgb24: SynetSetInputNchw3<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                    default: assert(0);
                    }
                    break;
                case SimdTensorFormatNhwc:
                    switch (srcFormat)
                    {
                    case SimdPixelFormatGray8: SynetSetInputNhwc3<SimdPixelFormatGray8, 1>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgr24: SynetSetInputNhwc3<SimdPixelFormatBgr24, 3>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatBgra32: SynetSetInputNhwc3<SimdPixelFormatBgr24, 4>(src, width, height, stride, scale, lower, dst); return;
                    case SimdPixelFormatRgb24: SynetSetInputNhwc3<SimdPixelFormatRgb24, 3>(src, width, height, stride, scale, lower, dst); return;
                    default: assert(0);
                    }
                    break;
                default: assert(0);
                }
            default: assert(0);
            }
        }

        //---------------------------------------------------------------------

        template<size_t N> SIMD_INLINE void Copy(const float * src, float * dst)
        {
            for (size_t i = 0; i < N; ++i)
                dst[i] = src[i];
        }

        void SynetReorderImage_Chw_Hwc(size_t channels, size_t spatial, const float * src, float * dst)
        {
            for (size_t s = 0; s < spatial; ++s, src += 1, dst += channels)
                for (size_t c = 0; c < channels; ++c)
                    dst[c] = src[c*spatial];
        }

        template<size_t N> void SynetReorderImage_Chw_ChwXc(size_t channels, size_t spatial, const float * src, float * dst)
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
        
        void SynetReorderImage_Hwc_Chw(size_t channels, size_t spatial, const float * src, float * dst)
        {
            SynetReorderImage_Chw_Hwc(spatial, channels, src, dst);
        }

        template<size_t N> void SynetReorderImage_Hwc_ChwXc(size_t channels, size_t spatial, const float * src, float * dst)
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

        template<size_t N> void SynetReorderImage_ChwXc_Chw(size_t channels, size_t spatial, const float * src, float * dst)
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

        template<size_t N> void SynetReorderImage_ChwXc_Hwc(size_t channels, size_t spatial, const float * src, float * dst)
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
                    return SynetReorderImage_Chw_Hwc;
                if (dst == SimdTensorFormatNchw4c)
                    return SynetReorderImage_Chw_ChwXc<4>;
                if (dst == SimdTensorFormatNchw8c)
                    return SynetReorderImage_Chw_ChwXc<8>;
                if (dst == SimdTensorFormatNchw16c)
                    return SynetReorderImage_Chw_ChwXc<16>;
            }
            if (src == SimdTensorFormatNhwc)
            {
                if(dst == SimdTensorFormatNchw)
                    return SynetReorderImage_Hwc_Chw;
                if (dst == SimdTensorFormatNchw4c)
                    return SynetReorderImage_Hwc_ChwXc<4>;
                if (dst == SimdTensorFormatNchw8c)
                    return SynetReorderImage_Hwc_ChwXc<8>;
                if (dst == SimdTensorFormatNchw16c)
                    return SynetReorderImage_Hwc_ChwXc<16>;
            }
            if (src == SimdTensorFormatNchw4c)
            {
                if (dst == SimdTensorFormatNchw)
                    return SynetReorderImage_ChwXc_Chw<4>;
                if (dst == SimdTensorFormatNhwc)
                    return SynetReorderImage_ChwXc_Hwc<4>;
            }
            if (src == SimdTensorFormatNchw8c)
            {
                if (dst == SimdTensorFormatNchw)
                    return SynetReorderImage_ChwXc_Chw<8>;
                if (dst == SimdTensorFormatNhwc)
                    return SynetReorderImage_ChwXc_Hwc<8>;
            }
            if (src == SimdTensorFormatNchw16c)
            {
                if (dst == SimdTensorFormatNchw)
                    return SynetReorderImage_ChwXc_Chw<16>;
                if (dst == SimdTensorFormatNhwc)
                    return SynetReorderImage_ChwXc_Hwc<16>;
            }
            return NULL;
        }

        void SynetReorderImage(size_t batch, size_t channels, size_t spatial, const float * src, SimdTensorFormatType srcFormat, float * dst, SimdTensorFormatType dstFormat)
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

        //---------------------------------------------------------------------

        void SynetReorderFilter_Oiyx_Yxio(size_t output, size_t input, size_t kernel, const float * src, float * dst)
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

        template<size_t N> void SynetReorderFilter_Oiyx_OyxiXo(size_t output, size_t input, size_t kernel, const float * src, float * dst)
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

        void SynetReorderFilter_Yxio_Oiyx(size_t output, size_t input, size_t kernel, const float * src, float * dst)
        {
            SynetReorderFilter_Oiyx_Yxio(kernel, input, output, src, dst);
        }

        template<size_t N> void SynetReorderFilter_Yxio_OyxiXo(size_t output, size_t input, size_t kernel, const float * src, float * dst)
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

        template<size_t N> void SynetReorderFilter_OyxiXo_Oiyx(size_t output, size_t input, size_t kernel, const float * src, float * dst)
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

        template<size_t N> void SynetReorderFilter_OyxiXo_Yxio(size_t output, size_t input, size_t kernel, const float * src, float * dst)
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
                    return SynetReorderFilter_Oiyx_Yxio;
                if (dst == SimdTensorFormatOyxi4o)
                    return SynetReorderFilter_Oiyx_OyxiXo<4>;
                if (dst == SimdTensorFormatOyxi8o)
                    return SynetReorderFilter_Oiyx_OyxiXo<8>;
                if (dst == SimdTensorFormatOyxi16o)
                    return SynetReorderFilter_Oiyx_OyxiXo<16>;
            }
            if (src == SimdTensorFormatYxio)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetReorderFilter_Yxio_Oiyx;
                if (dst == SimdTensorFormatOyxi4o)
                    return SynetReorderFilter_Yxio_OyxiXo<4>;
                if (dst == SimdTensorFormatOyxi8o)
                    return SynetReorderFilter_Yxio_OyxiXo<8>;
                if (dst == SimdTensorFormatOyxi16o)
                    return SynetReorderFilter_Yxio_OyxiXo<16>;
            }
            if (src == SimdTensorFormatOyxi4o)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetReorderFilter_OyxiXo_Oiyx<4>;
                if (dst == SimdTensorFormatYxio)
                    return SynetReorderFilter_OyxiXo_Yxio<4>;
            }
            if (src == SimdTensorFormatOyxi8o)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetReorderFilter_OyxiXo_Oiyx<8>;
                if (dst == SimdTensorFormatYxio)
                    return SynetReorderFilter_OyxiXo_Yxio<8>;
            }
            if (src == SimdTensorFormatOyxi16o)
            {
                if (dst == SimdTensorFormatOiyx)
                    return SynetReorderFilter_OyxiXo_Oiyx<16>;
                if (dst == SimdTensorFormatYxio)
                    return SynetReorderFilter_OyxiXo_Yxio<16>;
            }
            return NULL;
        }

        void SynetReorderFilter(size_t output, size_t input, size_t kernel, const float * src, SimdTensorFormatType srcFormat, float * dst, SimdTensorFormatType dstFormat)
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
    }
#endif
}
