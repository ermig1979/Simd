/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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

        //-------------------------------------------------------------------------------------------------

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

        //-------------------------------------------------------------------------------------------------

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
                break;
            default: assert(0);
            }
        }
    }
#endif
}
