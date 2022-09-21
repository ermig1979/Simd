/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
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
#include "Test/TestUtils.h"
#include "Test/TestTensor.h"
#include "Test/TestRandom.h"

#include "Simd/SimdSynet.h"

namespace Test
{
    void SetSrc32fTo8u(const Tensor32f& src, const float* min, const float* max, size_t channels, int negative, SimdSynetCompatibilityType compatibility, float* shift, float* scale, Tensor8u& dst)
    {
        assert(src.Shape() == dst.Shape() && src.Format() == dst.Format());
        int uMin = Simd::Base::Narrowed(compatibility) ? Simd::Base::U8_NARROWED_MIN : Simd::Base::U8_PRECISE_MIN;
        int uMax = Simd::Base::Narrowed(compatibility) ? Simd::Base::U8_NARROWED_MAX : Simd::Base::U8_PRECISE_MAX;
        int iMin = Simd::Base::Narrowed(compatibility) ? Simd::Base::I8_NARROWED_MIN : Simd::Base::I8_PRECISE_MIN;
        int iMax = Simd::Base::Narrowed(compatibility) ? Simd::Base::I8_NARROWED_MAX : Simd::Base::I8_PRECISE_MAX;
        Tensor32f buffer;
        if (scale == NULL && shift == NULL)
        {
            buffer.Reshape(Shp(2, channels));
            scale = buffer.Data(Shp(0, 0));
            shift = buffer.Data(Shp(1, 0));
        }
        for (size_t i = 0; i < channels; ++i)
        {
            float abs = std::max(Simd::Abs(min[i]), Simd::Abs(max[i]));
            scale[i] = (negative ? iMax : uMax) / abs;
            shift[i] = float(negative ? -iMin : uMin);
        }
        if (src.Count() == 4)
        {
            for (size_t b = 0; b < src.Axis(0); ++b)
            {
                if (src.Format() == SimdTensorFormatNhwc)
                {
                    for (size_t y = 0; y < src.Axis(1); ++y)
                        for (size_t x = 0; x < src.Axis(2); ++x)
                            for (size_t c = 0; c < src.Axis(3); ++c)
                                dst.Data({ b, y, x, c })[0] = Simd::Base::SynetConvert32fTo8u(src.Data({ b, y, x, c })[0], scale[c], shift[c], uMin, uMax);
                }
                else
                {
                    for (size_t c = 0; c < src.Axis(1); ++c)
                        for (size_t y = 0; y < src.Axis(2); ++y)
                            for (size_t x = 0; x < src.Axis(3); ++x)
                                dst.Data({ b, c, y, x })[0] = Simd::Base::SynetConvert32fTo8u(src.Data({ b, c, y, x })[0], scale[c], shift[c], uMin, uMax);
                }
            }
        }
        else if (src.Count() == 3)
        {
            for (size_t b = 0; b < src.Axis(0); ++b)
            {
                if (src.Format() == SimdTensorFormatNhwc)
                {
                    for (size_t s = 0; s < src.Axis(1); ++s)
                        for (size_t c = 0; c < src.Axis(2); ++c)
                            dst.Data({ b, s, c })[0] = Simd::Base::SynetConvert32fTo8u(src.Data({ b, s, c })[0], scale[c], shift[c], uMin, uMax);
                }
                else
                {
                    for (size_t c = 0; c < src.Axis(1); ++c)
                        for (size_t s = 0; s < src.Axis(2); ++s)
                            dst.Data({ b, c, s })[0] = Simd::Base::SynetConvert32fTo8u(src.Data({ b, c, s })[0], scale[c], shift[c], uMin, uMax);
                }
            }
        }
        else
            assert(0);
    }

    void SetDstStat(size_t channels, int negative, SimdSynetCompatibilityType compatibility, const Tensor32f& dst, float* min, float* max, float * scale, float * shift)
    {
        Fill(min, channels, FLT_MAX);
        Fill(max, channels, -FLT_MAX);
        if (dst.Count() == 4)
        {
            for (size_t b = 0; b < dst.Axis(0); ++b)
            {
                if (dst.Format() == SimdTensorFormatNhwc)
                {
                    for (size_t y = 0; y < dst.Axis(1); ++y)
                        for (size_t x = 0; x < dst.Axis(2); ++x)
                            for (size_t c = 0; c < dst.Axis(3); ++c)
                            {
                                min[c] = std::min(min[c], dst.Data({ b, y, x, c })[0]);
                                max[c] = std::max(max[c], dst.Data({ b, y, x, c })[0]);
                            }
                }
                else
                {
                    for (size_t c = 0; c < dst.Axis(1); ++c)
                        for (size_t y = 0; y < dst.Axis(2); ++y)
                            for (size_t x = 0; x < dst.Axis(3); ++x)
                            {
                                min[c] = std::min(min[c], dst.Data({ b, c, y, x })[0]);
                                max[c] = std::max(max[c], dst.Data({ b, c, y, x })[0]);
                            }
                }
            }
        }
        else if (dst.Count() == 3)
        {
            for (size_t b = 0; b < dst.Axis(0); ++b)
            {
                if (dst.Format() == SimdTensorFormatNhwc)
                {
                    for (size_t s = 0; s < dst.Axis(1); ++s)
                        for (size_t c = 0; c < dst.Axis(2); ++c)
                        {
                            min[c] = std::min(min[c], dst.Data({ b, s, c })[0]);
                            max[c] = std::max(max[c], dst.Data({ b, s, c })[0]);
                        }
                }
                else
                {
                    for (size_t c = 0; c < dst.Axis(1); ++c)
                        for (size_t s = 0; s < dst.Axis(2); ++s)
                        {
                            min[c] = std::min(min[c], dst.Data({ b, c, s })[0]);
                            max[c] = std::max(max[c], dst.Data({ b, c, s })[0]);
                        }
                }
            }
        }
        else
            assert(0);
        int uMin = Simd::Base::Narrowed(compatibility) ? Simd::Base::U8_NARROWED_MIN : Simd::Base::U8_PRECISE_MIN;
        int uMax = Simd::Base::Narrowed(compatibility) ? Simd::Base::U8_NARROWED_MAX : Simd::Base::U8_PRECISE_MAX;
        int iMin = Simd::Base::Narrowed(compatibility) ? Simd::Base::I8_NARROWED_MIN : Simd::Base::I8_PRECISE_MIN;
        int iMax = Simd::Base::Narrowed(compatibility) ? Simd::Base::I8_NARROWED_MAX : Simd::Base::I8_PRECISE_MAX;
        if (scale != NULL && shift != NULL)
        {
            for (size_t i = 0; i < channels; ++i)
            {
                float abs = std::max(Simd::Abs(min[i]), Simd::Abs(max[i]));
                scale[i] = abs / (negative ? iMax : uMax);
                shift[i] = float(negative ? iMin : uMin) * scale[i];
            }            
        }
    }
}
