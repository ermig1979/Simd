/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdDrawing.hpp"
#include "Simd/SimdFont.hpp"
#include "Simd/SimdSynet.h"

#ifdef WIN32
#define NOMINMAX
#include <windows.h>
#include <filesystem>
#endif

#ifdef __linux__
#include <unistd.h>
#include <dirent.h>
#endif

namespace Test
{
    void FillSequence(View & view)
    {
        for (size_t i = 0, n = view.DataSize(); i < n; ++i)
            view.data[i] = uint8_t(i);
    }

    template <size_t N> struct Color
    {
        uint8_t val[N];

        Color(uint8_t v = 0)
        {
            for (size_t i = 0; i < N; ++i)
                val[i] = v;
        }
    };

    template <size_t N> Color<N> RandomColor(uint8_t lo = 0, uint8_t hi = 255)
    {
        Color<N> color;
        for (size_t i = 0; i < N; ++i)
            color.val[i] = lo + Random(hi - lo + 1);
        return color;
    }

    template<size_t N> void FillPicture(View & view, uint64_t flag)
    {
        typedef Test::Color<N> Color;
        if (flag & 1)
        {
            Simd::Fill(view, 15);
        }
        if (flag & 2)
        {
            size_t d = view.height / 20;
            Rect rect(d, d, view.width - d, view.height - d);
            Simd::DrawRectangle(view, rect, RandomColor<N>(), d/2);
        }
        if (flag & 4)
        {
            size_t size = Simd::Min<size_t>(view.height / 2, 256);
            Simd::Font font(size);
            font.Draw(view, "A1", View::MiddleCenter, RandomColor<N>(128, 255));
        }
    }

    void FillPicture(View & view, uint64_t flag)
    {
        switch (view.PixelSize())
        {
        case 1: FillPicture<1>(view, flag); break;
        case 2: FillPicture<2>(view, flag); break;
        case 3: FillPicture<3>(view, flag); break;
        case 4: FillPicture<4>(view, flag); break;
        }
    }

    uint8_t g_rand8u[UINT16_MAX];
    bool InitRand8u()
    {
        for (size_t i = 0, n = UINT16_MAX; i < n; ++i)
            g_rand8u[i] = ::rand();
        return true;
    }
    bool g_rand8u_inited = InitRand8u();
    SIMD_INLINE const uint8_t * Rand8u()
    {
        return g_rand8u + (::rand()&INT16_MAX);
    }

    int16_t g_rand16i[UINT16_MAX];
    bool InitRand16i()
    {
        for (size_t i = 0, n = UINT16_MAX; i < n; ++i)
            g_rand16i[i] = (::rand() & INT16_MAX);
        return true;
    }
    bool g_rand16i_inited = InitRand16i();
    SIMD_INLINE const int16_t* Rand16i()
    {
        return g_rand16i + (::rand() & INT16_MAX);
    }

    void FillRandom(View & view, uint8_t lo, uint8_t hi)
    {
        assert(view.data);

        size_t width = view.width*View::PixelSize(view.format);
        bool fast = (lo == 0) && (hi == 255);
        for (size_t row = 0; row < view.height; ++row)
        {
            ptrdiff_t offset = row*view.stride;
            if (fast)
            {
                for (size_t col = 0; col < width; col += INT16_MAX)
                    memcpy(view.data + offset + col, Rand8u(), std::min<size_t>(INT16_MAX, width - col));
            }
            else
            {
                for (size_t col = 0; col < width; ++col, ++offset)
                    view.data[offset] = lo + Random(hi - lo + 1);
            }
        }
    }

    void FillRandom2(View & view, uint8_t lo, uint8_t hi, uint8_t step)
    {
        assert(view.data && view.height && view.width && view.format == View::Gray8);

        for (size_t row = 0; row < view.height; ++row)
        {
            if (row & 1)
            {
                for (ptrdiff_t col = view.width - 1; col >= 0; --col)
                {
                    int l = lo, h = hi;
                    if (row)
                    {
                        int v = view.At<uint8_t>(col, row - 1);
                        l = std::max(v - step, l);
                        h = std::min(v + step, h);
                    }
                    if (col != view.width - 1)
                    {
                        int v = view.At<uint8_t>(col + 1, row);
                        l = std::max(v - step / 2, l);
                        h = std::min(v + step / 2, h);
                    }
                    int r = h - l + 1;
                    int v = l + Random(r);
                    view.At<uint8_t>(col, row) = v;
                }
            }
            else
            {
                for (size_t col = 0; col < view.width; ++col)
                {
                    int l = lo, h = hi;
                    if (row)
                    {
                        int v = view.At<uint8_t>(col, row - 1);
                        l = std::max(v - step, l);
                        h = std::min(v + step, h);
                    }
                    if (col)
                    {
                        int v = view.At<uint8_t>(col - 1, row);
                        l = std::max(v - step / 2, l);
                        h = std::min(v + step / 2, h);
                    }
                    int r = h - l + 1;
                    int v = l + Random(r);
                    view.At<uint8_t>(col, row) = v;
                }
            }
        }

        View buff(view.Size(), View::Gray8);
        Simd::GaussianBlur3x3(view, buff);
        view.Swap(buff);
    }

    void FillRandomMask(View & view, uint8_t index)
    {
        assert(view.data);

        size_t width = view.width*View::PixelSize(view.format);
        for (size_t row = 0; row < view.height; ++row)
        {
            ptrdiff_t offset = row*view.stride;
            const uint8_t * rand = Rand8u();
            for (size_t col = 0; col < width; ++col, ++offset)
                view.data[offset] = (rand[col] & 1) ? index : 0;
        }
    }

    void FillRhombMask(View & mask, const Rect & rect, uint8_t index)
    {
        assert(mask.format == View::Gray8 && Rect(mask.Size()).Contains(rect));

        Simd::Fill(mask, 0);

        Point c = rect.Center();
        for (ptrdiff_t row = rect.top; row < rect.bottom; ++row)
        {
            ptrdiff_t indent = Simd::Abs(row - c.y)*rect.Width() / rect.Height();
            ptrdiff_t left = rect.left + indent;
            ptrdiff_t right = rect.right - indent;
            ptrdiff_t offset = row*mask.stride + left;
            const uint8_t * rand = Rand8u();
            for (ptrdiff_t col = left; col < right; ++col, ++offset)
                mask.data[offset] = (rand[col] & 1) ? index : 0;
        }
    }

    void FillRandom32f(View & view, float lo, float hi)
    {
        assert(view.format == View::Float);

        bool fast = view.Area() > 100000;
        float boost = (hi - lo) / UCHAR_MAX;
        for (size_t row = 0; row < view.height; ++row)
        {
            if (fast)
            {
                for (size_t col = 0; col < view.width; col += INT16_MAX)
                {
                    size_t size = std::min<size_t>(INT16_MAX, view.width - col);
                    float * dst = & view.At<float>(col, row);
                    const uint8_t * src = Rand8u();
                    for (size_t i = 0; i < size; ++i)
                        dst[i] = lo + boost*src[i];
                }
            }
            else
            {
                for (size_t col = 0; col < view.width; ++col)
                    view.At<float>(col, row) = lo + (hi - lo)*(float)Random();
            }
        }
    }

    void FillRandom(float * data, size_t size, float lo, float hi)
    {
        bool fast = size > 100000;
        float boost = (hi - lo) / SHRT_MAX;
        if (fast)
        {
            for (size_t i = 0; i < size; i += INT16_MAX)
            {
                size_t n = std::min<size_t>(INT16_MAX, size - i);
                const int16_t * src = Rand16i();
                for (size_t j = 0; j < n; ++j)
                    data[i + j] = lo + boost * src[j];
            }
        }
        else
        {
            for (size_t i = 0; i < size; ++i)
                data[i] = lo + (hi - lo)*(float)Random();
        }
    }

    void FillRandom(Buffer32f & buffer, float lo, float hi)
    {
        FillRandom(buffer.data(), buffer.size(), lo, hi);
    }

    void FillRandom(Tensor32f & tensor, float lo, float hi)
    {
        FillRandom(tensor.Data(), tensor.Size(), lo, hi);
    }

    void FillRandom(uint8_t* data, size_t size, uint8_t lo, uint8_t hi)
    {
        bool fast = (lo == 0) && (hi == 255);
        if (fast)
        {
            for (size_t i = 0; i < size; i += INT16_MAX)
                memcpy(data + i, Rand8u(), std::min<size_t>(INT16_MAX, size - i));
        }
        else
        {
            for (size_t i = 0; i < size; ++i)
                data[i] = lo + Random(hi - lo + 1);
        }
    }

    void FillRandom(Tensor8u & tensor, uint8_t lo, uint8_t hi)
    {
        FillRandom(tensor.Data(), tensor.Size(), lo, hi);
    }

    void FillRandom(Tensor8i& tensor, int8_t lo, int8_t hi)
    {
        for (size_t i = 0; i < tensor.Size(); ++i)
            tensor.Data()[i] = lo + Random(hi - lo + 1);
    }

    void FillRandom(Tensor32i& tensor, int32_t lo, int32_t hi)
    {
        for (size_t i = 0; i < tensor.Size(); ++i)
            tensor.Data()[i] = lo + Random(hi - lo + 1);
    }

    //-------------------------------------------------------------------------

    void FillRandom(Tensor32f & tensor, float* min, float* max, size_t channels, int negative, float upper, float range)
    {
        const float lower = negative ? -upper : 0.0f;
        Buffer32f buf(channels * 2);
        FillRandom(buf, lower + range, upper - range);
        for (size_t i = 0; i < channels; ++i)
        {
            min[i] = negative ? std::min(buf[i * 2 + 0], buf[i * 2 + 1]) - range : 0;
            max[i] = std::max(buf[i * 2 + 0], buf[i * 2 + 1]) + range;
        }
        FillRandom(tensor, 0.0f, 1.0f);
        if (tensor.Count() == 4)
        {
            for (size_t b = 0; b < tensor.Axis(0); ++b)
            {
                if (tensor.Format() == SimdTensorFormatNhwc)
                {
                    for (size_t y = 0; y < tensor.Axis(1); ++y)
                        for (size_t x = 0; x < tensor.Axis(2); ++x)
                            for (size_t c = 0; c < tensor.Axis(3); ++c)
                                tensor.Data({ b, y, x, c })[0] = min[c] + tensor.Data({ b, y, x, c })[0] * (max[c] - min[c]);
                }
                else
                {
                    for (size_t c = 0; c < tensor.Axis(1); ++c)
                        for (size_t y = 0; y < tensor.Axis(2); ++y)
                            for (size_t x = 0; x < tensor.Axis(3); ++x)
                                tensor.Data({ b, c, y, x })[0] = min[c] + tensor.Data({ b, c, y, x })[0] * (max[c] - min[c]);
                }
            }
        }
        else if (tensor.Count() == 3)
        {
            for (size_t b = 0; b < tensor.Axis(0); ++b)
            {
                if (tensor.Format() == SimdTensorFormatNhwc)
                {
                    for (size_t s = 0; s < tensor.Axis(1); ++s)
                        for (size_t c = 0; c < tensor.Axis(2); ++c)
                            tensor.Data({ b, s, c })[0] = min[c] + tensor.Data({ b, s, c })[0] * (max[c] - min[c]);
                }
                else
                {
                    for (size_t c = 0; c < tensor.Axis(1); ++c)
                        for (size_t s = 0; s < tensor.Axis(2); ++s)
                            tensor.Data({ b, c, s })[0] = min[c] + tensor.Data({ b, c, s })[0] * (max[c] - min[c]);
                }
            }
        }
        else
            assert(0);
    }

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

    //-------------------------------------------------------------------------

    template <class Channel> bool Compare(const View & a, const View & b, int differenceMax, bool printError, int errorCountMax, int valueCycle,
        const String & description)
    {
        std::stringstream message;
        int errorCount = 0;
        size_t channelCount = a.ChannelCount();
        size_t width = channelCount*a.width;
        for (size_t row = 0; row < a.height && errorCount < errorCountMax; ++row)
        {
            const Channel * pA = (const Channel*)(a.data + row*a.stride);
            const Channel * pB = (const Channel*)(b.data + row*b.stride);
            if (memcmp(pA, pB, width * sizeof(Channel)) == 0)
                continue;
            for (size_t offset = 0; offset < width; ++offset)
            {
                if (pA[offset] != pB[offset])
                {
                    if (differenceMax > 0)
                    {
                        Channel difference = Simd::Max(pA[offset], pB[offset]) - Simd::Min(pA[offset], pB[offset]);
                        if (valueCycle > 0)
                            difference = Simd::Min<Channel>(difference, valueCycle - difference);
                        if (difference <= differenceMax)
                            continue;
                    }
                    errorCount++;
                    if (printError)
                    {
                        if (errorCount == 1)
                            message << std::endl << "Fail comparison: " << description << std::endl;
                        size_t col = offset / channelCount;
                        message << "Error at [" << col << "," << row << "] : (" << (int64_t)pA[col*channelCount];
                        for (size_t channel = 1; channel < channelCount; ++channel)
                            message << "," << (int64_t)pA[col*channelCount + channel];
                        message << ") != (" << (int64_t)pB[col*channelCount];
                        for (size_t channel = 1; channel < channelCount; ++channel)
                            message << "," << (int64_t)pB[col*channelCount + channel];
                        message << ")." << std::endl;
                    }
                    if (errorCount >= errorCountMax)
                    {
                        if (printError)
                            message << "Stop comparison." << std::endl;
                        break;
                    }
                }
            }
        }
        if (printError && errorCount > 0)
            TEST_LOG_SS(Error, message.str());
        return errorCount == 0;
    }

    bool FullEqual(const View & a, const View & b)
    {
        size_t size = a.PixelSize()*a.width;
        for (size_t row = 0; row < a.height; ++row)
        {
            if (::memcmp(a.data + row*a.stride, b.data + row*b.stride, size))
                return false;
        }
        return true;
    }

    bool Compare(const View & a, const View & b, int differenceMax, bool printError, int errorCountMax, int valueCycle,
        const String & description)
    {
        assert(Simd::Compatible(a, b));

        if (FullEqual(a, b))
        	return true;

        if (a.format == View::Float)
            return Compare<float>(a, b, differenceMax, printError, errorCountMax, valueCycle, description);
        else if (a.format == View::Double)
            return Compare<double>(a, b, differenceMax, printError, errorCountMax, valueCycle, description);
        else
        {
            switch (a.ChannelSize())
            {
            case 1:
                return Compare<uint8_t>(a, b, differenceMax, printError, errorCountMax, valueCycle, description);
            case 2:
                return Compare<int16_t>(a, b, differenceMax, printError, errorCountMax, valueCycle, description);
            case 4:
                return Compare<int32_t>(a, b, differenceMax, printError, errorCountMax, valueCycle, description);
            case 8:
                return Compare<int64_t>(a, b, differenceMax, printError, errorCountMax, valueCycle, description);
            default:
                assert(0);
            }
        }

        return false;
    }

    template <class T> bool Compare(const T * a, const T * b, size_t size, int64_t differenceMax, bool printError, int errorCountMax, const String & description)
    {
        std::stringstream message;
        int errorCount = 0;
        for (size_t i = 0; i < size; ++i)
        {
            if (a[i] != b[i])
            {
                if (differenceMax > 0)
                {
                    int64_t difference = Simd::Max<int64_t>(a[i], b[i]) - Simd::Min<int64_t>(a[i], b[i]);
                    if (difference <= differenceMax)
                        continue;
                }
                errorCount++;
                if (printError)
                {
                    if (errorCount == 1)
                        message << std::endl << "Fail comparison: " << description << std::endl;
                    message << "Error at [" << i << "] : " << a[i] << " != " << b[i] << "." << std::endl;
                }
                if (errorCount > errorCountMax)
                {
                    if (printError)
                        message << "Stop comparison." << std::endl;
                    break;
                }
            }
        }
        if (printError && errorCount > 0)
            TEST_LOG_SS(Error, message.str());
        return errorCount == 0;
    }

    bool Compare(const Histogram a, const Histogram b, int differenceMax, bool printError, int errorCountMax, const String & description)
    {
        return Compare(a, b, Simd::HISTOGRAM_SIZE, differenceMax, printError, errorCountMax, description);
    }

    bool Compare(const Sums & a, const Sums & b, int differenceMax, bool printError, int errorCountMax, const String & description)
    {
        assert(a.size() == b.size());
        return Compare(a.data(), b.data(), a.size(), differenceMax, printError, errorCountMax, description);
    }

    bool Compare(const Sums64 & a, const Sums64 & b, int differenceMax, bool printError, int errorCountMax, const String & description)
    {
        assert(a.size() == b.size());
        return Compare(a.data(), b.data(), a.size(), differenceMax, printError, errorCountMax, description);
    }

    bool Compare(const Rect & a, const Rect & b, bool printError)
    {
        bool result(a == b);
        if (!result && printError)
        {
            TEST_LOG_SS(Error, "Rectangles is not equal: (" << a.left << ", " << a.top << ", " << a.right << ", " << a.bottom << ") != ("
                << b.left << ", " << b.top << ", " << b.right << ", " << b.bottom << ") !");
        }
        return result;
    }

    bool Compare(const float * a, size_t aStride, const float * b, size_t bStride, size_t width, size_t height, float differenceMax, bool printError,
        int errorCountMax, DifferenceType differenceType, const String & description)
    {
        std::stringstream message;
        int errorCount = 0;
        for (size_t row = 0; row < height; ++row)
        {
            for (size_t col = 0; col < width; ++col)
            {
                float absolute = ::fabs(a[col] - b[col]);
                float relative = ::fabs(a[col] - b[col]) / Simd::Max(::fabs(a[col]), ::fabs(b[col]));
                bool error = false;
                switch (differenceType)
                {
                case DifferenceAbsolute: error = absolute > differenceMax; break;
                case DifferenceRelative: error = relative > differenceMax; break;
                case DifferenceBoth: error = absolute > differenceMax && relative > differenceMax; break;
                case DifferenceAny: error = absolute > differenceMax || relative > differenceMax; break;
                }
                if (error)
                {
                    errorCount++;
                    if (printError)
                    {
                        if (errorCount == 1)
                            message << std::endl << "Fail comparison: " << description << std::endl;
                        message << "Error at [";
                        if (height > 1)
                            message << row << ", ";
                        message << col << "] : " << a[col] << " != " << b[col] << ";" 
                            << " (absolute = " << absolute << ", relative = " << relative << ")!" << std::endl;
                    }
                    if (errorCount > errorCountMax)
                    {
                        if (printError)
                            message << "Stop comparison." << std::endl;
                        goto tooMuchErrors;
                    }
                }
            }
            a += aStride;
            b += bStride;
        }
    tooMuchErrors:
        if (printError && errorCount > 0)
            TEST_LOG_SS(Error, message.str());
        return errorCount == 0;
    }

    bool Compare(const Buffer32f & a, const Buffer32f & b, float differenceMax, bool printError, int errorCountMax, DifferenceType differenceType, const String & description)
    {
        assert(a.size() == b.size());
        return Compare(a.data(), 0, b.data(), 0, a.size(), 1, differenceMax, printError, errorCountMax, differenceType, description);
    }

    bool Compare(const Buffer32f & a, const Buffer32f & b, float differenceMax, bool printError,
        int errorCountMax, bool relative, const String & description)
    {
        return Compare(a, b, differenceMax, printError, errorCountMax, relative ? DifferenceRelative : DifferenceAbsolute, description);
    }

    bool CompareCycle(const Buffer32f & a, const Buffer32f & b, size_t cycle, float differenceMax, bool printError, int errorCountMax, const String & description)
    {
        assert(a.size() == b.size() && a.size() % cycle == 0);
        std::stringstream message;
        Buffer32f rds(cycle, 0);
        int errorCount = 0;
        const size_t size = a.size() / cycle;
        for (size_t i = 0; i < size && errorCount <= errorCountMax; ++i)
        {
            const float * pa = a.data() + i*cycle;
            const float * pb = b.data() + i*cycle;
            float ds = 0, ns = 0;
            for (size_t c = 0; c < cycle; ++c)
            {
                float diff = pb[c] - pa[c];
                float norm = Simd::Max(::fabs(pa[c]), ::fabs(pb[c]));
                rds[c] = ::fabs(diff) / norm;
                ds += diff;
                ns += norm;
            }
            float rdn = float(::fabs(ds)*sqrt(cycle) / ns);
            if (rdn > differenceMax)
            {
                for (size_t c = 0; c < cycle && errorCount <= errorCountMax; ++c)
                {
                    if (rds[c] >= differenceMax)
                    {
                        errorCount++;
                        if (printError)
                        {
                            if (errorCount == 1)
                                message << std::endl << "Fail comparison: " << description << std::endl;
                            message << "Error at [" << i << ", " << c << "] : " << pa[c] << " != " << pb[c] << "; (relative difference = " << rds[c] << ")!" << std::endl;
                        }
                    }
                }
            }
        }
        if (printError && errorCount > 0)
            TEST_LOG_SS(Error, message.str());
        return errorCount == 0;
    }

    bool Compare(const View & a, const View & b, float differenceMax, bool printError, int errorCountMax, DifferenceType differenceType, const String & description)
    {
        assert(Simd::EqualSize(a, b) && a.format == View::Float);
        return Compare((float*)a.data, a.stride / 4, (float*)b.data, b.stride / 4, a.width, a.height, differenceMax, printError, errorCountMax, differenceType, description);
    }

    bool Compare(const View & a, const View & b, float differenceMax, bool printError,
        int errorCountMax, bool relative, const String & description)
    {
        return Compare(a, b, differenceMax, printError, errorCountMax, relative ? DifferenceRelative : DifferenceAbsolute, description);
    }

    bool Compare(const float & a, const float & b, float differenceMax, bool printError, DifferenceType differenceType, const String & description)
    {
        return Compare(&a, 0, &b, 0, 1, 1, differenceMax, printError, 0, differenceType, description);
    }

    String ColorDescription(View::Format format)
    {
        std::stringstream ss;
        ss << "[" << View::PixelSize(format) << "]";
        return ss.str();
    }

    String FormatDescription(View::Format format)
    {
        switch (format)
        {
        case View::None:      return "None";
        case View::Gray8:     return "8-bit Gray";
        case View::Uv16:      return "16-bit UV";
        case View::Bgr24:     return "24-bit BGR";
        case View::Bgra32:    return "32-bit BGRA";
        case View::Int16:     return "16-bit int";
        case View::Int32:     return "32-bit int";
        case View::Int64:     return "64-bit int";
        case View::Float:     return "32-bit float";
        case View::Double:    return "64-bit float";
        case View::BayerGrbg: return "Bayer GRBG";
        case View::BayerGbrg: return "Bayer GBRG";
        case View::BayerRggb: return "Bayer RGGB";
        case View::BayerBggr: return "Bayer BGGR";
        default: assert(0); return "";
        }
    }

    String ScaleDescription(const Point & scale)
    {
        std::stringstream ss;
        ss << "[" << scale.x << "x" << scale.y << "]";
        return ss.str();
    }

    String CompareTypeDescription(SimdCompareType type)
    {
        switch (type)
        {
        case SimdCompareEqual:
            return "[==]";
        case SimdCompareNotEqual:
            return "[!=]";
        case SimdCompareGreater:
            return "[> ]";
        case SimdCompareGreaterOrEqual:
            return "[>=]";
        case SimdCompareLesser:
            return "[< ]";
        case SimdCompareLesserOrEqual:
            return "[<=]";
        }
        assert(0);
        return "[Unknown]";
    }

    String ExpandToLeft(const String & value, size_t count)
    {
        assert(count >= value.size());
        std::stringstream ss;
        for (size_t i = value.size(); i < count; i++)
            ss << " ";
        ss << value;
        return ss.str();
    }

    String ExpandToRight(const String & value, size_t count)
    {
        assert(count >= value.size());
        std::stringstream ss;
        ss << value;
        for (size_t i = value.size(); i < count; i++)
            ss << " ";
        return ss.str();
    }

    String ToString(double value, size_t iCount, size_t fCount)
    {
        assert(iCount > 0);
        if (value > 0)
        {
            std::stringstream ss;
            ss << std::setprecision(fCount) << std::fixed << value;
            return ExpandToLeft(ss.str(), iCount + fCount + (fCount > 0 ? 1 : 0));
        }
        else
        {
            return ExpandToLeft("", iCount + fCount + 1);
        }
    }

    bool DirectoryExists(const String & path)
    {
#if defined(WIN32)
        DWORD fileAttribute = GetFileAttributes(path.c_str());
        return ((fileAttribute != INVALID_FILE_ATTRIBUTES) &&
            (fileAttribute & FILE_ATTRIBUTE_DIRECTORY) != 0);
#elif defined(__linux__)
        DIR * dir = opendir(path.c_str());
        if (dir != NULL)
        {
            ::closedir(dir);
            return true;
        }
        else
            return false;
#else
        return false;
#endif
    }

    String DirectoryByPath(const String & path)
    {
#ifdef WIN32
        String sep("\\");
#else
        String sep("/");
#endif
        size_t pos = path.find_last_of(sep);
        if (pos == std::string::npos)
            return path;
        else
            return path.substr(0, pos);
    }

    bool CreatePath(const String & path)
    {
#ifdef WIN32
        return std::system((String("mkdir ") + path).c_str()) == 0;
#else
        return std::system((String("mkdir -p ") + path).c_str()) == 0;
#endif
    }

    bool CreatePathIfNotExist(const String & path)
    {
        String directory = DirectoryByPath(path);
        if (!DirectoryExists(directory))
        {
            if (!CreatePath(directory))
            {
                TEST_LOG_SS(Info, "Can't create directory '" << directory << "' !");
                return false;
            }
        }
        return true;
    }
}
