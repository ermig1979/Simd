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

#include <bitset>

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

    template <class T> String Print(const T & a, const T & b)
    {
        std::stringstream ss;
        ss << a << " != " << b << ".";
        return ss.str();
    }

    template <> String Print<uint8_t>(const uint8_t& a, const uint8_t& b)
    {
        std::stringstream ss;
        ss << a << " != " << b << " | ";
        ss << std::bitset<8>(a) << " != " << std::bitset<8>(b) << " | ";
        ss << (int)a << " != " << (int)b << " | ";
        return ss.str();
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
                    message << "Error at [" << i << "] : " << Print(a[i], b[i]) << std::endl;
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

    bool Compare(const uint8_t* data1, size_t size1, const uint8_t* data2, size_t size2, int differenceMax,
        bool printError, int errorCountMax, const String& description)
    {
        if (size1 != size2)
        {
            if (printError)
            {
                std::stringstream message;
                message << std::endl << "Fail comparison: " << description << std::endl;
                message << "There are different sizes: " << size1 << " != " << size2 << "." << std::endl;
                TEST_LOG_SS(Error, message.str());
            }
            return false;
        }
        return Compare(data1, data2, size2, differenceMax, printError, errorCountMax, description);
    }
}
