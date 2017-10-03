/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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

namespace Test
{
    uint8_t g_rand[UINT16_MAX];
    bool InitRand()
    {
        for (size_t i = 0, n = UINT16_MAX; i < n; ++i)
            g_rand[i] = ::rand();
        return true;
    }
    bool g_inited = InitRand();
    SIMD_INLINE const uint8_t * Rand()
    {
        return g_rand + (::rand()&INT16_MAX);
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
                    memcpy(view.data + offset + col, Rand(), std::min<size_t>(INT16_MAX, width - col));
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
            const uint8_t * rand = Rand();
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
            ptrdiff_t indent = std::abs(row - c.y)*rect.Width() / rect.Height();
            ptrdiff_t left = rect.left + indent;
            ptrdiff_t right = rect.right - indent;
            ptrdiff_t offset = row*mask.stride + left;
            const uint8_t * rand = Rand();
            for (ptrdiff_t col = left; col < right; ++col, ++offset)
                mask.data[offset] = (rand[col] & 1) ? index : 0;
        }
    }

    void FillRandom32f(View & view, float lo, float hi)
    {
        assert(view.format == View::Float);

        for (size_t row = 0; row < view.height; ++row)
        {
            for (size_t col = 0; col < view.width; ++col)
            {
                view.At<float>(col, row) = lo + (hi - lo)*(float)Random();
            }
        }
    }

    void FillRandom32f(Buffer32f & buffer, float lo, float hi)
    {
        for (size_t i = 0; i < buffer.size(); ++i)
            buffer[i] = lo + (hi - lo)*(float)Random();
    }

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
                        message << "Error at [" << col << "," << row << "] : (" << (int)pA[col*channelCount];
                        for (size_t channel = 1; channel < channelCount; ++channel)
                            message << "," << (int)pA[col*channelCount + channel];
                        message << ") != (" << (int)pB[col*channelCount];
                        for (size_t channel = 1; channel < channelCount; ++channel)
                            message << "," << (int)pB[col*channelCount + channel];
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

        //if (FullEqual(a, b))
        //	return true;

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
        int errorCountMax, bool relative, const String & description)
    {
        std::stringstream message;
        int errorCount = 0;
        for (size_t row = 0; row < height; ++row)
        {
            for (size_t col = 0; col < width; ++col)
            {
                float difference = relative ? ::fabs(a[col] - b[col]) / Simd::Max(::fabs(a[col]), ::fabs(b[col])) : ::fabs(a[col] - b[col]);
                if (difference >= differenceMax)
                {
                    errorCount++;
                    if (printError)
                    {
                        if (errorCount == 1)
                            message << std::endl << "Fail comparison: " << description << std::endl;
                        message << "Error at [";
                        if (height > 1)
                            message << row << ", ";
                        message << col << "] : " << a[col] << " != " << b[col] << "; (" << (relative ? "relative" : "absolute") << " difference = " << difference << ")!" << std::endl;
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

    bool Compare(const Buffer32f & a, const Buffer32f & b, float relativeDifferenceMax, bool printError, int errorCountMax, bool relative, const String & description)
    {
        assert(a.size() == b.size());
        return Compare(a.data(), 0, b.data(), 0, a.size(), 1, relativeDifferenceMax, printError, errorCountMax, relative, description);
    }

    bool CompareCycle(const Buffer32f & a, const Buffer32f & b, size_t cycle, float relativeDifferenceMax, bool printError, int errorCountMax, const String & description)
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
            if (rdn > relativeDifferenceMax)
            {
                for (size_t c = 0; c < cycle && errorCount <= errorCountMax; ++c)
                {
                    if (rds[c] >= relativeDifferenceMax)
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

    bool Compare(const View & a, const View & b, float relativeDifferenceMax, bool printError, int errorCountMax, bool relative, const String & description)
    {
        assert(Simd::EqualSize(a, b) && a.format == View::Float);
        return Compare((float*)a.data, a.stride / 4, (float*)b.data, b.stride / 4, a.width, a.height, relativeDifferenceMax, printError, errorCountMax, relative, description);
    }

    bool Compare(const float & a, const float & b, float relativeDifferenceMax, bool printError, const String & description)
    {
        return Compare(&a, 0, &b, 0, 1, 1, relativeDifferenceMax, printError, 0, true, description);
    }

    String ColorDescription(View::Format format)
    {
        std::stringstream ss;
        ss << "<" << View::PixelSize(format) << ">";
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
            return "(==)";
        case SimdCompareNotEqual:
            return "(!=)";
        case SimdCompareGreater:
            return "(> )";
        case SimdCompareGreaterOrEqual:
            return "(>=)";
        case SimdCompareLesser:
            return "(< )";
        case SimdCompareLesserOrEqual:
            return "(<=)";
        }
        assert(0);
        return "(Unknown)";
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
}
