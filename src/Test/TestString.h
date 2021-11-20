/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
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
#ifndef __TestString_h__
#define __TestString_h__

#include "Test/TestConfig.h"

#include <locale>

namespace Test
{
    SIMD_INLINE String ToLower(const String& src)
    {
        std::locale loc;
        String dst(src);
        for (size_t i = 0; i < dst.size(); ++i)
            dst[i] = std::tolower(dst[i], loc);
        return dst;
    }

    String ColorDescription(View::Format format);

    String FormatDescription(View::Format format);

    String ScaleDescription(const Point& scale);

    String CompareTypeDescription(SimdCompareType type);

    String ExpandToLeft(const String& value, size_t count);
    String ExpandToRight(const String& value, size_t count);

    template <class T> SIMD_INLINE String ToString(const T& value)
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }

    template <> SIMD_INLINE String ToString<SimdBool>(const SimdBool& value)
    {
        std::stringstream ss;
        ss << (int)value;
        return ss.str();
    }

    template <> SIMD_INLINE String ToString<View::Format>(const View::Format& value)
    {
        switch (value)
        {
        case View::None:      return "None";
        case View::Gray8:     return "Gray8";
        case View::Uv16:      return "Uv16";
        case View::Bgr24:     return "Bgr24";
        case View::Bgra32:    return "Bgra32";
        case View::Int16:     return "Int16";
        case View::Int32:     return "Int32";
        case View::Int64:     return "Int64";
        case View::Float:     return "Float";
        case View::Double:    return "Double";
        case View::BayerGrbg: return "BayerGrbg";
        case View::BayerGbrg: return "BayerGbrg";
        case View::BayerRggb: return "BayerRggb";
        case View::BayerBggr: return "BayerBggr";
        case View::Hsv24:     return "Hsv24";
        case View::Hsl24:     return "Hsl24";
        case View::Rgb24:     return "Rgb24";
        case View::Rgba32:    return "Rgba32";
        case View::Uyvy16:    return "Uyvy16";
        default: assert(0);  return "";
        }
    }

    template <> SIMD_INLINE String ToString<SimdImageFileType>(const SimdImageFileType& value)
    {
        switch (value)
        {
        case SimdImageFileUndefined:    return "None";
        case SimdImageFilePgmTxt:       return "PgmT";
        case SimdImageFilePgmBin:       return "PgmB";
        case SimdImageFilePpmTxt:       return "PpmT";
        case SimdImageFilePpmBin:       return "PpmB";
        case SimdImageFilePng:          return "Png";
        case SimdImageFileJpeg:         return "Jpeg";
        default: assert(0);  return "";
        }
    }

    SIMD_INLINE String ToString(int value, int width)
    {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(width) << value;
        return ss.str();
    }

    SIMD_INLINE String ToString(double value, int precision, bool zero)
    {
        std::stringstream ss;
        if (value || zero)
            ss << std::setprecision(precision) << std::fixed << value;
        return ss.str();
    }

    template <class T> SIMD_INLINE T FromString(const String& str)
    {
        std::stringstream ss(str);
        T value;
        ss >> value;
        return value;
    }

    String ToString(double value, size_t iCount, size_t fCount);

    SIMD_INLINE String GetCurrentDateTimeString()
    {
        std::time_t t;
        std::time(&t);
        std::tm* tm = ::localtime(&t);
        std::stringstream ss;
        ss << ToString(tm->tm_year + 1900, 4) << "."
            << ToString(tm->tm_mon + 1, 2) << "."
            << ToString(tm->tm_mday, 2) << " "
            << ToString(tm->tm_hour, 2) << ":"
            << ToString(tm->tm_min, 2) << ":"
            << ToString(tm->tm_sec, 2);
        return ss.str();
    }

    SIMD_INLINE String ToExtension(const SimdImageFileType& value)
    {
        switch (value)
        {
        case SimdImageFilePgmTxt: return "pgm";
        case SimdImageFilePgmBin: return "pgm";
        case SimdImageFilePpmTxt: return "ppm";
        case SimdImageFilePpmBin: return "ppm";
        case SimdImageFilePng:    return "png";
        case SimdImageFileJpeg:   return "jpg";
        default: assert(0);  return "";
        }
    }
}

#endif//__TestString_h__
