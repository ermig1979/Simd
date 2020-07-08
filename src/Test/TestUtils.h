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
#ifndef __TestUtils_h__
#define __TestUtils_h__

#include "Test/TestLog.h"

namespace Test
{
    void FillSequence(View & view);
    void FillPicture(View & view, uint64_t flag = 0x000000000000000F);

    SIMD_INLINE int Random(int range)
    {
        return ((::rand()&INT16_MAX)*range) / INT16_MAX;
    }

    SIMD_INLINE double Random()
    {
        return ((::rand()&INT16_MAX)*1.0) / INT16_MAX;
    }

    template<class T> inline void Fill(T * data, size_t size, T value)
    {
        for (size_t i = 0; i < size; ++i)
            data[i] = value;
    }

    void FillRandom(View & view, uint8_t lo = 0, uint8_t hi = 255);

    void FillRandom2(View & view, uint8_t lo = 0, uint8_t hi = 255, uint8_t step = 1);

    void FillRandomMask(View & view, uint8_t index);

    void FillRhombMask(View & mask, const Rect & rect, uint8_t index);

    void FillRandom32f(View & view, float lo = 0, float hi = 4096.0f);

    void FillRandom(Buffer32f & buffer, float lo = 0, float hi = 4096.0f);

    void FillRandom(float * data, size_t size, float lo = 0, float hi = 4096.0f);

    void FillRandom(Tensor32f & tensor, float lo = -10.0f, float hi = 10.0f);

    void FillRandom(uint8_t * data, size_t size, uint8_t lo = 0, uint8_t hi = 255);

    void FillRandom(Tensor8u& tensor, uint8_t lo = 0, uint8_t hi = 255);

    void FillRandom(Tensor8i& tensor, int8_t lo = -128, int8_t hi = 127);

    void FillRandom(Tensor32i& tensor, int32_t lo, int32_t hi);

    void FillRandom(Tensor32f& tensor, float* min, float* max, size_t channels, int negative, float upper = 1.0f, float range = 0.01f);

    void SetSrc32fTo8u(const Tensor32f& src, const float* min, const float* max, size_t channels, int negative,
        SimdSynetCompatibilityType compatibility, float* shift, float* scale, Tensor8u& dst);

    void SetDstStat(size_t channels, int negative, SimdSynetCompatibilityType compatibility, 
        const Tensor32f& dst, float* min, float* max, float* scale, float* shift);

    bool Compare(const View & a, const View & b,
        int differenceMax = 0, bool printError = false, int errorCountMax = 0, int valueCycle = 0,
        const String & description = "");

    bool Compare(const Histogram a, const Histogram b,
        int differenceMax = 0, bool printError = false, int errorCountMax = 0, const String & description = "");

    bool Compare(const Sums & a, const Sums & b,
        int differenceMax = 0, bool printError = false, int errorCountMax = 0, const String & description = "");

    bool Compare(const Sums64 & a, const Sums64 & b,
        int differenceMax = 0, bool printError = false, int errorCountMax = 0, const String & description = "");

    bool Compare(const Rect & a, const Rect & b, bool printError = false);

    bool Compare(const Buffer32f & a, const Buffer32f & b, float differenceMax,
        bool printError, int errorCountMax, DifferenceType differenceType, const String & description = "");

    bool Compare(const Buffer32f & a, const Buffer32f & b, float differenceMax = EPS,
        bool printError = false, int errorCountMax = 0, bool relative = true, const String & description = "");

    bool CompareCycle(const Buffer32f & a, const Buffer32f & b, size_t cycle, float differenceMax = EPS,
        bool printError = false, int errorCountMax = 0, const String & description = "");

    bool Compare(const View & a, const View & b, float differenceMax, bool printError,
        int errorCountMax, DifferenceType differenceType, const String & description = "");

    bool Compare(const View & a, const View & b, float differenceMax = EPS, bool printError = false,
        int errorCountMax = 0, bool relative = true, const String & description = "");

    bool Compare(const float & a, const float & b, float differenceMax = EPS, bool printError = false, 
        DifferenceType differenceType = DifferenceRelative, const String & description = "");

    String ColorDescription(View::Format format);

    String FormatDescription(View::Format format);

    String ScaleDescription(const Point & scale);

    String CompareTypeDescription(SimdCompareType type);

    String ExpandToLeft(const String & value, size_t count);
    String ExpandToRight(const String & value, size_t count);

    template <class T> SIMD_INLINE String ToString(const T & value)
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }

    template <> SIMD_INLINE String ToString<SimdBool>(const SimdBool & value)
    {
        std::stringstream ss;
        ss << (int)value;
        return ss.str();
    }

    template <> SIMD_INLINE String ToString<View::Format>(const View::Format & value)
    {
        switch (value)
        {
        case View::None:      return "None";
        case View::Gray8:     return "Gray8";
        case View::Uv16:      return "Uv16:";
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
        if(value || zero)
            ss << std::setprecision(precision) << std::fixed << value;
        return ss.str();
    }

    template <class T> SIMD_INLINE T FromString(const String & str)
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
        std::tm * tm = ::localtime(&t);
        std::stringstream ss;
        ss << ToString(tm->tm_year + 1900, 4) << "."
            << ToString(tm->tm_mon + 1, 2) << "."
            << ToString(tm->tm_mday, 2) << " "
            << ToString(tm->tm_hour, 2) << ":"
            << ToString(tm->tm_min, 2) << ":"
            << ToString(tm->tm_sec, 2);
        return ss.str();
    }

    bool DirectoryExists(const String & path);
    String DirectoryByPath(const String & path);
    bool CreatePath(const String & path);
    bool CreatePathIfNotExist(const String & path);
}

#define TEST_CHECK_VALUE(name) \
    if(name##1 != name##2) \
    { \
        TEST_LOG_SS(Error, "Error " << #name << ": (" << name##1  << " != " << name##2 << ")! "); \
        return false; \
    } 

#endif//__TestUtils_h__
