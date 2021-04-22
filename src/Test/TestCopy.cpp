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
#include "Test/TestUtils.h"
#include "Test/TestPerformance.h"
#include "Test/TestData.h"
#include "Test/TestString.h"

namespace Test
{
    namespace
    {
        struct Func
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, uint8_t * dst, size_t dstStride);

            FuncPtr func;
            String description;

            Func(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, src.PixelSize(), dst.data, dst.stride);
            }
        };
    }

#define FUNC(function) \
    Func(function, std::string(#function))

    bool CopyAutoTest(View::Format format, int width, int height, const Func & f1, const Func & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View s(width, height, format, NULL, TEST_ALIGN(width));
        FillRandom(s);

        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

        result = result && Compare(d1, d2, 0, true, 32);

        return result;
    }

    bool CopyAutoTest(const Func & f1, const Func & f2)
    {
        bool result = true;

        for (View::Format format = View::Gray8; format <= View::BayerBggr; format = View::Format(format + 1))
        {
            if (format == View::Float || format == View::Double)
                continue;

            Func f1c = Func(f1.func, f1.description + ColorDescription(format));
            Func f2c = Func(f2.func, f2.description + ColorDescription(format));

            result = result && CopyAutoTest(format, W, H, f1c, f2c);
            result = result && CopyAutoTest(format, W + O, H - O, f1c, f2c);
            result = result && CopyAutoTest(format, W - O, H + O, f1c, f2c);
        }

        return result;
    }

    bool CopyAutoTest()
    {
        bool result = true;

        result = result && CopyAutoTest(FUNC(Simd::Base::Copy), FUNC(SimdCopy));

        return result;
    }

    namespace
    {
        struct FuncF
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize,
                size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uint8_t * dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, const Rect & frame, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, src.PixelSize(),
                    frame.left, frame.top, frame.right, frame.bottom, dst.data, dst.stride);
            }
        };
    }

#define FUNC_F(function) \
    FuncF(function, std::string(#function))

    bool CopyFrameAutoTest(View::Format format, int width, int height, const FuncF & f1, const FuncF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View s(width, height, format, NULL, TEST_ALIGN(width));
        FillRandom(s);

        Rect frame(width * 1 / 15, height * 2 / 15, width * 11 / 15, height * 12 / 15);

        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));
        Simd::Fill(d1, 0);
        Simd::Fill(d2, 0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, frame, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, frame, d2));

        result = result && Compare(d1, d2, 0, true, 32);

        return result;
    }

    bool CopyFrameAutoTest(const FuncF & f1, const FuncF & f2)
    {
        bool result = true;

        for (View::Format format = View::Gray8; format <= View::BayerBggr; format = View::Format(format + 1))
        {
            if (format == View::Float || format == View::Double)
                continue;

            FuncF f1c = FuncF(f1.func, f1.description + ColorDescription(format));
            FuncF f2c = FuncF(f2.func, f2.description + ColorDescription(format));

            result = result && CopyFrameAutoTest(format, W, H, f1c, f2c);
            result = result && CopyFrameAutoTest(format, W + O, H - O, f1c, f2c);
            result = result && CopyFrameAutoTest(format, W - O, H + O, f1c, f2c);
        }

        return result;
    }

    bool CopyFrameAutoTest()
    {
        bool result = true;

        result = result && CopyFrameAutoTest(FUNC_F(Simd::Base::CopyFrame), FUNC_F(SimdCopyFrame));

        return result;
    }

    //-----------------------------------------------------------------------

    bool CopyDataTest(bool create, View::Format format, int width, int height, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View s(width, height, format, NULL, TEST_ALIGN(width));

        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(s);

            TEST_SAVE(s);

            f.Call(s, d1);

            TEST_SAVE(d1);
        }
        else
        {
            TEST_LOAD(s);

            TEST_LOAD(d1);

            f.Call(s, d2);

            TEST_SAVE(d2);

            result = result && Compare(d1, d2, 0, true, 64);
        }

        return result;
    }

    bool CopyDataTest(bool create)
    {
        bool result = true;

        Func f = FUNC(SimdCopy);

        for (View::Format format = View::Gray8; format <= View::BayerBggr; format = View::Format(format + 1))
        {
            if (format == View::Float || format == View::Double)
                continue;

            result = result && CopyDataTest(create, format, DW, DH, Func(f.func, f.description + Data::Description(format)));
        }

        return result;
    }

    bool CopyFrameDataTest(bool create, View::Format format, int width, int height, const FuncF & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View s(width, height, format, NULL, TEST_ALIGN(width));

        Rect frame(width * 1 / 15, height * 2 / 15, width * 11 / 15, height * 12 / 15);

        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(s);

            TEST_SAVE(s);

            Simd::Fill(d1, 0);

            f.Call(s, frame, d1);

            TEST_SAVE(d1);
        }
        else
        {
            TEST_LOAD(s);

            Simd::Fill(d2, 0);

            TEST_LOAD(d1);

            f.Call(s, frame, d2);

            TEST_SAVE(d2);

            result = result && Compare(d1, d2, 0, true, 64);
        }

        return result;
    }

    bool CopyFrameDataTest(bool create)
    {
        bool result = true;

        FuncF f = FUNC_F(SimdCopyFrame);

        for (View::Format format = View::Gray8; format <= View::BayerBggr; format = View::Format(format + 1))
        {
            if (format == View::Float || format == View::Double)
                continue;

            result = result && CopyFrameDataTest(create, format, DW, DH, FuncF(f.func, f.description + Data::Description(format)));
        }

        return result;
    }
}
