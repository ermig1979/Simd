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
#include "Test/TestPerformance.h"
#include "Test/TestData.h"

namespace Test
{
    namespace
    {
        struct FuncHLEF
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * features, size_t featuresStride);

            FuncPtr func;
            String description;

            FuncHLEF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, float * features, size_t stride) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, features, stride);
            }
        };
    }

#define FUNC_HLEF(function) FuncHLEF(function, #function)

    void FillCircle(View & view)
    {
        assert(view.format == View::Gray8);
        Point c = view.Size() / 2;
        ptrdiff_t r2 = Simd::Square(Simd::Min(view.width, view.height)/4);
        for (size_t y = 0; y < view.height; ++y)
        {
            ptrdiff_t y2 = Simd::Square(y - c.y);
            uint8_t * data = view.data + view.stride*y;
            for (size_t x = 0; x < view.width; ++x)
            {
                ptrdiff_t x2 = Simd::Square(x - c.x);
                data[x] = x2 + y2 < r2 ? 255 : 0;
            }
        }
    }

    bool HogLiteExtractFeaturesAutoTest(size_t cell, size_t size, size_t width, size_t height, const FuncHLEF & f1, const FuncHLEF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(s);

        const size_t stride = size*(width / cell - 2);
        const size_t full = stride*(height / cell - 2);
        Buffer32f h1(full, 0), h2(full, 0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, h1.data(), stride));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, h2.data(), stride));

        result = result && Compare(h1, h2, EPS, true, 64);

        return result;
    }

    bool HogLiteExtractFeaturesAutoTest(size_t cell, size_t size, const FuncHLEF & f1, const FuncHLEF & f2)
    {
        bool result = true;

        result = result && HogLiteExtractFeaturesAutoTest(cell, size, W, H, f1, f2);
        result = result && HogLiteExtractFeaturesAutoTest(cell, size, W + O, H - O, f1, f2);

        return result;
    }

    bool HogLiteExtractFeatures8x8AutoTest()
    {
        bool result = true;

        result = result && HogLiteExtractFeaturesAutoTest(8, 16, FUNC_HLEF(Simd::Base::HogLiteExtractFeatures8x8), FUNC_HLEF(SimdHogLiteExtractFeatures8x8));

        return result;
    }

    //-----------------------------------------------------------------------

    bool HogLiteExtractFeaturesDataTest(bool create, size_t cell, size_t size, int width, int height, const FuncHLEF & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        const size_t stride = size*(width / cell - 2);
        const size_t full = stride*(height / cell - 2);
        Buffer32f h1(full, 0), h2(full, 0);

        if (create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, h1.data(), stride);

            TEST_SAVE(h1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(h1);

            f.Call(src, h2.data(), stride);

            TEST_SAVE(h2);

            result = result && Compare(h1, h2, EPS, true, 64);
        }

        return result;
    }

    bool HogLiteExtractFeatures8x8DataTest(bool create)
    {
        return HogLiteExtractFeaturesDataTest(create, 8, 16, DW, DH, FUNC_HLEF(SimdHogLiteExtractFeatures8x8));
    }
}
