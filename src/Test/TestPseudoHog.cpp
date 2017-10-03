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
        struct FuncPHEH
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * histogram, size_t histogramStride);

            FuncPtr func;
            String description;

            FuncPHEH(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, float * histogram, size_t stride) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, histogram, stride);
            }
        };
    }

#define FUNC_PHEH(function) FuncPHEH(function, #function)

    bool PseudoHogExtractHistogramAutoTest(size_t cell, size_t quantization, size_t width, size_t height, const FuncPHEH & f1, const FuncPHEH & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(s);

        const size_t stride = quantization*(width / cell);
        const size_t size = stride*(height / cell);
        Buffer32f h1(size, 0), h2(size, 0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, h1.data(), stride));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, h2.data(), stride));

        result = result && Compare(h1, h2, EPS, true, 64);

        return result;
    }

    bool PseudoHogExtractHistogramAutoTest(size_t cell, size_t quantization, const FuncPHEH & f1, const FuncPHEH & f2)
    {
        bool result = true;

        result = result && PseudoHogExtractHistogramAutoTest(cell, quantization, W, H, f1, f2);
        result = result && PseudoHogExtractHistogramAutoTest(cell, quantization, W + O, H - O, f1, f2);

        return result;
    }

    bool PseudoHogExtractHistogram8x8x8AutoTest()
    {
        bool result = true;

        result = result && PseudoHogExtractHistogramAutoTest(8, 8, FUNC_PHEH(Simd::Base::PseudoHogExtractHistogram8x8x8), FUNC_PHEH(SimdPseudoHogExtractHistogram8x8x8));

        return result;
    }

    //-----------------------------------------------------------------------

    bool PseudoHogExtractHistogramDataTest(bool create, size_t cell, size_t quantization, int width, int height, const FuncPHEH & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        const size_t stride = quantization*(width / cell);
        const size_t size = stride*(height / cell);
        Buffer32f h1(size, 0), h2(size, 0);

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

    bool PseudoHogExtractHistogram8x8x8DataTest(bool create)
    {
        bool result = true;

        result = result && PseudoHogExtractHistogramDataTest(create, 8, 8, DW, DH, FUNC_PHEH(SimdPseudoHogExtractHistogram8x8x8));

        return result;
    }
}
