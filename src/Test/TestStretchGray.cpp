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
        struct Func
        {
            typedef void(*FuncPtr)(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
                uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

            FuncPtr func;
            String description;

            Func(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
            }
        };
    }

#define FUNC(function) Func(function, #function)

    bool StretchGrayAutoTest(int width, int height, const Func & f1, const Func & f2, int stretch)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        const int stretchedWidth = width*stretch;
        const int stretchedHeight = height*stretch;

        View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(s);

        View d1(stretchedWidth, stretchedHeight, View::Gray8, NULL, TEST_ALIGN(stretchedWidth));
        View d2(stretchedWidth, stretchedHeight, View::Gray8, NULL, TEST_ALIGN(stretchedWidth));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool StretchGrayAutoTest(const Func & f1, const Func & f2, int stretch)
    {
        bool result = true;

        result = result && StretchGrayAutoTest(W, H, f1, f2, stretch);
        result = result && StretchGrayAutoTest(W + O, H - O, f1, f2, stretch);
        result = result && StretchGrayAutoTest(W - O, H + O, f1, f2, stretch);

        return result;
    }

    bool StretchGray2x2AutoTest()
    {
        bool result = true;

        result = result && StretchGrayAutoTest(FUNC(Simd::Base::StretchGray2x2), FUNC(SimdStretchGray2x2), 2);

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && StretchGrayAutoTest(FUNC(Simd::Sse2::StretchGray2x2), FUNC(SimdStretchGray2x2), 2);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && StretchGrayAutoTest(FUNC(Simd::Avx2::StretchGray2x2), FUNC(SimdStretchGray2x2), 2);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && StretchGrayAutoTest(FUNC(Simd::Avx512bw::StretchGray2x2), FUNC(SimdStretchGray2x2), 2);
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && StretchGrayAutoTest(FUNC(Simd::Vmx::StretchGray2x2), FUNC(SimdStretchGray2x2), 2);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && StretchGrayAutoTest(FUNC(Simd::Neon::StretchGray2x2), FUNC(SimdStretchGray2x2), 2);
#endif
        return result;
    }

    //-----------------------------------------------------------------------

    bool StretchGrayDataTest(bool create, int width, int height, const Func & f, int stretch)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        const int stretchedWidth = width*stretch;
        const int stretchedHeight = height*stretch;

        View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View d1(stretchedWidth, stretchedHeight, View::Gray8, NULL, TEST_ALIGN(stretchedWidth));
        View d2(stretchedWidth, stretchedHeight, View::Gray8, NULL, TEST_ALIGN(stretchedWidth));

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

    bool StretchGray2x2DataTest(bool create)
    {
        bool result = true;

        result = result && StretchGrayDataTest(create, DW, DH, FUNC(SimdStretchGray2x2), 2);

        return result;
    }
}
