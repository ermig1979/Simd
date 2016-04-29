/*
* Tests for Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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
        struct FuncHDH
        {
            typedef void (*FuncPtr)(const uint8_t * src, size_t stride, size_t width, size_t height, 
                size_t cellX, size_t cellY, size_t quantization, float * histograms);

            FuncPtr func;
            String description;

            FuncHDH(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, const Point & cell, size_t quantization, float * histograms) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, cell.x, cell.y, quantization, histograms);
            }
        };       
    }

#define FUNC_HDH(function) FuncHDH(function, #function)

    bool HogDirectionHistogramsAutoTest(const Point & cell, const Point & size, size_t quantization, const FuncHDH & f1, const FuncHDH & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size.x*cell.x << ", " << size.y*cell.y << "].");

        View s(int(size.x*cell.x), int(size.y*cell.y), View::Gray8, NULL, TEST_ALIGN(size.x*cell.x));
        FillRandom(s);

        const size_t _size = quantization*size.x*size.y;
        Buffer32f h1(_size, 0), h2(_size, 0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, cell, quantization, h1.data()));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, cell, quantization, h2.data()));

        result = result && Compare(h1, h2, EPS, true, 32);

        return result;
    }

    bool HogDirectionHistogramsAutoTest(const FuncHDH & f1, const FuncHDH & f2)
    {
        bool result = true;

        Point c(8, 8), s(32, 24);
        size_t q = 18;

        result = result && HogDirectionHistogramsAutoTest(c, s, q, f1, f2);
        result = result && HogDirectionHistogramsAutoTest(c, s + Point(1, 1), q, f1, f2);
        result = result && HogDirectionHistogramsAutoTest(c, s - Point(1, 1), q, f1, f2);

        return result;
    }

    bool HogDirectionHistogramsAutoTest()
    {
        bool result = true;

        result = result && HogDirectionHistogramsAutoTest(FUNC_HDH(Simd::Base::HogDirectionHistograms), FUNC_HDH(SimdHogDirectionHistograms));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && HogDirectionHistogramsAutoTest(FUNC_HDH(Simd::Sse2::HogDirectionHistograms), FUNC_HDH(SimdHogDirectionHistograms));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && HogDirectionHistogramsAutoTest(FUNC_HDH(Simd::Avx2::HogDirectionHistograms), FUNC_HDH(SimdHogDirectionHistograms));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && HogDirectionHistogramsAutoTest(FUNC_HDH(Simd::Vsx::HogDirectionHistograms), FUNC_HDH(SimdHogDirectionHistograms));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && HogDirectionHistogramsAutoTest(FUNC_HDH(Simd::Neon::HogDirectionHistograms), FUNC_HDH(SimdHogDirectionHistograms));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    bool HogDirectionHistogramsDataTest(bool create, int width, int height, const FuncHDH & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        const Point cell(8, 8);
        const size_t quantization = 18;
        const size_t size = quantization*width*height/cell.x/cell.y;
        Buffer32f h1(size, 0), h2(size, 0);

        if(create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, cell, quantization, h1.data());

            TEST_SAVE(h1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(h1);

            f.Call(src, cell, quantization, h2.data());

            TEST_SAVE(h2);

            result = result && Compare(h1, h2, EPS, true, 32);
        }

        return result;
    }

    bool HogDirectionHistogramsDataTest(bool create)
    {
        bool result = true;

        result = result && HogDirectionHistogramsDataTest(create, DW, DH, FUNC_HDH(SimdHogDirectionHistograms));

        return result;
    }
}
