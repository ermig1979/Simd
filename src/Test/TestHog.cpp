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

        const size_t C = 8;
        Point c(C, C), s(W / C, H / C);
        const size_t q = 18;

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

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && HogDirectionHistogramsAutoTest(FUNC_HDH(Simd::Sse41::HogDirectionHistograms), FUNC_HDH(SimdHogDirectionHistograms));
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

    namespace
    {
        struct FuncHEF
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t stride, size_t width, size_t height, float * features);

            FuncPtr func;
            String description;

            FuncHEF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, float * features) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, features);
            }
        };
    }

#define FUNC_HEF(function) FuncHEF(function, #function)

    bool HogExtractFeaturesAutoTest(int width, int height, const FuncHEF & f1, const FuncHEF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(src);

        const size_t size = (width/8)*(height/8)*31;
        Buffer32f features1(size, 0), features2(size, 0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, features1.data()));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, features2.data()));

        result = result && Compare(features1, features2, EPS, true, 64);

        return result;
    }

    bool HogExtractFeaturesAutoTest(const FuncHEF & f1, const FuncHEF & f2)
    {
        bool result = true;

        result = result && HogExtractFeaturesAutoTest(W, H, f1, f2);
        result = result && HogExtractFeaturesAutoTest(W + 8, H - 8, f1, f2);

        return result;
    }

    bool HogExtractFeaturesAutoTest()
    {
        bool result = true;

        result = result && HogExtractFeaturesAutoTest(FUNC_HEF(Simd::Base::HogExtractFeatures), FUNC_HEF(SimdHogExtractFeatures));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && HogExtractFeaturesAutoTest(FUNC_HEF(Simd::Sse41::HogExtractFeatures), FUNC_HEF(SimdHogExtractFeatures));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && HogExtractFeaturesAutoTest(FUNC_HEF(Simd::Avx2::HogExtractFeatures), FUNC_HEF(SimdHogExtractFeatures));
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

    bool HogExtractFeaturesDataTest(bool create, int width, int height, const FuncHEF & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        const size_t size = 31*(width/8)*(height/8);
        Buffer32f f1(size, 0), f2(size, 0);

        if (create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, f1.data());

            TEST_SAVE(f1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(f1);

            f.Call(src, f2.data());

            TEST_SAVE(f2);

            result = result && Compare(f1, f2, EPS, true, 64);
        }

        return result;
    }

    bool HogExtractFeaturesDataTest(bool create)
    {
        bool result = true;

        result = result && HogExtractFeaturesDataTest(create, DW, DH, FUNC_HEF(SimdHogExtractFeatures));

        return result;
    }
}
