/*
* Simd Library Tests.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
#include "Test/Test.h"

namespace Test
{
    namespace
    {
        struct FuncS
        {
            typedef void (*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

            FuncPtr func;
            std::string description;

            FuncS(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
            }
        };
    }

#define ARGS_S(function1, function2) \
    FuncS(function1, std::string(#function1)), FuncS(function2, std::string(#function2))

    bool SobelTest(int width, int height, const FuncS & f1, const FuncS & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(s);

        View d1(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Int16, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool SobelTest(const FuncS & f1, const FuncS & f2)
    {
        bool result = true;

        result = result && SobelTest(W, H, f1, f2);
        result = result && SobelTest(W + 1, H - 1, f1, f2);
        result = result && SobelTest(W - 1, H + 1, f1, f2);

        return result;
    }

    bool SobelDxTest()
    {
        bool result = true;

        result = result && SobelTest(ARGS_S(Simd::Base::SobelDx, SimdSobelDx));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
            result = result && SobelTest(ARGS_S(Simd::Sse2::SobelDx, Simd::Avx2::SobelDx));
#endif 

        return result;
    }

    bool SobelDxAbsTest()
    {
        bool result = true;

        result = result && SobelTest(ARGS_S(Simd::Base::SobelDxAbs, SimdSobelDxAbs));

#if defined(SIMD_SSSE3_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Ssse3::Enable && Simd::Avx2::Enable)
            result = result && SobelTest(ARGS_S(Simd::Ssse3::SobelDxAbs, Simd::Avx2::SobelDxAbs));
#endif 

        return result;
    }

    bool SobelDyTest()
    {
        bool result = true;

        result = result && SobelTest(ARGS_S(Simd::Base::SobelDy, SimdSobelDy));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
            result = result && SobelTest(ARGS_S(Simd::Sse2::SobelDy, Simd::Avx2::SobelDy));
#endif 

        return result;
    }

    bool SobelDyAbsTest()
    {
        bool result = true;

        result = result && SobelTest(ARGS_S(Simd::Base::SobelDyAbs, SimdSobelDyAbs));

#if defined(SIMD_SSSE3_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Ssse3::Enable && Simd::Avx2::Enable)
            result = result && SobelTest(ARGS_S(Simd::Ssse3::SobelDyAbs, Simd::Avx2::SobelDyAbs));
#endif 

        return result;
    }

    bool ContourMetricsTest()
    {
        bool result = true;

        result = result && SobelTest(ARGS_S(Simd::Base::ContourMetrics, SimdContourMetrics));

#if defined(SIMD_SSSE3_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Ssse3::Enable && Simd::Avx2::Enable)
            result = result && SobelTest(ARGS_S(Simd::Ssse3::ContourMetrics, Simd::Avx2::ContourMetrics));
#endif 

        return result;
    }

    namespace
    {
        struct FuncM
        {
            typedef void (*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
                const uint8_t * mask, size_t maskStride, uint8_t indexMin, uint8_t * dst, size_t dstStride);

            FuncPtr func;
            std::string description;

            FuncM(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & src, const View & mask, uint8_t indexMin, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, mask.data, mask.stride, indexMin, dst.data, dst.stride);
            }
        };
    }

#define ARGS_M(function1, function2) \
    FuncM(function1, std::string(#function1)), FuncM(function2, std::string(#function2))

    bool ContourMetricsMaskedTest(int width, int height, const FuncM & f1, const FuncM & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(s);

        View m(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(m);

        View d1(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Int16, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, m, 128, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, m, 128, d2));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool ContourMetricsMaskedTest(const FuncM & f1, const FuncM & f2)
    {
        bool result = true;

        result = result && ContourMetricsMaskedTest(W, H, f1, f2);
        result = result && ContourMetricsMaskedTest(W + 1, H - 1, f1, f2);
        result = result && ContourMetricsMaskedTest(W - 1, H + 1, f1, f2);

        return result;
    }

    bool ContourMetricsMaskedTest()
    {
        bool result = true;

        result = result && ContourMetricsMaskedTest(ARGS_M(Simd::Base::ContourMetricsMasked, SimdContourMetricsMasked));

#if defined(SIMD_SSSE3_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Ssse3::Enable && Simd::Avx2::Enable)
            result = result && ContourMetricsMaskedTest(ARGS_M(Simd::Ssse3::ContourMetricsMasked, Simd::Avx2::ContourMetricsMasked));
#endif 

        return result;
    }

    namespace
    {
        struct FuncA
        {
            typedef void (*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
                size_t step, int16_t threshold, uint8_t * dst, size_t dstStride);

            FuncPtr func;
            std::string description;

            FuncA(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & src, size_t step, int16_t threshold, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, step, threshold, dst.data, dst.stride);
            }
        };
    }

#define ARGS_A(function1, function2) \
    FuncA(function1, std::string(#function1)), FuncA(function2, std::string(#function2))

    bool ContourAnchorsTest(int width, int height, const FuncA & f1, const FuncA & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        View s(width, height, View::Int16, NULL, TEST_ALIGN(width));
        FillRandom(s);

        View d1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        Simd::Fill(d1, 0);
        Simd::Fill(d2, 0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, 3, 0, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, 3, 0, d2));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool ContourAnchorsTest(const FuncA & f1, const FuncA & f2)
    {
        bool result = true;

        result = result && ContourAnchorsTest(W, H, f1, f2);
        result = result && ContourAnchorsTest(W + 1, H - 1, f1, f2);
        result = result && ContourAnchorsTest(W - 1, H + 1, f1, f2);

        return result;
    }

    bool ContourAnchorsTest()
    {
        bool result = true;

        result = result && ContourAnchorsTest(ARGS_A(Simd::Base::ContourAnchors, SimdContourAnchors));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
            result = result && ContourAnchorsTest(ARGS_A(Simd::Sse2::ContourAnchors, Simd::Avx2::ContourAnchors));
#endif 

        return result;
    }
}

