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
#include "Test/TestData.h"
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

#define FUNC_S(function) \
    FuncS(function, std::string(#function))

    bool SobelAutoTest(int width, int height, const FuncS & f1, const FuncS & f2)
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

    bool SobelAutoTest(const FuncS & f1, const FuncS & f2)
    {
        bool result = true;

        result = result && SobelAutoTest(W, H, f1, f2);
        result = result && SobelAutoTest(W + O, H - O, f1, f2);
        result = result && SobelAutoTest(W - O, H + O, f1, f2);

        return result;
    }

    bool SobelDxAutoTest()
    {
        bool result = true;

        result = result && SobelAutoTest(FUNC_S(Simd::Base::SobelDx), FUNC_S(SimdSobelDx));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && SobelAutoTest(FUNC_S(Simd::Sse2::SobelDx), FUNC_S(SimdSobelDx));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && SobelAutoTest(FUNC_S(Simd::Avx2::SobelDx), FUNC_S(SimdSobelDx));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && SobelAutoTest(FUNC_S(Simd::Vsx::SobelDx), FUNC_S(SimdSobelDx));
#endif 

        return result;
    }

    bool SobelDxAbsAutoTest()
    {
        bool result = true;

        result = result && SobelAutoTest(FUNC_S(Simd::Base::SobelDxAbs), FUNC_S(SimdSobelDxAbs));

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && SobelAutoTest(FUNC_S(Simd::Ssse3::SobelDxAbs), FUNC_S(SimdSobelDxAbs));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && SobelAutoTest(FUNC_S(Simd::Avx2::SobelDxAbs), FUNC_S(SimdSobelDxAbs));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && SobelAutoTest(FUNC_S(Simd::Vsx::SobelDxAbs), FUNC_S(SimdSobelDxAbs));
#endif

        return result;
    }

    bool SobelDyAutoTest()
    {
        bool result = true;

        result = result && SobelAutoTest(FUNC_S(Simd::Base::SobelDy), FUNC_S(SimdSobelDy));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && SobelAutoTest(FUNC_S(Simd::Sse2::SobelDy), FUNC_S(SimdSobelDy));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && SobelAutoTest(FUNC_S(Simd::Avx2::SobelDy), FUNC_S(SimdSobelDy));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && SobelAutoTest(FUNC_S(Simd::Vsx::SobelDy), FUNC_S(SimdSobelDy));
#endif

        return result;
    }

    bool SobelDyAbsAutoTest()
    {
        bool result = true;

        result = result && SobelAutoTest(FUNC_S(Simd::Base::SobelDyAbs), FUNC_S(SimdSobelDyAbs));

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && SobelAutoTest(FUNC_S(Simd::Ssse3::SobelDyAbs), FUNC_S(SimdSobelDyAbs));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && SobelAutoTest(FUNC_S(Simd::Avx2::SobelDyAbs), FUNC_S(SimdSobelDyAbs));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && SobelAutoTest(FUNC_S(Simd::Vsx::SobelDyAbs), FUNC_S(SimdSobelDyAbs));
#endif

        return result;
    }

    bool ContourMetricsAutoTest()
    {
        bool result = true;

        result = result && SobelAutoTest(FUNC_S(Simd::Base::ContourMetrics), FUNC_S(SimdContourMetrics));

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && SobelAutoTest(FUNC_S(Simd::Ssse3::ContourMetrics), FUNC_S(SimdContourMetrics));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && SobelAutoTest(FUNC_S(Simd::Avx2::ContourMetrics), FUNC_S(SimdContourMetrics));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && SobelAutoTest(FUNC_S(Simd::Vsx::ContourMetrics), FUNC_S(SimdContourMetrics));
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

#define FUNC_M(function) \
    FuncM(function, std::string(#function))

    bool ContourMetricsMaskedAutoTest(int width, int height, const FuncM & f1, const FuncM & f2)
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

    bool ContourMetricsMaskedAutoTest(const FuncM & f1, const FuncM & f2)
    {
        bool result = true;

        result = result && ContourMetricsMaskedAutoTest(W, H, f1, f2);
        result = result && ContourMetricsMaskedAutoTest(W + O, H - O, f1, f2);
        result = result && ContourMetricsMaskedAutoTest(W - O, H + O, f1, f2);

        return result;
    }

    bool ContourMetricsMaskedAutoTest()
    {
        bool result = true;

        result = result && ContourMetricsMaskedAutoTest(FUNC_M(Simd::Base::ContourMetricsMasked), FUNC_M(SimdContourMetricsMasked));

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && ContourMetricsMaskedAutoTest(FUNC_M(Simd::Ssse3::ContourMetricsMasked), FUNC_M(SimdContourMetricsMasked));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && ContourMetricsMaskedAutoTest(FUNC_M(Simd::Avx2::ContourMetricsMasked), FUNC_M(SimdContourMetricsMasked));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && ContourMetricsMaskedAutoTest(FUNC_M(Simd::Vsx::ContourMetricsMasked), FUNC_M(SimdContourMetricsMasked));
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

#define FUNC_A(function) \
    FuncA(function, std::string(#function))

    bool ContourAnchorsAutoTest(int width, int height, const FuncA & f1, const FuncA & f2)
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

    bool ContourAnchorsAutoTest(const FuncA & f1, const FuncA & f2)
    {
        bool result = true;

        result = result && ContourAnchorsAutoTest(W, H, f1, f2);
        result = result && ContourAnchorsAutoTest(W + O, H - O, f1, f2);
        result = result && ContourAnchorsAutoTest(W - O, H + O, f1, f2);

        return result;
    }

    bool ContourAnchorsAutoTest()
    {
        bool result = true;

        result = result && ContourAnchorsAutoTest(FUNC_A(Simd::Base::ContourAnchors), FUNC_A(SimdContourAnchors));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && ContourAnchorsAutoTest(FUNC_A(Simd::Sse2::ContourAnchors), FUNC_A(SimdContourAnchors));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && ContourAnchorsAutoTest(FUNC_A(Simd::Avx2::ContourAnchors), FUNC_A(SimdContourAnchors));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && ContourAnchorsAutoTest(FUNC_A(Simd::Vsx::ContourAnchors), FUNC_A(SimdContourAnchors));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool SobelDataTest(bool create, int width, int height, const FuncS & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View dst1(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View dst2(width, height, View::Int16, NULL, TEST_ALIGN(width));

        if(create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 0, true, 32, 0);
        }

        return result;
    }

    bool SobelDxDataTest(bool create)
    {
        bool result = true;

        result = result && SobelDataTest(create, DW, DH, FUNC_S(SimdSobelDx));

        return result;
    }

    bool SobelDxAbsDataTest(bool create)
    {
        bool result = true;

        result = result && SobelDataTest(create, DW, DH, FUNC_S(SimdSobelDxAbs));

        return result;
    }

    bool SobelDyDataTest(bool create)
    {
        bool result = true;

        result = result && SobelDataTest(create, DW, DH, FUNC_S(SimdSobelDy));

        return result;
    }

    bool SobelDyAbsDataTest(bool create)
    {
        bool result = true;

        result = result && SobelDataTest(create, DW, DH, FUNC_S(SimdSobelDyAbs));

        return result;
    }

    bool ContourMetricsDataTest(bool create)
    {
        bool result = true;

        result = result && SobelDataTest(create, DW, DH, FUNC_S(SimdContourMetrics));

        return result;
    }

    bool ContourMetricsMaskedDataTest(bool create, int width, int height, const FuncM & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View mask(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View dst1(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View dst2(width, height, View::Int16, NULL, TEST_ALIGN(width));

        if(create)
        {
            FillRandom(src);
            FillRandom(mask);

            TEST_SAVE(src);
            TEST_SAVE(mask);

            f.Call(src, mask, 128, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(mask);

            TEST_LOAD(dst1);

            f.Call(src, mask, 128, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 0, true, 32, 0);
        }

        return result;
    }

    bool ContourMetricsMaskedDataTest(bool create)
    {
        bool result = true;

        result = result && ContourMetricsMaskedDataTest(create, DW, DH, FUNC_M(SimdContourMetricsMasked));

        return result;
    }

    bool ContourAnchorsDataTest(bool create, int width, int height, const FuncA & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View s(width, height, View::Int16, NULL, TEST_ALIGN(width));

        View d1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        if(create)
        {
            FillRandom(s);

            TEST_SAVE(s);

            Simd::Fill(d1, 0);

            f.Call(s, 3, 0, d1);

            TEST_SAVE(d1);
        }
        else
        {
            TEST_LOAD(s);

            TEST_LOAD(d1);

            Simd::Fill(d2, 0);

            f.Call(s, 3, 0, d2);

            TEST_SAVE(d2);

            result = result && Compare(d1, d2, 0, true, 32, 0);
        }

        return result;
    }

    bool ContourAnchorsDataTest(bool create)
    {
        bool result = true;

        result = result && ContourAnchorsDataTest(create, DW, DH, FUNC_A(SimdContourAnchors));

        return result;
    }
}

