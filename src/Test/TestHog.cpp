/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestRandom.h"
#include "Test/TestOptions.h"

#if defined(_MSC_VER)
#pragma warning (push)
#pragma warning (disable: 4996)
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace Test
{
    namespace
    {
        struct FuncHSF
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t width, size_t height, const float * rowFilter, size_t rowSize, const float * colFilter, size_t colSize, float * dst, size_t dstStride, int add);

            FuncPtr func;
            String description;

            FuncHSF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            FuncHSF(const FuncHSF & f, int add) : func(f.func), description(f.description + (add ? "[1]" : "[0]")) {}

            void Call(const View & src, const Buffer32f & row, const Buffer32f & col, const View & dstSrc, View & dstDst, int add) const
            {
                Simd::Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / 4, src.width, src.height, row.data(), row.size(), col.data(), col.size(), (float*)dstDst.data, dstDst.stride / 4, add);
            }
        };
    }

#define FUNC_HSF(function) FuncHSF(function, #function)

    bool HogFilterSeparableAutoTest(int width, int height, int rowSize, int colSize, int add, const FuncHSF & f1, const FuncHSF & f2)
    {
        bool result = true;

        colSize = std::min(colSize, height - 1);
        rowSize = std::min(rowSize, width - 1);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Float, NULL, TEST_ALIGN(width));
        FillRandom32f(src, -10.0f, 10.0f);

        Buffer32f col(colSize), row(rowSize);
        FillRandom(col, -1.0f, 1.0f);
        FillRandom(row, -1.0f, 1.0f);

        View dstSrc(width - rowSize + 1, height - colSize + 1, View::Float, NULL, TEST_ALIGN(width));
        FillRandom32f(dstSrc, -10.0f, 10.0f);
        View dstDst1(width - rowSize + 1, height - colSize + 1, View::Float, NULL, TEST_ALIGN(width));
        View dstDst2(width - rowSize + 1, height - colSize + 1, View::Float, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, row, col, dstSrc, dstDst1, add));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, row, col, dstSrc, dstDst2, add));

        result = result && Compare(dstDst1, dstDst2, EPS, true, 64, false);

        return result;
    }

    bool HogFilterSeparableAutoTest(const FuncHSF & f1, const FuncHSF & f2)
    {
        bool result = true;

        int w = (int)Simd::AlignHi(W / 4, SIMD_ALIGN), h = H / 4;

        for (int add = 0; result && add < 2; ++add)
        {
            result = result && HogFilterSeparableAutoTest(w, h, 10, 10, add, FuncHSF(f1, add), FuncHSF(f2, add));
            result = result && HogFilterSeparableAutoTest(w + 1, h - 1, 11, 9, add, FuncHSF(f1, add), FuncHSF(f2, add));
        }

        return result;
    }

    bool HogFilterSeparableAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && HogFilterSeparableAutoTest(FUNC_HSF(Simd::Base::HogFilterSeparable), FUNC_HSF(SimdHogFilterSeparable));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && HogFilterSeparableAutoTest(FUNC_HSF(Simd::Sse41::HogFilterSeparable), FUNC_HSF(SimdHogFilterSeparable));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && HogFilterSeparableAutoTest(FUNC_HSF(Simd::Avx2::HogFilterSeparable), FUNC_HSF(SimdHogFilterSeparable));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && HogFilterSeparableAutoTest(FUNC_HSF(Simd::Avx512bw::HogFilterSeparable), FUNC_HSF(SimdHogFilterSeparable));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options))
            result = result && HogFilterSeparableAutoTest(FUNC_HSF(Simd::Neon::HogFilterSeparable), FUNC_HSF(SimdHogFilterSeparable));
#endif 

#ifdef SIMD_SVE2_ENABLE
        if (Simd::Sve2::Enable && TestSve2(options))
            result = result && HogFilterSeparableAutoTest(FUNC_HSF(Simd::Sve2::HogFilterSeparable), FUNC_HSF(SimdHogFilterSeparable));
#endif 

        return result;
    }
}

#if defined(_MSC_VER)
#pragma warning (pop)
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
