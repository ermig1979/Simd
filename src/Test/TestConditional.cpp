/*
* Simd Library Tests.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
        struct FuncC
        {
            typedef void (*FuncPtr)(const uint8_t * src, size_t stride, size_t width, size_t height, 
                uint8_t value, SimdCompareType compareType, uint32_t * count);

            FuncPtr func;
            std::string description;

            FuncC(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & src, uint8_t value, SimdCompareType compareType, uint32_t & count) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, value, compareType, &count);
            }
        };
    }

#define ARGS_C1(width, height, type, function1, function2) \
    width, height, type, \
    FuncC(function1.func, function1.description + CompareTypeDescription(type)), \
    FuncC(function2.func, function2.description + CompareTypeDescription(type))

#define ARGS_C2(function1, function2) \
    FuncC(function1, std::string(#function1)), FuncC(function2, std::string(#function2))


    bool ConditionalCountTest(int width, int height, SimdCompareType type, const FuncC & f1, const FuncC & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(src);

        uint8_t value = 127;
        uint32_t c1, c2;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, value, type, c1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, value, type, c2));

        TEST_CHECK_VALUE(c);

        return result;
    }

    bool ConditionalCountTest(const FuncC & f1, const FuncC & f2)
    {
        bool result = true;

        for(SimdCompareType type = SimdCompareEqual; type <= SimdCompareLesserOrEqual && result; type = SimdCompareType(type + 1))
        {
            result = result && ConditionalCountTest(ARGS_C1(W, H, type, f1, f2));
            result = result && ConditionalCountTest(ARGS_C1(W + 1, H - 1, type, f1, f2));
            result = result && ConditionalCountTest(ARGS_C1(W - 1, H + 1, type, f1, f2));
        }

        return result;
    }

    bool ConditionalCountTest()
    {
        bool result = true;

        result = result && ConditionalCountTest(ARGS_C2(Simd::Base::ConditionalCount, SimdConditionalCount));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
            result = result && ConditionalCountTest(ARGS_C2(Simd::Avx2::ConditionalCount, Simd::Sse2::ConditionalCount));
#endif 

        return result;
    }

    namespace 
    {
        struct FuncS
        {
            typedef void (*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
                const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum);

            FuncPtr func;
            std::string description;

            FuncS(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & src, const View & mask, uint8_t value, SimdCompareType compareType, uint64_t & sum) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, mask.data, mask.stride, value, compareType, &sum);
            }
        };
    }

#define ARGS_S1(width, height, type, function1, function2) \
    width, height, type, \
    FuncS(function1.func, function1.description + CompareTypeDescription(type)), \
    FuncS(function2.func, function2.description + CompareTypeDescription(type))

#define ARGS_S2(function1, function2) \
    FuncS(function1, std::string(#function1)), FuncS(function2, std::string(#function2))

    bool ConditionalSumTest(int width, int height, SimdCompareType type, const FuncS & f1, const FuncS & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(src);
        View mask(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(mask);

        uint8_t value = 127;
        uint64_t s1, s2;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, mask, value, type, s1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, mask, value, type, s2));

        TEST_CHECK_VALUE(s);

        return result;
    }

    bool ConditionalSumTest(const FuncS & f1, const FuncS & f2)
    {
        bool result = true;

        for(SimdCompareType type = SimdCompareEqual; type <= SimdCompareLesserOrEqual && result; type = SimdCompareType(type + 1))
        {
            result = result && ConditionalSumTest(ARGS_S1(W, H, type, f1, f2));
            result = result && ConditionalSumTest(ARGS_S1(W + 1, H - 1, type, f1, f2));
            result = result && ConditionalSumTest(ARGS_S1(W - 1, H + 1, type, f1, f2));
        }

        return result;
    }

    bool ConditionalSumTest()
    {
        bool result = true;

        result = result && ConditionalSumTest(ARGS_S2(Simd::Base::ConditionalSum, SimdConditionalSum));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
            result = result && ConditionalSumTest(ARGS_S2(Simd::Avx2::ConditionalSum, Simd::Sse2::ConditionalSum));
#endif 

        return result;
    }

    bool ConditionalSquareSumTest()
    {
        bool result = true;

        result = result && ConditionalSumTest(ARGS_S2(Simd::Base::ConditionalSquareSum, SimdConditionalSquareSum));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
            result = result && ConditionalSumTest(ARGS_S2(Simd::Avx2::ConditionalSquareSum, Simd::Sse2::ConditionalSquareSum));
#endif 

        return result;
    }

    bool ConditionalSquareGradientSumTest()
    {
        bool result = true;

        result = result && ConditionalSumTest(ARGS_S2(Simd::Base::ConditionalSquareGradientSum, SimdConditionalSquareGradientSum));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
            result = result && ConditionalSumTest(ARGS_S2(Simd::Avx2::ConditionalSquareGradientSum, Simd::Sse2::ConditionalSquareGradientSum));
#endif 

        return result;
    }
}
