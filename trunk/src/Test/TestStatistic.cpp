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

#define TEST_CHECK_VALUE(name) \
    if(name##1 != name##2) \
    { \
        std::cout << "Error " << #name << ": (" << name##1  << " != " << name##2 << ")! " << std::endl; \
        return false; \
    }

namespace Test
{
	namespace
	{
		struct Func1
		{
			typedef void (*FuncPtr)(const uchar *src, size_t stride, size_t width, size_t height, 
				uchar * min, uchar * max, uchar * average);

			FuncPtr func;
			std::string description;

			Func1(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, uchar * min, uchar * max, uchar * average) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, min, max, average);
			}
		};
	}

#define FUNC1(function) Func1(function, #function)

	bool GetStatisticTest(int width, int height, const Func1 & f1, const Func1 & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(src);

		uchar min1, max1, average1;
		uchar min2, max2, average2;

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, &min1, &max1, &average1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, &min2, &max2, &average2));

        TEST_CHECK_VALUE(min);
        TEST_CHECK_VALUE(max);
        TEST_CHECK_VALUE(average);

		return result;
	}

	bool GetStatisticTest()
	{
		bool result = true;

		result = result && GetStatisticTest(W, H, FUNC1(Simd::Base::GetStatistic), FUNC1(Simd::GetStatistic));
		result = result && GetStatisticTest(W + 1, H - 1, FUNC1(Simd::Base::GetStatistic), FUNC1(Simd::GetStatistic));
        result = result && GetStatisticTest(W - 1, H + 1, FUNC1(Simd::Base::GetStatistic), FUNC1(Simd::GetStatistic));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && GetStatisticTest(W, H, FUNC1(Simd::Sse2::GetStatistic), FUNC1(Simd::Avx2::GetStatistic));
            result = result && GetStatisticTest(W + 1, H - 1, FUNC1(Simd::Sse2::GetStatistic), FUNC1(Simd::Avx2::GetStatistic));
            result = result && GetStatisticTest(W - 1, H + 1, FUNC1(Simd::Sse2::GetStatistic), FUNC1(Simd::Avx2::GetStatistic));
        }
#endif 

		return result;
	}

    namespace
    {
        struct Func2
        {
            typedef void (*FuncPtr)(const uchar * mask, size_t stride, size_t width, size_t height, uchar index, 
                uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy);

            FuncPtr func;
            std::string description;

            Func2(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & mask, uchar index, uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(mask.data, mask.stride, mask.width, mask.height, index, area, x, y, xx, xy, yy);
            }
        };
    }

#define FUNC2(function) Func2(function, #function)

    bool GetMomentsTest(int width, int height, const Func2 & f1, const Func2 & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        const uchar index = 7;
        View mask(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandomMask(mask, index);

        uint64_t area1, x1, y1, xx1, xy1, yy1;
        uint64_t area2, x2, y2, xx2, xy2, yy2;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(mask, index, &area1, &x1, &y1, &xx1, &xy1, &yy1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(mask, index, &area2, &x2, &y2, &xx2, &xy2, &yy2));

        TEST_CHECK_VALUE(area);
        TEST_CHECK_VALUE(x);
        TEST_CHECK_VALUE(y);
        TEST_CHECK_VALUE(xx);
        TEST_CHECK_VALUE(xy);
        TEST_CHECK_VALUE(yy);

        return result;
    }

    bool GetMomentsTest()
    {
        bool result = true;

        result = result && GetMomentsTest(W, H, FUNC2(Simd::Base::GetMoments), FUNC2(Simd::GetMoments));
        result = result && GetMomentsTest(W + 1, H - 1, FUNC2(Simd::Base::GetMoments), FUNC2(Simd::GetMoments));
        result = result && GetMomentsTest(W - 1, H + 1, FUNC2(Simd::Base::GetMoments), FUNC2(Simd::GetMoments));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && GetMomentsTest(W, H, FUNC2(Simd::Sse2::GetMoments), FUNC2(Simd::Avx2::GetMoments));
            result = result && GetMomentsTest(W + 1, H - 1, FUNC2(Simd::Sse2::GetMoments), FUNC2(Simd::Avx2::GetMoments));
            result = result && GetMomentsTest(W - 1, H + 1, FUNC2(Simd::Sse2::GetMoments), FUNC2(Simd::Avx2::GetMoments));
        }
#endif 

        return result;
    }

    namespace
    {
        struct Func3
        {
            typedef void (*FuncPtr)(const uchar * src, size_t stride, size_t width, size_t height, uint * sums);

            FuncPtr func;
            std::string description;

            Func3(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & src, uint * sums) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, sums);
            }
        };
    }

#define FUNC3(function) Func3(function, #function)

    bool GetSumsTest(int width, int height, const Func3 & f1, const Func3 & f2, bool isRow)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(src);

        size_t size = isRow ? height : width;
        Sums sums1(size, 0);
        Sums sums2(size, 0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, sums1.data()));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, sums2.data()));

        result = result && Compare(sums1, sums2, 0, true, 32);

        return result;
    }

    bool GetRowSumsTest()
    {
        bool result = true;

        result = result && GetSumsTest(W, H, FUNC3(Simd::Base::GetRowSums), FUNC3(Simd::GetRowSums), true);
        result = result && GetSumsTest(W + 1, H - 1, FUNC3(Simd::Base::GetRowSums), FUNC3(Simd::GetRowSums), true);
        result = result && GetSumsTest(W - 1, H + 1, FUNC3(Simd::Base::GetRowSums), FUNC3(Simd::GetRowSums), true);

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && GetSumsTest(W, H, FUNC3(Simd::Sse2::GetRowSums), FUNC3(Simd::Avx2::GetRowSums), true);
            result = result && GetSumsTest(W + 1, H - 1, FUNC3(Simd::Sse2::GetRowSums), FUNC3(Simd::Avx2::GetRowSums), true);
            result = result && GetSumsTest(W - 1, H + 1, FUNC3(Simd::Sse2::GetRowSums), FUNC3(Simd::Avx2::GetRowSums), true);
        }
#endif 

        return result;
    }

    bool GetColSumsTest()
    {
        bool result = true;

        result = result && GetSumsTest(W, H, FUNC3(Simd::Base::GetColSums), FUNC3(Simd::GetColSums), false);
        result = result && GetSumsTest(W + 1, H - 1, FUNC3(Simd::Base::GetColSums), FUNC3(Simd::GetColSums), false);
        result = result && GetSumsTest(W - 1, H + 1, FUNC3(Simd::Base::GetColSums), FUNC3(Simd::GetColSums), false);

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && GetSumsTest(W, H, FUNC3(Simd::Sse2::GetColSums), FUNC3(Simd::Avx2::GetColSums), false);
            result = result && GetSumsTest(W + 1, H - 1, FUNC3(Simd::Sse2::GetColSums), FUNC3(Simd::Avx2::GetColSums), false);
            result = result && GetSumsTest(W - 1, H + 1, FUNC3(Simd::Sse2::GetColSums), FUNC3(Simd::Avx2::GetColSums), false);
        }
#endif 

        return result;
    }

    bool GetAbsDyRowSumsTest()
    {
        bool result = true;

        result = result && GetSumsTest(W, H, FUNC3(Simd::Base::GetAbsDyRowSums), FUNC3(Simd::GetAbsDyRowSums), true);
        result = result && GetSumsTest(W + 1, H - 1, FUNC3(Simd::Base::GetAbsDyRowSums), FUNC3(Simd::GetAbsDyRowSums), true);
        result = result && GetSumsTest(W - 1, H + 1, FUNC3(Simd::Base::GetAbsDyRowSums), FUNC3(Simd::GetAbsDyRowSums), true);

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && GetSumsTest(W, H, FUNC3(Simd::Sse2::GetAbsDyRowSums), FUNC3(Simd::Avx2::GetAbsDyRowSums), true);
            result = result && GetSumsTest(W + 1, H - 1, FUNC3(Simd::Sse2::GetAbsDyRowSums), FUNC3(Simd::Avx2::GetAbsDyRowSums), true);
            result = result && GetSumsTest(W - 1, H + 1, FUNC3(Simd::Sse2::GetAbsDyRowSums), FUNC3(Simd::Avx2::GetAbsDyRowSums), true);
        }
#endif 

        return result;
    }

    bool GetAbsDxColSumsTest()
    {
        bool result = true;

        result = result && GetSumsTest(W, H, FUNC3(Simd::Base::GetAbsDxColSums), FUNC3(Simd::GetAbsDxColSums), false);
        result = result && GetSumsTest(W + 1, H - 1, FUNC3(Simd::Base::GetAbsDxColSums), FUNC3(Simd::GetAbsDxColSums), false);
        result = result && GetSumsTest(W - 1, H + 1, FUNC3(Simd::Base::GetAbsDxColSums), FUNC3(Simd::GetAbsDxColSums), false);

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && GetSumsTest(W, H, FUNC3(Simd::Sse2::GetAbsDxColSums), FUNC3(Simd::Avx2::GetAbsDxColSums), false);
            result = result && GetSumsTest(W + 1, H - 1, FUNC3(Simd::Sse2::GetAbsDxColSums), FUNC3(Simd::Avx2::GetAbsDxColSums), false);
            result = result && GetSumsTest(W - 1, H + 1, FUNC3(Simd::Sse2::GetAbsDxColSums), FUNC3(Simd::Avx2::GetAbsDxColSums), false);
        }
#endif 

        return result;
    }

    namespace
    {
        struct Func4
        {
            typedef void (*FuncPtr)(const uchar * src, size_t stride, size_t width, size_t height, 
                uchar value, SimdCompareType compareType, uint * count);

            FuncPtr func;
            std::string description;

            Func4(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & src, uchar value, SimdCompareType compareType, uint * count) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, value, compareType, count);
            }
        };
    }

#define ARGS41(width, height, type, function1, function2) \
    width, height, type, \
    Func4(function1.func, function1.description + CompareTypeDescription(type)), \
    Func4(function2.func, function2.description + CompareTypeDescription(type))

#define ARGS42(function1, function2) \
    Func4(function1, std::string(#function1)), Func4(function2, std::string(#function2))

    bool ConditionalCountTest(int width, int height, SimdCompareType type, const Func4 & f1, const Func4 & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(src);

        uchar value = 127;
        uint c1, c2;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, value, type, &c1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, value, type, &c2));

        TEST_CHECK_VALUE(c);

        return result;
    }

    bool ConditionalCountTest(const Func4 & f1, const Func4 & f2)
    {
        bool result = true;

        for(SimdCompareType type = SimdCompareEqual; type <= SimdCompareLesserOrEqual && result; type = SimdCompareType(type + 1))
        {
            result = result && ConditionalCountTest(ARGS41(W, H, type, f1, f2));
            result = result && ConditionalCountTest(ARGS41(W + 1, H - 1, type, f1, f2));
            result = result && ConditionalCountTest(ARGS41(W - 1, H + 1, type, f1, f2));
        }

        return result;
    }

    bool ConditionalCountTest()
    {
        bool result = true;

        result = result && ConditionalCountTest(ARGS42(Simd::Base::ConditionalCount, Simd::ConditionalCount));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
            result = result && ConditionalCountTest(ARGS42(Simd::Avx2::ConditionalCount, Simd::Sse2::ConditionalCount));
#endif 

        return result;
    }
}
