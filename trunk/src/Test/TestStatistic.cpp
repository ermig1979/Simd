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
		struct Func1
		{
			typedef void (*FuncPtr)(const uint8_t *src, size_t stride, size_t width, size_t height, 
				 uint8_t * min, uint8_t * max, uint8_t * average);

			FuncPtr func;
			std::string description;

			Func1(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, uint8_t * min, uint8_t * max, uint8_t * average) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, min, max, average);
			}
		};
	}

#define FUNC1(function) Func1(function, #function)

	bool GetStatisticAutoTest(int width, int height, const Func1 & f1, const Func1 & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(src);

		 uint8_t min1, max1, average1;
		 uint8_t min2, max2, average2;

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, &min1, &max1, &average1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, &min2, &max2, &average2));

        TEST_CHECK_VALUE(min);
        TEST_CHECK_VALUE(max);
        TEST_CHECK_VALUE(average);

		return result;
	}

    bool GetStatisticAutoTest(const Func1 & f1, const Func1 & f2)
    {
        bool result = true;

        result = result && GetStatisticAutoTest(W, H, f1, f2);
        result = result && GetStatisticAutoTest(W + 3, H - 3, f1, f2);
        result = result && GetStatisticAutoTest(W - 3, H + 3, f1, f2);

        return result;
    }

	bool GetStatisticAutoTest()
	{
		bool result = true;

		result = result && GetStatisticAutoTest(FUNC1(Simd::Base::GetStatistic), FUNC1(SimdGetStatistic));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && GetStatisticAutoTest(FUNC1(Simd::Sse2::GetStatistic), FUNC1(SimdGetStatistic));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && GetStatisticAutoTest(FUNC1(Simd::Avx2::GetStatistic), FUNC1(SimdGetStatistic));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && GetStatisticAutoTest(FUNC1(Simd::Vsx::GetStatistic), FUNC1(SimdGetStatistic));
#endif 

		return result;
	}

    namespace
    {
        struct FuncM
        {
            typedef void (*FuncPtr)(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index, 
                uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy);

            FuncPtr func;
            std::string description;

            FuncM(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & mask, uint8_t index, uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(mask.data, mask.stride, mask.width, mask.height, index, area, x, y, xx, xy, yy);
            }
        };
    }

#define FUNC_M(function) FuncM(function, #function)

#define ARGS_M(scale, function1, function2) \
    FuncM(function1.func, function1.description + ScaleDescription(scale)), \
    FuncM(function2.func, function2.description + ScaleDescription(scale)) 

    bool GetMomentsAutoTest(ptrdiff_t width, ptrdiff_t height, const FuncM & f1, const FuncM & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        const uint8_t index = 7;
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

    bool GetMomentsAutoTest(const Point & scale, const FuncM & f1, const FuncM & f2)
    {
        bool result = true;

        result = result && GetMomentsAutoTest(W*scale.x, H*scale.y, ARGS_M(scale, f1, f2));
        result = result && GetMomentsAutoTest(W*scale.x + 1, H*scale.y - 1, ARGS_M(scale, f1, f2));
        result = result && GetMomentsAutoTest(W*scale.x - 1, H*scale.y + 1, ARGS_M(scale, f1, f2));

        return result;
    }

    bool GetMomentsAutoTest(const FuncM & f1, const FuncM & f2)
    {
        bool result = true;

        result = result && GetMomentsAutoTest(Point(1, 1), f1, f2);
#ifdef NDEBUG
        result = result && GetMomentsAutoTest(Point(5, 2), f1, f2);
#else
        result = result && GetMomentsAutoTest(Point(50, 20), f1, f2);
#endif

        return result;
    }

    bool GetMomentsAutoTest()
    {
        bool result = true;

        result = result && GetMomentsAutoTest(FUNC_M(Simd::Base::GetMoments), FUNC_M(SimdGetMoments));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && GetMomentsAutoTest(FUNC_M(Simd::Sse2::GetMoments), FUNC_M(SimdGetMoments));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && GetMomentsAutoTest(FUNC_M(Simd::Avx2::GetMoments), FUNC_M(SimdGetMoments));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && GetMomentsAutoTest(FUNC_M(Simd::Vsx::GetMoments), FUNC_M(SimdGetMoments));
#endif 

        return result;
    }

    namespace
    {
        struct Func3
        {
            typedef void (*FuncPtr)(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums);

            FuncPtr func;
            std::string description;

            Func3(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & src, uint32_t * sums) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, sums);
            }
        };
    }

#define FUNC3(function) Func3(function, #function)

    bool GetSumsAutoTest(int width, int height, const Func3 & f1, const Func3 & f2, bool isRow)
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

    bool GetSumsAutoTest(const Func3 & f1, const Func3 & f2, bool isRow)
    {
        bool result = true;

        result = result && GetSumsAutoTest(W, H, f1, f2, isRow);
        result = result && GetSumsAutoTest(W + 1, H - 1, f1, f2, isRow);
        result = result && GetSumsAutoTest(W - 1, H + 1, f1, f2, isRow);

        return result;
    }

    bool GetRowSumsAutoTest()
    {
        bool result = true;

        result = result && GetSumsAutoTest(W, H, FUNC3(Simd::Base::GetRowSums), FUNC3(SimdGetRowSums), true);
        result = result && GetSumsAutoTest(W + 1, H - 1, FUNC3(Simd::Base::GetRowSums), FUNC3(SimdGetRowSums), true);
        result = result && GetSumsAutoTest(W - 1, H + 1, FUNC3(Simd::Base::GetRowSums), FUNC3(SimdGetRowSums), true);

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && GetSumsAutoTest(W, H, FUNC3(Simd::Sse2::GetRowSums), FUNC3(Simd::Avx2::GetRowSums), true);
            result = result && GetSumsAutoTest(W + 1, H - 1, FUNC3(Simd::Sse2::GetRowSums), FUNC3(Simd::Avx2::GetRowSums), true);
            result = result && GetSumsAutoTest(W - 1, H + 1, FUNC3(Simd::Sse2::GetRowSums), FUNC3(Simd::Avx2::GetRowSums), true);
        }
#endif 

        return result;
    }

    bool GetColSumsAutoTest()
    {
        bool result = true;

        result = result && GetSumsAutoTest(W, H, FUNC3(Simd::Base::GetColSums), FUNC3(SimdGetColSums), false);
        result = result && GetSumsAutoTest(W + 1, H - 1, FUNC3(Simd::Base::GetColSums), FUNC3(SimdGetColSums), false);
        result = result && GetSumsAutoTest(W - 1, H + 1, FUNC3(Simd::Base::GetColSums), FUNC3(SimdGetColSums), false);

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && GetSumsAutoTest(W, H, FUNC3(Simd::Sse2::GetColSums), FUNC3(Simd::Avx2::GetColSums), false);
            result = result && GetSumsAutoTest(W + 1, H - 1, FUNC3(Simd::Sse2::GetColSums), FUNC3(Simd::Avx2::GetColSums), false);
            result = result && GetSumsAutoTest(W - 1, H + 1, FUNC3(Simd::Sse2::GetColSums), FUNC3(Simd::Avx2::GetColSums), false);
        }
#endif 

        return result;
    }

    bool GetAbsDyRowSumsAutoTest()
    {
        bool result = true;

        result = result && GetSumsAutoTest(FUNC3(Simd::Base::GetAbsDyRowSums), FUNC3(SimdGetAbsDyRowSums), true);

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && GetSumsAutoTest(FUNC3(Simd::Sse2::GetAbsDyRowSums), FUNC3(SimdGetAbsDyRowSums), true);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && GetSumsAutoTest(FUNC3(Simd::Avx2::GetAbsDyRowSums), FUNC3(SimdGetAbsDyRowSums), true);
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && GetSumsAutoTest(FUNC3(Simd::Vsx::GetAbsDyRowSums), FUNC3(SimdGetAbsDyRowSums), true);
#endif 

        return result;
    }

    bool GetAbsDxColSumsAutoTest()
    {
        bool result = true;

        result = result && GetSumsAutoTest(FUNC3(Simd::Base::GetAbsDxColSums), FUNC3(SimdGetAbsDxColSums), false);

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && GetSumsAutoTest(FUNC3(Simd::Sse2::GetAbsDxColSums), FUNC3(SimdGetAbsDxColSums), false);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && GetSumsAutoTest(FUNC3(Simd::Avx2::GetAbsDxColSums), FUNC3(SimdGetAbsDxColSums), false);
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && GetSumsAutoTest(FUNC3(Simd::Vsx::GetAbsDxColSums), FUNC3(SimdGetAbsDxColSums), false);
#endif 

        return result;
    }

    namespace
    {
        struct Func4
        {
            typedef void (*FuncPtr)(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum);

            FuncPtr func;
            std::string description;

            Func4(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & src, uint64_t * sum) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, sum);
            }
        };
    }

#define FUNC4(function) Func4(function, #function)

    bool SumAutoTest(int width, int height, const Func4 & f1, const Func4 & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(src);

        uint64_t sum1;
        uint64_t sum2;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, &sum1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, &sum2));

        TEST_CHECK_VALUE(sum);

        return result;
    }

    bool SumAutoTest(const Func4 & f1, const Func4 & f2)
    {
        bool result = true;

        result = result && SumAutoTest(W, H, f1, f2);
        result = result && SumAutoTest(W + 3, H - 3, f1, f2);
        result = result && SumAutoTest(W - 3, H + 3, f1, f2);

        return result;
    }

    bool ValueSumAutoTest()
    {
        bool result = true;

        result = result && SumAutoTest(FUNC4(Simd::Base::ValueSum), FUNC4(SimdValueSum));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && SumAutoTest(FUNC4(Simd::Sse2::ValueSum), FUNC4(SimdValueSum));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && SumAutoTest(FUNC4(Simd::Avx2::ValueSum), FUNC4(SimdValueSum));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && SumAutoTest(FUNC4(Simd::Vsx::ValueSum), FUNC4(SimdValueSum));
#endif 

        return result;
    }

    bool SquareSumAutoTest()
    {
        bool result = true;

        result = result && SumAutoTest(FUNC4(Simd::Base::SquareSum), FUNC4(SimdSquareSum));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && SumAutoTest(FUNC4(Simd::Sse2::SquareSum), FUNC4(SimdSquareSum));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && SumAutoTest(FUNC4(Simd::Avx2::SquareSum), FUNC4(SimdSquareSum));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && SumAutoTest(FUNC4(Simd::Vsx::SquareSum), FUNC4(SimdSquareSum));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    bool GetStatisticDataTest(bool create, int width, int height, const Func1 & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        uint8_t min1, max1, average1;
        uint8_t min2, max2, average2;

        if(create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, &min1, &max1, &average1);

            TEST_SAVE(min1);
            TEST_SAVE(max1);
            TEST_SAVE(average1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(min1);
            TEST_LOAD(max1);
            TEST_LOAD(average1);

            f.Call(src, &min2, &max2, &average2);

            TEST_SAVE(min2);
            TEST_SAVE(max2);
            TEST_SAVE(average2);

            TEST_CHECK_VALUE(min);
            TEST_CHECK_VALUE(max);
            TEST_CHECK_VALUE(average);
        }

        return result;
    }

    bool GetStatisticDataTest(bool create)
    {
        bool result = true;

        result = result && GetStatisticDataTest(create, DW, DH, FUNC1(SimdGetStatistic));

        return result;
    }

    bool GetMomentsDataTest(bool create, int width, int height, const FuncM & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        const uint8_t index = 7;
        View mask(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        uint64_t area1, x1, y1, xx1, xy1, yy1;
        uint64_t area2, x2, y2, xx2, xy2, yy2;

        if(create)
        {
            FillRandomMask(mask, index);

            TEST_SAVE(mask);

            f.Call(mask, index, &area1, &x1, &y1, &xx1, &xy1, &yy1);

            TEST_SAVE(area1);
            TEST_SAVE(x1);
            TEST_SAVE(y1);
            TEST_SAVE(xx1);
            TEST_SAVE(xy1);
            TEST_SAVE(yy1);
        }
        else
        {
            TEST_LOAD(mask);

            TEST_LOAD(area1);
            TEST_LOAD(x1);
            TEST_LOAD(y1);
            TEST_LOAD(xx1);
            TEST_LOAD(xy1);
            TEST_LOAD(yy1);

            f.Call(mask, index, &area2, &x2, &y2, &xx2, &xy2, &yy2);

            TEST_SAVE(area2);
            TEST_SAVE(x2);
            TEST_SAVE(y2);
            TEST_SAVE(xx2);
            TEST_SAVE(xy2);
            TEST_SAVE(yy2);

            TEST_CHECK_VALUE(area);
            TEST_CHECK_VALUE(x);
            TEST_CHECK_VALUE(y);
            TEST_CHECK_VALUE(xx);
            TEST_CHECK_VALUE(xy);
            TEST_CHECK_VALUE(yy);
        }

        return result;
    }

    bool GetMomentsDataTest(bool create)
    {
        bool result = true;

        result = result && GetMomentsDataTest(create, DW, DH, FUNC_M(SimdGetMoments));

        return result;
    }

    bool GetSumsDataTest(bool create, int width, int height, const Func3 & f, bool isRow)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        size_t size = isRow ? height : width;
        Sums sums1(size, 0);
        Sums sums2(size, 0);

        if(create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, sums1.data());

            TEST_SAVE(sums1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(sums1);

            f.Call(src, sums2.data());

            TEST_SAVE(sums2);

            result = result && Compare(sums1, sums2, 0, true, 32);
        }

        return result;
    }

    bool GetAbsDyRowSumsDataTest(bool create)
    {
        bool result = true;

        result = result && GetSumsDataTest(create, DW, DH, FUNC3(SimdGetAbsDyRowSums), true);

        return result;
    }

    bool GetAbsDxColSumsDataTest(bool create)
    {
        bool result = true;

        result = result && GetSumsDataTest(create, DW, DH, FUNC3(SimdGetAbsDxColSums), false);

        return result;
    }

    bool SumDataTest(bool create, int width, int height, const Func4 & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        uint64_t sum1, sum2;

        if(create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, &sum1);

            TEST_SAVE(sum1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(sum1);

            f.Call(src, &sum2);

            TEST_SAVE(sum2);

            TEST_CHECK_VALUE(sum);
        }

        return result;
    }

    bool ValueSumDataTest(bool create)
    {
        bool result = true;

        result = result && SumDataTest(create, DW, DH, FUNC4(SimdValueSum));

        return result;
    }

    bool SquareSumDataTest(bool create)
    {
        bool result = true;

        result = result && SumDataTest(create, DW, DH, FUNC4(SimdSquareSum));

        return result;
    }
}
