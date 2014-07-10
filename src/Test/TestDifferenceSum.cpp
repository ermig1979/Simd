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
			typedef void (*FuncPtr)(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
				size_t width, size_t height, uint64_t * sum);

			FuncPtr func;
			std::string description;

			FuncS(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & a, const View & b, uint64_t * sum) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(a.data, a.stride, b.data, b.stride, a.width, a.height, sum);
			}
		};

		struct FuncM
		{
			typedef void (*FuncPtr)(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
				const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum);

			FuncPtr func;
			std::string description;

			FuncM(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & a, const View & b, const View & mask, uint8_t index, uint64_t * sum) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(a.data, a.stride, b.data, b.stride, mask.data, mask.stride, index, a.width, a.height, sum);
			}
		};
	}

#define FUNC_S(function) FuncS(function, #function)
#define FUNC_M(function) FuncM(function, #function)

	bool DifferenceSumsAutoTest(int width, int height, const FuncS & f1, const FuncS & f2, int count)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(a);

		View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(b);

        Sums64 s1(count, 0), s2(count, 0);

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, s1.data()));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, s2.data()));

        result = Compare(s1, s2, 0, true, count);

		return result;
	}

    bool DifferenceSumsAutoTest(const FuncS & f1, const FuncS & f2, int count)
    {
        bool result = true;

        result = result && DifferenceSumsAutoTest(W, H, f1, f2, count);
        result = result && DifferenceSumsAutoTest(W + O, H - O, f1, f2, count);
        result = result && DifferenceSumsAutoTest(W - O, H + O, f1, f2, count);

        return result;
    }

	bool DifferenceSumsMaskedAutoTest(int width, int height, const FuncM & f1, const FuncM & f2, int count)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(a);

		View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(b);

		View m(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		 uint8_t index = Random(256);
		FillRandomMask(m, index);

        Sums64 s1(count, 0), s2(count, 0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, m, index, s1.data()));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, m, index, s2.data()));

        result = Compare(s1, s2, 0, true, count);

		return result;
	}

    bool DifferenceSumsMaskedAutoTest(const FuncM & f1, const FuncM & f2, int count)
    {
        bool result = true;

        result = result && DifferenceSumsMaskedAutoTest(W, H, f1, f2, count);
        result = result && DifferenceSumsMaskedAutoTest(W + O, H - O, f1, f2, count);
        result = result && DifferenceSumsMaskedAutoTest(W - O, H + O, f1, f2, count);

        return result;
    }

	bool SquaredDifferenceSumAutoTest()
	{
		bool result = true;

		result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Base::SquaredDifferenceSum), FUNC_S(SimdSquaredDifferenceSum), 1);

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Sse2::SquaredDifferenceSum), FUNC_S(SimdSquaredDifferenceSum), 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Avx2::SquaredDifferenceSum), FUNC_S(SimdSquaredDifferenceSum), 1);
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Vsx::SquaredDifferenceSum), FUNC_S(SimdSquaredDifferenceSum), 1);
#endif 

		return result;
	}

	bool SquaredDifferenceSumMaskedAutoTest()
	{
        bool result = true;

        result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Base::SquaredDifferenceSumMasked), FUNC_M(SimdSquaredDifferenceSumMasked), 1);

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Sse2::SquaredDifferenceSumMasked), FUNC_M(SimdSquaredDifferenceSumMasked), 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Avx2::SquaredDifferenceSumMasked), FUNC_M(SimdSquaredDifferenceSumMasked), 1);
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Vsx::SquaredDifferenceSumMasked), FUNC_M(SimdSquaredDifferenceSumMasked), 1);
#endif 

		return result;
	}

    bool AbsDifferenceSumAutoTest()
    {
        bool result = true;

        result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Base::AbsDifferenceSum), FUNC_S(SimdAbsDifferenceSum), 1);

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Sse2::AbsDifferenceSum), FUNC_S(SimdAbsDifferenceSum), 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Avx2::AbsDifferenceSum), FUNC_S(SimdAbsDifferenceSum), 1);
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Vsx::AbsDifferenceSum), FUNC_S(SimdAbsDifferenceSum), 1);
#endif 

        return result;
    }

    bool AbsDifferenceSumMaskedAutoTest()
    {
        bool result = true;

        result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Base::AbsDifferenceSumMasked), FUNC_M(SimdAbsDifferenceSumMasked), 1);

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Sse2::AbsDifferenceSumMasked), FUNC_M(SimdAbsDifferenceSumMasked), 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Avx2::AbsDifferenceSumMasked), FUNC_M(SimdAbsDifferenceSumMasked), 1);
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Vsx::AbsDifferenceSumMasked), FUNC_M(SimdAbsDifferenceSumMasked), 1);
#endif 

        return result;
    }

    bool AbsDifferenceSums3x3AutoTest()
    {
        bool result = true;

        result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Base::AbsDifferenceSums3x3), FUNC_S(SimdAbsDifferenceSums3x3), 9);

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Sse2::AbsDifferenceSums3x3), FUNC_S(SimdAbsDifferenceSums3x3), 9);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Avx2::AbsDifferenceSums3x3), FUNC_S(SimdAbsDifferenceSums3x3), 9);
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Vsx::AbsDifferenceSums3x3), FUNC_S(SimdAbsDifferenceSums3x3), 9);
#endif 

        return result;
    }

    bool AbsDifferenceSums3x3MaskedAutoTest()
    {
        bool result = true;

        result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Base::AbsDifferenceSums3x3Masked), FUNC_M(SimdAbsDifferenceSums3x3Masked), 9);

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Sse2::AbsDifferenceSums3x3Masked), FUNC_M(SimdAbsDifferenceSums3x3Masked), 9);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Avx2::AbsDifferenceSums3x3Masked), FUNC_M(SimdAbsDifferenceSums3x3Masked), 9);
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Vsx::AbsDifferenceSums3x3Masked), FUNC_M(SimdAbsDifferenceSums3x3Masked), 9);
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool DifferenceSumsDataTest(bool create, int width, int height, const FuncS & f, int count)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        Sums64 s1(count, 0), s2(count, 0);

        if(create)
        {
            FillRandom(a);
            FillRandom(b);

            TEST_SAVE(a);
            TEST_SAVE(b);

            f.Call(a, b, s1.data());

            TEST_SAVE(s1);
        }
        else
        {
            TEST_LOAD(a);
            TEST_LOAD(b);

            TEST_LOAD(s1);

            f.Call(a, b, s2.data());

            TEST_SAVE(s2);

            result = result && Compare(s1, s2, 0, true, count);
        }

        return result;
    }

    bool AbsDifferenceSumDataTest(bool create)
    {
        bool result = true;

        result = result && DifferenceSumsDataTest(create, DW, DH, FUNC_S(SimdAbsDifferenceSum), 1);

        return result;
    }

    bool AbsDifferenceSums3x3DataTest(bool create)
    {
        bool result = true;

        result = result && DifferenceSumsDataTest(create, DW, DH, FUNC_S(SimdAbsDifferenceSums3x3), 9);

        return result;
    }

    bool SquaredDifferenceSumDataTest(bool create)
    {
        bool result = true;

        result = result && DifferenceSumsDataTest(create, DW, DH, FUNC_S(SimdSquaredDifferenceSum), 1);

        return result;
    }

    bool DifferenceSumsMaskedDataTest(bool create, int width, int height, const FuncM & f, int count)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View m(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        Sums64 s1(count, 0), s2(count, 0);

        uint8_t index = 17;

        if(create)
        {
            FillRandom(a);
            FillRandom(b);
            FillRandomMask(m, index);

            TEST_SAVE(a);
            TEST_SAVE(b);
            TEST_SAVE(m);

            f.Call(a, b, m, index, s1.data());

            TEST_SAVE(s1);
        }
        else
        {
            TEST_LOAD(a);
            TEST_LOAD(b);
            TEST_LOAD(m);

            TEST_LOAD(s1);

            f.Call(a, b, m, index, s2.data());

            TEST_SAVE(s2);

            result = result && Compare(s1, s2, 0, true, count);
        }

        return result;
    }

    bool AbsDifferenceSumMaskedDataTest(bool create)
    {
        bool result = true;

        result = result && DifferenceSumsMaskedDataTest(create, DW, DH, FUNC_M(SimdAbsDifferenceSumMasked), 1);

        return result;
    }

    bool AbsDifferenceSums3x3MaskedDataTest(bool create)
    {
        bool result = true;

        result = result && DifferenceSumsMaskedDataTest(create, DW, DH, FUNC_M(SimdAbsDifferenceSums3x3Masked), 9);

        return result;
    }

    bool SquaredDifferenceSumMaskedDataTest(bool create)
    {
        bool result = true;

        result = result && DifferenceSumsMaskedDataTest(create, DW, DH, FUNC_M(SimdSquaredDifferenceSumMasked), 1);

        return result;
    }
}
