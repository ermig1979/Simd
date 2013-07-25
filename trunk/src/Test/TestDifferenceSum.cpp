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

namespace Test
{
	namespace
	{
		struct Func1
		{
			typedef void (*FuncPtr)(const uchar *a, size_t aStride, const uchar *b, size_t bStride,
				size_t width, size_t height, uint64_t * sum);

			FuncPtr func;
			std::string description;

			Func1(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & a, const View & b, uint64_t & sum) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(a.data, a.stride, b.data, b.stride, a.width, a.height, &sum);
			}
		};

		struct Func2
		{
			typedef void (*FuncPtr)(const uchar *a, size_t aStride, const uchar *b, size_t bStride,
				const uchar *mask, size_t maskStride, uchar index, size_t width, size_t height, uint64_t * sum);

			FuncPtr func;
			std::string description;

			Func2(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & a, const View & b, const View & mask, uchar index, uint64_t & sum) const
			{
				TEST_PERFORMANCE_TEST(description + "<m>");
				func(a.data, a.stride, b.data, b.stride, mask.data, mask.stride, index, a.width, a.height, &sum);
			}
		};
	}

#define FUNC1(function) Func1(function, #function)
#define FUNC2(function) Func2(function, #function)

	bool DifferenceTest(int width, int height, const Func1 & f1, const Func1 & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(a);

		View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(b);

		uint64_t s1, s2;

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, s1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, s2));

		if(s1 != s2)
		{
			result = false;
			std::cout << "Error sum: (" << s1 << " != " << s2 << ")! " << std::endl;
		}
		return result;
	}

	bool MaskedDifferenceTest(int width, int height, const Func2 & f1, const Func2 & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(a);

		View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(b);

		View m(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		uchar index = Random(256);
		FillRandomMask(m, index);

		uint64_t s1, s2;

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, m, index, s1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, m, index, s2));

		if(s1 != s2)
		{
			result = false;
			std::cout << "Error sum: (" << s1 << " != " << s2 << ")! " << std::endl;
		}
		return result;
	}

	bool SquareDifferenceSumTest()
	{
		bool result = true;

		result = result && DifferenceTest(W, H, FUNC1(Simd::Base::SquaredDifferenceSum), FUNC1(Simd::SquaredDifferenceSum));
		result = result && DifferenceTest(W + 1, H - 1, FUNC1(Simd::Base::SquaredDifferenceSum), FUNC1(Simd::SquaredDifferenceSum));

		return result;
	}

	bool MaskedSquareDifferenceSumTest()
	{
		bool result = true;

		result = result && MaskedDifferenceTest(W, H, FUNC2(Simd::Base::SquaredDifferenceSum), FUNC2(Simd::SquaredDifferenceSum));
		result = result && MaskedDifferenceTest(W + 1, H - 1, FUNC2(Simd::Base::SquaredDifferenceSum), FUNC2(Simd::SquaredDifferenceSum));

		return result;
	}

	bool AbsDifferenceSumTest()
	{
		bool result = true;

		result = result && DifferenceTest(W, H, FUNC1(Simd::Base::AbsDifferenceSum), FUNC1(Simd::AbsDifferenceSum));
		result = result && DifferenceTest(W + 1, H - 1, FUNC1(Simd::Base::AbsDifferenceSum), FUNC1(Simd::AbsDifferenceSum));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && DifferenceTest(W, H, FUNC1(Simd::Sse2::AbsDifferenceSum), FUNC1(Simd::Avx2::AbsDifferenceSum));
            result = result && DifferenceTest(W + 1, H - 1, FUNC1(Simd::Sse2::AbsDifferenceSum), FUNC1(Simd::Avx2::AbsDifferenceSum));
        }
#endif 

		return result;
	}

	bool MaskedAbsDifferenceSumTest()
	{
		bool result = true;

		result = result && MaskedDifferenceTest(W, H, FUNC2(Simd::Base::AbsDifferenceSum), FUNC2(Simd::AbsDifferenceSum));
		result = result && MaskedDifferenceTest(W + 1, H - 1, FUNC2(Simd::Base::AbsDifferenceSum), FUNC2(Simd::AbsDifferenceSum));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && MaskedDifferenceTest(W, H, FUNC2(Simd::Sse2::AbsDifferenceSum), FUNC2(Simd::Avx2::AbsDifferenceSum));
            result = result && MaskedDifferenceTest(W + 1, H - 1, FUNC2(Simd::Sse2::AbsDifferenceSum), FUNC2(Simd::Avx2::AbsDifferenceSum));
        }
#endif 

		return result;
	}
}
