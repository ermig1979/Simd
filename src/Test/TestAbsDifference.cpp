/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar,
*               2019-2019 Facundo Galan.
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
		struct Func1
		{
			typedef void(*FuncPtr)(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, uint8_t *c, size_t cStride,
				size_t width, size_t height);

			FuncPtr func;
			String description;

			Func1(const FuncPtr & f, const String & d) : func(f), description(d) {}

			void Call(const View & a, const View & b, View & c) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(a.data, a.stride, b.data, b.stride, c.data, c.stride, a.width, a.height);
			}
		};
	}

#define FUNC1(function) \
    Func1(function, std::string(#function))

	bool AbsDifferenceAutoTest(int width, int height, const Func1 & f1, const Func1 & f2, int count)
	{
		bool result = true;

		TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

		View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(a);

		View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(b);

		View c1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View c2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, c1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, c2));

		result = Compare(c1, c2, 0, true, count);

		return result;
	}

	bool AbsDifferenceAutoTest(const Func1 & f1, const Func1 & f2, int count)
	{
		bool result = true;

		result = result && AbsDifferenceAutoTest(W, H, f1, f2, count);
		result = result && AbsDifferenceAutoTest(W + O, H - O, f1, f2, count);

		return result;
	}

	bool AbsDifferenceAutoTest()
	{
		bool result = true;

		result = result && AbsDifferenceAutoTest(FUNC1(Simd::Base::AbsDifference), FUNC1(SimdAbsDifference), 1);

#ifdef SIMD_SSE41_ENABLE
		if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
			result = result && AbsDifferenceAutoTest(FUNC1(Simd::Sse41::AbsDifference), FUNC1(SimdAbsDifference), 1);
#endif

#ifdef SIMD_AVX2_ENABLE
		if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
			result = result && AbsDifferenceAutoTest(FUNC1(Simd::Avx2::AbsDifference), FUNC1(SimdAbsDifference), 1);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
		if (Simd::Avx512bw::Enable)
			result = result && AbsDifferenceAutoTest(FUNC1(Simd::Avx512bw::AbsDifference), FUNC1(SimdAbsDifference), 1);
#endif

#ifdef SIMD_NEON_ENABLE
		if (Simd::Neon::Enable && W >= Simd::Neon::A)
			result = result && AbsDifferenceAutoTest(FUNC1(Simd::Neon::AbsDifference), FUNC1(SimdAbsDifference), 1);
#endif

		return result;
	}

	//-----------------------------------------------------------------------

	bool AbsDifferenceDataTest(bool create, int width, int height, const Func1 & f, int count)
	{
		bool result = true;

		Data data(f.description);

		TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

		View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		View c1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View c2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		if (create)
		{
			FillRandom(a);
			FillRandom(b);

			TEST_SAVE(a);
			TEST_SAVE(b);

			f.Call(a, b, c1);

			TEST_SAVE(c1);
		}
		else
		{
			TEST_LOAD(a);
			TEST_LOAD(b);

			TEST_LOAD(c1);

			f.Call(a, b, c2);

			TEST_SAVE(c2);

			result = result && Compare(c1, c2, 0, true, count);
		}

		return result;
	}

	bool AbsDifferenceDataTest(bool create)
	{
		bool result = true;

		result = result && AbsDifferenceDataTest(create, DW, DH, FUNC1(SimdAbsDifference), 1);

		return result;
	}

}