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
				TEST_PERFORMANCE_TEST(description + "<m>");
				func(a.data, a.stride, b.data, b.stride, mask.data, mask.stride, index, a.width, a.height, sum);
			}
		};
	}

#define FUNC_S(function) FuncS(function, #function)
#define FUNC_M(function) FuncM(function, #function)

	template <size_t count> bool DifferenceSumsTest(int width, int height, const FuncS & f1, const FuncS & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(a);

		View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(b);

		uint64_t s1[count], s2[count];

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, s1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, s2));

        for(size_t i = 0; i < count; ++i)
        {
		    if(s1[i] != s2[i])
		    {
			    result = false;
			    std::cout << "Error sum[" << i << "] : (" << s1[i] << " != " << s2[i] << ")! " << std::endl;
		    }        
        }

		return result;
	}

	template <size_t count> bool DifferenceSumsMaskedTest(int width, int height, const FuncM & f1, const FuncM & f2)
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

        uint64_t s1[count], s2[count];

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, m, index, s1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, m, index, s2));

        for(size_t i = 0; i < count; ++i)
        {
            if(s1[i] != s2[i])
            {
                result = false;
                std::cout << "Error sum[" << i << "] : (" << s1[i] << " != " << s2[i] << ")! " << std::endl;
            }        
        }

		return result;
	}

	bool SquareDifferenceSumTest()
	{
		bool result = true;

		result = result && DifferenceSumsTest<1>(W, H, FUNC_S(Simd::Base::SquaredDifferenceSum), FUNC_S(SimdSquaredDifferenceSum));
		result = result && DifferenceSumsTest<1>(W + 1, H - 1, FUNC_S(Simd::Base::SquaredDifferenceSum), FUNC_S(SimdSquaredDifferenceSum));
        result = result && DifferenceSumsTest<1>(W - 1, H + 1, FUNC_S(Simd::Base::SquaredDifferenceSum), FUNC_S(SimdSquaredDifferenceSum));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && DifferenceSumsTest<1>(W, H, FUNC_S(Simd::Sse2::SquaredDifferenceSum), FUNC_S(Simd::Avx2::SquaredDifferenceSum));
            result = result && DifferenceSumsTest<1>(W + 1, H - 1, FUNC_S(Simd::Sse2::SquaredDifferenceSum), FUNC_S(Simd::Avx2::SquaredDifferenceSum));
            result = result && DifferenceSumsTest<1>(W - 1, H + 1, FUNC_S(Simd::Sse2::SquaredDifferenceSum), FUNC_S(Simd::Avx2::SquaredDifferenceSum));
        }
#endif 

		return result;
	}

	bool SquareDifferenceSumMaskedTest()
	{
		bool result = true;

		result = result && DifferenceSumsMaskedTest<1>(W, H, FUNC_M(Simd::Base::SquaredDifferenceSum), FUNC_M(SimdSquaredDifferenceSumMasked));
		result = result && DifferenceSumsMaskedTest<1>(W + 1, H - 1, FUNC_M(Simd::Base::SquaredDifferenceSum), FUNC_M(SimdSquaredDifferenceSumMasked));
        result = result && DifferenceSumsMaskedTest<1>(W - 1, H + 1, FUNC_M(Simd::Base::SquaredDifferenceSum), FUNC_M(SimdSquaredDifferenceSumMasked));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && DifferenceSumsMaskedTest<1>(W, H, FUNC_M(Simd::Sse2::SquaredDifferenceSum), FUNC_M(Simd::Avx2::SquaredDifferenceSum));
            result = result && DifferenceSumsMaskedTest<1>(W + 1, H - 1, FUNC_M(Simd::Sse2::SquaredDifferenceSum), FUNC_M(Simd::Avx2::SquaredDifferenceSum));
            result = result && DifferenceSumsMaskedTest<1>(W - 1, H + 1, FUNC_M(Simd::Sse2::SquaredDifferenceSum), FUNC_M(Simd::Avx2::SquaredDifferenceSum));
        }
#endif 

		return result;
	}

	bool AbsDifferenceSumTest()
	{
		bool result = true;

		result = result && DifferenceSumsTest<1>(W, H, FUNC_S(Simd::Base::AbsDifferenceSum), FUNC_S(SimdAbsDifferenceSum));
		result = result && DifferenceSumsTest<1>(W + 1, H - 1, FUNC_S(Simd::Base::AbsDifferenceSum), FUNC_S(SimdAbsDifferenceSum));
        result = result && DifferenceSumsTest<1>(W - 1, H + 1, FUNC_S(Simd::Base::AbsDifferenceSum), FUNC_S(SimdAbsDifferenceSum));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && DifferenceSumsTest<1>(W, H, FUNC_S(Simd::Sse2::AbsDifferenceSum), FUNC_S(Simd::Avx2::AbsDifferenceSum));
            result = result && DifferenceSumsTest<1>(W + 1, H - 1, FUNC_S(Simd::Sse2::AbsDifferenceSum), FUNC_S(Simd::Avx2::AbsDifferenceSum));
            result = result && DifferenceSumsTest<1>(W - 1, H + 1, FUNC_S(Simd::Sse2::AbsDifferenceSum), FUNC_S(Simd::Avx2::AbsDifferenceSum));
        }
#endif 

		return result;
	}

	bool AbsDifferenceSumMaskedTest()
	{
		bool result = true;

		result = result && DifferenceSumsMaskedTest<1>(W, H, FUNC_M(Simd::Base::AbsDifferenceSum), FUNC_M(SimdAbsDifferenceSumMasked));
		result = result && DifferenceSumsMaskedTest<1>(W + 1, H - 1, FUNC_M(Simd::Base::AbsDifferenceSum), FUNC_M(SimdAbsDifferenceSumMasked));
        result = result && DifferenceSumsMaskedTest<1>(W - 1, H + 1, FUNC_M(Simd::Base::AbsDifferenceSum), FUNC_M(SimdAbsDifferenceSumMasked));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && DifferenceSumsMaskedTest<1>(W, H, FUNC_M(Simd::Sse2::AbsDifferenceSum), FUNC_M(Simd::Avx2::AbsDifferenceSum));
            result = result && DifferenceSumsMaskedTest<1>(W + 1, H - 1, FUNC_M(Simd::Sse2::AbsDifferenceSum), FUNC_M(Simd::Avx2::AbsDifferenceSum));
            result = result && DifferenceSumsMaskedTest<1>(W - 1, H + 1, FUNC_M(Simd::Sse2::AbsDifferenceSum), FUNC_M(Simd::Avx2::AbsDifferenceSum));
        }
#endif 

		return result;
	}

    bool AbsDifferenceSums3x3Test()
    {
        bool result = true;

        result = result && DifferenceSumsTest<9>(W, H, FUNC_S(Simd::Base::AbsDifferenceSums3x3), FUNC_S(SimdAbsDifferenceSums3x3));
        result = result && DifferenceSumsTest<9>(W + 1, H - 1, FUNC_S(Simd::Base::AbsDifferenceSums3x3), FUNC_S(SimdAbsDifferenceSums3x3));
        result = result && DifferenceSumsTest<9>(W - 1, H + 1, FUNC_S(Simd::Base::AbsDifferenceSums3x3), FUNC_S(SimdAbsDifferenceSums3x3));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && DifferenceSumsTest<9>(W, H, FUNC_S(Simd::Sse2::AbsDifferenceSums3x3), FUNC_S(Simd::Avx2::AbsDifferenceSums3x3));
            result = result && DifferenceSumsTest<9>(W + 1, H - 1, FUNC_S(Simd::Sse2::AbsDifferenceSums3x3), FUNC_S(Simd::Avx2::AbsDifferenceSums3x3));
            result = result && DifferenceSumsTest<9>(W - 1, H + 1, FUNC_S(Simd::Sse2::AbsDifferenceSums3x3), FUNC_S(Simd::Avx2::AbsDifferenceSums3x3));
        }
#endif 

        return result;
    }

    bool AbsDifferenceSums3x3MaskedTest()
    {
        bool result = true;

        result = result && DifferenceSumsMaskedTest<9>(W, H, FUNC_M(Simd::Base::AbsDifferenceSums3x3), FUNC_M(SimdAbsDifferenceSums3x3Masked));
        result = result && DifferenceSumsMaskedTest<9>(W + 1, H - 1, FUNC_M(Simd::Base::AbsDifferenceSums3x3), FUNC_M(SimdAbsDifferenceSums3x3Masked));
        result = result && DifferenceSumsMaskedTest<9>(W - 1, H + 1, FUNC_M(Simd::Base::AbsDifferenceSums3x3), FUNC_M(SimdAbsDifferenceSums3x3Masked));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && DifferenceSumsMaskedTest<9>(W, H, FUNC_M(Simd::Sse2::AbsDifferenceSums3x3), FUNC_M(Simd::Avx2::AbsDifferenceSums3x3));
            result = result && DifferenceSumsMaskedTest<9>(W + 1, H - 1, FUNC_M(Simd::Sse2::AbsDifferenceSums3x3), FUNC_M(Simd::Avx2::AbsDifferenceSums3x3));
            result = result && DifferenceSumsMaskedTest<9>(W - 1, H + 1, FUNC_M(Simd::Sse2::AbsDifferenceSums3x3), FUNC_M(Simd::Avx2::AbsDifferenceSums3x3));
        }
#endif 

        return result;
    }
}
