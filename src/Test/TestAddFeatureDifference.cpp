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
		struct Func
		{
            typedef void (*FuncPtr)(const uchar * value, size_t valueStride, size_t width, size_t height, 
                const uchar * lo, size_t loStride, const uchar * hi, size_t hiStride,
                ushort weight, uchar * difference, size_t differenceStride);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & value, const View & lo, const View & hi, ushort weight, const View & differenceSrc, View & differenceDst) const
			{
				Simd::Copy(differenceSrc, differenceDst);
				TEST_PERFORMANCE_TEST(description);
				func(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride,
                    weight, differenceDst.data, differenceDst.stride);
			}
		};
	}

#define FUNC(function) Func(function, std::string(#function))

	bool AddFeatureDifferenceTest(int width, int height, ushort weight, const Func & f1, const Func & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "] (" << weight/256 << "*256)." << std::endl;

		View value(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(value);
		View lo(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(lo);
		View hi(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(hi);
        View differenceSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(differenceSrc);

		View differenceDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View differenceDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(value, lo, hi, weight, differenceSrc, differenceDst1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(value, lo, hi, weight, differenceSrc, differenceDst2));

		result = result && Compare(differenceDst1, differenceDst2, 0, true, 10, 0);

		return result;
	}

    bool AddFeatureDifferenceTest(int width, int height, const Func & f1, const Func & f2)
    {
        bool result = true;

        const ushort delta = 256*7;
        for(ushort weight = 0; weight < 4 && result; ++weight)
            result = result &&  AddFeatureDifferenceTest(width, height, weight*delta, f1, f2);

        return result;
    }

    bool AddFeatureDifferenceTest()
    {
        bool result = true;

        result = result && AddFeatureDifferenceTest(W, H, FUNC(Simd::Base::AddFeatureDifference), FUNC(Simd::AddFeatureDifference));
        result = result && AddFeatureDifferenceTest(W + 1, H - 1, FUNC(Simd::Base::AddFeatureDifference), FUNC(Simd::AddFeatureDifference));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && AddFeatureDifferenceTest(W, H, FUNC(Simd::Sse2::AddFeatureDifference), FUNC(Simd::Avx2::AddFeatureDifference));
            result = result && AddFeatureDifferenceTest(W + 1, H - 1, FUNC(Simd::Sse2::AddFeatureDifference), FUNC(Simd::Avx2::AddFeatureDifference));
        }
#endif 

        return result;
    }
}
