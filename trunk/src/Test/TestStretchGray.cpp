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
		struct Func
		{
			typedef void (*FuncPtr)(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
				uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
			}
		};
	}

#define FUNC(function) Func(function, #function)

	bool StretchGrayTest(int width, int height, const Func & f1, const Func & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		const int stretchedWidth = width*2;
		const int stretchedHeight = height*2;

		View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(s);

		View d1(stretchedWidth, stretchedHeight, View::Gray8, NULL, TEST_ALIGN(stretchedWidth));
		View d2(stretchedWidth, stretchedHeight, View::Gray8, NULL, TEST_ALIGN(stretchedWidth));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

		result = result && Compare(d1, d2, 0, true, 64);

		return result;
	}

	bool StretchGray2x2Test()
	{
		bool result = true;

		result = result && StretchGrayTest(W, H, FUNC(Simd::Base::StretchGray2x2), FUNC(Simd::StretchGray2x2));
		result = result && StretchGrayTest(W - 1, H + 1, FUNC(Simd::Base::StretchGray2x2), FUNC(Simd::StretchGray2x2));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && StretchGrayTest(W, H, FUNC(Simd::Sse2::StretchGray2x2), FUNC(Simd::Avx2::StretchGray2x2));
            result = result && StretchGrayTest(W - 1, H + 1, FUNC(Simd::Sse2::StretchGray2x2), FUNC(Simd::Avx2::StretchGray2x2));
        }
#endif 

		return result;
	}
}