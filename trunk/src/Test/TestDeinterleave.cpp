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
		struct Func2
		{
			typedef void (*FuncPtr)(
				const uchar * uv, size_t uvStride, size_t width, size_t height,
				uchar * u, size_t uStride, uchar * v, size_t vStride);

			FuncPtr func;
			std::string description;

			Func2(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & uv, View & u, View & v) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(uv.data, uv.stride, uv.width, uv.height, u.data, u.stride, v.data, v.stride);
			}
		};
	}
#define FUNC2(function) Func2(function, #function)

	bool Deinterleave2Test(int width, int height, const Func2 & f1, const Func2 & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description
			<< " [" << width << ", " << height << "]." << std::endl;

		View uv(width, height, View::Uv16, NULL, TEST_ALIGN(width));
		FillRandom(uv);

		View u1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View v1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View u2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View v2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(uv, u1, v1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(uv, u2, v2));

		result = result && Compare(u1, u2, 0, true, 10);
		result = result && Compare(v1, v2, 0, true, 10);

		return result;
	}

	bool DeinterleaveUvTest()
	{
		bool result = true;

		result = result && Deinterleave2Test(W, H, FUNC2(Simd::Base::DeinterleaveUv), FUNC2(Simd::DeinterleaveUv));
		result = result && Deinterleave2Test(W - 1, H + 1, FUNC2(Simd::Base::DeinterleaveUv), FUNC2(Simd::DeinterleaveUv));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && Deinterleave2Test(W, H, FUNC2(Simd::Sse2::DeinterleaveUv), FUNC2(Simd::Avx2::DeinterleaveUv));
            result = result && Deinterleave2Test(W - 1, H + 1, FUNC2(Simd::Sse2::DeinterleaveUv), FUNC2(Simd::Avx2::DeinterleaveUv));
        }
#endif 

		return result;
	}
}
