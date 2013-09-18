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
		struct ColorFunc
		{
			typedef void (*FuncPtr)(const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride);

			FuncPtr func;
			std::string description;

			ColorFunc(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, View::PixelSize(src.format), dst.data, dst.stride);
			}
		};

		struct GrayFunc
		{
			typedef void (*FuncPtr)(const uchar * src, size_t srcStride, size_t width, size_t height, uchar * dst, size_t dstStride);

			FuncPtr func;
			std::string description;

			GrayFunc(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
			}
		};
	}

#define ARGS_C1(format, width, height, function1, function2) \
    format, width, height, \
    ColorFunc(function1.func, function1.description + ColorDescription(format)), \
    ColorFunc(function2.func, function2.description + ColorDescription(format))

#define ARGS_C2(function1, function2) \
    ColorFunc(function1, std::string(#function1)), ColorFunc(function2, std::string(#function2))

#define ARGS_G(function1, function2) \
    GrayFunc(function1, std::string(#function1)), GrayFunc(function2, std::string(#function2))

	bool ColorFilterTest(View::Format format, int width, int height, const ColorFunc & f1, const ColorFunc & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View s(width, height, format, NULL, TEST_ALIGN(width));
		FillRandom(s);

		View d1(width, height, format, NULL, TEST_ALIGN(width));
		View d2(width, height, format, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

		result = result && Compare(d1, d2, 0, true, 10);

		return result;
	}

	bool GrayFilterTest(int width, int height, const GrayFunc & f1, const GrayFunc & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(s);

		View d1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View d2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

		result = result && Compare(d1, d2, 0, true, 10);

		return result;
	}

    bool ColorFilterTest(const ColorFunc & f1, const ColorFunc & f2)
    {
        bool result = true;

        for(View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
        {
            result = result && ColorFilterTest(ARGS_C1(format, W, H, f1, f2));
            result = result && ColorFilterTest(ARGS_C1(format, W + 1, H - 1, f1, f2));
            result = result && ColorFilterTest(ARGS_C1(format, W - 1, H + 1, f1, f2));
        }

        return result;
    }

	bool MedianFilterSquare3x3Test()
	{
		bool result = true;

        result = result && ColorFilterTest(ARGS_C2(Simd::Base::MedianFilterSquare3x3, Simd::MedianFilterSquare3x3));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
            result = result && ColorFilterTest(ARGS_C2(Simd::Sse2::MedianFilterSquare3x3, Simd::Avx2::MedianFilterSquare3x3));
#endif 

		return result;
	}

	bool MedianFilterSquare5x5Test()
	{
		bool result = true;

        result = result && ColorFilterTest(ARGS_C2(Simd::Base::MedianFilterSquare5x5, Simd::MedianFilterSquare5x5));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
            result = result && ColorFilterTest(ARGS_C2(Simd::Sse2::MedianFilterSquare5x5, Simd::Avx2::MedianFilterSquare5x5));
#endif 

		return result;
	}

	bool GaussianBlur3x3Test()
	{
		bool result = true;

        result = result && ColorFilterTest(ARGS_C2(Simd::Base::GaussianBlur3x3, Simd::GaussianBlur3x3));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
            result = result && ColorFilterTest(ARGS_C2(Simd::Sse2::GaussianBlur3x3, Simd::Avx2::GaussianBlur3x3));
#endif 

		return result;
	}

	bool AbsGradientSaturatedSumTest()
	{
		bool result = true;

		result = result && GrayFilterTest(W, H, ARGS_G(Simd::Base::AbsGradientSaturatedSum, Simd::AbsGradientSaturatedSum));
		result = result && GrayFilterTest(W + 1, H - 1, ARGS_G(Simd::Base::AbsGradientSaturatedSum, Simd::AbsGradientSaturatedSum));
        result = result && GrayFilterTest(W - 1, H + 1, ARGS_G(Simd::Base::AbsGradientSaturatedSum, Simd::AbsGradientSaturatedSum));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && GrayFilterTest(W, H, ARGS_G(Simd::Sse2::AbsGradientSaturatedSum, Simd::Avx2::AbsGradientSaturatedSum));
            result = result && GrayFilterTest(W + 1, H - 1, ARGS_G(Simd::Sse2::AbsGradientSaturatedSum, Simd::Avx2::AbsGradientSaturatedSum));
            result = result && GrayFilterTest(W - 1, H + 1, ARGS_G(Simd::Sse2::AbsGradientSaturatedSum, Simd::Avx2::AbsGradientSaturatedSum));
        }
#endif 

		return result;
	}

    bool LbpEstimateTest()
    {
        bool result = true;

        result = result && GrayFilterTest(W, H, ARGS_G(Simd::Base::LbpEstimate, Simd::LbpEstimate));
        result = result && GrayFilterTest(W + 1, H - 1, ARGS_G(Simd::Base::LbpEstimate, Simd::LbpEstimate));
        result = result && GrayFilterTest(W - 1, H + 1, ARGS_G(Simd::Base::LbpEstimate, Simd::LbpEstimate));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && GrayFilterTest(W, H, ARGS_G(Simd::Sse2::LbpEstimate, Simd::Avx2::LbpEstimate));
            result = result && GrayFilterTest(W + 1, H - 1, ARGS_G(Simd::Sse2::LbpEstimate, Simd::Avx2::LbpEstimate));
            result = result && GrayFilterTest(W - 1, H + 1, ARGS_G(Simd::Sse2::LbpEstimate, Simd::Avx2::LbpEstimate));
        }
#endif 

        return result;
    }
}
