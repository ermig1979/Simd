/*
* Tests for Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
		struct ColorFunc
		{
			typedef void (*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride);

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
			typedef void (*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

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

#define ARGS_C(format, width, height, function1, function2) \
    format, width, height, \
    ColorFunc(function1.func, function1.description + ColorDescription(format)), \
    ColorFunc(function2.func, function2.description + ColorDescription(format))

#define FUNC_C(function) \
    ColorFunc(function, std::string(#function))

#define FUNC_G(function) \
    GrayFunc(function, std::string(#function))

	bool ColorFilterAutoTest(View::Format format, int width, int height, const ColorFunc & f1, const ColorFunc & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View s(width, height, format, NULL, TEST_ALIGN(width));
		FillRandom(s);

		View d1(width, height, format, NULL, TEST_ALIGN(width));
		View d2(width, height, format, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

		result = result && Compare(d1, d2, 0, true, 32);

		return result;
	}

	bool GrayFilterAutoTest(int width, int height, const GrayFunc & f1, const GrayFunc & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(s);

		View d1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View d2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

		result = result && Compare(d1, d2, 0, true, 32);

		return result;
	}

    bool ColorFilterAutoTest(const ColorFunc & f1, const ColorFunc & f2)
    {
        bool result = true;

        for(View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
        {
            result = result && ColorFilterAutoTest(ARGS_C(format, W, H, f1, f2));
            result = result && ColorFilterAutoTest(ARGS_C(format, W + O, H - O, f1, f2));
            result = result && ColorFilterAutoTest(ARGS_C(format, W - O, H + O, f1, f2));
        }

        return result;
    }

    bool GrayFilterAutoTest(const GrayFunc & f1, const GrayFunc & f2)
    {
        bool result = true;

        result = result && GrayFilterAutoTest(W, H, f1, f2);
        result = result && GrayFilterAutoTest(W + O, H - O, f1, f2);
        result = result && GrayFilterAutoTest(W - O, H + O, f1, f2);

        return result;
    }

    bool MedianFilterRhomb3x3AutoTest()
    {
        bool result = true;

        result = result && ColorFilterAutoTest(FUNC_C(Simd::Base::MedianFilterRhomb3x3), FUNC_C(SimdMedianFilterRhomb3x3));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Sse2::MedianFilterRhomb3x3), FUNC_C(SimdMedianFilterRhomb3x3));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx2::MedianFilterRhomb3x3), FUNC_C(SimdMedianFilterRhomb3x3));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Vsx::MedianFilterRhomb3x3), FUNC_C(SimdMedianFilterRhomb3x3));
#endif 

        return result;
    }

    bool MedianFilterRhomb5x5AutoTest()
    {
        bool result = true;

        result = result && ColorFilterAutoTest(FUNC_C(Simd::Base::MedianFilterRhomb5x5), FUNC_C(SimdMedianFilterRhomb5x5));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Sse2::MedianFilterRhomb5x5), FUNC_C(SimdMedianFilterRhomb5x5));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx2::MedianFilterRhomb5x5), FUNC_C(SimdMedianFilterRhomb5x5));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Vsx::MedianFilterRhomb5x5), FUNC_C(SimdMedianFilterRhomb5x5));
#endif 

        return result;
    }

	bool MedianFilterSquare3x3AutoTest()
	{
		bool result = true;

        result = result && ColorFilterAutoTest(FUNC_C(Simd::Base::MedianFilterSquare3x3), FUNC_C(SimdMedianFilterSquare3x3));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Sse2::MedianFilterSquare3x3), FUNC_C(SimdMedianFilterSquare3x3));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx2::MedianFilterSquare3x3), FUNC_C(SimdMedianFilterSquare3x3));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Vsx::MedianFilterSquare3x3), FUNC_C(SimdMedianFilterSquare3x3));
#endif 

		return result;
	}

	bool MedianFilterSquare5x5AutoTest()
	{
		bool result = true;

        result = result && ColorFilterAutoTest(FUNC_C(Simd::Base::MedianFilterSquare5x5), FUNC_C(SimdMedianFilterSquare5x5));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Sse2::MedianFilterSquare5x5), FUNC_C(SimdMedianFilterSquare5x5));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx2::MedianFilterSquare5x5), FUNC_C(SimdMedianFilterSquare5x5));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Vsx::MedianFilterSquare5x5), FUNC_C(SimdMedianFilterSquare5x5));
#endif

		return result;
	}

	bool GaussianBlur3x3AutoTest()
	{
        bool result = true;

        result = result && ColorFilterAutoTest(FUNC_C(Simd::Base::GaussianBlur3x3), FUNC_C(SimdGaussianBlur3x3));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Sse2::GaussianBlur3x3), FUNC_C(SimdGaussianBlur3x3));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx2::GaussianBlur3x3), FUNC_C(SimdGaussianBlur3x3));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Vsx::GaussianBlur3x3), FUNC_C(SimdGaussianBlur3x3));
#endif

		return result;
	}

	bool AbsGradientSaturatedSumAutoTest()
	{
		bool result = true;

		result = result && GrayFilterAutoTest(FUNC_G(Simd::Base::AbsGradientSaturatedSum), FUNC_G(SimdAbsGradientSaturatedSum));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && GrayFilterAutoTest(FUNC_G(Simd::Sse2::AbsGradientSaturatedSum), FUNC_G(SimdAbsGradientSaturatedSum));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && GrayFilterAutoTest(FUNC_G(Simd::Avx2::AbsGradientSaturatedSum), FUNC_G(SimdAbsGradientSaturatedSum));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && GrayFilterAutoTest(FUNC_G(Simd::Vsx::AbsGradientSaturatedSum), FUNC_G(SimdAbsGradientSaturatedSum));
#endif 

		return result;
	}

    bool LbpEstimateAutoTest()
    {
        bool result = true;

        result = result && GrayFilterAutoTest(FUNC_G(Simd::Base::LbpEstimate), FUNC_G(SimdLbpEstimate));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && GrayFilterAutoTest(FUNC_G(Simd::Sse2::LbpEstimate), FUNC_G(SimdLbpEstimate));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && GrayFilterAutoTest(FUNC_G(Simd::Avx2::LbpEstimate), FUNC_G(SimdLbpEstimate));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && GrayFilterAutoTest(FUNC_G(Simd::Vsx::LbpEstimate), FUNC_G(SimdLbpEstimate));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool GrayFilterDataTest(bool create, int width, int height, const GrayFunc & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View dst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View dst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        if(create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 0, true, 32, 0);
        }

        return result;
    }

    bool AbsGradientSaturatedSumDataTest(bool create)
    {
        bool result = true;

        result = result && GrayFilterDataTest(create, DW, DH, FUNC_G(SimdAbsGradientSaturatedSum));

        return result;
    }

    bool LbpEstimateDataTest(bool create)
    {
        bool result = true;

        result = result && GrayFilterDataTest(create, DW, DH, FUNC_G(SimdLbpEstimate));

        return result;
    }

    bool ColorFilterDataTest(bool create, int width, int height, View::Format format, const ColorFunc & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, format, NULL, TEST_ALIGN(width));

        View dst1(width, height, format, NULL, TEST_ALIGN(width));
        View dst2(width, height, format, NULL, TEST_ALIGN(width));

        if(create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 0, true, 32, 0);
        }

        return result;
    }

    bool ColorFilterDataTest(bool create, int width, int height, const ColorFunc & f)
    {
        bool result = true;

        for(View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
        {
            result = result && ColorFilterDataTest(create, width, height, format, ColorFunc(f.func, f.description + Data::Description(format)));
        }

        return result;
    }

    bool MedianFilterRhomb3x3DataTest(bool create)
    {
        bool result = true;

        result = result && ColorFilterDataTest(create, DW, DH, FUNC_C(SimdMedianFilterRhomb3x3));

        return result;
    }

    bool MedianFilterRhomb5x5DataTest(bool create)
    {
        bool result = true;

        result = result && ColorFilterDataTest(create, DW, DH, FUNC_C(SimdMedianFilterRhomb5x5));

        return result;
    }

    bool MedianFilterSquare3x3DataTest(bool create)
    {
        bool result = true;

        result = result && ColorFilterDataTest(create, DW, DH, FUNC_C(SimdMedianFilterSquare3x3));

        return result;
    }

    bool MedianFilterSquare5x5DataTest(bool create)
    {
        bool result = true;

        result = result && ColorFilterDataTest(create, DW, DH, FUNC_C(SimdMedianFilterSquare5x5));

        return result;
    }

    bool GaussianBlur3x3DataTest(bool create)
    {
        bool result = true;

        result = result && ColorFilterDataTest(create, DW, DH, FUNC_C(SimdGaussianBlur3x3));

        return result;
    }
}
