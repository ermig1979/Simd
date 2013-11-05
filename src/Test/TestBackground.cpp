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
		struct Func1
		{
			typedef void (*FuncPtr)(const uint8_t * value, size_t valueStride, size_t width, size_t height,
				 uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

			FuncPtr func;
			std::string description;

			Func1(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & value, const View & loSrc, const View & hiSrc, View & loDst, View & hiDst) const
			{
				Simd::Copy(loSrc, loDst);
				Simd::Copy(hiSrc, hiDst);
				TEST_PERFORMANCE_TEST(description);
				func(value.data, value.stride, value.width, value.height, loDst.data, loDst.stride, hiDst.data, hiDst.stride);
			}
		};
	}

#define FUNC1(function) Func1(function, std::string(#function))

	bool BackgroundChangeRangeTest(int width, int height, const Func1 & f1, const Func1 & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View value(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(value);
		View loSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(loSrc);
		View hiSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(hiSrc);

		View loDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View hiDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View loDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View hiDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(value, loSrc, hiSrc, loDst1, hiDst1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(value, loSrc, hiSrc, loDst2, hiDst2));

		result = result && Compare(loDst1, loDst2, 0, true, 10, 0, "lo");
		result = result && Compare(hiDst1, hiDst2, 0, true, 10, 0, "hi");

		return result;
	}

	namespace
	{
		struct Func2
		{
			typedef void (*FuncPtr)(const uint8_t * value, size_t valueStride, size_t width, size_t height,
				const uint8_t * loValue, size_t loValueStride, const uint8_t * hiValue, size_t hiValueStride,
				 uint8_t * loCount, size_t loCountStride, uint8_t * hiCount, size_t hiCountStride);

			FuncPtr func;
			std::string description;

			Func2(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & value, const View & loValue, const View & hiValue,
				const View & loCountSrc, const View & hiCountSrc, View & loCountDst, View & hiCountDst) const
			{
				Simd::Copy(loCountSrc, loCountDst);
				Simd::Copy(hiCountSrc, hiCountDst);
				TEST_PERFORMANCE_TEST(description);
				func(value.data, value.stride, value.width, value.height, 
					loValue.data, loValue.stride, hiValue.data, hiValue.stride,
					loCountDst.data, loCountDst.stride, hiCountDst.data, hiCountDst.stride);
			}
		};
	}

#define FUNC2(function) Func2(function, std::string(#function))

	bool BackgroundIncrementCountTest(int width, int height, const Func2 & f1, const Func2 & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View value(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(value);
		View loValue(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(loValue);
		View hiValue(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(hiValue);
		View loCountSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(loCountSrc);
		View hiCountSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(hiCountSrc);

		View loCountDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View hiCountDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View loCountDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View hiCountDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(value, loValue, hiValue, loCountSrc, hiCountSrc, loCountDst1, hiCountDst1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(value, loValue, hiValue, loCountSrc, hiCountSrc, loCountDst2, hiCountDst2));

		result = result && Compare(loCountDst1, loCountDst2, 0, true, 10, 0, "lo");
		result = result && Compare(hiCountDst1, hiCountDst2, 0, true, 10, 0, "hi");

		return result;
	}

	namespace
	{
		struct Func3
		{
			typedef void (*FuncPtr)(uint8_t * loCount, size_t loCountStride, size_t width, size_t height, 
				 uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride, 
				 uint8_t * hiValue, size_t hiValueStride, uint8_t threshold);

			FuncPtr func;
			std::string description;

			Func3(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & loCountSrc, const View & loValueSrc, const View & hiCountSrc, const View & hiValueSrc, 
				View & loCountDst, View & loValueDst, View & hiCountDst, View & hiValueDst, uint8_t threshold) const
			{
				Simd::Copy(loCountSrc, loCountDst);
				Simd::Copy(loValueSrc, loValueDst);
				Simd::Copy(hiCountSrc, hiCountDst);
				Simd::Copy(hiValueSrc, hiValueDst);
				TEST_PERFORMANCE_TEST(description);
				func(loCountDst.data, loCountDst.stride, loValueDst.width, loValueDst.height, loValueDst.data, loValueDst.stride, 
					hiCountDst.data, hiCountDst.stride, hiValueDst.data, hiValueDst.stride, threshold);
			}
		};
	}

#define FUNC3(function) Func3(function, std::string(#function))

	bool BackgroundAdjustRangeTest(int width, int height, const Func3 & f1, const Func3 & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View loCountSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(loCountSrc);
		View loValueSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(loValueSrc);
		View hiCountSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(hiCountSrc);
		View hiValueSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(hiValueSrc);

		View loCountDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View loValueDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View hiCountDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View hiValueDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View loCountDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View loValueDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View hiCountDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View hiValueDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(loCountSrc, loValueSrc,hiCountSrc, hiValueSrc,  
			loCountDst1, loValueDst1, hiCountDst1, hiValueDst1, 0x80));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(loCountSrc, loValueSrc,hiCountSrc, hiValueSrc,  
			loCountDst2, loValueDst2, hiCountDst2, hiValueDst2, 0x80));

		result = result && Compare(loCountDst1, loCountDst2, 0, true, 10, 0, "loCount");
		result = result && Compare(loValueDst1, loValueDst2, 0, true, 10, 0, "loValue");
		result = result && Compare(hiCountDst1, hiCountDst2, 0, true, 10, 0, "hiCount");
		result = result && Compare(hiValueDst1, hiValueDst2, 0, true, 10, 0, "hiValue");

		return result;
	}

	namespace
	{
		struct Func4
		{
			typedef void (*FuncPtr)(uint8_t * loCount, size_t loCountStride, size_t width, size_t height, 
				 uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride, 
				 uint8_t * hiValue, size_t hiValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride);

			FuncPtr func;
			std::string description;

			Func4(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & loCountSrc, const View & loValueSrc, const View & hiCountSrc, const View & hiValueSrc, 
				View & loCountDst, View & loValueDst, View & hiCountDst, View & hiValueDst, uint8_t threshold, const View & mask) const
			{
				Simd::Copy(loCountSrc, loCountDst);
				Simd::Copy(loValueSrc, loValueDst);
				Simd::Copy(hiCountSrc, hiCountDst);
				Simd::Copy(hiValueSrc, hiValueDst);
				TEST_PERFORMANCE_TEST(description + "<m>");
				func(loCountDst.data, loCountDst.stride, loValueDst.width, loValueDst.height, loValueDst.data, loValueDst.stride, 
					hiCountDst.data, hiCountDst.stride, hiValueDst.data, hiValueDst.stride, threshold, mask.data, mask.stride);
			}
		};
	}

#define FUNC4(function) Func4(function, std::string(#function))

	bool BackgroundAdjustRangeMaskedTest(int width, int height, const Func4 & f1, const Func4 & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View loCountSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(loCountSrc);
		View loValueSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(loValueSrc);
		View hiCountSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(hiCountSrc);
		View hiValueSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(hiValueSrc);
		View mask(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandomMask(mask, 0xFF);

		View loCountDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View loValueDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View hiCountDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View hiValueDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View loCountDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View loValueDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View hiCountDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View hiValueDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(loCountSrc, loValueSrc,hiCountSrc, hiValueSrc,  
			loCountDst1, loValueDst1, hiCountDst1, hiValueDst1, 0x80, mask));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(loCountSrc, loValueSrc,hiCountSrc, hiValueSrc,  
			loCountDst2, loValueDst2, hiCountDst2, hiValueDst2, 0x80, mask));

		result = result && Compare(loCountDst1, loCountDst2, 0, true, 10, 0, "loCount");
		result = result && Compare(loValueDst1, loValueDst2, 0, true, 10, 0, "loValue");
		result = result && Compare(hiCountDst1, hiCountDst2, 0, true, 10, 0, "hiCount");
		result = result && Compare(hiValueDst1, hiValueDst2, 0, true, 10, 0, "hiValue");

		return result;
	}

	namespace
	{
		struct Func5
		{
			typedef void (*FuncPtr)(const uint8_t * value, size_t valueStride, size_t width, size_t height,
				 uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride, const uint8_t * mask, size_t maskStride);

			FuncPtr func;
			std::string description;

			Func5(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & value, const View & loSrc, const View & hiSrc, View & loDst, View & hiDst, const View & mask) const
			{
				Simd::Copy(loSrc, loDst);
				Simd::Copy(hiSrc, hiDst);
				TEST_PERFORMANCE_TEST(description + "<m>");
				func(value.data, value.stride, value.width, value.height, loDst.data, loDst.stride, hiDst.data, hiDst.stride,
					mask.data, mask.stride);
			}
		};
	}

#define FUNC5(function) Func5(function, std::string(#function))

	bool BackgroundShiftRangeMaskedTest(int width, int height, const Func5 & f1, const Func5 & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View value(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(value);
		View loSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(loSrc);
		View hiSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(hiSrc);
		View mask(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandomMask(mask, 0xFF);

		View loDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View hiDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View loDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View hiDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(value, loSrc, hiSrc, loDst1, hiDst1, mask));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(value, loSrc, hiSrc, loDst2, hiDst2, mask));

		result = result && Compare(loDst1, loDst2, 0, true, 10, 0, "lo");
		result = result && Compare(hiDst1, hiDst2, 0, true, 10, 0, "hi");

		return result;
	}

	namespace
	{
		struct Func6
		{
			typedef void (*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height,
				 uint8_t index, uint8_t value, uint8_t * dst, size_t dstStride);

			FuncPtr func;
			std::string description;

			Func6(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, uint8_t index, uint8_t value, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, index, value, dst.data, dst.stride);
			}
		};
	}

#define FUNC6(function) Func6(function, std::string(#function))

	bool BackgroundInitMaskTest(int width, int height, const Func6 & f1, const Func6 & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		 uint8_t index = 1 + Random(255);
		View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandomMask(src, index);

		View dst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		memset(dst1.data, 0, dst1.stride*height);
		View dst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		memset(dst2.data, 0, dst2.stride*height);

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, index, 0xFF, dst1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, index, 0xFF, dst2));

		result = result && Compare(dst1, dst2, 0, true, 10);

		return result;
	}

	bool BackgroundGrowRangeSlowTest()
	{
		bool result = true;

		result = result && BackgroundChangeRangeTest(W, H, FUNC1(Simd::Base::BackgroundGrowRangeSlow), FUNC1(SimdBackgroundGrowRangeSlow));
		result = result && BackgroundChangeRangeTest(W + 1, H - 1, FUNC1(Simd::Base::BackgroundGrowRangeSlow), FUNC1(SimdBackgroundGrowRangeSlow));
        result = result && BackgroundChangeRangeTest(W - 1, H + 1, FUNC1(Simd::Base::BackgroundGrowRangeSlow), FUNC1(SimdBackgroundGrowRangeSlow));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && BackgroundChangeRangeTest(W, H, FUNC1(Simd::Sse2::BackgroundGrowRangeSlow), FUNC1(Simd::Avx2::BackgroundGrowRangeSlow));
            result = result && BackgroundChangeRangeTest(W + 1, H - 1, FUNC1(Simd::Sse2::BackgroundGrowRangeSlow), FUNC1(Simd::Avx2::BackgroundGrowRangeSlow));
            result = result && BackgroundChangeRangeTest(W - 1, H + 1, FUNC1(Simd::Sse2::BackgroundGrowRangeSlow), FUNC1(Simd::Avx2::BackgroundGrowRangeSlow));
        }
#endif 

		return result;
	}

	bool BackgroundGrowRangeFastTest()
	{
		bool result = true;

		result = result && BackgroundChangeRangeTest(W, H, FUNC1(Simd::Base::BackgroundGrowRangeFast), FUNC1(SimdBackgroundGrowRangeFast));
		result = result && BackgroundChangeRangeTest(W + 1, H - 1, FUNC1(Simd::Base::BackgroundGrowRangeFast), FUNC1(SimdBackgroundGrowRangeFast));
        result = result && BackgroundChangeRangeTest(W - 1, H + 1, FUNC1(Simd::Base::BackgroundGrowRangeFast), FUNC1(SimdBackgroundGrowRangeFast));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && BackgroundChangeRangeTest(W, H, FUNC1(Simd::Sse2::BackgroundGrowRangeFast), FUNC1(Simd::Avx2::BackgroundGrowRangeFast));
            result = result && BackgroundChangeRangeTest(W + 1, H - 1, FUNC1(Simd::Sse2::BackgroundGrowRangeFast), FUNC1(Simd::Avx2::BackgroundGrowRangeFast));
            result = result && BackgroundChangeRangeTest(W - 1, H + 1, FUNC1(Simd::Sse2::BackgroundGrowRangeFast), FUNC1(Simd::Avx2::BackgroundGrowRangeFast));
        }
#endif 

		return result;
	}

	bool BackgroundIncrementCountTest()
	{
		bool result = true;

		result = result && BackgroundIncrementCountTest(W, H, FUNC2(Simd::Base::BackgroundIncrementCount), FUNC2(SimdBackgroundIncrementCount));
		result = result && BackgroundIncrementCountTest(W + 1, H - 1, FUNC2(Simd::Base::BackgroundIncrementCount), FUNC2(SimdBackgroundIncrementCount));
        result = result && BackgroundIncrementCountTest(W - 1, H + 1, FUNC2(Simd::Base::BackgroundIncrementCount), FUNC2(SimdBackgroundIncrementCount));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && BackgroundIncrementCountTest(W, H, FUNC2(Simd::Sse2::BackgroundIncrementCount), FUNC2(Simd::Avx2::BackgroundIncrementCount));
            result = result && BackgroundIncrementCountTest(W + 1, H - 1, FUNC2(Simd::Sse2::BackgroundIncrementCount), FUNC2(Simd::Avx2::BackgroundIncrementCount));
            result = result && BackgroundIncrementCountTest(W - 1, H + 1, FUNC2(Simd::Sse2::BackgroundIncrementCount), FUNC2(Simd::Avx2::BackgroundIncrementCount));
        }
#endif 

		return result;
	}

	bool BackgroundAdjustRangeTest()
	{
		bool result = true;

		result = result && BackgroundAdjustRangeTest(W, H, FUNC3(Simd::Base::BackgroundAdjustRange), FUNC3(SimdBackgroundAdjustRange));
		result = result && BackgroundAdjustRangeTest(W + 1, H - 1, FUNC3(Simd::Base::BackgroundAdjustRange), FUNC3(SimdBackgroundAdjustRange));
        result = result && BackgroundAdjustRangeTest(W - 1, H + 1, FUNC3(Simd::Base::BackgroundAdjustRange), FUNC3(SimdBackgroundAdjustRange));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && BackgroundAdjustRangeTest(W, H, FUNC3(Simd::Sse2::BackgroundAdjustRange), FUNC3(Simd::Avx2::BackgroundAdjustRange));
            result = result && BackgroundAdjustRangeTest(W + 1, H - 1, FUNC3(Simd::Sse2::BackgroundAdjustRange), FUNC3(Simd::Avx2::BackgroundAdjustRange));
            result = result && BackgroundAdjustRangeTest(W - 1, H + 1, FUNC3(Simd::Sse2::BackgroundAdjustRange), FUNC3(Simd::Avx2::BackgroundAdjustRange));
        }
#endif 

		return result;
	}

	bool BackgroundAdjustRangeMaskedTest()
	{
		bool result = true;

		result = result && BackgroundAdjustRangeMaskedTest(W, H, FUNC4(Simd::Base::BackgroundAdjustRange), FUNC4(SimdBackgroundAdjustRangeMasked));
		result = result && BackgroundAdjustRangeMaskedTest(W + 1, H - 1, FUNC4(Simd::Base::BackgroundAdjustRange), FUNC4(SimdBackgroundAdjustRangeMasked));
        result = result && BackgroundAdjustRangeMaskedTest(W - 1, H + 1, FUNC4(Simd::Base::BackgroundAdjustRange), FUNC4(SimdBackgroundAdjustRangeMasked));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && BackgroundAdjustRangeMaskedTest(W, H, FUNC4(Simd::Sse2::BackgroundAdjustRange), FUNC4(Simd::Avx2::BackgroundAdjustRange));
            result = result && BackgroundAdjustRangeMaskedTest(W + 1, H - 1, FUNC4(Simd::Sse2::BackgroundAdjustRange), FUNC4(Simd::Avx2::BackgroundAdjustRange));
            result = result && BackgroundAdjustRangeMaskedTest(W - 1, H + 1, FUNC4(Simd::Sse2::BackgroundAdjustRange), FUNC4(Simd::Avx2::BackgroundAdjustRange));
        }
#endif 

		return result;
	}

	bool BackgroundShiftRangeTest()
	{
		bool result = true;

		result = result && BackgroundChangeRangeTest(W, H, FUNC1(Simd::Base::BackgroundShiftRange), FUNC1(SimdBackgroundShiftRange));
		result = result && BackgroundChangeRangeTest(W + 1, H - 1, FUNC1(Simd::Base::BackgroundShiftRange), FUNC1(SimdBackgroundShiftRange));
        result = result && BackgroundChangeRangeTest(W - 1, H + 1, FUNC1(Simd::Base::BackgroundShiftRange), FUNC1(SimdBackgroundShiftRange));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && BackgroundChangeRangeTest(W, H, FUNC1(Simd::Sse2::BackgroundShiftRange), FUNC1(Simd::Avx2::BackgroundShiftRange));
            result = result && BackgroundChangeRangeTest(W + 1, H - 1, FUNC1(Simd::Sse2::BackgroundShiftRange), FUNC1(Simd::Avx2::BackgroundShiftRange));
            result = result && BackgroundChangeRangeTest(W - 1, H + 1, FUNC1(Simd::Sse2::BackgroundShiftRange), FUNC1(Simd::Avx2::BackgroundShiftRange));
        }
#endif 

		return result;
	}

	bool BackgroundShiftRangeMaskedTest()
	{
		bool result = true;

		result = result && BackgroundShiftRangeMaskedTest(W, H, FUNC5(Simd::Base::BackgroundShiftRange), FUNC5(SimdBackgroundShiftRangeMasked));
		result = result && BackgroundShiftRangeMaskedTest(W + 1, H - 1, FUNC5(Simd::Base::BackgroundShiftRange), FUNC5(SimdBackgroundShiftRangeMasked));
        result = result && BackgroundShiftRangeMaskedTest(W - 1, H + 1, FUNC5(Simd::Base::BackgroundShiftRange), FUNC5(SimdBackgroundShiftRangeMasked));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && BackgroundShiftRangeMaskedTest(W, H, FUNC5(Simd::Sse2::BackgroundShiftRange), FUNC5(Simd::Avx2::BackgroundShiftRange));
            result = result && BackgroundShiftRangeMaskedTest(W + 1, H - 1, FUNC5(Simd::Sse2::BackgroundShiftRange), FUNC5(Simd::Avx2::BackgroundShiftRange));
            result = result && BackgroundShiftRangeMaskedTest(W - 1, H + 1, FUNC5(Simd::Sse2::BackgroundShiftRange), FUNC5(Simd::Avx2::BackgroundShiftRange));
        }
#endif 

		return result;
	}

	bool BackgroundInitMaskTest()
	{
		bool result = true;

		result = result && BackgroundInitMaskTest(W, H, FUNC6(Simd::Base::BackgroundInitMask), FUNC6(SimdBackgroundInitMask));
		result = result && BackgroundInitMaskTest(W + 1, H - 1, FUNC6(Simd::Base::BackgroundInitMask), FUNC6(SimdBackgroundInitMask));
        result = result && BackgroundInitMaskTest(W - 1, H + 1, FUNC6(Simd::Base::BackgroundInitMask), FUNC6(SimdBackgroundInitMask));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && BackgroundInitMaskTest(W, H, FUNC6(Simd::Sse2::BackgroundInitMask), FUNC6(Simd::Avx2::BackgroundInitMask));
            result = result && BackgroundInitMaskTest(W + 1, H - 1, FUNC6(Simd::Sse2::BackgroundInitMask), FUNC6(Simd::Avx2::BackgroundInitMask));
            result = result && BackgroundInitMaskTest(W - 1, H + 1, FUNC6(Simd::Sse2::BackgroundInitMask), FUNC6(Simd::Avx2::BackgroundInitMask));
        }
#endif 

		return result;
	}
}
