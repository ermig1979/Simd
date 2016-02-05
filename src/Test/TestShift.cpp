/*
* Tests for Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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
		struct Func
		{
			typedef void (*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, 
				const uint8_t * bkg, size_t bkgStride, const double * shiftX, const double * shiftY, 
				size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uint8_t * dst, size_t dstStride);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, const View & bkg, double shiftX, double shiftY, 
				size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, View::PixelSize(src.format), bkg.data, bkg.stride,
					&shiftX, &shiftY, cropLeft, cropTop, cropRight, cropBottom, dst.data, dst.stride);
			}
		};
	}

#define ARGS(format, width, height, function1, function2) \
	format, width, height, \
	Func(function1.func, function1.description + ColorDescription(format)), \
	Func(function2.func, function2.description + ColorDescription(format))

#define FUNC(function) \
    Func(function, std::string(#function))

	bool ShiftAutoTest(View::Format format, int width, int height, double dx, double dy, int crop, const Func & f1, const Func & f2)
	{
		bool result = true;

		TEST_LOG_SS(Info, std::setprecision(1) << std::fixed << "Test " << f1.description << " & " << f2.description 
            << " [" << width << ", " << height << "]," << " (" << dx << ", " << dy << ", " << crop << ").");

		View s(width, height, format, NULL, TEST_ALIGN(width));
		FillRandom(s);
		View b(width, height, format, NULL, TEST_ALIGN(width));
		FillRandom(b);

		View d1(width, height, format, NULL, TEST_ALIGN(width));
		View d2(width, height, format, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, b, dx, dy, crop, crop, width - crop, height - crop, d1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, b, dx, dy, crop, crop, width - crop, height - crop, d2));

		result = result && Compare(d1, d2, 0, true, 32);

		return result;
	}

	bool ShiftAutoTest(View::Format format, int width, int height, const Func & f1, const Func & f2)
	{
		bool result = true;

		const double x0 = 7.1, dx = -5.3, y0 = -5.2, dy = 3.7;
		for(int i = 0; i < 4; ++i)
			result = result && ShiftAutoTest(format, width, height, x0 + i*dx, y0 + i*dy, i*3, f1, f2);

		return result;
	}

    bool ShiftBilinearAutoTest(const Func & f1, const Func & f2)
    {
        bool result = true;

        for(View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
        {
            result = result && ShiftAutoTest(ARGS(format, W, H, f1, f2));
            result = result && ShiftAutoTest(ARGS(format, W + O, H - O, f1, f2));
            result = result && ShiftAutoTest(ARGS(format, W - O, H + O, f1, f2));
        }

        return result;
    }

	bool ShiftBilinearAutoTest()
	{
		bool result = true;

		result = result && ShiftBilinearAutoTest(FUNC(Simd::Base::ShiftBilinear), FUNC(SimdShiftBilinear));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && ShiftBilinearAutoTest(FUNC(Simd::Sse2::ShiftBilinear), FUNC(SimdShiftBilinear));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && ShiftBilinearAutoTest(FUNC(Simd::Avx2::ShiftBilinear), FUNC(SimdShiftBilinear));
#endif 

#ifdef SIMD_VMX_ENABLE
        if(Simd::Vmx::Enable)
            result = result && ShiftBilinearAutoTest(FUNC(Simd::Vmx::ShiftBilinear), FUNC(SimdShiftBilinear));
#endif 

#ifdef SIMD_NEON_ENABLE
		if (Simd::Neon::Enable)
			result = result && ShiftBilinearAutoTest(FUNC(Simd::Neon::ShiftBilinear), FUNC(SimdShiftBilinear));
#endif 

		return result;
	}

    //-----------------------------------------------------------------------

    bool ShiftBilinearDataTest(bool create, int width, int height, View::Format format, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View s(width, height, format, NULL, TEST_ALIGN(width));
        View b(width, height, format, NULL, TEST_ALIGN(width));
        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));

        const double dx = -5.3, dy = 3.7;
        const int crop = 3;

        if(create)
        {
            FillRandom(s);
            FillRandom(b);
            TEST_SAVE(s);
            TEST_SAVE(b);

            f.Call(s, b, dx, dy, crop, crop, width - crop, height - crop, d1);

            TEST_SAVE(d1);
        }
        else
        {
            TEST_LOAD(s);
            TEST_LOAD(b);
            TEST_LOAD(d1);

            f.Call(s, b, dx, dy, crop, crop, width - crop, height - crop, d2);

            TEST_SAVE(d2);

            result = result && Compare(d1, d2, 0, true, 64);
        }

        return result;
    }

    bool ShiftBilinearDataTest(bool create)
    {
        bool result = true;

        Func f = FUNC(SimdShiftBilinear);
        for(View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
        {
            result = result && ShiftBilinearDataTest(create, DW, DH, format, Func(f.func, f.description + Data::Description(format)));
        }

        return result;
    }
}
