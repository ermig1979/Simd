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

namespace Test
{
	namespace
	{
		struct Func
		{
			typedef void (*FuncPtr)(
				const uint8_t * uv, size_t uvStride, size_t width, size_t height,
				 uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & uv, View & u, View & v) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(uv.data, uv.stride, uv.width, uv.height, u.data, u.stride, v.data, v.stride);
			}
		};
	}
#define FUNC(function) Func(function, #function)

	bool DeinterleaveUvAutoTest(int width, int height, const Func & f1, const Func & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View uv(width, height, View::Uv16, NULL, TEST_ALIGN(width));
		FillRandom(uv);

		View u1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View v1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View u2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View v2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(uv, u1, v1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(uv, u2, v2));

		result = result && Compare(u1, u2, 0, true, 32);
		result = result && Compare(v1, v2, 0, true, 32);

		return result;
	}

    bool DeinterleaveUvAutoTest(const Func & f1, const Func & f2)
    {
        bool result = true;

        result = result && DeinterleaveUvAutoTest(W, H, f1, f2);
        result = result && DeinterleaveUvAutoTest(W + O, H - O, f1, f2);
        result = result && DeinterleaveUvAutoTest(W - O, H + O, f1, f2);

        return result;
    }

	bool DeinterleaveUvAutoTest()
	{
		bool result = true;

		result = result && DeinterleaveUvAutoTest(FUNC(Simd::Base::DeinterleaveUv), FUNC(SimdDeinterleaveUv));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && DeinterleaveUvAutoTest(FUNC(Simd::Sse2::DeinterleaveUv), FUNC(SimdDeinterleaveUv));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && DeinterleaveUvAutoTest(FUNC(Simd::Avx2::DeinterleaveUv), FUNC(SimdDeinterleaveUv));
#endif 

#ifdef SIMD_VMX_ENABLE
        if(Simd::Vmx::Enable)
            result = result && DeinterleaveUvAutoTest(FUNC(Simd::Vmx::DeinterleaveUv), FUNC(SimdDeinterleaveUv));
#endif 

		return result;
	}

    //-----------------------------------------------------------------------

    bool DeinterleaveUvDataTest(bool create, int width, int height, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View uv(width, height, View::Uv16, NULL, TEST_ALIGN(width));

        View u1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View v1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View u2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View v2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        if(create)
        {
            FillRandom(uv);

            TEST_SAVE(uv);

            f.Call(uv, u1, v1);

            TEST_SAVE(u1);
            TEST_SAVE(v1);
        }
        else
        {
            TEST_LOAD(uv);

            TEST_LOAD(u1);
            TEST_LOAD(v1);

            f.Call(uv, u2, v2);

            TEST_SAVE(u2);
            TEST_SAVE(v2);

            result = result && Compare(u1, u2, 0, true, 32, 0, "u");
            result = result && Compare(v1, v2, 0, true, 32, 0, "v");
        }

        return result;
    }

    bool DeinterleaveUvDataTest(bool create)
    {
        bool result = true;

        result = result && DeinterleaveUvDataTest(create, DW, DH, FUNC(SimdDeinterleaveUv));

        return result;
    }
}
