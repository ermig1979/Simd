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
			typedef void (*FuncPtr)(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, 
				size_t width, size_t height, uint8_t * uv, size_t uvStride);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & u, const View & v, View & uv) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(u.data, u.stride, v.data, v.stride, u.width, u.height, uv.data, uv.stride);
			}
		};
	}
#define FUNC(function) Func(function, #function)

	bool InterleaveUvAutoTest(int width, int height, const Func & f1, const Func & f2)
	{
		bool result = true;

		TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

		View u(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(u);
		View v(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(v);

		View uv1(width, height, View::Uv16, NULL, TEST_ALIGN(width));
		View uv2(width, height, View::Uv16, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(u, v, uv1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(u, v, uv2));

		result = result && Compare(uv1, uv2, 0, true, 32);

		return result;
	}

    bool InterleaveUvAutoTest(const Func & f1, const Func & f2)
    {
        bool result = true;

        result = result && InterleaveUvAutoTest(W, H, f1, f2);
        result = result && InterleaveUvAutoTest(W + O, H - O, f1, f2);
        result = result && InterleaveUvAutoTest(W - O, H + O, f1, f2);

        return result;
    }

	bool InterleaveUvAutoTest()
	{
		bool result = true;

		result = result && InterleaveUvAutoTest(FUNC(Simd::Base::InterleaveUv), FUNC(SimdInterleaveUv));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && InterleaveUvAutoTest(FUNC(Simd::Sse2::InterleaveUv), FUNC(SimdInterleaveUv));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && InterleaveUvAutoTest(FUNC(Simd::Avx2::InterleaveUv), FUNC(SimdInterleaveUv));
#endif 

#ifdef SIMD_VMX_ENABLE
        if(Simd::Vmx::Enable)
            result = result && InterleaveUvAutoTest(FUNC(Simd::Vmx::InterleaveUv), FUNC(SimdInterleaveUv));
#endif 

#ifdef SIMD_NEON_ENABLE
		if (Simd::Neon::Enable)
			result = result && InterleaveUvAutoTest(FUNC(Simd::Neon::InterleaveUv), FUNC(SimdInterleaveUv));
#endif

		return result;
	}

    //-----------------------------------------------------------------------

    bool InterleaveUvDataTest(bool create, int width, int height, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

		View u(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View v(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		View uv1(width, height, View::Uv16, NULL, TEST_ALIGN(width));
		View uv2(width, height, View::Uv16, NULL, TEST_ALIGN(width));

        if(create)
        {
            FillRandom(u);
			FillRandom(v);

            TEST_SAVE(u);
			TEST_SAVE(v);

            f.Call(u, v, uv1);

            TEST_SAVE(uv1);
        }
        else
        {
            TEST_LOAD(u);
            TEST_LOAD(v);

			TEST_LOAD(uv1);

            f.Call(u, v, uv2);

            TEST_SAVE(uv2);

            result = result && Compare(uv1, uv2, 0, true, 32);
        }

        return result;
    }

    bool InterleaveUvDataTest(bool create)
    {
        bool result = true;

        result = result && InterleaveUvDataTest(create, DW, DH, FUNC(SimdInterleaveUv));

        return result;
    }
}
