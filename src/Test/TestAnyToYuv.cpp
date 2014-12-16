/*
* Simd Library Tests.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar,
*               2014-2014 Antonenka Mikhail.
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
		struct Func
		{
			typedef void (*FuncPtr)(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
				uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & bgr, View & y, View & u, View & v) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(bgr.data, bgr.width, bgr.height, bgr.stride, y.data, y.stride, u.data, u.stride, v.data, v.stride);
			}
		};	
	}

#define FUNC(function) Func(function, #function)

	bool AnyToYuvAutoTest(int width, int height, View::Format srcType, bool is420, const Func & f1, const Func & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        const int uvWidth = is420 ? width/2 : width;
        const int uvHeight = is420 ? height/2 : height;

		View src(width, height, srcType, NULL, TEST_ALIGN(width));
		FillRandom(src);

		View y1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View u1(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));
		View v1(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));

		View y2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View u2(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));
		View v2(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, y1, u1, v1));
		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, y2, u2, v2));

		result = result && Compare(y1, y2, 0, true, 64, 0, "y");
		result = result && Compare(u1, u2, 0, true, 64, 0, "u");
		result = result && Compare(v1, v2, 0, true, 64, 0, "v");

		return result;
	}

	bool AnyToYuvAutoTest(View::Format srcType, bool is420, const Func & f1, const Func & f2)
	{
		bool result = true;

        int step = is420 ? E : O;
		
		result = result && AnyToYuvAutoTest(W, H, srcType, is420, f1, f2);
		result = result && AnyToYuvAutoTest(W + step, H - step, srcType, is420, f1, f2);
		result = result && AnyToYuvAutoTest(W - step, H + step, srcType, is420, f1, f2);
		
		return result;
	}

    bool BgraToYuv420pAutoTest()
    {
        bool result = true;

        result = result && AnyToYuvAutoTest(View::Bgra32, true, FUNC(Simd::Base::BgraToYuv420p), FUNC(SimdBgraToYuv420p));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && AnyToYuvAutoTest(View::Bgra32, true, FUNC(Simd::Sse2::BgraToYuv420p), FUNC(SimdBgraToYuv420p));
#endif 

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && AnyToYuvAutoTest(View::Bgra32, true, FUNC(Simd::Ssse3::BgraToYuv420p), FUNC(SimdBgraToYuv420p));
#endif

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && AnyToYuvAutoTest(View::Bgra32, true, FUNC(Simd::Avx2::BgraToYuv420p), FUNC(SimdBgraToYuv420p));
#endif

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && AnyToYuvAutoTest(View::Bgra32, true, FUNC(Simd::Vsx::BgraToYuv420p), FUNC(SimdBgraToYuv420p));
#endif

        return result;
    }

    bool BgraToYuv444pAutoTest()
    {
        bool result = true;

        result = result && AnyToYuvAutoTest(View::Bgra32, false, FUNC(Simd::Base::BgraToYuv444p), FUNC(SimdBgraToYuv444p));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && AnyToYuvAutoTest(View::Bgra32, false, FUNC(Simd::Sse2::BgraToYuv444p), FUNC(SimdBgraToYuv444p));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && AnyToYuvAutoTest(View::Bgra32, false, FUNC(Simd::Avx2::BgraToYuv444p), FUNC(SimdBgraToYuv444p));
#endif

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && AnyToYuvAutoTest(View::Bgra32, false, FUNC(Simd::Vsx::BgraToYuv444p), FUNC(SimdBgraToYuv444p));
#endif

        return result;
    }

	bool BgrToYuv420pAutoTest()
	{
		bool result = true;

		result = result && AnyToYuvAutoTest(View::Bgr24, true, FUNC(Simd::Base::BgrToYuv420p), FUNC(SimdBgrToYuv420p));

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && AnyToYuvAutoTest(View::Bgr24, true, FUNC(Simd::Ssse3::BgrToYuv420p), FUNC(SimdBgrToYuv420p));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && AnyToYuvAutoTest(View::Bgr24, true, FUNC(Simd::Avx2::BgrToYuv420p), FUNC(SimdBgrToYuv420p));
#endif

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && AnyToYuvAutoTest(View::Bgr24, true, FUNC(Simd::Vsx::BgrToYuv420p), FUNC(SimdBgrToYuv420p));
#endif

		return result;
	}

    bool BgrToYuv444pAutoTest()
    {
        bool result = true;

        result = result && AnyToYuvAutoTest(View::Bgr24, false, FUNC(Simd::Base::BgrToYuv444p), FUNC(SimdBgrToYuv444p));

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && AnyToYuvAutoTest(View::Bgr24, false, FUNC(Simd::Ssse3::BgrToYuv444p), FUNC(SimdBgrToYuv444p));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && AnyToYuvAutoTest(View::Bgr24, false, FUNC(Simd::Avx2::BgrToYuv444p), FUNC(SimdBgrToYuv444p));
#endif

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && AnyToYuvAutoTest(View::Bgr24, false, FUNC(Simd::Vsx::BgrToYuv444p), FUNC(SimdBgrToYuv444p));
#endif

        return result;
    }

	//-----------------------------------------------------------------------

	bool AnyToYuvDataTest(bool create, int width, int height, View::Format srcType, bool is420, const Func & f)
	{
		bool result = true;

		Data data(f.description);

		std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        const int uvWidth = is420 ? width/2 : width;
        const int uvHeight = is420 ? height/2 : height;

		View src(width, height, srcType, NULL, TEST_ALIGN(width));

		View y1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View u1(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));
		View v1(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));

		View y2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View u2(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));
		View v2(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));

		if(create)
		{
			FillRandom(src);

			TEST_SAVE(src);

			f.Call(src, y1, u1, v1);

			TEST_SAVE(y1);
			TEST_SAVE(u1);
			TEST_SAVE(v1);
		}
		else
		{
			TEST_LOAD(src);

			TEST_LOAD(y1);
			TEST_LOAD(u1);
			TEST_LOAD(v1);

			f.Call(src, y2, u2, v2);

			TEST_SAVE(y2);
			TEST_SAVE(u2);
			TEST_SAVE(v2);

            result = result && Compare(y1, y2, 0, true, 64, 0, "y");
            result = result && Compare(u1, u2, 0, true, 64, 0, "u");
            result = result && Compare(v1, v2, 0, true, 64, 0, "v");
		}

		return result;
	}

    bool BgraToYuv420pDataTest(bool create)
    {
        bool result = true;

        result = result && AnyToYuvDataTest(create, DW, DH, View::Bgra32, true, FUNC(SimdBgraToYuv420p));

        return result;
    }

    bool BgraToYuv444pDataTest(bool create)
    {
        bool result = true;

        result = result && AnyToYuvDataTest(create, DW, DH, View::Bgra32, false, FUNC(SimdBgraToYuv444p));

        return result;
    }

	bool BgrToYuv420pDataTest(bool create)
	{
		bool result = true;

		result = result && AnyToYuvDataTest(create, DW, DH, View::Bgr24, true, FUNC(SimdBgrToYuv420p));

		return result;
	}

    bool BgrToYuv444pDataTest(bool create)
    {
        bool result = true;

        result = result && AnyToYuvDataTest(create, DW, DH, View::Bgr24, false, FUNC(SimdBgrToYuv444p));

        return result;
    }
}