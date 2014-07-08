/*
* Simd Library Tests.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
		struct FuncBgra
		{
			typedef void (*FuncPtr)(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha);

			FuncPtr func;
			std::string description;

			FuncBgra(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(View & dst, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(dst.data, dst.stride, dst.width, dst.height, blue, green, red, alpha);
			}
		};
	}

#define ARGS_BGRA(width, height, function1, function2) \
	width, height, FuncBgra(function1, std::string(#function1)), FuncBgra(function2, std::string(#function2))

	bool FillBgraAutoTest(int width, int height, const FuncBgra & f1, const FuncBgra & f2)
	{
		bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description
            << " [" << width << ", " << height << "]." << std::endl;

        uint8_t blue = Random(256);
        uint8_t green = Random(256);
        uint8_t red = Random(256);
        uint8_t alpha = Random(256);

		View d1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
		View d2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(d1, blue, green, red, alpha));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(d2, blue, green, red, alpha));

		result = result && Compare(d1, d2, 0, true, 32);

		return result;
	}

	bool FillBgraAutoTest()
	{
		bool result = true;

		result = result && FillBgraAutoTest(ARGS_BGRA(W, H, Simd::Base::FillBgra, SimdFillBgra));
        result = result && FillBgraAutoTest(ARGS_BGRA(W + 1, H - 1, Simd::Base::FillBgra, SimdFillBgra));
        result = result && FillBgraAutoTest(ARGS_BGRA(W - 1, H + 1, Simd::Base::FillBgra, SimdFillBgra));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && FillBgraAutoTest(ARGS_BGRA(W, H, Simd::Sse2::FillBgra, Simd::Avx2::FillBgra));
            result = result && FillBgraAutoTest(ARGS_BGRA(W + 1, H - 1, Simd::Sse2::FillBgra, Simd::Avx2::FillBgra));
            result = result && FillBgraAutoTest(ARGS_BGRA(W - 1, H + 1, Simd::Sse2::FillBgra, Simd::Avx2::FillBgra));
        }
#endif 

		return result;
	}

    namespace
    {
        struct FuncBgr
        {
            typedef void (*FuncPtr)(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red);

            FuncPtr func;
            std::string description;

            FuncBgr(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(View & dst, uint8_t blue, uint8_t green, uint8_t red) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(dst.data, dst.stride, dst.width, dst.height, blue, green, red);
            }
        };
    }

#define FUNC_BGR(function) \
    FuncBgr(function, std::string(#function))

    bool FillBgrAutoTest(int width, int height, const FuncBgr & f1, const FuncBgr & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description
            << " [" << width << ", " << height << "]." << std::endl;

        uint8_t blue = Random(256);
        uint8_t green = Random(256);
        uint8_t red = Random(256);

        View d1(width, height, View::Bgr24, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Bgr24, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(d1, blue, green, red));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(d2, blue, green, red));

        result = result && Compare(d1, d2, 0, true, 32);

        return result;
    }

    bool FillBgrAutoTest(const FuncBgr & f1, const FuncBgr & f2)
    {
        bool result = true;

        result = result && FillBgrAutoTest(W, H, f1, f2);
        result = result && FillBgrAutoTest(W + O, H - O, f1, f2);
        result = result && FillBgrAutoTest(W - O, H + O, f1, f2);

        return result;
    }


    bool FillBgrAutoTest()
    {
        bool result = true;

        result = result && FillBgrAutoTest(FUNC_BGR(Simd::Base::FillBgr), FUNC_BGR(SimdFillBgr));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && FillBgrAutoTest(FUNC_BGR(Simd::Sse2::FillBgr), FUNC_BGR(SimdFillBgr));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && FillBgrAutoTest(FUNC_BGR(Simd::Avx2::FillBgr), FUNC_BGR(SimdFillBgr));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && FillBgrAutoTest(FUNC_BGR(Simd::Vsx::FillBgr), FUNC_BGR(SimdFillBgr));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool FillBgrDataTest(bool create, int width, int height, const FuncBgr & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View bgr1(width, height, View::Bgr24, NULL, TEST_ALIGN(width));
        View bgr2(width, height, View::Bgr24, NULL, TEST_ALIGN(width));

        const uint8_t blue = 0x11;
        const uint8_t green = 0xAA;
        const uint8_t red = 0x77;

        if(create)
        {
            f.Call(bgr1, blue, green, red);

            TEST_SAVE(bgr1);
        }
        else
        {
            TEST_LOAD(bgr1);

            f.Call(bgr2, blue, green, red);

            TEST_SAVE(bgr2);

            result = result && Compare(bgr1, bgr2, 0, true, 64);
        }

        return result;
    }

    bool FillBgrDataTest(bool create)
    {
        bool result = true;

        result = result && FillBgrDataTest(create, DW, DH, FUNC_BGR(SimdFillBgr));

        return result;
    }
}
