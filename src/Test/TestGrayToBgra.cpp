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
		struct Func
		{
			typedef void(*FuncPtr)(const uint8_t *gray, size_t width, size_t height, size_t grayStride, uint8_t *bgra, size_t bgraStride, uint8_t alpha);
			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, View & dst, uint8_t alpha) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.width, src.height, src.stride, dst.data, dst.stride, alpha);
			}
		};	
	}

#define FUNC(func) Func(func, #func)

    bool GrayToBgraAutoTest(int width, int height, const Func & f1, const Func & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " for size [" << width << "," << height << "]." << std::endl;

        View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(s);

        View d1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1, 0xFF));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2, 0xFF));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool GrayToBgraAutoTest(const Func & f1, const Func & f2)
    {
        bool result = true;

        result = result && GrayToBgraAutoTest(W, H, f1, f2);
        result = result && GrayToBgraAutoTest(W + 3, H - 3, f1, f2);
        result = result && GrayToBgraAutoTest(W - 3, H + 3, f1, f2);

        return result;    
    }

    bool GrayToBgraAutoTest()
    {
        bool result = true;

        result = result && GrayToBgraAutoTest(FUNC(Simd::Base::GrayToBgra), FUNC(SimdGrayToBgra));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && GrayToBgraAutoTest(FUNC(Simd::Sse2::GrayToBgra), FUNC(SimdGrayToBgra));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && GrayToBgraAutoTest(FUNC(Simd::Avx2::GrayToBgra), FUNC(SimdGrayToBgra));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && GrayToBgraAutoTest(FUNC(Simd::Vsx::GrayToBgra), FUNC(SimdGrayToBgra));
#endif 

        return result;    
    }

    //-----------------------------------------------------------------------

    bool GrayToBgraDataTest(bool create, int width, int height, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View gray(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View bgra1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View bgra2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        const uint8_t alpha = 127;

        if(create)
        {
            FillRandom(gray);

            TEST_SAVE(gray);

            f.Call(gray, bgra1, alpha);

            TEST_SAVE(bgra1);
        }
        else
        {
            TEST_LOAD(gray);

            TEST_LOAD(bgra1);

            f.Call(gray, bgra2, alpha);

            TEST_SAVE(bgra2);

            result = result && Compare(bgra1, bgra2, 0, true, 32, 0);
        }

        return result;
    }

    bool GrayToBgraDataTest(bool create)
    {
        bool result = true;

        result = result && GrayToBgraDataTest(create, DW, DH, FUNC(SimdGrayToBgra));

        return result;
    }
}