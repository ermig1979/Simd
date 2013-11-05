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
            typedef void(*FuncPtr)(const uint8_t * blue, size_t blueStride, size_t width, size_t height,
                const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha);
			FuncPtr func;
			std::string description;

			Func1(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & blue, const View & green, const View & red, View & bgra, uint8_t alpha) const
			{
				TEST_PERFORMANCE_TEST(description);
                func(blue.data, blue.stride, blue.width, blue.height, green.data, green.stride, red.data, red.stride, bgra.data, bgra.stride, alpha);
			}
		};	
	}

#define FUNC1(func) Func1(func, #func)

    bool Bgr48pToBgra32Test(int width, int height, const Func1 & f1, const Func1 & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " for size [" << width << "," << height << "]." << std::endl;

        View blue(width, height, View::Int16, NULL, TEST_ALIGN(width));
        FillRandom(blue);
        View green(width, height, View::Int16, NULL, TEST_ALIGN(width));
        FillRandom(green);
        View red(width, height, View::Int16, NULL, TEST_ALIGN(width));
        FillRandom(red);

        uint8_t alpha = 0xFF;

        View bgra1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View bgra2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(blue, green, red, bgra1, alpha));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(blue, green, red, bgra2, alpha));

        result = result && Compare(bgra1, bgra2, 0, true, 32);

        return result;
    }

    bool Bgr48pToBgra32Test()
    {
        bool result = true;
	
        result = result && Bgr48pToBgra32Test(W, H, FUNC1(Simd::Base::Bgr48pToBgra32), FUNC1(SimdBgr48pToBgra32));
        result = result && Bgr48pToBgra32Test(W + 1, H - 1, FUNC1(Simd::Base::Bgr48pToBgra32), FUNC1(SimdBgr48pToBgra32));
        result = result && Bgr48pToBgra32Test(W - 1, H + 1, FUNC1(Simd::Base::Bgr48pToBgra32), FUNC1(SimdBgr48pToBgra32));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && Bgr48pToBgra32Test(W, H, FUNC1(Simd::Sse2::Bgr48pToBgra32), FUNC1(Simd::Avx2::Bgr48pToBgra32));
            result = result && Bgr48pToBgra32Test(W + 1, H - 1, FUNC1(Simd::Sse2::Bgr48pToBgra32), FUNC1(Simd::Avx2::Bgr48pToBgra32));
            result = result && Bgr48pToBgra32Test(W - 1, H + 1, FUNC1(Simd::Sse2::Bgr48pToBgra32), FUNC1(Simd::Avx2::Bgr48pToBgra32));
        }
#endif 
		return result;    
    }
}