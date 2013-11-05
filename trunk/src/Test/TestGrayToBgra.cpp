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
		struct Func
		{
			typedef void(*FuncPtr)(const uchar *gray, size_t width, size_t height, size_t grayStride, uchar *bgra, size_t bgraStride, uchar alpha);
			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, View & dst, uchar alpha) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.width, src.height, src.stride, dst.data, dst.stride, alpha);
			}
		};	
	}

#define FUNC(func) Func(func, #func)

    bool GrayToBgraTest(int width, int height, const Func & f1, const Func & f2)
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

    bool GrayToBgraTest()
    {
        bool result = true;

        result = result && GrayToBgraTest(W, H, FUNC(Simd::Base::GrayToBgra), FUNC(SimdGrayToBgra));
        result = result && GrayToBgraTest(W + 1, H - 1, FUNC(Simd::Base::GrayToBgra), FUNC(SimdGrayToBgra));
        result = result && GrayToBgraTest(W - 1, H + 1, FUNC(Simd::Base::GrayToBgra), FUNC(SimdGrayToBgra));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && GrayToBgraTest(W, H, FUNC(Simd::Sse2::GrayToBgra), FUNC(Simd::Avx2::GrayToBgra));
            result = result && GrayToBgraTest(W + 1, H - 1, FUNC(Simd::Sse2::GrayToBgra), FUNC(Simd::Avx2::GrayToBgra));
            result = result && GrayToBgraTest(W - 1, H + 1, FUNC(Simd::Sse2::GrayToBgra), FUNC(Simd::Avx2::GrayToBgra));
        }
#endif 

        return result;    
    }
}