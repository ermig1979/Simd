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
		struct FuncAB
		{
            typedef void(*FuncPtr)(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount, 
                const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride);
			FuncPtr func;
			std::string description;

			FuncAB(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, const View & alpha, const View & dstSrc, View & dstDst) const
			{
                Simd::Copy(dstSrc, dstDst);
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, src.ChannelCount(), alpha.data, alpha.stride, dstDst.data, dstDst.stride);
			}
		};	
	}

#define FUNC_AB(func) FuncAB(func, #func)

    bool AlphaBlendingTest(View::Format format, int width, int height, const FuncAB & f1, const FuncAB & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " for size [" << width << "," << height << "]." << std::endl;

        View s(width, height, format, NULL, TEST_ALIGN(width));
        FillRandom(s);
        View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(a);
        View b(width, height, format, NULL, TEST_ALIGN(width));
        FillRandom(b);

        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, a, b, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, a, b, d2));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool AlphaBlendingTest(const FuncAB & f1, const FuncAB & f2)
    {
        bool result = true;

        for(View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
        {
            FuncAB f1c = FuncAB(f1.func, f1.description + ColorDescription(format));
            FuncAB f2c = FuncAB(f2.func, f2.description + ColorDescription(format));
            
            result = result && AlphaBlendingTest(format, W, H, f1c, f2c);
            result = result && AlphaBlendingTest(format, W + 1, H - 1, f1c, f2c);
            result = result && AlphaBlendingTest(format, W - 1, H + 1, f1c, f2c);
        }

        return result;
    }

    bool AlphaBlendingTest()
    {
        bool result = true;

        result = result && AlphaBlendingTest(FUNC_AB(Simd::Base::AlphaBlending), FUNC_AB(SimdAlphaBlending));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
            result = result && AlphaBlendingTest(FUNC_AB(Simd::Sse2::AlphaBlending), FUNC_AB(Simd::Avx2::AlphaBlending));
#endif 

        return result;    
    }
}