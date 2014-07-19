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
			typedef void(*FuncPtr)(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat);
			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.width, src.height, src.stride, dst.data, dst.stride, (SimdPixelFormatType)dst.format);
			}
		};	
	}

#define FUNC(func) Func(func, #func)

    bool BgraToBayerAutoTest(int width, int height, View::Format format, const Func & f1, const Func & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " for size [" << width << "," << height << "] of " << FormatDescription(format) << "." << std::endl;

        View s(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        FillRandom(s);

        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

        result = result && Compare(d1, d2, 0, true, 32);

        return result;
    }

    bool BgraToBayerAutoTest(const Func & f1, const Func & f2)
    {
        bool result = true;

        for(View::Format format = View::BayerGrbg; format <= View::BayerBggr; format = View::Format(format + 1))
        {
            result = result && BgraToBayerAutoTest(W, H, format, f1, f2);
            result = result && BgraToBayerAutoTest(W + E, H - E, format, f1, f2);
            result = result && BgraToBayerAutoTest(W - E, H + E, format, f1, f2);
        }

        return result;
    }

    bool BgraToBayerAutoTest()
    {
        bool result = true;

        result = result && BgraToBayerAutoTest(FUNC(Simd::Base::BgraToBayer), FUNC(SimdBgraToBayer));

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && BgraToBayerAutoTest(FUNC(Simd::Ssse3::BgraToBayer), FUNC(SimdBgraToBayer));
#endif

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && BgraToBayerAutoTest(FUNC(Simd::Vsx::BgraToBayer), FUNC(SimdBgraToBayer));
#endif 

        return result;    
    }

    //-----------------------------------------------------------------------

    bool BgraToBayerDataTest(bool create, int width, int height, View::Format format, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        View dst1(width, height, format, NULL, TEST_ALIGN(width));
        View dst2(width, height, format, NULL, TEST_ALIGN(width));

        if(create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 0, true, 32, 0);
        }

        return result;
    }

    bool BgraToBayerDataTest(bool create)
    {
        bool result = true;

        Func f = FUNC(SimdBgraToBayer);
        for(View::Format format = View::BayerGrbg; format <= View::BayerBggr; format = View::Format(format + 1))
        {
            Func fc = Func(f.func, f.description + Data::Description(format));
            result = result && BgraToBayerDataTest(create, DW, DH, format, fc);
        }

        return result;
    }
}