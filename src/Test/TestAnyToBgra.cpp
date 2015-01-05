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
#include "Test/Test.h"

namespace Test
{
	namespace
	{
        struct Func
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t width, size_t height, size_t srcStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha);
            FuncPtr func;
            std::string description;

            Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & src, View & bgra, uint8_t alpha) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.width, src.height, src.stride, bgra.data, bgra.stride, alpha);
            }
        };	
	}

#define FUNC(func) Func(func, #func)

    bool AnyToBgraAutoTest(int width, int height, View::Format srcType, const Func & f1, const Func & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " for size [" << width << "," << height << "]." << std::endl;

        View src(width, height, srcType, NULL, TEST_ALIGN(width));
        FillRandom(src);

        View dst1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View dst2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        uint8_t alpha = 0xFF;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1, alpha));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2, alpha));

        result = result && Compare(dst1, dst2, 0, true, 64);

        return result;
    }

    bool AnyToBgraAutoTest(View::Format srcType, const Func & f1, const Func & f2)
    {
        bool result = true;

        result = result && AnyToBgraAutoTest(W, H, srcType, f1, f2);
        result = result && AnyToBgraAutoTest(W + O, H - O, srcType, f1, f2);
        result = result && AnyToBgraAutoTest(W - O, H + O, srcType, f1, f2);

        return result;    
    }

    bool BgrToBgraAutoTest()
    {
        bool result = true;

        result = result && AnyToBgraAutoTest(View::Bgr24, FUNC(Simd::Base::BgrToBgra), FUNC(SimdBgrToBgra));

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && AnyToBgraAutoTest(View::Bgr24, FUNC(Simd::Ssse3::BgrToBgra), FUNC(SimdBgrToBgra));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && AnyToBgraAutoTest(View::Bgr24, FUNC(Simd::Avx2::BgrToBgra), FUNC(SimdBgrToBgra));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && AnyToBgraAutoTest(View::Bgr24, FUNC(Simd::Vsx::BgrToBgra), FUNC(SimdBgrToBgra));
#endif 

        return result;    
    }

    bool GrayToBgraAutoTest()
    {
        bool result = true;

        result = result && AnyToBgraAutoTest(View::Gray8, FUNC(Simd::Base::GrayToBgra), FUNC(SimdGrayToBgra));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && AnyToBgraAutoTest(View::Gray8, FUNC(Simd::Sse2::GrayToBgra), FUNC(SimdGrayToBgra));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && AnyToBgraAutoTest(View::Gray8, FUNC(Simd::Avx2::GrayToBgra), FUNC(SimdGrayToBgra));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && AnyToBgraAutoTest(View::Gray8, FUNC(Simd::Vsx::GrayToBgra), FUNC(SimdGrayToBgra));
#endif 

        return result;    
    }

    //-----------------------------------------------------------------------

    bool AnyToBgraDataTest(bool create, int width, int height, View::Format srcType, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, srcType, NULL, TEST_ALIGN(width));

        View dst1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View dst2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        if(create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, dst1, 0xFF);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, dst2, 0xFF);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 0, true, 32, 0);
        }

        return result;
    }

    bool BgrToBgraDataTest(bool create)
    {
        bool result = true;

        result = result && AnyToBgraDataTest(create, DW, DH, View::Bgr24, FUNC(SimdBgrToBgra));

        return result;
    }

    bool GrayToBgraDataTest(bool create)
    {
        bool result = true;

        result = result && AnyToBgraDataTest(create, DW, DH, View::Gray8, FUNC(SimdGrayToBgra));

        return result;
    }
}