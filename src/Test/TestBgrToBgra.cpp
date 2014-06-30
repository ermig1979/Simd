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
            typedef void(*FuncPtr)(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha);
            FuncPtr func;
            std::string description;

            Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & src, View & bgr, uint8_t alpha) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.width, src.height, src.stride, bgr.data, bgr.stride, alpha);
            }
        };	
	}

#define FUNC(func) Func(func, #func)

    bool BgrToBgraAutoTest(int width, int height, const Func & f1, const Func & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " for size [" << width << "," << height << "]." << std::endl;

        View s(width, height, View::Bgr24, NULL, TEST_ALIGN(width));
        FillRandom(s);

        View d1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        uint8_t alpha = 0xFF;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1, alpha));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2, alpha));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool BgrToBgraAutoTest(const Func & f1, const Func & f2)
    {
        bool result = true;

        result = result && BgrToBgraAutoTest(W, H, f1, f2);
        result = result && BgrToBgraAutoTest(W + 3, H - 3, f1, f2);
        result = result && BgrToBgraAutoTest(W - 3, H + 3, f1, f2);

        return result;    
    }

    bool BgrToBgraAutoTest()
    {
        bool result = true;

        result = result && BgrToBgraAutoTest(FUNC(Simd::Base::BgrToBgra), FUNC(SimdBgrToBgra));

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && BgrToBgraAutoTest(FUNC(Simd::Ssse3::BgrToBgra), FUNC(SimdBgrToBgra));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && BgrToBgraAutoTest(FUNC(Simd::Avx2::BgrToBgra), FUNC(SimdBgrToBgra));
#endif 

#ifdef SIMD_VSX_ENABLE
        if(Simd::Vsx::Enable)
            result = result && BgrToBgraAutoTest(FUNC(Simd::Vsx::BgrToBgra), FUNC(SimdBgrToBgra));
#endif 

        return result;    
    }

    namespace
    {
        struct FuncP
        {
            typedef void(*FuncPtr)(const uint8_t * blue, size_t blueStride, size_t width, size_t height,
                const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha);
            FuncPtr func;
            std::string description;

            FuncP(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & blue, const View & green, const View & red, View & bgra, uint8_t alpha) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(blue.data, blue.stride, blue.width, blue.height, green.data, green.stride, red.data, red.stride, bgra.data, bgra.stride, alpha);
            }
        };	
    }

#define FUNCP(func) FuncP(func, #func)

    bool Bgr48pToBgra32AutoTest(int width, int height, const FuncP & f1, const FuncP & f2)
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

    bool Bgr48pToBgra32AutoTest()
    {
        bool result = true;
	
        result = result && Bgr48pToBgra32AutoTest(W, H, FUNCP(Simd::Base::Bgr48pToBgra32), FUNCP(SimdBgr48pToBgra32));
        result = result && Bgr48pToBgra32AutoTest(W + 1, H - 1, FUNCP(Simd::Base::Bgr48pToBgra32), FUNCP(SimdBgr48pToBgra32));
        result = result && Bgr48pToBgra32AutoTest(W - 1, H + 1, FUNCP(Simd::Base::Bgr48pToBgra32), FUNCP(SimdBgr48pToBgra32));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && Bgr48pToBgra32AutoTest(W, H, FUNCP(Simd::Sse2::Bgr48pToBgra32), FUNCP(Simd::Avx2::Bgr48pToBgra32));
            result = result && Bgr48pToBgra32AutoTest(W + 1, H - 1, FUNCP(Simd::Sse2::Bgr48pToBgra32), FUNCP(Simd::Avx2::Bgr48pToBgra32));
            result = result && Bgr48pToBgra32AutoTest(W - 1, H + 1, FUNCP(Simd::Sse2::Bgr48pToBgra32), FUNCP(Simd::Avx2::Bgr48pToBgra32));
        }
#endif 
		return result;    
    }

    //-----------------------------------------------------------------------

    bool BgrToBgraDataTest(bool create, int width, int height, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        std::cout << (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, View::Bgr24, NULL, TEST_ALIGN(width));

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

        result = result && BgrToBgraDataTest(create, DW, DH, FUNC(SimdBgrToBgra));

        return result;
    }
}