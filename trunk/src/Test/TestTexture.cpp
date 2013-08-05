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
            typedef void (*FuncPtr)(const uchar * src, size_t srcStride, size_t width, size_t height, 
                uchar saturation, uchar boost, uchar * dx, size_t dxStride, uchar * dy, size_t dyStride);

            FuncPtr func;
            std::string description;

            Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & src, uchar saturation, uchar boost, View &  dx, View & dy) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, saturation, boost, dx.data, dx.stride, dy.data, dy.stride);
            }
        };
    }
#define FUNC(function) Func(function, #function)

    bool TextureBoostedSaturatedGradientTest(int width, int height, const Func & f1, const Func & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(src);

        View dx1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View dy1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View dx2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View dy2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, 16, 4, dx1, dy1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, 16, 4, dx2, dy2));

        result = result && Compare(dx1, dx2, 0, true, 32, 0, "dx");
        result = result && Compare(dy1, dy2, 0, true, 32, 0, "dy");

        return result;
    }

    bool TextureBoostedSaturatedGradientTest()
    {
        bool result = true;

        result = result && TextureBoostedSaturatedGradientTest(W, H, FUNC(Simd::Base::TextureBoostedSaturatedGradient), FUNC(Simd::TextureBoostedSaturatedGradient));
        result = result && TextureBoostedSaturatedGradientTest(W - 1, H + 1, FUNC(Simd::Base::TextureBoostedSaturatedGradient), FUNC(Simd::TextureBoostedSaturatedGradient));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && TextureBoostedSaturatedGradientTest(W, H, FUNC(Simd::Sse2::TextureBoostedSaturatedGradient), FUNC(Simd::Avx2::TextureBoostedSaturatedGradient));
            result = result && TextureBoostedSaturatedGradientTest(W - 1, H + 1, FUNC(Simd::Sse2::TextureBoostedSaturatedGradient), FUNC(Simd::Avx2::TextureBoostedSaturatedGradient));
        }
#endif 

        return result;
    }
}
