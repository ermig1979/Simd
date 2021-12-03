/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar,
*               2014-2016 Antonenka Mikhail.
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
#include "Test/TestString.h"

namespace Test
{
    namespace
    {
        struct Func0

        {
            typedef void(*FuncPtr)(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType);
            FuncPtr func;
            String description;

            Func0(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & uyvy, View & bgr, SimdYuvType yuvType) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(uyvy.data, uyvy.stride, uyvy.width, uyvy.height, bgr.data, bgr.stride, yuvType);
            }
        };
    }

#define FUNC_0(func) Func0(func, #func)

    bool Uyvy422ToBgrAutoTest(int width, int height, SimdYuvType yuvType, const Func0 & f1, const Func0 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " for size [" << width << "," << height << "].");

        View src(width, height, View::Uyvy16, NULL, TEST_ALIGN(width));
        FillRandom(src);
        //FillSequence(src);

        View dst1(width, height, View::Bgr24, NULL, TEST_ALIGN(width));
        View dst2(width, height, View::Bgr24, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1, yuvType));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2, yuvType));

        result = result && Compare(dst1, dst2, 0, true, 64);

        return result;
    }

    bool Uyvy422ToBgrAutoTest(const Func0 & f1, const Func0 & f2)
    {
        bool result = true;

        result = result && Uyvy422ToBgrAutoTest(W, H, SimdYuvBt601, f1, f2);
        result = result && Uyvy422ToBgrAutoTest(W + E, H - E, SimdYuvBt709, f1, f2);

        return result;
    }

    bool Uyvy422ToBgrAutoTest()
    {
        bool result = true;

        result = result && Uyvy422ToBgrAutoTest(FUNC_0(Simd::Base::Uyvy422ToBgr), FUNC_0(SimdUyvy422ToBgr));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && Uyvy422ToBgrAutoTest(FUNC_0(Simd::Sse41::Uyvy422ToBgr), FUNC_0(SimdUyvy422ToBgr));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && Uyvy422ToBgrAutoTest(FUNC_0(Simd::Avx2::Uyvy422ToBgr), FUNC_0(SimdUyvy422ToBgr));
#endif 

        return result;
    }
}
