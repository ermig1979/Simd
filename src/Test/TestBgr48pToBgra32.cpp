/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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

namespace Test
{
    namespace
    {
        struct Func
        {
            typedef void(*FuncPtr)(const uint8_t * blue, size_t blueStride, size_t width, size_t height,
                const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha);
            FuncPtr func;
            String description;

            Func(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & blue, const View & green, const View & red, View & bgra, uint8_t alpha) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(blue.data, blue.stride, blue.width, blue.height, green.data, green.stride, red.data, red.stride, bgra.data, bgra.stride, alpha);
            }
        };
    }

#define FUNC(func) Func(func, #func)

    bool Bgr48pToBgra32AutoTest(int width, int height, const Func & f1, const Func & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " for size [" << width << "," << height << "].");

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

        result = result && Compare(bgra1, bgra2, 0, true, 64);

        return result;
    }

    bool Bgr48pToBgra32AutoTest(const Func & f1, const Func & f2)
    {
        bool result = true;

        result = result && Bgr48pToBgra32AutoTest(W, H, f1, f2);
        result = result && Bgr48pToBgra32AutoTest(W + O, H - O, f1, f2);

        return result;
    }

    bool Bgr48pToBgra32AutoTest()
    {
        bool result = true;

        result = result && Bgr48pToBgra32AutoTest(FUNC(Simd::Base::Bgr48pToBgra32), FUNC(SimdBgr48pToBgra32));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::HA)
            result = result && Bgr48pToBgra32AutoTest(FUNC(Simd::Sse2::Bgr48pToBgra32), FUNC(SimdBgr48pToBgra32));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::HA)
            result = result && Bgr48pToBgra32AutoTest(FUNC(Simd::Avx2::Bgr48pToBgra32), FUNC(SimdBgr48pToBgra32));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && Bgr48pToBgra32AutoTest(FUNC(Simd::Avx512bw::Bgr48pToBgra32), FUNC(SimdBgr48pToBgra32));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::HA)
            result = result && Bgr48pToBgra32AutoTest(FUNC(Simd::Vmx::Bgr48pToBgra32), FUNC(SimdBgr48pToBgra32));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::HA)
            result = result && Bgr48pToBgra32AutoTest(FUNC(Simd::Neon::Bgr48pToBgra32), FUNC(SimdBgr48pToBgra32));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    bool Bgr48pToBgra32DataTest(bool create, int width, int height, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View blue(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View green(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View red(width, height, View::Int16, NULL, TEST_ALIGN(width));

        uint8_t alpha = 0xFF;

        View bgra1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View bgra2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(blue);
            FillRandom(green);
            FillRandom(red);

            TEST_SAVE(blue);
            TEST_SAVE(green);
            TEST_SAVE(red);

            f.Call(blue, green, red, bgra1, alpha);

            TEST_SAVE(bgra1);
        }
        else
        {
            TEST_LOAD(blue);
            TEST_LOAD(green);
            TEST_LOAD(red);

            TEST_LOAD(bgra1);

            f.Call(blue, green, red, bgra2, alpha);

            TEST_SAVE(bgra2);

            result = result && Compare(bgra1, bgra2, 0, true, 32);
        }

        return result;
    }

    bool Bgr48pToBgra32DataTest(bool create)
    {
        bool result = true;

        result = result && Bgr48pToBgra32DataTest(create, DW, DH, FUNC(SimdBgr48pToBgra32));

        return result;
    }
}
