/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
        struct Func
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t width, size_t height, size_t srcStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat);
            FuncPtr func;
            String description;

            Func(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.width, src.height, src.stride, dst.data, dst.stride, (SimdPixelFormatType)dst.format);
            }
        };
    }

#define FUNC(func) Func(func, #func)

    bool AnyToBayerAutoTest(int width, int height, View::Format srcType, View::Format dstType, const Func & f1, const Func & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " for size [" << width << "," << height << "] of " << FormatDescription(dstType) << ".");

        View src(width, height, srcType, NULL, TEST_ALIGN(width));
        FillRandom(src);

        View dst1(width, height, dstType, NULL, TEST_ALIGN(width));
        View dst2(width, height, dstType, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

        result = result && Compare(dst1, dst2, 0, true, 64);

        return result;
    }

    bool AnyToBayerAutoTest(View::Format srcType, const Func & f1, const Func & f2)
    {
        bool result = true;

        for (View::Format dstType = View::BayerGrbg; dstType <= View::BayerBggr; dstType = View::Format(dstType + 1))
        {
            result = result && AnyToBayerAutoTest(W, H, srcType, dstType, f1, f2);
            result = result && AnyToBayerAutoTest(W + E, H - E, srcType, dstType, f1, f2);
        }

        return result;
    }

    bool BgrToBayerAutoTest()
    {
        bool result = true;

        result = result && AnyToBayerAutoTest(View::Bgr24, FUNC(Simd::Base::BgrToBayer), FUNC(SimdBgrToBayer));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
            result = result && AnyToBayerAutoTest(View::Bgr24, FUNC(Simd::Sse41::BgrToBayer), FUNC(SimdBgrToBayer));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && AnyToBayerAutoTest(View::Bgr24, FUNC(Simd::Avx512bw::BgrToBayer), FUNC(SimdBgrToBayer));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && AnyToBayerAutoTest(View::Bgr24, FUNC(Simd::Vmx::BgrToBayer), FUNC(SimdBgrToBayer));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && AnyToBayerAutoTest(View::Bgr24, FUNC(Simd::Neon::BgrToBayer), FUNC(SimdBgrToBayer));
#endif

        return result;
    }

    bool BgraToBayerAutoTest()
    {
        bool result = true;

        result = result && AnyToBayerAutoTest(View::Bgra32, FUNC(Simd::Base::BgraToBayer), FUNC(SimdBgraToBayer));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
            result = result && AnyToBayerAutoTest(View::Bgra32, FUNC(Simd::Sse41::BgraToBayer), FUNC(SimdBgraToBayer));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && AnyToBayerAutoTest(View::Bgra32, FUNC(Simd::Avx512bw::BgraToBayer), FUNC(SimdBgraToBayer));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && AnyToBayerAutoTest(View::Bgra32, FUNC(Simd::Vmx::BgraToBayer), FUNC(SimdBgraToBayer));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && AnyToBayerAutoTest(View::Bgra32, FUNC(Simd::Neon::BgraToBayer), FUNC(SimdBgraToBayer));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    bool AnyToBayerDataTest(bool create, int width, int height, View::Format srcType, View::Format dstType, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, srcType, NULL, TEST_ALIGN(width));

        View dst1(width, height, dstType, NULL, TEST_ALIGN(width));
        View dst2(width, height, dstType, NULL, TEST_ALIGN(width));

        if (create)
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

    bool BgrToBayerDataTest(bool create)
    {
        bool result = true;

        Func f = FUNC(SimdBgrToBayer);
        for (View::Format dstType = View::BayerGrbg; dstType <= View::BayerBggr; dstType = View::Format(dstType + 1))
        {
            Func fc = Func(f.func, f.description + Data::Description(dstType));
            result = result && AnyToBayerDataTest(create, DW, DH, View::Bgr24, dstType, fc);
        }

        return result;
    }

    bool BgraToBayerDataTest(bool create)
    {
        bool result = true;

        Func f = FUNC(SimdBgraToBayer);
        for (View::Format dstType = View::BayerGrbg; dstType <= View::BayerBggr; dstType = View::Format(dstType + 1))
        {
            Func fc = Func(f.func, f.description + Data::Description(dstType));
            result = result && AnyToBayerDataTest(create, DW, DH, View::Bgra32, dstType, fc);
        }

        return result;
    }
}
