/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
            typedef void(*FuncPtr)(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t * bgra, size_t bgraStride, uint8_t alpha);
            FuncPtr func;
            String description;

            Func(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst, uint8_t alpha) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.width, src.height, src.stride, (SimdPixelFormatType)src.format, dst.data, dst.stride, alpha);
            }
        };
    }

#define FUNC(func) Func(func, #func)

    bool BayerToBgraAutoTest(int width, int height, View::Format format, const Func & f1, const Func & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " for size [" << width << "," << height << "] of " << FormatDescription(format) << ".");

        View s(width, height, format, NULL, TEST_ALIGN(width));
        FillRandom(s);

        View d1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1, 0xFF));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2, 0xFF));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool BayerToBgraAutoTest(const Func & f1, const Func & f2)
    {
        bool result = true;

        for (View::Format format = View::BayerGrbg; format <= View::BayerBggr; format = View::Format(format + 1))
        {
            result = result && BayerToBgraAutoTest(W, H, format, f1, f2);
            result = result && BayerToBgraAutoTest(W + E, H - E, format, f1, f2);
        }

        return result;
    }

    bool BayerToBgraAutoTest()
    {
        bool result = true;

        result = result && BayerToBgraAutoTest(FUNC(Simd::Base::BayerToBgra), FUNC(SimdBayerToBgra));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A + 2)
            result = result && BayerToBgraAutoTest(FUNC(Simd::Sse41::BayerToBgra), FUNC(SimdBayerToBgra));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A + 2)
            result = result && BayerToBgraAutoTest(FUNC(Simd::Avx2::BayerToBgra), FUNC(SimdBayerToBgra));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W >= Simd::Avx512bw::A + 2)
            result = result && BayerToBgraAutoTest(FUNC(Simd::Avx512bw::BayerToBgra), FUNC(SimdBayerToBgra));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A + 2)
            result = result && BayerToBgraAutoTest(FUNC(Simd::Neon::BayerToBgra), FUNC(SimdBayerToBgra));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool BayerToBgraDataTest(bool create, int width, int height, View::Format format, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, format, NULL, TEST_ALIGN(width));

        View dst1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View dst2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        if (create)
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

    bool BayerToBgraDataTest(bool create)
    {
        bool result = true;

        Func f = FUNC(SimdBayerToBgra);
        for (View::Format format = View::BayerGrbg; format <= View::BayerBggr; format = View::Format(format + 1))
        {
            Func fc = Func(f.func, f.description + Data::Description(format));
            result = result && BayerToBgraDataTest(create, DW, DH, format, fc);
        }

        return result;
    }
}
