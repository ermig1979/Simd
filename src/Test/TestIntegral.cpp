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
            typedef void(*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                uint8_t * sum, size_t sumStride, uint8_t * sqsum, size_t sqsumStride, uint8_t * tilted, size_t tiltedStride,
                SimdPixelFormatType sumFormat, SimdPixelFormatType sqsumFormat);

            FuncPtr func;
            String description;

            Func(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & sum, View & sqsum, View & tilted) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, sum.data, sum.stride, sqsum.data, sqsum.stride, tilted.data, tilted.stride,
                    (SimdPixelFormatType)sum.format, (SimdPixelFormatType)sqsum.format);
            }
        };
    }

#define FUNC(function) Func(function, #function)


    bool IntegralAutoTest(int width, int height, bool sqsumEnable, bool tiltedEnable, View::Format sumFormat, View::Format sqsumFormat, const Func & f1, const Func & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(src);

        View sum1(width + 1, height + 1, sumFormat, NULL, TEST_ALIGN(width));
        View sum2(width + 1, height + 1, sumFormat, NULL, TEST_ALIGN(width));
        View sqsum1, sqsum2, tilted1, tilted2;
        if (sqsumEnable)
        {
            sqsum1.Recreate(width + 1, height + 1, sqsumFormat, NULL, TEST_ALIGN(width));
            sqsum2.Recreate(width + 1, height + 1, sqsumFormat, NULL, TEST_ALIGN(width));
        }
        if (tiltedEnable)
        {
            tilted1.Recreate(width + 1, height + 1, sumFormat, NULL, TEST_ALIGN(width));
            tilted2.Recreate(width + 1, height + 1, sumFormat, NULL, TEST_ALIGN(width));
        }

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, sum1, sqsum1, tilted1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, sum2, sqsum2, tilted2));

        result = result && Compare(sum1, sum2, 0, true, 32, 0, "sum");
        if (sqsumEnable)
            result = result && Compare(sqsum1, sqsum2, 0, true, 32, 0, "sqsum");
        if (tiltedEnable)
            result = result && Compare(tilted1, tilted2, 0, true, 32, 0, "tilted");

        return result;
    }

    bool IntegralAutoTest(View::Format sumFormat, View::Format sqsumFormat, const Func & f1, const Func & f2)
    {
        bool result = true;

        for (int sqsumEnable = 0; sqsumEnable <= 1; ++sqsumEnable)
        {
            for (int tiltedEnable = 0; tiltedEnable <= 1; ++tiltedEnable)
            {
                std::stringstream ss;
                ss << ColorDescription(sumFormat) + ColorDescription(sqsumFormat);
                ss << "[1" << sqsumEnable << tiltedEnable << "]";

                Func f1d = Func(f1.func, f1.description + ss.str());
                Func f2d = Func(f2.func, f2.description + ss.str());
                result = result && IntegralAutoTest(W, H, sqsumEnable != 0, tiltedEnable != 0, sumFormat, sqsumFormat, f1d, f2d);
                result = result && IntegralAutoTest(W + O, H - O, sqsumEnable != 0, tiltedEnable != 0, sumFormat, sqsumFormat, f1d, f2d);
                result = result && IntegralAutoTest(W - O, H + O, sqsumEnable != 0, tiltedEnable != 0, sumFormat, sqsumFormat, f1d, f2d);
            }
        }

        return result;
    }

    bool IntegralAutoTest(const Func & f1, const Func & f2)
    {
        bool result = true;

        result = result && IntegralAutoTest(View::Int32, View::Int32, f1, f2);
        result = result && IntegralAutoTest(View::Int32, View::Double, f1, f2);

        return result;
    }

    bool IntegralAutoTest()
    {
        bool result = true;

        result = result && IntegralAutoTest(FUNC(Simd::Base::Integral), FUNC(SimdIntegral));

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && IntegralAutoTest(FUNC(Simd::Avx2::Integral), FUNC(SimdIntegral));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && IntegralAutoTest(FUNC(Simd::Avx512bw::Integral), FUNC(SimdIntegral));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    bool IntegralDataTest(bool create, int width, int height, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View sum1(width + 1, height + 1, View::Int32, NULL, TEST_ALIGN(width));
        View sum2(width + 1, height + 1, View::Int32, NULL, TEST_ALIGN(width));
        View sqsum1(width + 1, height + 1, View::Int32, NULL, TEST_ALIGN(width));
        View sqsum2(width + 1, height + 1, View::Int32, NULL, TEST_ALIGN(width));
        View tilted1(width + 1, height + 1, View::Int32, NULL, TEST_ALIGN(width));
        View tilted2(width + 1, height + 1, View::Int32, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, sum1, sqsum1, tilted1);

            TEST_SAVE(sum1);
            TEST_SAVE(sqsum1);
            TEST_SAVE(tilted1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(sum1);
            TEST_LOAD(sqsum1);
            TEST_LOAD(tilted1);

            f.Call(src, sum2, sqsum2, tilted2);

            TEST_SAVE(sum2);
            TEST_SAVE(sqsum2);
            TEST_SAVE(tilted2);

            result = result && Compare(sum1, sum2, 0, true, 32, 0, "sum");
            result = result && Compare(sqsum1, sqsum2, 0, true, 32, 0, "sqsum");
            result = result && Compare(tilted1, tilted2, 0, true, 32, 0, "tilted");
        }

        return result;
    }

    bool IntegralDataTest(bool create)
    {
        bool result = true;

        result = result && IntegralDataTest(create, DW, DH, FUNC(SimdIntegral));

        return result;
    }
}
