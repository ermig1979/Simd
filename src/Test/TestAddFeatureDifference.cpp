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

namespace Test
{
    namespace
    {
        struct Func
        {
            typedef void(*FuncPtr)(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride,
                uint16_t weight, uint8_t * difference, size_t differenceStride);

            FuncPtr func;
            String description;

            Func(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & value, const View & lo, const View & hi, uint16_t weight, const View & differenceSrc, View & differenceDst) const
            {
                Simd::Copy(differenceSrc, differenceDst);
                TEST_PERFORMANCE_TEST(description);
                func(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride,
                    weight, differenceDst.data, differenceDst.stride);
            }
        };
    }

#define FUNC(function) Func(function, std::string(#function))

    bool AddFeatureDifferenceAutoTest(int width, int height, uint16_t weight, const Func & f1, const Func & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "] (" << weight / 256 << "*256).");

        View value(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(value);
        View lo(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(lo);
        View hi(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(hi);
        View differenceSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(differenceSrc);

        View differenceDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View differenceDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(value, lo, hi, weight, differenceSrc, differenceDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(value, lo, hi, weight, differenceSrc, differenceDst2));

        result = result && Compare(differenceDst1, differenceDst2, 0, true, 32, 0);

        return result;
    }

    bool AddFeatureDifferenceAutoTest(const Func & f1, const Func & f2)
    {
        bool result = true;

        const uint16_t delta = 256 * 7;
        for (uint16_t weight = 0; weight < 4 && result; ++weight)
        {
            result = result &&  AddFeatureDifferenceAutoTest(W, H, weight*delta, f1, f2);
            result = result &&  AddFeatureDifferenceAutoTest(W + O, H - O, weight*delta, f1, f2);
        }

        return result;
    }

    bool AddFeatureDifferenceAutoTest()
    {
        bool result = true;

        result = result && AddFeatureDifferenceAutoTest(FUNC(Simd::Base::AddFeatureDifference), FUNC(SimdAddFeatureDifference));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
            result = result && AddFeatureDifferenceAutoTest(FUNC(Simd::Sse41::AddFeatureDifference), FUNC(SimdAddFeatureDifference));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && AddFeatureDifferenceAutoTest(FUNC(Simd::Avx2::AddFeatureDifference), FUNC(SimdAddFeatureDifference));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && AddFeatureDifferenceAutoTest(FUNC(Simd::Avx512bw::AddFeatureDifference), FUNC(SimdAddFeatureDifference));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && AddFeatureDifferenceAutoTest(FUNC(Simd::Vmx::AddFeatureDifference), FUNC(SimdAddFeatureDifference));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && AddFeatureDifferenceAutoTest(FUNC(Simd::Neon::AddFeatureDifference), FUNC(SimdAddFeatureDifference));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool AddFeatureDifferenceDataTest(bool create, int width, int height, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View value(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View lo(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View hi(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View differenceSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View differenceDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View differenceDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        const uint16_t weight = 256 * 7;

        if (create)
        {
            FillRandom(value);
            FillRandom(lo);
            FillRandom(hi);
            FillRandom(differenceSrc);

            TEST_SAVE(value);
            TEST_SAVE(lo);
            TEST_SAVE(hi);
            TEST_SAVE(differenceSrc);

            f.Call(value, lo, hi, weight, differenceSrc, differenceDst1);

            TEST_SAVE(differenceDst1);
        }
        else
        {
            TEST_LOAD(value);
            TEST_LOAD(lo);
            TEST_LOAD(hi);

            TEST_LOAD(differenceSrc);
            TEST_LOAD(differenceDst1);

            f.Call(value, lo, hi, weight, differenceSrc, differenceDst2);

            TEST_SAVE(differenceDst2);

            result = result && Compare(differenceDst1, differenceDst2, 0, true, 32, 0);
        }

        return result;
    }

    bool AddFeatureDifferenceDataTest(bool create)
    {
        bool result = true;

        result = result && AddFeatureDifferenceDataTest(create, DW, DH, FUNC(SimdAddFeatureDifference));

        return result;
    }
}
