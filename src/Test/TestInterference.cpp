/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
        struct Func1
        {
            typedef void(*FuncPtr)(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t value, int16_t saturation);

            FuncPtr func;
            String description;

            Func1(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & statisticSrc, View & statisticDst, uint8_t value, int16_t saturation) const
            {
                Simd::Copy(statisticSrc, statisticDst);
                TEST_PERFORMANCE_TEST(description);
                func(statisticDst.data, statisticDst.stride, statisticDst.width, statisticDst.height, value, saturation);
            }
        };

        struct Func2
        {
            typedef void(*FuncPtr)(uint8_t * statistic, size_t statisticStride, size_t width, size_t height,
                uint8_t value, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index);

            FuncPtr func;
            String description;

            Func2(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & statisticSrc, View & statisticDst, uint8_t value, int16_t saturation, const View & mask, uint8_t index) const
            {
                Simd::Copy(statisticSrc, statisticDst);
                TEST_PERFORMANCE_TEST(description);
                func(statisticDst.data, statisticDst.stride, statisticDst.width, statisticDst.height,
                    value, saturation, mask.data, mask.stride, index);
            }
        };
    }

#define FUNC1(function) Func1(function, std::string(#function))

#define FUNC2(function) Func2(function, std::string(#function))

    bool InterferenceChangeAutoTest(int width, int height, const Func1 & f1, const Func1 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View statisticSrc(width, height, View::Int16, NULL, TEST_ALIGN(width));
        FillRandom(statisticSrc, 0, 64);

        uint8_t value = 3;
        int16_t saturation = 8888;

        View statisticDst1(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View statisticDst2(width, height, View::Int16, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(statisticSrc, statisticDst1, value, saturation));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(statisticSrc, statisticDst2, value, saturation));

        result = result && Compare(statisticDst1, statisticDst2, 0, true, 32, 0);

        return result;
    }

    bool InterferenceChangeAutoTest(const Func1 & f1, const Func1 & f2)
    {
        bool result = true;

        result = result && InterferenceChangeAutoTest(W, H, f1, f2);
        result = result && InterferenceChangeAutoTest(W + O, H - O, f1, f2);
        result = result && InterferenceChangeAutoTest(W - O, H + O, f1, f2);

        return result;
    }

    bool InterferenceIncrementAutoTest()
    {
        bool result = true;

        result = result && InterferenceChangeAutoTest(FUNC1(Simd::Base::InterferenceIncrement), FUNC1(SimdInterferenceIncrement));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && InterferenceChangeAutoTest(FUNC1(Simd::Sse2::InterferenceIncrement), FUNC1(SimdInterferenceIncrement));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && InterferenceChangeAutoTest(FUNC1(Simd::Avx2::InterferenceIncrement), FUNC1(SimdInterferenceIncrement));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && InterferenceChangeAutoTest(FUNC1(Simd::Avx512bw::InterferenceIncrement), FUNC1(SimdInterferenceIncrement));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && InterferenceChangeAutoTest(FUNC1(Simd::Vmx::InterferenceIncrement), FUNC1(SimdInterferenceIncrement));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && InterferenceChangeAutoTest(FUNC1(Simd::Neon::InterferenceIncrement), FUNC1(SimdInterferenceIncrement));
#endif

        return result;
    }

    bool InterferenceDecrementAutoTest()
    {
        bool result = true;

        result = result && InterferenceChangeAutoTest(FUNC1(Simd::Base::InterferenceDecrement), FUNC1(SimdInterferenceDecrement));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && InterferenceChangeAutoTest(FUNC1(Simd::Sse2::InterferenceDecrement), FUNC1(SimdInterferenceDecrement));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && InterferenceChangeAutoTest(FUNC1(Simd::Avx2::InterferenceDecrement), FUNC1(SimdInterferenceDecrement));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && InterferenceChangeAutoTest(FUNC1(Simd::Avx512bw::InterferenceDecrement), FUNC1(SimdInterferenceDecrement));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && InterferenceChangeAutoTest(FUNC1(Simd::Vmx::InterferenceDecrement), FUNC1(SimdInterferenceDecrement));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && InterferenceChangeAutoTest(FUNC1(Simd::Neon::InterferenceDecrement), FUNC1(SimdInterferenceDecrement));
#endif

        return result;
    }

    bool InterferenceChangeMaskedAutoTest(int width, int height, const Func2 & f1, const Func2 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View statisticSrc(width, height, View::Int16, NULL, TEST_ALIGN(width));
        FillRandom(statisticSrc, 0, 64);

        uint8_t value = 3, index = 11;
        int16_t saturation = 8888;

        View mask(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandomMask(mask, index);

        View statisticDst1(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View statisticDst2(width, height, View::Int16, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(statisticSrc, statisticDst1, value, saturation, mask, index));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(statisticSrc, statisticDst2, value, saturation, mask, index));

        result = result && Compare(statisticDst1, statisticDst2, 0, true, 32, 0);

        return result;
    }

    bool InterferenceChangeMaskedAutoTest(const Func2 & f1, const Func2 & f2)
    {
        bool result = true;

        result = result && InterferenceChangeMaskedAutoTest(W, H, f1, f2);
        result = result && InterferenceChangeMaskedAutoTest(W + O, H - O, f1, f2);
        result = result && InterferenceChangeMaskedAutoTest(W - O, H + O, f1, f2);

        return result;
    }

    bool InterferenceIncrementMaskedAutoTest()
    {
        bool result = true;

        result = result && InterferenceChangeMaskedAutoTest(FUNC2(Simd::Base::InterferenceIncrementMasked), FUNC2(SimdInterferenceIncrementMasked));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && InterferenceChangeMaskedAutoTest(FUNC2(Simd::Sse2::InterferenceIncrementMasked), FUNC2(SimdInterferenceIncrementMasked));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && InterferenceChangeMaskedAutoTest(FUNC2(Simd::Avx2::InterferenceIncrementMasked), FUNC2(SimdInterferenceIncrementMasked));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && InterferenceChangeMaskedAutoTest(FUNC2(Simd::Avx512bw::InterferenceIncrementMasked), FUNC2(SimdInterferenceIncrementMasked));
#endif

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && InterferenceChangeMaskedAutoTest(FUNC2(Simd::Vmx::InterferenceIncrementMasked), FUNC2(SimdInterferenceIncrementMasked));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && InterferenceChangeMaskedAutoTest(FUNC2(Simd::Neon::InterferenceIncrementMasked), FUNC2(SimdInterferenceIncrementMasked));
#endif

        return result;
    }

    bool InterferenceDecrementMaskedAutoTest()
    {
        bool result = true;

        result = result && InterferenceChangeMaskedAutoTest(FUNC2(Simd::Base::InterferenceDecrementMasked), FUNC2(SimdInterferenceDecrementMasked));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && InterferenceChangeMaskedAutoTest(FUNC2(Simd::Sse2::InterferenceDecrementMasked), FUNC2(SimdInterferenceDecrementMasked));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && InterferenceChangeMaskedAutoTest(FUNC2(Simd::Avx2::InterferenceDecrementMasked), FUNC2(SimdInterferenceDecrementMasked));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && InterferenceChangeMaskedAutoTest(FUNC2(Simd::Avx512bw::InterferenceDecrementMasked), FUNC2(SimdInterferenceDecrementMasked));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && InterferenceChangeMaskedAutoTest(FUNC2(Simd::Vmx::InterferenceDecrementMasked), FUNC2(SimdInterferenceDecrementMasked));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && InterferenceChangeMaskedAutoTest(FUNC2(Simd::Neon::InterferenceDecrementMasked), FUNC2(SimdInterferenceDecrementMasked));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool InterferenceChangeDataTest(bool create, int width, int height, const Func1 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View statisticSrc(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View statisticDst1(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View statisticDst2(width, height, View::Int16, NULL, TEST_ALIGN(width));

        uint8_t value = 3;
        int16_t saturation = 8888;

        if (create)
        {
            FillRandom(statisticSrc, 0, 64);

            TEST_SAVE(statisticSrc);

            f.Call(statisticSrc, statisticDst1, value, saturation);

            TEST_SAVE(statisticDst1);
        }
        else
        {
            TEST_LOAD(statisticSrc);

            TEST_LOAD(statisticDst1);

            f.Call(statisticSrc, statisticDst2, value, saturation);

            TEST_SAVE(statisticDst2);

            result = result && Compare(statisticDst1, statisticDst2, 0, true, 32, 0);
        }

        return result;
    }

    bool InterferenceIncrementDataTest(bool create)
    {
        bool result = true;

        result = result && InterferenceChangeDataTest(create, DW, DH, FUNC1(SimdInterferenceIncrement));

        return result;
    }

    bool InterferenceDecrementDataTest(bool create)
    {
        bool result = true;

        result = result && InterferenceChangeDataTest(create, DW, DH, FUNC1(SimdInterferenceDecrement));

        return result;
    }

    bool InterferenceChangeMaskedDataTest(bool create, int width, int height, const Func2 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View statisticSrc(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View statisticDst1(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View statisticDst2(width, height, View::Int16, NULL, TEST_ALIGN(width));

        uint8_t value = 3, index = 11;
        int16_t saturation = 8888;

        View mask(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(statisticSrc, 0, 64);
            FillRandomMask(mask, index);

            TEST_SAVE(statisticSrc);
            TEST_SAVE(mask);

            f.Call(statisticSrc, statisticDst1, value, saturation, mask, index);

            TEST_SAVE(statisticDst1);
        }
        else
        {
            TEST_LOAD(statisticSrc);
            TEST_LOAD(mask);

            TEST_LOAD(statisticDst1);

            f.Call(statisticSrc, statisticDst2, value, saturation, mask, index);

            TEST_SAVE(statisticDst2);

            result = result && Compare(statisticDst1, statisticDst2, 0, true, 32, 0);
        }

        return result;
    }

    bool InterferenceIncrementMaskedDataTest(bool create)
    {
        bool result = true;

        result = result && InterferenceChangeMaskedDataTest(create, DW, DH, FUNC2(SimdInterferenceIncrementMasked));

        return result;
    }

    bool InterferenceDecrementMaskedDataTest(bool create)
    {
        bool result = true;

        result = result && InterferenceChangeMaskedDataTest(create, DW, DH, FUNC2(SimdInterferenceDecrementMasked));

        return result;
    }
}
