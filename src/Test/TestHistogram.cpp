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
        struct FuncH
        {
            typedef void(*FuncPtr)(
                const uint8_t *src, size_t width, size_t height, size_t stride, uint32_t * histogram);

            FuncPtr func;
            String description;

            FuncH(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, uint32_t * histogram) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.width, src.height, src.stride, histogram);
            }
        };

        struct FuncHM
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                const uint8_t * mask, size_t maskStride, uint8_t index, uint32_t * histogram);

            FuncPtr func;
            String description;

            FuncHM(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, const View & mask, uint8_t index, uint32_t * histogram) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, mask.data, mask.stride, index, histogram);
            }
        };


        struct FuncASDH
        {
            typedef void(*FuncPtr)(
                const uint8_t *src, size_t width, size_t height, size_t stride,
                size_t step, size_t indent, uint32_t * histogram);

            FuncPtr func;
            String description;

            FuncASDH(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, size_t step, size_t indent, uint32_t * histogram) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.width, src.height, src.stride,
                    step, indent, histogram);
            }
        };
    }

#define FUNC_H(function) FuncH(function, #function)

#define FUNC_HM(function) FuncHM(function, #function)

#define FUNC_ASDH(function) FuncASDH(function, #function)

    bool HistogramAutoTest(int width, int height, const FuncH & f1, const FuncH & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View s(int(width), int(height), View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(s);

        Histogram h1 = { 0 }, h2 = { 0 };

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, h1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, h2));

        result = result && Compare(h1, h2, 0, true, 32);

        return result;
    }

    bool HistogramAutoTest(const FuncH & f1, const FuncH & f2)
    {
        bool result = true;

        result = result && HistogramAutoTest(W, H, f1, f2);
        result = result && HistogramAutoTest(W + O, H - O, f1, f2);

        return result;
    }

    bool HistogramAutoTest()
    {
        bool result = true;

        result = result && HistogramAutoTest(FUNC_H(Simd::Base::Histogram), FUNC_H(SimdHistogram));

        return result;
    }

    bool HistogramMaskedAutoTest(int width, int height, const FuncHM & f1, const FuncHM & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View s(int(width), int(height), View::Gray8, NULL, TEST_ALIGN(width));
        View m(int(width), int(height), View::Gray8, NULL, TEST_ALIGN(width));

        const uint8_t index = 77;
        FillRandom(s);
        FillRandomMask(m, index);

        Histogram h1 = { 0 }, h2 = { 0 };

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, m, index, h1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, m, index, h2));

        result = result && Compare(h1, h2, 0, true, 32);

        return result;
    }

    bool HistogramMaskedAutoTest(const FuncHM & f1, const FuncHM & f2)
    {
        bool result = true;

        result = result && HistogramMaskedAutoTest(W, H, f1, f2);
        result = result && HistogramMaskedAutoTest(W + O, H - O, f1, f2);

        return result;
    }

    bool HistogramMaskedAutoTest()
    {
        bool result = true;

        result = result && HistogramMaskedAutoTest(FUNC_HM(Simd::Base::HistogramMasked), FUNC_HM(SimdHistogramMasked));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A)
            result = result && HistogramMaskedAutoTest(FUNC_HM(Simd::Sse2::HistogramMasked), FUNC_HM(SimdHistogramMasked));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && HistogramMaskedAutoTest(FUNC_HM(Simd::Avx2::HistogramMasked), FUNC_HM(SimdHistogramMasked));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && HistogramMaskedAutoTest(FUNC_HM(Simd::Avx512bw::HistogramMasked), FUNC_HM(SimdHistogramMasked));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && HistogramMaskedAutoTest(FUNC_HM(Simd::Vmx::HistogramMasked), FUNC_HM(SimdHistogramMasked));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && HistogramMaskedAutoTest(FUNC_HM(Simd::Neon::HistogramMasked), FUNC_HM(SimdHistogramMasked));
#endif 

        return result;
    }

    bool AbsSecondDerivativeHistogramAutoTest(int width, int height, int step, int indent, int A, const FuncASDH & f1, const FuncASDH & f2)
    {
        bool result = true;

        if (width > 2 * indent && height > 2 * indent && indent >= step && width >= A + 2 * indent)
        {
            TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "] (" << step << ", " << indent << ").");

            View s(int(width), int(height), View::Gray8, NULL, TEST_ALIGN(width));
            FillRandom(s);

            Histogram h1 = { 0 }, h2 = { 0 };

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, step, indent, h1));

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, step, indent, h2));

            result = result && Compare(h1, h2, 0, true, 32);
        }

        return result;
    }

    bool AbsSecondDerivativeHistogramAutoTest(int A, const FuncASDH & f1, const FuncASDH & f2)
    {
        bool result = true;

        result = result && AbsSecondDerivativeHistogramAutoTest(W, H, 1, 16, A, f1, f2);
        result = result && AbsSecondDerivativeHistogramAutoTest(W + O, H - O, 2, 16, A, f1, f2);
        result = result && AbsSecondDerivativeHistogramAutoTest(W, H, 3, 8, A, f1, f2);

        return result;
    }

    bool AbsSecondDerivativeHistogramAutoTest()
    {
        bool result = true;

        result = result && AbsSecondDerivativeHistogramAutoTest(1, FUNC_ASDH(Simd::Base::AbsSecondDerivativeHistogram), FUNC_ASDH(SimdAbsSecondDerivativeHistogram));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && AbsSecondDerivativeHistogramAutoTest(Simd::Sse2::A, FUNC_ASDH(Simd::Sse2::AbsSecondDerivativeHistogram), FUNC_ASDH(SimdAbsSecondDerivativeHistogram));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && AbsSecondDerivativeHistogramAutoTest(Simd::Avx2::A, FUNC_ASDH(Simd::Avx2::AbsSecondDerivativeHistogram), FUNC_ASDH(SimdAbsSecondDerivativeHistogram));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && AbsSecondDerivativeHistogramAutoTest(Simd::Avx512bw::A, FUNC_ASDH(Simd::Avx512bw::AbsSecondDerivativeHistogram), FUNC_ASDH(SimdAbsSecondDerivativeHistogram));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && AbsSecondDerivativeHistogramAutoTest(Simd::Vmx::A, FUNC_ASDH(Simd::Vmx::AbsSecondDerivativeHistogram), FUNC_ASDH(SimdAbsSecondDerivativeHistogram));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && AbsSecondDerivativeHistogramAutoTest(Simd::Neon::A, FUNC_ASDH(Simd::Neon::AbsSecondDerivativeHistogram), FUNC_ASDH(SimdAbsSecondDerivativeHistogram));
#endif 

        return result;
    }

    namespace
    {
        struct FuncCC
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                const uint8_t * colors, uint8_t * dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncCC(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, const View & colors, const View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, colors.data, dst.data, dst.stride);
            }
        };
    }

#define FUNC_CC(function) \
    FuncCC(function, std::string(#function))

    bool ChangeColorsAutoTest(int width, int height, const FuncCC & f1, const FuncCC & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(s);

        View c(Simd::HISTOGRAM_SIZE, 1, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(c, 64, 191);

        View d1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, c, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, c, d2));

        result = result && Compare(d1, d2, 0, true, 32);

        return result;
    }

    bool ChangeColorsAutoTest(const FuncCC & f1, const FuncCC & f2)
    {
        bool result = true;

        result = result && ChangeColorsAutoTest(W, H, f1, f2);
        result = result && ChangeColorsAutoTest(W + O, H - O, f1, f2);

        return result;
    }

    bool ChangeColorsAutoTest()
    {
        bool result = true;

        result = result && ChangeColorsAutoTest(FUNC_CC(Simd::Base::ChangeColors), FUNC_CC(SimdChangeColors));

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W >= Simd::Avx512bw::HA)
            result = result && ChangeColorsAutoTest(FUNC_CC(Simd::Avx512bw::ChangeColors), FUNC_CC(SimdChangeColors));
#endif 

        return result;
    }

    namespace
    {
        struct FuncHC
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint32_t * histogram);

            FuncPtr func;
            String description;

            FuncHC(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, const View & mask, uint8_t value, SimdCompareType compareType, uint32_t * histogram) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, mask.data, mask.stride, value, compareType, histogram);
            }
        };
    }

#define ARGS_HC(width, height, type, function1, function2) \
    width, height, type, \
    FuncHC(function1.func, function1.description + CompareTypeDescription(type)), \
    FuncHC(function2.func, function2.description + CompareTypeDescription(type))

#define FUNC_HC(function) \
    FuncHC(function, std::string(#function))

    bool HistogramConditionalAutoTest(int width, int height, SimdCompareType type, const FuncHC & f1, const FuncHC & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View s(int(width), int(height), View::Gray8, NULL, TEST_ALIGN(width));
        View m(int(width), int(height), View::Gray8, NULL, TEST_ALIGN(width));

        uint8_t value = 127;
        FillRandom(s);
        FillRandom(m);

        Histogram h1 = { 0 }, h2 = { 0 };

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, m, value, type, h1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, m, value, type, h2));

        result = result && Compare(h1, h2, 0, true, 32);

        return result;
    }

    bool HistogramConditionalAutoTest(const FuncHC & f1, const FuncHC & f2)
    {
        bool result = true;

        for (SimdCompareType type = SimdCompareEqual; type <= SimdCompareLesserOrEqual && result; type = SimdCompareType(type + 1))
        {
            result = result && HistogramConditionalAutoTest(ARGS_HC(W, H, type, f1, f2));
            result = result && HistogramConditionalAutoTest(ARGS_HC(W + O, H - O, type, f1, f2));
        }

        return result;
    }

    bool HistogramConditionalAutoTest()
    {
        bool result = true;

        result = result && HistogramConditionalAutoTest(FUNC_HC(Simd::Base::HistogramConditional), FUNC_HC(SimdHistogramConditional));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A)
            result = result && HistogramConditionalAutoTest(FUNC_HC(Simd::Sse2::HistogramConditional), FUNC_HC(SimdHistogramConditional));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && HistogramConditionalAutoTest(FUNC_HC(Simd::Avx2::HistogramConditional), FUNC_HC(SimdHistogramConditional));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && HistogramConditionalAutoTest(FUNC_HC(Simd::Avx512bw::HistogramConditional), FUNC_HC(SimdHistogramConditional));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && HistogramConditionalAutoTest(FUNC_HC(Simd::Neon::HistogramConditional), FUNC_HC(SimdHistogramConditional));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool HistogramDataTest(bool create, int width, int height, const FuncH & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        Histogram h1, h2;

        if (create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, h1);

            TEST_SAVE(h1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(h1);

            f.Call(src, h2);

            TEST_SAVE(h2);

            result = result && Compare(h1, h2, 0, true, 32);
        }

        return result;
    }

    bool HistogramDataTest(bool create)
    {
        bool result = true;

        result = result && HistogramDataTest(create, DW, DH, FUNC_H(SimdHistogram));

        return result;
    }

    bool HistogramMaskedDataTest(bool create, int width, int height, const FuncHM & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View mask(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        const uint8_t index = 77;
        Histogram h1, h2;

        if (create)
        {
            FillRandom(src);
            FillRandomMask(mask, index);

            TEST_SAVE(src);
            TEST_SAVE(mask);

            f.Call(src, mask, index, h1);

            TEST_SAVE(h1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(mask);

            TEST_LOAD(h1);

            f.Call(src, mask, index, h2);

            TEST_SAVE(h2);

            result = result && Compare(h1, h2, 0, true, 32);
        }

        return result;
    }

    bool HistogramMaskedDataTest(bool create)
    {
        bool result = true;

        result = result && HistogramMaskedDataTest(create, DW, DH, FUNC_HM(SimdHistogramMasked));

        return result;
    }

    bool AbsSecondDerivativeHistogramDataTest(bool create, int width, int height, const FuncASDH & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        size_t step = 1, indent = 16;
        Histogram h1, h2;

        if (create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, step, indent, h1);

            TEST_SAVE(h1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(h1);

            f.Call(src, step, indent, h2);

            TEST_SAVE(h2);

            result = result && Compare(h1, h2, 0, true, 32);
        }

        return result;
    }

    bool AbsSecondDerivativeHistogramDataTest(bool create)
    {
        bool result = true;

        result = result && AbsSecondDerivativeHistogramDataTest(create, DW, DH, FUNC_ASDH(SimdAbsSecondDerivativeHistogram));

        return result;
    }

    bool ChangeColorsDataTest(bool create, int width, int height, const FuncCC & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View colors(Simd::HISTOGRAM_SIZE, 1, View::Gray8, NULL, TEST_ALIGN(width));

        View dst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View dst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(src);

            FillRandom(colors);

            TEST_SAVE(src);

            TEST_SAVE(colors);

            f.Call(src, colors, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(colors);

            TEST_LOAD(dst1);

            f.Call(src, colors, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 0, true, 32);
        }

        return result;
    }

    bool ChangeColorsDataTest(bool create)
    {
        return ChangeColorsDataTest(create, DW, DH, FUNC_CC(SimdChangeColors));
    }

    bool HistogramConditionalDataTest(bool create, int width, int height, SimdCompareType type, const FuncHC & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View mask(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        const uint8_t value = 127;
        Histogram h1, h2;

        if (create)
        {
            FillRandom(src);
            FillRandom(mask);

            TEST_SAVE(src);
            TEST_SAVE(mask);

            f.Call(src, mask, value, type, h1);

            TEST_SAVE(h1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(mask);

            TEST_LOAD(h1);

            f.Call(src, mask, value, type, h2);

            TEST_SAVE(h2);

            result = result && Compare(h1, h2, 0, true, 32);
        }

        return result;
    }

    bool HistogramConditionalDataTest(bool create)
    {
        bool result = true;

        FuncHC f = FUNC_HC(SimdHistogramConditional);
        for (SimdCompareType type = SimdCompareEqual; type <= SimdCompareLesserOrEqual && result; type = SimdCompareType(type + 1))
        {
            result = result && HistogramConditionalDataTest(create, DW, DH, type, FuncHC(f.func, f.description + Data::Description(type)));
        }

        return result;
    }
}
