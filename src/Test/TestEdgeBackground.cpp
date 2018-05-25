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
        struct Func1
        {
            typedef void(*FuncPtr)(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                uint8_t * background, size_t backgroundStride);

            FuncPtr func;
            String description;

            Func1(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & value, const View & backgroundSrc, View & backgroundDst) const
            {
                Simd::Copy(backgroundSrc, backgroundDst);
                TEST_PERFORMANCE_TEST(description);
                func(value.data, value.stride, value.width, value.height, backgroundDst.data, backgroundDst.stride);
            }
        };
    }

#define FUNC1(function) Func1(function, std::string(#function))

    bool EdgeBackgroundChangeRangeAutoTest(int width, int height, const Func1 & f1, const Func1 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View value(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(value);
        View backgroundSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(backgroundSrc);

        View backgroundDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(value, backgroundSrc, backgroundDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(value, backgroundSrc, backgroundDst2));

        result = result && Compare(backgroundDst1, backgroundDst2, 0, true, 32, 0);

        return result;
    }

    bool EdgeBackgroundChangeRangeAutoTest(const Func1 & f1, const Func1 & f2)
    {
        bool result = true;

        result = result && EdgeBackgroundChangeRangeAutoTest(W, H, f1, f2);
        result = result && EdgeBackgroundChangeRangeAutoTest(W + O, H - O, f1, f2);

        return result;
    }

    namespace
    {
        struct Func2
        {
            typedef void(*FuncPtr)(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                const uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t * backgroundCount, size_t backgroundCountStride);

            FuncPtr func;
            String description;

            Func2(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & value, const View & backgroundValue,
                const View & backgroundCountSrc, View & backgroundCountDst) const
            {
                Simd::Copy(backgroundCountSrc, backgroundCountDst);
                TEST_PERFORMANCE_TEST(description);
                func(value.data, value.stride, value.width, value.height,
                    backgroundValue.data, backgroundValue.stride, backgroundCountDst.data, backgroundCountDst.stride);
            }
        };
    }

#define FUNC2(function) Func2(function, std::string(#function))

    bool EdgeBackgroundIncrementCountAutoTest(int width, int height, const Func2 & f1, const Func2 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View value(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(value);
        View backgroundValue(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(backgroundValue);
        View backgroundCountSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(backgroundCountSrc);

        View backgroundCountDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundCountDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(value, backgroundValue, backgroundCountSrc, backgroundCountDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(value, backgroundValue, backgroundCountSrc, backgroundCountDst2));

        result = result && Compare(backgroundCountDst1, backgroundCountDst2, 0, true, 32, 0);

        return result;
    }

    bool EdgeBackgroundIncrementCountAutoTest(const Func2 & f1, const Func2 & f2)
    {
        bool result = true;

        result = result && EdgeBackgroundIncrementCountAutoTest(W, H, f1, f2);
        result = result && EdgeBackgroundIncrementCountAutoTest(W + O, H - O, f1, f2);

        return result;
    }

    namespace
    {
        struct Func3
        {
            typedef void(*FuncPtr)(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
                uint8_t * backgroundalue, size_t backgroundValueStride, uint8_t threshold);

            FuncPtr func;
            String description;

            Func3(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & backgroundCountSrc, const View & backgroundValueSrc,
                View & backgroundCountDst, View & backgroundValueDst, uint8_t threshold) const
            {
                Simd::Copy(backgroundCountSrc, backgroundCountDst);
                Simd::Copy(backgroundValueSrc, backgroundValueDst);
                TEST_PERFORMANCE_TEST(description);
                func(backgroundCountDst.data, backgroundCountDst.stride, backgroundValueDst.width, backgroundValueDst.height,
                    backgroundValueDst.data, backgroundValueDst.stride, threshold);
            }
        };
    }

#define FUNC3(function) Func3(function, std::string(#function))

    bool EdgeBackgroundAdjustRangeAutoTest(int width, int height, const Func3 & f1, const Func3 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View backgroundCountSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(backgroundCountSrc);
        View backgroundValueSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(backgroundValueSrc);

        View backgroundCountDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundValueDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundCountDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundValueDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(backgroundCountSrc, backgroundValueSrc, backgroundCountDst1, backgroundValueDst1, 0x80));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(backgroundCountSrc, backgroundValueSrc, backgroundCountDst2, backgroundValueDst2, 0x80));

        result = result && Compare(backgroundCountDst1, backgroundCountDst2, 0, true, 32, 0, "backgroundCount");
        result = result && Compare(backgroundValueDst1, backgroundValueDst2, 0, true, 32, 0, "backgroundValue");

        return result;
    }

    bool EdgeBackgroundAdjustRangeAutoTest(const Func3 & f1, const Func3 & f2)
    {
        bool result = true;

        result = result && EdgeBackgroundAdjustRangeAutoTest(W, H, f1, f2);
        result = result && EdgeBackgroundAdjustRangeAutoTest(W + O, H - O, f1, f2);

        return result;
    }

    namespace
    {
        struct Func4
        {
            typedef void(*FuncPtr)(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
                uint8_t * backgroundalue, size_t backgroundValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride);

            FuncPtr func;
            String description;

            Func4(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & backgroundCountSrc, const View & backgroundValueSrc,
                View & backgroundCountDst, View & backgroundValueDst, uint8_t threshold, const View & mask) const
            {
                Simd::Copy(backgroundCountSrc, backgroundCountDst);
                Simd::Copy(backgroundValueSrc, backgroundValueDst);
                TEST_PERFORMANCE_TEST(description);
                func(backgroundCountDst.data, backgroundCountDst.stride, backgroundValueDst.width, backgroundValueDst.height,
                    backgroundValueDst.data, backgroundValueDst.stride, threshold, mask.data, mask.stride);
            }
        };
    }

#define FUNC4(function) Func4(function, std::string(#function))

    bool EdgeBackgroundAdjustRangeMaskedAutoTest(int width, int height, const Func4 & f1, const Func4 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View backgroundCountSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(backgroundCountSrc);
        View backgroundValueSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(backgroundValueSrc);
        View mask(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandomMask(mask, 0xFF);

        View backgroundCountDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundValueDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundCountDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundValueDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(backgroundCountSrc, backgroundValueSrc, backgroundCountDst1, backgroundValueDst1, 0x80, mask));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(backgroundCountSrc, backgroundValueSrc, backgroundCountDst2, backgroundValueDst2, 0x80, mask));

        result = result && Compare(backgroundCountDst1, backgroundCountDst2, 0, true, 32, 0, "backgroundCount");
        result = result && Compare(backgroundValueDst1, backgroundValueDst2, 0, true, 32, 0, "backgroundValue");

        return result;
    }

    bool EdgeBackgroundAdjustRangeMaskedAutoTest(const Func4 & f1, const Func4 & f2)
    {
        bool result = true;

        result = result && EdgeBackgroundAdjustRangeMaskedAutoTest(W, H, f1, f2);
        result = result && EdgeBackgroundAdjustRangeMaskedAutoTest(W + O, H - O, f1, f2);

        return result;
    }

    namespace
    {
        struct Func5
        {
            typedef void(*FuncPtr)(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                uint8_t * background, size_t backgroundStride, const uint8_t * mask, size_t maskStride);

            FuncPtr func;
            String description;

            Func5(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & value, const View & backgroundSrc, View & backgroundDst, const View & mask) const
            {
                Simd::Copy(backgroundSrc, backgroundDst);
                TEST_PERFORMANCE_TEST(description);
                func(value.data, value.stride, value.width, value.height, backgroundDst.data, backgroundDst.stride, mask.data, mask.stride);
            }
        };
    }

#define FUNC5(function) Func5(function, std::string(#function))

    bool EdgeBackgroundShiftRangeMaskedAutoTest(int width, int height, const Func5 & f1, const Func5 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View value(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(value);
        View backgroundSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(backgroundSrc);
        View mask(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandomMask(mask, 0xFF);

        View backgroundDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(value, backgroundSrc, backgroundDst1, mask));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(value, backgroundSrc, backgroundDst2, mask));

        result = result && Compare(backgroundDst1, backgroundDst2, 0, true, 32, 0);

        return result;
    }

    bool EdgeBackgroundShiftRangeMaskedAutoTest(const Func5 & f1, const Func5 & f2)
    {
        bool result = true;

        result = result && EdgeBackgroundShiftRangeMaskedAutoTest(W, H, f1, f2);
        result = result && EdgeBackgroundShiftRangeMaskedAutoTest(W + O, H - O, f1, f2);

        return result;
    }

    bool EdgeBackgroundGrowRangeSlowAutoTest()
    {
        bool result = true;

        result = result && EdgeBackgroundChangeRangeAutoTest(FUNC1(Simd::Base::EdgeBackgroundGrowRangeSlow), FUNC1(SimdEdgeBackgroundGrowRangeSlow));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A)
            result = result && EdgeBackgroundChangeRangeAutoTest(FUNC1(Simd::Sse2::EdgeBackgroundGrowRangeSlow), FUNC1(SimdEdgeBackgroundGrowRangeSlow));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && EdgeBackgroundChangeRangeAutoTest(FUNC1(Simd::Avx2::EdgeBackgroundGrowRangeSlow), FUNC1(SimdEdgeBackgroundGrowRangeSlow));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && EdgeBackgroundChangeRangeAutoTest(FUNC1(Simd::Avx512bw::EdgeBackgroundGrowRangeSlow), FUNC1(SimdEdgeBackgroundGrowRangeSlow));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && EdgeBackgroundChangeRangeAutoTest(FUNC1(Simd::Vmx::EdgeBackgroundGrowRangeSlow), FUNC1(SimdEdgeBackgroundGrowRangeSlow));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && EdgeBackgroundChangeRangeAutoTest(FUNC1(Simd::Neon::EdgeBackgroundGrowRangeSlow), FUNC1(SimdEdgeBackgroundGrowRangeSlow));
#endif

        return result;
    }

    bool EdgeBackgroundGrowRangeFastAutoTest()
    {
        bool result = true;

        result = result && EdgeBackgroundChangeRangeAutoTest(FUNC1(Simd::Base::EdgeBackgroundGrowRangeFast), FUNC1(SimdEdgeBackgroundGrowRangeFast));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A)
            result = result && EdgeBackgroundChangeRangeAutoTest(FUNC1(Simd::Sse2::EdgeBackgroundGrowRangeFast), FUNC1(SimdEdgeBackgroundGrowRangeFast));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && EdgeBackgroundChangeRangeAutoTest(FUNC1(Simd::Avx2::EdgeBackgroundGrowRangeFast), FUNC1(SimdEdgeBackgroundGrowRangeFast));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && EdgeBackgroundChangeRangeAutoTest(FUNC1(Simd::Avx512bw::EdgeBackgroundGrowRangeFast), FUNC1(SimdEdgeBackgroundGrowRangeFast));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && EdgeBackgroundChangeRangeAutoTest(FUNC1(Simd::Vmx::EdgeBackgroundGrowRangeFast), FUNC1(SimdEdgeBackgroundGrowRangeFast));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && EdgeBackgroundChangeRangeAutoTest(FUNC1(Simd::Neon::EdgeBackgroundGrowRangeFast), FUNC1(SimdEdgeBackgroundGrowRangeFast));
#endif

        return result;
    }

    bool EdgeBackgroundIncrementCountAutoTest()
    {
        bool result = true;

        result = result && EdgeBackgroundIncrementCountAutoTest(FUNC2(Simd::Base::EdgeBackgroundIncrementCount), FUNC2(SimdEdgeBackgroundIncrementCount));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A)
            result = result && EdgeBackgroundIncrementCountAutoTest(FUNC2(Simd::Sse2::EdgeBackgroundIncrementCount), FUNC2(SimdEdgeBackgroundIncrementCount));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && EdgeBackgroundIncrementCountAutoTest(FUNC2(Simd::Avx2::EdgeBackgroundIncrementCount), FUNC2(SimdEdgeBackgroundIncrementCount));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && EdgeBackgroundIncrementCountAutoTest(FUNC2(Simd::Avx512bw::EdgeBackgroundIncrementCount), FUNC2(SimdEdgeBackgroundIncrementCount));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && EdgeBackgroundIncrementCountAutoTest(FUNC2(Simd::Vmx::EdgeBackgroundIncrementCount), FUNC2(SimdEdgeBackgroundIncrementCount));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && EdgeBackgroundIncrementCountAutoTest(FUNC2(Simd::Neon::EdgeBackgroundIncrementCount), FUNC2(SimdEdgeBackgroundIncrementCount));
#endif 

        return result;
    }

    bool EdgeBackgroundAdjustRangeAutoTest()
    {
        bool result = true;

        result = result && EdgeBackgroundAdjustRangeAutoTest(FUNC3(Simd::Base::EdgeBackgroundAdjustRange), FUNC3(SimdEdgeBackgroundAdjustRange));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A)
            result = result && EdgeBackgroundAdjustRangeAutoTest(FUNC3(Simd::Sse2::EdgeBackgroundAdjustRange), FUNC3(SimdEdgeBackgroundAdjustRange));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && EdgeBackgroundAdjustRangeAutoTest(FUNC3(Simd::Avx2::EdgeBackgroundAdjustRange), FUNC3(SimdEdgeBackgroundAdjustRange));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && EdgeBackgroundAdjustRangeAutoTest(FUNC3(Simd::Avx512bw::EdgeBackgroundAdjustRange), FUNC3(SimdEdgeBackgroundAdjustRange));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && EdgeBackgroundAdjustRangeAutoTest(FUNC3(Simd::Vmx::EdgeBackgroundAdjustRange), FUNC3(SimdEdgeBackgroundAdjustRange));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && EdgeBackgroundAdjustRangeAutoTest(FUNC3(Simd::Neon::EdgeBackgroundAdjustRange), FUNC3(SimdEdgeBackgroundAdjustRange));
#endif 

        return result;
    }

    bool EdgeBackgroundAdjustRangeMaskedAutoTest()
    {
        bool result = true;

        result = result && EdgeBackgroundAdjustRangeMaskedAutoTest(FUNC4(Simd::Base::EdgeBackgroundAdjustRangeMasked), FUNC4(SimdEdgeBackgroundAdjustRangeMasked));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A)
            result = result && EdgeBackgroundAdjustRangeMaskedAutoTest(FUNC4(Simd::Sse2::EdgeBackgroundAdjustRangeMasked), FUNC4(SimdEdgeBackgroundAdjustRangeMasked));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && EdgeBackgroundAdjustRangeMaskedAutoTest(FUNC4(Simd::Avx2::EdgeBackgroundAdjustRangeMasked), FUNC4(SimdEdgeBackgroundAdjustRangeMasked));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && EdgeBackgroundAdjustRangeMaskedAutoTest(FUNC4(Simd::Avx512bw::EdgeBackgroundAdjustRangeMasked), FUNC4(SimdEdgeBackgroundAdjustRangeMasked));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && EdgeBackgroundAdjustRangeMaskedAutoTest(FUNC4(Simd::Vmx::EdgeBackgroundAdjustRangeMasked), FUNC4(SimdEdgeBackgroundAdjustRangeMasked));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && EdgeBackgroundAdjustRangeMaskedAutoTest(FUNC4(Simd::Neon::EdgeBackgroundAdjustRangeMasked), FUNC4(SimdEdgeBackgroundAdjustRangeMasked));
#endif 

        return result;
    }

    bool EdgeBackgroundShiftRangeAutoTest()
    {
        bool result = true;

        result = result && EdgeBackgroundChangeRangeAutoTest(FUNC1(Simd::Base::EdgeBackgroundShiftRange), FUNC1(SimdEdgeBackgroundShiftRange));

        return result;
    }

    bool EdgeBackgroundShiftRangeMaskedAutoTest()
    {
        bool result = true;

        result = result && EdgeBackgroundShiftRangeMaskedAutoTest(FUNC5(Simd::Base::EdgeBackgroundShiftRangeMasked), FUNC5(SimdEdgeBackgroundShiftRangeMasked));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A)
            result = result && EdgeBackgroundShiftRangeMaskedAutoTest(FUNC5(Simd::Sse2::EdgeBackgroundShiftRangeMasked), FUNC5(SimdEdgeBackgroundShiftRangeMasked));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && EdgeBackgroundShiftRangeMaskedAutoTest(FUNC5(Simd::Avx2::EdgeBackgroundShiftRangeMasked), FUNC5(SimdEdgeBackgroundShiftRangeMasked));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && EdgeBackgroundShiftRangeMaskedAutoTest(FUNC5(Simd::Avx512bw::EdgeBackgroundShiftRangeMasked), FUNC5(SimdEdgeBackgroundShiftRangeMasked));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && EdgeBackgroundShiftRangeMaskedAutoTest(FUNC5(Simd::Vmx::EdgeBackgroundShiftRangeMasked), FUNC5(SimdEdgeBackgroundShiftRangeMasked));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && EdgeBackgroundShiftRangeMaskedAutoTest(FUNC5(Simd::Neon::EdgeBackgroundShiftRangeMasked), FUNC5(SimdEdgeBackgroundShiftRangeMasked));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool EdgeBackgroundChangeRangeDataTest(bool create, int width, int height, const Func1 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View value(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View backgroundDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(value);
            FillRandom(backgroundSrc);

            TEST_SAVE(value);
            TEST_SAVE(backgroundSrc);

            f.Call(value, backgroundSrc, backgroundDst1);

            TEST_SAVE(backgroundDst1);
        }
        else
        {
            TEST_LOAD(value);
            TEST_LOAD(backgroundSrc);

            TEST_LOAD(backgroundDst1);

            f.Call(value, backgroundSrc, backgroundDst2);

            TEST_SAVE(backgroundDst2);

            result = result && Compare(backgroundDst1, backgroundDst2, 0, true, 32, 0);
        }

        return result;
    }

    bool EdgeBackgroundGrowRangeSlowDataTest(bool create)
    {
        bool result = true;

        result = result && EdgeBackgroundChangeRangeDataTest(create, DW, DH, FUNC1(SimdEdgeBackgroundGrowRangeSlow));

        return result;
    }

    bool EdgeBackgroundGrowRangeFastDataTest(bool create)
    {
        bool result = true;

        result = result && EdgeBackgroundChangeRangeDataTest(create, DW, DH, FUNC1(SimdEdgeBackgroundGrowRangeFast));

        return result;
    }

    bool EdgeBackgroundShiftRangeDataTest(bool create)
    {
        bool result = true;

        result = result && EdgeBackgroundChangeRangeDataTest(create, DW, DH, FUNC1(SimdEdgeBackgroundShiftRange));

        return result;
    }

    bool EdgeBackgroundIncrementCountDataTest(bool create, int width, int height, const Func2 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View value(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundValue(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundCountSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View backgroundCountDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundCountDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(value);
            FillRandom(backgroundValue);
            FillRandom(backgroundCountSrc);

            TEST_SAVE(value);
            TEST_SAVE(backgroundValue);
            TEST_SAVE(backgroundCountSrc);

            f.Call(value, backgroundValue, backgroundCountSrc, backgroundCountDst1);

            TEST_SAVE(backgroundCountDst1);
        }
        else
        {
            TEST_LOAD(value);
            TEST_LOAD(backgroundValue);
            TEST_LOAD(backgroundCountSrc);

            TEST_LOAD(backgroundCountDst1);

            f.Call(value, backgroundValue, backgroundCountSrc, backgroundCountDst2);

            TEST_SAVE(backgroundCountDst2);

            result = result && Compare(backgroundCountDst1, backgroundCountDst2, 0, true, 32, 0);
        }

        return result;
    }

    bool EdgeBackgroundIncrementCountDataTest(bool create)
    {
        bool result = true;

        result = result && EdgeBackgroundIncrementCountDataTest(create, DW, DH, FUNC2(SimdEdgeBackgroundIncrementCount));

        return result;
    }

    bool EdgeBackgroundAdjustRangeDataTest(bool create, int width, int height, const Func3 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View backgroundCountSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundValueSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View backgroundCountDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundValueDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundCountDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundValueDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(backgroundCountSrc);
            FillRandom(backgroundValueSrc);

            TEST_SAVE(backgroundCountSrc);
            TEST_SAVE(backgroundValueSrc);

            f.Call(backgroundCountSrc, backgroundValueSrc, backgroundCountDst1, backgroundValueDst1, 0x80);

            TEST_SAVE(backgroundCountDst1);
            TEST_SAVE(backgroundValueDst1);
        }
        else
        {
            TEST_LOAD(backgroundCountSrc);
            TEST_LOAD(backgroundValueSrc);

            TEST_LOAD(backgroundCountDst1);
            TEST_LOAD(backgroundValueDst1);

            f.Call(backgroundCountSrc, backgroundValueSrc, backgroundCountDst2, backgroundValueDst2, 0x80);

            TEST_SAVE(backgroundCountDst2);
            TEST_SAVE(backgroundValueDst2);

            result = result && Compare(backgroundCountDst1, backgroundCountDst2, 0, true, 32, 0, "backgroundCount");
            result = result && Compare(backgroundValueDst1, backgroundValueDst2, 0, true, 32, 0, "backgroundValue");
        }

        return result;
    }

    bool EdgeBackgroundAdjustRangeDataTest(bool create)
    {
        bool result = true;

        result = result && EdgeBackgroundAdjustRangeDataTest(create, DW, DH, FUNC3(SimdEdgeBackgroundAdjustRange));

        return result;
    }

    bool EdgeBackgroundAdjustRangeMaskedDataTest(bool create, int width, int height, const Func4 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View backgroundCountSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundValueSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View mask(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View backgroundCountDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundValueDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundCountDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundValueDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(backgroundCountSrc);
            FillRandom(backgroundValueSrc);
            FillRandomMask(mask, 0xFF);

            TEST_SAVE(backgroundCountSrc);
            TEST_SAVE(backgroundValueSrc);
            TEST_SAVE(mask);

            f.Call(backgroundCountSrc, backgroundValueSrc, backgroundCountDst1, backgroundValueDst1, 0x80, mask);

            TEST_SAVE(backgroundCountDst1);
            TEST_SAVE(backgroundValueDst1);
        }
        else
        {
            TEST_LOAD(backgroundCountSrc);
            TEST_LOAD(backgroundValueSrc);
            TEST_LOAD(mask);

            TEST_LOAD(backgroundCountDst1);
            TEST_LOAD(backgroundValueDst1);

            f.Call(backgroundCountSrc, backgroundValueSrc, backgroundCountDst2, backgroundValueDst2, 0x80, mask);

            TEST_SAVE(backgroundCountDst2);
            TEST_SAVE(backgroundValueDst2);

            result = result && Compare(backgroundCountDst1, backgroundCountDst2, 0, true, 32, 0, "backgroundCount");
            result = result && Compare(backgroundValueDst1, backgroundValueDst2, 0, true, 32, 0, "backgroundValue");
        }

        return result;
    }

    bool EdgeBackgroundAdjustRangeMaskedDataTest(bool create)
    {
        bool result = true;

        result = result && EdgeBackgroundAdjustRangeMaskedDataTest(create, DW, DH, FUNC4(SimdEdgeBackgroundAdjustRangeMasked));

        return result;
    }

    bool EdgeBackgroundShiftRangeMaskedDataTest(bool create, int width, int height, const Func5 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View value(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundSrc(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View mask(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View backgroundDst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View backgroundDst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(value);
            FillRandom(backgroundSrc);
            FillRandomMask(mask, 0xFF);

            TEST_SAVE(value);
            TEST_SAVE(backgroundSrc);
            TEST_SAVE(mask);

            f.Call(value, backgroundSrc, backgroundDst1, mask);

            TEST_SAVE(backgroundDst1);
        }
        else
        {
            TEST_LOAD(value);
            TEST_LOAD(backgroundSrc);
            TEST_LOAD(mask);

            TEST_LOAD(backgroundDst1);

            f.Call(value, backgroundSrc, backgroundDst2, mask);

            TEST_SAVE(backgroundDst2);

            result = result && Compare(backgroundDst1, backgroundDst2, 0, true, 32, 0);
        }

        return result;
    }

    bool EdgeBackgroundShiftRangeMaskedDataTest(bool create)
    {
        bool result = true;

        result = result && EdgeBackgroundShiftRangeMaskedDataTest(create, DW, DH, FUNC5(SimdEdgeBackgroundShiftRangeMasked));

        return result;
    }
}
