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
            typedef void(*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride, SimdCompareType type);

            FuncPtr func;
            String description;

            Func1(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, uint8_t value, uint8_t positive, uint8_t negative, View & dst, SimdCompareType type) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, value, positive, negative, dst.data, dst.stride, type);
            }
        };
    }

#define ARGS1(width, height, type, function1, function2) \
    width, height, type, \
    Func1(function1.func, function1.description + CompareTypeDescription(type)), \
    Func1(function2.func, function2.description + CompareTypeDescription(type))

#define FUNC1(function) \
    Func1(function, std::string(#function))

    bool BinarizationAutoTest(int width, int height, SimdCompareType type, const Func1 & f1, const Func1 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(src);

        uint8_t value = Random(256);
        uint8_t positive = Random(256);
        uint8_t negative = Random(256);

        View d1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, value, positive, negative, d1, type));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, value, positive, negative, d2, type));

        result = result && Compare(d1, d2, 0, true, 32);

        return result;
    }

    bool BinarizationAutoTest(const Func1 & f1, const Func1 & f2)
    {
        bool result = true;

        for (SimdCompareType type = SimdCompareEqual; type <= SimdCompareLesserOrEqual && result; type = SimdCompareType(type + 1))
        {
            result = result && BinarizationAutoTest(ARGS1(W, H, type, f1, f2));
            result = result && BinarizationAutoTest(ARGS1(W + O, H - O, type, f1, f2));
            result = result && BinarizationAutoTest(ARGS1(W - O, H + O, type, f1, f2));
        }

        return result;
    }

    bool BinarizationAutoTest()
    {
        bool result = true;

        result = result && BinarizationAutoTest(FUNC1(Simd::Base::Binarization), FUNC1(SimdBinarization));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && BinarizationAutoTest(FUNC1(Simd::Sse2::Binarization), FUNC1(SimdBinarization));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && BinarizationAutoTest(FUNC1(Simd::Avx2::Binarization), FUNC1(SimdBinarization));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && BinarizationAutoTest(FUNC1(Simd::Avx512bw::Binarization), FUNC1(SimdBinarization));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && BinarizationAutoTest(FUNC1(Simd::Vmx::Binarization), FUNC1(SimdBinarization));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && BinarizationAutoTest(FUNC1(Simd::Neon::Binarization), FUNC1(SimdBinarization));
#endif

        return result;
    }

    namespace
    {
        struct Func2
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative,
                uint8_t * dst, size_t dstStride, SimdCompareType type);

            FuncPtr func;
            String description;

            Func2(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative, View & dst, SimdCompareType type) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, value, neighborhood, threshold, positive, negative, dst.data, dst.stride, type);
            }
        };
    }

#define ARGS2(width, height, type, function1, function2) \
    width, height, type, \
    Func2(function1.func, function1.description + CompareTypeDescription(type)), \
    Func2(function2.func, function2.description + CompareTypeDescription(type))

#define FUNC2(function) \
    Func2(function, std::string(#function))

    bool AveragingBinarizationAutoTest(int width, int height, SimdCompareType type, const Func2 & f1, const Func2 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(src);

        uint8_t value = 127;
        size_t neighborhood = 17;
        uint8_t threshold = 128;
        uint8_t positive = 7;
        uint8_t negative = 3;

        View d1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, value, neighborhood, threshold, positive, negative, d1, type));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, value, neighborhood, threshold, positive, negative, d2, type));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool AveragingBinarizationAutoTest(const Func2 & f1, const Func2 & f2)
    {
        bool result = true;

        for (SimdCompareType type = SimdCompareEqual; type <= SimdCompareLesserOrEqual && result; type = SimdCompareType(type + 1))
        {
            result = result && AveragingBinarizationAutoTest(ARGS2(W, H, type, f1, f2));
            result = result && AveragingBinarizationAutoTest(ARGS2(W + O, H - O, type, f1, f2));
            result = result && AveragingBinarizationAutoTest(ARGS2(W - O, H + O, type, f1, f2));
        }

        return result;
    }

    bool AveragingBinarizationAutoTest()
    {
        bool result = true;

        result = result && AveragingBinarizationAutoTest(FUNC2(Simd::Base::AveragingBinarization), FUNC2(SimdAveragingBinarization));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && AveragingBinarizationAutoTest(FUNC2(Simd::Sse2::AveragingBinarization), FUNC2(SimdAveragingBinarization));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && AveragingBinarizationAutoTest(FUNC2(Simd::Avx2::AveragingBinarization), FUNC2(SimdAveragingBinarization));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && AveragingBinarizationAutoTest(FUNC2(Simd::Avx512bw::AveragingBinarization), FUNC2(SimdAveragingBinarization));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && AveragingBinarizationAutoTest(FUNC2(Simd::Vmx::AveragingBinarization), FUNC2(SimdAveragingBinarization));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && AveragingBinarizationAutoTest(FUNC2(Simd::Neon::AveragingBinarization), FUNC2(SimdAveragingBinarization));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool BinarizationDataTest(bool create, int width, int height, SimdCompareType type, const Func1 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View dst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View dst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        const uint8_t value = 127;
        const uint8_t positive = 0xAA;
        const uint8_t negative = 0x11;

        if (create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, value, positive, negative, dst1, type);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, value, positive, negative, dst2, type);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 0, true, 64);
        }

        return result;
    }

    bool BinarizationDataTest(bool create)
    {
        bool result = true;

        Func1 f = FUNC1(SimdBinarization);
        for (SimdCompareType type = SimdCompareEqual; type <= SimdCompareLesserOrEqual && result; type = SimdCompareType(type + 1))
        {
            result = result && BinarizationDataTest(create, DW, DH, type, Func1(f.func, f.description + Data::Description(type)));
        }

        return result;
    }

    bool AveragingBinarizationDataTest(bool create, int width, int height, SimdCompareType type, const Func2 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View dst1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View dst2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        const uint8_t value = 127;
        const size_t neighborhood = 17;
        const uint8_t threshold = 128;
        const uint8_t positive = 7;
        const uint8_t negative = 3;

        if (create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, value, neighborhood, threshold, positive, negative, dst1, type);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, value, neighborhood, threshold, positive, negative, dst2, type);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 0, true, 64);
        }

        return result;
    }

    bool AveragingBinarizationDataTest(bool create)
    {
        bool result = true;

        Func2 f = FUNC2(SimdAveragingBinarization);
        for (SimdCompareType type = SimdCompareEqual; type <= SimdCompareLesserOrEqual && result; type = SimdCompareType(type + 1))
        {
            result = result && AveragingBinarizationDataTest(create, DW, DH, type, Func2(f.func, f.description + Data::Description(type)));
        }

        return result;
    }
}
