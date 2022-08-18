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
        struct FuncS
        {
            typedef void(*FuncPtr)(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
                size_t width, size_t height, uint64_t * sum);

            FuncPtr func;
            String description;

            FuncS(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & a, const View & b, uint64_t * sum) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(a.data, a.stride, b.data, b.stride, a.width, a.height, sum);
            }
        };

        struct FuncM
        {
            typedef void(*FuncPtr)(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
                const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum);

            FuncPtr func;
            String description;

            FuncM(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & a, const View & b, const View & mask, uint8_t index, uint64_t * sum) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(a.data, a.stride, b.data, b.stride, mask.data, mask.stride, index, a.width, a.height, sum);
            }
        };

        struct FuncF
        {
            typedef void(*FuncPtr)(const float * a, const float * b, size_t size, float * sum);

            FuncPtr func;
            String description;

            FuncF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & a, const View & b, float * sum) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)a.data, (float*)b.data, a.width, sum);
            }
        };
    }

#define FUNC_S(function) FuncS(function, #function)
#define FUNC_M(function) FuncM(function, #function)
#define FUNC_F(function) FuncF(function, #function)

    bool DifferenceSumsAutoTest(int width, int height, const FuncS & f1, const FuncS & f2, int count)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(a);

        View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(b);

        Sums64 s1(count, 0), s2(count, 0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, s1.data()));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, s2.data()));

        result = Compare(s1, s2, 0, true, count);

        return result;
    }

    bool DifferenceSumsAutoTest(const FuncS & f1, const FuncS & f2, int count)
    {
        bool result = true;

        result = result && DifferenceSumsAutoTest(W, H, f1, f2, count);
        result = result && DifferenceSumsAutoTest(W + O, H - O, f1, f2, count);

        return result;
    }

    bool DifferenceSumsMaskedAutoTest(int width, int height, const FuncM & f1, const FuncM & f2, int count)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(a);

        View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(b);

        View m(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        uint8_t index = Random(256);
        FillRandomMask(m, index);

        Sums64 s1(count, 0), s2(count, 0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, m, index, s1.data()));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, m, index, s2.data()));

        result = Compare(s1, s2, 0, true, count);

        return result;
    }

    bool DifferenceSumsMaskedAutoTest(const FuncM & f1, const FuncM & f2, int count)
    {
        bool result = true;

        result = result && DifferenceSumsMaskedAutoTest(W, H, f1, f2, count);
        result = result && DifferenceSumsMaskedAutoTest(W + O, H - O, f1, f2, count);

        return result;
    }

    bool DifferenceSum32fAutoTest(int size, float eps, const FuncF & f1, const FuncF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View a(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(a);

        View b(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(b);

        float s1, s2;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, &s1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, &s2));

        result = Compare(s1, s2, eps, true, DifferenceBoth);

        return result;
    }

    bool DifferenceSum32fAutoTest(float eps, const FuncF & f1, const FuncF & f2)
    {
        bool result = true;

        result = result && DifferenceSum32fAutoTest(W*H, eps, f1, f2);
        result = result && DifferenceSum32fAutoTest(W*H + O, eps, f1, f2);

        return result;
    }

    bool SquaredDifferenceSumAutoTest()
    {
        bool result = true;

        result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Base::SquaredDifferenceSum), FUNC_S(SimdSquaredDifferenceSum), 1);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Sse41::SquaredDifferenceSum), FUNC_S(SimdSquaredDifferenceSum), 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Avx2::SquaredDifferenceSum), FUNC_S(SimdSquaredDifferenceSum), 1);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Avx512bw::SquaredDifferenceSum), FUNC_S(SimdSquaredDifferenceSum), 1);
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Vmx::SquaredDifferenceSum), FUNC_S(SimdSquaredDifferenceSum), 1);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Neon::SquaredDifferenceSum), FUNC_S(SimdSquaredDifferenceSum), 1);
#endif

        return result;
    }

    bool SquaredDifferenceSumMaskedAutoTest()
    {
        bool result = true;

        result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Base::SquaredDifferenceSumMasked), FUNC_M(SimdSquaredDifferenceSumMasked), 1);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Sse41::SquaredDifferenceSumMasked), FUNC_M(SimdSquaredDifferenceSumMasked), 1);
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Avx2::SquaredDifferenceSumMasked), FUNC_M(SimdSquaredDifferenceSumMasked), 1);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Avx512bw::SquaredDifferenceSumMasked), FUNC_M(SimdSquaredDifferenceSumMasked), 1);
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Vmx::SquaredDifferenceSumMasked), FUNC_M(SimdSquaredDifferenceSumMasked), 1);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Neon::SquaredDifferenceSumMasked), FUNC_M(SimdSquaredDifferenceSumMasked), 1);
#endif

        return result;
    }

    bool AbsDifferenceSumAutoTest()
    {
        bool result = true;

        result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Base::AbsDifferenceSum), FUNC_S(SimdAbsDifferenceSum), 1);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Sse41::AbsDifferenceSum), FUNC_S(SimdAbsDifferenceSum), 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Avx2::AbsDifferenceSum), FUNC_S(SimdAbsDifferenceSum), 1);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Avx512bw::AbsDifferenceSum), FUNC_S(SimdAbsDifferenceSum), 1);
#endif

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Vmx::AbsDifferenceSum), FUNC_S(SimdAbsDifferenceSum), 1);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Neon::AbsDifferenceSum), FUNC_S(SimdAbsDifferenceSum), 1);
#endif 

        return result;
    }

    bool AbsDifferenceSumMaskedAutoTest()
    {
        bool result = true;

        result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Base::AbsDifferenceSumMasked), FUNC_M(SimdAbsDifferenceSumMasked), 1);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Sse41::AbsDifferenceSumMasked), FUNC_M(SimdAbsDifferenceSumMasked), 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Avx2::AbsDifferenceSumMasked), FUNC_M(SimdAbsDifferenceSumMasked), 1);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Avx512bw::AbsDifferenceSumMasked), FUNC_M(SimdAbsDifferenceSumMasked), 1);
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Vmx::AbsDifferenceSumMasked), FUNC_M(SimdAbsDifferenceSumMasked), 1);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Neon::AbsDifferenceSumMasked), FUNC_M(SimdAbsDifferenceSumMasked), 1);
#endif 

        return result;
    }

    bool AbsDifferenceSums3x3AutoTest()
    {
        bool result = true;

        result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Base::AbsDifferenceSums3x3), FUNC_S(SimdAbsDifferenceSums3x3), 9);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A + 2)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Sse41::AbsDifferenceSums3x3), FUNC_S(SimdAbsDifferenceSums3x3), 9);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A + 2)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Avx2::AbsDifferenceSums3x3), FUNC_S(SimdAbsDifferenceSums3x3), 9);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Avx512bw::AbsDifferenceSums3x3), FUNC_S(SimdAbsDifferenceSums3x3), 9);
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A + 2)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Vmx::AbsDifferenceSums3x3), FUNC_S(SimdAbsDifferenceSums3x3), 9);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A + 2)
            result = result && DifferenceSumsAutoTest(FUNC_S(Simd::Neon::AbsDifferenceSums3x3), FUNC_S(SimdAbsDifferenceSums3x3), 9);
#endif

        return result;
    }

    bool AbsDifferenceSums3x3MaskedAutoTest()
    {
        bool result = true;

        result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Base::AbsDifferenceSums3x3Masked), FUNC_M(SimdAbsDifferenceSums3x3Masked), 9);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A + 2)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Sse41::AbsDifferenceSums3x3Masked), FUNC_M(SimdAbsDifferenceSums3x3Masked), 9);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A + 2)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Avx2::AbsDifferenceSums3x3Masked), FUNC_M(SimdAbsDifferenceSums3x3Masked), 9);
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Avx512bw::AbsDifferenceSums3x3Masked), FUNC_M(SimdAbsDifferenceSums3x3Masked), 9);
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A + 2)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Vmx::AbsDifferenceSums3x3Masked), FUNC_M(SimdAbsDifferenceSums3x3Masked), 9);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A + 2)
            result = result && DifferenceSumsMaskedAutoTest(FUNC_M(Simd::Neon::AbsDifferenceSums3x3Masked), FUNC_M(SimdAbsDifferenceSums3x3Masked), 9);
#endif 

        return result;
    }

    bool SquaredDifferenceSum32fAutoTest()
    {
        bool result = true;

        result = result && DifferenceSum32fAutoTest(EPS, FUNC_F(Simd::Base::SquaredDifferenceSum32f), FUNC_F(SimdSquaredDifferenceSum32f));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && DifferenceSum32fAutoTest(EPS, FUNC_F(Simd::Sse41::SquaredDifferenceSum32f), FUNC_F(SimdSquaredDifferenceSum32f));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && DifferenceSum32fAutoTest(EPS, FUNC_F(Simd::Avx::SquaredDifferenceSum32f), FUNC_F(SimdSquaredDifferenceSum32f));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DifferenceSum32fAutoTest(EPS, FUNC_F(Simd::Avx512bw::SquaredDifferenceSum32f), FUNC_F(SimdSquaredDifferenceSum32f));
#endif

#ifdef SIMD_VSX_ENABLE
        if (Simd::Vsx::Enable)
            result = result && DifferenceSum32fAutoTest(EPS, FUNC_F(Simd::Vsx::SquaredDifferenceSum32f), FUNC_F(SimdSquaredDifferenceSum32f));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && DifferenceSum32fAutoTest(EPS, FUNC_F(Simd::Neon::SquaredDifferenceSum32f), FUNC_F(SimdSquaredDifferenceSum32f));
#endif

        return result;
    }

    bool SquaredDifferenceKahanSum32fAutoTest()
    {
        bool result = true;

        result = result && DifferenceSum32fAutoTest(EPS*EPS, FUNC_F(Simd::Base::SquaredDifferenceKahanSum32f), FUNC_F(SimdSquaredDifferenceKahanSum32f));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && DifferenceSum32fAutoTest(EPS*EPS, FUNC_F(Simd::Sse41::SquaredDifferenceKahanSum32f), FUNC_F(SimdSquaredDifferenceKahanSum32f));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && DifferenceSum32fAutoTest(EPS*EPS, FUNC_F(Simd::Avx::SquaredDifferenceKahanSum32f), FUNC_F(SimdSquaredDifferenceKahanSum32f));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DifferenceSum32fAutoTest(EPS*EPS, FUNC_F(Simd::Avx512bw::SquaredDifferenceKahanSum32f), FUNC_F(SimdSquaredDifferenceKahanSum32f));
#endif

#ifdef SIMD_VSX_ENABLE
        if (Simd::Vsx::Enable)
            result = result && DifferenceSum32fAutoTest(EPS*EPS, FUNC_F(Simd::Vsx::SquaredDifferenceKahanSum32f), FUNC_F(SimdSquaredDifferenceKahanSum32f));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && DifferenceSum32fAutoTest(EPS*EPS, FUNC_F(Simd::Neon::SquaredDifferenceKahanSum32f), FUNC_F(SimdSquaredDifferenceKahanSum32f));
#endif 

        return result;
    }

    bool CosineDistance32fAutoTest()
    {
        bool result = true;

        result = result && DifferenceSum32fAutoTest(EPS, FUNC_F(Simd::Base::CosineDistance32f), FUNC_F(SimdCosineDistance32f));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && DifferenceSum32fAutoTest(EPS, FUNC_F(Simd::Sse41::CosineDistance32f), FUNC_F(SimdCosineDistance32f));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && DifferenceSum32fAutoTest(EPS, FUNC_F(Simd::Avx::CosineDistance32f), FUNC_F(SimdCosineDistance32f));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && DifferenceSum32fAutoTest(EPS, FUNC_F(Simd::Avx2::CosineDistance32f), FUNC_F(SimdCosineDistance32f));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DifferenceSum32fAutoTest(EPS, FUNC_F(Simd::Avx512bw::CosineDistance32f), FUNC_F(SimdCosineDistance32f));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && DifferenceSum32fAutoTest(EPS, FUNC_F(Simd::Neon::CosineDistance32f), FUNC_F(SimdCosineDistance32f));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool DifferenceSumsDataTest(bool create, int width, int height, const FuncS & f, int count)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        Sums64 s1(count, 0), s2(count, 0);

        if (create)
        {
            FillRandom(a);
            FillRandom(b);

            TEST_SAVE(a);
            TEST_SAVE(b);

            f.Call(a, b, s1.data());

            TEST_SAVE(s1);
        }
        else
        {
            TEST_LOAD(a);
            TEST_LOAD(b);

            TEST_LOAD(s1);

            f.Call(a, b, s2.data());

            TEST_SAVE(s2);

            result = result && Compare(s1, s2, 0, true, count);
        }

        return result;
    }

    bool AbsDifferenceSumDataTest(bool create)
    {
        bool result = true;

        result = result && DifferenceSumsDataTest(create, DW, DH, FUNC_S(SimdAbsDifferenceSum), 1);

        return result;
    }

    bool AbsDifferenceSums3x3DataTest(bool create)
    {
        bool result = true;

        result = result && DifferenceSumsDataTest(create, DW, DH, FUNC_S(SimdAbsDifferenceSums3x3), 9);

        return result;
    }

    bool SquaredDifferenceSumDataTest(bool create)
    {
        bool result = true;

        result = result && DifferenceSumsDataTest(create, DW, DH, FUNC_S(SimdSquaredDifferenceSum), 1);

        return result;
    }

    bool DifferenceSumsMaskedDataTest(bool create, int width, int height, const FuncM & f, int count)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View m(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        Sums64 s1(count, 0), s2(count, 0);

        uint8_t index = 17;

        if (create)
        {
            FillRandom(a);
            FillRandom(b);
            FillRandomMask(m, index);

            TEST_SAVE(a);
            TEST_SAVE(b);
            TEST_SAVE(m);

            f.Call(a, b, m, index, s1.data());

            TEST_SAVE(s1);
        }
        else
        {
            TEST_LOAD(a);
            TEST_LOAD(b);
            TEST_LOAD(m);

            TEST_LOAD(s1);

            f.Call(a, b, m, index, s2.data());

            TEST_SAVE(s2);

            result = result && Compare(s1, s2, 0, true, count);
        }

        return result;
    }

    bool AbsDifferenceSumMaskedDataTest(bool create)
    {
        bool result = true;

        result = result && DifferenceSumsMaskedDataTest(create, DW, DH, FUNC_M(SimdAbsDifferenceSumMasked), 1);

        return result;
    }

    bool AbsDifferenceSums3x3MaskedDataTest(bool create)
    {
        bool result = true;

        result = result && DifferenceSumsMaskedDataTest(create, DW, DH, FUNC_M(SimdAbsDifferenceSums3x3Masked), 9);

        return result;
    }

    bool SquaredDifferenceSumMaskedDataTest(bool create)
    {
        bool result = true;

        result = result && DifferenceSumsMaskedDataTest(create, DW, DH, FUNC_M(SimdSquaredDifferenceSumMasked), 1);

        return result;
    }

    bool DifferenceSum32fDataTest(bool create, int size, float eps, const FuncF & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View a(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View b(size, 1, View::Float, NULL, TEST_ALIGN(size));

        float s1, s2;

        if (create)
        {
            FillRandom32f(a);
            FillRandom32f(b);

            TEST_SAVE(a);
            TEST_SAVE(b);

            f.Call(a, b, &s1);

            TEST_SAVE(s1);
        }
        else
        {
            TEST_LOAD(a);
            TEST_LOAD(b);

            TEST_LOAD(s1);

            f.Call(a, b, &s2);

            TEST_SAVE(s2);

            result = result && Compare(s1, s2, eps, true);
        }

        return result;
    }

    bool SquaredDifferenceSum32fDataTest(bool create)
    {
        return DifferenceSum32fDataTest(create, DH, EPS, FUNC_F(SimdSquaredDifferenceSum32f));
    }

    bool SquaredDifferenceKahanSum32fDataTest(bool create)
    {
        return DifferenceSum32fDataTest(create, DH, (float)::pow(EPS, 1.5), FUNC_F(SimdSquaredDifferenceKahanSum32f));
    }

    bool CosineDistance32fDataTest(bool create)
    {
        return DifferenceSum32fDataTest(create, DH, EPS, FUNC_F(SimdCosineDistance32f));
    }
}
