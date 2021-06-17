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

namespace Test
{
    namespace
    {
        struct FuncSL
        {
            typedef void(*FuncPtr)(const float * x, const float * svs, const float * weights, size_t length, size_t count, float * sum);

            FuncPtr func;
            String description;

            FuncSL(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & x, const View & svs, const View & weights, size_t length, size_t count, float * sum) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)x.data, (float*)svs.data, (float*)weights.data, length, count, sum);
            }
        };
    }

#define FUNC_SL(function) FuncSL(function, #function)

    bool SvmSumLinearAutoTest(size_t length, size_t count, const FuncSL & f1, const FuncSL & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << length << ", " << count << "].");

        View svs(length*count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View weights(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View x(length, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(svs, -10.0, 10.0);
        FillRandom32f(weights, -10.0, 10.0);
        FillRandom32f(x, -10.0, 10.0);

        float s1, s2;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(x, svs, weights, length, count, &s1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(x, svs, weights, length, count, &s2));

        result = result && Compare(s1, s2, EPS, true);

        return result;
    }

    bool SvmSumLinearAutoTest(const FuncSL & f1, const FuncSL & f2)
    {
        bool result = true;

        result = result && SvmSumLinearAutoTest(W / 9, H * 9, f1, f2);
        result = result && SvmSumLinearAutoTest(W / 10, H * 10, f1, f2);
        result = result && SvmSumLinearAutoTest(W / 11, H * 11, f1, f2);

        return result;
    }

    bool SvmSumLinearAutoTest()
    {
        bool result = true;

        result = result && SvmSumLinearAutoTest(FUNC_SL(Simd::Base::SvmSumLinear), FUNC_SL(SimdSvmSumLinear));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SvmSumLinearAutoTest(FUNC_SL(Simd::Sse2::SvmSumLinear), FUNC_SL(SimdSvmSumLinear));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SvmSumLinearAutoTest(FUNC_SL(Simd::Avx::SvmSumLinear), FUNC_SL(SimdSvmSumLinear));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SvmSumLinearAutoTest(FUNC_SL(Simd::Avx512f::SvmSumLinear), FUNC_SL(SimdSvmSumLinear));
#endif

#ifdef SIMD_VSX_ENABLE
        if (Simd::Vsx::Enable)
            result = result && SvmSumLinearAutoTest(FUNC_SL(Simd::Vsx::SvmSumLinear), FUNC_SL(SimdSvmSumLinear));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SvmSumLinearAutoTest(FUNC_SL(Simd::Neon::SvmSumLinear), FUNC_SL(SimdSvmSumLinear));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool SvmSumLinearDataTest(bool create, size_t length, size_t count, const FuncSL & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << length << ", " << count << "].");

        View svs(length*count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View weights(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View x(length, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        float s1, s2;

        if (create)
        {
            FillRandom32f(svs, -10.0, 10.0);
            FillRandom32f(weights, -10.0, 10.0);
            FillRandom32f(x, -10.0, 10.0);

            TEST_SAVE(svs);
            TEST_SAVE(weights);
            TEST_SAVE(x);

            f.Call(x, svs, weights, length, count, &s1);

            TEST_SAVE(s1);
        }
        else
        {
            TEST_LOAD(svs);
            TEST_LOAD(weights);
            TEST_LOAD(x);

            TEST_LOAD(s1);

            f.Call(x, svs, weights, length, count, &s2);

            TEST_SAVE(s2);

            result = result && Compare(s1, s2, EPS, true);
        }

        return result;
    }

    bool SvmSumLinearDataTest(bool create)
    {
        bool result = true;

        result = result && SvmSumLinearDataTest(create, DW, DH, FUNC_SL(SimdSvmSumLinear));

        return result;
    }
}
