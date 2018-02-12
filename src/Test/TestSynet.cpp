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
        struct FuncAB
        {
            typedef void(*FuncPtr)(const float * bias, size_t count, size_t size, float * dst);

            FuncPtr func;
            String description;

            FuncAB(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & bias, size_t count, size_t size, const View & dstSrc, View & dstDst) const
            {
                Simd::Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)bias.data, count, size, (float*)dstDst.data);
            }
        };
    }

#define FUNC_AB(function) FuncAB(function, #function)

    bool SynetAddBiasAutoTest(size_t count, size_t size, const FuncAB & f1, const FuncAB & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << count << ", " << size << "].");

        View bias(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstSrc(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstDst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstDst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(bias, -10.0, 10.0);
        FillRandom32f(dstSrc, -10.0, 10.0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(bias, count, size, dstSrc, dstDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(bias, count, size, dstSrc, dstDst2));

        result = result && Compare(dstDst1, dstDst2, EPS, true, 32, false);

        return result;
    }

    bool SynetAddBiasAutoTest(const FuncAB & f1, const FuncAB & f2)
    {
        bool result = true;

        result = result && SynetAddBiasAutoTest(H, W, f1, f2);
        result = result && SynetAddBiasAutoTest(H - O, W + O, f1, f2);

        return result;
    }

    bool SynetAddBiasAutoTest()
    {
        bool result = true;

        result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Base::SynetAddBias), FUNC_AB(SimdSynetAddBias));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Sse::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Avx::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Avx512f::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool SynetAddBiasDataTest(bool create, size_t count, size_t size, const FuncAB & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << count << ", " << size << "].");

        View bias(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstSrc(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstDst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstDst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        if (create)
        {
            FillRandom32f(bias, -10.0, 10.0);
            FillRandom32f(dstSrc, -10.0, 10.0);

            TEST_SAVE(bias);
            TEST_SAVE(dstSrc);

            f.Call(bias, count, size, dstSrc, dstDst1);

            TEST_SAVE(dstDst1);
        }
        else
        {
            TEST_LOAD(bias);
            TEST_LOAD(dstSrc);

            TEST_LOAD(dstDst1);

            f.Call(bias, count, size, dstSrc, dstDst2);

            TEST_SAVE(dstDst2);

            result = result && Compare(dstDst1, dstDst2, EPS, true, 32, false);
        }

        return result;
    }

    bool SynetAddBiasDataTest(bool create)
    {
        return SynetAddBiasDataTest(create, DH, DW, FUNC_AB(SimdSynetAddBias));
    }
}
