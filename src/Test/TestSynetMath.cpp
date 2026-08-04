/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestRandom.h"
#include "Test/TestOptions.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncAVMV
        {
            typedef void(*FuncPtr)(const float * src, size_t size, const float * value, float * dst);

            FuncPtr func;
            String description;

            FuncAVMV(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, float value, const View & dstSrc, View & dstDst) const
            {
                Simd::Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.width, &value, (float*)dstDst.data);
            }
        };
    }
#define FUNC_AVMV(function) FuncAVMV(function, #function)

    bool SynetAddVectorMultipliedByValueAutoTest(int size, float eps, const FuncAVMV & f1, const FuncAVMV & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(src);

        View dstSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(dstSrc);

        const float value = 0.3f;

        View dstDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, value, dstSrc, dstDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, value, dstSrc, dstDst2));

        result = Compare(dstDst1, dstDst2, eps, true);

        return result;
    }

    bool SynetAddVectorMultipliedByValueAutoTest(float eps, const FuncAVMV & f1, const FuncAVMV & f2)
    {
        bool result = true;

        result = result && SynetAddVectorMultipliedByValueAutoTest(W*H, eps, f1, f2);
        result = result && SynetAddVectorMultipliedByValueAutoTest(W*H + O, eps, f1, f2);
        result = result && SynetAddVectorMultipliedByValueAutoTest(W*H - O, eps, f1, f2);

        return result;
    }

    bool SynetAddVectorMultipliedByValueAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetAddVectorMultipliedByValueAutoTest(EPS, FUNC_AVMV(Simd::Base::SynetAddVectorMultipliedByValue), FUNC_AVMV(SimdSynetAddVectorMultipliedByValue));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetAddVectorMultipliedByValueAutoTest(EPS, FUNC_AVMV(Simd::Sse41::SynetAddVectorMultipliedByValue), FUNC_AVMV(SimdSynetAddVectorMultipliedByValue));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && SynetAddVectorMultipliedByValueAutoTest(EPS, FUNC_AVMV(Simd::Avx2::SynetAddVectorMultipliedByValue), FUNC_AVMV(SimdSynetAddVectorMultipliedByValue));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && SynetAddVectorMultipliedByValueAutoTest(EPS, FUNC_AVMV(Simd::Avx512bw::SynetAddVectorMultipliedByValue), FUNC_AVMV(SimdSynetAddVectorMultipliedByValue));
#endif

#ifdef SIMD_SVE2_ENABLE
        if (Simd::Sve2::Enable && TestSve2(options))
            result = result && SynetAddVectorMultipliedByValueAutoTest(EPS, FUNC_AVMV(Simd::Sve2::SynetAddVectorMultipliedByValue), FUNC_AVMV(SimdSynetAddVectorMultipliedByValue));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options))
            result = result && SynetAddVectorMultipliedByValueAutoTest(EPS, FUNC_AVMV(Simd::Neon::SynetAddVectorMultipliedByValue), FUNC_AVMV(SimdSynetAddVectorMultipliedByValue));
#endif

        return result;
    }
#endif
}
