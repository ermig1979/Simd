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
        struct FuncGemm32f
        {
            typedef void(*FuncPtr)(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc);

            FuncPtr func;
            String description;

            FuncGemm32f(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(float alpha, const View & A, const View & B, float beta, const View & srcC, View & dstC) const
            {
                Simd::Copy(srcC, dstC);
                TEST_PERFORMANCE_TEST(description);
                func(A.height, B.width, A.width, &alpha, (float*)A.data, A.stride / sizeof(float), 
                    (float*)B.data, B.stride / sizeof(float), &beta, (float*)dstC.data, dstC.stride / sizeof(float));
            }

            void Update(size_t M, size_t N, size_t K)
            {
                std::stringstream ss;
                ss << description;
                ss << "[" << M << "-" << N << "-" << K << "]";
                description = ss.str();
            }
        };
    }

#define FUNC_GEMM32F(function) FuncGemm32f(function, #function)

    bool Gemm32fAutoTest(size_t M, size_t N, size_t K, FuncGemm32f f1, FuncGemm32f f2)
    {
        bool result = true;

        f1.Update(M, N, K);
        f2.Update(M, N, K);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << M << ", " << N << ", " << K << "].");

        View A(M, K, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View B(K, N, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View srcC(M, N, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstC1(M, N, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstC2(M, N, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        const float alpha = 1.5f, beta = 0.5f;
        FillRandom32f(A, -1.0f, 1.0f);
        FillRandom32f(B, -1.0f, 1.0f);
        FillRandom32f(srcC, -1.0f, 1.0f);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(alpha, A, B, beta, srcC, dstC1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(alpha, A, B, beta, srcC, dstC2));

        result = result && Compare(dstC1, dstC2, EPS, true, 32, false);

        return result;
    }

    bool Gemm32fAutoTest(const FuncGemm32f & f1, const FuncGemm32f & f2)
    {
        bool result = true;

        result = result && Gemm32fAutoTest(999, 999, 999, f1, f2);
        result = result && Gemm32fAutoTest(666, 666, 666, f1, f2);
        result = result && Gemm32fAutoTest(333, 333, 333, f1, f2);

        return result;
    }

    bool Gemm32fNNAutoTest()
    {
        bool result = true;

        result = result && Gemm32fAutoTest(FUNC_GEMM32F(Simd::Base::Gemm32fNN), FUNC_GEMM32F(SimdGemm32fNN));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && Gemm32fAutoTest(FUNC_GEMM32F(Simd::Sse::Gemm32fNN), FUNC_GEMM32F(SimdGemm32fNN));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool Gemm32fDataTest(bool create, size_t M, size_t N, size_t K, const FuncGemm32f & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << M << ", " << N << ", " << K << "].");

        View A(M, K, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View B(K, N, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View srcC(M, N, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstC1(M, N, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstC2(M, N, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        const float alpha = 1.5f, beta = 0.5f;

        if (create)
        {
            FillRandom32f(A, -1.0f, 1.0f);
            FillRandom32f(B, -1.0f, 1.0f);
            FillRandom32f(srcC, -1.0f, 1.0f);

            TEST_SAVE(A);
            TEST_SAVE(B);
            TEST_SAVE(srcC);

            f.Call(alpha, A, B, beta, srcC, dstC1);

            TEST_SAVE(dstC1);
        }
        else
        {
            TEST_LOAD(A);
            TEST_LOAD(B);
            TEST_LOAD(srcC);

            TEST_LOAD(dstC1);

            f.Call(alpha, A, B, beta, srcC, dstC2);

            TEST_SAVE(dstC2);

            result = result && Compare(dstC1, dstC2, EPS, true, 32, false);
        }

        return result;
    }

    bool Gemm32fNNDataTest(bool create)
    {
        return Gemm32fDataTest(create, 16, 18, 20, FUNC_GEMM32F(SimdGemm32fNN));
    }
}
