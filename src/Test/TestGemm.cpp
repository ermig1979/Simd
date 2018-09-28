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
        struct FuncGemm32f
        {
            typedef void(*FuncPtr)(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc);

            FuncPtr func;
            String description;

            FuncGemm32f(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(size_t M, size_t N, size_t K, float alpha, const View & A, const View & B, float beta, const View & srcC, View & dstC) const
            {
                Simd::Copy(srcC, dstC);
                TEST_PERFORMANCE_TEST(description);
                func(M, N, K, &alpha, (float*)A.data, A.stride / sizeof(float), 
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

    bool Gemm32fAutoTest(int transA, int transB, size_t M, size_t N, size_t K, FuncGemm32f f1, FuncGemm32f f2)
    {
        bool result = true;

        f1.Update(M, N, K);
        f2.Update(M, N, K);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << M << ", " << N << ", " << K << "].");

        View A(transA ? M : K, transA ? K : M, View::Float, NULL, TEST_ALIGN(1));
        View B(transB ? K : N, transB ? N : K, View::Float, NULL, TEST_ALIGN(1));
        View dstC1(N, M, View::Float, NULL, TEST_ALIGN(1));
        View dstC2(N, M, View::Float, NULL, TEST_ALIGN(1));
        View srcC(N, M, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        const float alpha = 1.5f, beta = 0.5f;
        FillRandom32f(A, -1.0f, 1.0f);
        FillRandom32f(B, -1.0f, 1.0f);
        FillRandom32f(srcC, -1.0f, 1.0f);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(M, N, K, alpha, A, B, beta, srcC, dstC1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(M, N, K, alpha, A, B, beta, srcC, dstC2));

        result = result && Compare(dstC1, dstC2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool Gemm32fNNAutoTest(const FuncGemm32f & f1, const FuncGemm32f & f2)
    {
        bool result = true;

        //result = result && Gemm32fAutoTest(0, 0, 666, 666, 666, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 555, 555, 555, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 999, 999, 999, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 333, 333, 333, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 999, 399, 379, f1, f2);

        //result = result && Gemm32fAutoTest(0, 0, 32, 173056, 27, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 64, 43264, 288, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 128, 10816, 576, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 64, 10816, 128, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 256, 2704, 1152, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 256, 2704, 128, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 512, 676, 2304, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 256, 676, 512, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 1024, 169, 4608, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 512, 169, 1024, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 1024, 169, 9216, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 64, 676, 512, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 1024, 169, 11520, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 425, 169, 1024, f1, f2);

        result = result && Gemm32fAutoTest(0, 0, 512, 25, 256, f1, f2);
        result = result && Gemm32fAutoTest(0, 0, 256, 4, 128, f1, f2);
 
        return result;
    }

    bool Gemm32fNNAutoTest()
    {
        bool result = true;

        result = result && Gemm32fNNAutoTest(FUNC_GEMM32F(Simd::Base::Gemm32fNN), FUNC_GEMM32F(SimdGemm32fNN));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && Gemm32fNNAutoTest(FUNC_GEMM32F(Simd::Sse::Gemm32fNN), FUNC_GEMM32F(SimdGemm32fNN));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && Gemm32fNNAutoTest(FUNC_GEMM32F(Simd::Avx::Gemm32fNN), FUNC_GEMM32F(SimdGemm32fNN));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && Gemm32fNNAutoTest(FUNC_GEMM32F(Simd::Avx2::Gemm32fNN), FUNC_GEMM32F(SimdGemm32fNN));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && Gemm32fNNAutoTest(FUNC_GEMM32F(Simd::Avx512f::Gemm32fNN), FUNC_GEMM32F(SimdGemm32fNN));
#endif

        return result;
    }

    bool Gemm32fNTAutoTest(const FuncGemm32f & f1, const FuncGemm32f & f2)
    {
        bool result = true;

        result = result && Gemm32fAutoTest(0, 1, 512, 25, 256, f1, f2);
        result = result && Gemm32fAutoTest(0, 1, 256, 4, 128, f1, f2);

        return result;
    }

    bool Gemm32fNTAutoTest()
    {
        bool result = true;

        result = result && Gemm32fNTAutoTest(FUNC_GEMM32F(Simd::Base::Gemm32fNT), FUNC_GEMM32F(SimdGemm32fNT));

#ifdef SIMD_SSE3_ENABLE
        if (Simd::Sse3::Enable)
            result = result && Gemm32fNTAutoTest(FUNC_GEMM32F(Simd::Sse3::Gemm32fNT), FUNC_GEMM32F(SimdGemm32fNT));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool Gemm32fDataTest(bool create, int transA, int transB, size_t M, size_t N, size_t K, const FuncGemm32f & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << M << ", " << N << ", " << K << "].");

        View A(transA ? M : K, transA ? K : M, View::Float, NULL, TEST_ALIGN(1));
        View B(transB ? K : N, transB ? N : K, View::Float, NULL, TEST_ALIGN(1));
        View dstC1(N, M, View::Float, NULL, TEST_ALIGN(1));
        View dstC2(N, M, View::Float, NULL, TEST_ALIGN(1));
        View srcC(N, M, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        const float alpha = 1.5f, beta = 0.5f;

        if (create)
        {
            FillRandom32f(A, -1.0f, 1.0f);
            FillRandom32f(B, -1.0f, 1.0f);
            FillRandom32f(srcC, -1.0f, 1.0f);

            TEST_SAVE(A);
            TEST_SAVE(B);
            TEST_SAVE(srcC);

            f.Call(M, N, K, alpha, A, B, beta, srcC, dstC1);

            TEST_SAVE(dstC1);
        }
        else
        {
            TEST_LOAD(A);
            TEST_LOAD(B);
            TEST_LOAD(srcC);

            TEST_LOAD(dstC1);

            f.Call(M, N, K, alpha, A, B, beta, srcC, dstC2);

            TEST_SAVE(dstC2);

            result = result && Compare(dstC1, dstC2, EPS, true, 32, false);
        }

        return result;
    }

    bool Gemm32fNNDataTest(bool create)
    {
        return Gemm32fDataTest(create, 0, 0, 16, 18, 20, FUNC_GEMM32F(SimdGemm32fNN));
    }

    bool Gemm32fNTDataTest(bool create)
    {
        return Gemm32fDataTest(create, 0, 1, 16, 18, 20, FUNC_GEMM32F(SimdGemm32fNT));
    }
}
