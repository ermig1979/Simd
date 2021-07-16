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
#include "Test/TestTensor.h"

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

            void Call(size_t M, size_t N, size_t K, float alpha, const Tensor32f & A, const Tensor32f & B, float beta, const Tensor32f & srcC, Tensor32f & dstC) const
            {
                memcpy(dstC.Data(), srcC.Data(), sizeof(float)*srcC.Size());
                TEST_PERFORMANCE_TEST(description);
                func(M, N, K, &alpha, A.Data(), A.Axis(1), B.Data(), B.Axis(1), &beta, dstC.Data(), dstC.Axis(1));
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

        Tensor32f A({ transA ? K : M, transA ? M : K });
        Tensor32f B({ transB ? N : K, transB ? K : N });
        Tensor32f dstC1({ M, N });
        Tensor32f dstC2({ M, N });
        Tensor32f srcC({ M, N });

        const float alpha = 1.5f, beta = 0.5f;
        FillRandom(A.Data(), A.Size(), -1.0, 1.0f);
        FillRandom(B.Data(), B.Size(), -1.0, 1.0f);
        FillRandom(srcC.Data(), srcC.Size(), -1.0, 1.0f);

        TEST_ALIGN(SIMD_ALIGN);

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
        //result = result && Gemm32fAutoTest(0, 0, 333, 334, 335, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 334, 334, 334, f1, f2);
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

        //result = result && Gemm32fAutoTest(0, 0, 1280, 100, 256, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 512, 25, 256, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 256, 4, 128, f1, f2);

        //result = result && Gemm32fAutoTest(0, 0, 666, 666, 666, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 728, 196, 728, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 728, 192, 728, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 728, 4, 728, f1, f2);

        //result = result && Gemm32fAutoTest(0, 0, 32, 1216, 144, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 1216, 32, 144, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 64, 304, 32, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 304, 64, 32, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 128, 42, 64, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 42, 128, 64, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 256, 12, 128, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 12, 256, 128, f1, f2);

        //result = result && Gemm32fAutoTest(0, 0, 96, 22500, 16, f1, f2);       
        //result = result && Gemm32fAutoTest(0, 0, 22500, 96, 16, f1, f2);

        //result = result && Gemm32fAutoTest(0, 0, 5625, 144, 24, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 144, 5625, 24, f1, f2);

        //result = result && Gemm32fAutoTest(0, 0, 728, 196, 728, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 196, 728, 728, f1, f2);

        //result = result && Gemm32fAutoTest(0, 0, 9, 256, 256, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 256, 9, 256, f1, f2);

        //result = result && Gemm32fAutoTest(0, 0, 997, 998, 999, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 999, 998, 997, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 667, 666, 665, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 665, 666, 667, f1, f2);
        
        //result = result && Gemm32fAutoTest(0, 0, 32, 22500, 27, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 22500, 32, 27, f1, f2);

        //result = result && Gemm32fAutoTest(0, 0, 16, 22500, 27, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 22500, 16, 27, f1, f2);

        //result = result && Gemm32fAutoTest(0, 0, 728, 196, 728, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 196, 728, 728, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 728, 192, 728, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 192, 728, 728, f1, f2);
        
        //result = result && Gemm32fAutoTest(0, 0, 1002, 1001, 3000, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 1000, 1001, 3002, f1, f2);

        //result = result && Gemm32fAutoTest(0, 0, 4096, 64, 1200, f1, f2);

        //result = result && Gemm32fAutoTest(0, 0, 25600, 48, 8, f1, f2);        
        //result = result && Gemm32fAutoTest(0, 0, 25600, 8, 16, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 25600/10, 48, 8, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 25600/10, 8, 16, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 25600, 48, 16, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 6400, 8, 48, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 1024, 1024, 1024, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 256, 256, 256, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 128, 128, 128, f1, f2);

        //result = result && Gemm32fAutoTest(0, 0, 7245, 2, 32, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 7245, 4, 32, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 2, 7245, 32, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 4, 7245, 32, f1, f2);

        //result = result && Gemm32fAutoTest(0, 0, 2048, 36, 448, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 36, 2048, 448, f1, f2);

        result = result && Gemm32fAutoTest(0, 0, 1, 192, 192, f1, f2);
        //result = result && Gemm32fAutoTest(0, 0, 10, 192, 192, f1, f2);

        return result;
    }

    bool Gemm32fNNAutoTest()
    {
        bool result = true;

        result = result && Gemm32fNNAutoTest(FUNC_GEMM32F(Simd::Base::Gemm32fNN), FUNC_GEMM32F(SimdGemm32fNN));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && Gemm32fNNAutoTest(FUNC_GEMM32F(Simd::Sse2::Gemm32fNN), FUNC_GEMM32F(SimdGemm32fNN));
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

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && Gemm32fNNAutoTest(FUNC_GEMM32F(Simd::Neon::Gemm32fNN), FUNC_GEMM32F(SimdGemm32fNN));
#endif

        return result;
    }

    bool Gemm32fNTAutoTest(const FuncGemm32f & f1, const FuncGemm32f & f2)
    {
        bool result = true;

        //result = result && Gemm32fAutoTest(0, 1, 666, 666, 666, f1, f2);
        //result = result && Gemm32fAutoTest(0, 1, 997, 998, 999, f1, f2);
        //result = result && Gemm32fAutoTest(0, 1, 999, 998, 997, f1, f2);
        //result = result && Gemm32fAutoTest(0, 1, 333, 334, 335, f1, f2);
        //result = result && Gemm32fAutoTest(0, 1, 7245, 2, 32, f1, f2);
        //result = result && Gemm32fAutoTest(0, 1, 7245, 4, 32, f1, f2);
        //result = result && Gemm32fAutoTest(0, 1, 2, 7245, 32, f1, f2);
        //result = result && Gemm32fAutoTest(0, 1, 4, 7245, 32, f1, f2);


        //result = result && Gemm32fAutoTest(0, 1, 1280, 100, 256, f1, f2);
        //result = result && Gemm32fAutoTest(0, 1, 512, 25, 256, f1, f2);
        //result = result && Gemm32fAutoTest(0, 1, 256, 4, 128, f1, f2);
        //result = result && Gemm32fAutoTest(0, 1, 16, 1, 1152, f1, f2);
        //result = result && Gemm32fAutoTest(0, 1, 16, 25, 4608, f1, f2);

        //result = result && Gemm32fAutoTest(0, 1, 728, 196, 728, f1, f2);
        //result = result && Gemm32fAutoTest(0, 1, 728, 192, 728, f1, f2);
        //result = result && Gemm32fAutoTest(0, 1, 728, 4, 728, f1, f2);

        //result = result && Gemm32fAutoTest(0, 1, 6400, 8, 48, f1, f2);
        //result = result && Gemm32fAutoTest(0, 1, 2048, 36, 448, f1, f2);
        //result = result && Gemm32fAutoTest(0, 1, 36, 2048, 448, f1, f2);

        result = result && Gemm32fAutoTest(0, 1, 1, 192, 192, f1, f2);
        result = result && Gemm32fAutoTest(0, 1, 10, 192, 192, f1, f2);

        return result;
    }

    bool Gemm32fNTAutoTest()
    {
        bool result = true;

        result = result && Gemm32fNTAutoTest(FUNC_GEMM32F(Simd::Base::Gemm32fNT), FUNC_GEMM32F(SimdGemm32fNT));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && Gemm32fNTAutoTest(FUNC_GEMM32F(Simd::Sse41::Gemm32fNT), FUNC_GEMM32F(SimdGemm32fNT));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && Gemm32fNTAutoTest(FUNC_GEMM32F(Simd::Avx::Gemm32fNT), FUNC_GEMM32F(SimdGemm32fNT));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && Gemm32fNTAutoTest(FUNC_GEMM32F(Simd::Avx2::Gemm32fNT), FUNC_GEMM32F(SimdGemm32fNT));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && Gemm32fNTAutoTest(FUNC_GEMM32F(Simd::Avx512f::Gemm32fNT), FUNC_GEMM32F(SimdGemm32fNT));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && Gemm32fNTAutoTest(FUNC_GEMM32F(Simd::Neon::Gemm32fNT), FUNC_GEMM32F(SimdGemm32fNT));
#endif

        return result;
    }
}
