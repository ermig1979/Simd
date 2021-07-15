/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE __m128 Tail(size_t tail)
        {
            const int32_t mask[DF] = { 0, 0, 0, 0, -1, -1, -1, -1 };
            return _mm_loadu_ps((float*)(mask + tail));
        }

        SIMD_INLINE void Add4ExtractedSums(const __m128 & sum0, const __m128 & sum1, const __m128 & sum2, const __m128 & sum3, const __m128 & alpha, float * dst)
        {
            _mm_storeu_ps(dst, _mm_add_ps(_mm_loadu_ps(dst), _mm_mul_ps(alpha, _mm_hadd_ps(_mm_hadd_ps(sum0, sum1), _mm_hadd_ps(sum2, sum3)))));
        }

        static void Kernel1x1x4nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K4 = K & (~3);
            const float * A0 = A + 0 * lda;
            const float * B0 = B + 0 * ldb;
            __m128 c00 = _mm_setzero_ps();
            __m128 a0, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = _mm_loadu_ps(A0 + k);
                b0 = _mm_loadu_ps(B0 + k);
                c00 = _mm_add_ps(c00, _mm_mul_ps(a0, b0));
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                __m128 tail = Tail(K - K4);
                a0 = _mm_and_ps(tail, _mm_loadu_ps(A0 + k));
                b0 = _mm_loadu_ps(B0 + k);
                c00 = _mm_add_ps(c00, _mm_mul_ps(a0, b0));
            }
            C[0] += alpha * ExtractSum(c00);
        }

        static void Kernel1x4x4nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K4 = K & (~3);
            const float * A0 = A + 0 * lda;
            const float * B0 = B + 0 * ldb;
            const float * B1 = B + 1 * ldb;
            const float * B2 = B + 2 * ldb;
            const float * B3 = B + 3 * ldb;
            __m128 c00 = _mm_setzero_ps();
            __m128 c01 = _mm_setzero_ps();
            __m128 c02 = _mm_setzero_ps();
            __m128 c03 = _mm_setzero_ps();
            __m128 a0, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = _mm_loadu_ps(A0 + k);
                b0 = _mm_loadu_ps(B0 + k);
                c00 = _mm_add_ps(c00, _mm_mul_ps(a0, b0));
                b0 = _mm_loadu_ps(B1 + k);
                c01 = _mm_add_ps(c01, _mm_mul_ps(a0, b0));
                b0 = _mm_loadu_ps(B2 + k);
                c02 = _mm_add_ps(c02, _mm_mul_ps(a0, b0));
                b0 = _mm_loadu_ps(B3 + k);
                c03 = _mm_add_ps(c03, _mm_mul_ps(a0, b0));
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                __m128 tail = Tail(K - K4);
                a0 = _mm_and_ps(tail, _mm_loadu_ps(A0 + k));
                b0 = _mm_loadu_ps(B0 + k);
                c00 = _mm_add_ps(c00, _mm_mul_ps(a0, b0));
                b0 = _mm_loadu_ps(B1 + k);
                c01 = _mm_add_ps(c01, _mm_mul_ps(a0, b0));
                b0 = _mm_loadu_ps(B2 + k);
                c02 = _mm_add_ps(c02, _mm_mul_ps(a0, b0));
                b0 = _mm_loadu_ps(B3 + k);
                c03 = _mm_add_ps(c03, _mm_mul_ps(a0, b0));
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            Add4ExtractedSums(c00, c01, c02, c03, _alpha, C + 0*ldc);
        }

        static void Kernel2x1x4nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K4 = K & (~3);
            const float * A0 = A + 0 * lda;
            const float * A1 = A + 1 * lda;
            const float * B0 = B + 0 * ldb;
            __m128 c00 = _mm_setzero_ps();
            __m128 c10 = _mm_setzero_ps();
            __m128 a0, a1, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = _mm_loadu_ps(A0 + k);
                a1 = _mm_loadu_ps(A1 + k);
                b0 = _mm_loadu_ps(B0 + k);
                c00 = _mm_add_ps(c00, _mm_mul_ps(a0, b0));
                c10 = _mm_add_ps(c10, _mm_mul_ps(a1, b0));
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                __m128 tail = Tail(K - K4);
                a0 = _mm_and_ps(tail, _mm_loadu_ps(A0 + k));
                a1 = _mm_and_ps(tail, _mm_loadu_ps(A1 + k));
                b0 = _mm_loadu_ps(B0 + k);
                c00 = _mm_add_ps(c00, _mm_mul_ps(a0, b0));
                c10 = _mm_add_ps(c10, _mm_mul_ps(a1, b0));
            }
            C[0*ldc] += alpha * ExtractSum(c00);
            C[1*ldc] += alpha * ExtractSum(c10);
        }

        static void Kernel2x4x4nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K4 = K & (~3);
            const float * A0 = A + 0 * lda;
            const float * A1 = A + 1 * lda;
            const float * B0 = B + 0 * ldb;
            const float * B1 = B + 1 * ldb;
            const float * B2 = B + 2 * ldb;
            const float * B3 = B + 3 * ldb;
            __m128 c00 = _mm_setzero_ps();
            __m128 c01 = _mm_setzero_ps();
            __m128 c02 = _mm_setzero_ps();
            __m128 c03 = _mm_setzero_ps();
            __m128 c10 = _mm_setzero_ps();
            __m128 c11 = _mm_setzero_ps();
            __m128 c12 = _mm_setzero_ps();
            __m128 c13 = _mm_setzero_ps();
            __m128 a0, a1, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = _mm_loadu_ps(A0 + k);
                a1 = _mm_loadu_ps(A1 + k);
                b0 = _mm_loadu_ps(B0 + k);
                c00 = _mm_add_ps(c00, _mm_mul_ps(a0, b0));
                c10 = _mm_add_ps(c10, _mm_mul_ps(a1, b0));
                b0 = _mm_loadu_ps(B1 + k);
                c01 = _mm_add_ps(c01, _mm_mul_ps(a0, b0));
                c11 = _mm_add_ps(c11, _mm_mul_ps(a1, b0));
                b0 = _mm_loadu_ps(B2 + k);
                c02 = _mm_add_ps(c02, _mm_mul_ps(a0, b0));
                c12 = _mm_add_ps(c12, _mm_mul_ps(a1, b0));
                b0 = _mm_loadu_ps(B3 + k);
                c03 = _mm_add_ps(c03, _mm_mul_ps(a0, b0));
                c13 = _mm_add_ps(c13, _mm_mul_ps(a1, b0));
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                __m128 tail = Tail(K - K4);
                a0 = _mm_and_ps(tail, _mm_loadu_ps(A0 + k));
                a1 = _mm_and_ps(tail, _mm_loadu_ps(A1 + k));
                b0 = _mm_loadu_ps(B0 + k);
                c00 = _mm_add_ps(c00, _mm_mul_ps(a0, b0));
                c10 = _mm_add_ps(c10, _mm_mul_ps(a1, b0));
                b0 = _mm_loadu_ps(B1 + k);
                c01 = _mm_add_ps(c01, _mm_mul_ps(a0, b0));
                c11 = _mm_add_ps(c11, _mm_mul_ps(a1, b0));
                b0 = _mm_loadu_ps(B2 + k);
                c02 = _mm_add_ps(c02, _mm_mul_ps(a0, b0));
                c12 = _mm_add_ps(c12, _mm_mul_ps(a1, b0));
                b0 = _mm_loadu_ps(B3 + k);
                c03 = _mm_add_ps(c03, _mm_mul_ps(a0, b0));
                c13 = _mm_add_ps(c13, _mm_mul_ps(a1, b0));
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            Add4ExtractedSums(c00, c01, c02, c03, _alpha, C + 0 * ldc);
            Add4ExtractedSums(c10, c11, c12, c13, _alpha, C + 1 * ldc);
        }

        static void Kernel3x1x4nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K4 = K & (~3);
            const float * A0 = A + 0 * lda;
            const float * A1 = A + 1 * lda;
            const float * A2 = A + 2 * lda;
            const float * B0 = B + 0 * ldb;
            __m128 c00 = _mm_setzero_ps();
            __m128 c10 = _mm_setzero_ps();
            __m128 c20 = _mm_setzero_ps();
            __m128 a0, a1, a2, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = _mm_loadu_ps(A0 + k);
                a1 = _mm_loadu_ps(A1 + k);
                a2 = _mm_loadu_ps(A2 + k);
                b0 = _mm_loadu_ps(B0 + k);
                c00 = _mm_add_ps(c00, _mm_mul_ps(a0, b0));
                c10 = _mm_add_ps(c10, _mm_mul_ps(a1, b0));
                c20 = _mm_add_ps(c20, _mm_mul_ps(a2, b0));
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                __m128 tail = Tail(K - K4);
                a0 = _mm_and_ps(tail, _mm_loadu_ps(A0 + k));
                a1 = _mm_and_ps(tail, _mm_loadu_ps(A1 + k));
                a2 = _mm_and_ps(tail, _mm_loadu_ps(A2 + k));
                b0 = _mm_loadu_ps(B0 + k);
                c00 = _mm_add_ps(c00, _mm_mul_ps(a0, b0));
                c10 = _mm_add_ps(c10, _mm_mul_ps(a1, b0));
                c20 = _mm_add_ps(c20, _mm_mul_ps(a2, b0));
            }
            C[0 * ldc] += alpha * ExtractSum(c00);
            C[1 * ldc] += alpha * ExtractSum(c10);
            C[2 * ldc] += alpha * ExtractSum(c20);
        }

        static void Kernel3x4x4nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K4 = K & (~3);
            const float * A0 = A + 0 * lda;
            const float * A1 = A + 1 * lda;
            const float * A2 = A + 2 * lda;
            const float * B0 = B + 0 * ldb;
            const float * B1 = B + 1 * ldb;
            const float * B2 = B + 2 * ldb;
            const float * B3 = B + 3 * ldb;
            __m128 c00 = _mm_setzero_ps();
            __m128 c01 = _mm_setzero_ps();
            __m128 c02 = _mm_setzero_ps();
            __m128 c03 = _mm_setzero_ps();
            __m128 c10 = _mm_setzero_ps();
            __m128 c11 = _mm_setzero_ps();
            __m128 c12 = _mm_setzero_ps();
            __m128 c13 = _mm_setzero_ps();
            __m128 c20 = _mm_setzero_ps();
            __m128 c21 = _mm_setzero_ps();
            __m128 c22 = _mm_setzero_ps();
            __m128 c23 = _mm_setzero_ps();
            __m128 a0, a1, a2, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = _mm_loadu_ps(A0 + k);
                a1 = _mm_loadu_ps(A1 + k);
                a2 = _mm_loadu_ps(A2 + k);
                b0 = _mm_loadu_ps(B0 + k);
                c00 = _mm_add_ps(c00, _mm_mul_ps(a0, b0));
                c10 = _mm_add_ps(c10, _mm_mul_ps(a1, b0));
                c20 = _mm_add_ps(c20, _mm_mul_ps(a2, b0));
                b0 = _mm_loadu_ps(B1 + k);
                c01 = _mm_add_ps(c01, _mm_mul_ps(a0, b0));
                c11 = _mm_add_ps(c11, _mm_mul_ps(a1, b0));
                c21 = _mm_add_ps(c21, _mm_mul_ps(a2, b0));
                b0 = _mm_loadu_ps(B2 + k);
                c02 = _mm_add_ps(c02, _mm_mul_ps(a0, b0));
                c12 = _mm_add_ps(c12, _mm_mul_ps(a1, b0));
                c22 = _mm_add_ps(c22, _mm_mul_ps(a2, b0));
                b0 = _mm_loadu_ps(B3 + k);
                c03 = _mm_add_ps(c03, _mm_mul_ps(a0, b0));
                c13 = _mm_add_ps(c13, _mm_mul_ps(a1, b0));
                c23 = _mm_add_ps(c23, _mm_mul_ps(a2, b0));
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                __m128 tail = Tail(K - K4);
                a0 = _mm_and_ps(tail, _mm_loadu_ps(A0 + k));
                a1 = _mm_and_ps(tail, _mm_loadu_ps(A1 + k));
                a2 = _mm_and_ps(tail, _mm_loadu_ps(A2 + k));
                b0 = _mm_loadu_ps(B0 + k);
                c00 = _mm_add_ps(c00, _mm_mul_ps(a0, b0));
                c10 = _mm_add_ps(c10, _mm_mul_ps(a1, b0));
                c20 = _mm_add_ps(c20, _mm_mul_ps(a2, b0));
                b0 = _mm_loadu_ps(B1 + k);
                c01 = _mm_add_ps(c01, _mm_mul_ps(a0, b0));
                c11 = _mm_add_ps(c11, _mm_mul_ps(a1, b0));
                c21 = _mm_add_ps(c21, _mm_mul_ps(a2, b0));
                b0 = _mm_loadu_ps(B2 + k);
                c02 = _mm_add_ps(c02, _mm_mul_ps(a0, b0));
                c12 = _mm_add_ps(c12, _mm_mul_ps(a1, b0));
                c22 = _mm_add_ps(c22, _mm_mul_ps(a2, b0));
                b0 = _mm_loadu_ps(B3 + k);
                c03 = _mm_add_ps(c03, _mm_mul_ps(a0, b0));
                c13 = _mm_add_ps(c13, _mm_mul_ps(a1, b0));
                c23 = _mm_add_ps(c23, _mm_mul_ps(a2, b0));
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            Add4ExtractedSums(c00, c01, c02, c03, _alpha, C + 0 * ldc);
            Add4ExtractedSums(c10, c11, c12, c13, _alpha, C + 1 * ldc);
            Add4ExtractedSums(c20, c21, c22, c23, _alpha, C + 2 * ldc);
        }

        void Gemm32fNT(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            typedef Simd::GemmNT<float, F> GemmNT;
#ifdef SIMD_X64_ENABLE
            GemmNT gemmNT(M, N, K, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), Sse2::GemmScaleC,
                Kernel1x1x4nt, Kernel1x4x4nt, Kernel2x1x4nt, Kernel2x4x4nt, Kernel3x1x4nt, Kernel3x4x4nt, NULL, NULL);
#else
            GemmNT gemmNT(M, N, K, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), Sse2::GemmScaleC,
                Kernel1x1x4nt, Kernel1x4x4nt, NULL, NULL, NULL, NULL, NULL, NULL);
#endif
            gemmNT.Run(alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
#endif// SIMD_SSE41_ENABLE
}
