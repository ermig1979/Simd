/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"

namespace Simd
{
#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
#define GEMM_VER 2

        SIMD_INLINE void AddDot1x1(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register float c0 = 0;
            for (size_t k = 0; k < K; k++)
            {
                float a = alpha * A[k];
                c0 += a * B[0];
                B += ldb;
            }
            C[0] += c0;
        }        
        
        SIMD_INLINE void AddDot1x4(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register float c0 = 0;
            register float c1 = 0;
            register float c2 = 0;
            register float c3 = 0;
            for (size_t k = 0; k < K; k++) 
            {
                float a = alpha * A[k];
                c0 += a * B[0];
                c1 += a * B[1];
                c2 += a * B[2];
                c3 += a * B[3];
                B += ldb;
            }
            C[0] += c0;
            C[1] += c1;
            C[2] += c2;
            C[3] += c3;
        }

        SIMD_INLINE void AddDot4x1(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register float c0 = 0;
            register float c1 = 0;
            register float c2 = 0;
            register float c3 = 0;
            const float * a0 = A + lda * 0;
            const float * a1 = A + lda * 1;
            const float * a2 = A + lda * 2;
            const float * a3 = A + lda * 3;
            for (size_t k = 0; k < K; k++)
            {
                float b = alpha * B[0];
                c0 += b * (*a0++);
                c1 += b * (*a1++);
                c2 += b * (*a2++);
                c3 += b * (*a3++);
                B += ldb;
            }
            C[0 * ldc] += c0;
            C[1 * ldc] += c1;
            C[2 * ldc] += c2;
            C[3 * ldc] += c3;
        }

        SIMD_INLINE void AddDot4x2(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register float c00 = 0;
            register float c10 = 0;
            register float c20 = 0;
            register float c30 = 0;
            register float c01 = 0;
            register float c11 = 0;
            register float c21 = 0;
            register float c31 = 0;
            const float * a0 = A + lda * 0;
            const float * a1 = A + lda * 1;
            const float * a2 = A + lda * 2;
            const float * a3 = A + lda * 3;
            for (size_t k = 0; k < K; k++)
            {
                float b0 = alpha * B[0];
                float b1 = alpha * B[1];
                c00 += b0 * (*a0);
                c10 += b0 * (*a1);
                c20 += b0 * (*a2);
                c30 += b0 * (*a3);
                c01 += b1 * (*a0++);
                c11 += b1 * (*a1++);
                c21 += b1 * (*a2++);
                c31 += b1 * (*a3++);
                B += ldb;
            }
            C[0 * ldc + 0] += c00;
            C[0 * ldc + 1] += c01;
            C[1 * ldc + 0] += c10;
            C[1 * ldc + 1] += c11;
            C[2 * ldc + 0] += c20;
            C[2 * ldc + 1] += c21;
            C[3 * ldc + 0] += c30;
            C[3 * ldc + 1] += c31;
        }

#if GEMM_VER == 0
        void Gemm32fNN(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            register float b = beta[0];
            for (size_t i = 0; i < M; ++i)
            {
                float * pC = C + i * ldc;
                for (size_t j = 0; j < N; ++j)
                    pC[j] = b * pC[j];
            }
            size_t N4 = AlignLo(N, 4);
            register float a = alpha[0];
            for (size_t i = 0; i < M; ++i) 
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    AddDot1x4(K, a, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
                for (; j < N; j += 1)
                    AddDot1x1(K, a, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
            }
        }
#elif GEMM_VER == 1
        void Gemm32fNN(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            register float b = beta[0];
            for (size_t i = 0; i < M; ++i)
            {
                float * pC = C + i * ldc;
                for (size_t j = 0; j < N; ++j)
                    pC[j] = b * pC[j];
            }
            size_t M4 = AlignLo(M, 4);
            register float a = alpha[0];
            size_t i = 0;
            for (; i < M4; i += 4)
            {
                for (size_t j = 0; j < N; ++j)
                    AddDot4x1(K, a, A + i*lda, lda, B + j, ldb, C + i * ldc + j, ldc);
            }
            for (; i < M; i += 1)
            {
                for (size_t j = 0; j < N; ++j)
                    AddDot1x1(K, a, A + i*lda, lda, B + j, ldb, C + i * ldc + j, ldc);
            }
        }
#elif GEMM_VER == 2
        void Gemm32fNN(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            register float b = beta[0];
            for (size_t i = 0; i < M; ++i)
            {
                float * pC = C + i * ldc;
                for (size_t j = 0; j < N; ++j)
                    pC[j] = b * pC[j];
            }
            size_t M4 = AlignLo(M, 4);
            size_t N2 = AlignLo(N, 2);
            register float a = alpha[0];
            size_t i = 0;
            for (; i < M4; i += 4)
            {
                size_t j = 0;
                for (; j < N; j += 2)
                    AddDot4x2(K, a, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
                for (0; j < N; ++j)
                    AddDot4x1(K, a, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
            }
            for (; i < M; i += 1)
            {
                for (size_t j = 0; j < N; ++j)
                    AddDot1x1(K, a, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
            }
        }
#elif GEMM_VER == 3
        void Gemm32fNN(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            register float b = beta[0];
            for (size_t i = 0; i < M; ++i)
            {
                float * pC = C + i * ldc;
                for (size_t j = 0; j < N; ++j)
                    pC[j] = b * pC[j];
            }
            size_t M4 = AlignLo(M, 4);
            size_t N2 = AlignLo(N, 2);
            register float a = alpha[0];
            size_t i = 0;
            for (; i < M4; i += 4)
            {
                size_t j = 0;
                for (; j < N; j += 2)
                    AddDot4x2(K, a, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
                for (0; j < N; ++j)
                    AddDot4x1(K, a, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
            }
            for (; i < M; i += 1)
            {
                for (size_t j = 0; j < N; ++j)
                    AddDot1x1(K, a, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
            }
        }
#endif
    }
#endif// SIMD_SSE_ENABLE
}
