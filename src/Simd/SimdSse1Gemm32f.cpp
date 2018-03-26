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
#include "Simd/SimdSse1.h"

namespace Simd
{
#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        SIMD_INLINE void MulBy(float * ptr, __m128 value)
        {
            _mm_storeu_ps(ptr, _mm_mul_ps(_mm_loadu_ps(ptr), value));
        }

        SIMD_INLINE void AddTo(float * ptr, __m128 value)
        {
            _mm_storeu_ps(ptr, _mm_add_ps(_mm_loadu_ps(ptr), value));
        }

        SIMD_INLINE void AddDot1x1(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register float c0 = 0;
            for (size_t k = 0; k < K; k++)
            {
                c0 += A[k] * B[0];
                B += ldb;
            }
            C[0] += c0 * alpha;
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
                float b = B[0];
                c0 += b * (*a0++);
                c1 += b * (*a1++);
                c2 += b * (*a2++);
                c3 += b * (*a3++);
                B += ldb;
            }
            C[0 * ldc] += c0 * alpha;
            C[1 * ldc] += c1 * alpha;
            C[2 * ldc] += c2 * alpha;
            C[3 * ldc] += c3 * alpha;
        }

        SIMD_INLINE void AddDot6x1(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register float c0 = 0;
            register float c1 = 0;
            register float c2 = 0;
            register float c3 = 0;
            register float c4 = 0;
            register float c5 = 0;
            const float * a0 = A + lda * 0;
            const float * a1 = A + lda * 1;
            const float * a2 = A + lda * 2;
            const float * a3 = A + lda * 3;
            const float * a4 = A + lda * 4;
            const float * a5 = A + lda * 5;
            for (size_t k = 0; k < K; k++)
            {
                float b = B[0];
                c0 += b * (*a0++);
                c1 += b * (*a1++);
                c2 += b * (*a2++);
                c3 += b * (*a3++);
                c4 += b * (*a4++);
                c5 += b * (*a5++);
                B += ldb;
            }
            C[0 * ldc] += c0 * alpha;
            C[1 * ldc] += c1 * alpha;
            C[2 * ldc] += c2 * alpha;
            C[3 * ldc] += c3 * alpha;
            C[4 * ldc] += c4 * alpha;
            C[5 * ldc] += c5 * alpha;
        }

        SIMD_INLINE void AddDot1x4(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register __m128 c0 = _mm_setzero_ps();
            for (size_t k = 0; k < K; k++)
            {
                __m128 b0 = _mm_loadu_ps(B);
                c0 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[k])), c0);
                B += ldb;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            _mm_storeu_ps(C, _mm_add_ps(_mm_mul_ps(_alpha, c0), _mm_loadu_ps(C)));
        }

        SIMD_INLINE void AddDot4x4(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register __m128 c0 = _mm_setzero_ps();
            register __m128 c1 = _mm_setzero_ps();
            register __m128 c2 = _mm_setzero_ps();
            register __m128 c3 = _mm_setzero_ps();
            const float * a0 = A + lda * 0;
            const float * a1 = A + lda * 1;
            const float * a2 = A + lda * 2;
            const float * a3 = A + lda * 3;
            for (size_t k = 0; k < K; k++)
            {
                __m128 b0 = _mm_loadu_ps(B);
                c0 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(*a0++)), c0);
                c1 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(*a1++)), c1);
                c2 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(*a2++)), c2);
                c3 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(*a3++)), c3);
                B += ldb;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            _mm_storeu_ps(C + 0 * ldc, _mm_add_ps(_mm_mul_ps(_alpha, c0), _mm_loadu_ps(C + 0 * ldc)));
            _mm_storeu_ps(C + 1 * ldc, _mm_add_ps(_mm_mul_ps(_alpha, c1), _mm_loadu_ps(C + 1 * ldc)));
            _mm_storeu_ps(C + 2 * ldc, _mm_add_ps(_mm_mul_ps(_alpha, c2), _mm_loadu_ps(C + 2 * ldc)));
            _mm_storeu_ps(C + 3 * ldc, _mm_add_ps(_mm_mul_ps(_alpha, c3), _mm_loadu_ps(C + 3 * ldc)));
        }

        SIMD_INLINE void AddDot4x8(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register __m128 c00 = _mm_setzero_ps();
            register __m128 c10 = _mm_setzero_ps();
            register __m128 c20 = _mm_setzero_ps();
            register __m128 c30 = _mm_setzero_ps();
            register __m128 c01 = _mm_setzero_ps();
            register __m128 c11 = _mm_setzero_ps();
            register __m128 c21 = _mm_setzero_ps();
            register __m128 c31 = _mm_setzero_ps();
            const float * A0 = A + lda * 0;
            const float * A1 = A + lda * 1;
            const float * A2 = A + lda * 2;
            const float * A3 = A + lda * 3;
            register __m128 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + 0 * F);
                b1 = _mm_loadu_ps(B + 1 * F);
                a0 = _mm_set1_ps(*A0++);
                c00 = _mm_add_ps(_mm_mul_ps(a0, b0), c00);
                c01 = _mm_add_ps(_mm_mul_ps(a0, b1), c01);
                a0 = _mm_set1_ps(*A1++);
                c10 = _mm_add_ps(_mm_mul_ps(a0, b0), c10);
                c11 = _mm_add_ps(_mm_mul_ps(a0, b1), c11);
                a0 = _mm_set1_ps(*A2++);
                c20 = _mm_add_ps(_mm_mul_ps(a0, b0), c20);
                c21 = _mm_add_ps(_mm_mul_ps(a0, b1), c21);
                a0 = _mm_set1_ps(*A3++);
                c30 = _mm_add_ps(_mm_mul_ps(a0, b0), c30);
                c31 = _mm_add_ps(_mm_mul_ps(a0, b1), c31);
                B += ldb;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            _mm_storeu_ps(C + 0 * ldc + 0 * F, _mm_add_ps(_mm_mul_ps(_alpha, c00), _mm_loadu_ps(C + 0 * ldc + 0 * F)));
            _mm_storeu_ps(C + 0 * ldc + 1 * F, _mm_add_ps(_mm_mul_ps(_alpha, c01), _mm_loadu_ps(C + 0 * ldc + 1 * F)));
            _mm_storeu_ps(C + 1 * ldc + 0 * F, _mm_add_ps(_mm_mul_ps(_alpha, c10), _mm_loadu_ps(C + 1 * ldc + 0 * F)));
            _mm_storeu_ps(C + 1 * ldc + 1 * F, _mm_add_ps(_mm_mul_ps(_alpha, c11), _mm_loadu_ps(C + 1 * ldc + 1 * F)));
            _mm_storeu_ps(C + 2 * ldc + 0 * F, _mm_add_ps(_mm_mul_ps(_alpha, c20), _mm_loadu_ps(C + 2 * ldc + 0 * F)));
            _mm_storeu_ps(C + 2 * ldc + 1 * F, _mm_add_ps(_mm_mul_ps(_alpha, c21), _mm_loadu_ps(C + 2 * ldc + 1 * F)));
            _mm_storeu_ps(C + 3 * ldc + 0 * F, _mm_add_ps(_mm_mul_ps(_alpha, c30), _mm_loadu_ps(C + 3 * ldc + 0 * F)));
            _mm_storeu_ps(C + 3 * ldc + 1 * F, _mm_add_ps(_mm_mul_ps(_alpha, c31), _mm_loadu_ps(C + 3 * ldc + 1 * F)));
        }

        SIMD_INLINE void AddDot4x12(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register __m128 c00 = _mm_setzero_ps();
            register __m128 c10 = _mm_setzero_ps();
            register __m128 c20 = _mm_setzero_ps();
            register __m128 c30 = _mm_setzero_ps();
            register __m128 c01 = _mm_setzero_ps();
            register __m128 c11 = _mm_setzero_ps();
            register __m128 c21 = _mm_setzero_ps();
            register __m128 c31 = _mm_setzero_ps();
            register __m128 c02 = _mm_setzero_ps();
            register __m128 c12 = _mm_setzero_ps();
            register __m128 c22 = _mm_setzero_ps();
            register __m128 c32 = _mm_setzero_ps();
            const float * A0 = A + lda * 0;
            const float * A1 = A + lda * 1;
            const float * A2 = A + lda * 2;
            const float * A3 = A + lda * 3;
            register __m128 b0, b1, b2, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + 0 * F);
                b1 = _mm_loadu_ps(B + 1 * F);
                b2 = _mm_loadu_ps(B + 2 * F);
                a0 = _mm_set1_ps(*A0++);
                c00 = _mm_add_ps(_mm_mul_ps(a0, b0), c00);
                c01 = _mm_add_ps(_mm_mul_ps(a0, b1), c01);
                c02 = _mm_add_ps(_mm_mul_ps(a0, b2), c02);
                a0 = _mm_set1_ps(*A1++);
                c10 = _mm_add_ps(_mm_mul_ps(a0, b0), c10);
                c11 = _mm_add_ps(_mm_mul_ps(a0, b1), c11);
                c12 = _mm_add_ps(_mm_mul_ps(a0, b2), c12);
                a0 = _mm_set1_ps(*A2++);
                c20 = _mm_add_ps(_mm_mul_ps(a0, b0), c20);
                c21 = _mm_add_ps(_mm_mul_ps(a0, b1), c21);
                c22 = _mm_add_ps(_mm_mul_ps(a0, b2), c22);
                a0 = _mm_set1_ps(*A3++);
                c30 = _mm_add_ps(_mm_mul_ps(a0, b0), c30);
                c31 = _mm_add_ps(_mm_mul_ps(a0, b1), c31);
                c32 = _mm_add_ps(_mm_mul_ps(a0, b2), c32);
                B += ldb;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            _mm_storeu_ps(C + 0 * ldc + 0 * F, _mm_add_ps(_mm_mul_ps(_alpha, c00), _mm_loadu_ps(C + 0 * ldc + 0 * F)));
            _mm_storeu_ps(C + 0 * ldc + 1 * F, _mm_add_ps(_mm_mul_ps(_alpha, c01), _mm_loadu_ps(C + 0 * ldc + 1 * F)));
            _mm_storeu_ps(C + 0 * ldc + 2 * F, _mm_add_ps(_mm_mul_ps(_alpha, c02), _mm_loadu_ps(C + 0 * ldc + 2 * F)));
            _mm_storeu_ps(C + 1 * ldc + 0 * F, _mm_add_ps(_mm_mul_ps(_alpha, c10), _mm_loadu_ps(C + 1 * ldc + 0 * F)));
            _mm_storeu_ps(C + 1 * ldc + 1 * F, _mm_add_ps(_mm_mul_ps(_alpha, c11), _mm_loadu_ps(C + 1 * ldc + 1 * F)));
            _mm_storeu_ps(C + 1 * ldc + 2 * F, _mm_add_ps(_mm_mul_ps(_alpha, c12), _mm_loadu_ps(C + 1 * ldc + 2 * F)));
            _mm_storeu_ps(C + 2 * ldc + 0 * F, _mm_add_ps(_mm_mul_ps(_alpha, c20), _mm_loadu_ps(C + 2 * ldc + 0 * F)));
            _mm_storeu_ps(C + 2 * ldc + 1 * F, _mm_add_ps(_mm_mul_ps(_alpha, c21), _mm_loadu_ps(C + 2 * ldc + 1 * F)));
            _mm_storeu_ps(C + 2 * ldc + 2 * F, _mm_add_ps(_mm_mul_ps(_alpha, c22), _mm_loadu_ps(C + 2 * ldc + 2 * F)));
            _mm_storeu_ps(C + 3 * ldc + 0 * F, _mm_add_ps(_mm_mul_ps(_alpha, c30), _mm_loadu_ps(C + 3 * ldc + 0 * F)));
            _mm_storeu_ps(C + 3 * ldc + 1 * F, _mm_add_ps(_mm_mul_ps(_alpha, c31), _mm_loadu_ps(C + 3 * ldc + 1 * F)));
            _mm_storeu_ps(C + 3 * ldc + 2 * F, _mm_add_ps(_mm_mul_ps(_alpha, c32), _mm_loadu_ps(C + 3 * ldc + 2 * F)));
        }

        SIMD_INLINE void AddDot6x8(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register __m128 c00 = _mm_setzero_ps();
            register __m128 c10 = _mm_setzero_ps();
            register __m128 c20 = _mm_setzero_ps();
            register __m128 c30 = _mm_setzero_ps();
            register __m128 c40 = _mm_setzero_ps();
            register __m128 c50 = _mm_setzero_ps();
            register __m128 c01 = _mm_setzero_ps();
            register __m128 c11 = _mm_setzero_ps();
            register __m128 c21 = _mm_setzero_ps();
            register __m128 c31 = _mm_setzero_ps();
            register __m128 c41 = _mm_setzero_ps();
            register __m128 c51 = _mm_setzero_ps();
            const float * A0 = A + lda * 0;
            const float * A1 = A + lda * 1;
            const float * A2 = A + lda * 2;
            const float * A3 = A + lda * 3;
            const float * A4 = A + lda * 4;
            const float * A5 = A + lda * 5;
            register __m128 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + 0 * F);
                b1 = _mm_loadu_ps(B + 1 * F);
                a0 = _mm_set1_ps(*A0++);
                c00 = _mm_add_ps(_mm_mul_ps(a0, b0), c00);
                c01 = _mm_add_ps(_mm_mul_ps(a0, b1), c01);
                a0 = _mm_set1_ps(*A1++);
                c10 = _mm_add_ps(_mm_mul_ps(a0, b0), c10);
                c11 = _mm_add_ps(_mm_mul_ps(a0, b1), c11);
                a0 = _mm_set1_ps(*A2++);
                c20 = _mm_add_ps(_mm_mul_ps(a0, b0), c20);
                c21 = _mm_add_ps(_mm_mul_ps(a0, b1), c21);
                a0 = _mm_set1_ps(*A3++);
                c30 = _mm_add_ps(_mm_mul_ps(a0, b0), c30);
                c31 = _mm_add_ps(_mm_mul_ps(a0, b1), c31);
                a0 = _mm_set1_ps(*A4++);
                c40 = _mm_add_ps(_mm_mul_ps(a0, b0), c40);
                c41 = _mm_add_ps(_mm_mul_ps(a0, b1), c41);
                a0 = _mm_set1_ps(*A5++);
                c50 = _mm_add_ps(_mm_mul_ps(a0, b0), c50);
                c51 = _mm_add_ps(_mm_mul_ps(a0, b1), c51);
                B += ldb;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            _mm_storeu_ps(C + 0 * ldc + 0 * F, _mm_add_ps(_mm_mul_ps(_alpha, c00), _mm_loadu_ps(C + 0 * ldc + 0 * F)));
            _mm_storeu_ps(C + 0 * ldc + 1 * F, _mm_add_ps(_mm_mul_ps(_alpha, c01), _mm_loadu_ps(C + 0 * ldc + 1 * F)));
            _mm_storeu_ps(C + 1 * ldc + 0 * F, _mm_add_ps(_mm_mul_ps(_alpha, c10), _mm_loadu_ps(C + 1 * ldc + 0 * F)));
            _mm_storeu_ps(C + 1 * ldc + 1 * F, _mm_add_ps(_mm_mul_ps(_alpha, c11), _mm_loadu_ps(C + 1 * ldc + 1 * F)));
            _mm_storeu_ps(C + 2 * ldc + 0 * F, _mm_add_ps(_mm_mul_ps(_alpha, c20), _mm_loadu_ps(C + 2 * ldc + 0 * F)));
            _mm_storeu_ps(C + 2 * ldc + 1 * F, _mm_add_ps(_mm_mul_ps(_alpha, c21), _mm_loadu_ps(C + 2 * ldc + 1 * F)));
            _mm_storeu_ps(C + 3 * ldc + 0 * F, _mm_add_ps(_mm_mul_ps(_alpha, c30), _mm_loadu_ps(C + 3 * ldc + 0 * F)));
            _mm_storeu_ps(C + 3 * ldc + 1 * F, _mm_add_ps(_mm_mul_ps(_alpha, c31), _mm_loadu_ps(C + 3 * ldc + 1 * F)));
            _mm_storeu_ps(C + 4 * ldc + 0 * F, _mm_add_ps(_mm_mul_ps(_alpha, c40), _mm_loadu_ps(C + 4 * ldc + 0 * F)));
            _mm_storeu_ps(C + 4 * ldc + 1 * F, _mm_add_ps(_mm_mul_ps(_alpha, c41), _mm_loadu_ps(C + 4 * ldc + 1 * F)));
            _mm_storeu_ps(C + 5 * ldc + 0 * F, _mm_add_ps(_mm_mul_ps(_alpha, c50), _mm_loadu_ps(C + 5 * ldc + 0 * F)));
            _mm_storeu_ps(C + 5 * ldc + 1 * F, _mm_add_ps(_mm_mul_ps(_alpha, c51), _mm_loadu_ps(C + 5 * ldc + 1 * F)));
        }

        SIMD_INLINE void MulBy(float * ptr, size_t stride, size_t height, size_t width, float value)
        {
            size_t aligned = AlignLo(width, QF);
            size_t partial = AlignLo(width, F);
            __m128 _value = _mm_set1_ps(value);
            for (size_t i = 0; i < height; ++i)
            {
                size_t j = 0;
                for (; j < aligned; j += QF)
                {
                    MulBy(ptr + j + F * 0, _value);
                    MulBy(ptr + j + F * 1, _value);
                    MulBy(ptr + j + F * 2, _value);
                    MulBy(ptr + j + F * 3, _value);
                }
                for (; j < partial; j += F)
                    MulBy(ptr + j, _value);
                for (; j < width; ++j)
                    ptr[j] *= value;
                ptr += stride;
            }
        }

        void Gemm32fNN(size_t M, size_t N, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float beta, float * C, size_t ldc)
        {
            MulBy(C, ldc, M, N, beta);

            size_t M6 = M / 6 * 6;
            size_t M4 = AlignLo(M, 4);
            size_t N12 = N / 12 * 12;
            size_t N8 = AlignLo(N, 8);
            size_t N4 = AlignLo(N, 4);
            size_t i = 0;
#if 0
            for (; i < M6; i += 6)
            {
                size_t j = 0;
                for (; j < N8; j += 8)
                    AddDot6x8(K, alpha, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
                for (0; j < N; ++j)
                    AddDot6x1(K, alpha, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
            }
#else
            for (; i < M4; i += 4)
            {
                size_t j = 0;
                for (; j < N12; j += 12)
                    AddDot4x12(K, alpha, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
                if (j < N - 8)
                {
                    AddDot4x8(K, alpha, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
                    j += 8;
                }
                for (; j < N4; j += 4)
                    AddDot4x4(K, alpha, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
                for (0; j < N; ++j)
                    AddDot4x1(K, alpha, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
            }
#endif
            for (; i < M; i += 1)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    AddDot1x4(K, alpha, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
                for (0; j < N; ++j)
                    AddDot1x1(K, alpha, A + i * lda, lda, B + j, ldb, C + i * ldc + j, ldc);
            }
        }

        void Gemm32fNN(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            const size_t MB = 256;
            const size_t NB = 128;

            for (size_t i = 0; i < M; i += MB)
            {
                size_t MS = Min(M, i + MB) - i;
                for (size_t j = 0; j < N; j += NB)
                {
                    size_t NS = Min(N, j + NB) - j;
                    Gemm32fNN(MS, NS, K, *alpha, A + i * lda, lda, B + j, ldb, *beta, C + i * ldc + j, ldc);
                }
            }
        }
    }
#endif// SIMD_SSE_ENABLE
}
