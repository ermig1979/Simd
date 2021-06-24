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
#include "Simd/SimdAvx2.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdPrefetch.h"

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        SIMD_INLINE void Add4ExtractedSums(const __m512 & sum0, const __m512 & sum1, const __m512 & sum2, const __m512 & sum3, const __m128 & alpha, float * dst)
        {
            __m512 sum02 = _mm512_add_ps(_mm512_unpacklo_ps(sum0, sum2), _mm512_unpackhi_ps(sum0, sum2));
            __m512 sum13 = _mm512_add_ps(_mm512_unpacklo_ps(sum1, sum3), _mm512_unpackhi_ps(sum1, sum3));
            __m512 sum512 = _mm512_add_ps(_mm512_unpacklo_ps(sum02, sum13), _mm512_unpackhi_ps(sum02, sum13));
            __m128 sum128 = _mm_add_ps(_mm_add_ps(_mm512_extractf32x4_ps(sum512, 0), _mm512_extractf32x4_ps(sum512, 1)),
                _mm_add_ps(_mm512_extractf32x4_ps(sum512, 2), _mm512_extractf32x4_ps(sum512, 3)));
            _mm_storeu_ps(dst, _mm_fmadd_ps(alpha, sum128, _mm_loadu_ps(dst)));
        }

        static void Kernel1x1x16nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K16 = K & (~15);
            const float * A0 = A + 0 * lda;
            const float * B0 = B + 0 * ldb;
            __m512 c00 = _mm512_setzero_ps();
            __m512 a0, b0;
            size_t k = 0;
            for (; k < K16; k += 16)
            {
                a0 = _mm512_loadu_ps(A0 + k);
                b0 = _mm512_loadu_ps(B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
            }
            if (k < K)
            {
                __mmask16 tail = __mmask16(-1) >> (16 + k - K);
                a0 = _mm512_maskz_loadu_ps(tail, A0 + k);
                b0 = _mm512_maskz_loadu_ps(tail, B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
            }
            C[0] += alpha * Avx512f::ExtractSum(c00);
        }

        static void Kernel1x4x16nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K16 = K & (~15);
            const float * A0 = A + 0 * lda;
            const float * B0 = B + 0 * ldb;
            const float * B1 = B + 1 * ldb;
            const float * B2 = B + 2 * ldb;
            const float * B3 = B + 3 * ldb;
            __m512 c00 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c02 = _mm512_setzero_ps();
            __m512 c03 = _mm512_setzero_ps();
            __m512 a0, b0;
            size_t k = 0;
            for (; k < K16; k += 16)
            {
                a0 = _mm512_loadu_ps(A0 + k);
                b0 = _mm512_loadu_ps(B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                b0 = _mm512_loadu_ps(B1 + k);
                c01 = _mm512_fmadd_ps(a0, b0, c01);
                b0 = _mm512_loadu_ps(B2 + k);
                c02 = _mm512_fmadd_ps(a0, b0, c02);
                b0 = _mm512_loadu_ps(B3 + k);
                c03 = _mm512_fmadd_ps(a0, b0, c03);
            }
            if (k < K)
            {
                __mmask16 tail = __mmask16(-1) >> (16 + k - K);
                a0 = _mm512_maskz_loadu_ps(tail, A0 + k);
                b0 = _mm512_maskz_loadu_ps(tail, B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                b0 = _mm512_maskz_loadu_ps(tail, B1 + k);
                c01 = _mm512_fmadd_ps(a0, b0, c01);
                b0 = _mm512_maskz_loadu_ps(tail, B2 + k);
                c02 = _mm512_fmadd_ps(a0, b0, c02);
                b0 = _mm512_maskz_loadu_ps(tail, B3 + k);
                c03 = _mm512_fmadd_ps(a0, b0, c03);
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            Add4ExtractedSums(c00, c01, c02, c03, _alpha, C + 0 * ldc);
        }

        static void Kernel2x1x16nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K16 = K & (~15);
            const float * A0 = A + 0 * lda;
            const float * A1 = A + 1 * lda;
            const float * B0 = B + 0 * ldb;
            __m512 c00 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 a0, a1, b0;
            size_t k = 0;
            for (; k < K16; k += 16)
            {
                a0 = _mm512_loadu_ps(A0 + k);
                a1 = _mm512_loadu_ps(A1 + k);
                b0 = _mm512_loadu_ps(B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
            }
            if (k < K)
            {
                __mmask16 tail = __mmask16(-1) >> (16 + k - K);
                a0 = _mm512_maskz_loadu_ps(tail, A0 + k);
                a1 = _mm512_maskz_loadu_ps(tail, A1 + k);
                b0 = _mm512_maskz_loadu_ps(tail, B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
            }
            C[0 * ldc] += alpha * Avx512f::ExtractSum(c00);
            C[1 * ldc] += alpha * Avx512f::ExtractSum(c10);
        }

        static void Kernel2x4x16nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K16 = K & (~15);
            const float * A0 = A + 0 * lda;
            const float * A1 = A + 1 * lda;
            const float * B0 = B + 0 * ldb;
            const float * B1 = B + 1 * ldb;
            const float * B2 = B + 2 * ldb;
            const float * B3 = B + 3 * ldb;
            __m512 c00 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c02 = _mm512_setzero_ps();
            __m512 c03 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c11 = _mm512_setzero_ps();
            __m512 c12 = _mm512_setzero_ps();
            __m512 c13 = _mm512_setzero_ps();
            __m512 a0, a1, b0;
            size_t k = 0;
            for (; k < K16; k += 16)
            {
                a0 = _mm512_loadu_ps(A0 + k);
                a1 = _mm512_loadu_ps(A1 + k);
                b0 = _mm512_loadu_ps(B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                b0 = _mm512_loadu_ps(B1 + k);
                c01 = _mm512_fmadd_ps(a0, b0, c01);
                c11 = _mm512_fmadd_ps(a1, b0, c11);
                b0 = _mm512_loadu_ps(B2 + k);
                c02 = _mm512_fmadd_ps(a0, b0, c02);
                c12 = _mm512_fmadd_ps(a1, b0, c12);
                b0 = _mm512_loadu_ps(B3 + k);
                c03 = _mm512_fmadd_ps(a0, b0, c03);
                c13 = _mm512_fmadd_ps(a1, b0, c13);
            }
            if (k < K)
            {
                __mmask16 tail = __mmask16(-1) >> (16 + k - K);
                a0 = _mm512_maskz_loadu_ps(tail, A0 + k);
                a1 = _mm512_maskz_loadu_ps(tail, A1 + k);
                b0 = _mm512_maskz_loadu_ps(tail, B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                b0 = _mm512_maskz_loadu_ps(tail, B1 + k);
                c01 = _mm512_fmadd_ps(a0, b0, c01);
                c11 = _mm512_fmadd_ps(a1, b0, c11);
                b0 = _mm512_maskz_loadu_ps(tail, B2 + k);
                c02 = _mm512_fmadd_ps(a0, b0, c02);
                c12 = _mm512_fmadd_ps(a1, b0, c12);
                b0 = _mm512_maskz_loadu_ps(tail, B3 + k);
                c03 = _mm512_fmadd_ps(a0, b0, c03);
                c13 = _mm512_fmadd_ps(a1, b0, c13);
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            Add4ExtractedSums(c00, c01, c02, c03, _alpha, C + 0 * ldc);
            Add4ExtractedSums(c10, c11, c12, c13, _alpha, C + 1 * ldc);
        }

        static void Kernel3x1x16nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K16 = K & (~15);
            const float * A0 = A + 0 * lda;
            const float * A1 = A + 1 * lda;
            const float * A2 = A + 2 * lda;
            const float * B0 = B + 0 * ldb;
            __m512 c00 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 a0, a1, a2, b0;
            size_t k = 0;
            for (; k < K16; k += 16)
            {
                a0 = _mm512_loadu_ps(A0 + k);
                a1 = _mm512_loadu_ps(A1 + k);
                a2 = _mm512_loadu_ps(A2 + k);
                b0 = _mm512_loadu_ps(B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                c20 = _mm512_fmadd_ps(a2, b0, c20);
            }
            if (k < K)
            {
                __mmask16 tail = __mmask16(-1) >> (16 + k - K);
                a0 = _mm512_maskz_loadu_ps(tail, A0 + k);
                a1 = _mm512_maskz_loadu_ps(tail, A1 + k);
                a2 = _mm512_maskz_loadu_ps(tail, A2 + k);
                b0 = _mm512_maskz_loadu_ps(tail, B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                c20 = _mm512_fmadd_ps(a2, b0, c20);
            }
            C[0 * ldc] += alpha * Avx512f::ExtractSum(c00);
            C[1 * ldc] += alpha * Avx512f::ExtractSum(c10);
            C[2 * ldc] += alpha * Avx512f::ExtractSum(c20);
        }

        static void Kernel3x4x16nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K16 = K & (~15);
            const float * A0 = A + 0 * lda;
            const float * A1 = A + 1 * lda;
            const float * A2 = A + 2 * lda;
            const float * B0 = B + 0 * ldb;
            const float * B1 = B + 1 * ldb;
            const float * B2 = B + 2 * ldb;
            const float * B3 = B + 3 * ldb;
            __m512 c00 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c02 = _mm512_setzero_ps();
            __m512 c03 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c11 = _mm512_setzero_ps();
            __m512 c12 = _mm512_setzero_ps();
            __m512 c13 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c21 = _mm512_setzero_ps();
            __m512 c22 = _mm512_setzero_ps();
            __m512 c23 = _mm512_setzero_ps();
            __m512 a0, a1, a2, b0;
            size_t k = 0;
            for (; k < K16; k += 16)
            {
                a0 = _mm512_loadu_ps(A0 + k);
                a1 = _mm512_loadu_ps(A1 + k);
                a2 = _mm512_loadu_ps(A2 + k);
                b0 = _mm512_loadu_ps(B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                c20 = _mm512_fmadd_ps(a2, b0, c20);
                b0 = _mm512_loadu_ps(B1 + k);
                c01 = _mm512_fmadd_ps(a0, b0, c01);
                c11 = _mm512_fmadd_ps(a1, b0, c11);
                c21 = _mm512_fmadd_ps(a2, b0, c21);
                b0 = _mm512_loadu_ps(B2 + k);
                c02 = _mm512_fmadd_ps(a0, b0, c02);
                c12 = _mm512_fmadd_ps(a1, b0, c12);
                c22 = _mm512_fmadd_ps(a2, b0, c22);
                b0 = _mm512_loadu_ps(B3 + k);
                c03 = _mm512_fmadd_ps(a0, b0, c03);
                c13 = _mm512_fmadd_ps(a1, b0, c13);
                c23 = _mm512_fmadd_ps(a2, b0, c23);
            }
            if (k < K)
            {
                __mmask16 tail = __mmask16(-1) >> (16 + k - K);
                a0 = _mm512_maskz_loadu_ps(tail, A0 + k);
                a1 = _mm512_maskz_loadu_ps(tail, A1 + k);
                a2 = _mm512_maskz_loadu_ps(tail, A2 + k);
                b0 = _mm512_maskz_loadu_ps(tail, B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                c20 = _mm512_fmadd_ps(a2, b0, c20);
                b0 = _mm512_maskz_loadu_ps(tail, B1 + k);
                c01 = _mm512_fmadd_ps(a0, b0, c01);
                c11 = _mm512_fmadd_ps(a1, b0, c11);
                c21 = _mm512_fmadd_ps(a2, b0, c21);
                b0 = _mm512_maskz_loadu_ps(tail, B2 + k);
                c02 = _mm512_fmadd_ps(a0, b0, c02);
                c12 = _mm512_fmadd_ps(a1, b0, c12);
                c22 = _mm512_fmadd_ps(a2, b0, c22);
                b0 = _mm512_maskz_loadu_ps(tail, B3 + k);
                c03 = _mm512_fmadd_ps(a0, b0, c03);
                c13 = _mm512_fmadd_ps(a1, b0, c13);
                c23 = _mm512_fmadd_ps(a2, b0, c23);
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            Add4ExtractedSums(c00, c01, c02, c03, _alpha, C + 0 * ldc);
            Add4ExtractedSums(c10, c11, c12, c13, _alpha, C + 1 * ldc);
            Add4ExtractedSums(c20, c21, c22, c23, _alpha, C + 2 * ldc);
        }

        static void Kernel6x1x16nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K16 = K & (~15);
            const float * A0 = A + 0 * lda;
            const float * A1 = A + 1 * lda;
            const float * A2 = A + 2 * lda;
            const float * A3 = A + 3 * lda;
            const float * A4 = A + 4 * lda;
            const float * A5 = A + 5 * lda;
            const float * B0 = B + 0 * ldb;
            __m512 c00 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps();
            __m512 a0, a1, a2, a3, a4, a5, b0;
            size_t k = 0;
            for (; k < K16; k += 16)
            {
                a0 = _mm512_loadu_ps(A0 + k);
                a1 = _mm512_loadu_ps(A1 + k);
                a2 = _mm512_loadu_ps(A2 + k);
                a3 = _mm512_loadu_ps(A3 + k);
                a4 = _mm512_loadu_ps(A4 + k);
                a5 = _mm512_loadu_ps(A5 + k);
                b0 = _mm512_loadu_ps(B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                c20 = _mm512_fmadd_ps(a2, b0, c20);
                c30 = _mm512_fmadd_ps(a3, b0, c30);
                c40 = _mm512_fmadd_ps(a4, b0, c40);
                c50 = _mm512_fmadd_ps(a5, b0, c50);
            }
            if (k < K)
            {
                __mmask16 tail = __mmask16(-1) >> (16 + k - K);
                a0 = _mm512_maskz_loadu_ps(tail, A0 + k);
                a1 = _mm512_maskz_loadu_ps(tail, A1 + k);
                a2 = _mm512_maskz_loadu_ps(tail, A2 + k);
                a3 = _mm512_maskz_loadu_ps(tail, A3 + k);
                a4 = _mm512_maskz_loadu_ps(tail, A4 + k);
                a5 = _mm512_maskz_loadu_ps(tail, A5 + k);
                b0 = _mm512_maskz_loadu_ps(tail, B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                c20 = _mm512_fmadd_ps(a2, b0, c20);
                c30 = _mm512_fmadd_ps(a3, b0, c30);
                c40 = _mm512_fmadd_ps(a4, b0, c40);
                c50 = _mm512_fmadd_ps(a5, b0, c50);
            }
            C[0 * ldc] += alpha * Avx512f::ExtractSum(c00);
            C[1 * ldc] += alpha * Avx512f::ExtractSum(c10);
            C[2 * ldc] += alpha * Avx512f::ExtractSum(c20);
            C[3 * ldc] += alpha * Avx512f::ExtractSum(c30);
            C[4 * ldc] += alpha * Avx512f::ExtractSum(c40);
            C[5 * ldc] += alpha * Avx512f::ExtractSum(c50);
        }

        static void Kernel6x4x16nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K16 = K & (~15);
            const float * A0 = A + 0 * lda;
            const float * A1 = A + 1 * lda;
            const float * A2 = A + 2 * lda;
            const float * A3 = A + 3 * lda;
            const float * A4 = A + 4 * lda;
            const float * A5 = A + 5 * lda;
            const float * B0 = B + 0 * ldb;
            const float * B1 = B + 1 * ldb;
            const float * B2 = B + 2 * ldb;
            const float * B3 = B + 3 * ldb;
            __m512 c00 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c02 = _mm512_setzero_ps();
            __m512 c03 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c11 = _mm512_setzero_ps();
            __m512 c12 = _mm512_setzero_ps();
            __m512 c13 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c21 = _mm512_setzero_ps();
            __m512 c22 = _mm512_setzero_ps();
            __m512 c23 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c31 = _mm512_setzero_ps();
            __m512 c32 = _mm512_setzero_ps();
            __m512 c33 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps();
            __m512 c41 = _mm512_setzero_ps();
            __m512 c42 = _mm512_setzero_ps();
            __m512 c43 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps();
            __m512 c51 = _mm512_setzero_ps();
            __m512 c52 = _mm512_setzero_ps();
            __m512 c53 = _mm512_setzero_ps();
            __m512 a0, a1, a2, a3, a4, a5, b0;
            size_t k = 0;
            for (; k < K16; k += 16)
            {
                a0 = _mm512_loadu_ps(A0 + k);
                a1 = _mm512_loadu_ps(A1 + k);
                a2 = _mm512_loadu_ps(A2 + k);
                a3 = _mm512_loadu_ps(A3 + k);
                a4 = _mm512_loadu_ps(A4 + k);
                a5 = _mm512_loadu_ps(A5 + k);
                b0 = _mm512_loadu_ps(B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                c20 = _mm512_fmadd_ps(a2, b0, c20);
                c30 = _mm512_fmadd_ps(a3, b0, c30);
                c40 = _mm512_fmadd_ps(a4, b0, c40);
                c50 = _mm512_fmadd_ps(a5, b0, c50);
                b0 = _mm512_loadu_ps(B1 + k);
                c01 = _mm512_fmadd_ps(a0, b0, c01);
                c11 = _mm512_fmadd_ps(a1, b0, c11);
                c21 = _mm512_fmadd_ps(a2, b0, c21);
                c31 = _mm512_fmadd_ps(a3, b0, c31);
                c41 = _mm512_fmadd_ps(a4, b0, c41);
                c51 = _mm512_fmadd_ps(a5, b0, c51);
                b0 = _mm512_loadu_ps(B2 + k);
                c02 = _mm512_fmadd_ps(a0, b0, c02);
                c12 = _mm512_fmadd_ps(a1, b0, c12);
                c22 = _mm512_fmadd_ps(a2, b0, c22);
                c32 = _mm512_fmadd_ps(a3, b0, c32);
                c42 = _mm512_fmadd_ps(a4, b0, c42);
                c52 = _mm512_fmadd_ps(a5, b0, c52);
                b0 = _mm512_loadu_ps(B3 + k);
                c03 = _mm512_fmadd_ps(a0, b0, c03);
                c13 = _mm512_fmadd_ps(a1, b0, c13);
                c23 = _mm512_fmadd_ps(a2, b0, c23);
                c33 = _mm512_fmadd_ps(a3, b0, c33);
                c43 = _mm512_fmadd_ps(a4, b0, c43);
                c53 = _mm512_fmadd_ps(a5, b0, c53);
            }
            if (k < K)
            {
                __mmask16 tail = __mmask16(-1) >> (16 + k - K);
                a0 = _mm512_maskz_loadu_ps(tail, A0 + k);
                a1 = _mm512_maskz_loadu_ps(tail, A1 + k);
                a2 = _mm512_maskz_loadu_ps(tail, A2 + k);
                a3 = _mm512_maskz_loadu_ps(tail, A3 + k);
                a4 = _mm512_maskz_loadu_ps(tail, A4 + k);
                a5 = _mm512_maskz_loadu_ps(tail, A5 + k);
                b0 = _mm512_maskz_loadu_ps(tail, B0 + k);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                c20 = _mm512_fmadd_ps(a2, b0, c20);
                c30 = _mm512_fmadd_ps(a3, b0, c30);
                c40 = _mm512_fmadd_ps(a4, b0, c40);
                c50 = _mm512_fmadd_ps(a5, b0, c50);
                b0 = _mm512_maskz_loadu_ps(tail, B1 + k);
                c01 = _mm512_fmadd_ps(a0, b0, c01);
                c11 = _mm512_fmadd_ps(a1, b0, c11);
                c21 = _mm512_fmadd_ps(a2, b0, c21);
                c31 = _mm512_fmadd_ps(a3, b0, c31);
                c41 = _mm512_fmadd_ps(a4, b0, c41);
                c51 = _mm512_fmadd_ps(a5, b0, c51);
                b0 = _mm512_maskz_loadu_ps(tail, B2 + k);
                c02 = _mm512_fmadd_ps(a0, b0, c02);
                c12 = _mm512_fmadd_ps(a1, b0, c12);
                c22 = _mm512_fmadd_ps(a2, b0, c22);
                c32 = _mm512_fmadd_ps(a3, b0, c32);
                c42 = _mm512_fmadd_ps(a4, b0, c42);
                c52 = _mm512_fmadd_ps(a5, b0, c52);
                b0 = _mm512_maskz_loadu_ps(tail, B3 + k);
                c03 = _mm512_fmadd_ps(a0, b0, c03);
                c13 = _mm512_fmadd_ps(a1, b0, c13);
                c23 = _mm512_fmadd_ps(a2, b0, c23);
                c33 = _mm512_fmadd_ps(a3, b0, c33);
                c43 = _mm512_fmadd_ps(a4, b0, c43);
                c53 = _mm512_fmadd_ps(a5, b0, c53);
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            Add4ExtractedSums(c00, c01, c02, c03, _alpha, C + 0 * ldc);
            Add4ExtractedSums(c10, c11, c12, c13, _alpha, C + 1 * ldc);
            Add4ExtractedSums(c20, c21, c22, c23, _alpha, C + 2 * ldc);
            Add4ExtractedSums(c30, c31, c32, c33, _alpha, C + 3 * ldc);
            Add4ExtractedSums(c40, c41, c42, c43, _alpha, C + 4 * ldc);
            Add4ExtractedSums(c50, c51, c52, c53, _alpha, C + 5 * ldc);
        }

        void Gemm32fNT(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            SIMD_PERF_BEGF(Simd::ToStr(M) + "-" + Simd::ToStr(N) + "-" + Simd::ToStr(K), M*N*K * 2);
            if (K <= Avx2::F)
            {
                Avx2::Gemm32fNT(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
                return;
            }
            typedef Simd::GemmNT<float, F> GemmNT;
#if SIMD_ZMM_COUNT == 32
            GemmNT gemmNT(M, N, K, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), Avx::GemmScaleC,
                Kernel1x1x16nt, Kernel1x4x16nt, Kernel2x1x16nt, Kernel2x4x16nt, Kernel3x1x16nt, Kernel3x4x16nt, Kernel6x1x16nt, Kernel6x4x16nt);
#elif defined(SIMD_X64_ENABLE)
            GemmNT gemmNT(M, N, K, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), Avx::GemmScaleC,
                Kernel1x1x16nt, Kernel1x4x16nt, Kernel2x1x16nt, Kernel2x4x16nt, Kernel3x1x16nt, Kernel3x4x16nt, NULL, NULL);
#else
            GemmNT gemmNT(M, N, K, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), Sse2::GemmScaleC,
                Kernel1x1x16nt, Kernel1x4x16nt, NULL, NULL, NULL, NULL, NULL, NULL);
#endif
            gemmNT.Run(alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
#endif// SIMD_AVX512F_ENABLE
}
