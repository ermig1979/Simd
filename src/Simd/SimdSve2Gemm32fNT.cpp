/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#include "Simd/SimdGemm.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE float ExtractSum32f(const svfloat32_t& value)
        {
            return svaddv_f32(svptrue_b32(), value);
        }

        SIMD_INLINE void Add4ExtractedSums(const svfloat32_t& sum0, const svfloat32_t& sum1,
            const svfloat32_t& sum2, const svfloat32_t& sum3, float alpha, float* dst)
        {
            dst[0] += alpha * ExtractSum32f(sum0);
            dst[1] += alpha * ExtractSum32f(sum1);
            dst[2] += alpha * ExtractSum32f(sum2);
            dst[3] += alpha * ExtractSum32f(sum3);
        }

        static void Kernel1x1nt(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            size_t F = svcntw(), DF = 2 * F, k = 0;
            const float* A0 = A + 0 * lda;
            const float* B0 = B + 0 * ldb;
            const svbool_t body = svptrue_b32();
            svfloat32_t c00 = svdup_n_f32(0.0f);
            svfloat32_t c00b = svdup_n_f32(0.0f);
            for (; k + DF <= K; k += DF)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
                a0 = svld1_f32(body, A0 + k + F);
                b0 = svld1_f32(body, B0 + k + F);
                c00b = svmla_f32_x(body, c00b, a0, b0);
            }
            for (; k + F <= K; k += F)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                svfloat32_t a0 = svld1_f32(mask, A0 + k);
                svfloat32_t b0 = svld1_f32(mask, B0 + k);
                c00 = svmla_f32_m(mask, c00, a0, b0);
            }
            C[0] += alpha * ExtractSum32f(svadd_f32_x(body, c00, c00b));
        }

        static void Kernel1x4nt(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            size_t F = svcntw(), DF = 2 * F, k = 0;
            const float* A0 = A + 0 * lda;
            const float* B0 = B + 0 * ldb;
            const float* B1 = B + 1 * ldb;
            const float* B2 = B + 2 * ldb;
            const float* B3 = B + 3 * ldb;
            const svbool_t body = svptrue_b32();
            svfloat32_t c00 = svdup_n_f32(0.0f);
            svfloat32_t c01 = svdup_n_f32(0.0f);
            svfloat32_t c02 = svdup_n_f32(0.0f);
            svfloat32_t c03 = svdup_n_f32(0.0f);
            svfloat32_t c00b = svdup_n_f32(0.0f);
            svfloat32_t c01b = svdup_n_f32(0.0f);
            svfloat32_t c02b = svdup_n_f32(0.0f);
            svfloat32_t c03b = svdup_n_f32(0.0f);
            for (; k + DF <= K; k += DF)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
                b0 = svld1_f32(body, B1 + k);
                c01 = svmla_f32_x(body, c01, a0, b0);
                b0 = svld1_f32(body, B2 + k);
                c02 = svmla_f32_x(body, c02, a0, b0);
                b0 = svld1_f32(body, B3 + k);
                c03 = svmla_f32_x(body, c03, a0, b0);
                a0 = svld1_f32(body, A0 + k + F);
                b0 = svld1_f32(body, B0 + k + F);
                c00b = svmla_f32_x(body, c00b, a0, b0);
                b0 = svld1_f32(body, B1 + k + F);
                c01b = svmla_f32_x(body, c01b, a0, b0);
                b0 = svld1_f32(body, B2 + k + F);
                c02b = svmla_f32_x(body, c02b, a0, b0);
                b0 = svld1_f32(body, B3 + k + F);
                c03b = svmla_f32_x(body, c03b, a0, b0);
            }
            for (; k + F <= K; k += F)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
                b0 = svld1_f32(body, B1 + k);
                c01 = svmla_f32_x(body, c01, a0, b0);
                b0 = svld1_f32(body, B2 + k);
                c02 = svmla_f32_x(body, c02, a0, b0);
                b0 = svld1_f32(body, B3 + k);
                c03 = svmla_f32_x(body, c03, a0, b0);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                svfloat32_t a0 = svld1_f32(mask, A0 + k);
                svfloat32_t b0 = svld1_f32(mask, B0 + k);
                c00 = svmla_f32_m(mask, c00, a0, b0);
                b0 = svld1_f32(mask, B1 + k);
                c01 = svmla_f32_m(mask, c01, a0, b0);
                b0 = svld1_f32(mask, B2 + k);
                c02 = svmla_f32_m(mask, c02, a0, b0);
                b0 = svld1_f32(mask, B3 + k);
                c03 = svmla_f32_m(mask, c03, a0, b0);
            }
            Add4ExtractedSums(
                svadd_f32_x(body, c00, c00b),
                svadd_f32_x(body, c01, c01b),
                svadd_f32_x(body, c02, c02b),
                svadd_f32_x(body, c03, c03b),
                alpha, C + 0 * ldc);
        }

        static void Kernel2x1nt(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            size_t F = svcntw(), DF = 2 * F, k = 0;
            const float* A0 = A + 0 * lda;
            const float* A1 = A + 1 * lda;
            const float* B0 = B + 0 * ldb;
            const svbool_t body = svptrue_b32();
            svfloat32_t c00 = svdup_n_f32(0.0f);
            svfloat32_t c10 = svdup_n_f32(0.0f);
            svfloat32_t c00b = svdup_n_f32(0.0f);
            svfloat32_t c10b = svdup_n_f32(0.0f);
            for (; k + DF <= K; k += DF)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t a1 = svld1_f32(body, A1 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
                a0 = svld1_f32(body, A0 + k + F);
                a1 = svld1_f32(body, A1 + k + F);
                b0 = svld1_f32(body, B0 + k + F);
                c00b = svmla_f32_x(body, c00b, a0, b0);
                c10b = svmla_f32_x(body, c10b, a1, b0);
            }
            for (; k + F <= K; k += F)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t a1 = svld1_f32(body, A1 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                svfloat32_t a0 = svld1_f32(mask, A0 + k);
                svfloat32_t a1 = svld1_f32(mask, A1 + k);
                svfloat32_t b0 = svld1_f32(mask, B0 + k);
                c00 = svmla_f32_m(mask, c00, a0, b0);
                c10 = svmla_f32_m(mask, c10, a1, b0);
            }
            C[0 * ldc] += alpha * ExtractSum32f(svadd_f32_x(body, c00, c00b));
            C[1 * ldc] += alpha * ExtractSum32f(svadd_f32_x(body, c10, c10b));
        }

        static void Kernel2x4nt(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            size_t F = svcntw(), DF = 2 * F, k = 0;
            const float* A0 = A + 0 * lda;
            const float* A1 = A + 1 * lda;
            const float* B0 = B + 0 * ldb;
            const float* B1 = B + 1 * ldb;
            const float* B2 = B + 2 * ldb;
            const float* B3 = B + 3 * ldb;
            const svbool_t body = svptrue_b32();
            svfloat32_t c00 = svdup_n_f32(0.0f);
            svfloat32_t c01 = svdup_n_f32(0.0f);
            svfloat32_t c02 = svdup_n_f32(0.0f);
            svfloat32_t c03 = svdup_n_f32(0.0f);
            svfloat32_t c10 = svdup_n_f32(0.0f);
            svfloat32_t c11 = svdup_n_f32(0.0f);
            svfloat32_t c12 = svdup_n_f32(0.0f);
            svfloat32_t c13 = svdup_n_f32(0.0f);
            for (; k + DF <= K; k += DF)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t a1 = svld1_f32(body, A1 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
                b0 = svld1_f32(body, B1 + k);
                c01 = svmla_f32_x(body, c01, a0, b0);
                c11 = svmla_f32_x(body, c11, a1, b0);
                b0 = svld1_f32(body, B2 + k);
                c02 = svmla_f32_x(body, c02, a0, b0);
                c12 = svmla_f32_x(body, c12, a1, b0);
                b0 = svld1_f32(body, B3 + k);
                c03 = svmla_f32_x(body, c03, a0, b0);
                c13 = svmla_f32_x(body, c13, a1, b0);
                a0 = svld1_f32(body, A0 + k + F);
                a1 = svld1_f32(body, A1 + k + F);
                b0 = svld1_f32(body, B0 + k + F);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
                b0 = svld1_f32(body, B1 + k + F);
                c01 = svmla_f32_x(body, c01, a0, b0);
                c11 = svmla_f32_x(body, c11, a1, b0);
                b0 = svld1_f32(body, B2 + k + F);
                c02 = svmla_f32_x(body, c02, a0, b0);
                c12 = svmla_f32_x(body, c12, a1, b0);
                b0 = svld1_f32(body, B3 + k + F);
                c03 = svmla_f32_x(body, c03, a0, b0);
                c13 = svmla_f32_x(body, c13, a1, b0);
            }
            for (; k + F <= K; k += F)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t a1 = svld1_f32(body, A1 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
                b0 = svld1_f32(body, B1 + k);
                c01 = svmla_f32_x(body, c01, a0, b0);
                c11 = svmla_f32_x(body, c11, a1, b0);
                b0 = svld1_f32(body, B2 + k);
                c02 = svmla_f32_x(body, c02, a0, b0);
                c12 = svmla_f32_x(body, c12, a1, b0);
                b0 = svld1_f32(body, B3 + k);
                c03 = svmla_f32_x(body, c03, a0, b0);
                c13 = svmla_f32_x(body, c13, a1, b0);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                svfloat32_t a0 = svld1_f32(mask, A0 + k);
                svfloat32_t a1 = svld1_f32(mask, A1 + k);
                svfloat32_t b0 = svld1_f32(mask, B0 + k);
                c00 = svmla_f32_m(mask, c00, a0, b0);
                c10 = svmla_f32_m(mask, c10, a1, b0);
                b0 = svld1_f32(mask, B1 + k);
                c01 = svmla_f32_m(mask, c01, a0, b0);
                c11 = svmla_f32_m(mask, c11, a1, b0);
                b0 = svld1_f32(mask, B2 + k);
                c02 = svmla_f32_m(mask, c02, a0, b0);
                c12 = svmla_f32_m(mask, c12, a1, b0);
                b0 = svld1_f32(mask, B3 + k);
                c03 = svmla_f32_m(mask, c03, a0, b0);
                c13 = svmla_f32_m(mask, c13, a1, b0);
            }
            Add4ExtractedSums(c00, c01, c02, c03, alpha, C + 0 * ldc);
            Add4ExtractedSums(c10, c11, c12, c13, alpha, C + 1 * ldc);
        }

        static void Kernel3x1nt(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            size_t F = svcntw(), DF = 2 * F, k = 0;
            const float* A0 = A + 0 * lda;
            const float* A1 = A + 1 * lda;
            const float* A2 = A + 2 * lda;
            const float* B0 = B + 0 * ldb;
            const svbool_t body = svptrue_b32();
            svfloat32_t c00 = svdup_n_f32(0.0f);
            svfloat32_t c10 = svdup_n_f32(0.0f);
            svfloat32_t c20 = svdup_n_f32(0.0f);
            svfloat32_t c00b = svdup_n_f32(0.0f);
            svfloat32_t c10b = svdup_n_f32(0.0f);
            svfloat32_t c20b = svdup_n_f32(0.0f);
            for (; k + DF <= K; k += DF)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t a1 = svld1_f32(body, A1 + k);
                svfloat32_t a2 = svld1_f32(body, A2 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
                c20 = svmla_f32_x(body, c20, a2, b0);
                a0 = svld1_f32(body, A0 + k + F);
                a1 = svld1_f32(body, A1 + k + F);
                a2 = svld1_f32(body, A2 + k + F);
                b0 = svld1_f32(body, B0 + k + F);
                c00b = svmla_f32_x(body, c00b, a0, b0);
                c10b = svmla_f32_x(body, c10b, a1, b0);
                c20b = svmla_f32_x(body, c20b, a2, b0);
            }
            for (; k + F <= K; k += F)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t a1 = svld1_f32(body, A1 + k);
                svfloat32_t a2 = svld1_f32(body, A2 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
                c20 = svmla_f32_x(body, c20, a2, b0);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                svfloat32_t a0 = svld1_f32(mask, A0 + k);
                svfloat32_t a1 = svld1_f32(mask, A1 + k);
                svfloat32_t a2 = svld1_f32(mask, A2 + k);
                svfloat32_t b0 = svld1_f32(mask, B0 + k);
                c00 = svmla_f32_m(mask, c00, a0, b0);
                c10 = svmla_f32_m(mask, c10, a1, b0);
                c20 = svmla_f32_m(mask, c20, a2, b0);
            }
            C[0 * ldc] += alpha * ExtractSum32f(svadd_f32_x(body, c00, c00b));
            C[1 * ldc] += alpha * ExtractSum32f(svadd_f32_x(body, c10, c10b));
            C[2 * ldc] += alpha * ExtractSum32f(svadd_f32_x(body, c20, c20b));
        }

        static void Kernel3x4nt(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            size_t F = svcntw(), DF = 2 * F, k = 0;
            const float* A0 = A + 0 * lda;
            const float* A1 = A + 1 * lda;
            const float* A2 = A + 2 * lda;
            const float* B0 = B + 0 * ldb;
            const float* B1 = B + 1 * ldb;
            const float* B2 = B + 2 * ldb;
            const float* B3 = B + 3 * ldb;
            const svbool_t body = svptrue_b32();
            svfloat32_t c00 = svdup_n_f32(0.0f);
            svfloat32_t c01 = svdup_n_f32(0.0f);
            svfloat32_t c02 = svdup_n_f32(0.0f);
            svfloat32_t c03 = svdup_n_f32(0.0f);
            svfloat32_t c10 = svdup_n_f32(0.0f);
            svfloat32_t c11 = svdup_n_f32(0.0f);
            svfloat32_t c12 = svdup_n_f32(0.0f);
            svfloat32_t c13 = svdup_n_f32(0.0f);
            svfloat32_t c20 = svdup_n_f32(0.0f);
            svfloat32_t c21 = svdup_n_f32(0.0f);
            svfloat32_t c22 = svdup_n_f32(0.0f);
            svfloat32_t c23 = svdup_n_f32(0.0f);
            for (; k + DF <= K; k += DF)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t a1 = svld1_f32(body, A1 + k);
                svfloat32_t a2 = svld1_f32(body, A2 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
                c20 = svmla_f32_x(body, c20, a2, b0);
                b0 = svld1_f32(body, B1 + k);
                c01 = svmla_f32_x(body, c01, a0, b0);
                c11 = svmla_f32_x(body, c11, a1, b0);
                c21 = svmla_f32_x(body, c21, a2, b0);
                b0 = svld1_f32(body, B2 + k);
                c02 = svmla_f32_x(body, c02, a0, b0);
                c12 = svmla_f32_x(body, c12, a1, b0);
                c22 = svmla_f32_x(body, c22, a2, b0);
                b0 = svld1_f32(body, B3 + k);
                c03 = svmla_f32_x(body, c03, a0, b0);
                c13 = svmla_f32_x(body, c13, a1, b0);
                c23 = svmla_f32_x(body, c23, a2, b0);
                a0 = svld1_f32(body, A0 + k + F);
                a1 = svld1_f32(body, A1 + k + F);
                a2 = svld1_f32(body, A2 + k + F);
                b0 = svld1_f32(body, B0 + k + F);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
                c20 = svmla_f32_x(body, c20, a2, b0);
                b0 = svld1_f32(body, B1 + k + F);
                c01 = svmla_f32_x(body, c01, a0, b0);
                c11 = svmla_f32_x(body, c11, a1, b0);
                c21 = svmla_f32_x(body, c21, a2, b0);
                b0 = svld1_f32(body, B2 + k + F);
                c02 = svmla_f32_x(body, c02, a0, b0);
                c12 = svmla_f32_x(body, c12, a1, b0);
                c22 = svmla_f32_x(body, c22, a2, b0);
                b0 = svld1_f32(body, B3 + k + F);
                c03 = svmla_f32_x(body, c03, a0, b0);
                c13 = svmla_f32_x(body, c13, a1, b0);
                c23 = svmla_f32_x(body, c23, a2, b0);
            }
            for (; k + F <= K; k += F)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t a1 = svld1_f32(body, A1 + k);
                svfloat32_t a2 = svld1_f32(body, A2 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
                c20 = svmla_f32_x(body, c20, a2, b0);
                b0 = svld1_f32(body, B1 + k);
                c01 = svmla_f32_x(body, c01, a0, b0);
                c11 = svmla_f32_x(body, c11, a1, b0);
                c21 = svmla_f32_x(body, c21, a2, b0);
                b0 = svld1_f32(body, B2 + k);
                c02 = svmla_f32_x(body, c02, a0, b0);
                c12 = svmla_f32_x(body, c12, a1, b0);
                c22 = svmla_f32_x(body, c22, a2, b0);
                b0 = svld1_f32(body, B3 + k);
                c03 = svmla_f32_x(body, c03, a0, b0);
                c13 = svmla_f32_x(body, c13, a1, b0);
                c23 = svmla_f32_x(body, c23, a2, b0);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                svfloat32_t a0 = svld1_f32(mask, A0 + k);
                svfloat32_t a1 = svld1_f32(mask, A1 + k);
                svfloat32_t a2 = svld1_f32(mask, A2 + k);
                svfloat32_t b0 = svld1_f32(mask, B0 + k);
                c00 = svmla_f32_m(mask, c00, a0, b0);
                c10 = svmla_f32_m(mask, c10, a1, b0);
                c20 = svmla_f32_m(mask, c20, a2, b0);
                b0 = svld1_f32(mask, B1 + k);
                c01 = svmla_f32_m(mask, c01, a0, b0);
                c11 = svmla_f32_m(mask, c11, a1, b0);
                c21 = svmla_f32_m(mask, c21, a2, b0);
                b0 = svld1_f32(mask, B2 + k);
                c02 = svmla_f32_m(mask, c02, a0, b0);
                c12 = svmla_f32_m(mask, c12, a1, b0);
                c22 = svmla_f32_m(mask, c22, a2, b0);
                b0 = svld1_f32(mask, B3 + k);
                c03 = svmla_f32_m(mask, c03, a0, b0);
                c13 = svmla_f32_m(mask, c13, a1, b0);
                c23 = svmla_f32_m(mask, c23, a2, b0);
            }
            Add4ExtractedSums(c00, c01, c02, c03, alpha, C + 0 * ldc);
            Add4ExtractedSums(c10, c11, c12, c13, alpha, C + 1 * ldc);
            Add4ExtractedSums(c20, c21, c22, c23, alpha, C + 2 * ldc);
        }

        static void Kernel6x1nt(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            size_t F = svcntw(), DF = 2 * F, k = 0;
            const float* A0 = A + 0 * lda;
            const float* A1 = A + 1 * lda;
            const float* A2 = A + 2 * lda;
            const float* A3 = A + 3 * lda;
            const float* A4 = A + 4 * lda;
            const float* A5 = A + 5 * lda;
            const float* B0 = B + 0 * ldb;
            const svbool_t body = svptrue_b32();
            svfloat32_t c00 = svdup_n_f32(0.0f);
            svfloat32_t c10 = svdup_n_f32(0.0f);
            svfloat32_t c20 = svdup_n_f32(0.0f);
            svfloat32_t c30 = svdup_n_f32(0.0f);
            svfloat32_t c40 = svdup_n_f32(0.0f);
            svfloat32_t c50 = svdup_n_f32(0.0f);
            for (; k + DF <= K; k += DF)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t a1 = svld1_f32(body, A1 + k);
                svfloat32_t a2 = svld1_f32(body, A2 + k);
                svfloat32_t a3 = svld1_f32(body, A3 + k);
                svfloat32_t a4 = svld1_f32(body, A4 + k);
                svfloat32_t a5 = svld1_f32(body, A5 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
                c20 = svmla_f32_x(body, c20, a2, b0);
                c30 = svmla_f32_x(body, c30, a3, b0);
                c40 = svmla_f32_x(body, c40, a4, b0);
                c50 = svmla_f32_x(body, c50, a5, b0);
                a0 = svld1_f32(body, A0 + k + F);
                a1 = svld1_f32(body, A1 + k + F);
                a2 = svld1_f32(body, A2 + k + F);
                a3 = svld1_f32(body, A3 + k + F);
                a4 = svld1_f32(body, A4 + k + F);
                a5 = svld1_f32(body, A5 + k + F);
                b0 = svld1_f32(body, B0 + k + F);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
                c20 = svmla_f32_x(body, c20, a2, b0);
                c30 = svmla_f32_x(body, c30, a3, b0);
                c40 = svmla_f32_x(body, c40, a4, b0);
                c50 = svmla_f32_x(body, c50, a5, b0);
            }
            for (; k + F <= K; k += F)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t a1 = svld1_f32(body, A1 + k);
                svfloat32_t a2 = svld1_f32(body, A2 + k);
                svfloat32_t a3 = svld1_f32(body, A3 + k);
                svfloat32_t a4 = svld1_f32(body, A4 + k);
                svfloat32_t a5 = svld1_f32(body, A5 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
                c20 = svmla_f32_x(body, c20, a2, b0);
                c30 = svmla_f32_x(body, c30, a3, b0);
                c40 = svmla_f32_x(body, c40, a4, b0);
                c50 = svmla_f32_x(body, c50, a5, b0);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                svfloat32_t a0 = svld1_f32(mask, A0 + k);
                svfloat32_t a1 = svld1_f32(mask, A1 + k);
                svfloat32_t a2 = svld1_f32(mask, A2 + k);
                svfloat32_t a3 = svld1_f32(mask, A3 + k);
                svfloat32_t a4 = svld1_f32(mask, A4 + k);
                svfloat32_t a5 = svld1_f32(mask, A5 + k);
                svfloat32_t b0 = svld1_f32(mask, B0 + k);
                c00 = svmla_f32_m(mask, c00, a0, b0);
                c10 = svmla_f32_m(mask, c10, a1, b0);
                c20 = svmla_f32_m(mask, c20, a2, b0);
                c30 = svmla_f32_m(mask, c30, a3, b0);
                c40 = svmla_f32_m(mask, c40, a4, b0);
                c50 = svmla_f32_m(mask, c50, a5, b0);
            }
            C[0 * ldc] += alpha * ExtractSum32f(c00);
            C[1 * ldc] += alpha * ExtractSum32f(c10);
            C[2 * ldc] += alpha * ExtractSum32f(c20);
            C[3 * ldc] += alpha * ExtractSum32f(c30);
            C[4 * ldc] += alpha * ExtractSum32f(c40);
            C[5 * ldc] += alpha * ExtractSum32f(c50);
        }

        static void Kernel6x4nt(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            size_t F = svcntw(), DF = 2 * F, k = 0;
            const float* A0 = A + 0 * lda;
            const float* A1 = A + 1 * lda;
            const float* A2 = A + 2 * lda;
            const float* A3 = A + 3 * lda;
            const float* A4 = A + 4 * lda;
            const float* A5 = A + 5 * lda;
            const float* B0 = B + 0 * ldb;
            const float* B1 = B + 1 * ldb;
            const float* B2 = B + 2 * ldb;
            const float* B3 = B + 3 * ldb;
            const svbool_t body = svptrue_b32();
            svfloat32_t c00 = svdup_n_f32(0.0f);
            svfloat32_t c01 = svdup_n_f32(0.0f);
            svfloat32_t c02 = svdup_n_f32(0.0f);
            svfloat32_t c03 = svdup_n_f32(0.0f);
            svfloat32_t c10 = svdup_n_f32(0.0f);
            svfloat32_t c11 = svdup_n_f32(0.0f);
            svfloat32_t c12 = svdup_n_f32(0.0f);
            svfloat32_t c13 = svdup_n_f32(0.0f);
            svfloat32_t c20 = svdup_n_f32(0.0f);
            svfloat32_t c21 = svdup_n_f32(0.0f);
            svfloat32_t c22 = svdup_n_f32(0.0f);
            svfloat32_t c23 = svdup_n_f32(0.0f);
            svfloat32_t c30 = svdup_n_f32(0.0f);
            svfloat32_t c31 = svdup_n_f32(0.0f);
            svfloat32_t c32 = svdup_n_f32(0.0f);
            svfloat32_t c33 = svdup_n_f32(0.0f);
            svfloat32_t c40 = svdup_n_f32(0.0f);
            svfloat32_t c41 = svdup_n_f32(0.0f);
            svfloat32_t c42 = svdup_n_f32(0.0f);
            svfloat32_t c43 = svdup_n_f32(0.0f);
            svfloat32_t c50 = svdup_n_f32(0.0f);
            svfloat32_t c51 = svdup_n_f32(0.0f);
            svfloat32_t c52 = svdup_n_f32(0.0f);
            svfloat32_t c53 = svdup_n_f32(0.0f);
            for (; k + DF <= K; k += DF)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t a1 = svld1_f32(body, A1 + k);
                svfloat32_t a2 = svld1_f32(body, A2 + k);
                svfloat32_t a3 = svld1_f32(body, A3 + k);
                svfloat32_t a4 = svld1_f32(body, A4 + k);
                svfloat32_t a5 = svld1_f32(body, A5 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
                c20 = svmla_f32_x(body, c20, a2, b0);
                c30 = svmla_f32_x(body, c30, a3, b0);
                c40 = svmla_f32_x(body, c40, a4, b0);
                c50 = svmla_f32_x(body, c50, a5, b0);
                b0 = svld1_f32(body, B1 + k);
                c01 = svmla_f32_x(body, c01, a0, b0);
                c11 = svmla_f32_x(body, c11, a1, b0);
                c21 = svmla_f32_x(body, c21, a2, b0);
                c31 = svmla_f32_x(body, c31, a3, b0);
                c41 = svmla_f32_x(body, c41, a4, b0);
                c51 = svmla_f32_x(body, c51, a5, b0);
                b0 = svld1_f32(body, B2 + k);
                c02 = svmla_f32_x(body, c02, a0, b0);
                c12 = svmla_f32_x(body, c12, a1, b0);
                c22 = svmla_f32_x(body, c22, a2, b0);
                c32 = svmla_f32_x(body, c32, a3, b0);
                c42 = svmla_f32_x(body, c42, a4, b0);
                c52 = svmla_f32_x(body, c52, a5, b0);
                b0 = svld1_f32(body, B3 + k);
                c03 = svmla_f32_x(body, c03, a0, b0);
                c13 = svmla_f32_x(body, c13, a1, b0);
                c23 = svmla_f32_x(body, c23, a2, b0);
                c33 = svmla_f32_x(body, c33, a3, b0);
                c43 = svmla_f32_x(body, c43, a4, b0);
                c53 = svmla_f32_x(body, c53, a5, b0);
                a0 = svld1_f32(body, A0 + k + F);
                a1 = svld1_f32(body, A1 + k + F);
                a2 = svld1_f32(body, A2 + k + F);
                a3 = svld1_f32(body, A3 + k + F);
                a4 = svld1_f32(body, A4 + k + F);
                a5 = svld1_f32(body, A5 + k + F);
                b0 = svld1_f32(body, B0 + k + F);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
                c20 = svmla_f32_x(body, c20, a2, b0);
                c30 = svmla_f32_x(body, c30, a3, b0);
                c40 = svmla_f32_x(body, c40, a4, b0);
                c50 = svmla_f32_x(body, c50, a5, b0);
                b0 = svld1_f32(body, B1 + k + F);
                c01 = svmla_f32_x(body, c01, a0, b0);
                c11 = svmla_f32_x(body, c11, a1, b0);
                c21 = svmla_f32_x(body, c21, a2, b0);
                c31 = svmla_f32_x(body, c31, a3, b0);
                c41 = svmla_f32_x(body, c41, a4, b0);
                c51 = svmla_f32_x(body, c51, a5, b0);
                b0 = svld1_f32(body, B2 + k + F);
                c02 = svmla_f32_x(body, c02, a0, b0);
                c12 = svmla_f32_x(body, c12, a1, b0);
                c22 = svmla_f32_x(body, c22, a2, b0);
                c32 = svmla_f32_x(body, c32, a3, b0);
                c42 = svmla_f32_x(body, c42, a4, b0);
                c52 = svmla_f32_x(body, c52, a5, b0);
                b0 = svld1_f32(body, B3 + k + F);
                c03 = svmla_f32_x(body, c03, a0, b0);
                c13 = svmla_f32_x(body, c13, a1, b0);
                c23 = svmla_f32_x(body, c23, a2, b0);
                c33 = svmla_f32_x(body, c33, a3, b0);
                c43 = svmla_f32_x(body, c43, a4, b0);
                c53 = svmla_f32_x(body, c53, a5, b0);
            }
            for (; k + F <= K; k += F)
            {
                svfloat32_t a0 = svld1_f32(body, A0 + k);
                svfloat32_t a1 = svld1_f32(body, A1 + k);
                svfloat32_t a2 = svld1_f32(body, A2 + k);
                svfloat32_t a3 = svld1_f32(body, A3 + k);
                svfloat32_t a4 = svld1_f32(body, A4 + k);
                svfloat32_t a5 = svld1_f32(body, A5 + k);
                svfloat32_t b0 = svld1_f32(body, B0 + k);
                c00 = svmla_f32_x(body, c00, a0, b0);
                c10 = svmla_f32_x(body, c10, a1, b0);
                c20 = svmla_f32_x(body, c20, a2, b0);
                c30 = svmla_f32_x(body, c30, a3, b0);
                c40 = svmla_f32_x(body, c40, a4, b0);
                c50 = svmla_f32_x(body, c50, a5, b0);
                b0 = svld1_f32(body, B1 + k);
                c01 = svmla_f32_x(body, c01, a0, b0);
                c11 = svmla_f32_x(body, c11, a1, b0);
                c21 = svmla_f32_x(body, c21, a2, b0);
                c31 = svmla_f32_x(body, c31, a3, b0);
                c41 = svmla_f32_x(body, c41, a4, b0);
                c51 = svmla_f32_x(body, c51, a5, b0);
                b0 = svld1_f32(body, B2 + k);
                c02 = svmla_f32_x(body, c02, a0, b0);
                c12 = svmla_f32_x(body, c12, a1, b0);
                c22 = svmla_f32_x(body, c22, a2, b0);
                c32 = svmla_f32_x(body, c32, a3, b0);
                c42 = svmla_f32_x(body, c42, a4, b0);
                c52 = svmla_f32_x(body, c52, a5, b0);
                b0 = svld1_f32(body, B3 + k);
                c03 = svmla_f32_x(body, c03, a0, b0);
                c13 = svmla_f32_x(body, c13, a1, b0);
                c23 = svmla_f32_x(body, c23, a2, b0);
                c33 = svmla_f32_x(body, c33, a3, b0);
                c43 = svmla_f32_x(body, c43, a4, b0);
                c53 = svmla_f32_x(body, c53, a5, b0);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                svfloat32_t a0 = svld1_f32(mask, A0 + k);
                svfloat32_t a1 = svld1_f32(mask, A1 + k);
                svfloat32_t a2 = svld1_f32(mask, A2 + k);
                svfloat32_t a3 = svld1_f32(mask, A3 + k);
                svfloat32_t a4 = svld1_f32(mask, A4 + k);
                svfloat32_t a5 = svld1_f32(mask, A5 + k);
                svfloat32_t b0 = svld1_f32(mask, B0 + k);
                c00 = svmla_f32_m(mask, c00, a0, b0);
                c10 = svmla_f32_m(mask, c10, a1, b0);
                c20 = svmla_f32_m(mask, c20, a2, b0);
                c30 = svmla_f32_m(mask, c30, a3, b0);
                c40 = svmla_f32_m(mask, c40, a4, b0);
                c50 = svmla_f32_m(mask, c50, a5, b0);
                b0 = svld1_f32(mask, B1 + k);
                c01 = svmla_f32_m(mask, c01, a0, b0);
                c11 = svmla_f32_m(mask, c11, a1, b0);
                c21 = svmla_f32_m(mask, c21, a2, b0);
                c31 = svmla_f32_m(mask, c31, a3, b0);
                c41 = svmla_f32_m(mask, c41, a4, b0);
                c51 = svmla_f32_m(mask, c51, a5, b0);
                b0 = svld1_f32(mask, B2 + k);
                c02 = svmla_f32_m(mask, c02, a0, b0);
                c12 = svmla_f32_m(mask, c12, a1, b0);
                c22 = svmla_f32_m(mask, c22, a2, b0);
                c32 = svmla_f32_m(mask, c32, a3, b0);
                c42 = svmla_f32_m(mask, c42, a4, b0);
                c52 = svmla_f32_m(mask, c52, a5, b0);
                b0 = svld1_f32(mask, B3 + k);
                c03 = svmla_f32_m(mask, c03, a0, b0);
                c13 = svmla_f32_m(mask, c13, a1, b0);
                c23 = svmla_f32_m(mask, c23, a2, b0);
                c33 = svmla_f32_m(mask, c33, a3, b0);
                c43 = svmla_f32_m(mask, c43, a4, b0);
                c53 = svmla_f32_m(mask, c53, a5, b0);
            }
            Add4ExtractedSums(c00, c01, c02, c03, alpha, C + 0 * ldc);
            Add4ExtractedSums(c10, c11, c12, c13, alpha, C + 1 * ldc);
            Add4ExtractedSums(c20, c21, c22, c23, alpha, C + 2 * ldc);
            Add4ExtractedSums(c30, c31, c32, c33, alpha, C + 3 * ldc);
            Add4ExtractedSums(c40, c41, c42, c43, alpha, C + 4 * ldc);
            Add4ExtractedSums(c50, c51, c52, c53, alpha, C + 5 * ldc);
        }

        void Gemm32fNT(size_t M, size_t N, size_t K, const float* alpha, const float* A, size_t lda, const float* B, size_t ldb, const float* beta, float* C, size_t ldc)
        {
            typedef Simd::GemmNT<float, 4> GemmNT;
            GemmNT gemmNT(M, N, K, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), GemmScaleC,
                Kernel1x1nt, Kernel1x4nt, Kernel2x1nt, Kernel2x4nt, Kernel3x1nt, Kernel3x4nt, Kernel6x1nt, Kernel6x4nt);
            gemmNT.Run(alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
#endif
}
