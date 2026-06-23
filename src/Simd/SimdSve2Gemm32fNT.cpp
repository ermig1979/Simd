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

#define SIMD_SVE2_GEMM_INIT1(row) \
        svfloat32_t c##row##0 = zero;

#define SIMD_SVE2_GEMM_INIT4(row) \
        svfloat32_t c##row##0 = zero, c##row##1 = zero, c##row##2 = zero, c##row##3 = zero;

#define SIMD_SVE2_GEMM_ROW1(row, offset) \
        { \
            svfloat32_t a##row = svld1_f32(mask, A##row + (offset)); \
            c##row##0 = svmla_f32_m(mask, c##row##0, a##row, b0); \
        }

#define SIMD_SVE2_GEMM_ROW4(row, offset) \
        { \
            svfloat32_t a##row = svld1_f32(mask, A##row + (offset)); \
            c##row##0 = svmla_f32_m(mask, c##row##0, a##row, b0); \
            c##row##1 = svmla_f32_m(mask, c##row##1, a##row, b1); \
            c##row##2 = svmla_f32_m(mask, c##row##2, a##row, b2); \
            c##row##3 = svmla_f32_m(mask, c##row##3, a##row, b3); \
        }

#define SIMD_SVE2_GEMM_STEP1_1(offset) \
        { \
            svfloat32_t b0 = svld1_f32(mask, B0 + (offset)); \
            SIMD_SVE2_GEMM_ROW1(0, offset); \
        }

#define SIMD_SVE2_GEMM_STEP1_2(offset) \
        { \
            svfloat32_t b0 = svld1_f32(mask, B0 + (offset)); \
            SIMD_SVE2_GEMM_ROW1(0, offset); \
            SIMD_SVE2_GEMM_ROW1(1, offset); \
        }

#define SIMD_SVE2_GEMM_STEP1_3(offset) \
        { \
            svfloat32_t b0 = svld1_f32(mask, B0 + (offset)); \
            SIMD_SVE2_GEMM_ROW1(0, offset); \
            SIMD_SVE2_GEMM_ROW1(1, offset); \
            SIMD_SVE2_GEMM_ROW1(2, offset); \
        }

#define SIMD_SVE2_GEMM_STEP1_6(offset) \
        { \
            svfloat32_t b0 = svld1_f32(mask, B0 + (offset)); \
            SIMD_SVE2_GEMM_ROW1(0, offset); \
            SIMD_SVE2_GEMM_ROW1(1, offset); \
            SIMD_SVE2_GEMM_ROW1(2, offset); \
            SIMD_SVE2_GEMM_ROW1(3, offset); \
            SIMD_SVE2_GEMM_ROW1(4, offset); \
            SIMD_SVE2_GEMM_ROW1(5, offset); \
        }

#define SIMD_SVE2_GEMM_STEP4_1(offset) \
        { \
            svfloat32_t b0 = svld1_f32(mask, B0 + (offset)); \
            svfloat32_t b1 = svld1_f32(mask, B1 + (offset)); \
            svfloat32_t b2 = svld1_f32(mask, B2 + (offset)); \
            svfloat32_t b3 = svld1_f32(mask, B3 + (offset)); \
            SIMD_SVE2_GEMM_ROW4(0, offset); \
        }

#define SIMD_SVE2_GEMM_STEP4_2(offset) \
        { \
            svfloat32_t b0 = svld1_f32(mask, B0 + (offset)); \
            svfloat32_t b1 = svld1_f32(mask, B1 + (offset)); \
            svfloat32_t b2 = svld1_f32(mask, B2 + (offset)); \
            svfloat32_t b3 = svld1_f32(mask, B3 + (offset)); \
            SIMD_SVE2_GEMM_ROW4(0, offset); \
            SIMD_SVE2_GEMM_ROW4(1, offset); \
        }

#define SIMD_SVE2_GEMM_STEP4_3(offset) \
        { \
            svfloat32_t b0 = svld1_f32(mask, B0 + (offset)); \
            svfloat32_t b1 = svld1_f32(mask, B1 + (offset)); \
            svfloat32_t b2 = svld1_f32(mask, B2 + (offset)); \
            svfloat32_t b3 = svld1_f32(mask, B3 + (offset)); \
            SIMD_SVE2_GEMM_ROW4(0, offset); \
            SIMD_SVE2_GEMM_ROW4(1, offset); \
            SIMD_SVE2_GEMM_ROW4(2, offset); \
        }

#define SIMD_SVE2_GEMM_STEP4_6(offset) \
        { \
            svfloat32_t b0 = svld1_f32(mask, B0 + (offset)); \
            svfloat32_t b1 = svld1_f32(mask, B1 + (offset)); \
            svfloat32_t b2 = svld1_f32(mask, B2 + (offset)); \
            svfloat32_t b3 = svld1_f32(mask, B3 + (offset)); \
            SIMD_SVE2_GEMM_ROW4(0, offset); \
            SIMD_SVE2_GEMM_ROW4(1, offset); \
            SIMD_SVE2_GEMM_ROW4(2, offset); \
            SIMD_SVE2_GEMM_ROW4(3, offset); \
            SIMD_SVE2_GEMM_ROW4(4, offset); \
            SIMD_SVE2_GEMM_ROW4(5, offset); \
        }

#define SIMD_SVE2_GEMM_SAVE1(row) \
        C[(row) * ldc] += alpha * ExtractSum32f(c##row##0);

#define SIMD_SVE2_GEMM_SAVE4(row) \
        C[(row) * ldc + 0] += alpha * ExtractSum32f(c##row##0); \
        C[(row) * ldc + 1] += alpha * ExtractSum32f(c##row##1); \
        C[(row) * ldc + 2] += alpha * ExtractSum32f(c##row##2); \
        C[(row) * ldc + 3] += alpha * ExtractSum32f(c##row##3);

        static void Kernel1x1nt(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            size_t F = svcntw(), DF = 2 * F, k = 0;
            const float* A0 = A + 0 * lda;
            const float* B0 = B + 0 * ldb;
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            SIMD_SVE2_GEMM_INIT1(0);
            for (; k + DF <= K; k += DF)
            {
                const svbool_t mask = body;
                SIMD_SVE2_GEMM_STEP1_1(k);
                SIMD_SVE2_GEMM_STEP1_1(k + F);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                SIMD_SVE2_GEMM_STEP1_1(k);
            }
            SIMD_SVE2_GEMM_SAVE1(0);
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
            const svfloat32_t zero = svdup_n_f32(0.0f);
            SIMD_SVE2_GEMM_INIT4(0);
            for (; k + DF <= K; k += DF)
            {
                const svbool_t mask = body;
                SIMD_SVE2_GEMM_STEP4_1(k);
                SIMD_SVE2_GEMM_STEP4_1(k + F);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                SIMD_SVE2_GEMM_STEP4_1(k);
            }
            SIMD_SVE2_GEMM_SAVE4(0);
        }

        static void Kernel2x1nt(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            size_t F = svcntw(), DF = 2 * F, k = 0;
            const float* A0 = A + 0 * lda;
            const float* A1 = A + 1 * lda;
            const float* B0 = B + 0 * ldb;
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            SIMD_SVE2_GEMM_INIT1(0);
            SIMD_SVE2_GEMM_INIT1(1);
            for (; k + DF <= K; k += DF)
            {
                const svbool_t mask = body;
                SIMD_SVE2_GEMM_STEP1_2(k);
                SIMD_SVE2_GEMM_STEP1_2(k + F);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                SIMD_SVE2_GEMM_STEP1_2(k);
            }
            SIMD_SVE2_GEMM_SAVE1(0);
            SIMD_SVE2_GEMM_SAVE1(1);
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
            const svfloat32_t zero = svdup_n_f32(0.0f);
            SIMD_SVE2_GEMM_INIT4(0);
            SIMD_SVE2_GEMM_INIT4(1);
            for (; k + DF <= K; k += DF)
            {
                const svbool_t mask = body;
                SIMD_SVE2_GEMM_STEP4_2(k);
                SIMD_SVE2_GEMM_STEP4_2(k + F);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                SIMD_SVE2_GEMM_STEP4_2(k);
            }
            SIMD_SVE2_GEMM_SAVE4(0);
            SIMD_SVE2_GEMM_SAVE4(1);
        }

        static void Kernel3x1nt(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            size_t F = svcntw(), DF = 2 * F, k = 0;
            const float* A0 = A + 0 * lda;
            const float* A1 = A + 1 * lda;
            const float* A2 = A + 2 * lda;
            const float* B0 = B + 0 * ldb;
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            SIMD_SVE2_GEMM_INIT1(0);
            SIMD_SVE2_GEMM_INIT1(1);
            SIMD_SVE2_GEMM_INIT1(2);
            for (; k + DF <= K; k += DF)
            {
                const svbool_t mask = body;
                SIMD_SVE2_GEMM_STEP1_3(k);
                SIMD_SVE2_GEMM_STEP1_3(k + F);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                SIMD_SVE2_GEMM_STEP1_3(k);
            }
            SIMD_SVE2_GEMM_SAVE1(0);
            SIMD_SVE2_GEMM_SAVE1(1);
            SIMD_SVE2_GEMM_SAVE1(2);
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
            const svfloat32_t zero = svdup_n_f32(0.0f);
            SIMD_SVE2_GEMM_INIT4(0);
            SIMD_SVE2_GEMM_INIT4(1);
            SIMD_SVE2_GEMM_INIT4(2);
            for (; k + DF <= K; k += DF)
            {
                const svbool_t mask = body;
                SIMD_SVE2_GEMM_STEP4_3(k);
                SIMD_SVE2_GEMM_STEP4_3(k + F);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                SIMD_SVE2_GEMM_STEP4_3(k);
            }
            SIMD_SVE2_GEMM_SAVE4(0);
            SIMD_SVE2_GEMM_SAVE4(1);
            SIMD_SVE2_GEMM_SAVE4(2);
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
            const svfloat32_t zero = svdup_n_f32(0.0f);
            SIMD_SVE2_GEMM_INIT1(0);
            SIMD_SVE2_GEMM_INIT1(1);
            SIMD_SVE2_GEMM_INIT1(2);
            SIMD_SVE2_GEMM_INIT1(3);
            SIMD_SVE2_GEMM_INIT1(4);
            SIMD_SVE2_GEMM_INIT1(5);
            for (; k + DF <= K; k += DF)
            {
                const svbool_t mask = body;
                SIMD_SVE2_GEMM_STEP1_6(k);
                SIMD_SVE2_GEMM_STEP1_6(k + F);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                SIMD_SVE2_GEMM_STEP1_6(k);
            }
            SIMD_SVE2_GEMM_SAVE1(0);
            SIMD_SVE2_GEMM_SAVE1(1);
            SIMD_SVE2_GEMM_SAVE1(2);
            SIMD_SVE2_GEMM_SAVE1(3);
            SIMD_SVE2_GEMM_SAVE1(4);
            SIMD_SVE2_GEMM_SAVE1(5);
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
            const svfloat32_t zero = svdup_n_f32(0.0f);
            SIMD_SVE2_GEMM_INIT4(0);
            SIMD_SVE2_GEMM_INIT4(1);
            SIMD_SVE2_GEMM_INIT4(2);
            SIMD_SVE2_GEMM_INIT4(3);
            SIMD_SVE2_GEMM_INIT4(4);
            SIMD_SVE2_GEMM_INIT4(5);
            for (; k + DF <= K; k += DF)
            {
                const svbool_t mask = body;
                SIMD_SVE2_GEMM_STEP4_6(k);
                SIMD_SVE2_GEMM_STEP4_6(k + F);
            }
            if (k < K)
            {
                const svbool_t mask = svwhilelt_b32(k, K);
                SIMD_SVE2_GEMM_STEP4_6(k);
            }
            SIMD_SVE2_GEMM_SAVE4(0);
            SIMD_SVE2_GEMM_SAVE4(1);
            SIMD_SVE2_GEMM_SAVE4(2);
            SIMD_SVE2_GEMM_SAVE4(3);
            SIMD_SVE2_GEMM_SAVE4(4);
            SIMD_SVE2_GEMM_SAVE4(5);
        }

        void Gemm32fNT(size_t M, size_t N, size_t K, const float* alpha, const float* A, size_t lda, const float* B, size_t ldb, const float* beta, float* C, size_t ldc)
        {
            typedef Simd::GemmNT<float, 4> GemmNT;
            GemmNT gemmNT(M, N, K, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), GemmScaleC,
                Kernel1x1nt, Kernel1x4nt, Kernel2x1nt, Kernel2x4nt, Kernel3x1nt, Kernel3x4nt, Kernel6x1nt, Kernel6x4nt);
            gemmNT.Run(alpha, A, lda, B, ldb, beta, C, ldc);
        }

#undef SIMD_SVE2_GEMM_INIT1
#undef SIMD_SVE2_GEMM_INIT4
#undef SIMD_SVE2_GEMM_ROW1
#undef SIMD_SVE2_GEMM_ROW4
#undef SIMD_SVE2_GEMM_STEP1_1
#undef SIMD_SVE2_GEMM_STEP1_2
#undef SIMD_SVE2_GEMM_STEP1_3
#undef SIMD_SVE2_GEMM_STEP1_6
#undef SIMD_SVE2_GEMM_STEP4_1
#undef SIMD_SVE2_GEMM_STEP4_2
#undef SIMD_SVE2_GEMM_STEP4_3
#undef SIMD_SVE2_GEMM_STEP4_6
#undef SIMD_SVE2_GEMM_SAVE1
#undef SIMD_SVE2_GEMM_SAVE4
    }
#endif
}
