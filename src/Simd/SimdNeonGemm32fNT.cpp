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
#ifdef SIMD_NEON_ENABLE 
    namespace Neon
    {
        SIMD_INLINE float32x4_t Tail(size_t tail)
        {
            const int32_t mask[DF] = { 0, 0, 0, 0, -1, -1, -1, -1 };
            return Load<false>((float*)(mask + tail));
        }

        SIMD_INLINE void Add4ExtractedSums(const float32x4_t & sum0, const float32x4_t & sum1, const float32x4_t & sum2, const float32x4_t & sum3, const float32x4_t & alpha, float * dst)
        {
            float32x4x2_t a02 = vzipq_f32(sum0, sum2);
            float32x4x2_t a13 = vzipq_f32(sum1, sum3);
            float32x4x2_t b0 = vzipq_f32(a02.val[0], a13.val[0]);
            float32x4x2_t b1 = vzipq_f32(a02.val[1], a13.val[1]);
            Store<false>(dst, vmlaq_f32(Load<false>(dst), alpha, vaddq_f32(vaddq_f32(b0.val[0], b0.val[1]), vaddq_f32(b1.val[0], b1.val[1]))));
        }

        static void Kernel1x1x4nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K4 = K & (~3);
            const float * A0 = A + 0 * lda;
            const float * B0 = B + 0 * ldb;
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t a0, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = Load<false>(A0 + k);
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                float32x4_t tail = Tail(K - K4);
                a0 = And(tail, Load<false>(A0 + k));
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
            }
            C[0] += alpha * ExtractSum32f(c00);
        }

        static void Kernel1x4x4nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K4 = K & (~3);
            const float * A0 = A + 0 * lda;
            const float * B0 = B + 0 * ldb;
            const float * B1 = B + 1 * ldb;
            const float * B2 = B + 2 * ldb;
            const float * B3 = B + 3 * ldb;
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c02 = vdupq_n_f32(0);
            float32x4_t c03 = vdupq_n_f32(0);
            float32x4_t a0, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = Load<false>(A0 + k);
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
                b0 = Load<false>(B1 + k);
                c01 = vmlaq_f32(c01, a0, b0);
                b0 = Load<false>(B2 + k);
                c02 = vmlaq_f32(c02, a0, b0);
                b0 = Load<false>(B3 + k);
                c03 = vmlaq_f32(c03, a0, b0);
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                float32x4_t tail = Tail(K - K4);
                a0 = And(tail, Load<false>(A0 + k));
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
                b0 = Load<false>(B1 + k);
                c01 = vmlaq_f32(c01, a0, b0);
                b0 = Load<false>(B2 + k);
                c02 = vmlaq_f32(c02, a0, b0);
                b0 = Load<false>(B3 + k);
                c03 = vmlaq_f32(c03, a0, b0);
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            Add4ExtractedSums(c00, c01, c02, c03, _alpha, C + 0 * ldc);
        }

        static void Kernel2x1x4nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K4 = K & (~3);
            const float * A0 = A + 0 * lda;
            const float * A1 = A + 1 * lda;
            const float * B0 = B + 0 * ldb;
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t a0, a1, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = Load<false>(A0 + k);
                a1 = Load<false>(A1 + k);
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                float32x4_t tail = Tail(K - K4);
                a0 = And(tail, Load<false>(A0 + k));
                a1 = And(tail, Load<false>(A1 + k));
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
            }
            C[0 * ldc] += alpha * ExtractSum32f(c00);
            C[1 * ldc] += alpha * ExtractSum32f(c10);
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
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c02 = vdupq_n_f32(0);
            float32x4_t c03 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c12 = vdupq_n_f32(0);
            float32x4_t c13 = vdupq_n_f32(0);
            float32x4_t a0, a1, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = Load<false>(A0 + k);
                a1 = Load<false>(A1 + k);
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
                b0 = Load<false>(B1 + k);
                c01 = vmlaq_f32(c01, a0, b0);
                c11 = vmlaq_f32(c11, a1, b0);
                b0 = Load<false>(B2 + k);
                c02 = vmlaq_f32(c02, a0, b0);
                c12 = vmlaq_f32(c12, a1, b0);
                b0 = Load<false>(B3 + k);
                c03 = vmlaq_f32(c03, a0, b0);
                c13 = vmlaq_f32(c13, a1, b0);
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                float32x4_t tail = Tail(K - K4);
                a0 = And(tail, Load<false>(A0 + k));
                a1 = And(tail, Load<false>(A1 + k));
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
                b0 = Load<false>(B1 + k);
                c01 = vmlaq_f32(c01, a0, b0);
                c11 = vmlaq_f32(c11, a1, b0);
                b0 = Load<false>(B2 + k);
                c02 = vmlaq_f32(c02, a0, b0);
                c12 = vmlaq_f32(c12, a1, b0);
                b0 = Load<false>(B3 + k);
                c03 = vmlaq_f32(c03, a0, b0);
                c13 = vmlaq_f32(c13, a1, b0);
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
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
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t a0, a1, a2, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = Load<false>(A0 + k);
                a1 = Load<false>(A1 + k);
                a2 = Load<false>(A2 + k);
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
                c20 = vmlaq_f32(c20, a2, b0);
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                float32x4_t tail = Tail(K - K4);
                a0 = And(tail, Load<false>(A0 + k));
                a1 = And(tail, Load<false>(A1 + k));
                a2 = And(tail, Load<false>(A2 + k));
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
                c20 = vmlaq_f32(c20, a2, b0);
            }
            C[0 * ldc] += alpha * ExtractSum32f(c00);
            C[1 * ldc] += alpha * ExtractSum32f(c10);
            C[2 * ldc] += alpha * ExtractSum32f(c20);
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
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c02 = vdupq_n_f32(0);
            float32x4_t c03 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c12 = vdupq_n_f32(0);
            float32x4_t c13 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c21 = vdupq_n_f32(0);
            float32x4_t c22 = vdupq_n_f32(0);
            float32x4_t c23 = vdupq_n_f32(0);
            float32x4_t a0, a1, a2, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = Load<false>(A0 + k);
                a1 = Load<false>(A1 + k);
                a2 = Load<false>(A2 + k);
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
                c20 = vmlaq_f32(c20, a2, b0);
                b0 = Load<false>(B1 + k);
                c01 = vmlaq_f32(c01, a0, b0);
                c11 = vmlaq_f32(c11, a1, b0);
                c21 = vmlaq_f32(c21, a2, b0);
                b0 = Load<false>(B2 + k);
                c02 = vmlaq_f32(c02, a0, b0);
                c12 = vmlaq_f32(c12, a1, b0);
                c22 = vmlaq_f32(c22, a2, b0);
                b0 = Load<false>(B3 + k);
                c03 = vmlaq_f32(c03, a0, b0);
                c13 = vmlaq_f32(c13, a1, b0);
                c23 = vmlaq_f32(c23, a2, b0);
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                float32x4_t tail = Tail(K - K4);
                a0 = And(tail, Load<false>(A0 + k));
                a1 = And(tail, Load<false>(A1 + k));
                a2 = And(tail, Load<false>(A2 + k));
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
                c20 = vmlaq_f32(c20, a2, b0);
                b0 = Load<false>(B1 + k);
                c01 = vmlaq_f32(c01, a0, b0);
                c11 = vmlaq_f32(c11, a1, b0);
                c21 = vmlaq_f32(c21, a2, b0);
                b0 = Load<false>(B2 + k);
                c02 = vmlaq_f32(c02, a0, b0);
                c12 = vmlaq_f32(c12, a1, b0);
                c22 = vmlaq_f32(c22, a2, b0);
                b0 = Load<false>(B3 + k);
                c03 = vmlaq_f32(c03, a0, b0);
                c13 = vmlaq_f32(c13, a1, b0);
                c23 = vmlaq_f32(c23, a2, b0);
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            Add4ExtractedSums(c00, c01, c02, c03, _alpha, C + 0 * ldc);
            Add4ExtractedSums(c10, c11, c12, c13, _alpha, C + 1 * ldc);
            Add4ExtractedSums(c20, c21, c22, c23, _alpha, C + 2 * ldc);
        }

        static void Kernel6x1x4nt(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            size_t K4 = K & (~3);
            const float* A0 = A + 0 * lda;
            const float* A1 = A + 1 * lda;
            const float* A2 = A + 2 * lda;
            const float* A3 = A + 3 * lda;
            const float* A4 = A + 4 * lda;
            const float* A5 = A + 5 * lda;
            const float* B0 = B + 0 * ldb;
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c40 = vdupq_n_f32(0);
            float32x4_t c50 = vdupq_n_f32(0);
            float32x4_t a0, a1, a2, a3, a4, a5, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = Load<false>(A0 + k);
                a1 = Load<false>(A1 + k);
                a2 = Load<false>(A2 + k);
                a3 = Load<false>(A3 + k);
                a4 = Load<false>(A4 + k);
                a5 = Load<false>(A5 + k);
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
                c20 = vmlaq_f32(c20, a2, b0);
                c30 = vmlaq_f32(c30, a3, b0);
                c40 = vmlaq_f32(c40, a4, b0);
                c50 = vmlaq_f32(c50, a5, b0);
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                float32x4_t tail = Tail(K - K4);
                a0 = And(tail, Load<false>(A0 + k));
                a1 = And(tail, Load<false>(A1 + k));
                a2 = And(tail, Load<false>(A2 + k));
                a3 = And(tail, Load<false>(A3 + k));
                a4 = And(tail, Load<false>(A4 + k));
                a5 = And(tail, Load<false>(A5 + k));
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
                c20 = vmlaq_f32(c20, a2, b0);
                c30 = vmlaq_f32(c30, a3, b0);
                c40 = vmlaq_f32(c40, a4, b0);
                c50 = vmlaq_f32(c50, a5, b0);
            }
            C[0 * ldc] += alpha * ExtractSum32f(c00);
            C[1 * ldc] += alpha * ExtractSum32f(c10);
            C[2 * ldc] += alpha * ExtractSum32f(c20);
            C[3 * ldc] += alpha * ExtractSum32f(c30);
            C[4 * ldc] += alpha * ExtractSum32f(c40);
            C[5 * ldc] += alpha * ExtractSum32f(c50);
        }

        static void Kernel6x4x4nt(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            size_t K4 = K & (~3);
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
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c02 = vdupq_n_f32(0);
            float32x4_t c03 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c12 = vdupq_n_f32(0);
            float32x4_t c13 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c21 = vdupq_n_f32(0);
            float32x4_t c22 = vdupq_n_f32(0);
            float32x4_t c23 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c31 = vdupq_n_f32(0);
            float32x4_t c32 = vdupq_n_f32(0);
            float32x4_t c33 = vdupq_n_f32(0);
            float32x4_t c40 = vdupq_n_f32(0);
            float32x4_t c41 = vdupq_n_f32(0);
            float32x4_t c42 = vdupq_n_f32(0);
            float32x4_t c43 = vdupq_n_f32(0);
            float32x4_t c50 = vdupq_n_f32(0);
            float32x4_t c51 = vdupq_n_f32(0);
            float32x4_t c52 = vdupq_n_f32(0);
            float32x4_t c53 = vdupq_n_f32(0);
            float32x4_t a0, a1, a2, a3, a4, a5, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = Load<false>(A0 + k);
                a1 = Load<false>(A1 + k);
                a2 = Load<false>(A2 + k);
                a3 = Load<false>(A3 + k);
                a4 = Load<false>(A4 + k);
                a5 = Load<false>(A5 + k);
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
                c20 = vmlaq_f32(c20, a2, b0);
                c30 = vmlaq_f32(c30, a3, b0);
                c40 = vmlaq_f32(c40, a4, b0);
                c50 = vmlaq_f32(c50, a5, b0);
                b0 = Load<false>(B1 + k);
                c01 = vmlaq_f32(c01, a0, b0);
                c11 = vmlaq_f32(c11, a1, b0);
                c21 = vmlaq_f32(c21, a2, b0);
                c31 = vmlaq_f32(c31, a3, b0);
                c41 = vmlaq_f32(c41, a4, b0);
                c51 = vmlaq_f32(c51, a5, b0);
                b0 = Load<false>(B2 + k);
                c02 = vmlaq_f32(c02, a0, b0);
                c12 = vmlaq_f32(c12, a1, b0);
                c22 = vmlaq_f32(c22, a2, b0);
                c32 = vmlaq_f32(c32, a3, b0);
                c42 = vmlaq_f32(c42, a4, b0);
                c52 = vmlaq_f32(c52, a5, b0);
                b0 = Load<false>(B3 + k);
                c03 = vmlaq_f32(c03, a0, b0);
                c13 = vmlaq_f32(c13, a1, b0);
                c23 = vmlaq_f32(c23, a2, b0);
                c33 = vmlaq_f32(c33, a3, b0);
                c43 = vmlaq_f32(c43, a4, b0);
                c53 = vmlaq_f32(c53, a5, b0);
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                float32x4_t tail = Tail(K - K4);
                a0 = And(tail, Load<false>(A0 + k));
                a1 = And(tail, Load<false>(A1 + k));
                a2 = And(tail, Load<false>(A2 + k));
                a3 = And(tail, Load<false>(A3 + k));
                a4 = And(tail, Load<false>(A4 + k));
                a5 = And(tail, Load<false>(A5 + k));
                b0 = Load<false>(B0 + k);
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
                c20 = vmlaq_f32(c20, a2, b0);
                c30 = vmlaq_f32(c30, a3, b0);
                c40 = vmlaq_f32(c40, a4, b0);
                c50 = vmlaq_f32(c50, a5, b0);
                b0 = Load<false>(B1 + k);
                c01 = vmlaq_f32(c01, a0, b0);
                c11 = vmlaq_f32(c11, a1, b0);
                c21 = vmlaq_f32(c21, a2, b0);
                c31 = vmlaq_f32(c31, a3, b0);
                c41 = vmlaq_f32(c41, a4, b0);
                c51 = vmlaq_f32(c51, a5, b0);
                b0 = Load<false>(B2 + k);
                c02 = vmlaq_f32(c02, a0, b0);
                c12 = vmlaq_f32(c12, a1, b0);
                c22 = vmlaq_f32(c22, a2, b0);
                c32 = vmlaq_f32(c32, a3, b0);
                c42 = vmlaq_f32(c42, a4, b0);
                c52 = vmlaq_f32(c52, a5, b0);
                b0 = Load<false>(B3 + k);
                c03 = vmlaq_f32(c03, a0, b0);
                c13 = vmlaq_f32(c13, a1, b0);
                c23 = vmlaq_f32(c23, a2, b0);
                c33 = vmlaq_f32(c33, a3, b0);
                c43 = vmlaq_f32(c43, a4, b0);
                c53 = vmlaq_f32(c53, a5, b0);
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            Add4ExtractedSums(c00, c01, c02, c03, _alpha, C + 0 * ldc);
            Add4ExtractedSums(c10, c11, c12, c13, _alpha, C + 1 * ldc);
            Add4ExtractedSums(c20, c21, c22, c23, _alpha, C + 2 * ldc);
            Add4ExtractedSums(c30, c31, c32, c33, _alpha, C + 3 * ldc);
            Add4ExtractedSums(c40, c41, c42, c43, _alpha, C + 4 * ldc);
            Add4ExtractedSums(c50, c51, c52, c53, _alpha, C + 5 * ldc);
        }

        void Gemm32fNT(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            typedef Simd::GemmNT<float, F> GemmNT;
#if defined(SIMD_ARM64_ENABLE)
            GemmNT gemmNT(M, N, K, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), GemmScaleC,
                Kernel1x1x4nt, Kernel1x4x4nt, Kernel2x1x4nt, Kernel2x4x4nt, Kernel3x1x4nt, Kernel3x4x4nt, Kernel6x1x4nt, Kernel6x4x4nt);
#else
            GemmNT gemmNT(M, N, K, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), GemmScaleC,
                Kernel1x1x4nt, Kernel1x4x4nt, Kernel2x1x4nt, Kernel2x4x4nt, Kernel3x1x4nt, Kernel3x4x4nt, NULL, NULL);
#endif
            gemmNT.Run(alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
#endif// SIMD_NEON_ENABLE
}
