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
#if defined(SIMD_ARM64_ENABLE)
//#define SIMD_ARM64_KERNEL_9X
//#define SIMD_ARM64_KERNEL_14X
#endif
    namespace Neon
    {
        SIMD_INLINE void AddProduct(float * ptr, float32x4_t value, float32x4_t alpha)
        {
            Store<false>(ptr, vmlaq_f32(Load<false>(ptr), value, alpha));
        }

        SIMD_INLINE void AddProduct(float * ptr, float32x4_t value, float32x4_t alpha, size_t tail)
        {
            if (tail == F)
                AddProduct(ptr, value, alpha);
            else
            {
                float tmp[F];
                Store<false>(tmp, vmlaq_f32(Load<false>(ptr), value, alpha));
                for (size_t i = 0; i < tail; ++i)
                    ptr[i] = tmp[i];
            }
        }

        void GemmKernel4x12nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c21 = vdupq_n_f32(0);
            float32x4_t c31 = vdupq_n_f32(0);
            float32x4_t c02 = vdupq_n_f32(0);
            float32x4_t c12 = vdupq_n_f32(0);
            float32x4_t c22 = vdupq_n_f32(0);
            float32x4_t c32 = vdupq_n_f32(0);
            float32x4_t b0, b1, b2, a0;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                b2 = Load<false>(B + ob2);
                a0 = vld1q_dup_f32(A + oa0);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                c02 = vmlaq_f32(c02, a0, b2);
                a0 = vld1q_dup_f32(A + oa1);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                c12 = vmlaq_f32(c12, a0, b2);
                a0 = vld1q_dup_f32(A + oa2);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                c22 = vmlaq_f32(c22, a0, b2);
                a0 = vld1q_dup_f32(A + oa3);
                c30 = vmlaq_f32(c30, a0, b0);
                c31 = vmlaq_f32(c31, a0, b1);
                c32 = vmlaq_f32(c32, a0, b2);
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01);
            AddProduct(C + 2 * F, _alpha, c02, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11);
            AddProduct(C + 2 * F, _alpha, c12, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21);
            AddProduct(C + 2 * F, _alpha, c22, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31);
            AddProduct(C + 2 * F, _alpha, c32, tail);
        }

        void GemmKernel4x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c21 = vdupq_n_f32(0);
            float32x4_t c31 = vdupq_n_f32(0);
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            float32x4_t b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                a0 = vld1q_dup_f32(A + oa0);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                a0 = vld1q_dup_f32(A + oa1);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                a0 = vld1q_dup_f32(A + oa2);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                a0 = vld1q_dup_f32(A + oa3);
                c30 = vmlaq_f32(c30, a0, b0);
                c31 = vmlaq_f32(c31, a0, b1);
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31, tail);
        }

        void GemmKernel4x4nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c0 = vdupq_n_f32(0);
            float32x4_t c1 = vdupq_n_f32(0);
            float32x4_t c2 = vdupq_n_f32(0);
            float32x4_t c3 = vdupq_n_f32(0);
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            const size_t ob0 = ldb * 0;
            float32x4_t b0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                c0 = vmlaq_f32(c0, b0, vld1q_dup_f32(A + oa0));
                c1 = vmlaq_f32(c1, b0, vld1q_dup_f32(A + oa1));
                c2 = vmlaq_f32(c2, b0, vld1q_dup_f32(A + oa2));
                c3 = vmlaq_f32(c3, b0, vld1q_dup_f32(A + oa3));
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * ldc, _alpha, c0, tail);
            AddProduct(C + 1 * ldc, _alpha, c1, tail);
            AddProduct(C + 2 * ldc, _alpha, c2, tail);
            AddProduct(C + 3 * ldc, _alpha, c3, tail);
        }

#if defined(SIMD_ARM64_ENABLE)
        void GemmKernel6x16nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c40 = vdupq_n_f32(0);
            float32x4_t c50 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c21 = vdupq_n_f32(0);
            float32x4_t c31 = vdupq_n_f32(0);
            float32x4_t c41 = vdupq_n_f32(0);
            float32x4_t c51 = vdupq_n_f32(0);
            float32x4_t c02 = vdupq_n_f32(0);
            float32x4_t c12 = vdupq_n_f32(0);
            float32x4_t c22 = vdupq_n_f32(0);
            float32x4_t c32 = vdupq_n_f32(0);
            float32x4_t c42 = vdupq_n_f32(0);
            float32x4_t c52 = vdupq_n_f32(0);
            float32x4_t c03 = vdupq_n_f32(0);
            float32x4_t c13 = vdupq_n_f32(0);
            float32x4_t c23 = vdupq_n_f32(0);
            float32x4_t c33 = vdupq_n_f32(0);
            float32x4_t c43 = vdupq_n_f32(0);
            float32x4_t c53 = vdupq_n_f32(0);
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            const size_t ob3 = ldb * 3;
            float32x4_t b0, b1, b2, b3, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                b2 = Load<false>(B + ob2);
                b3 = Load<false>(B + ob3);
                a0 = vld1q_dup_f32(A + oa0);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                c02 = vmlaq_f32(c02, a0, b2);
                c03 = vmlaq_f32(c03, a0, b3);
                a0 = vld1q_dup_f32(A + oa1);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                c12 = vmlaq_f32(c12, a0, b2);
                c13 = vmlaq_f32(c13, a0, b3);
                a0 = vld1q_dup_f32(A + oa2);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                c22 = vmlaq_f32(c22, a0, b2);
                c23 = vmlaq_f32(c23, a0, b3);
                a0 = vld1q_dup_f32(A + oa3);
                c30 = vmlaq_f32(c30, a0, b0);
                c31 = vmlaq_f32(c31, a0, b1);
                c32 = vmlaq_f32(c32, a0, b2);
                c33 = vmlaq_f32(c33, a0, b3);
                a0 = vld1q_dup_f32(A + oa4);
                c40 = vmlaq_f32(c40, a0, b0);
                c41 = vmlaq_f32(c41, a0, b1);
                c42 = vmlaq_f32(c42, a0, b2);
                c43 = vmlaq_f32(c43, a0, b3);
                a0 = vld1q_dup_f32(A + oa5);
                c50 = vmlaq_f32(c50, a0, b0);
                c51 = vmlaq_f32(c51, a0, b1);
                c52 = vmlaq_f32(c52, a0, b2);
                c53 = vmlaq_f32(c53, a0, b3);
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01);
            AddProduct(C + 2 * F, _alpha, c02);
            AddProduct(C + 3 * F, _alpha, c03, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11);
            AddProduct(C + 2 * F, _alpha, c12);
            AddProduct(C + 3 * F, _alpha, c13, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21);
            AddProduct(C + 2 * F, _alpha, c22);
            AddProduct(C + 3 * F, _alpha, c23, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31);
            AddProduct(C + 2 * F, _alpha, c32);
            AddProduct(C + 3 * F, _alpha, c33, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41);
            AddProduct(C + 2 * F, _alpha, c42);
            AddProduct(C + 3 * F, _alpha, c43, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51);
            AddProduct(C + 2 * F, _alpha, c52);
            AddProduct(C + 3 * F, _alpha, c53, tail);
        }

        void GemmKernel6x12nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c40 = vdupq_n_f32(0);
            float32x4_t c50 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c21 = vdupq_n_f32(0);
            float32x4_t c31 = vdupq_n_f32(0);
            float32x4_t c41 = vdupq_n_f32(0);
            float32x4_t c51 = vdupq_n_f32(0);
            float32x4_t c02 = vdupq_n_f32(0);
            float32x4_t c12 = vdupq_n_f32(0);
            float32x4_t c22 = vdupq_n_f32(0);
            float32x4_t c32 = vdupq_n_f32(0);
            float32x4_t c42 = vdupq_n_f32(0);
            float32x4_t c52 = vdupq_n_f32(0);
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            float32x4_t b0, b1, b2, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                b2 = Load<false>(B + ob2);
                a0 = vld1q_dup_f32(A + oa0);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                c02 = vmlaq_f32(c02, a0, b2);
                a0 = vld1q_dup_f32(A + oa1);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                c12 = vmlaq_f32(c12, a0, b2);
                a0 = vld1q_dup_f32(A + oa2);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                c22 = vmlaq_f32(c22, a0, b2);
                a0 = vld1q_dup_f32(A + oa3);
                c30 = vmlaq_f32(c30, a0, b0);
                c31 = vmlaq_f32(c31, a0, b1);
                c32 = vmlaq_f32(c32, a0, b2);
                a0 = vld1q_dup_f32(A + oa4);
                c40 = vmlaq_f32(c40, a0, b0);
                c41 = vmlaq_f32(c41, a0, b1);
                c42 = vmlaq_f32(c42, a0, b2);
                a0 = vld1q_dup_f32(A + oa5);
                c50 = vmlaq_f32(c50, a0, b0);
                c51 = vmlaq_f32(c51, a0, b1);
                c52 = vmlaq_f32(c52, a0, b2);
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01);
            AddProduct(C + 2 * F, _alpha, c02, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11);
            AddProduct(C + 2 * F, _alpha, c12, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21);
            AddProduct(C + 2 * F, _alpha, c22, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31);
            AddProduct(C + 2 * F, _alpha, c32, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41);
            AddProduct(C + 2 * F, _alpha, c42, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51);
            AddProduct(C + 2 * F, _alpha, c52, tail);
        }
#endif

        void GemmKernel6x8nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c40 = vdupq_n_f32(0);
            float32x4_t c50 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c21 = vdupq_n_f32(0);
            float32x4_t c31 = vdupq_n_f32(0);
            float32x4_t c41 = vdupq_n_f32(0);
            float32x4_t c51 = vdupq_n_f32(0);
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            float32x4_t b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                a0 = vld1q_dup_f32(A + oa0);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                a0 = vld1q_dup_f32(A + oa1);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                a0 = vld1q_dup_f32(A + oa2);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                a0 = vld1q_dup_f32(A + oa3);
                c30 = vmlaq_f32(c30, a0, b0);
                c31 = vmlaq_f32(c31, a0, b1);
                a0 = vld1q_dup_f32(A + oa4);
                c40 = vmlaq_f32(c40, a0, b0);
                c41 = vmlaq_f32(c41, a0, b1);
                a0 = vld1q_dup_f32(A + oa5);
                c50 = vmlaq_f32(c50, a0, b0);
                c51 = vmlaq_f32(c51, a0, b1);
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51, tail);
        }

        void GemmKernel6x4nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c0 = vdupq_n_f32(0);
            float32x4_t c1 = vdupq_n_f32(0);
            float32x4_t c2 = vdupq_n_f32(0);
            float32x4_t c3 = vdupq_n_f32(0);
            float32x4_t c4 = vdupq_n_f32(0);
            float32x4_t c5 = vdupq_n_f32(0);
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            const size_t ob0 = ldb * 0;
            float32x4_t b0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                c0 = vmlaq_f32(c0, b0, vld1q_dup_f32(A + oa0));
                c1 = vmlaq_f32(c1, b0, vld1q_dup_f32(A + oa1));
                c2 = vmlaq_f32(c2, b0, vld1q_dup_f32(A + oa2));
                c3 = vmlaq_f32(c3, b0, vld1q_dup_f32(A + oa3));
                c4 = vmlaq_f32(c4, b0, vld1q_dup_f32(A + oa4));
                c5 = vmlaq_f32(c5, b0, vld1q_dup_f32(A + oa5));
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * ldc, _alpha, c0, tail);
            AddProduct(C + 1 * ldc, _alpha, c1, tail);
            AddProduct(C + 2 * ldc, _alpha, c2, tail);
            AddProduct(C + 3 * ldc, _alpha, c3, tail);
            AddProduct(C + 4 * ldc, _alpha, c4, tail);
            AddProduct(C + 5 * ldc, _alpha, c5, tail);
        }

#if defined(SIMD_ARM64_ENABLE)
        void GemmKernel8x12nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c02 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c12 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c21 = vdupq_n_f32(0);
            float32x4_t c22 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c31 = vdupq_n_f32(0);
            float32x4_t c32 = vdupq_n_f32(0);
            float32x4_t c40 = vdupq_n_f32(0);
            float32x4_t c41 = vdupq_n_f32(0);
            float32x4_t c42 = vdupq_n_f32(0);
            float32x4_t c50 = vdupq_n_f32(0);
            float32x4_t c51 = vdupq_n_f32(0);
            float32x4_t c52 = vdupq_n_f32(0);
            float32x4_t c60 = vdupq_n_f32(0);
            float32x4_t c61 = vdupq_n_f32(0);
            float32x4_t c62 = vdupq_n_f32(0);
            float32x4_t c70 = vdupq_n_f32(0);
            float32x4_t c71 = vdupq_n_f32(0);
            float32x4_t c72 = vdupq_n_f32(0);
            float32x4_t b0, b1, b2, a0;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t oa6 = lda * 6;
            const size_t oa7 = lda * 7;
            const size_t sa = lda == 1 ? 8 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                b2 = Load<false>(B + ob2);
                a0 = vld1q_dup_f32(A + oa0);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                c02 = vmlaq_f32(c02, a0, b2);
                a0 = vld1q_dup_f32(A + oa1);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                c12 = vmlaq_f32(c12, a0, b2);
                a0 = vld1q_dup_f32(A + oa2);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                c22 = vmlaq_f32(c22, a0, b2);
                a0 = vld1q_dup_f32(A + oa3);
                c30 = vmlaq_f32(c30, a0, b0);
                c31 = vmlaq_f32(c31, a0, b1);
                c32 = vmlaq_f32(c32, a0, b2);
                a0 = vld1q_dup_f32(A + oa4);
                c40 = vmlaq_f32(c40, a0, b0);
                c41 = vmlaq_f32(c41, a0, b1);
                c42 = vmlaq_f32(c42, a0, b2);
                a0 = vld1q_dup_f32(A + oa5);
                c50 = vmlaq_f32(c50, a0, b0);
                c51 = vmlaq_f32(c51, a0, b1);
                c52 = vmlaq_f32(c52, a0, b2);
                a0 = vld1q_dup_f32(A + oa6);
                c60 = vmlaq_f32(c60, a0, b0);
                c61 = vmlaq_f32(c61, a0, b1);
                c62 = vmlaq_f32(c62, a0, b2);
                a0 = vld1q_dup_f32(A + oa7);
                c70 = vmlaq_f32(c70, a0, b0);
                c71 = vmlaq_f32(c71, a0, b1);
                c72 = vmlaq_f32(c72, a0, b2);
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01);
            AddProduct(C + 2 * F, _alpha, c02, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11);
            AddProduct(C + 2 * F, _alpha, c12, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21);
            AddProduct(C + 2 * F, _alpha, c22, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31);
            AddProduct(C + 2 * F, _alpha, c32, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41);
            AddProduct(C + 2 * F, _alpha, c42, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51);
            AddProduct(C + 2 * F, _alpha, c52, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60);
            AddProduct(C + 1 * F, _alpha, c61);
            AddProduct(C + 2 * F, _alpha, c62, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70);
            AddProduct(C + 1 * F, _alpha, c71);
            AddProduct(C + 2 * F, _alpha, c72, tail);
        }

        void GemmKernel8x8nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c21 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c31 = vdupq_n_f32(0);
            float32x4_t c40 = vdupq_n_f32(0);
            float32x4_t c41 = vdupq_n_f32(0);
            float32x4_t c50 = vdupq_n_f32(0);
            float32x4_t c51 = vdupq_n_f32(0);
            float32x4_t c60 = vdupq_n_f32(0);
            float32x4_t c61 = vdupq_n_f32(0);
            float32x4_t c70 = vdupq_n_f32(0);
            float32x4_t c71 = vdupq_n_f32(0);
            float32x4_t b0, b1, a0;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t oa6 = lda * 6;
            const size_t oa7 = lda * 7;
            const size_t sa = lda == 1 ? 8 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                a0 = vld1q_dup_f32(A + oa0);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                a0 = vld1q_dup_f32(A + oa1);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                a0 = vld1q_dup_f32(A + oa2);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                a0 = vld1q_dup_f32(A + oa3);
                c30 = vmlaq_f32(c30, a0, b0);
                c31 = vmlaq_f32(c31, a0, b1);
                a0 = vld1q_dup_f32(A + oa4);
                c40 = vmlaq_f32(c40, a0, b0);
                c41 = vmlaq_f32(c41, a0, b1);
                a0 = vld1q_dup_f32(A + oa5);
                c50 = vmlaq_f32(c50, a0, b0);
                c51 = vmlaq_f32(c51, a0, b1);
                a0 = vld1q_dup_f32(A + oa6);
                c60 = vmlaq_f32(c60, a0, b0);
                c61 = vmlaq_f32(c61, a0, b1);
                a0 = vld1q_dup_f32(A + oa7);
                c70 = vmlaq_f32(c70, a0, b0);
                c71 = vmlaq_f32(c71, a0, b1);
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60);
            AddProduct(C + 1 * F, _alpha, c61, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70);
            AddProduct(C + 1 * F, _alpha, c71, tail);
        }

        void GemmKernel8x4nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c40 = vdupq_n_f32(0);
            float32x4_t c50 = vdupq_n_f32(0);
            float32x4_t c60 = vdupq_n_f32(0);
            float32x4_t c70 = vdupq_n_f32(0);
            float32x4_t b0, a0;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t oa6 = lda * 6;
            const size_t oa7 = lda * 7;
            const size_t sa = lda == 1 ? 8 : 1;
            const size_t ob0 = ldb * 0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                a0 = vld1q_dup_f32(A + oa0);
                c00 = vmlaq_f32(c00, a0, b0);
                a0 = vld1q_dup_f32(A + oa1);
                c10 = vmlaq_f32(c10, a0, b0);
                a0 = vld1q_dup_f32(A + oa2);
                c20 = vmlaq_f32(c20, a0, b0);
                a0 = vld1q_dup_f32(A + oa3);
                c30 = vmlaq_f32(c30, a0, b0);
                a0 = vld1q_dup_f32(A + oa4);
                c40 = vmlaq_f32(c40, a0, b0);
                a0 = vld1q_dup_f32(A + oa5);
                c50 = vmlaq_f32(c50, a0, b0);
                a0 = vld1q_dup_f32(A + oa6);
                c60 = vmlaq_f32(c60, a0, b0);
                a0 = vld1q_dup_f32(A + oa7);
                c70 = vmlaq_f32(c70, a0, b0);
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * F, _alpha, c00, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70, tail);
        }
#endif

#if defined(SIMD_ARM64_ENABLE) && defined(SIMD_ARM64_KERNEL_9X)
        void GemmKernel9x12nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c02 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c12 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c21 = vdupq_n_f32(0);
            float32x4_t c22 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c31 = vdupq_n_f32(0);
            float32x4_t c32 = vdupq_n_f32(0);
            float32x4_t c40 = vdupq_n_f32(0);
            float32x4_t c41 = vdupq_n_f32(0);
            float32x4_t c42 = vdupq_n_f32(0);
            float32x4_t c50 = vdupq_n_f32(0);
            float32x4_t c51 = vdupq_n_f32(0);
            float32x4_t c52 = vdupq_n_f32(0);
            float32x4_t c60 = vdupq_n_f32(0);
            float32x4_t c61 = vdupq_n_f32(0);
            float32x4_t c62 = vdupq_n_f32(0);
            float32x4_t c70 = vdupq_n_f32(0);
            float32x4_t c71 = vdupq_n_f32(0);
            float32x4_t c72 = vdupq_n_f32(0);
            float32x4_t c80 = vdupq_n_f32(0);
            float32x4_t c81 = vdupq_n_f32(0);
            float32x4_t c82 = vdupq_n_f32(0);
            float32x4_t b0, b1, b2, a0;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t oa6 = lda * 6;
            const size_t oa7 = lda * 7;
            const size_t oa8 = lda * 8;
            const size_t sa = lda == 1 ? 9 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                b2 = Load<false>(B + ob2);
                a0 = vld1q_dup_f32(A + oa0);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                c02 = vmlaq_f32(c02, a0, b2);
                a0 = vld1q_dup_f32(A + oa1);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                c12 = vmlaq_f32(c12, a0, b2);
                a0 = vld1q_dup_f32(A + oa2);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                c22 = vmlaq_f32(c22, a0, b2);
                a0 = vld1q_dup_f32(A + oa3);
                c30 = vmlaq_f32(c30, a0, b0);
                c31 = vmlaq_f32(c31, a0, b1);
                c32 = vmlaq_f32(c32, a0, b2);
                a0 = vld1q_dup_f32(A + oa4);
                c40 = vmlaq_f32(c40, a0, b0);
                c41 = vmlaq_f32(c41, a0, b1);
                c42 = vmlaq_f32(c42, a0, b2);
                a0 = vld1q_dup_f32(A + oa5);
                c50 = vmlaq_f32(c50, a0, b0);
                c51 = vmlaq_f32(c51, a0, b1);
                c52 = vmlaq_f32(c52, a0, b2);
                a0 = vld1q_dup_f32(A + oa6);
                c60 = vmlaq_f32(c60, a0, b0);
                c61 = vmlaq_f32(c61, a0, b1);
                c62 = vmlaq_f32(c62, a0, b2);
                a0 = vld1q_dup_f32(A + oa7);
                c70 = vmlaq_f32(c70, a0, b0);
                c71 = vmlaq_f32(c71, a0, b1);
                c72 = vmlaq_f32(c72, a0, b2);
                a0 = vld1q_dup_f32(A + oa8);
                c80 = vmlaq_f32(c80, a0, b0);
                c81 = vmlaq_f32(c81, a0, b1);
                c82 = vmlaq_f32(c82, a0, b2);
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01);
            AddProduct(C + 2 * F, _alpha, c02, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11);
            AddProduct(C + 2 * F, _alpha, c12, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21);
            AddProduct(C + 2 * F, _alpha, c22, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31);
            AddProduct(C + 2 * F, _alpha, c32, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41);
            AddProduct(C + 2 * F, _alpha, c42, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51);
            AddProduct(C + 2 * F, _alpha, c52, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60);
            AddProduct(C + 1 * F, _alpha, c61);
            AddProduct(C + 2 * F, _alpha, c62, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70);
            AddProduct(C + 1 * F, _alpha, c71);
            AddProduct(C + 2 * F, _alpha, c72, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c80);
            AddProduct(C + 1 * F, _alpha, c81);
            AddProduct(C + 2 * F, _alpha, c82, tail);
        }

        void GemmKernel9x8nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c21 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c31 = vdupq_n_f32(0);
            float32x4_t c40 = vdupq_n_f32(0);
            float32x4_t c41 = vdupq_n_f32(0);
            float32x4_t c50 = vdupq_n_f32(0);
            float32x4_t c51 = vdupq_n_f32(0);
            float32x4_t c60 = vdupq_n_f32(0);
            float32x4_t c61 = vdupq_n_f32(0);
            float32x4_t c70 = vdupq_n_f32(0);
            float32x4_t c71 = vdupq_n_f32(0);
            float32x4_t c80 = vdupq_n_f32(0);
            float32x4_t c81 = vdupq_n_f32(0);
            float32x4_t b0, b1, a0;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t oa6 = lda * 6;
            const size_t oa7 = lda * 7;
            const size_t oa8 = lda * 8;
            const size_t sa = lda == 1 ? 9 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                a0 = vld1q_dup_f32(A + oa0);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                a0 = vld1q_dup_f32(A + oa1);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                a0 = vld1q_dup_f32(A + oa2);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                a0 = vld1q_dup_f32(A + oa3);
                c30 = vmlaq_f32(c30, a0, b0);
                c31 = vmlaq_f32(c31, a0, b1);
                a0 = vld1q_dup_f32(A + oa4);
                c40 = vmlaq_f32(c40, a0, b0);
                c41 = vmlaq_f32(c41, a0, b1);
                a0 = vld1q_dup_f32(A + oa5);
                c50 = vmlaq_f32(c50, a0, b0);
                c51 = vmlaq_f32(c51, a0, b1);
                a0 = vld1q_dup_f32(A + oa6);
                c60 = vmlaq_f32(c60, a0, b0);
                c61 = vmlaq_f32(c61, a0, b1);
                a0 = vld1q_dup_f32(A + oa7);
                c70 = vmlaq_f32(c70, a0, b0);
                c71 = vmlaq_f32(c71, a0, b1);
                a0 = vld1q_dup_f32(A + oa8);
                c80 = vmlaq_f32(c80, a0, b0);
                c81 = vmlaq_f32(c81, a0, b1);
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60);
            AddProduct(C + 1 * F, _alpha, c61, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70);
            AddProduct(C + 1 * F, _alpha, c71, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c80);
            AddProduct(C + 1 * F, _alpha, c81, tail);
        }

        void GemmKernel9x4nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c40 = vdupq_n_f32(0);
            float32x4_t c50 = vdupq_n_f32(0);
            float32x4_t c60 = vdupq_n_f32(0);
            float32x4_t c70 = vdupq_n_f32(0);
            float32x4_t c80 = vdupq_n_f32(0);
            float32x4_t b0, a0;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t oa6 = lda * 6;
            const size_t oa7 = lda * 7;
            const size_t oa8 = lda * 8;
            const size_t sa = lda == 1 ? 9 : 1;
            const size_t ob0 = ldb * 0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                a0 = vld1q_dup_f32(A + oa0);
                c00 = vmlaq_f32(c00, a0, b0);
                a0 = vld1q_dup_f32(A + oa1);
                c10 = vmlaq_f32(c10, a0, b0);
                a0 = vld1q_dup_f32(A + oa2);
                c20 = vmlaq_f32(c20, a0, b0);
                a0 = vld1q_dup_f32(A + oa3);
                c30 = vmlaq_f32(c30, a0, b0);
                a0 = vld1q_dup_f32(A + oa4);
                c40 = vmlaq_f32(c40, a0, b0);
                a0 = vld1q_dup_f32(A + oa5);
                c50 = vmlaq_f32(c50, a0, b0);
                a0 = vld1q_dup_f32(A + oa6);
                c60 = vmlaq_f32(c60, a0, b0);
                a0 = vld1q_dup_f32(A + oa7);
                c70 = vmlaq_f32(c70, a0, b0);
                a0 = vld1q_dup_f32(A + oa8);
                c80 = vmlaq_f32(c80, a0, b0);
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * F, _alpha, c00, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c80, tail);
        }
#endif

#if defined(SIMD_ARM64_ENABLE)
        void GemmKernel12x8nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c21 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c31 = vdupq_n_f32(0);
            float32x4_t c40 = vdupq_n_f32(0);
            float32x4_t c41 = vdupq_n_f32(0);
            float32x4_t c50 = vdupq_n_f32(0);
            float32x4_t c51 = vdupq_n_f32(0);
            float32x4_t c60 = vdupq_n_f32(0);
            float32x4_t c61 = vdupq_n_f32(0);
            float32x4_t c70 = vdupq_n_f32(0);
            float32x4_t c71 = vdupq_n_f32(0);
            float32x4_t c80 = vdupq_n_f32(0);
            float32x4_t c81 = vdupq_n_f32(0);
            float32x4_t c90 = vdupq_n_f32(0);
            float32x4_t c91 = vdupq_n_f32(0);
            float32x4_t cA0 = vdupq_n_f32(0);
            float32x4_t cA1 = vdupq_n_f32(0);
            float32x4_t cB0 = vdupq_n_f32(0);
            float32x4_t cB1 = vdupq_n_f32(0);
            const float* A0 = A, * A6 = A + 6 * lda;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 12 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            float32x4_t b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                a0 = vld1q_dup_f32(A0 + oa0);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                a0 = vld1q_dup_f32(A0 + oa1);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                a0 = vld1q_dup_f32(A0 + oa2);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                a0 = vld1q_dup_f32(A0 + oa3);
                c30 = vmlaq_f32(c30, a0, b0);
                c31 = vmlaq_f32(c31, a0, b1);
                a0 = vld1q_dup_f32(A0 + oa4);
                c40 = vmlaq_f32(c40, a0, b0);
                c41 = vmlaq_f32(c41, a0, b1);
                a0 = vld1q_dup_f32(A0 + oa5);
                c50 = vmlaq_f32(c50, a0, b0);
                c51 = vmlaq_f32(c51, a0, b1);
                a0 = vld1q_dup_f32(A6 + oa0);
                c60 = vmlaq_f32(c60, a0, b0);
                c61 = vmlaq_f32(c61, a0, b1);
                a0 = vld1q_dup_f32(A6 + oa1);
                c70 = vmlaq_f32(c70, a0, b0);
                c71 = vmlaq_f32(c71, a0, b1);
                a0 = vld1q_dup_f32(A6 + oa2);
                c80 = vmlaq_f32(c80, a0, b0);
                c81 = vmlaq_f32(c81, a0, b1);
                a0 = vld1q_dup_f32(A6 + oa3);
                c90 = vmlaq_f32(c90, a0, b0);
                c91 = vmlaq_f32(c91, a0, b1);
                a0 = vld1q_dup_f32(A6 + oa4);
                cA0 = vmlaq_f32(cA0, a0, b0);
                cA1 = vmlaq_f32(cA1, a0, b1);
                a0 = vld1q_dup_f32(A6 + oa5);
                cB0 = vmlaq_f32(cB0, a0, b0);
                cB1 = vmlaq_f32(cB1, a0, b1);
                B += sb;
                A0 += sa;
                A6 += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60);
            AddProduct(C + 1 * F, _alpha, c61, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70);
            AddProduct(C + 1 * F, _alpha, c71, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c80);
            AddProduct(C + 1 * F, _alpha, c81, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c90);
            AddProduct(C + 1 * F, _alpha, c91, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cA0);
            AddProduct(C + 1 * F, _alpha, cA1, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cB0);
            AddProduct(C + 1 * F, _alpha, cB1, tail);
        }

        void GemmKernel12x4nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c40 = vdupq_n_f32(0);
            float32x4_t c50 = vdupq_n_f32(0);
            float32x4_t c60 = vdupq_n_f32(0);
            float32x4_t c70 = vdupq_n_f32(0);
            float32x4_t c80 = vdupq_n_f32(0);
            float32x4_t c90 = vdupq_n_f32(0);
            float32x4_t cA0 = vdupq_n_f32(0);
            float32x4_t cB0 = vdupq_n_f32(0);
            const float* A0 = A, * A6 = A + 6 * lda;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 12 : 1;
            const size_t ob0 = ldb * 0;
            float32x4_t b0, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                a0 = vld1q_dup_f32(A0 + oa0);
                c00 = vmlaq_f32(c00, a0, b0);
                a0 = vld1q_dup_f32(A0 + oa1);
                c10 = vmlaq_f32(c10, a0, b0);
                a0 = vld1q_dup_f32(A0 + oa2);
                c20 = vmlaq_f32(c20, a0, b0);
                a0 = vld1q_dup_f32(A0 + oa3);
                c30 = vmlaq_f32(c30, a0, b0);
                a0 = vld1q_dup_f32(A0 + oa4);
                c40 = vmlaq_f32(c40, a0, b0);
                a0 = vld1q_dup_f32(A0 + oa5);
                c50 = vmlaq_f32(c50, a0, b0);
                a0 = vld1q_dup_f32(A6 + oa0);
                c60 = vmlaq_f32(c60, a0, b0);
                a0 = vld1q_dup_f32(A6 + oa1);
                c70 = vmlaq_f32(c70, a0, b0);
                a0 = vld1q_dup_f32(A6 + oa2);
                c80 = vmlaq_f32(c80, a0, b0);
                a0 = vld1q_dup_f32(A6 + oa3);
                c90 = vmlaq_f32(c90, a0, b0);
                a0 = vld1q_dup_f32(A6 + oa4);
                cA0 = vmlaq_f32(cA0, a0, b0);
                a0 = vld1q_dup_f32(A6 + oa5);
                cB0 = vmlaq_f32(cB0, a0, b0);
                B += sb;
                A0 += sa;
                A6 += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * F, _alpha, c00, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c80, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c90, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cA0, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cB0, tail);
        }
#endif

#if defined(SIMD_ARM64_ENABLE) && defined(SIMD_ARM64_KERNEL_14X)
        void GemmKernel14x8nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c21 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c31 = vdupq_n_f32(0);
            float32x4_t c40 = vdupq_n_f32(0);
            float32x4_t c41 = vdupq_n_f32(0);
            float32x4_t c50 = vdupq_n_f32(0);
            float32x4_t c51 = vdupq_n_f32(0);
            float32x4_t c60 = vdupq_n_f32(0);
            float32x4_t c61 = vdupq_n_f32(0);
            float32x4_t c70 = vdupq_n_f32(0);
            float32x4_t c71 = vdupq_n_f32(0);
            float32x4_t c80 = vdupq_n_f32(0);
            float32x4_t c81 = vdupq_n_f32(0);
            float32x4_t c90 = vdupq_n_f32(0);
            float32x4_t c91 = vdupq_n_f32(0);
            float32x4_t cA0 = vdupq_n_f32(0);
            float32x4_t cA1 = vdupq_n_f32(0);
            float32x4_t cB0 = vdupq_n_f32(0);
            float32x4_t cB1 = vdupq_n_f32(0);
            float32x4_t cC0 = vdupq_n_f32(0);
            float32x4_t cC1 = vdupq_n_f32(0);
            float32x4_t cD0 = vdupq_n_f32(0);
            float32x4_t cD1 = vdupq_n_f32(0);
            const float* A0 = A, * A7 = A + 7 * lda;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t oa6 = lda * 6;
            const size_t sa = lda == 1 ? 14 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            float32x4_t b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                a0 = vld1q_dup_f32(A0 + oa0);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                a0 = vld1q_dup_f32(A0 + oa1);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                a0 = vld1q_dup_f32(A0 + oa2);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                a0 = vld1q_dup_f32(A0 + oa3);
                c30 = vmlaq_f32(c30, a0, b0);
                c31 = vmlaq_f32(c31, a0, b1);
                a0 = vld1q_dup_f32(A0 + oa4);
                c40 = vmlaq_f32(c40, a0, b0);
                c41 = vmlaq_f32(c41, a0, b1);
                a0 = vld1q_dup_f32(A0 + oa5);
                c50 = vmlaq_f32(c50, a0, b0);
                c51 = vmlaq_f32(c51, a0, b1);
                a0 = vld1q_dup_f32(A0 + oa6);
                c60 = vmlaq_f32(c60, a0, b0);
                c61 = vmlaq_f32(c61, a0, b1);
                a0 = vld1q_dup_f32(A7 + oa0);
                c70 = vmlaq_f32(c70, a0, b0);
                c71 = vmlaq_f32(c71, a0, b1);
                a0 = vld1q_dup_f32(A7 + oa1);
                c80 = vmlaq_f32(c80, a0, b0);
                c81 = vmlaq_f32(c81, a0, b1);
                a0 = vld1q_dup_f32(A7 + oa2);
                c90 = vmlaq_f32(c90, a0, b0);
                c91 = vmlaq_f32(c91, a0, b1);
                a0 = vld1q_dup_f32(A7 + oa3);
                cA0 = vmlaq_f32(cA0, a0, b0);
                cA1 = vmlaq_f32(cA1, a0, b1);
                a0 = vld1q_dup_f32(A7 + oa4);
                cB0 = vmlaq_f32(cB0, a0, b0);
                cB1 = vmlaq_f32(cB1, a0, b1);
                a0 = vld1q_dup_f32(A7 + oa5);
                cC0 = vmlaq_f32(cC0, a0, b0);
                cC1 = vmlaq_f32(cC1, a0, b1);
                a0 = vld1q_dup_f32(A7 + oa6);
                cD0 = vmlaq_f32(cD0, a0, b0);
                cD1 = vmlaq_f32(cD1, a0, b1);               
                B += sb;
                A0 += sa;
                A7 += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60);
            AddProduct(C + 1 * F, _alpha, c61, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70);
            AddProduct(C + 1 * F, _alpha, c71, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c80);
            AddProduct(C + 1 * F, _alpha, c81, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c90);
            AddProduct(C + 1 * F, _alpha, c91, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cA0);
            AddProduct(C + 1 * F, _alpha, cA1, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cB0);
            AddProduct(C + 1 * F, _alpha, cB1, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cC0);
            AddProduct(C + 1 * F, _alpha, cC1, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cD0);
            AddProduct(C + 1 * F, _alpha, cD1, tail);
        }

        void GemmKernel14x4nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c40 = vdupq_n_f32(0);
            float32x4_t c50 = vdupq_n_f32(0);
            float32x4_t c60 = vdupq_n_f32(0);
            float32x4_t c70 = vdupq_n_f32(0);
            float32x4_t c80 = vdupq_n_f32(0);
            float32x4_t c90 = vdupq_n_f32(0);
            float32x4_t cA0 = vdupq_n_f32(0);
            float32x4_t cB0 = vdupq_n_f32(0);
            float32x4_t cC0 = vdupq_n_f32(0);
            float32x4_t cD0 = vdupq_n_f32(0);
            const float* A0 = A, * A7 = A + 7 * lda;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t oa6 = lda * 6;
            const size_t sa = lda == 1 ? 14 : 1;
            const size_t ob0 = ldb * 0;
            float32x4_t b0, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                a0 = vld1q_dup_f32(A0 + oa0);
                c00 = vmlaq_f32(c00, a0, b0);
                a0 = vld1q_dup_f32(A0 + oa1);
                c10 = vmlaq_f32(c10, a0, b0);
                a0 = vld1q_dup_f32(A0 + oa2);
                c20 = vmlaq_f32(c20, a0, b0);
                a0 = vld1q_dup_f32(A0 + oa3);
                c30 = vmlaq_f32(c30, a0, b0);
                a0 = vld1q_dup_f32(A0 + oa4);
                c40 = vmlaq_f32(c40, a0, b0);
                a0 = vld1q_dup_f32(A0 + oa5);
                c50 = vmlaq_f32(c50, a0, b0);
                a0 = vld1q_dup_f32(A0 + oa6);
                c60 = vmlaq_f32(c60, a0, b0);
                a0 = vld1q_dup_f32(A7 + oa0);
                c70 = vmlaq_f32(c70, a0, b0);
                a0 = vld1q_dup_f32(A7 + oa1);
                c80 = vmlaq_f32(c80, a0, b0);
                a0 = vld1q_dup_f32(A7 + oa2);
                c90 = vmlaq_f32(c90, a0, b0);
                a0 = vld1q_dup_f32(A7 + oa3);
                cA0 = vmlaq_f32(cA0, a0, b0);
                a0 = vld1q_dup_f32(A7 + oa4);
                cB0 = vmlaq_f32(cB0, a0, b0);
                a0 = vld1q_dup_f32(A7 + oa5);
                cC0 = vmlaq_f32(cC0, a0, b0);
                a0 = vld1q_dup_f32(A7 + oa6);
                cD0 = vmlaq_f32(cD0, a0, b0);
                B += sb;
                A0 += sa;
                A7 += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * F, _alpha, c00, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c80, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c90, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cA0, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cB0, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cC0, tail);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cD0, tail);
        }
#endif

        void GemmKernelMx16nn(size_t M, size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, size_t tail)
        {
#ifdef SIMD_ARM64_ENABLE
            float32x4_t c[6][4];
            size_t oa[6];
#else
            float32x4_t c[3][4];
            size_t oa[3];
#endif
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            const size_t ob3 = ldb * 3;
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = vdupq_n_f32(0);
                c[i][1] = vdupq_n_f32(0);
                c[i][2] = vdupq_n_f32(0);
                c[i][3] = vdupq_n_f32(0);
                oa[i] = lda * i;
            }
            float32x4_t b0, b1, b2, b3, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                b2 = Load<false>(B + ob2);
                b3 = Load<false>(B + ob3);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = vld1q_dup_f32(A + oa[i]);
                    c[i][0] = vmlaq_f32(c[i][0], b0, a0);
                    c[i][1] = vmlaq_f32(c[i][1], b1, a0);
                    c[i][2] = vmlaq_f32(c[i][2], b2, a0);
                    c[i][3] = vmlaq_f32(c[i][3], b3, a0);
                }
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            for (size_t i = 0; i < M; ++i)
            {
                AddProduct(C + 0 * F, _alpha, c[i][0]);
                AddProduct(C + 1 * F, _alpha, c[i][1]);
                AddProduct(C + 2 * F, _alpha, c[i][2]);
                AddProduct(C + 3 * F, _alpha, c[i][3], tail);
                C += ldc;
            }
        }

        void GemmKernelMx12nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
#ifdef SIMD_ARM64_ENABLE
            float32x4_t c[8][3];
            size_t oa[8];
#else
            float32x4_t c[4][3];
            size_t oa[4];
#endif
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = vdupq_n_f32(0);
                c[i][1] = vdupq_n_f32(0);
                c[i][2] = vdupq_n_f32(0);
                oa[i] = lda * i;
            }
            float32x4_t b0, b1, b2, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                b2 = Load<false>(B + ob2);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = vld1q_dup_f32(A + oa[i]);
                    c[i][0] = vmlaq_f32(c[i][0], b0, a0);
                    c[i][1] = vmlaq_f32(c[i][1], b1, a0);
                    c[i][2] = vmlaq_f32(c[i][2], b2, a0);
                }
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            for (size_t i = 0; i < M; ++i)
            {
                AddProduct(C + 0 * F, _alpha, c[i][0]);
                AddProduct(C + 1 * F, _alpha, c[i][1]);
                AddProduct(C + 2 * F, _alpha, c[i][2], tail);
                C += ldc;
            }
        }

        void GemmKernelMx8nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
#ifdef SIMD_ARM64_ENABLE
            float32x4_t c[12][2];
            size_t oa[12];
#else
            float32x4_t c[6][2];
            size_t oa[6];
#endif
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = vdupq_n_f32(0);
                c[i][1] = vdupq_n_f32(0);
                oa[i] = lda * i;
            }
            float32x4_t b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = vld1q_dup_f32(A + oa[i]);
                    c[i][0] = vmlaq_f32(c[i][0], b0, a0);
                    c[i][1] = vmlaq_f32(c[i][1], b1, a0);
                }
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            for (size_t i = 0; i < M; ++i)
            {
                AddProduct(C + 0 * F, _alpha, c[i][0]);
                AddProduct(C + 1 * F, _alpha, c[i][1], tail);
                C += ldc;
            }
        }

        void GemmKernelMx4nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
#ifdef SIMD_ARM64_ENABLE
            float32x4_t c[12];
            size_t oa[12];
#else
            float32x4_t c[6];
            size_t oa[6];
#endif
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            for (size_t i = 0; i < M; ++i)
            {
                c[i] = vdupq_n_f32(0);
                oa[i] = lda * i;
            }
            float32x4_t b0, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = vld1q_dup_f32(A + oa[i]);
                    c[i] = vmlaq_f32(c[i], b0, a0);
                }
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            for (size_t i = 0; i < M; ++i)
                AddProduct(C + i * ldc, _alpha, c[i], tail);
        }

        template<int M> void GemmKernelMx16nnT(size_t, size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, size_t tail)
        {
            float32x4_t c00, c01, c02, c03, c04, c05, c10, c11, c12, c13, c14, c15, c20, c21, c22, c23, c24, c25, c30, c31, c32, c33, c34, c35, b0, b1, b2, b3, a0;
            if (M > 0) c00 = vdupq_n_f32(0), c10 = vdupq_n_f32(0), c20 = vdupq_n_f32(0), c30 = vdupq_n_f32(0);
            if (M > 1) c01 = vdupq_n_f32(0), c11 = vdupq_n_f32(0), c21 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
            if (M > 2) c02 = vdupq_n_f32(0), c12 = vdupq_n_f32(0), c22 = vdupq_n_f32(0), c32 = vdupq_n_f32(0);
            if (M > 3) c03 = vdupq_n_f32(0), c13 = vdupq_n_f32(0), c23 = vdupq_n_f32(0), c33 = vdupq_n_f32(0);
            if (M > 4) c04 = vdupq_n_f32(0), c14 = vdupq_n_f32(0), c24 = vdupq_n_f32(0), c34 = vdupq_n_f32(0);
            if (M > 5) c05 = vdupq_n_f32(0), c15 = vdupq_n_f32(0), c25 = vdupq_n_f32(0), c35 = vdupq_n_f32(0);
            size_t oa0, oa1, oa2, oa3, oa4, oa5;
            if (M > 0) oa0 = lda * 0;
            if (M > 1) oa1 = lda * 1;
            if (M > 2) oa2 = lda * 2;
            if (M > 3) oa3 = lda * 3;
            if (M > 4) oa4 = lda * 4;
            if (M > 5) oa5 = lda * 5;
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            const size_t ob3 = ldb * 3;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                b2 = Load<false>(B + ob2);
                b3 = Load<false>(B + ob3);
                if (M > 0) a0 = vld1q_dup_f32(A + oa0), c00 = vmlaq_f32(c00, b0, a0), c10 = vmlaq_f32(c10, b1, a0), c20 = vmlaq_f32(c20, b2, a0), c30 = vmlaq_f32(c30, b3, a0);
                if (M > 1) a0 = vld1q_dup_f32(A + oa1), c01 = vmlaq_f32(c01, b0, a0), c11 = vmlaq_f32(c11, b1, a0), c21 = vmlaq_f32(c21, b2, a0), c31 = vmlaq_f32(c31, b3, a0);
                if (M > 2) a0 = vld1q_dup_f32(A + oa2), c02 = vmlaq_f32(c02, b0, a0), c12 = vmlaq_f32(c12, b1, a0), c22 = vmlaq_f32(c22, b2, a0), c32 = vmlaq_f32(c32, b3, a0);
                if (M > 3) a0 = vld1q_dup_f32(A + oa3), c03 = vmlaq_f32(c03, b0, a0), c13 = vmlaq_f32(c13, b1, a0), c23 = vmlaq_f32(c23, b2, a0), c33 = vmlaq_f32(c33, b3, a0);
                if (M > 4) a0 = vld1q_dup_f32(A + oa4), c04 = vmlaq_f32(c04, b0, a0), c14 = vmlaq_f32(c14, b1, a0), c24 = vmlaq_f32(c24, b2, a0), c34 = vmlaq_f32(c34, b3, a0);
                if (M > 5) a0 = vld1q_dup_f32(A + oa5), c05 = vmlaq_f32(c05, b0, a0), c15 = vmlaq_f32(c15, b1, a0), c25 = vmlaq_f32(c25, b2, a0), c35 = vmlaq_f32(c35, b3, a0);
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            if (M > 0) AddProduct(C + 0 * F, _alpha, c00), AddProduct(C + 1 * F, _alpha, c10), AddProduct(C + 2 * F, _alpha, c20), AddProduct(C + 3 * F, _alpha, c30, tail), C += ldc;
            if (M > 1) AddProduct(C + 0 * F, _alpha, c01), AddProduct(C + 1 * F, _alpha, c11), AddProduct(C + 2 * F, _alpha, c21), AddProduct(C + 3 * F, _alpha, c31, tail), C += ldc;
            if (M > 2) AddProduct(C + 0 * F, _alpha, c02), AddProduct(C + 1 * F, _alpha, c12), AddProduct(C + 2 * F, _alpha, c22), AddProduct(C + 3 * F, _alpha, c32, tail), C += ldc;
            if (M > 3) AddProduct(C + 0 * F, _alpha, c03), AddProduct(C + 1 * F, _alpha, c13), AddProduct(C + 2 * F, _alpha, c23), AddProduct(C + 3 * F, _alpha, c33, tail), C += ldc;
            if (M > 4) AddProduct(C + 0 * F, _alpha, c04), AddProduct(C + 1 * F, _alpha, c14), AddProduct(C + 2 * F, _alpha, c24), AddProduct(C + 3 * F, _alpha, c34, tail), C += ldc;
            if (M > 5) AddProduct(C + 0 * F, _alpha, c05), AddProduct(C + 1 * F, _alpha, c15), AddProduct(C + 2 * F, _alpha, c25), AddProduct(C + 3 * F, _alpha, c35, tail), C += ldc;
        }

        template<int M> void GemmKernelMx12nnT(size_t, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c00, c01, c02, c03, c04, c05, c06, c07, c10, c11, c12, c13, c14, c15, c16, c17, c20, c21, c22, c23, c24, c25, c26, c27, b0, b1, b2, a0;
            if (M > 0) c00 = vdupq_n_f32(0), c10 = vdupq_n_f32(0), c20 = vdupq_n_f32(0);
            if (M > 1) c01 = vdupq_n_f32(0), c11 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
            if (M > 2) c02 = vdupq_n_f32(0), c12 = vdupq_n_f32(0), c22 = vdupq_n_f32(0);
            if (M > 3) c03 = vdupq_n_f32(0), c13 = vdupq_n_f32(0), c23 = vdupq_n_f32(0);
            if (M > 4) c04 = vdupq_n_f32(0), c14 = vdupq_n_f32(0), c24 = vdupq_n_f32(0);
            if (M > 5) c05 = vdupq_n_f32(0), c15 = vdupq_n_f32(0), c25 = vdupq_n_f32(0);
            if (M > 6) c06 = vdupq_n_f32(0), c16 = vdupq_n_f32(0), c26 = vdupq_n_f32(0);
            if (M > 7) c07 = vdupq_n_f32(0), c17 = vdupq_n_f32(0), c27 = vdupq_n_f32(0);
            const float* A0 = A, * A4 = A + 4 * lda;
            size_t oa0, oa1, oa2, oa3;
            if (M > 0) oa0 = lda * 0;
            if (M > 1) oa1 = lda * 1;
            if (M > 2) oa2 = lda * 2;
            if (M > 3) oa3 = lda * 3;
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                b2 = Load<false>(B + ob2);
                if (M > 0) a0 = vld1q_dup_f32(A0 + oa0), c00 = vmlaq_f32(c00, b0, a0), c10 = vmlaq_f32(c10, b1, a0), c20 = vmlaq_f32(c20, b2, a0);
                if (M > 1) a0 = vld1q_dup_f32(A0 + oa1), c01 = vmlaq_f32(c01, b0, a0), c11 = vmlaq_f32(c11, b1, a0), c21 = vmlaq_f32(c21, b2, a0);
                if (M > 2) a0 = vld1q_dup_f32(A0 + oa2), c02 = vmlaq_f32(c02, b0, a0), c12 = vmlaq_f32(c12, b1, a0), c22 = vmlaq_f32(c22, b2, a0);
                if (M > 3) a0 = vld1q_dup_f32(A0 + oa3), c03 = vmlaq_f32(c03, b0, a0), c13 = vmlaq_f32(c13, b1, a0), c23 = vmlaq_f32(c23, b2, a0);
                if (M > 4) a0 = vld1q_dup_f32(A4 + oa0), c04 = vmlaq_f32(c04, b0, a0), c14 = vmlaq_f32(c14, b1, a0), c24 = vmlaq_f32(c24, b2, a0);
                if (M > 5) a0 = vld1q_dup_f32(A4 + oa1), c05 = vmlaq_f32(c05, b0, a0), c15 = vmlaq_f32(c15, b1, a0), c25 = vmlaq_f32(c25, b2, a0);
                if (M > 6) a0 = vld1q_dup_f32(A4 + oa2), c06 = vmlaq_f32(c06, b0, a0), c16 = vmlaq_f32(c16, b1, a0), c26 = vmlaq_f32(c26, b2, a0);
                if (M > 7) a0 = vld1q_dup_f32(A4 + oa3), c07 = vmlaq_f32(c07, b0, a0), c17 = vmlaq_f32(c17, b1, a0), c27 = vmlaq_f32(c27, b2, a0);
                B += sb;
                A0 += sa;
                A4 += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            if (M > 0) AddProduct(C + 0 * F, _alpha, c00), AddProduct(C + 1 * F, _alpha, c10), AddProduct(C + 2 * F, _alpha, c20, tail), C += ldc;
            if (M > 1) AddProduct(C + 0 * F, _alpha, c01), AddProduct(C + 1 * F, _alpha, c11), AddProduct(C + 2 * F, _alpha, c21, tail), C += ldc;
            if (M > 2) AddProduct(C + 0 * F, _alpha, c02), AddProduct(C + 1 * F, _alpha, c12), AddProduct(C + 2 * F, _alpha, c22, tail), C += ldc;
            if (M > 3) AddProduct(C + 0 * F, _alpha, c03), AddProduct(C + 1 * F, _alpha, c13), AddProduct(C + 2 * F, _alpha, c23, tail), C += ldc;
            if (M > 4) AddProduct(C + 0 * F, _alpha, c04), AddProduct(C + 1 * F, _alpha, c14), AddProduct(C + 2 * F, _alpha, c24, tail), C += ldc;
            if (M > 5) AddProduct(C + 0 * F, _alpha, c05), AddProduct(C + 1 * F, _alpha, c15), AddProduct(C + 2 * F, _alpha, c25, tail), C += ldc;
            if (M > 6) AddProduct(C + 0 * F, _alpha, c06), AddProduct(C + 1 * F, _alpha, c16), AddProduct(C + 2 * F, _alpha, c26, tail), C += ldc;
            if (M > 7) AddProduct(C + 0 * F, _alpha, c07), AddProduct(C + 1 * F, _alpha, c17), AddProduct(C + 2 * F, _alpha, c27, tail), C += ldc;
        }

        template<int M> void GemmKernelMx8nnT(size_t, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, b0, b1, a0;
            if (M > 0x0) c00 = vdupq_n_f32(0), c10 = vdupq_n_f32(0);
            if (M > 0x1) c01 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
            if (M > 0x2) c02 = vdupq_n_f32(0), c12 = vdupq_n_f32(0);
            if (M > 0x3) c03 = vdupq_n_f32(0), c13 = vdupq_n_f32(0);
            if (M > 0x4) c04 = vdupq_n_f32(0), c14 = vdupq_n_f32(0);
            if (M > 0x5) c05 = vdupq_n_f32(0), c15 = vdupq_n_f32(0);
            if (M > 0x6) c06 = vdupq_n_f32(0), c16 = vdupq_n_f32(0);
            if (M > 0x7) c07 = vdupq_n_f32(0), c17 = vdupq_n_f32(0);
            if (M > 0x8) c08 = vdupq_n_f32(0), c18 = vdupq_n_f32(0);
            if (M > 0x9) c09 = vdupq_n_f32(0), c19 = vdupq_n_f32(0);
            if (M > 0xA) c0A = vdupq_n_f32(0), c1A = vdupq_n_f32(0);
            if (M > 0xB) c0B = vdupq_n_f32(0), c1B = vdupq_n_f32(0);
            const float* A0 = A, * A6 = A + 6 * lda;
            size_t oa0, oa1, oa2, oa3, oa4, oa5, oa6;
            if (M > 0) oa0 = lda * 0;
            if (M > 1) oa1 = lda * 1;
            if (M > 2) oa2 = lda * 2;
            if (M > 3) oa3 = lda * 3;
            if (M > 4) oa4 = lda * 4;
            if (M > 5) oa5 = lda * 5;
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                if (M > 0x0) a0 = vld1q_dup_f32(A0 + oa0), c00 = vmlaq_f32(c00, b0, a0), c10 = vmlaq_f32(c10, b1, a0);
                if (M > 0x1) a0 = vld1q_dup_f32(A0 + oa1), c01 = vmlaq_f32(c01, b0, a0), c11 = vmlaq_f32(c11, b1, a0);
                if (M > 0x2) a0 = vld1q_dup_f32(A0 + oa2), c02 = vmlaq_f32(c02, b0, a0), c12 = vmlaq_f32(c12, b1, a0);
                if (M > 0x3) a0 = vld1q_dup_f32(A0 + oa3), c03 = vmlaq_f32(c03, b0, a0), c13 = vmlaq_f32(c13, b1, a0);
                if (M > 0x4) a0 = vld1q_dup_f32(A0 + oa4), c04 = vmlaq_f32(c04, b0, a0), c14 = vmlaq_f32(c14, b1, a0);
                if (M > 0x5) a0 = vld1q_dup_f32(A0 + oa5), c05 = vmlaq_f32(c05, b0, a0), c15 = vmlaq_f32(c15, b1, a0);
                if (M > 0x6) a0 = vld1q_dup_f32(A6 + oa0), c06 = vmlaq_f32(c06, b0, a0), c16 = vmlaq_f32(c16, b1, a0);
                if (M > 0x7) a0 = vld1q_dup_f32(A6 + oa1), c07 = vmlaq_f32(c07, b0, a0), c17 = vmlaq_f32(c17, b1, a0);
                if (M > 0x8) a0 = vld1q_dup_f32(A6 + oa2), c08 = vmlaq_f32(c08, b0, a0), c18 = vmlaq_f32(c18, b1, a0);
                if (M > 0x9) a0 = vld1q_dup_f32(A6 + oa3), c09 = vmlaq_f32(c09, b0, a0), c19 = vmlaq_f32(c19, b1, a0);
                if (M > 0xA) a0 = vld1q_dup_f32(A6 + oa4), c0A = vmlaq_f32(c0A, b0, a0), c1A = vmlaq_f32(c1A, b1, a0);
                if (M > 0xB) a0 = vld1q_dup_f32(A6 + oa5), c0B = vmlaq_f32(c0B, b0, a0), c1B = vmlaq_f32(c1B, b1, a0);
                B += sb;
                A0 += sa;
                A6 += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            if (M > 0x0) AddProduct(C + 0 * F, _alpha, c00), AddProduct(C + 1 * F, _alpha, c10, tail), C += ldc;
            if (M > 0x1) AddProduct(C + 0 * F, _alpha, c01), AddProduct(C + 1 * F, _alpha, c11, tail), C += ldc;
            if (M > 0x2) AddProduct(C + 0 * F, _alpha, c02), AddProduct(C + 1 * F, _alpha, c12, tail), C += ldc;
            if (M > 0x3) AddProduct(C + 0 * F, _alpha, c03), AddProduct(C + 1 * F, _alpha, c13, tail), C += ldc;
            if (M > 0x4) AddProduct(C + 0 * F, _alpha, c04), AddProduct(C + 1 * F, _alpha, c14, tail), C += ldc;
            if (M > 0x5) AddProduct(C + 0 * F, _alpha, c05), AddProduct(C + 1 * F, _alpha, c15, tail), C += ldc;
            if (M > 0x6) AddProduct(C + 0 * F, _alpha, c06), AddProduct(C + 1 * F, _alpha, c16, tail), C += ldc;
            if (M > 0x7) AddProduct(C + 0 * F, _alpha, c07), AddProduct(C + 1 * F, _alpha, c17, tail), C += ldc;
            if (M > 0x8) AddProduct(C + 0 * F, _alpha, c08), AddProduct(C + 1 * F, _alpha, c18, tail), C += ldc;
            if (M > 0x9) AddProduct(C + 0 * F, _alpha, c09), AddProduct(C + 1 * F, _alpha, c19, tail), C += ldc;
            if (M > 0xA) AddProduct(C + 0 * F, _alpha, c0A), AddProduct(C + 1 * F, _alpha, c1A, tail), C += ldc;
            if (M > 0xB) AddProduct(C + 0 * F, _alpha, c0B), AddProduct(C + 1 * F, _alpha, c1B, tail), C += ldc;
        }

        template<int M> void GemmKernelMx4nnT(size_t, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, b0, a0;
            if (M > 0x0) c00 = vdupq_n_f32(0);
            if (M > 0x1) c01 = vdupq_n_f32(0);
            if (M > 0x2) c02 = vdupq_n_f32(0);
            if (M > 0x3) c03 = vdupq_n_f32(0);
            if (M > 0x4) c04 = vdupq_n_f32(0);
            if (M > 0x5) c05 = vdupq_n_f32(0);
            if (M > 0x6) c06 = vdupq_n_f32(0);
            if (M > 0x7) c07 = vdupq_n_f32(0);
            if (M > 0x8) c08 = vdupq_n_f32(0);
            if (M > 0x9) c09 = vdupq_n_f32(0);
            if (M > 0xA) c0A = vdupq_n_f32(0);
            if (M > 0xB) c0B = vdupq_n_f32(0);
            const float* A0 = A, * A6 = A + 6 * lda;
            size_t oa0, oa1, oa2, oa3, oa4, oa5;
            if (M > 0) oa0 = lda * 0;
            if (M > 1) oa1 = lda * 1;
            if (M > 2) oa2 = lda * 2;
            if (M > 3) oa3 = lda * 3;
            if (M > 4) oa4 = lda * 4;
            if (M > 5) oa5 = lda * 5;
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                if (M > 0x0) c00 = vmlaq_f32(c00, b0, vld1q_dup_f32(A0 + oa0));
                if (M > 0x1) c01 = vmlaq_f32(c01, b0, vld1q_dup_f32(A0 + oa1));
                if (M > 0x2) c02 = vmlaq_f32(c02, b0, vld1q_dup_f32(A0 + oa2));
                if (M > 0x3) c03 = vmlaq_f32(c03, b0, vld1q_dup_f32(A0 + oa3));
                if (M > 0x4) c04 = vmlaq_f32(c04, b0, vld1q_dup_f32(A0 + oa4));
                if (M > 0x5) c05 = vmlaq_f32(c05, b0, vld1q_dup_f32(A0 + oa5));
                if (M > 0x6) c06 = vmlaq_f32(c06, b0, vld1q_dup_f32(A6 + oa0));
                if (M > 0x7) c07 = vmlaq_f32(c07, b0, vld1q_dup_f32(A6 + oa1));
                if (M > 0x8) c08 = vmlaq_f32(c08, b0, vld1q_dup_f32(A6 + oa2));
                if (M > 0x9) c09 = vmlaq_f32(c09, b0, vld1q_dup_f32(A6 + oa3));
                if (M > 0xA) c0A = vmlaq_f32(c0A, b0, vld1q_dup_f32(A6 + oa4));
                if (M > 0xB) c0B = vmlaq_f32(c0B, b0, vld1q_dup_f32(A6 + oa5));
                B += sb;
                A0 += sa;
                A6 += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            if (M > 0x0) AddProduct(C, _alpha, c00, tail), C += ldc;
            if (M > 0x1) AddProduct(C, _alpha, c01, tail), C += ldc;
            if (M > 0x2) AddProduct(C, _alpha, c02, tail), C += ldc;
            if (M > 0x3) AddProduct(C, _alpha, c03, tail), C += ldc;
            if (M > 0x4) AddProduct(C, _alpha, c04, tail), C += ldc;
            if (M > 0x5) AddProduct(C, _alpha, c05, tail), C += ldc;
            if (M > 0x6) AddProduct(C, _alpha, c06, tail), C += ldc;
            if (M > 0x7) AddProduct(C, _alpha, c07, tail), C += ldc;
            if (M > 0x8) AddProduct(C, _alpha, c08, tail), C += ldc;
            if (M > 0x9) AddProduct(C, _alpha, c09, tail), C += ldc;
            if (M > 0xA) AddProduct(C, _alpha, c0A, tail), C += ldc;
            if (M > 0xB) AddProduct(C, _alpha, c0B, tail), C += ldc;
        }

        SIMD_INLINE Simd::GemmNN<float, F, size_t>::Tail GetGemmTail(size_t M, size_t N)
        {
            if (N <= 4)
            {
                switch (M)
                {
                case 0: return GemmKernelMx4nnT<0>;
                case 1: return GemmKernelMx4nnT<1>;
                case 2: return GemmKernelMx4nnT<2>;
                case 3: return GemmKernelMx4nnT<3>;
                case 4: return GemmKernelMx4nnT<4>;
                case 5: return GemmKernelMx4nnT<5>;
                case 6: return GemmKernelMx4nnT<6>;
                case 7: return GemmKernelMx4nnT<7>;
                case 8: return GemmKernelMx4nnT<8>;
                case 9: return GemmKernelMx4nnT<9>;
                case 10: return GemmKernelMx4nnT<10>;
                case 11: return GemmKernelMx4nnT<11>;
                case 12: return GemmKernelMx4nnT<12>;
                }
            }
            else if (N <= 8)
            {
                switch (M)
                {
                case 0: return GemmKernelMx8nnT<0>;
                case 1: return GemmKernelMx8nnT<1>;
                case 2: return GemmKernelMx8nnT<2>;
                case 3: return GemmKernelMx8nnT<3>;
                case 4: return GemmKernelMx8nnT<4>;
                case 5: return GemmKernelMx8nnT<5>;
                case 6: return GemmKernelMx8nnT<6>;
                case 7: return GemmKernelMx8nnT<7>;
                case 8: return GemmKernelMx8nnT<8>;
                case 9: return GemmKernelMx8nnT<9>;
                case 10: return GemmKernelMx8nnT<10>;
                case 11: return GemmKernelMx8nnT<11>;
                case 12: return GemmKernelMx8nnT<12>;
                }
            }
            else if (N <= 12)
            {
                switch (M)
                {
                case 0: return GemmKernelMx12nnT<0>;
                case 1: return GemmKernelMx12nnT<1>;
                case 2: return GemmKernelMx12nnT<2>;
                case 3: return GemmKernelMx12nnT<3>;
                case 4: return GemmKernelMx12nnT<4>;
                case 5: return GemmKernelMx12nnT<5>;
                case 6: return GemmKernelMx12nnT<6>;
                case 7: return GemmKernelMx12nnT<7>;
                case 8: return GemmKernelMx12nnT<8>;
                }
            }
            else if (N <= 16)
            {
                switch (M)
                {
                case 0: return GemmKernelMx16nnT<0>;
                case 1: return GemmKernelMx16nnT<1>;
                case 2: return GemmKernelMx16nnT<2>;
                case 3: return GemmKernelMx16nnT<3>;
                case 4: return GemmKernelMx16nnT<4>;
                case 5: return GemmKernelMx16nnT<5>;
                case 6: return GemmKernelMx16nnT<6>;
                }
            }
            assert(0);
            return NULL;
        }

        void Gemm32fNN(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            typedef Simd::GemmNN<float, F, size_t> GemmNN;
            GemmNN::Main kernelMM, kernelMT;
            GemmNN::Tail kernelTM, kernelTT;
            size_t microM, microN, L1, L2;
#if defined(SIMD_ARM64_ENABLE) 
            if (N == 8 || M == 12 || M * 8 < N)
            {
                microM = 12;
                microN = 8;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel12x8nn;
                kernelMT = tail > F ? GemmKernel12x8nn : GemmKernel12x4nn;
                kernelTM = GemmKernelMx8nn;
                kernelTT = tail > F ? GemmKernelMx8nn : GemmKernelMx4nn;
            }
            else if(N == 12 || N == 24 || M == 8 || M == 16)
            {
                microM = 8;
                microN = 12;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel8x12nn;
                kernelMT = tail > DF ? GemmKernel8x12nn : (tail > F ? GemmKernel8x8nn : GemmKernel8x4nn);
                kernelTM = GemmKernelMx12nn;
                kernelTT = tail > DF ? GemmKernelMx12nn : (tail > F ? GemmKernelMx8nn : GemmKernelMx4nn);
            }
            else
            {
                microM = 6;
                microN = 16;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel6x16nn;
                kernelMT = tail > 3 * F ? GemmKernel6x16nn : (tail > 2 * F ? GemmKernel6x12nn : (tail > F ? GemmKernel6x8nn : GemmKernel6x4nn));
                kernelTM = GemmKernelMx16nn;
                kernelTT = tail > 3 * F ? GemmKernelMx16nn : (tail > 2 * F ? GemmKernelMx12nn : (tail > F ? GemmKernelMx8nn : GemmKernelMx4nn));
            }
#else
            if (N != 12 && M != 4 && M != 8)
            {
                microM = 6;
                microN = 8;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel6x8nn;
                kernelMT = tail > F ? GemmKernel6x8nn : GemmKernel6x4nn;
                kernelTM = GemmKernelMx8nn;
                kernelTT = tail > F ? GemmKernelMx8nn : GemmKernelMx4nn;
            }
            else
            {
                microM = 4;
                microN = 12;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel4x12nn;
                kernelMT = tail > DF ? GemmKernel4x12nn : (tail > F ? GemmKernel4x8nn : GemmKernel4x4nn);
                kernelTM = GemmKernelMx12nn;
                kernelTT = tail > DF ? GemmKernelMx12nn : (tail > F ? GemmKernelMx8nn : GemmKernelMx4nn);
            }
#endif
            GemmNN::PackA packA = GemmPackA;
            L1 = N > 4096 ? Base::AlgCacheL2() : Base::AlgCacheL1();
            L2 = N > 4096 ? Base::AlgCacheL3() : Base::AlgCacheL2();
            GemmNN gemmNN(M, N, K, microM, microN, L1, L2, Base::AlgCacheL3(), 
                kernelMM, kernelMT, kernelTM, kernelTT, packA, GemmPackB, GemmScaleC, NULL);
            gemmNN.Run(alpha, A, lda, B, ldb, beta, C, ldc);
        }

        //---------------------------------------------------------------------

        typedef Simd::GemmNNcb<float, F, size_t> Gemm32fNNcb;

        SIMD_INLINE Gemm32fNNcb CreateGemm32fNNcb(size_t M, size_t N, size_t K, GemmKernelType type, bool compatibility)
        {
            Gemm32fNNcb::Main kernelMM, kernelMT;
            Gemm32fNNcb::Tail kernelTM, kernelTT;
            size_t microM, microN;
#if defined(SIMD_ARM64_ENABLE)
            if (type == GemmKernelF4 || (type == GemmKernelAny && (M != 4 && M != 8 && M != 16) && N > 12))
            {
                microN = 16;
                size_t tail = N - AlignLoAny(N, microN);
                microM = 6;
                kernelMM = Neon::GemmKernel6x16nn;
                kernelMT = tail > 3 * F ? Neon::GemmKernel6x16nn : (tail > DF ? Neon::GemmKernel6x12nn : (tail > F ? Neon::GemmKernel6x8nn : Neon::GemmKernel6x4nn));
                kernelTM = Neon::GetGemmTail(M % microM, microN);
                kernelTT = Neon::GetGemmTail(M % microM, tail);
                type = GemmKernelF4;
            }
            if (type == GemmKernelF3 || (type == GemmKernelAny && (M == 4 || M == 8 || M == 16) && N > 8))
            {
                microN = 12;
                size_t tail = N - AlignLoAny(N, microN);
                if (M == 4)
                {
                    microM = 4;
                    kernelMM = Neon::GemmKernel4x12nn;
                    kernelMT = tail > DF ? Neon::GemmKernel4x12nn : (tail > F ? Neon::GemmKernel4x8nn : Neon::GemmKernel4x4nn);
                }
                else
                {
                    microM = 8;
                    kernelMM = Neon::GemmKernel8x12nn;
                    kernelMT = tail > DF ? Neon::GemmKernel8x12nn : (tail > F ? Neon::GemmKernel8x8nn : Neon::GemmKernel8x4nn);
                }
                kernelTM = Neon::GetGemmTail(M % microM, microN);
                kernelTT = Neon::GetGemmTail(M % microM, tail);
                type = GemmKernelF3;
            }
            if (type == GemmKernelF2 || (type == GemmKernelF3 && N <= 8) || (type == GemmKernelAny && N > 4))
            {
                microN = 8;
                size_t tail = N - AlignLoAny(N, microN);
                if (M == 6)
                {
                    microM = 6;
                    kernelMM = Neon::GemmKernel6x8nn;
                    kernelMT = tail > F ? Neon::GemmKernel6x8nn : Neon::GemmKernel6x4nn;
                }
                else
                {
                    microM = 12;
                    kernelMM = Neon::GemmKernel12x8nn;
                    kernelMT = tail > F ? Neon::GemmKernel12x8nn : Neon::GemmKernel12x4nn;
                }
                kernelTM = Neon::GetGemmTail(M % microM, microN);
                kernelTT = Neon::GetGemmTail(M % microM, tail);
                type = GemmKernelF2;
            }
            if (type == GemmKernelF1 || (type == GemmKernelF2 && N <= 4) || type == GemmKernelAny)
            {
                microM = 12;
                microN = 4;
                kernelMM = Neon::GemmKernel12x4nn;
                kernelMT = Neon::GemmKernel12x4nn;
                kernelTM = Neon::GetGemmTail(M % microM, microN);
                kernelTT = Neon::GetGemmTail(M % microM, microN);
                type = GemmKernelF1;
            }
#else
            if (type == GemmKernelF3 || (type == GemmKernelAny && (M == 4 || M == 8 || M == 16) && N > 8))
            {
                microM = 4;
                microN = 12;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Neon::GemmKernel4x12nn;
                kernelMT = tail > DF ? Neon::GemmKernel4x12nn : (tail > F ? Neon::GemmKernel4x8nn : Neon::GemmKernel4x4nn);
                kernelTM = Neon::GetGemmTail(M%microM, microN);
                kernelTT = Neon::GetGemmTail(M%microM, tail);
                type = GemmKernelF3;
            }
            if (type == GemmKernelF2 || (type == GemmKernelF3 && N <= 8) || (type == GemmKernelAny && N > 4))
            {
                microM = 6;
                microN = 8;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Neon::GemmKernel6x8nn;
                kernelMT = tail > F ? Neon::GemmKernel6x8nn : Neon::GemmKernel6x4nn;
                kernelTM = Neon::GetGemmTail(M%microM, microN);
                kernelTT = Neon::GetGemmTail(M%microM, tail);
                type = GemmKernelF2;
            }
            if (type == GemmKernelF1 || (type == GemmKernelF2 && N <= 4) || type == GemmKernelAny)
            {
                microM = 6;
                microN = 4;
                kernelMM = Neon::GemmKernel6x4nn;
                kernelMT = Neon::GemmKernel6x4nn;
                kernelTM = Neon::GetGemmTail(M%microM, microN);
                kernelTT = Neon::GetGemmTail(M%microM, microN);
                type = GemmKernelF1;
            }
#endif
            return Gemm32fNNcb(M, N, K, microM, microN, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), 
                kernelMM, kernelMT, kernelTM, kernelTT, Neon::GemmPackA, Neon::GemmPackB, Neon::GemmScaleC, NULL, compatibility);
        }

        size_t Gemm32fNNcbBufferSize(size_t M, size_t N, size_t K, GemmKernelType type, bool compatibility)
        {
            Gemm32fNNcb gemm = CreateGemm32fNNcb(M, N, K, type, compatibility);
            return gemm.BufferSize();
        }

        void Gemm32fNNcbReorderB(size_t M, size_t N, size_t K, const float * B, float * pB, GemmKernelType type, bool compatibility)
        {
            Gemm32fNNcb gemm = CreateGemm32fNNcb(M, N, K, type, compatibility);
            gemm.ReorderB(B, N, pB);
        }

        void Gemm32fNNcbRun(size_t M, size_t N, size_t K, const float * A, const float * pB, float * C, GemmKernelType type, bool compatibility)
        {
            Gemm32fNNcb gemm = CreateGemm32fNNcb(M, N, K, type, compatibility);
            gemm.Run(A, K, pB, C, N);
        }
    }
#endif// SIMD_NEON_ENABLE
}
