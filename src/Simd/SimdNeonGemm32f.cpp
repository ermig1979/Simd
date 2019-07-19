/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
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
                a0 = vdupq_n_f32(A[oa0]);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                c02 = vmlaq_f32(c02, a0, b2);
                a0 = vdupq_n_f32(A[oa1]);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                c12 = vmlaq_f32(c12, a0, b2);
                a0 = vdupq_n_f32(A[oa2]);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                c22 = vmlaq_f32(c22, a0, b2);
                a0 = vdupq_n_f32(A[oa3]);
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
                a0 = vdupq_n_f32(A[oa0]);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                a0 = vdupq_n_f32(A[oa1]);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                a0 = vdupq_n_f32(A[oa2]);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                a0 = vdupq_n_f32(A[oa3]);
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
                c0 = vmlaq_f32(c0, b0, vdupq_n_f32(A[oa0]));
                c1 = vmlaq_f32(c1, b0, vdupq_n_f32(A[oa1]));
                c2 = vmlaq_f32(c2, b0, vdupq_n_f32(A[oa2]));
                c3 = vmlaq_f32(c3, b0, vdupq_n_f32(A[oa3]));
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * ldc, _alpha, c0, tail);
            AddProduct(C + 1 * ldc, _alpha, c1, tail);
            AddProduct(C + 2 * ldc, _alpha, c2, tail);
            AddProduct(C + 3 * ldc, _alpha, c3, tail);
        }

        void GemmKernel6x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
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
                a0 = vdupq_n_f32(A[oa0]);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                a0 = vdupq_n_f32(A[oa1]);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                a0 = vdupq_n_f32(A[oa2]);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                a0 = vdupq_n_f32(A[oa3]);
                c30 = vmlaq_f32(c30, a0, b0);
                c31 = vmlaq_f32(c31, a0, b1);
                a0 = vdupq_n_f32(A[oa4]);
                c40 = vmlaq_f32(c40, a0, b0);
                c41 = vmlaq_f32(c41, a0, b1);
                a0 = vdupq_n_f32(A[oa5]);
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
                c0 = vmlaq_f32(c0, b0, vdupq_n_f32(A[oa0]));
                c1 = vmlaq_f32(c1, b0, vdupq_n_f32(A[oa1]));
                c2 = vmlaq_f32(c2, b0, vdupq_n_f32(A[oa2]));
                c3 = vmlaq_f32(c3, b0, vdupq_n_f32(A[oa3]));
                c4 = vmlaq_f32(c4, b0, vdupq_n_f32(A[oa4]));
                c5 = vmlaq_f32(c5, b0, vdupq_n_f32(A[oa5]));
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

        void GemmKernelMx12nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c[4][3];
            size_t oa[4];
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
                    a0 = vdupq_n_f32(A[oa[i]]);
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
            float32x4_t c[6][2];
            size_t oa[6];
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
                    a0 = vdupq_n_f32(A[oa[i]]);
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
            float32x4_t c[6];
            size_t oa[6];
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
                    a0 = vdupq_n_f32(A[oa[i]]);
                    c[i] = vmlaq_f32(c[i], b0, a0);
                }
                B += ldb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            for (size_t i = 0; i < M; ++i)
                AddProduct(C + i * ldc, _alpha, c[i], tail);
        }

        template<int M> void GemmKernelMx12nnT(size_t, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, b0, b1, b2, a0;
            if (M > 0) c00 = vdupq_n_f32(0.0), c10 = vdupq_n_f32(0.0), c20 = vdupq_n_f32(0.0);
            if (M > 1) c01 = vdupq_n_f32(0.0), c11 = vdupq_n_f32(0.0), c21 = vdupq_n_f32(0.0);
            if (M > 2) c02 = vdupq_n_f32(0.0), c12 = vdupq_n_f32(0.0), c22 = vdupq_n_f32(0.0);
            if (M > 3) c03 = vdupq_n_f32(0.0), c13 = vdupq_n_f32(0.0), c23 = vdupq_n_f32(0.0);
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
                if (M > 0) a0 = vdupq_n_f32(A[oa0]), c00 = vmlaq_f32(c00, b0, a0), c10 = vmlaq_f32(c10, b1, a0), c20 = vmlaq_f32(c20, b2, a0);
                if (M > 1) a0 = vdupq_n_f32(A[oa1]), c01 = vmlaq_f32(c01, b0, a0), c11 = vmlaq_f32(c11, b1, a0), c21 = vmlaq_f32(c21, b2, a0);
                if (M > 2) a0 = vdupq_n_f32(A[oa2]), c02 = vmlaq_f32(c02, b0, a0), c12 = vmlaq_f32(c12, b1, a0), c22 = vmlaq_f32(c22, b2, a0);
                if (M > 3) a0 = vdupq_n_f32(A[oa3]), c03 = vmlaq_f32(c03, b0, a0), c13 = vmlaq_f32(c13, b1, a0), c23 = vmlaq_f32(c23, b2, a0);
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            if (M > 0) AddProduct(C + 0 * F, _alpha, c00), AddProduct(C + 1 * F, _alpha, c10), AddProduct(C + 2 * F, _alpha, c20, tail), C += ldc;
            if (M > 1) AddProduct(C + 0 * F, _alpha, c01), AddProduct(C + 1 * F, _alpha, c11), AddProduct(C + 2 * F, _alpha, c21, tail), C += ldc;
            if (M > 2) AddProduct(C + 0 * F, _alpha, c02), AddProduct(C + 1 * F, _alpha, c12), AddProduct(C + 2 * F, _alpha, c22, tail), C += ldc;
            if (M > 3) AddProduct(C + 0 * F, _alpha, c03), AddProduct(C + 1 * F, _alpha, c13), AddProduct(C + 2 * F, _alpha, c23, tail), C += ldc;
        }

        template<int M> void GemmKernelMx8nnT(size_t, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c00, c01, c02, c03, c04, c05, c10, c11, c12, c13, c14, c15, b0, b1, a0;
            if (M > 0) c00 = vdupq_n_f32(0.0), c10 = vdupq_n_f32(0.0);
            if (M > 1) c01 = vdupq_n_f32(0.0), c11 = vdupq_n_f32(0.0);
            if (M > 2) c02 = vdupq_n_f32(0.0), c12 = vdupq_n_f32(0.0);
            if (M > 3) c03 = vdupq_n_f32(0.0), c13 = vdupq_n_f32(0.0);
            if (M > 4) c04 = vdupq_n_f32(0.0), c14 = vdupq_n_f32(0.0);
            if (M > 5) c05 = vdupq_n_f32(0.0), c15 = vdupq_n_f32(0.0);
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
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + ob0);
                b1 = Load<false>(B + ob1);
                if (M > 0) a0 = vdupq_n_f32(A[oa0]), c00 = vmlaq_f32(c00, b0, a0), c10 = vmlaq_f32(c10, b1, a0);
                if (M > 1) a0 = vdupq_n_f32(A[oa1]), c01 = vmlaq_f32(c01, b0, a0), c11 = vmlaq_f32(c11, b1, a0);
                if (M > 2) a0 = vdupq_n_f32(A[oa2]), c02 = vmlaq_f32(c02, b0, a0), c12 = vmlaq_f32(c12, b1, a0);
                if (M > 3) a0 = vdupq_n_f32(A[oa3]), c03 = vmlaq_f32(c03, b0, a0), c13 = vmlaq_f32(c13, b1, a0);
                if (M > 4) a0 = vdupq_n_f32(A[oa4]), c04 = vmlaq_f32(c04, b0, a0), c14 = vmlaq_f32(c14, b1, a0);
                if (M > 5) a0 = vdupq_n_f32(A[oa5]), c05 = vmlaq_f32(c00, b0, a0), c15 = vmlaq_f32(c15, b1, a0);
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            if (M > 0) AddProduct(C + 0 * F, _alpha, c00), AddProduct(C + 1 * F, _alpha, c10, tail), C += ldc;
            if (M > 1) AddProduct(C + 0 * F, _alpha, c01), AddProduct(C + 1 * F, _alpha, c11, tail), C += ldc;
            if (M > 2) AddProduct(C + 0 * F, _alpha, c02), AddProduct(C + 1 * F, _alpha, c12, tail), C += ldc;
            if (M > 3) AddProduct(C + 0 * F, _alpha, c03), AddProduct(C + 1 * F, _alpha, c13, tail), C += ldc;
            if (M > 4) AddProduct(C + 0 * F, _alpha, c04), AddProduct(C + 1 * F, _alpha, c14, tail), C += ldc;
            if (M > 5) AddProduct(C + 0 * F, _alpha, c05), AddProduct(C + 1 * F, _alpha, c15, tail), C += ldc;
        }

        template<int M> void GemmKernelMx4nnT(size_t, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c00, c01, c02, c03, c04, c05, b0;
            if (M > 0) c00 = vdupq_n_f32(0.0);
            if (M > 1) c01 = vdupq_n_f32(0.0);
            if (M > 2) c02 = vdupq_n_f32(0.0);
            if (M > 3) c03 = vdupq_n_f32(0.0);
            if (M > 4) c04 = vdupq_n_f32(0.0);
            if (M > 5) c05 = vdupq_n_f32(0.0);
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
                if (M > 0) c00 = vmlaq_f32(c00, b0, vdupq_n_f32(A[oa0]));
                if (M > 1) c01 = vmlaq_f32(c01, b0, vdupq_n_f32(A[oa1]));
                if (M > 2) c02 = vmlaq_f32(c02, b0, vdupq_n_f32(A[oa2]));
                if (M > 3) c03 = vmlaq_f32(c03, b0, vdupq_n_f32(A[oa3]));
                if (M > 4) c04 = vmlaq_f32(c04, b0, vdupq_n_f32(A[oa4]));
                if (M > 5) c05 = vmlaq_f32(c05, b0, vdupq_n_f32(A[oa5]));
                B += sb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            if (M > 0) AddProduct(C + 0 * ldc, _alpha, c00, tail);
            if (M > 1) AddProduct(C + 1 * ldc, _alpha, c01, tail);
            if (M > 2) AddProduct(C + 2 * ldc, _alpha, c02, tail);
            if (M > 3) AddProduct(C + 3 * ldc, _alpha, c03, tail);
            if (M > 4) AddProduct(C + 4 * ldc, _alpha, c04, tail);
            if (M > 5) AddProduct(C + 5 * ldc, _alpha, c05, tail);
        }

        SIMD_INLINE Simd::GemmNN<float, size_t>::Tail GetGemmTail(size_t M, size_t N)
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
                }
            }
            assert(0);
            return NULL;
        }

        void GemmPackA(const float * src, size_t stride, size_t M, size_t K, size_t cell, float * dst)
        {
            for (size_t i = 0; i < M; i += cell)
            {
                size_t m = Simd::Min(cell, M - i), k = 0;
                if (cell == 6 && m == 6)
                {
                    size_t K4 = AlignLo(K, 4);
                    for (; k < K4; k += 4)
                    {
                        const float * ps = src + k;
                        float32x4_t src0 = Load<false>(ps + 0 * stride);
                        float32x4_t src1 = Load<false>(ps + 1 * stride);
                        float32x4_t src2 = Load<false>(ps + 2 * stride);
                        float32x4_t src3 = Load<false>(ps + 3 * stride);
                        float32x4_t src4 = Load<false>(ps + 4 * stride);
                        float32x4_t src5 = Load<false>(ps + 5 * stride);
                        float32x4x2_t src03 = vzipq_f32(src0, src3);
                        float32x4x2_t src14 = vzipq_f32(src1, src4);
                        float32x4x2_t src25 = vzipq_f32(src2, src5);
                        float32x4x3_t dst0;
                        dst0.val[0] = src03.val[0];
                        dst0.val[1] = src14.val[0];
                        dst0.val[2] = src25.val[0];
                        Store3<false>(dst, dst0);
                        float32x4x3_t dst1;
                        dst1.val[0] = src03.val[1];
                        dst1.val[1] = src14.val[1];
                        dst1.val[2] = src25.val[1];
                        Store3<false>(dst + 12, dst1);
                        dst += 24;
                    }
                }
                if (cell == 4 && m == 4)
                {
                    size_t K4 = AlignLo(K, 4);
                    for (; k < K4; k += 4)
                    {
                        const float * ps = src + k;
                        float32x4x4_t _dst;
                        _dst.val[0] = Load<false>(ps + 0 * stride);
                        _dst.val[1] = Load<false>(ps + 1 * stride);
                        _dst.val[2] = Load<false>(ps + 2 * stride);
                        _dst.val[3] = Load<false>(ps + 3 * stride);
                        Store4<false>(dst, _dst);
                        dst += 16;
                    }
                }
                for (; k < K; ++k)
                {
                    for (size_t c = 0; c < m; ++c)
                        *(dst++) = src[c*stride + k];
                }
                src += cell * stride;
            }
        }

        void GemmPackB(const float * B, size_t ldb, size_t K, size_t N, size_t microN, float * pB)
        {
            for (size_t j = 0; j < N; j += microN)
            {
                size_t n = Simd::Min(microN, N - j);
                size_t k = 0;
                if (microN == 1 * F)
                {
                    if (n == microN)
                    {
                        for (; k < K; ++k)
                        {
                            const float * b = B + k * ldb;
                            Store<false>(pB + 0 * F, Load<false>(b + 0 * F));
                            pB += microN;
                        }
                    }
                    else
                    {
                        float32x4_t mask0 = LeftNotZero(n - 0 * F);
                        for (; k < K - 1; ++k)
                        {
                            const float * b = B + k * ldb;
                            Store<false>(pB + 0 * F, And(mask0, Load<false>(b + 0 * F)));
                            pB += microN;
                        }
                    }
                }
                else if (microN == 2 * F)
                {
                    if (n == microN)
                    {
                        for (; k < K; ++k)
                        {
                            const float * b = B + k * ldb;
                            Store<false>(pB + 0 * F, Load<false>(b + 0 * F));
                            Store<false>(pB + 1 * F, Load<false>(b + 1 * F));
                            pB += microN;
                        }
                    }
                    else
                    {
                        float32x4_t mask0 = LeftNotZero(n - 0 * F);
                        float32x4_t mask1 = LeftNotZero(n - 1 * F);
                        for (; k < K - 1; ++k)
                        {
                            const float * b = B + k * ldb;
                            Store<false>(pB + 0 * F, And(mask0, Load<false>(b + 0 * F)));
                            Store<false>(pB + 1 * F, And(mask1, Load<false>(b + 1 * F)));
                            pB += microN;
                        }
                    }
                }
                else if (microN == 3 * F)
                {
                    if (n == microN)
                    {
                        for (; k < K; ++k)
                        {
                            const float * b = B + k * ldb;
                            Store<false>(pB + 0 * F, Load<false>(b + 0 * F));
                            Store<false>(pB + 1 * F, Load<false>(b + 1 * F));
                            Store<false>(pB + 2 * F, Load<false>(b + 2 * F));
                            pB += microN;
                        }
                    }
                    else
                    {
                        float32x4_t mask0 = LeftNotZero(n - 0 * F);
                        float32x4_t mask1 = LeftNotZero(n - 1 * F);
                        float32x4_t mask2 = LeftNotZero(n - 2 * F);
                        for (; k < K - 1; ++k)
                        {
                            const float * b = B + k * ldb;
                            Store<false>(pB + 0 * F, And(mask0, Load<false>(b + 0 * F)));
                            Store<false>(pB + 1 * F, And(mask1, Load<false>(b + 1 * F)));
                            Store<false>(pB + 2 * F, And(mask2, Load<false>(b + 2 * F)));
                            pB += microN;
}
                    }
                }
                for (; k < K; ++k)
                {
                    const float * b = B + k * ldb;
                    size_t c = 0;
                    for (; c < n; ++c)
                        *(pB++) = *(b++);
                    for (; c < microN; ++c)
                        *(pB++) = 0;
                }
                B += microN;
            }
        }

        SIMD_INLINE void ScaleC(float * C, float32x4_t beta)
        {
            Store<false>(C, vmulq_f32(Load<false>(C), beta));
        }

        void GemmScaleC(size_t M, size_t N, float beta, float * C, size_t ldc)
        {
            if (beta == 1.0f)
                return;
            else if (beta == 0.0f)
            {
                for (size_t i = 0; i < M; ++i)
                    memset(C + i * ldc, 0, N * sizeof(float));
            }
            else
            {
                size_t NQF = AlignLo(N, QF);
                size_t NF = AlignLo(N, F);
                float32x4_t _beta = vdupq_n_f32(beta);
                for (size_t i = 0; i < M; ++i)
                {
                    size_t j = 0;
                    for (; j < NQF; j += QF)
                    {
                        ScaleC(C + j + F * 0, _beta);
                        ScaleC(C + j + F * 1, _beta);
                        ScaleC(C + j + F * 2, _beta);
                        ScaleC(C + j + F * 3, _beta);
                    }
                    for (; j < NF; j += F)
                        ScaleC(C + j, _beta);
                    for (; j < N; ++j)
                        C[j] *= beta;
                    C += ldc;
                }
            }
        }

        void Gemm32fNN(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            const size_t CACHE_L1_SIZE = 16 * 1024;
            const size_t CACHE_L2_SIZE = 512 * 1024;
            const size_t CACHE_L3_SIZE = 2 * 1024 * 1024;
            typedef Simd::GemmNN<float, size_t> GemmNN;
            GemmNN::Main kernelMM, kernelMT;
            GemmNN::Tail kernelTM, kernelTT;
            size_t microM, microN, L1, L2;
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
            GemmNN::PackA packA = GemmPackA;
            L1 = N > 4096 ? CACHE_L2_SIZE : CACHE_L1_SIZE;
            L2 = N > 4096 ? CACHE_L3_SIZE : CACHE_L2_SIZE;
            GemmNN gemmNN(M, N, K, microM, microN, L1, L2, CACHE_L3_SIZE, F,
                kernelMM, kernelMT, kernelTM, kernelTT, packA, GemmPackB, GemmScaleC, NULL);
            gemmNN.Run(alpha, A, lda, B, ldb, beta, C, ldc);
        }

        //---------------------------------------------------------------------

        typedef Simd::GemmNNcb<float, size_t> Gemm32fNNcb;

        SIMD_INLINE Gemm32fNNcb CreateGemm32fNNcb(size_t M, size_t N, size_t K, GemmKernelType type, bool compatibility)
        {
            const size_t L1 = 32 * 1024;
            const size_t L2 = 256 * 1024;
            const size_t L3 = 2 * 1024 * 1024;
            Gemm32fNNcb::Main kernelMM, kernelMT;
            Gemm32fNNcb::Tail kernelTM, kernelTT;
            size_t microM, microN;
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
            return Gemm32fNNcb(M, N, K, microM, microN, L1, L2, L3, F, kernelMM, kernelMT, kernelTM, kernelTT, Neon::GemmPackB, Neon::GemmScaleC, NULL, compatibility);
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

        //---------------------------------------------------------------------

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

        void Gemm32fNT(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            const size_t CACHE_L1_SIZE = 16 * 1024;
            const size_t CACHE_L2_SIZE = 512 * 1024;
            const size_t CACHE_L3_SIZE = 2 * 1024 * 1024;
            typedef Simd::GemmNT<float> GemmNT;
            GemmNT gemmNT(M, N, K, CACHE_L1_SIZE, CACHE_L2_SIZE, CACHE_L3_SIZE, F, GemmScaleC,
                Kernel1x1x4nt, Kernel1x4x4nt, Kernel2x1x4nt, Kernel2x4x4nt, Kernel3x1x4nt, Kernel3x4x4nt, NULL, NULL);
            gemmNT.Run(alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
#endif// SIMD_NEON_ENABLE
}
