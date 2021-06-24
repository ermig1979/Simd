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
#include "Simd/SimdGemm.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        SIMD_INLINE void AddProduct(float * ptr, __m128 value, __m128 alpha)
        {
            _mm_storeu_ps(ptr, _mm_add_ps(_mm_mul_ps(value, alpha), _mm_loadu_ps(ptr)));
        }

        SIMD_INLINE void AddProduct(float * ptr, __m128 value, __m128 alpha, size_t tail)
        {
            if (tail == F)
                AddProduct(ptr, value, alpha);
            else
            {
                float tmp[F];
                _mm_storeu_ps(tmp, _mm_add_ps(_mm_mul_ps(value, alpha), _mm_loadu_ps(ptr)));
                for (size_t i = 0; i < tail; ++i)
                    ptr[i] = tmp[i];
            }
        }

        void GemmKernel4x12nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m128 c00 = _mm_setzero_ps();
            __m128 c10 = _mm_setzero_ps();
            __m128 c20 = _mm_setzero_ps();
            __m128 c30 = _mm_setzero_ps();
            __m128 c01 = _mm_setzero_ps();
            __m128 c11 = _mm_setzero_ps();
            __m128 c21 = _mm_setzero_ps();
            __m128 c31 = _mm_setzero_ps();
            __m128 c02 = _mm_setzero_ps();
            __m128 c12 = _mm_setzero_ps();
            __m128 c22 = _mm_setzero_ps();
            __m128 c32 = _mm_setzero_ps();
            __m128 b0, b1, b2, a0;
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
                b0 = _mm_loadu_ps(B + ob0);
                b1 = _mm_loadu_ps(B + ob1);
                b2 = _mm_loadu_ps(B + ob2);
                a0 = _mm_set1_ps(A[oa0]);
                c00 = _mm_add_ps(_mm_mul_ps(a0, b0), c00);
                c01 = _mm_add_ps(_mm_mul_ps(a0, b1), c01);
                c02 = _mm_add_ps(_mm_mul_ps(a0, b2), c02);
                a0 = _mm_set1_ps(A[oa1]);
                c10 = _mm_add_ps(_mm_mul_ps(a0, b0), c10);
                c11 = _mm_add_ps(_mm_mul_ps(a0, b1), c11);
                c12 = _mm_add_ps(_mm_mul_ps(a0, b2), c12);
                a0 = _mm_set1_ps(A[oa2]);
                c20 = _mm_add_ps(_mm_mul_ps(a0, b0), c20);
                c21 = _mm_add_ps(_mm_mul_ps(a0, b1), c21);
                c22 = _mm_add_ps(_mm_mul_ps(a0, b2), c22);
                a0 = _mm_set1_ps(A[oa3]);
                c30 = _mm_add_ps(_mm_mul_ps(a0, b0), c30);
                c31 = _mm_add_ps(_mm_mul_ps(a0, b1), c31);
                c32 = _mm_add_ps(_mm_mul_ps(a0, b2), c32);
                B += sb;
                A += sa;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
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
            __m128 c00 = _mm_setzero_ps();
            __m128 c10 = _mm_setzero_ps();
            __m128 c20 = _mm_setzero_ps();
            __m128 c30 = _mm_setzero_ps();
            __m128 c01 = _mm_setzero_ps();
            __m128 c11 = _mm_setzero_ps();
            __m128 c21 = _mm_setzero_ps();
            __m128 c31 = _mm_setzero_ps();
            __m128 b0, b1, a0;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + ob0);
                b1 = _mm_loadu_ps(B + ob1);
                a0 = _mm_set1_ps(A[oa0]);
                c00 = _mm_add_ps(_mm_mul_ps(a0, b0), c00);
                c01 = _mm_add_ps(_mm_mul_ps(a0, b1), c01);
                a0 = _mm_set1_ps(A[oa1]);
                c10 = _mm_add_ps(_mm_mul_ps(a0, b0), c10);
                c11 = _mm_add_ps(_mm_mul_ps(a0, b1), c11);
                a0 = _mm_set1_ps(A[oa2]);
                c20 = _mm_add_ps(_mm_mul_ps(a0, b0), c20);
                c21 = _mm_add_ps(_mm_mul_ps(a0, b1), c21);
                a0 = _mm_set1_ps(A[oa3]);
                c30 = _mm_add_ps(_mm_mul_ps(a0, b0), c30);
                c31 = _mm_add_ps(_mm_mul_ps(a0, b1), c31);
                B += sb;
                A += sa;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
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
            __m128 c0 = _mm_setzero_ps();
            __m128 c1 = _mm_setzero_ps();
            __m128 c2 = _mm_setzero_ps();
            __m128 c3 = _mm_setzero_ps();
            __m128 b0;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            const size_t ob0 = ldb * 0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + ob0);
                c0 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa0])), c0);
                c1 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa1])), c1);
                c2 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa2])), c2);
                c3 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa3])), c3);
                B += sb;
                A += sa;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            AddProduct(C + 0 * ldc, _alpha, c0, tail);
            AddProduct(C + 1 * ldc, _alpha, c1, tail);
            AddProduct(C + 2 * ldc, _alpha, c2, tail);
            AddProduct(C + 3 * ldc, _alpha, c3, tail);
        }

        void GemmKernel6x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m128 c00 = _mm_setzero_ps();
            __m128 c10 = _mm_setzero_ps();
            __m128 c20 = _mm_setzero_ps();
            __m128 c30 = _mm_setzero_ps();
            __m128 c40 = _mm_setzero_ps();
            __m128 c50 = _mm_setzero_ps();
            __m128 c01 = _mm_setzero_ps();
            __m128 c11 = _mm_setzero_ps();
            __m128 c21 = _mm_setzero_ps();
            __m128 c31 = _mm_setzero_ps();
            __m128 c41 = _mm_setzero_ps();
            __m128 c51 = _mm_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            __m128 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + ob0);
                b1 = _mm_loadu_ps(B + ob1);
                a0 = _mm_set1_ps(A[oa0]);
                c00 = _mm_add_ps(_mm_mul_ps(a0, b0), c00);
                c01 = _mm_add_ps(_mm_mul_ps(a0, b1), c01);
                a0 = _mm_set1_ps(A[oa1]);
                c10 = _mm_add_ps(_mm_mul_ps(a0, b0), c10);
                c11 = _mm_add_ps(_mm_mul_ps(a0, b1), c11);
                a0 = _mm_set1_ps(A[oa2]);
                c20 = _mm_add_ps(_mm_mul_ps(a0, b0), c20);
                c21 = _mm_add_ps(_mm_mul_ps(a0, b1), c21);
                a0 = _mm_set1_ps(A[oa3]);
                c30 = _mm_add_ps(_mm_mul_ps(a0, b0), c30);
                c31 = _mm_add_ps(_mm_mul_ps(a0, b1), c31);
                a0 = _mm_set1_ps(A[oa4]);
                c40 = _mm_add_ps(_mm_mul_ps(a0, b0), c40);
                c41 = _mm_add_ps(_mm_mul_ps(a0, b1), c41);
                a0 = _mm_set1_ps(A[oa5]);
                c50 = _mm_add_ps(_mm_mul_ps(a0, b0), c50);
                c51 = _mm_add_ps(_mm_mul_ps(a0, b1), c51);
                B += sb;
                A += sa;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
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
            __m128 c0 = _mm_setzero_ps();
            __m128 c1 = _mm_setzero_ps();
            __m128 c2 = _mm_setzero_ps();
            __m128 c3 = _mm_setzero_ps();
            __m128 c4 = _mm_setzero_ps();
            __m128 c5 = _mm_setzero_ps();
            __m128 b0;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            const size_t ob0 = ldb * 0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + ob0);
                c0 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa0])), c0);
                c1 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa1])), c1);
                c2 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa2])), c2);
                c3 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa3])), c3);
                c4 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa4])), c4);
                c5 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa5])), c5);
                B += sb;
                A += sa;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            AddProduct(C + 0 * ldc, _alpha, c0, tail);
            AddProduct(C + 1 * ldc, _alpha, c1, tail);
            AddProduct(C + 2 * ldc, _alpha, c2, tail);
            AddProduct(C + 3 * ldc, _alpha, c3, tail);
            AddProduct(C + 4 * ldc, _alpha, c4, tail);
            AddProduct(C + 5 * ldc, _alpha, c5, tail);
        }

        void GemmKernelMx12nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m128 c[4][3];
            __m128 b0, b1, b2, a0;
            size_t oa[4];
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = _mm_setzero_ps();
                c[i][1] = _mm_setzero_ps();
                c[i][2] = _mm_setzero_ps();
                oa[i] = lda * i;
            }
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + ob0);
                b1 = _mm_loadu_ps(B + ob1);
                b2 = _mm_loadu_ps(B + ob2);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = _mm_set1_ps(A[oa[i]]);
                    c[i][0] = _mm_add_ps(_mm_mul_ps(b0, a0), c[i][0]);
                    c[i][1] = _mm_add_ps(_mm_mul_ps(b1, a0), c[i][1]);
                    c[i][2] = _mm_add_ps(_mm_mul_ps(b2, a0), c[i][2]);
                }
                B += sb;
                A += sa;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
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
            __m128 c[6][2];
            __m128 b0, b1, a0;
            size_t oa[6];
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t sa = lda == 1 ? M : 1;
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = _mm_setzero_ps();
                c[i][1] = _mm_setzero_ps();
                oa[i] = lda * i;
            }
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + ob0);
                b1 = _mm_loadu_ps(B + ob1);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = _mm_set1_ps(A[oa[i]]);
                    c[i][0] = _mm_add_ps(_mm_mul_ps(b0, a0), c[i][0]);
                    c[i][1] = _mm_add_ps(_mm_mul_ps(b1, a0), c[i][1]);
                }
                B += sb;
                A += sa;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            for (size_t i = 0; i < M; ++i)
            {
                AddProduct(C + 0 * F, _alpha, c[i][0]);
                AddProduct(C + 1 * F, _alpha, c[i][1], tail);
                C += ldc;
            }
        }

        void GemmKernelMx4nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
#ifdef SIMD_X64_ENABLE
            __m128 c[6];
            size_t oa[6];
#else
            __m128 c[4];
            size_t oa[4];
#endif
            __m128 b0, a0;
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            for (size_t i = 0; i < M; ++i)
            {
                c[i] = _mm_setzero_ps();
                oa[i] = lda * i;
            }
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + ob0);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = _mm_set1_ps(A[oa[i]]);
                    c[i] = _mm_add_ps(_mm_mul_ps(b0, a0), c[i]);
                }
                B += sb;
                A += sa;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            for (size_t i = 0; i < M; ++i)
                AddProduct(C + i * ldc, _alpha, c[i], tail);
        }

        template<int M> void GemmKernelMx12nnT(size_t, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m128 c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, b0, b1, b2, a0;
            if (M > 0) c00 = _mm_setzero_ps(), c10 = _mm_setzero_ps(), c20 = _mm_setzero_ps();
            if (M > 1) c01 = _mm_setzero_ps(), c11 = _mm_setzero_ps(), c21 = _mm_setzero_ps();
            if (M > 2) c02 = _mm_setzero_ps(), c12 = _mm_setzero_ps(), c22 = _mm_setzero_ps();
            if (M > 3) c03 = _mm_setzero_ps(), c13 = _mm_setzero_ps(), c23 = _mm_setzero_ps();
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
                b0 = _mm_loadu_ps(B + ob0);
                b1 = _mm_loadu_ps(B + ob1);
                b2 = _mm_loadu_ps(B + ob2);
                if (M > 0) a0 = _mm_set1_ps(A[oa0]), c00 = _mm_add_ps(_mm_mul_ps(b0, a0), c00), c10 = _mm_add_ps(_mm_mul_ps(b1, a0), c10), c20 = _mm_add_ps(_mm_mul_ps(b2, a0), c20);
                if (M > 1) a0 = _mm_set1_ps(A[oa1]), c01 = _mm_add_ps(_mm_mul_ps(b0, a0), c01), c11 = _mm_add_ps(_mm_mul_ps(b1, a0), c11), c21 = _mm_add_ps(_mm_mul_ps(b2, a0), c21);
                if (M > 2) a0 = _mm_set1_ps(A[oa2]), c02 = _mm_add_ps(_mm_mul_ps(b0, a0), c02), c12 = _mm_add_ps(_mm_mul_ps(b1, a0), c12), c22 = _mm_add_ps(_mm_mul_ps(b2, a0), c22);
                if (M > 3) a0 = _mm_set1_ps(A[oa3]), c03 = _mm_add_ps(_mm_mul_ps(b0, a0), c03), c13 = _mm_add_ps(_mm_mul_ps(b1, a0), c13), c23 = _mm_add_ps(_mm_mul_ps(b2, a0), c23);
                B += sb;
                A += sa;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            if (M > 0) AddProduct(C + 0 * F, _alpha, c00), AddProduct(C + 1 * F, _alpha, c10), AddProduct(C + 2 * F, _alpha, c20, tail), C += ldc;
            if (M > 1) AddProduct(C + 0 * F, _alpha, c01), AddProduct(C + 1 * F, _alpha, c11), AddProduct(C + 2 * F, _alpha, c21, tail), C += ldc;
            if (M > 2) AddProduct(C + 0 * F, _alpha, c02), AddProduct(C + 1 * F, _alpha, c12), AddProduct(C + 2 * F, _alpha, c22, tail), C += ldc;
            if (M > 3) AddProduct(C + 0 * F, _alpha, c03), AddProduct(C + 1 * F, _alpha, c13), AddProduct(C + 2 * F, _alpha, c23, tail), C += ldc;
        }

        template<int M> void GemmKernelMx8nnT(size_t, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m128 c00, c01, c02, c03, c04, c05, c10, c11, c12, c13, c14, c15, b0, b1, a0;
            if (M > 0) c00 = _mm_setzero_ps(), c10 = _mm_setzero_ps();
            if (M > 1) c01 = _mm_setzero_ps(), c11 = _mm_setzero_ps();
            if (M > 2) c02 = _mm_setzero_ps(), c12 = _mm_setzero_ps();
            if (M > 3) c03 = _mm_setzero_ps(), c13 = _mm_setzero_ps();
            if (M > 4) c04 = _mm_setzero_ps(), c14 = _mm_setzero_ps();
            if (M > 5) c05 = _mm_setzero_ps(), c15 = _mm_setzero_ps();
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
                b0 = _mm_loadu_ps(B + ob0);
                b1 = _mm_loadu_ps(B + ob1);
                if (M > 0) a0 = _mm_set1_ps(A[oa0]), c00 = _mm_add_ps(_mm_mul_ps(b0, a0), c00), c10 = _mm_add_ps(_mm_mul_ps(b1, a0), c10);
                if (M > 1) a0 = _mm_set1_ps(A[oa1]), c01 = _mm_add_ps(_mm_mul_ps(b0, a0), c01), c11 = _mm_add_ps(_mm_mul_ps(b1, a0), c11);
                if (M > 2) a0 = _mm_set1_ps(A[oa2]), c02 = _mm_add_ps(_mm_mul_ps(b0, a0), c02), c12 = _mm_add_ps(_mm_mul_ps(b1, a0), c12);
                if (M > 3) a0 = _mm_set1_ps(A[oa3]), c03 = _mm_add_ps(_mm_mul_ps(b0, a0), c03), c13 = _mm_add_ps(_mm_mul_ps(b1, a0), c13);
                if (M > 4) a0 = _mm_set1_ps(A[oa4]), c04 = _mm_add_ps(_mm_mul_ps(b0, a0), c04), c14 = _mm_add_ps(_mm_mul_ps(b1, a0), c14);
                if (M > 5) a0 = _mm_set1_ps(A[oa5]), c05 = _mm_add_ps(_mm_mul_ps(b0, a0), c05), c15 = _mm_add_ps(_mm_mul_ps(b1, a0), c15);
                B += sb;
                A += sa;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            if (M > 0) AddProduct(C + 0 * F, _alpha, c00), AddProduct(C + 1 * F, _alpha, c10, tail), C += ldc;
            if (M > 1) AddProduct(C + 0 * F, _alpha, c01), AddProduct(C + 1 * F, _alpha, c11, tail), C += ldc;
            if (M > 2) AddProduct(C + 0 * F, _alpha, c02), AddProduct(C + 1 * F, _alpha, c12, tail), C += ldc;
            if (M > 3) AddProduct(C + 0 * F, _alpha, c03), AddProduct(C + 1 * F, _alpha, c13, tail), C += ldc;
            if (M > 4) AddProduct(C + 0 * F, _alpha, c04), AddProduct(C + 1 * F, _alpha, c14, tail), C += ldc;
            if (M > 5) AddProduct(C + 0 * F, _alpha, c05), AddProduct(C + 1 * F, _alpha, c15, tail), C += ldc;
        }

        template<int M> void GemmKernelMx4nnT(size_t, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m128 c00, c01, c02, c03, c04, c05, b0;
            if (M > 0) c00 = _mm_setzero_ps();
            if (M > 1) c01 = _mm_setzero_ps();
            if (M > 2) c02 = _mm_setzero_ps();
            if (M > 3) c03 = _mm_setzero_ps();
            if (M > 4) c04 = _mm_setzero_ps();
            if (M > 5) c05 = _mm_setzero_ps();
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
                b0 = _mm_loadu_ps(B + ob0);
                if (M > 0) c00 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa0])), c00);
                if (M > 1) c01 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa1])), c01);
                if (M > 2) c02 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa2])), c02);
                if (M > 3) c03 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa3])), c03);
                if (M > 4) c04 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa4])), c04);
                if (M > 5) c05 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[oa5])), c05);
                B += sb;
                A += sa;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            if (M > 0) AddProduct(C + 0 * ldc, _alpha, c00, tail);
            if (M > 1) AddProduct(C + 1 * ldc, _alpha, c01, tail);
            if (M > 2) AddProduct(C + 2 * ldc, _alpha, c02, tail);
            if (M > 3) AddProduct(C + 3 * ldc, _alpha, c03, tail);
            if (M > 4) AddProduct(C + 4 * ldc, _alpha, c04, tail);
            if (M > 5) AddProduct(C + 5 * ldc, _alpha, c05, tail);
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
                        __m128 s0 = _mm_loadu_ps(ps + 0 * stride);
                        __m128 s1 = _mm_loadu_ps(ps + 1 * stride);
                        __m128 s2 = _mm_loadu_ps(ps + 2 * stride);
                        __m128 s3 = _mm_loadu_ps(ps + 3 * stride);
                        __m128 s4 = _mm_loadu_ps(ps + 4 * stride);
                        __m128 s5 = _mm_loadu_ps(ps + 5 * stride);
                        __m128 s00 = _mm_unpacklo_ps(s0, s2);
                        __m128 s01 = _mm_unpacklo_ps(s1, s3);
                        __m128 s10 = _mm_unpackhi_ps(s0, s2);
                        __m128 s11 = _mm_unpackhi_ps(s1, s3);
                        __m128 s20 = _mm_unpacklo_ps(s4, s5);
                        __m128 s21 = _mm_unpackhi_ps(s4, s5);
                        _mm_storeu_ps(dst + 0, _mm_unpacklo_ps(s00, s01));
                        _mm_storel_pi((__m64*)(dst + 4), s20);
                        _mm_storeu_ps(dst + 6, _mm_unpackhi_ps(s00, s01));
                        _mm_storeh_pi((__m64*)(dst + 10), s20);
                        _mm_storeu_ps(dst + 12, _mm_unpacklo_ps(s10, s11));
                        _mm_storel_pi((__m64*)(dst + 16), s21);
                        _mm_storeu_ps(dst + 18, _mm_unpackhi_ps(s10, s11));
                        _mm_storeh_pi((__m64*)(dst + 22), s21);
                        dst += 24;
                    }
                }
                if (cell == 4 && m == 4)
                {
                    size_t K4 = AlignLo(K, 4);
                    for (; k < K4; k += 4)
                    {
                        const float * ps = src + k;
                        __m128 s0 = _mm_loadu_ps(ps + 0 * stride);
                        __m128 s1 = _mm_loadu_ps(ps + 1 * stride);
                        __m128 s2 = _mm_loadu_ps(ps + 2 * stride);
                        __m128 s3 = _mm_loadu_ps(ps + 3 * stride);
                        __m128 s00 = _mm_unpacklo_ps(s0, s2);
                        __m128 s01 = _mm_unpacklo_ps(s1, s3);
                        __m128 s10 = _mm_unpackhi_ps(s0, s2);
                        __m128 s11 = _mm_unpackhi_ps(s1, s3);
                        _mm_storeu_ps(dst + 0, _mm_unpacklo_ps(s00, s01));
                        _mm_storeu_ps(dst + 4, _mm_unpackhi_ps(s00, s01));
                        _mm_storeu_ps(dst + 8, _mm_unpacklo_ps(s10, s11));
                        _mm_storeu_ps(dst + 12, _mm_unpackhi_ps(s10, s11));
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
                            _mm_storeu_ps(pB + 0 * F, _mm_loadu_ps(b + 0 * F));
                            pB += microN;
                        }
                    }
                    else
                    {
                        __m128 mask0 = LeftNotZero32f(n - 0 * F);
                        for (; k < K - 1; ++k)
                        {
                            const float * b = B + k * ldb;
                            _mm_storeu_ps(pB + 0 * F, _mm_and_ps(mask0, _mm_loadu_ps(b + 0 * F)));
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
                            _mm_storeu_ps(pB + 0 * F, _mm_loadu_ps(b + 0 * F));
                            _mm_storeu_ps(pB + 1 * F, _mm_loadu_ps(b + 1 * F));
                            pB += microN;
                        }
                    }
                    else
                    {
                        __m128 mask0 = LeftNotZero32f(n - 0 * F);
                        __m128 mask1 = LeftNotZero32f(n - 1 * F);
                        for (; k < K - 1; ++k)
                        {
                            const float * b = B + k * ldb;
                            _mm_storeu_ps(pB + 0 * F, _mm_and_ps(mask0, _mm_loadu_ps(b + 0 * F)));
                            _mm_storeu_ps(pB + 1 * F, _mm_and_ps(mask1, _mm_loadu_ps(b + 1 * F)));
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
                            _mm_storeu_ps(pB + 0 * F, _mm_loadu_ps(b + 0 * F));
                            _mm_storeu_ps(pB + 1 * F, _mm_loadu_ps(b + 1 * F));
                            _mm_storeu_ps(pB + 2 * F, _mm_loadu_ps(b + 2 * F));
                            pB += microN;
                        }
                    }
                    else
                    {
                        __m128 mask0 = LeftNotZero32f(n - 0 * F);
                        __m128 mask1 = LeftNotZero32f(n - 1 * F);
                        __m128 mask2 = LeftNotZero32f(n - 2 * F);
                        for (; k < K - 1; ++k)
                        {
                            const float * b = B + k * ldb;
                            _mm_storeu_ps(pB + 0 * F, _mm_and_ps(mask0, _mm_loadu_ps(b + 0 * F)));
                            _mm_storeu_ps(pB + 1 * F, _mm_and_ps(mask1, _mm_loadu_ps(b + 1 * F)));
                            _mm_storeu_ps(pB + 2 * F, _mm_and_ps(mask2, _mm_loadu_ps(b + 2 * F)));
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

        SIMD_INLINE void ScaleC(float * C, __m128 beta)
        {
            _mm_storeu_ps(C, _mm_mul_ps(_mm_loadu_ps(C), beta));
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
                __m128 _beta = _mm_set1_ps(beta);
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
            typedef Simd::GemmNN<float, F, size_t> GemmNN;
            GemmNN::Main kernelMM, kernelMT;
            GemmNN::Tail kernelTM, kernelTT;
            size_t microM, microN, L1, L2;
#ifdef SIMD_X64_ENABLE
            if (N < K)
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
#else
            microM = 4;
            microN = 4;
            kernelMM = GemmKernel4x4nn;
            kernelMT = GemmKernel4x4nn;
            kernelTM = GemmKernelMx4nn;
            kernelTT = GemmKernelMx4nn;
#endif
            GemmNN::PackA packA = NULL;
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
#ifdef SIMD_X64_ENABLE
            if (type == GemmKernelF3 || (type == GemmKernelAny && (M == 4 || M == 8 || M == 16) && N > 8))
            {
                microM = 4;
                microN = 12;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Sse2::GemmKernel4x12nn;
                kernelMT = tail > DF ? Sse2::GemmKernel4x12nn : (tail > F ? Sse2::GemmKernel4x8nn : Sse2::GemmKernel4x4nn);
                kernelTM = Sse2::GetGemmTail(M%microM, microN);
                kernelTT = Sse2::GetGemmTail(M%microM, tail);
                type = GemmKernelF3;
            }
            if (type == GemmKernelF2 || (type == GemmKernelF3 && N <= 8) || (type == GemmKernelAny && N > 4))
            {
                microM = 6;
                microN = 8;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Sse2::GemmKernel6x8nn;
                kernelMT = tail > F ? Sse2::GemmKernel6x8nn : Sse2::GemmKernel6x4nn;
                kernelTM = Sse2::GetGemmTail(M%microM, microN);
                kernelTT = Sse2::GetGemmTail(M%microM, tail);
                type = GemmKernelF2;
            }
            if (type == GemmKernelF1 || (type == GemmKernelF2 && N <= 4) || type == GemmKernelAny)
            {
                microM = 6;
                microN = 4;
                kernelMM = Sse2::GemmKernel6x4nn;
                kernelMT = Sse2::GemmKernel6x4nn;
                kernelTM = Sse2::GetGemmTail(M%microM, microN);
                kernelTT = Sse2::GetGemmTail(M%microM, microN);
                type = GemmKernelF1;
            }
#else
            microM = 4;
            microN = 4;
            kernelMM = Sse2::GemmKernel4x4nn;
            kernelMT = Sse2::GemmKernel4x4nn;
            kernelTM = Sse2::GetGemmTail(M%microM, microN);
            kernelTT = Sse2::GetGemmTail(M%microM, microN);
#endif
            return Gemm32fNNcb(M, N, K, microM, microN, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(),  
                kernelMM, kernelMT, kernelTM, kernelTT, NULL, Sse2::GemmPackB, Sse2::GemmScaleC, NULL, compatibility);
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
#endif// SIMD_SSE2_ENABLE
}
