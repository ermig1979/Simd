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
#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        SIMD_INLINE void AddProduct(float * ptr, __m256 value, __m256 alpha)
        {
            _mm256_storeu_ps(ptr, _mm256_add_ps(_mm256_mul_ps(value, alpha), _mm256_loadu_ps(ptr)));
        }

        SIMD_INLINE void AddProduct(float * ptr, __m256 value, __m256 alpha, size_t tail)
        {
            if (tail == F)
                AddProduct(ptr, value, alpha);
            else
            {
                float tmp[F];
                _mm256_storeu_ps(tmp, _mm256_add_ps(_mm256_mul_ps(value, alpha), _mm256_loadu_ps(ptr)));
                for (size_t i = 0; i < tail; ++i)
                    ptr[i] = tmp[i];
            }
        }

        void GemmKernel4x24nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m256 c00 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps();
            __m256 c30 = _mm256_setzero_ps();
            __m256 c01 = _mm256_setzero_ps();
            __m256 c11 = _mm256_setzero_ps();
            __m256 c21 = _mm256_setzero_ps();
            __m256 c31 = _mm256_setzero_ps();
            __m256 c02 = _mm256_setzero_ps();
            __m256 c12 = _mm256_setzero_ps();
            __m256 c22 = _mm256_setzero_ps();
            __m256 c32 = _mm256_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            __m256 b0, b1, b2, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm256_loadu_ps(B + ob0);
                b1 = _mm256_loadu_ps(B + ob1);
                b2 = _mm256_loadu_ps(B + ob2);
                a0 = _mm256_set1_ps(A[oa0]);
                c00 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c00);
                c01 = _mm256_add_ps(_mm256_mul_ps(a0, b1), c01);
                c02 = _mm256_add_ps(_mm256_mul_ps(a0, b2), c02);
                a0 = _mm256_set1_ps(A[oa1]);
                c10 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c10);
                c11 = _mm256_add_ps(_mm256_mul_ps(a0, b1), c11);
                c12 = _mm256_add_ps(_mm256_mul_ps(a0, b2), c12);
                a0 = _mm256_set1_ps(A[oa2]);
                c20 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c20);
                c21 = _mm256_add_ps(_mm256_mul_ps(a0, b1), c21);
                c22 = _mm256_add_ps(_mm256_mul_ps(a0, b2), c22);
                a0 = _mm256_set1_ps(A[oa3]);
                c30 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c30);
                c31 = _mm256_add_ps(_mm256_mul_ps(a0, b1), c31);
                c32 = _mm256_add_ps(_mm256_mul_ps(a0, b2), c32);
                B += sb;
                A += sa;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
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

        void GemmKernel4x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m256 c00 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps();
            __m256 c30 = _mm256_setzero_ps();
            __m256 c01 = _mm256_setzero_ps();
            __m256 c11 = _mm256_setzero_ps();
            __m256 c21 = _mm256_setzero_ps();
            __m256 c31 = _mm256_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            __m256 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm256_loadu_ps(B + ob0);
                b1 = _mm256_loadu_ps(B + ob1);
                a0 = _mm256_set1_ps(A[oa0]);
                c00 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c00);
                c01 = _mm256_add_ps(_mm256_mul_ps(a0, b1), c01);
                a0 = _mm256_set1_ps(A[oa1]);
                c10 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c10);
                c11 = _mm256_add_ps(_mm256_mul_ps(a0, b1), c11);
                a0 = _mm256_set1_ps(A[oa2]);
                c20 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c20);
                c21 = _mm256_add_ps(_mm256_mul_ps(a0, b1), c21);
                a0 = _mm256_set1_ps(A[oa3]);
                c30 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c30);
                c31 = _mm256_add_ps(_mm256_mul_ps(a0, b1), c31);
                B += sb;
                A += sa;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
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

        void GemmKernel4x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            const size_t ob0 = ldb * 0;
            __m256 b0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm256_loadu_ps(B + ob0);
                c0 = _mm256_add_ps(_mm256_mul_ps(b0, _mm256_set1_ps(A[oa0])), c0);
                c1 = _mm256_add_ps(_mm256_mul_ps(b0, _mm256_set1_ps(A[oa1])), c1);
                c2 = _mm256_add_ps(_mm256_mul_ps(b0, _mm256_set1_ps(A[oa2])), c2);
                c3 = _mm256_add_ps(_mm256_mul_ps(b0, _mm256_set1_ps(A[oa3])), c3);
                B += sb;
                A += sa;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
            AddProduct(C + 0 * ldc, _alpha, c0, tail);
            AddProduct(C + 1 * ldc, _alpha, c1, tail);
            AddProduct(C + 2 * ldc, _alpha, c2, tail);
            AddProduct(C + 3 * ldc, _alpha, c3, tail);
        }

        void GemmKernel6x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m256 c00 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps();
            __m256 c30 = _mm256_setzero_ps();
            __m256 c40 = _mm256_setzero_ps();
            __m256 c50 = _mm256_setzero_ps();
            __m256 c01 = _mm256_setzero_ps();
            __m256 c11 = _mm256_setzero_ps();
            __m256 c21 = _mm256_setzero_ps();
            __m256 c31 = _mm256_setzero_ps();
            __m256 c41 = _mm256_setzero_ps();
            __m256 c51 = _mm256_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            __m256 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm256_loadu_ps(B + ob0);
                b1 = _mm256_loadu_ps(B + ob1);
                a0 = _mm256_set1_ps(A[oa0]);
                c00 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c00);
                c01 = _mm256_add_ps(_mm256_mul_ps(a0, b1), c01);
                a0 = _mm256_set1_ps(A[oa1]);
                c10 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c10);
                c11 = _mm256_add_ps(_mm256_mul_ps(a0, b1), c11);
                a0 = _mm256_set1_ps(A[oa2]);
                c20 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c20);
                c21 = _mm256_add_ps(_mm256_mul_ps(a0, b1), c21);
                a0 = _mm256_set1_ps(A[oa3]);
                c30 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c30);
                c31 = _mm256_add_ps(_mm256_mul_ps(a0, b1), c31);
                a0 = _mm256_set1_ps(A[oa4]);
                c40 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c40);
                c41 = _mm256_add_ps(_mm256_mul_ps(a0, b1), c41);
                a0 = _mm256_set1_ps(A[oa5]);
                c50 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c50);
                c51 = _mm256_add_ps(_mm256_mul_ps(a0, b1), c51);
                B += sb;
                A += sa;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
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

        void GemmKernel6x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m256 c00 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps();
            __m256 c30 = _mm256_setzero_ps();
            __m256 c40 = _mm256_setzero_ps();
            __m256 c50 = _mm256_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            const size_t ob0 = ldb * 0;
            __m256 b0, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm256_loadu_ps(B + ob0);
                a0 = _mm256_set1_ps(A[oa0]);
                c00 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c00);
                a0 = _mm256_set1_ps(A[oa1]);
                c10 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c10);
                a0 = _mm256_set1_ps(A[oa2]);
                c20 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c20);
                a0 = _mm256_set1_ps(A[oa3]);
                c30 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c30);
                a0 = _mm256_set1_ps(A[oa4]);
                c40 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c40);
                a0 = _mm256_set1_ps(A[oa5]);
                c50 = _mm256_add_ps(_mm256_mul_ps(a0, b0), c50);
                B += sb;
                A += sa;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
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
        }

        void GemmKernelMx24nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m256 c[4][3];
            size_t oa[4];
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = _mm256_setzero_ps();
                c[i][1] = _mm256_setzero_ps();
                c[i][2] = _mm256_setzero_ps();
                oa[i] = lda * i;
            }
            __m256 b0, b1, b2, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm256_loadu_ps(B + ob0);
                b1 = _mm256_loadu_ps(B + ob1);
                b2 = _mm256_loadu_ps(B + ob2);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = _mm256_set1_ps(A[oa[i]]);
                    c[i][0] = _mm256_add_ps(_mm256_mul_ps(b0, a0), c[i][0]);
                    c[i][1] = _mm256_add_ps(_mm256_mul_ps(b1, a0), c[i][1]);
                    c[i][2] = _mm256_add_ps(_mm256_mul_ps(b2, a0), c[i][2]);
                }
                B += sb;
                A += sa;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
            for (size_t i = 0; i < M; ++i)
            {
                AddProduct(C + 0 * F, _alpha, c[i][0]);
                AddProduct(C + 1 * F, _alpha, c[i][1]);
                AddProduct(C + 2 * F, _alpha, c[i][2], tail);
                C += ldc;
            }
        }

        void GemmKernelMx16nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m256 c[6][2];
            size_t oa[6];
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = _mm256_setzero_ps();
                c[i][1] = _mm256_setzero_ps();
                oa[i] = lda * i;
            }
            __m256 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm256_loadu_ps(B + ob0);
                b1 = _mm256_loadu_ps(B + ob1);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = _mm256_set1_ps(A[oa[i]]);
                    c[i][0] = _mm256_add_ps(_mm256_mul_ps(b0, a0), c[i][0]);
                    c[i][1] = _mm256_add_ps(_mm256_mul_ps(b1, a0), c[i][1]);
                }
                B += sb;
                A += sa;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
            for (size_t i = 0; i < M; ++i)
            {
                AddProduct(C + 0 * F, _alpha, c[i][0]);
                AddProduct(C + 1 * F, _alpha, c[i][1], tail);
                C += ldc;
            }
        }

        void GemmKernelMx8nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
#ifdef SIMD_X64_ENABLE
            __m256 c[6];
            size_t oa[6];
#else
            __m256 c[4];
            size_t oa[4];
#endif
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            for (size_t i = 0; i < M; ++i)
            {
                c[i] = _mm256_setzero_ps();
                oa[i] = lda * i;
            }
            __m256 b0, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm256_loadu_ps(B + ob0);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = _mm256_set1_ps(A[oa[i]]);
                    c[i] = _mm256_add_ps(_mm256_mul_ps(b0, a0), c[i]);
                }
                B += sb;
                A += sa;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
            for (size_t i = 0; i < M; ++i)
                AddProduct(C + i * ldc, _alpha, c[i], tail);
        }

        template<int M> void GemmKernelMx24nnT(size_t, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m256 c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, b0, b1, b2, a0;
            if (M > 0) c00 = _mm256_setzero_ps(), c10 = _mm256_setzero_ps(), c20 = _mm256_setzero_ps();
            if (M > 1) c01 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
            if (M > 2) c02 = _mm256_setzero_ps(), c12 = _mm256_setzero_ps(), c22 = _mm256_setzero_ps();
            if (M > 3) c03 = _mm256_setzero_ps(), c13 = _mm256_setzero_ps(), c23 = _mm256_setzero_ps();
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
                b0 = _mm256_loadu_ps(B + ob0);
                b1 = _mm256_loadu_ps(B + ob1);
                b2 = _mm256_loadu_ps(B + ob2);
                if (M > 0) a0 = _mm256_set1_ps(A[oa0]), c00 = _mm256_add_ps(_mm256_mul_ps(b0, a0), c00), c10 = _mm256_add_ps(_mm256_mul_ps(b1, a0), c10), c20 = _mm256_add_ps(_mm256_mul_ps(b2, a0), c20);
                if (M > 1) a0 = _mm256_set1_ps(A[oa1]), c01 = _mm256_add_ps(_mm256_mul_ps(b0, a0), c01), c11 = _mm256_add_ps(_mm256_mul_ps(b1, a0), c11), c21 = _mm256_add_ps(_mm256_mul_ps(b2, a0), c21);
                if (M > 2) a0 = _mm256_set1_ps(A[oa2]), c02 = _mm256_add_ps(_mm256_mul_ps(b0, a0), c02), c12 = _mm256_add_ps(_mm256_mul_ps(b1, a0), c12), c22 = _mm256_add_ps(_mm256_mul_ps(b2, a0), c22);
                if (M > 3) a0 = _mm256_set1_ps(A[oa3]), c03 = _mm256_add_ps(_mm256_mul_ps(b0, a0), c03), c13 = _mm256_add_ps(_mm256_mul_ps(b1, a0), c13), c23 = _mm256_add_ps(_mm256_mul_ps(b2, a0), c23);
                B += sb;
                A += sa;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
            if (M > 0) AddProduct(C + 0 * F, _alpha, c00), AddProduct(C + 1 * F, _alpha, c10), AddProduct(C + 2 * F, _alpha, c20, tail), C += ldc;
            if (M > 1) AddProduct(C + 0 * F, _alpha, c01), AddProduct(C + 1 * F, _alpha, c11), AddProduct(C + 2 * F, _alpha, c21, tail), C += ldc;
            if (M > 2) AddProduct(C + 0 * F, _alpha, c02), AddProduct(C + 1 * F, _alpha, c12), AddProduct(C + 2 * F, _alpha, c22, tail), C += ldc;
            if (M > 3) AddProduct(C + 0 * F, _alpha, c03), AddProduct(C + 1 * F, _alpha, c13), AddProduct(C + 2 * F, _alpha, c23, tail), C += ldc;
        }

        template<int M> void GemmKernelMx16nnT(size_t, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m256 c00, c01, c02, c03, c04, c05, c10, c11, c12, c13, c14, c15, b0, b1, a0;
            if (M > 0) c00 = _mm256_setzero_ps(), c10 = _mm256_setzero_ps();
            if (M > 1) c01 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
            if (M > 2) c02 = _mm256_setzero_ps(), c12 = _mm256_setzero_ps();
            if (M > 3) c03 = _mm256_setzero_ps(), c13 = _mm256_setzero_ps();
            if (M > 4) c04 = _mm256_setzero_ps(), c14 = _mm256_setzero_ps();
            if (M > 5) c05 = _mm256_setzero_ps(), c15 = _mm256_setzero_ps();
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
                b0 = _mm256_loadu_ps(B + ob0);
                b1 = _mm256_loadu_ps(B + ob1);
                if (M > 0) a0 = _mm256_set1_ps(A[oa0]), c00 = _mm256_add_ps(_mm256_mul_ps(b0, a0), c00), c10 = _mm256_add_ps(_mm256_mul_ps(b1, a0), c10);
                if (M > 1) a0 = _mm256_set1_ps(A[oa1]), c01 = _mm256_add_ps(_mm256_mul_ps(b0, a0), c01), c11 = _mm256_add_ps(_mm256_mul_ps(b1, a0), c11);
                if (M > 2) a0 = _mm256_set1_ps(A[oa2]), c02 = _mm256_add_ps(_mm256_mul_ps(b0, a0), c02), c12 = _mm256_add_ps(_mm256_mul_ps(b1, a0), c12);
                if (M > 3) a0 = _mm256_set1_ps(A[oa3]), c03 = _mm256_add_ps(_mm256_mul_ps(b0, a0), c03), c13 = _mm256_add_ps(_mm256_mul_ps(b1, a0), c13);
                if (M > 4) a0 = _mm256_set1_ps(A[oa4]), c04 = _mm256_add_ps(_mm256_mul_ps(b0, a0), c04), c14 = _mm256_add_ps(_mm256_mul_ps(b1, a0), c14);
                if (M > 5) a0 = _mm256_set1_ps(A[oa5]), c05 = _mm256_add_ps(_mm256_mul_ps(b0, a0), c05), c15 = _mm256_add_ps(_mm256_mul_ps(b1, a0), c15);
                B += sb;
                A += sa;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
            if (M > 0) AddProduct(C + 0 * F, _alpha, c00), AddProduct(C + 1 * F, _alpha, c10, tail), C += ldc;
            if (M > 1) AddProduct(C + 0 * F, _alpha, c01), AddProduct(C + 1 * F, _alpha, c11, tail), C += ldc;
            if (M > 2) AddProduct(C + 0 * F, _alpha, c02), AddProduct(C + 1 * F, _alpha, c12, tail), C += ldc;
            if (M > 3) AddProduct(C + 0 * F, _alpha, c03), AddProduct(C + 1 * F, _alpha, c13, tail), C += ldc;
            if (M > 4) AddProduct(C + 0 * F, _alpha, c04), AddProduct(C + 1 * F, _alpha, c14, tail), C += ldc;
            if (M > 5) AddProduct(C + 0 * F, _alpha, c05), AddProduct(C + 1 * F, _alpha, c15, tail), C += ldc;
        }

        template<int M> void GemmKernelMx8nnT(size_t, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, size_t tail)
        {
            __m256 c00, c01, c02, c03, c04, c05, b0;
            if (M > 0) c00 = _mm256_setzero_ps();
            if (M > 1) c01 = _mm256_setzero_ps();
            if (M > 2) c02 = _mm256_setzero_ps();
            if (M > 3) c03 = _mm256_setzero_ps();
            if (M > 4) c04 = _mm256_setzero_ps();
            if (M > 5) c05 = _mm256_setzero_ps();
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
                b0 = _mm256_loadu_ps(B + ob0);
                if (M > 0) c00 = _mm256_add_ps(_mm256_mul_ps(b0, _mm256_set1_ps(A[oa0])), c00);
                if (M > 1) c01 = _mm256_add_ps(_mm256_mul_ps(b0, _mm256_set1_ps(A[oa1])), c01);
                if (M > 2) c02 = _mm256_add_ps(_mm256_mul_ps(b0, _mm256_set1_ps(A[oa2])), c02);
                if (M > 3) c03 = _mm256_add_ps(_mm256_mul_ps(b0, _mm256_set1_ps(A[oa3])), c03);
                if (M > 4) c04 = _mm256_add_ps(_mm256_mul_ps(b0, _mm256_set1_ps(A[oa4])), c04);
                if (M > 5) c05 = _mm256_add_ps(_mm256_mul_ps(b0, _mm256_set1_ps(A[oa5])), c05);
                B += sb;
                A += sa;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
            if (M > 0) AddProduct(C + 0 * ldc, _alpha, c00, tail);
            if (M > 1) AddProduct(C + 1 * ldc, _alpha, c01, tail);
            if (M > 2) AddProduct(C + 2 * ldc, _alpha, c02, tail);
            if (M > 3) AddProduct(C + 3 * ldc, _alpha, c03, tail);
            if (M > 4) AddProduct(C + 4 * ldc, _alpha, c04, tail);
            if (M > 5) AddProduct(C + 5 * ldc, _alpha, c05, tail);
        }

        SIMD_INLINE Simd::GemmNN<float, F, size_t>::Tail GetGemmTail(size_t M, size_t N)
        {
            if (N <= 8)
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
                }
            }
            else if (N <= 24)
            {
                switch (M)
                {
                case 0: return GemmKernelMx24nnT<0>;
                case 1: return GemmKernelMx24nnT<1>;
                case 2: return GemmKernelMx24nnT<2>;
                case 3: return GemmKernelMx24nnT<3>;
                }
            }
            assert(0);
            return NULL;
        }

        SIMD_INLINE void GemmPackA_4x8(const float* src, size_t stride, float* dst)
        {
            __m256 s0 = _mm256_loadu_ps(src + 0 * stride);
            __m256 s1 = _mm256_loadu_ps(src + 1 * stride);
            __m256 s2 = _mm256_loadu_ps(src + 2 * stride);
            __m256 s3 = _mm256_loadu_ps(src + 3 * stride);
            __m256 s00 = _mm256_unpacklo_ps(s0, s2);
            __m256 s01 = _mm256_unpacklo_ps(s1, s3);
            __m256 s10 = _mm256_unpackhi_ps(s0, s2);
            __m256 s11 = _mm256_unpackhi_ps(s1, s3);
            __m256 d0 = _mm256_unpacklo_ps(s00, s01);
            __m256 d1 = _mm256_unpackhi_ps(s00, s01);
            __m256 d2 = _mm256_unpacklo_ps(s10, s11);
            __m256 d3 = _mm256_unpackhi_ps(s10, s11);
            _mm256_storeu_ps(dst + 0x00, _mm256_permute2f128_ps(d0, d1, 0x20));
            _mm256_storeu_ps(dst + 0x08, _mm256_permute2f128_ps(d2, d3, 0x20));
            _mm256_storeu_ps(dst + 0x10, _mm256_permute2f128_ps(d0, d1, 0x31));
            _mm256_storeu_ps(dst + 0x18, _mm256_permute2f128_ps(d2, d3, 0x31));
        }

        SIMD_INLINE void GemmPackA_4x4(const float* src, size_t stride, float* dst)
        {
            __m128 s0 = _mm_loadu_ps(src + 0 * stride);
            __m128 s1 = _mm_loadu_ps(src + 1 * stride);
            __m128 s2 = _mm_loadu_ps(src + 2 * stride);
            __m128 s3 = _mm_loadu_ps(src + 3 * stride);
            __m128 s00 = _mm_unpacklo_ps(s0, s2);
            __m128 s01 = _mm_unpacklo_ps(s1, s3);
            __m128 s10 = _mm_unpackhi_ps(s0, s2);
            __m128 s11 = _mm_unpackhi_ps(s1, s3);
            _mm_storeu_ps(dst + 0, _mm_unpacklo_ps(s00, s01));
            _mm_storeu_ps(dst + 4, _mm_unpackhi_ps(s00, s01));
            _mm_storeu_ps(dst + 8, _mm_unpacklo_ps(s10, s11));
            _mm_storeu_ps(dst + 12, _mm_unpackhi_ps(s10, s11));
        }

        SIMD_INLINE void GemmPackA_6x4(const float* src, size_t stride, float* dst)
        {
            __m128 s0 = _mm_loadu_ps(src + 0 * stride);
            __m128 s1 = _mm_loadu_ps(src + 1 * stride);
            __m128 s2 = _mm_loadu_ps(src + 2 * stride);
            __m128 s3 = _mm_loadu_ps(src + 3 * stride);
            __m128 s4 = _mm_loadu_ps(src + 4 * stride);
            __m128 s5 = _mm_loadu_ps(src + 5 * stride);
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
        }

        void GemmPackA(const float * src, size_t stride, size_t M, size_t K, size_t cell, float * dst)
        {
            size_t K4 = AlignLo(K, 4), K8 = AlignLo(K, 8);
            for (size_t i = 0; i < M; i += cell)
            {
                size_t m = Simd::Min(cell, M - i), k = 0;
                if (cell == 4 && m == 4)
                {
                    for (; k < K8; k += 8, dst += 32)
                        GemmPackA_4x8(src + k, stride, dst);
                    for (; k < K4; k += 4, dst += 16)
                        GemmPackA_4x4(src + k, stride, dst);
                }
                else if (cell == 6 && m == 6)
                {
                    for (; k < K4; k += 4, dst += 24)
                        GemmPackA_6x4(src + k, stride, dst);
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
                            _mm256_storeu_ps(pB + 0 * F, _mm256_loadu_ps(b + 0 * F));
                            pB += microN;
                        }
                    }
                    else
                    {
                        __m256 mask0 = Avx::LeftNotZero32f(n - 0 * F);
                        for (; k < K - 1; ++k)
                        {
                            const float * b = B + k * ldb;
                            _mm256_storeu_ps(pB + 0 * F, _mm256_and_ps(mask0, _mm256_loadu_ps(b + 0 * F)));
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
                            _mm256_storeu_ps(pB + 0 * F, _mm256_loadu_ps(b + 0 * F));
                            _mm256_storeu_ps(pB + 1 * F, _mm256_loadu_ps(b + 1 * F));
                            pB += microN;
                        }
                    }
                    else
                    {
                        __m256 mask0 = Avx::LeftNotZero32f(n - 0 * F);
                        __m256 mask1 = Avx::LeftNotZero32f(n - 1 * F);
                        for (; k < K - 1; ++k)
                        {
                            const float * b = B + k * ldb;
                            _mm256_storeu_ps(pB + 0 * F, _mm256_and_ps(mask0, _mm256_loadu_ps(b + 0 * F)));
                            _mm256_storeu_ps(pB + 1 * F, _mm256_and_ps(mask1, _mm256_loadu_ps(b + 1 * F)));
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
                            _mm256_storeu_ps(pB + 0 * F, _mm256_loadu_ps(b + 0 * F));
                            _mm256_storeu_ps(pB + 1 * F, _mm256_loadu_ps(b + 1 * F));
                            _mm256_storeu_ps(pB + 2 * F, _mm256_loadu_ps(b + 2 * F));
                            pB += microN;
                        }
                    }
                    else
                    {
                        __m256 mask0 = Avx::LeftNotZero32f(n - 0 * F);
                        __m256 mask1 = Avx::LeftNotZero32f(n - 1 * F);
                        __m256 mask2 = Avx::LeftNotZero32f(n - 2 * F);
                        for (; k < K - 1; ++k)
                        {
                            const float * b = B + k * ldb;
                            _mm256_storeu_ps(pB + 0 * F, _mm256_and_ps(mask0, _mm256_loadu_ps(b + 0 * F)));
                            _mm256_storeu_ps(pB + 1 * F, _mm256_and_ps(mask1, _mm256_loadu_ps(b + 1 * F)));
                            _mm256_storeu_ps(pB + 2 * F, _mm256_and_ps(mask2, _mm256_loadu_ps(b + 2 * F)));
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

        SIMD_INLINE void ScaleC(float * C, __m256 beta)
        {
            _mm256_storeu_ps(C, _mm256_mul_ps(_mm256_loadu_ps(C), beta));
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
                __m256 _beta = _mm256_set1_ps(beta);
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
                microN = 16;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel6x16nn;
                kernelMT = tail > F ? GemmKernel6x16nn : GemmKernel6x8nn;
                kernelTM = GemmKernelMx16nn;
                kernelTT = tail > F ? GemmKernelMx16nn : GemmKernelMx8nn;
            }
            else
            {
                microM = 4;
                microN = 24;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel4x24nn;
                kernelMT = tail > DF ? GemmKernel4x24nn : (tail > F ? GemmKernel4x16nn : GemmKernel4x8nn);
                kernelTM = GemmKernelMx24nn;
                kernelTT = tail > DF ? GemmKernelMx24nn : (tail > F ? GemmKernelMx16nn : GemmKernelMx8nn);
            }
#else
            microM = 4;
            microN = 8;
            kernelMM = GemmKernel4x8nn;
            kernelMT = GemmKernel4x8nn;
            kernelTM = GemmKernelMx8nn;
            kernelTT = GemmKernelMx8nn;
#endif
            GemmNN::PackA packA = NULL;
            L1 = N > 4096 ? Base::AlgCacheL2() : Base::AlgCacheL1();
            L2 = N > 4096 ? Base::AlgCacheL3() : Base::AlgCacheL2();
            GemmNN gemmNN(M, N, K, microM, microN, L1, L2, Base::AlgCacheL3(), 
                kernelMM, kernelMT, kernelTM, kernelTT, packA, Avx::GemmPackB, Avx::GemmScaleC, NULL);
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
            if (type == GemmKernelF3 || (type == GemmKernelAny && (M == 4 || M == 8 || M == 16) && N > 16))
            {
                microM = 4;
                microN = 24;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx::GemmKernel4x24nn;
                kernelMT = tail > DF ? Avx::GemmKernel4x24nn : (tail > F ? Avx::GemmKernel4x16nn : Avx::GemmKernel4x8nn);
                kernelTM = Avx::GetGemmTail(M%microM, microN);
                kernelTT = Avx::GetGemmTail(M%microM, tail);
                type = GemmKernelF3;
            }
            if (type == GemmKernelF2 || (type == GemmKernelF3 && N <= 16) || (type == GemmKernelAny && N > 8))
            {
                microM = 6;
                microN = 16;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx::GemmKernel6x16nn;
                kernelMT = tail > F ? Avx::GemmKernel6x16nn : Avx::GemmKernel6x8nn;
                kernelTM = Avx::GetGemmTail(M%microM, microN);
                kernelTT = Avx::GetGemmTail(M%microM, tail);
                type = GemmKernelF2;
            }
            if (type == GemmKernelF1 || (type == GemmKernelF2 && N <= 8) || type == GemmKernelAny)
            {
                microM = 6;
                microN = 8;
                kernelMM = Avx::GemmKernel6x8nn;
                kernelMT = Avx::GemmKernel6x8nn;
                kernelTM = Avx::GetGemmTail(M%microM, microN);
                kernelTT = Avx::GetGemmTail(M%microM, microN);
                type = GemmKernelF1;
            }
#else
            microM = 4;
            microN = 8;
            kernelMM = Avx::GemmKernel4x8nn;
            kernelMT = Avx::GemmKernel4x8nn;
            kernelTM = Avx::GetGemmTail(M%microM, microN);
            kernelTT = Avx::GetGemmTail(M%microM, microN);
#endif
            return Gemm32fNNcb(M, N, K, microM, microN, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(),  
                kernelMM, kernelMT, kernelTM, kernelTT, NULL, Avx::GemmPackB, Avx::GemmScaleC, NULL, compatibility);
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

        SIMD_INLINE __m256 Tail(size_t tail)
        {
            const int32_t mask[DF] = { 0, 0, 0, 0, 0, 0, 0, 0 , -1, -1, -1, -1, -1, -1, -1, -1 };
            return _mm256_loadu_ps((float*)(mask + tail));
        }

        SIMD_INLINE void Add4ExtractedSums(const __m256 & sum0, const __m256 & sum1, const __m256 & sum2, const __m256 & sum3, const __m128 & alpha, float * dst)
        {
            __m256 sum256 = _mm256_hadd_ps(_mm256_hadd_ps(sum0, sum1), _mm256_hadd_ps(sum2, sum3));
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            _mm_storeu_ps(dst, _mm_add_ps(_mm_loadu_ps(dst), _mm_mul_ps(alpha, sum128)));
        }

        static void Kernel1x1x8nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K8 = K & (~7);
            const float * A0 = A + 0 * lda;
            const float * B0 = B + 0 * ldb;
            __m256 c00 = _mm256_setzero_ps();
            __m256 a0, b0;
            for (size_t k = 0; k < K8; k += 8)
            {
                a0 = _mm256_loadu_ps(A0 + k);
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_add_ps(c00, _mm256_mul_ps(a0, b0));
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                a0 = _mm256_and_ps(tail, _mm256_loadu_ps(A0 + k));
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_add_ps(c00, _mm256_mul_ps(a0, b0));
            }
            C[0] += alpha * Avx::ExtractSum(c00);
        }

        static void Kernel1x4x8nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K8 = K & (~7);
            const float * A0 = A + 0 * lda;
            const float * B0 = B + 0 * ldb;
            const float * B1 = B + 1 * ldb;
            const float * B2 = B + 2 * ldb;
            const float * B3 = B + 3 * ldb;
            __m256 c00 = _mm256_setzero_ps();
            __m256 c01 = _mm256_setzero_ps();
            __m256 c02 = _mm256_setzero_ps();
            __m256 c03 = _mm256_setzero_ps();
            __m256 a0, b0;
            for (size_t k = 0; k < K8; k += 8)
            {
                a0 = _mm256_loadu_ps(A0 + k);
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_add_ps(c00, _mm256_mul_ps(a0, b0));
                b0 = _mm256_loadu_ps(B1 + k);
                c01 = _mm256_add_ps(c01, _mm256_mul_ps(a0, b0));
                b0 = _mm256_loadu_ps(B2 + k);
                c02 = _mm256_add_ps(c02, _mm256_mul_ps(a0, b0));
                b0 = _mm256_loadu_ps(B3 + k);
                c03 = _mm256_add_ps(c03, _mm256_mul_ps(a0, b0));
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                a0 = _mm256_and_ps(tail, _mm256_loadu_ps(A0 + k));
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_add_ps(c00, _mm256_mul_ps(a0, b0));
                b0 = _mm256_loadu_ps(B1 + k);
                c01 = _mm256_add_ps(c01, _mm256_mul_ps(a0, b0));
                b0 = _mm256_loadu_ps(B2 + k);
                c02 = _mm256_add_ps(c02, _mm256_mul_ps(a0, b0));
                b0 = _mm256_loadu_ps(B3 + k);
                c03 = _mm256_add_ps(c03, _mm256_mul_ps(a0, b0));
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            Add4ExtractedSums(c00, c01, c02, c03, _alpha, C + 0 * ldc);
        }

        static void Kernel2x1x8nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K8 = K & (~7);
            const float * A0 = A + 0 * lda;
            const float * A1 = A + 1 * lda;
            const float * B0 = B + 0 * ldb;
            __m256 c00 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps();
            __m256 a0, a1, b0;
            for (size_t k = 0; k < K8; k += 8)
            {
                a0 = _mm256_loadu_ps(A0 + k);
                a1 = _mm256_loadu_ps(A1 + k);
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_add_ps(c00, _mm256_mul_ps(a0, b0));
                c10 = _mm256_add_ps(c10, _mm256_mul_ps(a1, b0));
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                a0 = _mm256_and_ps(tail, _mm256_loadu_ps(A0 + k));
                a1 = _mm256_and_ps(tail, _mm256_loadu_ps(A1 + k));
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_add_ps(c00, _mm256_mul_ps(a0, b0));
                c10 = _mm256_add_ps(c10, _mm256_mul_ps(a1, b0));
            }
            C[0 * ldc] += alpha * Avx::ExtractSum(c00);
            C[1 * ldc] += alpha * Avx::ExtractSum(c10);
        }

        static void Kernel2x4x8nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K8 = K & (~7);
            const float * A0 = A + 0 * lda;
            const float * A1 = A + 1 * lda;
            const float * B0 = B + 0 * ldb;
            const float * B1 = B + 1 * ldb;
            const float * B2 = B + 2 * ldb;
            const float * B3 = B + 3 * ldb;
            __m256 c00 = _mm256_setzero_ps();
            __m256 c01 = _mm256_setzero_ps();
            __m256 c02 = _mm256_setzero_ps();
            __m256 c03 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps();
            __m256 c11 = _mm256_setzero_ps();
            __m256 c12 = _mm256_setzero_ps();
            __m256 c13 = _mm256_setzero_ps();
            __m256 a0, a1, b0;
            for (size_t k = 0; k < K8; k += 8)
            {
                a0 = _mm256_loadu_ps(A0 + k);
                a1 = _mm256_loadu_ps(A1 + k);
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_add_ps(c00, _mm256_mul_ps(a0, b0));
                c10 = _mm256_add_ps(c10, _mm256_mul_ps(a1, b0));
                b0 = _mm256_loadu_ps(B1 + k);
                c01 = _mm256_add_ps(c01, _mm256_mul_ps(a0, b0));
                c11 = _mm256_add_ps(c11, _mm256_mul_ps(a1, b0));
                b0 = _mm256_loadu_ps(B2 + k);
                c02 = _mm256_add_ps(c02, _mm256_mul_ps(a0, b0));
                c12 = _mm256_add_ps(c12, _mm256_mul_ps(a1, b0));
                b0 = _mm256_loadu_ps(B3 + k);
                c03 = _mm256_add_ps(c03, _mm256_mul_ps(a0, b0));
                c13 = _mm256_add_ps(c13, _mm256_mul_ps(a1, b0));
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                a0 = _mm256_and_ps(tail, _mm256_loadu_ps(A0 + k));
                a1 = _mm256_and_ps(tail, _mm256_loadu_ps(A1 + k));
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_add_ps(c00, _mm256_mul_ps(a0, b0));
                c10 = _mm256_add_ps(c10, _mm256_mul_ps(a1, b0));
                b0 = _mm256_loadu_ps(B1 + k);
                c01 = _mm256_add_ps(c01, _mm256_mul_ps(a0, b0));
                c11 = _mm256_add_ps(c11, _mm256_mul_ps(a1, b0));
                b0 = _mm256_loadu_ps(B2 + k);
                c02 = _mm256_add_ps(c02, _mm256_mul_ps(a0, b0));
                c12 = _mm256_add_ps(c12, _mm256_mul_ps(a1, b0));
                b0 = _mm256_loadu_ps(B3 + k);
                c03 = _mm256_add_ps(c03, _mm256_mul_ps(a0, b0));
                c13 = _mm256_add_ps(c13, _mm256_mul_ps(a1, b0));
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            Add4ExtractedSums(c00, c01, c02, c03, _alpha, C + 0 * ldc);
            Add4ExtractedSums(c10, c11, c12, c13, _alpha, C + 1 * ldc);
        }

        static void Kernel3x1x8nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K8 = K & (~7);
            const float * A0 = A + 0 * lda;
            const float * A1 = A + 1 * lda;
            const float * A2 = A + 2 * lda;
            const float * B0 = B + 0 * ldb;
            __m256 c00 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps();
            __m256 a0, a1, a2, b0;
            for (size_t k = 0; k < K8; k += 8)
            {
                a0 = _mm256_loadu_ps(A0 + k);
                a1 = _mm256_loadu_ps(A1 + k);
                a2 = _mm256_loadu_ps(A2 + k);
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_add_ps(c00, _mm256_mul_ps(a0, b0));
                c10 = _mm256_add_ps(c10, _mm256_mul_ps(a1, b0));
                c20 = _mm256_add_ps(c20, _mm256_mul_ps(a2, b0));
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                a0 = _mm256_and_ps(tail, _mm256_loadu_ps(A0 + k));
                a1 = _mm256_and_ps(tail, _mm256_loadu_ps(A1 + k));
                a2 = _mm256_and_ps(tail, _mm256_loadu_ps(A2 + k));
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_add_ps(c00, _mm256_mul_ps(a0, b0));
                c10 = _mm256_add_ps(c10, _mm256_mul_ps(a1, b0));
                c20 = _mm256_add_ps(c20, _mm256_mul_ps(a2, b0));
            }
            C[0 * ldc] += alpha * Avx::ExtractSum(c00);
            C[1 * ldc] += alpha * Avx::ExtractSum(c10);
            C[2 * ldc] += alpha * Avx::ExtractSum(c20);
        }

        static void Kernel3x4x8nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K8 = K & (~7);
            const float * A0 = A + 0 * lda;
            const float * A1 = A + 1 * lda;
            const float * A2 = A + 2 * lda;
            const float * B0 = B + 0 * ldb;
            const float * B1 = B + 1 * ldb;
            const float * B2 = B + 2 * ldb;
            const float * B3 = B + 3 * ldb;
            __m256 c00 = _mm256_setzero_ps();
            __m256 c01 = _mm256_setzero_ps();
            __m256 c02 = _mm256_setzero_ps();
            __m256 c03 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps();
            __m256 c11 = _mm256_setzero_ps();
            __m256 c12 = _mm256_setzero_ps();
            __m256 c13 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps();
            __m256 c21 = _mm256_setzero_ps();
            __m256 c22 = _mm256_setzero_ps();
            __m256 c23 = _mm256_setzero_ps();
            __m256 a0, a1, a2, b0;
            for (size_t k = 0; k < K8; k += 8)
            {
                a0 = _mm256_loadu_ps(A0 + k);
                a1 = _mm256_loadu_ps(A1 + k);
                a2 = _mm256_loadu_ps(A2 + k);
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_add_ps(c00, _mm256_mul_ps(a0, b0));
                c10 = _mm256_add_ps(c10, _mm256_mul_ps(a1, b0));
                c20 = _mm256_add_ps(c20, _mm256_mul_ps(a2, b0));
                b0 = _mm256_loadu_ps(B1 + k);
                c01 = _mm256_add_ps(c01, _mm256_mul_ps(a0, b0));
                c11 = _mm256_add_ps(c11, _mm256_mul_ps(a1, b0));
                c21 = _mm256_add_ps(c21, _mm256_mul_ps(a2, b0));
                b0 = _mm256_loadu_ps(B2 + k);
                c02 = _mm256_add_ps(c02, _mm256_mul_ps(a0, b0));
                c12 = _mm256_add_ps(c12, _mm256_mul_ps(a1, b0));
                c22 = _mm256_add_ps(c22, _mm256_mul_ps(a2, b0));
                b0 = _mm256_loadu_ps(B3 + k);
                c03 = _mm256_add_ps(c03, _mm256_mul_ps(a0, b0));
                c13 = _mm256_add_ps(c13, _mm256_mul_ps(a1, b0));
                c23 = _mm256_add_ps(c23, _mm256_mul_ps(a2, b0));
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                a0 = _mm256_and_ps(tail, _mm256_loadu_ps(A0 + k));
                a1 = _mm256_and_ps(tail, _mm256_loadu_ps(A1 + k));
                a2 = _mm256_and_ps(tail, _mm256_loadu_ps(A2 + k));
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_add_ps(c00, _mm256_mul_ps(a0, b0));
                c10 = _mm256_add_ps(c10, _mm256_mul_ps(a1, b0));
                c20 = _mm256_add_ps(c20, _mm256_mul_ps(a2, b0));
                b0 = _mm256_loadu_ps(B1 + k);
                c01 = _mm256_add_ps(c01, _mm256_mul_ps(a0, b0));
                c11 = _mm256_add_ps(c11, _mm256_mul_ps(a1, b0));
                c21 = _mm256_add_ps(c21, _mm256_mul_ps(a2, b0));
                b0 = _mm256_loadu_ps(B2 + k);
                c02 = _mm256_add_ps(c02, _mm256_mul_ps(a0, b0));
                c12 = _mm256_add_ps(c12, _mm256_mul_ps(a1, b0));
                c22 = _mm256_add_ps(c22, _mm256_mul_ps(a2, b0));
                b0 = _mm256_loadu_ps(B3 + k);
                c03 = _mm256_add_ps(c03, _mm256_mul_ps(a0, b0));
                c13 = _mm256_add_ps(c13, _mm256_mul_ps(a1, b0));
                c23 = _mm256_add_ps(c23, _mm256_mul_ps(a2, b0));
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
            GemmNT gemmNT(M, N, K, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), Avx::GemmScaleC,
                Kernel1x1x8nt, Kernel1x4x8nt, Kernel2x1x8nt, Kernel2x4x8nt, Kernel3x1x8nt, Kernel3x4x8nt, NULL, NULL);
#else
            GemmNT gemmNT(M, N, K, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), Sse2::GemmScaleC,
                Kernel1x1x8nt, Kernel1x4x8nt, NULL, NULL, NULL, NULL, NULL, NULL);
#endif
            gemmNT.Run(alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
#endif// SIMD_AVX_ENABLE
}
