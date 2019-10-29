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
#include "Simd/SimdAvx2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        SIMD_INLINE void AddProduct(float * ptr, __m512 value, __m512 alpha)
        {
            _mm512_storeu_ps(ptr, _mm512_fmadd_ps(value, alpha, _mm512_loadu_ps(ptr)));
        }

        SIMD_INLINE void AddProduct(float * ptr, __m512 value, __m512 alpha, __mmask16 mask)
        {
            _mm512_mask_storeu_ps(ptr, mask, _mm512_fmadd_ps(value, alpha, _mm512_maskz_loadu_ps(mask, ptr)));
        }

        void GemmKernel4x48nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c11 = _mm512_setzero_ps();
            __m512 c21 = _mm512_setzero_ps();
            __m512 c31 = _mm512_setzero_ps();
            __m512 c02 = _mm512_setzero_ps();
            __m512 c12 = _mm512_setzero_ps();
            __m512 c22 = _mm512_setzero_ps();
            __m512 c32 = _mm512_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            __m512 b0, b1, b2, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                b2 = _mm512_loadu_ps(B + ob2);
                a0 = _mm512_set1_ps(A[oa0]);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);
                c02 = _mm512_fmadd_ps(a0, b2, c02);
                a0 = _mm512_set1_ps(A[oa1]);
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                c11 = _mm512_fmadd_ps(a0, b1, c11);
                c12 = _mm512_fmadd_ps(a0, b2, c12);
                a0 = _mm512_set1_ps(A[oa2]);
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                c21 = _mm512_fmadd_ps(a0, b1, c21);
                c22 = _mm512_fmadd_ps(a0, b2, c22);
                a0 = _mm512_set1_ps(A[oa3]);
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                c31 = _mm512_fmadd_ps(a0, b1, c31);
                c32 = _mm512_fmadd_ps(a0, b2, c32);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01);
            AddProduct(C + 2 * F, _alpha, c02, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11);
            AddProduct(C + 2 * F, _alpha, c12, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21);
            AddProduct(C + 2 * F, _alpha, c22, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31);
            AddProduct(C + 2 * F, _alpha, c32, mask);
        }

        void GemmKernel4x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c11 = _mm512_setzero_ps();
            __m512 c21 = _mm512_setzero_ps();
            __m512 c31 = _mm512_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            __m512 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                a0 = _mm512_set1_ps(A[oa0]);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);
                a0 = _mm512_set1_ps(A[oa1]);
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                c11 = _mm512_fmadd_ps(a0, b1, c11);
                a0 = _mm512_set1_ps(A[oa2]);
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                c21 = _mm512_fmadd_ps(a0, b1, c21);
                a0 = _mm512_set1_ps(A[oa3]);
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                c31 = _mm512_fmadd_ps(a0, b1, c31);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31, mask);
        }

        void GemmKernel4x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c0 = _mm512_setzero_ps();
            __m512 c1 = _mm512_setzero_ps();
            __m512 c2 = _mm512_setzero_ps();
            __m512 c3 = _mm512_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            const size_t ob0 = ldb * 0;
            __m512 b0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                c0 = _mm512_fmadd_ps(b0, _mm512_set1_ps(A[oa0]), c0);
                c1 = _mm512_fmadd_ps(b0, _mm512_set1_ps(A[oa1]), c1);
                c2 = _mm512_fmadd_ps(b0, _mm512_set1_ps(A[oa2]), c2);
                c3 = _mm512_fmadd_ps(b0, _mm512_set1_ps(A[oa3]), c3);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            AddProduct(C + 0 * ldc, _alpha, c0, mask);
            AddProduct(C + 1 * ldc, _alpha, c1, mask);
            AddProduct(C + 2 * ldc, _alpha, c2, mask);
            AddProduct(C + 3 * ldc, _alpha, c3, mask);
        }

        void GemmKernel6x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c11 = _mm512_setzero_ps();
            __m512 c21 = _mm512_setzero_ps();
            __m512 c31 = _mm512_setzero_ps();
            __m512 c41 = _mm512_setzero_ps();
            __m512 c51 = _mm512_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            __m512 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                a0 = _mm512_set1_ps(A[oa0]);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);
                a0 = _mm512_set1_ps(A[oa1]);
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                c11 = _mm512_fmadd_ps(a0, b1, c11);
                a0 = _mm512_set1_ps(A[oa2]);
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                c21 = _mm512_fmadd_ps(a0, b1, c21);
                a0 = _mm512_set1_ps(A[oa3]);
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                c31 = _mm512_fmadd_ps(a0, b1, c31);
                a0 = _mm512_set1_ps(A[oa4]);
                c40 = _mm512_fmadd_ps(a0, b0, c40);
                c41 = _mm512_fmadd_ps(a0, b1, c41);
                a0 = _mm512_set1_ps(A[oa5]);
                c50 = _mm512_fmadd_ps(a0, b0, c50);
                c51 = _mm512_fmadd_ps(a0, b1, c51);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51, mask);
        }

        void GemmKernel6x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            const size_t ob0 = ldb * 0;
            __m512 b0, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                a0 = _mm512_set1_ps(A[oa0]);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                a0 = _mm512_set1_ps(A[oa1]);
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                a0 = _mm512_set1_ps(A[oa2]);
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                a0 = _mm512_set1_ps(A[oa3]);
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                a0 = _mm512_set1_ps(A[oa4]);
                c40 = _mm512_fmadd_ps(a0, b0, c40);
                a0 = _mm512_set1_ps(A[oa5]);
                c50 = _mm512_fmadd_ps(a0, b0, c50);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            AddProduct(C + 0 * F, _alpha, c00, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50, mask);
        }

        void GemmKernel8x48nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c02 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c11 = _mm512_setzero_ps();
            __m512 c12 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c21 = _mm512_setzero_ps();
            __m512 c22 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c31 = _mm512_setzero_ps();
            __m512 c32 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps();
            __m512 c41 = _mm512_setzero_ps();
            __m512 c42 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps();
            __m512 c51 = _mm512_setzero_ps();
            __m512 c52 = _mm512_setzero_ps();
            __m512 c60 = _mm512_setzero_ps();
            __m512 c61 = _mm512_setzero_ps();
            __m512 c62 = _mm512_setzero_ps();
            __m512 c70 = _mm512_setzero_ps();
            __m512 c71 = _mm512_setzero_ps();
            __m512 c72 = _mm512_setzero_ps();
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
            __m512 b0, b1, b2, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                b2 = _mm512_loadu_ps(B + ob2);
                a0 = _mm512_set1_ps(A[oa0]);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);
                c02 = _mm512_fmadd_ps(a0, b2, c02);
                a0 = _mm512_set1_ps(A[oa1]);
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                c11 = _mm512_fmadd_ps(a0, b1, c11);
                c12 = _mm512_fmadd_ps(a0, b2, c12);
                a0 = _mm512_set1_ps(A[oa2]);
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                c21 = _mm512_fmadd_ps(a0, b1, c21);
                c22 = _mm512_fmadd_ps(a0, b2, c22);
                a0 = _mm512_set1_ps(A[oa3]);
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                c31 = _mm512_fmadd_ps(a0, b1, c31);
                c32 = _mm512_fmadd_ps(a0, b2, c32);
                a0 = _mm512_set1_ps(A[oa4]);
                c40 = _mm512_fmadd_ps(a0, b0, c40);
                c41 = _mm512_fmadd_ps(a0, b1, c41);
                c42 = _mm512_fmadd_ps(a0, b2, c42);
                a0 = _mm512_set1_ps(A[oa5]);
                c50 = _mm512_fmadd_ps(a0, b0, c50);
                c51 = _mm512_fmadd_ps(a0, b1, c51);
                c52 = _mm512_fmadd_ps(a0, b2, c52);
                a0 = _mm512_set1_ps(A[oa6]);
                c60 = _mm512_fmadd_ps(a0, b0, c60);
                c61 = _mm512_fmadd_ps(a0, b1, c61);
                c62 = _mm512_fmadd_ps(a0, b2, c62);
                a0 = _mm512_set1_ps(A[oa7]);
                c70 = _mm512_fmadd_ps(a0, b0, c70);
                c71 = _mm512_fmadd_ps(a0, b1, c71);
                c72 = _mm512_fmadd_ps(a0, b2, c72);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01);
            AddProduct(C + 2 * F, _alpha, c02, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11);
            AddProduct(C + 2 * F, _alpha, c12, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21);
            AddProduct(C + 2 * F, _alpha, c22, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31);
            AddProduct(C + 2 * F, _alpha, c32, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41);
            AddProduct(C + 2 * F, _alpha, c42, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51);
            AddProduct(C + 2 * F, _alpha, c52, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60);
            AddProduct(C + 1 * F, _alpha, c61);
            AddProduct(C + 2 * F, _alpha, c62, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70);
            AddProduct(C + 1 * F, _alpha, c71);
            AddProduct(C + 2 * F, _alpha, c72, mask);
        }

        void GemmKernel8x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc,  __mmask16 mask)
        {
            __m512 c00 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c11 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c21 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c31 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps();
            __m512 c41 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps();
            __m512 c51 = _mm512_setzero_ps();
            __m512 c60 = _mm512_setzero_ps();
            __m512 c61 = _mm512_setzero_ps();
            __m512 c70 = _mm512_setzero_ps();
            __m512 c71 = _mm512_setzero_ps();
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
            __m512 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                a0 = _mm512_set1_ps(A[oa0]);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);
                a0 = _mm512_set1_ps(A[oa1]);
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                c11 = _mm512_fmadd_ps(a0, b1, c11);
                a0 = _mm512_set1_ps(A[oa2]);
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                c21 = _mm512_fmadd_ps(a0, b1, c21);
                a0 = _mm512_set1_ps(A[oa3]);
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                c31 = _mm512_fmadd_ps(a0, b1, c31);
                a0 = _mm512_set1_ps(A[oa4]);
                c40 = _mm512_fmadd_ps(a0, b0, c40);
                c41 = _mm512_fmadd_ps(a0, b1, c41);
                a0 = _mm512_set1_ps(A[oa5]);
                c50 = _mm512_fmadd_ps(a0, b0, c50);
                c51 = _mm512_fmadd_ps(a0, b1, c51);
                a0 = _mm512_set1_ps(A[oa6]);
                c60 = _mm512_fmadd_ps(a0, b0, c60);
                c61 = _mm512_fmadd_ps(a0, b1, c61);
                a0 = _mm512_set1_ps(A[oa7]);
                c70 = _mm512_fmadd_ps(a0, b0, c70);
                c71 = _mm512_fmadd_ps(a0, b1, c71);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60);
            AddProduct(C + 1 * F, _alpha, c61, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70);
            AddProduct(C + 1 * F, _alpha, c71, mask);
        }

        void GemmKernel8x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps();
            __m512 c60 = _mm512_setzero_ps();
            __m512 c70 = _mm512_setzero_ps();
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
            __m512 b0, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                a0 = _mm512_set1_ps(A[oa0]);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                a0 = _mm512_set1_ps(A[oa1]);
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                a0 = _mm512_set1_ps(A[oa2]);
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                a0 = _mm512_set1_ps(A[oa3]);
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                a0 = _mm512_set1_ps(A[oa4]);
                c40 = _mm512_fmadd_ps(a0, b0, c40);
                a0 = _mm512_set1_ps(A[oa5]);
                c50 = _mm512_fmadd_ps(a0, b0, c50);
                a0 = _mm512_set1_ps(A[oa6]);
                c60 = _mm512_fmadd_ps(a0, b0, c60);
                a0 = _mm512_set1_ps(A[oa7]);
                c70 = _mm512_fmadd_ps(a0, b0, c70);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            AddProduct(C + 0 * F, _alpha, c00, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70, mask);
        }

        void GemmKernel9x48nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c02 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c11 = _mm512_setzero_ps();
            __m512 c12 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c21 = _mm512_setzero_ps();
            __m512 c22 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c31 = _mm512_setzero_ps();
            __m512 c32 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps();
            __m512 c41 = _mm512_setzero_ps();
            __m512 c42 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps();
            __m512 c51 = _mm512_setzero_ps();
            __m512 c52 = _mm512_setzero_ps();
            __m512 c60 = _mm512_setzero_ps();
            __m512 c61 = _mm512_setzero_ps();
            __m512 c62 = _mm512_setzero_ps();
            __m512 c70 = _mm512_setzero_ps();
            __m512 c71 = _mm512_setzero_ps();
            __m512 c72 = _mm512_setzero_ps();
            __m512 c80 = _mm512_setzero_ps();
            __m512 c81 = _mm512_setzero_ps();
            __m512 c82 = _mm512_setzero_ps();
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
            __m512 b0, b1, b2, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                b2 = _mm512_loadu_ps(B + ob2);
                a0 = _mm512_set1_ps(A[oa0]);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);
                c02 = _mm512_fmadd_ps(a0, b2, c02);
                a0 = _mm512_set1_ps(A[oa1]);
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                c11 = _mm512_fmadd_ps(a0, b1, c11);
                c12 = _mm512_fmadd_ps(a0, b2, c12);
                a0 = _mm512_set1_ps(A[oa2]);
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                c21 = _mm512_fmadd_ps(a0, b1, c21);
                c22 = _mm512_fmadd_ps(a0, b2, c22);
                a0 = _mm512_set1_ps(A[oa3]);
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                c31 = _mm512_fmadd_ps(a0, b1, c31);
                c32 = _mm512_fmadd_ps(a0, b2, c32);
                a0 = _mm512_set1_ps(A[oa4]);
                c40 = _mm512_fmadd_ps(a0, b0, c40);
                c41 = _mm512_fmadd_ps(a0, b1, c41);
                c42 = _mm512_fmadd_ps(a0, b2, c42);
                a0 = _mm512_set1_ps(A[oa5]);
                c50 = _mm512_fmadd_ps(a0, b0, c50);
                c51 = _mm512_fmadd_ps(a0, b1, c51);
                c52 = _mm512_fmadd_ps(a0, b2, c52);
                a0 = _mm512_set1_ps(A[oa6]);
                c60 = _mm512_fmadd_ps(a0, b0, c60);
                c61 = _mm512_fmadd_ps(a0, b1, c61);
                c62 = _mm512_fmadd_ps(a0, b2, c62);
                a0 = _mm512_set1_ps(A[oa7]);
                c70 = _mm512_fmadd_ps(a0, b0, c70);
                c71 = _mm512_fmadd_ps(a0, b1, c71);
                c72 = _mm512_fmadd_ps(a0, b2, c72);
                a0 = _mm512_set1_ps(A[oa8]);
                c80 = _mm512_fmadd_ps(a0, b0, c80);
                c81 = _mm512_fmadd_ps(a0, b1, c81);
                c82 = _mm512_fmadd_ps(a0, b2, c82);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01);
            AddProduct(C + 2 * F, _alpha, c02, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11);
            AddProduct(C + 2 * F, _alpha, c12, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21);
            AddProduct(C + 2 * F, _alpha, c22, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31);
            AddProduct(C + 2 * F, _alpha, c32, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41);
            AddProduct(C + 2 * F, _alpha, c42, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51);
            AddProduct(C + 2 * F, _alpha, c52, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60);
            AddProduct(C + 1 * F, _alpha, c61);
            AddProduct(C + 2 * F, _alpha, c62, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70);
            AddProduct(C + 1 * F, _alpha, c71);
            AddProduct(C + 2 * F, _alpha, c72, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c80);
            AddProduct(C + 1 * F, _alpha, c81);
            AddProduct(C + 2 * F, _alpha, c82, mask);
        }

        void GemmKernel9x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c11 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c21 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c31 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps();
            __m512 c41 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps();
            __m512 c51 = _mm512_setzero_ps();
            __m512 c60 = _mm512_setzero_ps();
            __m512 c61 = _mm512_setzero_ps();
            __m512 c70 = _mm512_setzero_ps();
            __m512 c71 = _mm512_setzero_ps();
            __m512 c80 = _mm512_setzero_ps();
            __m512 c81 = _mm512_setzero_ps();
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
            __m512 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                a0 = _mm512_set1_ps(A[oa0]);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);
                a0 = _mm512_set1_ps(A[oa1]);
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                c11 = _mm512_fmadd_ps(a0, b1, c11);
                a0 = _mm512_set1_ps(A[oa2]);
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                c21 = _mm512_fmadd_ps(a0, b1, c21);
                a0 = _mm512_set1_ps(A[oa3]);
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                c31 = _mm512_fmadd_ps(a0, b1, c31);
                a0 = _mm512_set1_ps(A[oa4]);
                c40 = _mm512_fmadd_ps(a0, b0, c40);
                c41 = _mm512_fmadd_ps(a0, b1, c41);
                a0 = _mm512_set1_ps(A[oa5]);
                c50 = _mm512_fmadd_ps(a0, b0, c50);
                c51 = _mm512_fmadd_ps(a0, b1, c51);
                a0 = _mm512_set1_ps(A[oa6]);
                c60 = _mm512_fmadd_ps(a0, b0, c60);
                c61 = _mm512_fmadd_ps(a0, b1, c61);
                a0 = _mm512_set1_ps(A[oa7]);
                c70 = _mm512_fmadd_ps(a0, b0, c70);
                c71 = _mm512_fmadd_ps(a0, b1, c71);
                a0 = _mm512_set1_ps(A[oa8]);
                c80 = _mm512_fmadd_ps(a0, b0, c80);
                c81 = _mm512_fmadd_ps(a0, b1, c81);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60);
            AddProduct(C + 1 * F, _alpha, c61, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70);
            AddProduct(C + 1 * F, _alpha, c71, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c80);
            AddProduct(C + 1 * F, _alpha, c81, mask);
        }

        void GemmKernel9x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps();
            __m512 c60 = _mm512_setzero_ps();
            __m512 c70 = _mm512_setzero_ps();
            __m512 c80 = _mm512_setzero_ps();
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
            __m512 b0, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                a0 = _mm512_set1_ps(A[oa0]);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                a0 = _mm512_set1_ps(A[oa1]);
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                a0 = _mm512_set1_ps(A[oa2]);
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                a0 = _mm512_set1_ps(A[oa3]);
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                a0 = _mm512_set1_ps(A[oa4]);
                c40 = _mm512_fmadd_ps(a0, b0, c40);
                a0 = _mm512_set1_ps(A[oa5]);
                c50 = _mm512_fmadd_ps(a0, b0, c50);
                a0 = _mm512_set1_ps(A[oa6]);
                c60 = _mm512_fmadd_ps(a0, b0, c60);
                a0 = _mm512_set1_ps(A[oa7]);
                c70 = _mm512_fmadd_ps(a0, b0, c70);
                a0 = _mm512_set1_ps(A[oa8]);
                c80 = _mm512_fmadd_ps(a0, b0, c80);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            AddProduct(C + 0 * F, _alpha, c00, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c80, mask);
        }

        void GemmKernel12x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c11 = _mm512_setzero_ps();
            __m512 c21 = _mm512_setzero_ps();
            __m512 c31 = _mm512_setzero_ps();
            __m512 c41 = _mm512_setzero_ps();
            __m512 c51 = _mm512_setzero_ps();
            __m512 c60 = _mm512_setzero_ps();
            __m512 c70 = _mm512_setzero_ps();
            __m512 c80 = _mm512_setzero_ps();
            __m512 c90 = _mm512_setzero_ps();
            __m512 cA0 = _mm512_setzero_ps();
            __m512 cB0 = _mm512_setzero_ps();
            __m512 c61 = _mm512_setzero_ps();
            __m512 c71 = _mm512_setzero_ps();
            __m512 c81 = _mm512_setzero_ps();
            __m512 c91 = _mm512_setzero_ps();
            __m512 cA1 = _mm512_setzero_ps();
            __m512 cB1 = _mm512_setzero_ps();
            const float * A0 = A, *A6 = A + 6 * lda;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 12 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            __m512 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                a0 = _mm512_set1_ps(A0[oa0]);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);
                a0 = _mm512_set1_ps(A0[oa1]);
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                c11 = _mm512_fmadd_ps(a0, b1, c11);
                a0 = _mm512_set1_ps(A0[oa2]);
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                c21 = _mm512_fmadd_ps(a0, b1, c21);
                a0 = _mm512_set1_ps(A0[oa3]);
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                c31 = _mm512_fmadd_ps(a0, b1, c31);
                a0 = _mm512_set1_ps(A0[oa4]);
                c40 = _mm512_fmadd_ps(a0, b0, c40);
                c41 = _mm512_fmadd_ps(a0, b1, c41);
                a0 = _mm512_set1_ps(A0[oa5]);
                c50 = _mm512_fmadd_ps(a0, b0, c50);
                c51 = _mm512_fmadd_ps(a0, b1, c51);
                a0 = _mm512_set1_ps(A6[oa0]);
                c60 = _mm512_fmadd_ps(a0, b0, c60);
                c61 = _mm512_fmadd_ps(a0, b1, c61);
                a0 = _mm512_set1_ps(A6[oa1]);
                c70 = _mm512_fmadd_ps(a0, b0, c70);
                c71 = _mm512_fmadd_ps(a0, b1, c71);
                a0 = _mm512_set1_ps(A6[oa2]);
                c80 = _mm512_fmadd_ps(a0, b0, c80);
                c81 = _mm512_fmadd_ps(a0, b1, c81);
                a0 = _mm512_set1_ps(A6[oa3]);
                c90 = _mm512_fmadd_ps(a0, b0, c90);
                c91 = _mm512_fmadd_ps(a0, b1, c91);
                a0 = _mm512_set1_ps(A6[oa4]);
                cA0 = _mm512_fmadd_ps(a0, b0, cA0);
                cA1 = _mm512_fmadd_ps(a0, b1, cA1);
                a0 = _mm512_set1_ps(A6[oa5]);
                cB0 = _mm512_fmadd_ps(a0, b0, cB0);
                cB1 = _mm512_fmadd_ps(a0, b1, cB1);
                B += sb;
                A0 += sa;
                A6 += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60);
            AddProduct(C + 1 * F, _alpha, c61, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70);
            AddProduct(C + 1 * F, _alpha, c71, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c80);
            AddProduct(C + 1 * F, _alpha, c81, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c90);
            AddProduct(C + 1 * F, _alpha, c91, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cA0);
            AddProduct(C + 1 * F, _alpha, cA1, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cB0);
            AddProduct(C + 1 * F, _alpha, cB1, mask);
        }

        void GemmKernel12x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps();
            __m512 c60 = _mm512_setzero_ps();
            __m512 c70 = _mm512_setzero_ps();
            __m512 c80 = _mm512_setzero_ps();
            __m512 c90 = _mm512_setzero_ps();
            __m512 cA0 = _mm512_setzero_ps();
            __m512 cB0 = _mm512_setzero_ps();
            const float * A0 = A, *A6 = A + 6 * lda;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 12 : 1;
            const size_t ob0 = ldb * 0;
            __m512 b0, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                a0 = _mm512_set1_ps(A0[oa0]);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                a0 = _mm512_set1_ps(A0[oa1]);
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                a0 = _mm512_set1_ps(A0[oa2]);
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                a0 = _mm512_set1_ps(A0[oa3]);
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                a0 = _mm512_set1_ps(A0[oa4]);
                c40 = _mm512_fmadd_ps(a0, b0, c40);
                a0 = _mm512_set1_ps(A0[oa5]);
                c50 = _mm512_fmadd_ps(a0, b0, c50);
                a0 = _mm512_set1_ps(A6[oa0]);
                c60 = _mm512_fmadd_ps(a0, b0, c60);
                a0 = _mm512_set1_ps(A6[oa1]);
                c70 = _mm512_fmadd_ps(a0, b0, c70);
                a0 = _mm512_set1_ps(A6[oa2]);
                c80 = _mm512_fmadd_ps(a0, b0, c80);
                a0 = _mm512_set1_ps(A6[oa3]);
                c90 = _mm512_fmadd_ps(a0, b0, c90);
                a0 = _mm512_set1_ps(A6[oa4]);
                cA0 = _mm512_fmadd_ps(a0, b0, cA0);
                a0 = _mm512_set1_ps(A6[oa5]);
                cB0 = _mm512_fmadd_ps(a0, b0, cB0);
                B += sb;
                A0 += sa;
                A6 += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            AddProduct(C + 0 * F, _alpha, c00, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c80, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c90, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cA0, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cB0, mask);
        }

        void GemmKernel14x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c11 = _mm512_setzero_ps();
            __m512 c21 = _mm512_setzero_ps();
            __m512 c31 = _mm512_setzero_ps();
            __m512 c41 = _mm512_setzero_ps();
            __m512 c51 = _mm512_setzero_ps();
            __m512 c60 = _mm512_setzero_ps();
            __m512 c70 = _mm512_setzero_ps();
            __m512 c80 = _mm512_setzero_ps();
            __m512 c90 = _mm512_setzero_ps();
            __m512 cA0 = _mm512_setzero_ps();
            __m512 cB0 = _mm512_setzero_ps();
            __m512 c61 = _mm512_setzero_ps();
            __m512 c71 = _mm512_setzero_ps();
            __m512 c81 = _mm512_setzero_ps();
            __m512 c91 = _mm512_setzero_ps();
            __m512 cA1 = _mm512_setzero_ps();
            __m512 cB1 = _mm512_setzero_ps();
            __m512 cC0 = _mm512_setzero_ps();
            __m512 cC1 = _mm512_setzero_ps();
            __m512 cD0 = _mm512_setzero_ps();
            __m512 cD1 = _mm512_setzero_ps();
            const float * A0 = A, *A7 = A + 7 * lda;
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
            __m512 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                a0 = _mm512_set1_ps(A0[oa0]);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);
                a0 = _mm512_set1_ps(A0[oa1]);
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                c11 = _mm512_fmadd_ps(a0, b1, c11);
                a0 = _mm512_set1_ps(A0[oa2]);
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                c21 = _mm512_fmadd_ps(a0, b1, c21);
                a0 = _mm512_set1_ps(A0[oa3]);
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                c31 = _mm512_fmadd_ps(a0, b1, c31);
                a0 = _mm512_set1_ps(A0[oa4]);
                c40 = _mm512_fmadd_ps(a0, b0, c40);
                c41 = _mm512_fmadd_ps(a0, b1, c41);
                a0 = _mm512_set1_ps(A0[oa5]);
                c50 = _mm512_fmadd_ps(a0, b0, c50);
                c51 = _mm512_fmadd_ps(a0, b1, c51);
                a0 = _mm512_set1_ps(A0[oa6]);
                c60 = _mm512_fmadd_ps(a0, b0, c60);
                c61 = _mm512_fmadd_ps(a0, b1, c61);
                a0 = _mm512_set1_ps(A7[oa0]);
                c70 = _mm512_fmadd_ps(a0, b0, c70);
                c71 = _mm512_fmadd_ps(a0, b1, c71);
                a0 = _mm512_set1_ps(A7[oa1]);
                c80 = _mm512_fmadd_ps(a0, b0, c80);
                c81 = _mm512_fmadd_ps(a0, b1, c81);
                a0 = _mm512_set1_ps(A7[oa2]);
                c90 = _mm512_fmadd_ps(a0, b0, c90);
                c91 = _mm512_fmadd_ps(a0, b1, c91);
                a0 = _mm512_set1_ps(A7[oa3]);
                cA0 = _mm512_fmadd_ps(a0, b0, cA0);
                cA1 = _mm512_fmadd_ps(a0, b1, cA1);
                a0 = _mm512_set1_ps(A7[oa4]);
                cB0 = _mm512_fmadd_ps(a0, b0, cB0);
                cB1 = _mm512_fmadd_ps(a0, b1, cB1);
                a0 = _mm512_set1_ps(A7[oa5]);
                cC0 = _mm512_fmadd_ps(a0, b0, cC0);
                cC1 = _mm512_fmadd_ps(a0, b1, cC1);
                a0 = _mm512_set1_ps(A7[oa6]);
                cD0 = _mm512_fmadd_ps(a0, b0, cD0);
                cD1 = _mm512_fmadd_ps(a0, b1, cD1);
                B += sb;
                A0 += sa;
                A7 += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c60);
            AddProduct(C + 1 * F, _alpha, c61, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c70);
            AddProduct(C + 1 * F, _alpha, c71, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c80);
            AddProduct(C + 1 * F, _alpha, c81, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c90);
            AddProduct(C + 1 * F, _alpha, c91, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cA0);
            AddProduct(C + 1 * F, _alpha, cA1, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cB0);
            AddProduct(C + 1 * F, _alpha, cB1, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cC0);
            AddProduct(C + 1 * F, _alpha, cC1, mask);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, cD0);
            AddProduct(C + 1 * F, _alpha, cD1, mask);
        }

        void GemmKernel14x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c0 = _mm512_setzero_ps();
            __m512 c1 = _mm512_setzero_ps();
            __m512 c2 = _mm512_setzero_ps();
            __m512 c3 = _mm512_setzero_ps();
            __m512 c4 = _mm512_setzero_ps();
            __m512 c5 = _mm512_setzero_ps();
            __m512 c6 = _mm512_setzero_ps();
            __m512 c7 = _mm512_setzero_ps();
            __m512 c8 = _mm512_setzero_ps();
            __m512 c9 = _mm512_setzero_ps();
            __m512 cA = _mm512_setzero_ps();
            __m512 cB = _mm512_setzero_ps();
            __m512 cC = _mm512_setzero_ps();
            __m512 cD = _mm512_setzero_ps();
            const float * A0 = A, * A7 = A + 7*lda;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t oa6 = lda * 6;
            const size_t sa = lda == 1 ? 14 : 1;
            const size_t ob0 = ldb * 0;
            __m512 b0, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                a0 = _mm512_set1_ps(A0[oa0]);
                c0 = _mm512_fmadd_ps(a0, b0, c0);
                a0 = _mm512_set1_ps(A0[oa1]);
                c1 = _mm512_fmadd_ps(a0, b0, c1);
                a0 = _mm512_set1_ps(A0[oa2]);
                c2 = _mm512_fmadd_ps(a0, b0, c2);
                a0 = _mm512_set1_ps(A0[oa3]);
                c3 = _mm512_fmadd_ps(a0, b0, c3);
                a0 = _mm512_set1_ps(A0[oa4]);
                c4 = _mm512_fmadd_ps(a0, b0, c4);
                a0 = _mm512_set1_ps(A0[oa5]);
                c5 = _mm512_fmadd_ps(a0, b0, c5);
                a0 = _mm512_set1_ps(A0[oa6]);
                c6 = _mm512_fmadd_ps(a0, b0, c6);
                a0 = _mm512_set1_ps(A7[oa0]);
                c7 = _mm512_fmadd_ps(a0, b0, c7);
                a0 = _mm512_set1_ps(A7[oa1]);
                c8 = _mm512_fmadd_ps(a0, b0, c8);
                a0 = _mm512_set1_ps(A7[oa2]);
                c9 = _mm512_fmadd_ps(a0, b0, c9);
                a0 = _mm512_set1_ps(A7[oa3]);
                cA = _mm512_fmadd_ps(a0, b0, cA);
                a0 = _mm512_set1_ps(A7[oa4]);
                cB = _mm512_fmadd_ps(a0, b0, cB);
                a0 = _mm512_set1_ps(A7[oa5]);
                cC = _mm512_fmadd_ps(a0, b0, cC);
                a0 = _mm512_set1_ps(A7[oa6]);
                cD = _mm512_fmadd_ps(a0, b0, cD);
                B += sb;
                A0 += sa;
                A7 += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            AddProduct(C, _alpha, c0, mask);
            C += ldc;
            AddProduct(C, _alpha, c1, mask);
            C += ldc;
            AddProduct(C, _alpha, c2, mask);
            C += ldc;
            AddProduct(C, _alpha, c3, mask);
            C += ldc;
            AddProduct(C, _alpha, c4, mask);
            C += ldc;
            AddProduct(C, _alpha, c5, mask);
            C += ldc;
            AddProduct(C, _alpha, c6, mask);
            C += ldc;
            AddProduct(C, _alpha, c7, mask);
            C += ldc;
            AddProduct(C, _alpha, c8, mask);
            C += ldc;
            AddProduct(C, _alpha, c9, mask);
            C += ldc;
            AddProduct(C, _alpha, cA, mask);
            C += ldc;
            AddProduct(C, _alpha, cB, mask);
            C += ldc;
            AddProduct(C, _alpha, cC, mask);
            C += ldc;
            AddProduct(C, _alpha, cD, mask);
        }

        void GemmKernelMx48nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
#if SIMD_ZMM_COUNT == 32
            __m512 c[9][3];
            size_t oa[9];
#else
            __m512 c[4][3];
            size_t oa[4];
#endif
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = _mm512_setzero_ps();
                c[i][1] = _mm512_setzero_ps();
                c[i][2] = _mm512_setzero_ps();
                oa[i] = lda * i;
            }
            __m512 b0, b1, b2, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                b2 = _mm512_loadu_ps(B + ob2);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = _mm512_set1_ps(A[oa[i]]);
                    c[i][0] = _mm512_add_ps(_mm512_mul_ps(b0, a0), c[i][0]);
                    c[i][1] = _mm512_add_ps(_mm512_mul_ps(b1, a0), c[i][1]);
                    c[i][2] = _mm512_add_ps(_mm512_mul_ps(b2, a0), c[i][2]);
                }
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            for (size_t i = 0; i < M; ++i)
            {
                AddProduct(C + 0 * F, _alpha, c[i][0]);
                AddProduct(C + 1 * F, _alpha, c[i][1]);
                AddProduct(C + 2 * F, _alpha, c[i][2], mask);
                C += ldc;
            }
        }

        void GemmKernelMx32nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
#if SIMD_ZMM_COUNT == 32
            __m512 c[14][2];
            size_t oa[14];
#else
            __m512 c[6][2];
            size_t oa[6];
#endif
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = _mm512_setzero_ps();
                c[i][1] = _mm512_setzero_ps();
                oa[i] = lda * i;
            }
            __m512 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = _mm512_set1_ps(A[oa[i]]);
                    c[i][0] = _mm512_fmadd_ps(b0, a0, c[i][0]);
                    c[i][1] = _mm512_fmadd_ps(b1, a0, c[i][1]);
                }
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            for (size_t i = 0; i < M; ++i)
            {
                AddProduct(C + 0 * F, _alpha, c[i][0]);
                AddProduct(C + 1 * F, _alpha, c[i][1], mask);
                C += ldc;
            }
        }

        void GemmKernelMx16nn(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            //SIMD_PERF_BEG(Simd::ToStr(M) + "-" + Simd::ToStr(N) + "-" + Simd::ToStr(K));

#if SIMD_ZMM_COUNT == 32
            __m512 c[14];
            size_t oa[14];
#elif SIMD_ZMM_COUNT == 16
            __m512 c[6];
            size_t oa[6];
#else
            __m512 c[4];
            size_t oa[4];
#endif
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            for (size_t i = 0; i < M; ++i)
            {
                c[i] = _mm512_setzero_ps();
                oa[i] = lda * i;
            }
            __m512 b0, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = _mm512_set1_ps(A[oa[i]]);
                    c[i] = _mm512_fmadd_ps(b0, a0, c[i]);
                }
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            for (size_t i = 0; i < M; ++i)
                AddProduct(C + i * ldc, _alpha, c[i], mask);
        }

        template<int M> void GemmKernelMx48nnT(size_t, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c01, c02, c03, c04, c05, c06, c07, c08, c10, c11, c12, c13, c14, c15, c16, c17, c18, c20, c21, c22, c23, c24, c25, c26, c27, c28, b0, b1, b2, a0;
            if (M > 0x0) c00 = _mm512_setzero_ps(), c10 = _mm512_setzero_ps(), c20 = _mm512_setzero_ps();
            if (M > 0x1) c01 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
            if (M > 0x2) c02 = _mm512_setzero_ps(), c12 = _mm512_setzero_ps(), c22 = _mm512_setzero_ps();
            if (M > 0x3) c03 = _mm512_setzero_ps(), c13 = _mm512_setzero_ps(), c23 = _mm512_setzero_ps();
            if (M > 0x4) c04 = _mm512_setzero_ps(), c14 = _mm512_setzero_ps(), c24 = _mm512_setzero_ps();
            if (M > 0x5) c05 = _mm512_setzero_ps(), c15 = _mm512_setzero_ps(), c25 = _mm512_setzero_ps();
            if (M > 0x6) c06 = _mm512_setzero_ps(), c16 = _mm512_setzero_ps(), c26 = _mm512_setzero_ps();
            if (M > 0x7) c07 = _mm512_setzero_ps(), c17 = _mm512_setzero_ps(), c27 = _mm512_setzero_ps();
            if (M > 0x8) c08 = _mm512_setzero_ps(), c18 = _mm512_setzero_ps(), c28 = _mm512_setzero_ps();
            const float * A0 = A, *A5 = A + 5 * lda;
            size_t oa0, oa1, oa2, oa3, oa4;
            if (M > 0) oa0 = lda * 0;
            if (M > 1) oa1 = lda * 1;
            if (M > 2) oa2 = lda * 2;
            if (M > 3) oa3 = lda * 3;
            if (M > 4) oa4 = lda * 4;
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                b2 = _mm512_loadu_ps(B + ob2);
                if (M > 0x0) a0 = _mm512_set1_ps(A0[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c10 = _mm512_fmadd_ps(a0, b1, c10), c20 = _mm512_fmadd_ps(a0, b2, c20);
                if (M > 0x1) a0 = _mm512_set1_ps(A0[oa1]), c01 = _mm512_fmadd_ps(a0, b0, c01), c11 = _mm512_fmadd_ps(a0, b1, c11), c21 = _mm512_fmadd_ps(a0, b2, c21);
                if (M > 0x2) a0 = _mm512_set1_ps(A0[oa2]), c02 = _mm512_fmadd_ps(a0, b0, c02), c12 = _mm512_fmadd_ps(a0, b1, c12), c22 = _mm512_fmadd_ps(a0, b2, c22);
                if (M > 0x3) a0 = _mm512_set1_ps(A0[oa3]), c03 = _mm512_fmadd_ps(a0, b0, c03), c13 = _mm512_fmadd_ps(a0, b1, c13), c23 = _mm512_fmadd_ps(a0, b2, c23);
                if (M > 0x4) a0 = _mm512_set1_ps(A0[oa4]), c04 = _mm512_fmadd_ps(a0, b0, c04), c14 = _mm512_fmadd_ps(a0, b1, c14), c24 = _mm512_fmadd_ps(a0, b2, c24);
                if (M > 0x5) a0 = _mm512_set1_ps(A5[oa0]), c05 = _mm512_fmadd_ps(a0, b0, c05), c15 = _mm512_fmadd_ps(a0, b1, c15), c25 = _mm512_fmadd_ps(a0, b2, c25);
                if (M > 0x6) a0 = _mm512_set1_ps(A5[oa1]), c06 = _mm512_fmadd_ps(a0, b0, c06), c16 = _mm512_fmadd_ps(a0, b1, c16), c26 = _mm512_fmadd_ps(a0, b2, c26);
                if (M > 0x7) a0 = _mm512_set1_ps(A5[oa2]), c07 = _mm512_fmadd_ps(a0, b0, c07), c17 = _mm512_fmadd_ps(a0, b1, c17), c27 = _mm512_fmadd_ps(a0, b2, c27);
                if (M > 0x8) a0 = _mm512_set1_ps(A5[oa3]), c08 = _mm512_fmadd_ps(a0, b0, c08), c18 = _mm512_fmadd_ps(a0, b1, c18), c28 = _mm512_fmadd_ps(a0, b2, c28);
                B += sb;
                A0 += sa;
                A5 += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            if (M > 0x0) AddProduct(C + 0 * F, _alpha, c00), AddProduct(C + 1 * F, _alpha, c10), AddProduct(C + 2 * F, _alpha, c20, mask), C += ldc;
            if (M > 0x1) AddProduct(C + 0 * F, _alpha, c01), AddProduct(C + 1 * F, _alpha, c11), AddProduct(C + 2 * F, _alpha, c21, mask), C += ldc;
            if (M > 0x2) AddProduct(C + 0 * F, _alpha, c02), AddProduct(C + 1 * F, _alpha, c12), AddProduct(C + 2 * F, _alpha, c22, mask), C += ldc;
            if (M > 0x3) AddProduct(C + 0 * F, _alpha, c03), AddProduct(C + 1 * F, _alpha, c13), AddProduct(C + 2 * F, _alpha, c23, mask), C += ldc;
            if (M > 0x4) AddProduct(C + 0 * F, _alpha, c04), AddProduct(C + 1 * F, _alpha, c14), AddProduct(C + 2 * F, _alpha, c24, mask), C += ldc;
            if (M > 0x5) AddProduct(C + 0 * F, _alpha, c05), AddProduct(C + 1 * F, _alpha, c15), AddProduct(C + 2 * F, _alpha, c25, mask), C += ldc;
            if (M > 0x6) AddProduct(C + 0 * F, _alpha, c06), AddProduct(C + 1 * F, _alpha, c16), AddProduct(C + 2 * F, _alpha, c26, mask), C += ldc;
            if (M > 0x7) AddProduct(C + 0 * F, _alpha, c07), AddProduct(C + 1 * F, _alpha, c17), AddProduct(C + 2 * F, _alpha, c27, mask), C += ldc;
            if (M > 0x8) AddProduct(C + 0 * F, _alpha, c08), AddProduct(C + 1 * F, _alpha, c18), AddProduct(C + 2 * F, _alpha, c28, mask), C += ldc;
        }

        template<int M> void GemmKernelMx32nnT(size_t, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c1A, c1B, c1C, c1D, b0, b1, a0;
            if (M > 0x0) c00 = _mm512_setzero_ps(), c10 = _mm512_setzero_ps();
            if (M > 0x1) c01 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
            if (M > 0x2) c02 = _mm512_setzero_ps(), c12 = _mm512_setzero_ps();
            if (M > 0x3) c03 = _mm512_setzero_ps(), c13 = _mm512_setzero_ps();
            if (M > 0x4) c04 = _mm512_setzero_ps(), c14 = _mm512_setzero_ps();
            if (M > 0x5) c05 = _mm512_setzero_ps(), c15 = _mm512_setzero_ps();
            if (M > 0x6) c06 = _mm512_setzero_ps(), c16 = _mm512_setzero_ps();
            if (M > 0x7) c07 = _mm512_setzero_ps(), c17 = _mm512_setzero_ps();
            if (M > 0x8) c08 = _mm512_setzero_ps(), c18 = _mm512_setzero_ps();
            if (M > 0x9) c09 = _mm512_setzero_ps(), c19 = _mm512_setzero_ps();
            if (M > 0xA) c0A = _mm512_setzero_ps(), c1A = _mm512_setzero_ps();
            if (M > 0xB) c0B = _mm512_setzero_ps(), c1B = _mm512_setzero_ps();
            if (M > 0xC) c0C = _mm512_setzero_ps(), c1C = _mm512_setzero_ps();
            if (M > 0xD) c0D = _mm512_setzero_ps(), c1D = _mm512_setzero_ps();
            const float * A0 = A, *A7 = A + 7 * lda;
            size_t oa0, oa1, oa2, oa3, oa4, oa5, oa6;
            if (M > 0) oa0 = lda * 0;
            if (M > 1) oa1 = lda * 1;
            if (M > 2) oa2 = lda * 2;
            if (M > 3) oa3 = lda * 3;
            if (M > 4) oa4 = lda * 4;
            if (M > 5) oa5 = lda * 5;
            if (M > 6) oa6 = lda * 6;
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                if (M > 0x0) a0 = _mm512_set1_ps(A0[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c10 = _mm512_fmadd_ps(a0, b1, c10);
                if (M > 0x1) a0 = _mm512_set1_ps(A0[oa1]), c01 = _mm512_fmadd_ps(a0, b0, c01), c11 = _mm512_fmadd_ps(a0, b1, c11);
                if (M > 0x2) a0 = _mm512_set1_ps(A0[oa2]), c02 = _mm512_fmadd_ps(a0, b0, c02), c12 = _mm512_fmadd_ps(a0, b1, c12);
                if (M > 0x3) a0 = _mm512_set1_ps(A0[oa3]), c03 = _mm512_fmadd_ps(a0, b0, c03), c13 = _mm512_fmadd_ps(a0, b1, c13);
                if (M > 0x4) a0 = _mm512_set1_ps(A0[oa4]), c04 = _mm512_fmadd_ps(a0, b0, c04), c14 = _mm512_fmadd_ps(a0, b1, c14);
                if (M > 0x5) a0 = _mm512_set1_ps(A0[oa5]), c05 = _mm512_fmadd_ps(a0, b0, c05), c15 = _mm512_fmadd_ps(a0, b1, c15);
                if (M > 0x6) a0 = _mm512_set1_ps(A0[oa6]), c06 = _mm512_fmadd_ps(a0, b0, c06), c16 = _mm512_fmadd_ps(a0, b1, c16);
                if (M > 0x7) a0 = _mm512_set1_ps(A7[oa0]), c07 = _mm512_fmadd_ps(a0, b0, c07), c17 = _mm512_fmadd_ps(a0, b1, c17);
                if (M > 0x8) a0 = _mm512_set1_ps(A7[oa1]), c08 = _mm512_fmadd_ps(a0, b0, c08), c18 = _mm512_fmadd_ps(a0, b1, c18);
                if (M > 0x9) a0 = _mm512_set1_ps(A7[oa2]), c09 = _mm512_fmadd_ps(a0, b0, c09), c19 = _mm512_fmadd_ps(a0, b1, c19);
                if (M > 0xA) a0 = _mm512_set1_ps(A7[oa3]), c0A = _mm512_fmadd_ps(a0, b0, c0A), c1A = _mm512_fmadd_ps(a0, b1, c1A);
                if (M > 0xB) a0 = _mm512_set1_ps(A7[oa4]), c0B = _mm512_fmadd_ps(a0, b0, c0B), c1B = _mm512_fmadd_ps(a0, b1, c1B);
                if (M > 0xC) a0 = _mm512_set1_ps(A7[oa5]), c0C = _mm512_fmadd_ps(a0, b0, c0C), c1C = _mm512_fmadd_ps(a0, b1, c1C);
                if (M > 0xD) a0 = _mm512_set1_ps(A7[oa6]), c0D = _mm512_fmadd_ps(a0, b0, c0D), c1D = _mm512_fmadd_ps(a0, b1, c1D);
                B += sb;
                A0 += sa;
                A7 += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            if (M > 0x0) AddProduct(C + 0 * F, _alpha, c00), AddProduct(C + 1 * F, _alpha, c10, mask), C += ldc;
            if (M > 0x1) AddProduct(C + 0 * F, _alpha, c01), AddProduct(C + 1 * F, _alpha, c11, mask), C += ldc;
            if (M > 0x2) AddProduct(C + 0 * F, _alpha, c02), AddProduct(C + 1 * F, _alpha, c12, mask), C += ldc;
            if (M > 0x3) AddProduct(C + 0 * F, _alpha, c03), AddProduct(C + 1 * F, _alpha, c13, mask), C += ldc;
            if (M > 0x4) AddProduct(C + 0 * F, _alpha, c04), AddProduct(C + 1 * F, _alpha, c14, mask), C += ldc;
            if (M > 0x5) AddProduct(C + 0 * F, _alpha, c05), AddProduct(C + 1 * F, _alpha, c15, mask), C += ldc;
            if (M > 0x6) AddProduct(C + 0 * F, _alpha, c06), AddProduct(C + 1 * F, _alpha, c16, mask), C += ldc;
            if (M > 0x7) AddProduct(C + 0 * F, _alpha, c07), AddProduct(C + 1 * F, _alpha, c17, mask), C += ldc;
            if (M > 0x8) AddProduct(C + 0 * F, _alpha, c08), AddProduct(C + 1 * F, _alpha, c18, mask), C += ldc;
            if (M > 0x9) AddProduct(C + 0 * F, _alpha, c09), AddProduct(C + 1 * F, _alpha, c19, mask), C += ldc;
            if (M > 0xA) AddProduct(C + 0 * F, _alpha, c0A), AddProduct(C + 1 * F, _alpha, c1A, mask), C += ldc;
            if (M > 0xB) AddProduct(C + 0 * F, _alpha, c0B), AddProduct(C + 1 * F, _alpha, c1B, mask), C += ldc;
            if (M > 0xC) AddProduct(C + 0 * F, _alpha, c0C), AddProduct(C + 1 * F, _alpha, c1C, mask), C += ldc;
            if (M > 0xD) AddProduct(C + 0 * F, _alpha, c0D), AddProduct(C + 1 * F, _alpha, c1D, mask), C += ldc;
        }

        template<int M> void GemmKernelMx16nnT(size_t, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c01, c02, c03, c04, c05, c06, c07, c08, c09, c0A, c0B, c0C, c0D, b0, a0;
            if (M > 0x0) c00 = _mm512_setzero_ps();
            if (M > 0x1) c01 = _mm512_setzero_ps();
            if (M > 0x2) c02 = _mm512_setzero_ps();
            if (M > 0x3) c03 = _mm512_setzero_ps();
            if (M > 0x4) c04 = _mm512_setzero_ps();
            if (M > 0x5) c05 = _mm512_setzero_ps();
            if (M > 0x6) c06 = _mm512_setzero_ps();
            if (M > 0x7) c07 = _mm512_setzero_ps();
            if (M > 0x8) c08 = _mm512_setzero_ps();
            if (M > 0x9) c09 = _mm512_setzero_ps();
            if (M > 0xA) c0A = _mm512_setzero_ps();
            if (M > 0xB) c0B = _mm512_setzero_ps();
            if (M > 0xC) c0C = _mm512_setzero_ps();
            if (M > 0xD) c0D = _mm512_setzero_ps();
            const float * A0 = A, *A7 = A + 7 * lda;
            size_t oa0, oa1, oa2, oa3, oa4, oa5, oa6;
            if (M > 0) oa0 = lda * 0;
            if (M > 1) oa1 = lda * 1;
            if (M > 2) oa2 = lda * 2;
            if (M > 3) oa3 = lda * 3;
            if (M > 4) oa4 = lda * 4;
            if (M > 5) oa5 = lda * 5;
            if (M > 6) oa6 = lda * 6;
            const size_t sa = lda == 1 ? M : 1;
            const size_t ob0 = ldb * 0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm512_loadu_ps(B + ob0);
                if (M > 0x0) a0 = _mm512_set1_ps(A0[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00);
                if (M > 0x1) a0 = _mm512_set1_ps(A0[oa1]), c01 = _mm512_fmadd_ps(a0, b0, c01);
                if (M > 0x2) a0 = _mm512_set1_ps(A0[oa2]), c02 = _mm512_fmadd_ps(a0, b0, c02);
                if (M > 0x3) a0 = _mm512_set1_ps(A0[oa3]), c03 = _mm512_fmadd_ps(a0, b0, c03);
                if (M > 0x4) a0 = _mm512_set1_ps(A0[oa4]), c04 = _mm512_fmadd_ps(a0, b0, c04);
                if (M > 0x5) a0 = _mm512_set1_ps(A0[oa5]), c05 = _mm512_fmadd_ps(a0, b0, c05);
                if (M > 0x6) a0 = _mm512_set1_ps(A0[oa6]), c06 = _mm512_fmadd_ps(a0, b0, c06);
                if (M > 0x7) a0 = _mm512_set1_ps(A7[oa0]), c07 = _mm512_fmadd_ps(a0, b0, c07);
                if (M > 0x8) a0 = _mm512_set1_ps(A7[oa1]), c08 = _mm512_fmadd_ps(a0, b0, c08);
                if (M > 0x9) a0 = _mm512_set1_ps(A7[oa2]), c09 = _mm512_fmadd_ps(a0, b0, c09);
                if (M > 0xA) a0 = _mm512_set1_ps(A7[oa3]), c0A = _mm512_fmadd_ps(a0, b0, c0A);
                if (M > 0xB) a0 = _mm512_set1_ps(A7[oa4]), c0B = _mm512_fmadd_ps(a0, b0, c0B);
                if (M > 0xC) a0 = _mm512_set1_ps(A7[oa5]), c0C = _mm512_fmadd_ps(a0, b0, c0C);
                if (M > 0xD) a0 = _mm512_set1_ps(A7[oa6]), c0D = _mm512_fmadd_ps(a0, b0, c0D);
                B += sb;
                A0 += sa;
                A7 += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            if (M > 0x0) AddProduct(C, _alpha, c00, mask), C += ldc;
            if (M > 0x1) AddProduct(C, _alpha, c01, mask), C += ldc;
            if (M > 0x2) AddProduct(C, _alpha, c02, mask), C += ldc;
            if (M > 0x3) AddProduct(C, _alpha, c03, mask), C += ldc;
            if (M > 0x4) AddProduct(C, _alpha, c04, mask), C += ldc;
            if (M > 0x5) AddProduct(C, _alpha, c05, mask), C += ldc;
            if (M > 0x6) AddProduct(C, _alpha, c06, mask), C += ldc;
            if (M > 0x7) AddProduct(C, _alpha, c07, mask), C += ldc;
            if (M > 0x8) AddProduct(C, _alpha, c08, mask), C += ldc;
            if (M > 0x9) AddProduct(C, _alpha, c09, mask), C += ldc;
            if (M > 0xA) AddProduct(C, _alpha, c0A, mask), C += ldc;
            if (M > 0xB) AddProduct(C, _alpha, c0B, mask), C += ldc;
            if (M > 0xC) AddProduct(C, _alpha, c0C, mask), C += ldc;
            if (M > 0xD) AddProduct(C, _alpha, c0D, mask), C += ldc;
        }

        SIMD_INLINE Simd::GemmNN<float, __mmask16>::Tail GetGemmTail(size_t M, size_t N)
        {
            if (N <= 16)
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
                case 7: return GemmKernelMx16nnT<7>;
                case 8: return GemmKernelMx16nnT<8>;
                case 9: return GemmKernelMx16nnT<9>;
                case 10: return GemmKernelMx16nnT<10>;
                case 11: return GemmKernelMx16nnT<11>;
                case 12: return GemmKernelMx16nnT<12>;
                case 13: return GemmKernelMx16nnT<13>;
                case 14: return GemmKernelMx16nnT<14>;
                }
            }
            else if (N <= 32)
            {
                switch (M)
                {
                case 0: return GemmKernelMx32nnT<0>;
                case 1: return GemmKernelMx32nnT<1>;
                case 2: return GemmKernelMx32nnT<2>;
                case 3: return GemmKernelMx32nnT<3>;
                case 4: return GemmKernelMx32nnT<4>;
                case 5: return GemmKernelMx32nnT<5>;
                case 6: return GemmKernelMx32nnT<6>;
                case 7: return GemmKernelMx32nnT<7>;
                case 8: return GemmKernelMx32nnT<8>;
                case 9: return GemmKernelMx32nnT<9>;
                case 10: return GemmKernelMx32nnT<10>;
                case 11: return GemmKernelMx32nnT<11>;
                case 12: return GemmKernelMx32nnT<12>;
                case 13: return GemmKernelMx32nnT<13>;
                case 14: return GemmKernelMx32nnT<14>;
                }
            }
            else if (N <= 48)
            {
                switch (M)
                {
                case 0: return GemmKernelMx48nnT<0>;
                case 1: return GemmKernelMx48nnT<1>;
                case 2: return GemmKernelMx48nnT<2>;
                case 3: return GemmKernelMx48nnT<3>;
                case 4: return GemmKernelMx48nnT<4>;
                case 5: return GemmKernelMx48nnT<5>;
                case 6: return GemmKernelMx48nnT<6>;
                case 7: return GemmKernelMx48nnT<7>;
                case 8: return GemmKernelMx48nnT<8>;
                }
            }
            assert(0);
            return NULL;
        }

        void GemmPackA(const float * src, size_t stride, size_t M, size_t K, size_t cell, float * dst)
        {
            size_t K4 = AlignLo(K, 4), K8 = AlignLo(K, 8);
            for (size_t i = 0; i < M; i += cell)
            {
                size_t m = Simd::Min(cell, M - i), k = 0;
                if (cell == 4 && m == 4)
                {
                    for (; k < K8; k += 8)
                    {
                        const float * ps = src + k;
                        __m256 s0 = _mm256_loadu_ps(ps + 0 * K);
                        __m256 s1 = _mm256_loadu_ps(ps + 1 * K);
                        __m256 s2 = _mm256_loadu_ps(ps + 2 * K);
                        __m256 s3 = _mm256_loadu_ps(ps + 3 * K);
                        __m256 s00 = _mm256_unpacklo_ps(s0, s2);
                        __m256 s01 = _mm256_unpacklo_ps(s1, s3);
                        __m256 s10 = _mm256_unpackhi_ps(s0, s2);
                        __m256 s11 = _mm256_unpackhi_ps(s1, s3);
                        __m256 d0 = _mm256_unpacklo_ps(s00, s01);
                        __m256 d1 = _mm256_unpackhi_ps(s00, s01);
                        __m256 d2 = _mm256_unpacklo_ps(s10, s11);
                        __m256 d3 = _mm256_unpackhi_ps(s10, s11);
                        _mm256_storeu_ps(dst + 0, _mm256_permute2f128_ps(d0, d1, 0x20));
                        _mm256_storeu_ps(dst + 8, _mm256_permute2f128_ps(d2, d3, 0x20));
                        _mm256_storeu_ps(dst + 16, _mm256_permute2f128_ps(d0, d1, 0x31));
                        _mm256_storeu_ps(dst + 24, _mm256_permute2f128_ps(d2, d3, 0x31));
                        dst += 32;
                    };
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
                if (microN == 1 * F)
                {
                    __mmask16 mask0 = TailMask16(n - 0 * F);
                    for (size_t k = 0; k < K; ++k)
                    {
                        const float * b = B + k * ldb;
                        _mm512_storeu_ps(pB + 0 * F, _mm512_maskz_loadu_ps(mask0, b + 0 * F));
                        pB += microN;
                    }
                }
                else if (microN == 2 * F)
                {
                    __mmask16 mask0 = TailMask16(n - 0 * F);
                    __mmask16 mask1 = TailMask16(n - 1 * F);
                    for (size_t k = 0; k < K; ++k)
                    {
                        const float * b = B + k * ldb;
                        _mm512_storeu_ps(pB + 0 * F, _mm512_maskz_loadu_ps(mask0, b + 0 * F));
                        _mm512_storeu_ps(pB + 1 * F, _mm512_maskz_loadu_ps(mask1, b + 1 * F));
                        pB += microN;
                    }
                }
                else if (microN == 3 * F)
                {
                    __mmask16 mask0 = TailMask16(n - 0 * F);
                    __mmask16 mask1 = TailMask16(n - 1 * F);
                    __mmask16 mask2 = TailMask16(n - 2 * F);
                    for (size_t k = 0; k < K; ++k)
                    {
                        const float * b = B + k * ldb;
                        _mm512_storeu_ps(pB + 0 * F, _mm512_maskz_loadu_ps(mask0, b + 0 * F));
                        _mm512_storeu_ps(pB + 1 * F, _mm512_maskz_loadu_ps(mask1, b + 1 * F));
                        _mm512_storeu_ps(pB + 2 * F, _mm512_maskz_loadu_ps(mask2, b + 2 * F));
                        pB += microN;
                    }
                }
                else
                {
                    for (size_t k = 0; k < K; ++k)
                    {
                        const float * b = B + k * ldb;
                        size_t c = 0;
                        for (; c < n; ++c)
                            *(pB++) = *(b++);
                        for (; c < microN; ++c)
                            *(pB++) = 0;
                    }
                }
                B += microN;
            }
        }

        SIMD_INLINE void ScaleC(float * ptr, __m512 value, __mmask16 mask = -1)
        {
            _mm512_mask_storeu_ps(ptr, mask, _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, ptr), value));
        }

        void GemmScaleC(size_t M, size_t N, float value, float * C, size_t ldc)
        {
            size_t NQF = AlignLo(N, QF);
            size_t NF = AlignLo(N, F);
            __m512 _value = _mm512_set1_ps(value);
            __mmask16 tail = TailMask16(N - NF);
            for (size_t i = 0; i < M; ++i)
            {
                size_t j = 0;
                for (; j < NQF; j += QF)
                {
                    ScaleC(C + j + F * 0, _value);
                    ScaleC(C + j + F * 1, _value);
                    ScaleC(C + j + F * 2, _value);
                    ScaleC(C + j + F * 3, _value);
                }
                for (; j < NF; j += F)
                    ScaleC(C + j, _value);
                if (j < N)
                    ScaleC(C + j, _value, tail);
                C += ldc;
            }
        }

        void Gemm32fNN(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            SIMD_PERF_BEGF(Simd::ToStr(M) + "-" + Simd::ToStr(N) + "-" + Simd::ToStr(K), M*N*K * 2);

            typedef Simd::GemmNN<float, __mmask16> GemmNN;
            GemmNN::Main kernelMM, kernelMT;
            GemmNN::Tail kernelTM, kernelTT;
            size_t microM, microN;
            if (N <= 8)
            {
                Avx2::Gemm32fNN(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
                return;
            }
#if SIMD_ZMM_COUNT == 32 
            if (N < K || M * 8 < N)
            {
                microM = 14;
                microN = 32;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel14x32nn;
                kernelMT = tail > F ? GemmKernel14x32nn : GemmKernel14x16nn;
                kernelTM = GemmKernelMx32nn;
                kernelTT = tail > F ? GemmKernelMx32nn : GemmKernelMx16nn;
            }
            else
            {
                microM = 9;
                microN = 48;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel9x48nn;
                kernelMT = tail > DF ? GemmKernel9x48nn : (tail > F ? GemmKernel9x32nn : GemmKernel9x16nn);
                kernelTM = GemmKernelMx48nn;
                kernelTT = tail > DF ? GemmKernelMx48nn : (tail > F ? GemmKernelMx32nn : GemmKernelMx16nn);
            }
            if (M == 16)
            {
                microM = 8;
                microN = 48;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel8x48nn;
                kernelMT = tail > DF ? GemmKernel8x48nn : (tail > F ? GemmKernel8x32nn : GemmKernel8x16nn);
                kernelTM = GemmKernelMx48nn;
                kernelTT = tail > DF ? GemmKernelMx48nn : (tail > F ? GemmKernelMx32nn : GemmKernelMx16nn);
            }
#elif SIMD_ZMM_COUNT == 16 
            if (N < K || M * 8 < N)
            {
                microM = 6;
                microN = 32;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel6x32nn;
                kernelMT = tail > F ? GemmKernel6x32nn : GemmKernel6x16nn;
                kernelTM = GemmKernelMx32nn;
                kernelTT = tail > F ? GemmKernelMx32nn : GemmKernelMx16nn;
            }
            else
            {
                microM = 4;
                microN = 48;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel4x48nn;
                kernelMT = tail > DF ? GemmKernel4x48nn : (tail > F ? GemmKernel4x32nn : GemmKernel4x16nn);
                kernelTM = GemmKernelMx48nn;
                kernelTT = tail > DF ? GemmKernelMx48nn : (tail > F ? GemmKernelMx32nn : GemmKernelMx16nn);
            }
#else
            microM = 4;
            microN = 16;
            kernelMM = GemmKernel4x16nn;
            kernelMT = GemmKernel4x16nn;
            kernelTM = GemmKernelMx16nn;
            kernelTT = GemmKernelMx16nn;
#endif
#if SIMD_ZMM_COUNT >= 16 
            if (M == 4)
            {
                microM = 4;
                microN = 48;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel4x48nn;
                kernelMT = tail > DF ? GemmKernel4x48nn : (tail > F ? GemmKernel4x32nn : GemmKernel4x16nn);
                kernelTM = GemmKernelMx48nn;
                kernelTT = tail > DF ? GemmKernelMx48nn : (tail > F ? GemmKernelMx32nn : GemmKernelMx16nn);
            }
#endif
            GemmNN::PackA packA = (microM > 6 && M*N*K > 700*700*700) ? Avx::GemmPackA : NULL;
            GemmNN gemmNN(M, N, K, microM, microN, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), F,
                kernelMM, kernelMT, kernelTM, kernelTT, packA, Avx512f::GemmPackB, Avx512f::GemmScaleC, TailMask16);
            gemmNN.Run(alpha, A, lda, B, ldb, beta, C, ldc);
        }

        //---------------------------------------------------------------------

        typedef Simd::GemmNNcb<float, __mmask16> Gemm32fNNcb;

        SIMD_INLINE Gemm32fNNcb CreateGemm32fNNcb(size_t M, size_t N, size_t K, GemmKernelType type, bool compatibility)
        {
            Gemm32fNNcb::Main kernelMM, kernelMT;
            Gemm32fNNcb::Tail kernelTM, kernelTT;
            size_t microM, microN;
#if SIMD_ZMM_COUNT == 32
            if (type == GemmKernelF3 || (type == GemmKernelAny && (M == 4 || M == 8 || M == 9 || M == 16 || M == 18 || M == 32 || N == 48 || N == 96 || (M < 14 && M != 6 && M != 12)) && N > 32))
            {
                microN = 48;
                size_t tail = N - AlignLoAny(N, microN);
                if (M == 4)
                {
                    microM = 4;
                    kernelMM = Avx512f::GemmKernel4x48nn;
                    kernelMT = tail > DF ? Avx512f::GemmKernel4x48nn : (tail > F ? Avx512f::GemmKernel4x32nn : Avx512f::GemmKernel4x16nn);
                    kernelTM = Avx512f::GetGemmTail(M%microM, microN);
                    kernelTT = Avx512f::GetGemmTail(M%microM, tail);
                }
                else if (M == 8 || M == 16 || M == 32)
                {
                    microM = 8;
                    kernelMM = Avx512f::GemmKernel8x48nn;
                    kernelMT = tail > DF ? Avx512f::GemmKernel8x48nn : (tail > F ? Avx512f::GemmKernel8x32nn : Avx512f::GemmKernel8x16nn);
                    kernelTM = Avx512f::GetGemmTail(M%microM, microN);
                    kernelTT = Avx512f::GetGemmTail(M%microM, tail);
                }
                else
                {
                    microM = 9;
                    kernelMM = Avx512f::GemmKernel9x48nn;
                    kernelMT = tail > DF ? Avx512f::GemmKernel9x48nn : (tail > F ? Avx512f::GemmKernel9x32nn : Avx512f::GemmKernel9x16nn);
                    kernelTM = Avx512f::GetGemmTail(M%microM, microN);
                    kernelTT = Avx512f::GetGemmTail(M%microM, tail);
                }
                type = GemmKernelF3;
            }
            if (type == GemmKernelF2 || (type == GemmKernelF3 && N <= 32) || (type == GemmKernelAny && N > 16))
            {
                microN = 32;
                size_t tail = N - AlignLoAny(N, microN);
                if (M <= 6)
                {
                    microM = 6;
                    kernelMM = Avx512f::GemmKernel6x32nn;
                    kernelMT = tail > F ? Avx512f::GemmKernel6x32nn : Avx512f::GemmKernel6x16nn;
                    kernelTM = Avx512f::GetGemmTail(M%microM, microN);
                    kernelTT = Avx512f::GetGemmTail(M%microM, tail);
                }
                else if (M <= 12 || M == 24)
                {
                    microM = 12;
                    kernelMM = Avx512f::GemmKernel12x32nn;
                    kernelMT = tail > F ? Avx512f::GemmKernel12x32nn : Avx512f::GemmKernel12x16nn;
                    kernelTM = Avx512f::GetGemmTail(M%microM, microN);
                    kernelTT = Avx512f::GetGemmTail(M%microM, tail);
                }
                else
                {
                    microM = 14;
                    kernelMM = Avx512f::GemmKernel14x32nn;
                    kernelMT = tail > F ? Avx512f::GemmKernel14x32nn : Avx512f::GemmKernel14x16nn;
                    kernelTM = Avx512f::GetGemmTail(M%microM, microN);
                    kernelTT = Avx512f::GetGemmTail(M%microM, tail);
                }
                type = GemmKernelF2;
            }
            if (type == GemmKernelF1 || (type == GemmKernelF2 && N <= 16) || type == GemmKernelAny)
            {
                microM = 14;
                microN = 16;
                kernelMM = Avx512f::GemmKernel14x16nn;
                kernelMT = Avx512f::GemmKernel14x16nn;
                kernelTM = Avx512f::GetGemmTail(M%microM, microN);
                kernelTT = Avx512f::GetGemmTail(M%microM, microN);
                type = GemmKernelF1;
            }
#elif SIMD_ZMM_COUNT == 16
            if (type == GemmKernelF3 || (type == GemmKernelAny && (M == 4 || M == 8 || M == 16 || N == 48 || N == 96) && N > 32))
            {
                microM = 4;
                microN = 48;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx512f::GemmKernel4x48nn;
                kernelMT = tail > DF ? Avx512f::GemmKernel4x48nn : (tail > F ? Avx512f::GemmKernel4x32nn : Avx512f::GemmKernel4x16nn);
                kernelTM = Avx512f::GetGemmTail(M%microM, microN);
                kernelTT = Avx512f::GetGemmTail(M%microM, tail);
                type = GemmKernelF3;
            }
            if (type == GemmKernelF2 || (type == GemmKernelF3 && N <= 32) || (type == GemmKernelAny && N > 16))
            {
                microM = 6;
                microN = 32;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx512f::GemmKernel6x32nn;
                kernelMT = tail > F ? Avx512f::GemmKernel6x32nn : Avx512f::GemmKernel6x16nn;
                kernelTM = Avx512f::GetGemmTail(M%microM, microN);
                kernelTT = Avx512f::GetGemmTail(M%microM, tail);
                type = GemmKernelF2;
            }
            if (type == GemmKernelF1 || (type == GemmKernelF2 && N <= 16) || type == GemmKernelAny)
            {
                microM = 6;
                microN = 16;
                kernelMM = Avx512f::GemmKernel6x16nn;
                kernelMT = Avx512f::GemmKernel6x16nn;
                kernelTM = Avx512f::GetGemmTail(M%microM, microN);
                kernelTT = Avx512f::GetGemmTail(M%microM, microN);
                type = GemmKernelF1;
            }
#else
            microM = 4;
            microN = 16;
            kernelMM = Avx512f::GemmKernel4x16nn;
            kernelMT = Avx512f::GemmKernel4x16nn;
            kernelTM = Avx512f::GetGemmTail(M%microM, microN);
            kernelTT = Avx512f::GetGemmTail(M%microM, microN);
#endif
            return Gemm32fNNcb(M, N, K, microM, microN, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), F, 
                kernelMM, kernelMT, kernelTM, kernelTT, Avx512f::GemmPackB, Avx512f::GemmScaleC, TailMask16, compatibility);
        }

        size_t Gemm32fNNcbBufferSize(size_t M, size_t N, size_t K, GemmKernelType type, bool compatibility)
        {
            if (N > Avx::F)
            {
                Gemm32fNNcb gemm = CreateGemm32fNNcb(M, N, K, type, compatibility);
                return gemm.BufferSize();
            }
            else
                return Avx2::Gemm32fNNcbBufferSize(M, N, K, type, compatibility);
        }

        void Gemm32fNNcbReorderB(size_t M, size_t N, size_t K, const float * B, float * pB, GemmKernelType type, bool compatibility)
        {
            if (N > Avx::F)
            {
                Gemm32fNNcb gemm = CreateGemm32fNNcb(M, N, K, type, compatibility);
                gemm.ReorderB(B, N, pB);
            }
            else
                Avx2::Gemm32fNNcbReorderB(M, N, K, B, pB, type, compatibility);
        }

        void Gemm32fNNcbRun(size_t M, size_t N, size_t K, const float * A, const float * pB, float * C, GemmKernelType type, bool compatibility)
        {
            if (N > Avx::F)
            {
                Gemm32fNNcb gemm = CreateGemm32fNNcb(M, N, K, type, compatibility);
                gemm.Run(A, K, pB, C, N);
            }
            else
                Avx2::Gemm32fNNcbRun(M, N, K, A, pB, C, type, compatibility);
        }

        //---------------------------------------------------------------------

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
            size_t K16 = K & (~16);
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

            typedef Simd::GemmNT<float> GemmNT;
#if SIMD_ZMM_COUNT == 32
            GemmNT gemmNT(M, N, K, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), F, Avx::GemmScaleC,
                Kernel1x1x16nt, Kernel1x4x16nt, Kernel2x1x16nt, Kernel2x4x16nt, Kernel3x1x16nt, Kernel3x4x16nt, Kernel6x1x16nt, Kernel6x4x16nt);
#elif defined(SIMD_X64_ENABLE)
            GemmNT gemmNT(M, N, K, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), F, Avx::GemmScaleC,
                Kernel1x1x16nt, Kernel1x4x16nt, Kernel2x1x16nt, Kernel2x4x16nt, Kernel3x1x16nt, Kernel3x4x16nt, NULL, NULL);
#else
            GemmNT gemmNT(M, N, K, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), F, Sse::GemmScaleC,
                Kernel1x1x16nt, Kernel1x4x16nt, NULL, NULL, NULL, NULL, NULL, NULL);
#endif
            gemmNT.Run(alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
#endif// SIMD_AVX512F_ENABLE
}
