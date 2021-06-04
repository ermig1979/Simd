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
        SIMD_INLINE void AddProduct(float * ptr, __m512 value, __m512 alpha)
        {
            _mm512_storeu_ps(ptr, _mm512_fmadd_ps(value, alpha, _mm512_loadu_ps(ptr)));
        }

        SIMD_INLINE void AddProduct(float * ptr, __m512 value, __m512 alpha, __mmask16 mask)
        {
            _mm512_mask_storeu_ps(ptr, mask, _mm512_fmadd_ps(value, alpha, _mm512_maskz_loadu_ps(mask, ptr)));
        }

        SIMD_INLINE void UpdateC1(float * C, const __m512 & c0, const __m512& alpha, __mmask16 tail)
        {
            AddProduct(C + 0 * F, c0, alpha, tail);
        }

        SIMD_INLINE void UpdateC2(float* C, const __m512& c0, const __m512& c1, const __m512& alpha, __mmask16 tail)
        {
            AddProduct(C + 0 * F, c0, alpha);
            AddProduct(C + 1 * F, c1, alpha, tail);
        }

        SIMD_INLINE void UpdateC3(float* C, const __m512& c0, const __m512& c1, const __m512& c2, const __m512& alpha, __mmask16 tail)
        {
            AddProduct(C + 0 * F, c0, alpha);
            AddProduct(C + 1 * F, c1, alpha);
            AddProduct(C + 2 * F, c2, alpha, tail);
        }

        SIMD_INLINE void UpdateC4(float* C, const __m512& c0, const __m512& c1, const __m512& c2, const __m512& c3, const __m512& alpha, __mmask16 tail)
        {
            AddProduct(C + 0 * F, c0, alpha);
            AddProduct(C + 1 * F, c1, alpha);
            AddProduct(C + 2 * F, c2, alpha);
            AddProduct(C + 3 * F, c3, alpha, tail);
        }

        void GemmKernel4x48nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32, b0, b1, b2, a0;
            c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps(), c02 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps(), c12 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps(), c22 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps(), c32 = _mm512_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            const size_t ob2 = ldb * 2;
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                PrefetchL1(B + ob1);
                PrefetchL1(B + ob2);
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                b2 = _mm512_loadu_ps(B + ob2);
                a0 = _mm512_set1_ps(A[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c01 = _mm512_fmadd_ps(a0, b1, c01), c02 = _mm512_fmadd_ps(a0, b2, c02);
                a0 = _mm512_set1_ps(A[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10), c11 = _mm512_fmadd_ps(a0, b1, c11), c12 = _mm512_fmadd_ps(a0, b2, c12);
                a0 = _mm512_set1_ps(A[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20), c21 = _mm512_fmadd_ps(a0, b1, c21), c22 = _mm512_fmadd_ps(a0, b2, c22);
                a0 = _mm512_set1_ps(A[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30), c31 = _mm512_fmadd_ps(a0, b1, c31), c32 = _mm512_fmadd_ps(a0, b2, c32);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC3(C + 0 * ldc, c00, c01, c02, _alpha, mask);
            UpdateC3(C + 1 * ldc, c10, c11, c12, _alpha, mask);
            UpdateC3(C + 2 * ldc, c20, c21, c22, _alpha, mask);
            UpdateC3(C + 3 * ldc, c30, c31, c32, _alpha, mask);
        }

        void GemmKernel4x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c01, c10, c11, c20, c21, c30, c31, b0, b1, a0;
            c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                PrefetchL1(B + ob1);
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                a0 = _mm512_set1_ps(A[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c01 = _mm512_fmadd_ps(a0, b1, c01);
                a0 = _mm512_set1_ps(A[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10), c11 = _mm512_fmadd_ps(a0, b1, c11);
                a0 = _mm512_set1_ps(A[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20), c21 = _mm512_fmadd_ps(a0, b1, c21);
                a0 = _mm512_set1_ps(A[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30), c31 = _mm512_fmadd_ps(a0, b1, c31);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC2(C + 0 * ldc, c00, c01, _alpha, mask);
            UpdateC2(C + 1 * ldc, c10, c11, _alpha, mask);
            UpdateC2(C + 2 * ldc, c20, c21, _alpha, mask);
            UpdateC2(C + 3 * ldc, c30, c31, _alpha, mask);
        }

        void GemmKernel4x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c10, c20, c30, b0, a0;
            c00 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            const size_t ob0 = ldb * 0;
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                b0 = _mm512_loadu_ps(B + ob0);
                a0 = _mm512_set1_ps(A[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00);
                a0 = _mm512_set1_ps(A[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10);
                a0 = _mm512_set1_ps(A[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20);
                a0 = _mm512_set1_ps(A[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC1(C + 0 * ldc, c00, _alpha, mask);
            UpdateC1(C + 1 * ldc, c10, _alpha, mask);
            UpdateC1(C + 2 * ldc, c20, _alpha, mask);
            UpdateC1(C + 3 * ldc, c30, _alpha, mask);
        }

        void GemmKernel6x64nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33, c40, c41, c42, c43, c50, c51, c52, c53, b0, b1, b2, b3, a0;
            c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps(), c02 = _mm512_setzero_ps(), c03 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps(), c12 = _mm512_setzero_ps(), c13 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps(), c22 = _mm512_setzero_ps(), c23 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps(), c32 = _mm512_setzero_ps(), c33 = _mm512_setzero_ps();
            c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps(), c42 = _mm512_setzero_ps(), c43 = _mm512_setzero_ps();
            c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps(), c52 = _mm512_setzero_ps(), c53 = _mm512_setzero_ps();
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
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                PrefetchL1(B + ob1);
                PrefetchL1(B + ob2);
                PrefetchL1(B + ob3);
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                b2 = _mm512_loadu_ps(B + ob2);
                b3 = _mm512_loadu_ps(B + ob3);
                a0 = _mm512_set1_ps(A[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c01 = _mm512_fmadd_ps(a0, b1, c01), c02 = _mm512_fmadd_ps(a0, b2, c02), c03 = _mm512_fmadd_ps(a0, b3, c03);
                a0 = _mm512_set1_ps(A[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10), c11 = _mm512_fmadd_ps(a0, b1, c11), c12 = _mm512_fmadd_ps(a0, b2, c12), c13 = _mm512_fmadd_ps(a0, b3, c13);
                a0 = _mm512_set1_ps(A[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20), c21 = _mm512_fmadd_ps(a0, b1, c21), c22 = _mm512_fmadd_ps(a0, b2, c22), c23 = _mm512_fmadd_ps(a0, b3, c23);
                a0 = _mm512_set1_ps(A[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30), c31 = _mm512_fmadd_ps(a0, b1, c31), c32 = _mm512_fmadd_ps(a0, b2, c32), c33 = _mm512_fmadd_ps(a0, b3, c33);
                a0 = _mm512_set1_ps(A[oa4]), c40 = _mm512_fmadd_ps(a0, b0, c40), c41 = _mm512_fmadd_ps(a0, b1, c41), c42 = _mm512_fmadd_ps(a0, b2, c42), c43 = _mm512_fmadd_ps(a0, b3, c43);
                a0 = _mm512_set1_ps(A[oa5]), c50 = _mm512_fmadd_ps(a0, b0, c50), c51 = _mm512_fmadd_ps(a0, b1, c51), c52 = _mm512_fmadd_ps(a0, b2, c52), c53 = _mm512_fmadd_ps(a0, b3, c53);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC4(C + 0 * ldc, c00, c01, c02, c03, _alpha, mask);
            UpdateC4(C + 1 * ldc, c10, c11, c12, c13, _alpha, mask);
            UpdateC4(C + 2 * ldc, c20, c21, c22, c23, _alpha, mask);
            UpdateC4(C + 3 * ldc, c30, c31, c32, c33, _alpha, mask);
            UpdateC4(C + 4 * ldc, c40, c41, c42, c43, _alpha, mask);
            UpdateC4(C + 5 * ldc, c50, c51, c52, c53, _alpha, mask);
        }

        void GemmKernel6x48nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32, c40, c41, c42, c50, c51, c52, b0, b1, b2, a0;
            c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps(), c02 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps(), c12 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps(), c22 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps(), c32 = _mm512_setzero_ps();
            c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps(), c42 = _mm512_setzero_ps();
            c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps(), c52 = _mm512_setzero_ps();
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
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                PrefetchL1(B + ob1);
                PrefetchL1(B + ob2);
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                b2 = _mm512_loadu_ps(B + ob2);
                a0 = _mm512_set1_ps(A[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c01 = _mm512_fmadd_ps(a0, b1, c01), c02 = _mm512_fmadd_ps(a0, b2, c02);
                a0 = _mm512_set1_ps(A[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10), c11 = _mm512_fmadd_ps(a0, b1, c11), c12 = _mm512_fmadd_ps(a0, b2, c12);
                a0 = _mm512_set1_ps(A[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20), c21 = _mm512_fmadd_ps(a0, b1, c21), c22 = _mm512_fmadd_ps(a0, b2, c22);
                a0 = _mm512_set1_ps(A[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30), c31 = _mm512_fmadd_ps(a0, b1, c31), c32 = _mm512_fmadd_ps(a0, b2, c32);
                a0 = _mm512_set1_ps(A[oa4]), c40 = _mm512_fmadd_ps(a0, b0, c40), c41 = _mm512_fmadd_ps(a0, b1, c41), c42 = _mm512_fmadd_ps(a0, b2, c42);
                a0 = _mm512_set1_ps(A[oa5]), c50 = _mm512_fmadd_ps(a0, b0, c50), c51 = _mm512_fmadd_ps(a0, b1, c51), c52 = _mm512_fmadd_ps(a0, b2, c52);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC3(C + 0 * ldc, c00, c01, c02, _alpha, mask);
            UpdateC3(C + 1 * ldc, c10, c11, c12, _alpha, mask);
            UpdateC3(C + 2 * ldc, c20, c21, c22, _alpha, mask);
            UpdateC3(C + 3 * ldc, c30, c31, c32, _alpha, mask);
            UpdateC3(C + 4 * ldc, c40, c41, c42, _alpha, mask);
            UpdateC3(C + 5 * ldc, c50, c51, c52, _alpha, mask);
        }

        void GemmKernel6x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, b0, b1, a0;
            c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
            c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
            c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            const size_t ob0 = ldb * 0;
            const size_t ob1 = ldb * 1;
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                PrefetchL1(B + ob1);
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                a0 = _mm512_set1_ps(A[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c01 = _mm512_fmadd_ps(a0, b1, c01);
                a0 = _mm512_set1_ps(A[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10), c11 = _mm512_fmadd_ps(a0, b1, c11);
                a0 = _mm512_set1_ps(A[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20), c21 = _mm512_fmadd_ps(a0, b1, c21);
                a0 = _mm512_set1_ps(A[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30), c31 = _mm512_fmadd_ps(a0, b1, c31);
                a0 = _mm512_set1_ps(A[oa4]), c40 = _mm512_fmadd_ps(a0, b0, c40), c41 = _mm512_fmadd_ps(a0, b1, c41);
                a0 = _mm512_set1_ps(A[oa5]), c50 = _mm512_fmadd_ps(a0, b0, c50), c51 = _mm512_fmadd_ps(a0, b1, c51);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC2(C + 0 * ldc, c00, c01, _alpha, mask);
            UpdateC2(C + 1 * ldc, c10, c11, _alpha, mask);
            UpdateC2(C + 2 * ldc, c20, c21, _alpha, mask);
            UpdateC2(C + 3 * ldc, c30, c31, _alpha, mask);
            UpdateC2(C + 4 * ldc, c40, c41, _alpha, mask);
            UpdateC2(C + 5 * ldc, c50, c51, _alpha, mask);
        }

        void GemmKernel6x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c10, c20, c30, c40, c50, b0, a0;
            c00 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps();
            c40 = _mm512_setzero_ps();
            c50 = _mm512_setzero_ps();
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            const size_t ob0 = ldb * 0;
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                b0 = _mm512_loadu_ps(B + ob0);
                a0 = _mm512_set1_ps(A[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00);
                a0 = _mm512_set1_ps(A[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10);
                a0 = _mm512_set1_ps(A[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20);
                a0 = _mm512_set1_ps(A[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30);
                a0 = _mm512_set1_ps(A[oa4]), c40 = _mm512_fmadd_ps(a0, b0, c40);
                a0 = _mm512_set1_ps(A[oa5]), c50 = _mm512_fmadd_ps(a0, b0, c50);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC1(C + 0 * ldc, c00, _alpha, mask);
            UpdateC1(C + 1 * ldc, c10, _alpha, mask);
            UpdateC1(C + 2 * ldc, c20, _alpha, mask);
            UpdateC1(C + 3 * ldc, c30, _alpha, mask);
            UpdateC1(C + 4 * ldc, c40, _alpha, mask);
            UpdateC1(C + 5 * ldc, c50, _alpha, mask);
        }

        void GemmKernel8x48nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32, c40, c41, c42, c50, c51, c52, c60, c61, c62, c70, c71,c72, b0, b1, b2, a0;
            c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps(), c02 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps(), c12 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps(), c22 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps(), c32 = _mm512_setzero_ps();
            c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps(), c42 = _mm512_setzero_ps();
            c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps(), c52 = _mm512_setzero_ps();
            c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps(), c62 = _mm512_setzero_ps();
            c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps(), c72 = _mm512_setzero_ps();
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
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                PrefetchL1(B + ob1);
                PrefetchL1(B + ob2);
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                b2 = _mm512_loadu_ps(B + ob2);
                a0 = _mm512_set1_ps(A[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c01 = _mm512_fmadd_ps(a0, b1, c01), c02 = _mm512_fmadd_ps(a0, b2, c02);
                a0 = _mm512_set1_ps(A[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10), c11 = _mm512_fmadd_ps(a0, b1, c11), c12 = _mm512_fmadd_ps(a0, b2, c12);
                a0 = _mm512_set1_ps(A[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20), c21 = _mm512_fmadd_ps(a0, b1, c21), c22 = _mm512_fmadd_ps(a0, b2, c22);
                a0 = _mm512_set1_ps(A[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30), c31 = _mm512_fmadd_ps(a0, b1, c31), c32 = _mm512_fmadd_ps(a0, b2, c32);
                a0 = _mm512_set1_ps(A[oa4]), c40 = _mm512_fmadd_ps(a0, b0, c40), c41 = _mm512_fmadd_ps(a0, b1, c41), c42 = _mm512_fmadd_ps(a0, b2, c42);
                a0 = _mm512_set1_ps(A[oa5]), c50 = _mm512_fmadd_ps(a0, b0, c50), c51 = _mm512_fmadd_ps(a0, b1, c51), c52 = _mm512_fmadd_ps(a0, b2, c52);
                a0 = _mm512_set1_ps(A[oa6]), c60 = _mm512_fmadd_ps(a0, b0, c60), c61 = _mm512_fmadd_ps(a0, b1, c61), c62 = _mm512_fmadd_ps(a0, b2, c62);
                a0 = _mm512_set1_ps(A[oa7]), c70 = _mm512_fmadd_ps(a0, b0, c70), c71 = _mm512_fmadd_ps(a0, b1, c71), c72 = _mm512_fmadd_ps(a0, b2, c72);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC3(C + 0 * ldc, c00, c01, c02, _alpha, mask);
            UpdateC3(C + 1 * ldc, c10, c11, c12, _alpha, mask);
            UpdateC3(C + 2 * ldc, c20, c21, c22, _alpha, mask);
            UpdateC3(C + 3 * ldc, c30, c31, c32, _alpha, mask);
            UpdateC3(C + 4 * ldc, c40, c41, c42, _alpha, mask);
            UpdateC3(C + 5 * ldc, c50, c51, c52, _alpha, mask);
            UpdateC3(C + 6 * ldc, c60, c61, c62, _alpha, mask);
            UpdateC3(C + 7 * ldc, c70, c71, c72, _alpha, mask);
        }

        void GemmKernel8x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc,  __mmask16 mask)
        {
            __m512 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, b0, b1, a0;
            c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
            c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
            c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
            c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps();
            c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps();
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
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                PrefetchL1(B + ob1);
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                a0 = _mm512_set1_ps(A[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c01 = _mm512_fmadd_ps(a0, b1, c01);
                a0 = _mm512_set1_ps(A[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10), c11 = _mm512_fmadd_ps(a0, b1, c11);
                a0 = _mm512_set1_ps(A[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20), c21 = _mm512_fmadd_ps(a0, b1, c21);
                a0 = _mm512_set1_ps(A[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30), c31 = _mm512_fmadd_ps(a0, b1, c31);
                a0 = _mm512_set1_ps(A[oa4]), c40 = _mm512_fmadd_ps(a0, b0, c40), c41 = _mm512_fmadd_ps(a0, b1, c41);
                a0 = _mm512_set1_ps(A[oa5]), c50 = _mm512_fmadd_ps(a0, b0, c50), c51 = _mm512_fmadd_ps(a0, b1, c51);
                a0 = _mm512_set1_ps(A[oa6]), c60 = _mm512_fmadd_ps(a0, b0, c60), c61 = _mm512_fmadd_ps(a0, b1, c61);
                a0 = _mm512_set1_ps(A[oa7]), c70 = _mm512_fmadd_ps(a0, b0, c70), c71 = _mm512_fmadd_ps(a0, b1, c71);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC2(C + 0 * ldc, c00, c01, _alpha, mask);
            UpdateC2(C + 1 * ldc, c10, c11, _alpha, mask);
            UpdateC2(C + 2 * ldc, c20, c21, _alpha, mask);
            UpdateC2(C + 3 * ldc, c30, c31, _alpha, mask);
            UpdateC2(C + 4 * ldc, c40, c41, _alpha, mask);
            UpdateC2(C + 5 * ldc, c50, c51, _alpha, mask);
            UpdateC2(C + 6 * ldc, c60, c61, _alpha, mask);
            UpdateC2(C + 7 * ldc, c70, c71, _alpha, mask);
        }

        void GemmKernel8x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c10, c20, c30, c40, c50, c60, c70, b0, a0;
            c00 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps();
            c40 = _mm512_setzero_ps();
            c50 = _mm512_setzero_ps();
            c60 = _mm512_setzero_ps();
            c70 = _mm512_setzero_ps();
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
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                b0 = _mm512_loadu_ps(B + ob0);
                a0 = _mm512_set1_ps(A[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00);
                a0 = _mm512_set1_ps(A[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10);
                a0 = _mm512_set1_ps(A[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20);
                a0 = _mm512_set1_ps(A[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30);
                a0 = _mm512_set1_ps(A[oa4]), c40 = _mm512_fmadd_ps(a0, b0, c40);
                a0 = _mm512_set1_ps(A[oa5]), c50 = _mm512_fmadd_ps(a0, b0, c50);
                a0 = _mm512_set1_ps(A[oa6]), c60 = _mm512_fmadd_ps(a0, b0, c60);
                a0 = _mm512_set1_ps(A[oa7]), c70 = _mm512_fmadd_ps(a0, b0, c70);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC1(C + 0 * ldc, c00, _alpha, mask);
            UpdateC1(C + 1 * ldc, c10, _alpha, mask);
            UpdateC1(C + 2 * ldc, c20, _alpha, mask);
            UpdateC1(C + 3 * ldc, c30, _alpha, mask);
            UpdateC1(C + 4 * ldc, c40, _alpha, mask);
            UpdateC1(C + 5 * ldc, c50, _alpha, mask);
            UpdateC1(C + 6 * ldc, c60, _alpha, mask);
            UpdateC1(C + 7 * ldc, c70, _alpha, mask);
        }

        void GemmKernel9x48nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32, c40, c41, c42, c50, c51, c52, c60, c61, c62, c70, c71, c72, c80, c81, c82, b0, b1, b2, a0;
            c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps(), c02 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps(), c12 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps(), c22 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps(), c32 = _mm512_setzero_ps();
            c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps(), c42 = _mm512_setzero_ps();
            c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps(), c52 = _mm512_setzero_ps();
            c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps(), c62 = _mm512_setzero_ps();
            c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps(), c72 = _mm512_setzero_ps();
            c80 = _mm512_setzero_ps(), c81 = _mm512_setzero_ps(), c82 = _mm512_setzero_ps();
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
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                PrefetchL1(B + ob1);
                PrefetchL1(B + ob2);
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                b2 = _mm512_loadu_ps(B + ob2);
                a0 = _mm512_set1_ps(A[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c01 = _mm512_fmadd_ps(a0, b1, c01), c02 = _mm512_fmadd_ps(a0, b2, c02);
                a0 = _mm512_set1_ps(A[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10), c11 = _mm512_fmadd_ps(a0, b1, c11), c12 = _mm512_fmadd_ps(a0, b2, c12);
                a0 = _mm512_set1_ps(A[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20), c21 = _mm512_fmadd_ps(a0, b1, c21), c22 = _mm512_fmadd_ps(a0, b2, c22);
                a0 = _mm512_set1_ps(A[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30), c31 = _mm512_fmadd_ps(a0, b1, c31), c32 = _mm512_fmadd_ps(a0, b2, c32);
                a0 = _mm512_set1_ps(A[oa4]), c40 = _mm512_fmadd_ps(a0, b0, c40), c41 = _mm512_fmadd_ps(a0, b1, c41), c42 = _mm512_fmadd_ps(a0, b2, c42);
                a0 = _mm512_set1_ps(A[oa5]), c50 = _mm512_fmadd_ps(a0, b0, c50), c51 = _mm512_fmadd_ps(a0, b1, c51), c52 = _mm512_fmadd_ps(a0, b2, c52);
                a0 = _mm512_set1_ps(A[oa6]), c60 = _mm512_fmadd_ps(a0, b0, c60), c61 = _mm512_fmadd_ps(a0, b1, c61), c62 = _mm512_fmadd_ps(a0, b2, c62);
                a0 = _mm512_set1_ps(A[oa7]), c70 = _mm512_fmadd_ps(a0, b0, c70), c71 = _mm512_fmadd_ps(a0, b1, c71), c72 = _mm512_fmadd_ps(a0, b2, c72);
                a0 = _mm512_set1_ps(A[oa8]), c80 = _mm512_fmadd_ps(a0, b0, c80), c81 = _mm512_fmadd_ps(a0, b1, c81), c82 = _mm512_fmadd_ps(a0, b2, c82);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC3(C + 0 * ldc, c00, c01, c02, _alpha, mask);
            UpdateC3(C + 1 * ldc, c10, c11, c12, _alpha, mask);
            UpdateC3(C + 2 * ldc, c20, c21, c22, _alpha, mask);
            UpdateC3(C + 3 * ldc, c30, c31, c32, _alpha, mask);
            UpdateC3(C + 4 * ldc, c40, c41, c42, _alpha, mask);
            UpdateC3(C + 5 * ldc, c50, c51, c52, _alpha, mask);
            UpdateC3(C + 6 * ldc, c60, c61, c62, _alpha, mask);
            UpdateC3(C + 7 * ldc, c70, c71, c72, _alpha, mask);
            UpdateC3(C + 8 * ldc, c80, c81, c82, _alpha, mask);
        }

        void GemmKernel9x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, c80, c81, b0, b1, a0;
            c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
            c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
            c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
            c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps();
            c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps();
            c80 = _mm512_setzero_ps(), c81 = _mm512_setzero_ps();
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
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                PrefetchL1(B + ob1);
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                a0 = _mm512_set1_ps(A[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c01 = _mm512_fmadd_ps(a0, b1, c01);
                a0 = _mm512_set1_ps(A[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10), c11 = _mm512_fmadd_ps(a0, b1, c11);
                a0 = _mm512_set1_ps(A[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20), c21 = _mm512_fmadd_ps(a0, b1, c21);
                a0 = _mm512_set1_ps(A[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30), c31 = _mm512_fmadd_ps(a0, b1, c31);
                a0 = _mm512_set1_ps(A[oa4]), c40 = _mm512_fmadd_ps(a0, b0, c40), c41 = _mm512_fmadd_ps(a0, b1, c41);
                a0 = _mm512_set1_ps(A[oa5]), c50 = _mm512_fmadd_ps(a0, b0, c50), c51 = _mm512_fmadd_ps(a0, b1, c51);
                a0 = _mm512_set1_ps(A[oa6]), c60 = _mm512_fmadd_ps(a0, b0, c60), c61 = _mm512_fmadd_ps(a0, b1, c61);
                a0 = _mm512_set1_ps(A[oa7]), c70 = _mm512_fmadd_ps(a0, b0, c70), c71 = _mm512_fmadd_ps(a0, b1, c71);
                a0 = _mm512_set1_ps(A[oa8]), c80 = _mm512_fmadd_ps(a0, b0, c80), c81 = _mm512_fmadd_ps(a0, b1, c81);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC2(C + 0 * ldc, c00, c01, _alpha, mask);
            UpdateC2(C + 1 * ldc, c10, c11, _alpha, mask);
            UpdateC2(C + 2 * ldc, c20, c21, _alpha, mask);
            UpdateC2(C + 3 * ldc, c30, c31, _alpha, mask);
            UpdateC2(C + 4 * ldc, c40, c41, _alpha, mask);
            UpdateC2(C + 5 * ldc, c50, c51, _alpha, mask);
            UpdateC2(C + 6 * ldc, c60, c61, _alpha, mask);
            UpdateC2(C + 7 * ldc, c70, c71, _alpha, mask);
            UpdateC2(C + 8 * ldc, c80, c81, _alpha, mask);
        }

        void GemmKernel9x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c10, c20, c30, c40, c50, c60, c70, c80, b0, a0;
            c00 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps();
            c40 = _mm512_setzero_ps();
            c50 = _mm512_setzero_ps();
            c60 = _mm512_setzero_ps();
            c70 = _mm512_setzero_ps();
            c80 = _mm512_setzero_ps();
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
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                b0 = _mm512_loadu_ps(B + ob0);
                a0 = _mm512_set1_ps(A[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00);
                a0 = _mm512_set1_ps(A[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10);
                a0 = _mm512_set1_ps(A[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20);
                a0 = _mm512_set1_ps(A[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30);
                a0 = _mm512_set1_ps(A[oa4]), c40 = _mm512_fmadd_ps(a0, b0, c40);
                a0 = _mm512_set1_ps(A[oa5]), c50 = _mm512_fmadd_ps(a0, b0, c50);
                a0 = _mm512_set1_ps(A[oa6]), c60 = _mm512_fmadd_ps(a0, b0, c60);
                a0 = _mm512_set1_ps(A[oa7]), c70 = _mm512_fmadd_ps(a0, b0, c70);
                a0 = _mm512_set1_ps(A[oa8]), c80 = _mm512_fmadd_ps(a0, b0, c80);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC1(C + 0 * ldc, c00, _alpha, mask);
            UpdateC1(C + 1 * ldc, c10, _alpha, mask);
            UpdateC1(C + 2 * ldc, c20, _alpha, mask);
            UpdateC1(C + 3 * ldc, c30, _alpha, mask);
            UpdateC1(C + 4 * ldc, c40, _alpha, mask);
            UpdateC1(C + 5 * ldc, c50, _alpha, mask);
            UpdateC1(C + 6 * ldc, c60, _alpha, mask);
            UpdateC1(C + 7 * ldc, c70, _alpha, mask);
            UpdateC1(C + 8 * ldc, c80, _alpha, mask);
        }

        void GemmKernel12x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, c80, c81, c90, c91, ca0, ca1, cb0, cb1, b0, b1, a0;
            c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
            c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
            c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
            c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps();
            c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps();
            c80 = _mm512_setzero_ps(), c81 = _mm512_setzero_ps();
            c90 = _mm512_setzero_ps(), c91 = _mm512_setzero_ps();
            ca0 = _mm512_setzero_ps(), ca1 = _mm512_setzero_ps();
            cb0 = _mm512_setzero_ps(), cb1 = _mm512_setzero_ps();
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
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                PrefetchL1(B + ob1);
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                a0 = _mm512_set1_ps(A0[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c01 = _mm512_fmadd_ps(a0, b1, c01);
                a0 = _mm512_set1_ps(A0[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10), c11 = _mm512_fmadd_ps(a0, b1, c11);
                a0 = _mm512_set1_ps(A0[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20), c21 = _mm512_fmadd_ps(a0, b1, c21);
                a0 = _mm512_set1_ps(A0[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30), c31 = _mm512_fmadd_ps(a0, b1, c31);
                a0 = _mm512_set1_ps(A0[oa4]), c40 = _mm512_fmadd_ps(a0, b0, c40), c41 = _mm512_fmadd_ps(a0, b1, c41);
                a0 = _mm512_set1_ps(A0[oa5]), c50 = _mm512_fmadd_ps(a0, b0, c50), c51 = _mm512_fmadd_ps(a0, b1, c51);
                a0 = _mm512_set1_ps(A6[oa0]), c60 = _mm512_fmadd_ps(a0, b0, c60), c61 = _mm512_fmadd_ps(a0, b1, c61);
                a0 = _mm512_set1_ps(A6[oa1]), c70 = _mm512_fmadd_ps(a0, b0, c70), c71 = _mm512_fmadd_ps(a0, b1, c71);
                a0 = _mm512_set1_ps(A6[oa2]), c80 = _mm512_fmadd_ps(a0, b0, c80), c81 = _mm512_fmadd_ps(a0, b1, c81);
                a0 = _mm512_set1_ps(A6[oa3]), c90 = _mm512_fmadd_ps(a0, b0, c90), c91 = _mm512_fmadd_ps(a0, b1, c91);
                a0 = _mm512_set1_ps(A6[oa4]), ca0 = _mm512_fmadd_ps(a0, b0, ca0), ca1 = _mm512_fmadd_ps(a0, b1, ca1);
                a0 = _mm512_set1_ps(A6[oa5]), cb0 = _mm512_fmadd_ps(a0, b0, cb0), cb1 = _mm512_fmadd_ps(a0, b1, cb1);
                B += sb;
                A0 += sa;
                A6 += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC2(C + 0x0 * ldc, c00, c01, _alpha, mask);
            UpdateC2(C + 0x1 * ldc, c10, c11, _alpha, mask);
            UpdateC2(C + 0x2 * ldc, c20, c21, _alpha, mask);
            UpdateC2(C + 0x3 * ldc, c30, c31, _alpha, mask);
            UpdateC2(C + 0x4 * ldc, c40, c41, _alpha, mask);
            UpdateC2(C + 0x5 * ldc, c50, c51, _alpha, mask);
            UpdateC2(C + 0x6 * ldc, c60, c61, _alpha, mask);
            UpdateC2(C + 0x7 * ldc, c70, c71, _alpha, mask);
            UpdateC2(C + 0x8 * ldc, c80, c81, _alpha, mask);
            UpdateC2(C + 0x9 * ldc, c90, c91, _alpha, mask);
            UpdateC2(C + 0xA * ldc, ca0, ca1, _alpha, mask);
            UpdateC2(C + 0xB * ldc, cb0, cb1, _alpha, mask);
        }

        void GemmKernel12x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c10, c20, c30, c40, c50, c60, c70, c80, c90, ca0, cb0, b0, a0;
            c00 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps();
            c40 = _mm512_setzero_ps();
            c50 = _mm512_setzero_ps();
            c60 = _mm512_setzero_ps();
            c70 = _mm512_setzero_ps();
            c80 = _mm512_setzero_ps();
            c90 = _mm512_setzero_ps();
            ca0 = _mm512_setzero_ps();
            cb0 = _mm512_setzero_ps();
            const float* A0 = A, * A6 = A + 6 * lda;
            const size_t oa0 = lda * 0;
            const size_t oa1 = lda * 1;
            const size_t oa2 = lda * 2;
            const size_t oa3 = lda * 3;
            const size_t oa4 = lda * 4;
            const size_t oa5 = lda * 5;
            const size_t sa = lda == 1 ? 12 : 1;
            const size_t ob0 = ldb * 0;
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                b0 = _mm512_loadu_ps(B + ob0);
                a0 = _mm512_set1_ps(A0[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00);
                a0 = _mm512_set1_ps(A0[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10);
                a0 = _mm512_set1_ps(A0[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20);
                a0 = _mm512_set1_ps(A0[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30);
                a0 = _mm512_set1_ps(A0[oa4]), c40 = _mm512_fmadd_ps(a0, b0, c40);
                a0 = _mm512_set1_ps(A0[oa5]), c50 = _mm512_fmadd_ps(a0, b0, c50);
                a0 = _mm512_set1_ps(A6[oa0]), c60 = _mm512_fmadd_ps(a0, b0, c60);
                a0 = _mm512_set1_ps(A6[oa1]), c70 = _mm512_fmadd_ps(a0, b0, c70);
                a0 = _mm512_set1_ps(A6[oa2]), c80 = _mm512_fmadd_ps(a0, b0, c80);
                a0 = _mm512_set1_ps(A6[oa3]), c90 = _mm512_fmadd_ps(a0, b0, c90);
                a0 = _mm512_set1_ps(A6[oa4]), ca0 = _mm512_fmadd_ps(a0, b0, ca0);
                a0 = _mm512_set1_ps(A6[oa5]), cb0 = _mm512_fmadd_ps(a0, b0, cb0);
                B += sb;
                A0 += sa;
                A6 += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC1(C + 0x0 * ldc, c00, _alpha, mask);
            UpdateC1(C + 0x1 * ldc, c10, _alpha, mask);
            UpdateC1(C + 0x2 * ldc, c20, _alpha, mask);
            UpdateC1(C + 0x3 * ldc, c30, _alpha, mask);
            UpdateC1(C + 0x4 * ldc, c40, _alpha, mask);
            UpdateC1(C + 0x5 * ldc, c50, _alpha, mask);
            UpdateC1(C + 0x6 * ldc, c60, _alpha, mask);
            UpdateC1(C + 0x7 * ldc, c70, _alpha, mask);
            UpdateC1(C + 0x8 * ldc, c80, _alpha, mask);
            UpdateC1(C + 0x9 * ldc, c90, _alpha, mask);
            UpdateC1(C + 0xA * ldc, ca0, _alpha, mask);
            UpdateC1(C + 0xB * ldc, cb0, _alpha, mask);
        }

        void GemmKernel14x32nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, c80, c81, c90, c91, ca0, ca1, cb0, cb1, cc0, cc1, cd0, cd1, b0, b1, a0;
            c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
            c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
            c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
            c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps();
            c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps();
            c80 = _mm512_setzero_ps(), c81 = _mm512_setzero_ps();
            c90 = _mm512_setzero_ps(), c91 = _mm512_setzero_ps();
            ca0 = _mm512_setzero_ps(), ca1 = _mm512_setzero_ps();
            cb0 = _mm512_setzero_ps(), cb1 = _mm512_setzero_ps();
            cc0 = _mm512_setzero_ps(), cc1 = _mm512_setzero_ps();
            cd0 = _mm512_setzero_ps(), cd1 = _mm512_setzero_ps();
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
            for (const float* endB = B + sb * K; B < endB;)
            {
                PrefetchL1(B + ob0);
                PrefetchL1(B + ob1);
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                a0 = _mm512_set1_ps(A0[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c01 = _mm512_fmadd_ps(a0, b1, c01);
                a0 = _mm512_set1_ps(A0[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10), c11 = _mm512_fmadd_ps(a0, b1, c11);
                a0 = _mm512_set1_ps(A0[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20), c21 = _mm512_fmadd_ps(a0, b1, c21);
                a0 = _mm512_set1_ps(A0[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30), c31 = _mm512_fmadd_ps(a0, b1, c31);
                a0 = _mm512_set1_ps(A0[oa4]), c40 = _mm512_fmadd_ps(a0, b0, c40), c41 = _mm512_fmadd_ps(a0, b1, c41);
                a0 = _mm512_set1_ps(A0[oa5]), c50 = _mm512_fmadd_ps(a0, b0, c50), c51 = _mm512_fmadd_ps(a0, b1, c51);
                a0 = _mm512_set1_ps(A0[oa6]), c60 = _mm512_fmadd_ps(a0, b0, c60), c61 = _mm512_fmadd_ps(a0, b1, c61);
                a0 = _mm512_set1_ps(A7[oa0]), c70 = _mm512_fmadd_ps(a0, b0, c70), c71 = _mm512_fmadd_ps(a0, b1, c71);
                a0 = _mm512_set1_ps(A7[oa1]), c80 = _mm512_fmadd_ps(a0, b0, c80), c81 = _mm512_fmadd_ps(a0, b1, c81);
                a0 = _mm512_set1_ps(A7[oa2]), c90 = _mm512_fmadd_ps(a0, b0, c90), c91 = _mm512_fmadd_ps(a0, b1, c91);
                a0 = _mm512_set1_ps(A7[oa3]), ca0 = _mm512_fmadd_ps(a0, b0, ca0), ca1 = _mm512_fmadd_ps(a0, b1, ca1);
                a0 = _mm512_set1_ps(A7[oa4]), cb0 = _mm512_fmadd_ps(a0, b0, cb0), cb1 = _mm512_fmadd_ps(a0, b1, cb1);
                a0 = _mm512_set1_ps(A7[oa5]), cc0 = _mm512_fmadd_ps(a0, b0, cc0), cc1 = _mm512_fmadd_ps(a0, b1, cc1);
                a0 = _mm512_set1_ps(A7[oa6]), cd0 = _mm512_fmadd_ps(a0, b0, cd0), cd1 = _mm512_fmadd_ps(a0, b1, cd1);
                B += sb;
                A0 += sa;
                A7 += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC2(C + 0x0 * ldc, c00, c01, _alpha, mask);
            UpdateC2(C + 0x1 * ldc, c10, c11, _alpha, mask);
            UpdateC2(C + 0x2 * ldc, c20, c21, _alpha, mask);
            UpdateC2(C + 0x3 * ldc, c30, c31, _alpha, mask);
            UpdateC2(C + 0x4 * ldc, c40, c41, _alpha, mask);
            UpdateC2(C + 0x5 * ldc, c50, c51, _alpha, mask);
            UpdateC2(C + 0x6 * ldc, c60, c61, _alpha, mask);
            UpdateC2(C + 0x7 * ldc, c70, c71, _alpha, mask);
            UpdateC2(C + 0x8 * ldc, c80, c81, _alpha, mask);
            UpdateC2(C + 0x9 * ldc, c90, c91, _alpha, mask);
            UpdateC2(C + 0xA * ldc, ca0, ca1, _alpha, mask);
            UpdateC2(C + 0xB * ldc, cb0, cb1, _alpha, mask);
            UpdateC2(C + 0xC * ldc, cc0, cc1, _alpha, mask);
            UpdateC2(C + 0xD * ldc, cd0, cd1, _alpha, mask);
        }

        void GemmKernel14x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, size_t sb, float * C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c10, c20, c30, c40, c50, c60, c70, c80, c90, ca0, cb0, cc0, cd0, b0, a0;
            c00 = _mm512_setzero_ps();
            c10 = _mm512_setzero_ps();
            c20 = _mm512_setzero_ps();
            c30 = _mm512_setzero_ps();
            c40 = _mm512_setzero_ps();
            c50 = _mm512_setzero_ps();
            c60 = _mm512_setzero_ps();
            c70 = _mm512_setzero_ps();
            c80 = _mm512_setzero_ps();
            c90 = _mm512_setzero_ps();
            ca0 = _mm512_setzero_ps();
            cb0 = _mm512_setzero_ps();
            cc0 = _mm512_setzero_ps();
            cd0 = _mm512_setzero_ps();
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
            for (size_t k = 0; k < K; k++)
            {
                PrefetchL1(B + ob0);
                b0 = _mm512_loadu_ps(B + ob0);
                a0 = _mm512_set1_ps(A0[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00);
                a0 = _mm512_set1_ps(A0[oa1]), c10 = _mm512_fmadd_ps(a0, b0, c10);
                a0 = _mm512_set1_ps(A0[oa2]), c20 = _mm512_fmadd_ps(a0, b0, c20);
                a0 = _mm512_set1_ps(A0[oa3]), c30 = _mm512_fmadd_ps(a0, b0, c30);
                a0 = _mm512_set1_ps(A0[oa4]), c40 = _mm512_fmadd_ps(a0, b0, c40);
                a0 = _mm512_set1_ps(A0[oa5]), c50 = _mm512_fmadd_ps(a0, b0, c50);
                a0 = _mm512_set1_ps(A0[oa6]), c60 = _mm512_fmadd_ps(a0, b0, c60);
                a0 = _mm512_set1_ps(A7[oa0]), c70 = _mm512_fmadd_ps(a0, b0, c70);
                a0 = _mm512_set1_ps(A7[oa1]), c80 = _mm512_fmadd_ps(a0, b0, c80);
                a0 = _mm512_set1_ps(A7[oa2]), c90 = _mm512_fmadd_ps(a0, b0, c90);
                a0 = _mm512_set1_ps(A7[oa3]), ca0 = _mm512_fmadd_ps(a0, b0, ca0);
                a0 = _mm512_set1_ps(A7[oa4]), cb0 = _mm512_fmadd_ps(a0, b0, cb0);
                a0 = _mm512_set1_ps(A7[oa5]), cc0 = _mm512_fmadd_ps(a0, b0, cc0);
                a0 = _mm512_set1_ps(A7[oa6]), cd0 = _mm512_fmadd_ps(a0, b0, cd0);
                B += sb;
                A0 += sa;
                A7 += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            UpdateC1(C + 0x0 * ldc, c00, _alpha, mask);
            UpdateC1(C + 0x1 * ldc, c10, _alpha, mask);
            UpdateC1(C + 0x2 * ldc, c20, _alpha, mask);
            UpdateC1(C + 0x3 * ldc, c30, _alpha, mask);
            UpdateC1(C + 0x4 * ldc, c40, _alpha, mask);
            UpdateC1(C + 0x5 * ldc, c50, _alpha, mask);
            UpdateC1(C + 0x6 * ldc, c60, _alpha, mask);
            UpdateC1(C + 0x7 * ldc, c70, _alpha, mask);
            UpdateC1(C + 0x8 * ldc, c80, _alpha, mask);
            UpdateC1(C + 0x9 * ldc, c90, _alpha, mask);
            UpdateC1(C + 0xA * ldc, ca0, _alpha, mask);
            UpdateC1(C + 0xB * ldc, cb0, _alpha, mask);
            UpdateC1(C + 0xC * ldc, cc0, _alpha, mask);
            UpdateC1(C + 0xD * ldc, cd0, _alpha, mask);
        }

        template<int M> void GemmKernelMx64nnT(size_t, size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t sb, float* C, size_t ldc, __mmask16 mask)
        {
            __m512 c00, c01, c02, c03, c04, c05, c10, c11, c12, c13, c14, c15, c20, c21, c22, c23, c24, c25, c30, c31, c32, c33, c34, c35, b0, b1, b2, b3, a0;
            if (M > 0x0) c00 = _mm512_setzero_ps(), c10 = _mm512_setzero_ps(), c20 = _mm512_setzero_ps(), c30 = _mm512_setzero_ps();
            if (M > 0x1) c01 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
            if (M > 0x2) c02 = _mm512_setzero_ps(), c12 = _mm512_setzero_ps(), c22 = _mm512_setzero_ps(), c32 = _mm512_setzero_ps();
            if (M > 0x3) c03 = _mm512_setzero_ps(), c13 = _mm512_setzero_ps(), c23 = _mm512_setzero_ps(), c33 = _mm512_setzero_ps();
            if (M > 0x4) c04 = _mm512_setzero_ps(), c14 = _mm512_setzero_ps(), c24 = _mm512_setzero_ps(), c34 = _mm512_setzero_ps();
            if (M > 0x5) c05 = _mm512_setzero_ps(), c15 = _mm512_setzero_ps(), c25 = _mm512_setzero_ps(), c35 = _mm512_setzero_ps();
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
                b0 = _mm512_loadu_ps(B + ob0);
                b1 = _mm512_loadu_ps(B + ob1);
                b2 = _mm512_loadu_ps(B + ob2);
                b3 = _mm512_loadu_ps(B + ob3);
                if (M > 0x0) a0 = _mm512_set1_ps(A[oa0]), c00 = _mm512_fmadd_ps(a0, b0, c00), c10 = _mm512_fmadd_ps(a0, b1, c10), c20 = _mm512_fmadd_ps(a0, b2, c20), c30 = _mm512_fmadd_ps(a0, b3, c30);
                if (M > 0x1) a0 = _mm512_set1_ps(A[oa1]), c01 = _mm512_fmadd_ps(a0, b0, c01), c11 = _mm512_fmadd_ps(a0, b1, c11), c21 = _mm512_fmadd_ps(a0, b2, c21), c31 = _mm512_fmadd_ps(a0, b3, c31);
                if (M > 0x2) a0 = _mm512_set1_ps(A[oa2]), c02 = _mm512_fmadd_ps(a0, b0, c02), c12 = _mm512_fmadd_ps(a0, b1, c12), c22 = _mm512_fmadd_ps(a0, b2, c22), c32 = _mm512_fmadd_ps(a0, b3, c32);
                if (M > 0x3) a0 = _mm512_set1_ps(A[oa3]), c03 = _mm512_fmadd_ps(a0, b0, c03), c13 = _mm512_fmadd_ps(a0, b1, c13), c23 = _mm512_fmadd_ps(a0, b2, c23), c33 = _mm512_fmadd_ps(a0, b3, c33);
                if (M > 0x4) a0 = _mm512_set1_ps(A[oa4]), c04 = _mm512_fmadd_ps(a0, b0, c04), c14 = _mm512_fmadd_ps(a0, b1, c14), c24 = _mm512_fmadd_ps(a0, b2, c24), c34 = _mm512_fmadd_ps(a0, b3, c34);
                if (M > 0x5) a0 = _mm512_set1_ps(A[oa5]), c05 = _mm512_fmadd_ps(a0, b0, c05), c15 = _mm512_fmadd_ps(a0, b1, c15), c25 = _mm512_fmadd_ps(a0, b2, c25), c35 = _mm512_fmadd_ps(a0, b3, c35);
                B += sb;
                A += sa;
            }
            __m512 _alpha = _mm512_set1_ps(alpha);
            if (M > 0x0) AddProduct(C + 0 * F, _alpha, c00), AddProduct(C + 1 * F, _alpha, c10), AddProduct(C + 2 * F, _alpha, c20), AddProduct(C + 3 * F, _alpha, c30, mask), C += ldc;
            if (M > 0x1) AddProduct(C + 0 * F, _alpha, c01), AddProduct(C + 1 * F, _alpha, c11), AddProduct(C + 2 * F, _alpha, c21), AddProduct(C + 3 * F, _alpha, c31, mask), C += ldc;
            if (M > 0x2) AddProduct(C + 0 * F, _alpha, c02), AddProduct(C + 1 * F, _alpha, c12), AddProduct(C + 2 * F, _alpha, c22), AddProduct(C + 3 * F, _alpha, c32, mask), C += ldc;
            if (M > 0x3) AddProduct(C + 0 * F, _alpha, c03), AddProduct(C + 1 * F, _alpha, c13), AddProduct(C + 2 * F, _alpha, c23), AddProduct(C + 3 * F, _alpha, c33, mask), C += ldc;
            if (M > 0x4) AddProduct(C + 0 * F, _alpha, c04), AddProduct(C + 1 * F, _alpha, c14), AddProduct(C + 2 * F, _alpha, c24), AddProduct(C + 3 * F, _alpha, c34, mask), C += ldc;
            if (M > 0x5) AddProduct(C + 0 * F, _alpha, c05), AddProduct(C + 1 * F, _alpha, c15), AddProduct(C + 2 * F, _alpha, c25), AddProduct(C + 3 * F, _alpha, c35, mask), C += ldc;
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

        SIMD_INLINE Simd::GemmNN<float, F, __mmask16>::Tail GetGemmTail(size_t M, size_t N)
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
            else if (N <= 64)
            {
                switch (M)
                {
                case 0: return GemmKernelMx64nnT<0>;
                case 1: return GemmKernelMx64nnT<1>;
                case 2: return GemmKernelMx64nnT<2>;
                case 3: return GemmKernelMx64nnT<3>;
                case 4: return GemmKernelMx64nnT<4>;
                case 5: return GemmKernelMx64nnT<5>;
                }
            }
            assert(0);
            return NULL;
        }

        //---------------------------------------------------------------------

        void Gemm32fNN(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            SIMD_PERF_BEGF(Simd::ToStr(M) + "-" + Simd::ToStr(N) + "-" + Simd::ToStr(K), M*N*K * 2);

            typedef Simd::GemmNN<float, F, __mmask16> GemmNN;
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
                kernelTM = Avx512f::GetGemmTail(M % microM, microN);
                kernelTT = Avx512f::GetGemmTail(M % microM, tail);
            }
            else
            {
                microM = 9;
                microN = 48;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel9x48nn;
                kernelMT = tail > DF ? GemmKernel9x48nn : (tail > F ? GemmKernel9x32nn : GemmKernel9x16nn);
                kernelTM = Avx512f::GetGemmTail(M % microM, microN);
                kernelTT = Avx512f::GetGemmTail(M % microM, tail);
            }
            if (M == 16)
            {
                microM = 8;
                microN = 48;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel8x48nn;
                kernelMT = tail > DF ? GemmKernel8x48nn : (tail > F ? GemmKernel8x32nn : GemmKernel8x16nn);
                kernelTM = Avx512f::GetGemmTail(M % microM, microN);
                kernelTT = Avx512f::GetGemmTail(M % microM, tail);
            }
#elif SIMD_ZMM_COUNT == 16 
            if (N < K || M * 8 < N)
            {
                microM = 6;
                microN = 32;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel6x32nn;
                kernelMT = tail > F ? GemmKernel6x32nn : GemmKernel6x16nn;
                kernelTM = Avx512f::GetGemmTail(M % microM, microN);
                kernelTT = Avx512f::GetGemmTail(M % microM, tail);
            }
            else
            {
                microM = 4;
                microN = 48;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel4x48nn;
                kernelMT = tail > DF ? GemmKernel4x48nn : (tail > F ? GemmKernel4x32nn : GemmKernel4x16nn);
                kernelTM = Avx512f::GetGemmTail(M % microM, microN);
                kernelTT = Avx512f::GetGemmTail(M % microM, tail);
            }
#else
            microM = 4;
            microN = 16;
            size_t tail = N - AlignLoAny(N, microN);
            kernelMM = GemmKernel4x16nn;
            kernelMT = GemmKernel4x16nn;
            kernelTM = Avx512f::GetGemmTail(M % microM, microN);
            kernelTT = Avx512f::GetGemmTail(M % microM, tail);
#endif
#if SIMD_ZMM_COUNT >= 16 
            if (M == 4)
            {
                microM = 4;
                microN = 48;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = GemmKernel4x48nn;
                kernelMT = tail > DF ? GemmKernel4x48nn : (tail > F ? GemmKernel4x32nn : GemmKernel4x16nn);
                kernelTM = Avx512f::GetGemmTail(M % microM, microN);
                kernelTT = Avx512f::GetGemmTail(M % microM, tail);
            }
#endif
            GemmNN::PackA packA = (microM > 6 && M*N*K > 700*700*700) ? Avx::GemmPackA : NULL;
            GemmNN gemmNN(M, N, K, microM, microN, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), 
                kernelMM, kernelMT, kernelTM, kernelTT, packA, Avx512f::GemmPackB, Avx512f::GemmScaleC, TailMask16);
            gemmNN.Run(alpha, A, lda, B, ldb, beta, C, ldc);
        }

        //---------------------------------------------------------------------

        typedef Simd::GemmNNcb<float, F, __mmask16> Gemm32fNNcb;

        SIMD_INLINE Gemm32fNNcb CreateGemm32fNNcb(size_t M, size_t N, size_t K, GemmKernelType type, bool compatibility)
        {
            Gemm32fNNcb::Main kernelMM, kernelMT;
            Gemm32fNNcb::Tail kernelTM, kernelTT;
            size_t microM, microN;
#if SIMD_ZMM_COUNT == 32
            if (type == GemmKernelF4 || (type == GemmKernelAny && (M == 6 || N == 64)))
            {
                microN = 64;
                size_t tail = N - AlignLoAny(N, microN);
                {
                    microM = 6;
                    kernelMM = Avx512f::GemmKernel6x64nn;
                    kernelMT = tail > 3*F ? Avx512f::GemmKernel6x64nn : (tail > DF ? Avx512f::GemmKernel6x48nn : (tail > F ? Avx512f::GemmKernel6x32nn : Avx512f::GemmKernel6x16nn));
                    kernelTM = Avx512f::GetGemmTail(M % microM, microN);
                    kernelTT = Avx512f::GetGemmTail(M % microM, tail);
                }
                type = GemmKernelF4;
            }
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
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx512f::GemmKernel14x16nn;
                kernelMT = Avx512f::GemmKernel14x16nn;
                kernelTM = Avx512f::GetGemmTail(M%microM, microN);
                kernelTT = Avx512f::GetGemmTail(M%microM, tail);
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
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx512f::GemmKernel6x16nn;
                kernelMT = Avx512f::GemmKernel6x16nn;
                kernelTM = Avx512f::GetGemmTail(M%microM, microN);
                kernelTT = Avx512f::GetGemmTail(M%microM, tail);
                type = GemmKernelF1;
            }
#else
            microM = 4;
            microN = 16;
            size_t tail = N - AlignLoAny(N, microN);
            kernelMM = Avx512f::GemmKernel4x16nn;
            kernelMT = Avx512f::GemmKernel4x16nn;
            kernelTM = Avx512f::GetGemmTail(M%microM, microN);
            kernelTT = Avx512f::GetGemmTail(M%microM, tail);
#endif
            return Gemm32fNNcb(M, N, K, microM, microN, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), 
                kernelMM, kernelMT, kernelTM, kernelTT, Avx512f::GemmPackA, Avx512f::GemmPackB, Avx512f::GemmScaleC, TailMask16, compatibility);
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
            //SIMD_PERF_BEGF(Simd::ToStr(M) + "-" + Simd::ToStr(N) + "-" + Simd::ToStr(K), M * N * K * 2);
            if (N > Avx::F)
            {
                Gemm32fNNcb gemm = CreateGemm32fNNcb(M, N, K, type, compatibility);
                gemm.Run(A, K, pB, C, N);
            }
            else
                Avx2::Gemm32fNNcbRun(M, N, K, A, pB, C, type, compatibility);
        }
    }
#endif// SIMD_AVX512F_ENABLE
}
