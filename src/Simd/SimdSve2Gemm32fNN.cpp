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
        SIMD_INLINE void AddProduct(float* ptr, const svfloat32_t& value, const svfloat32_t& alpha, const svbool_t& mask)
        {
            svst1_f32(mask, ptr, svmla_f32_x(mask, svld1_f32(mask, ptr), value, alpha));
        }

        SIMD_INLINE void AddProduct(float* ptr, const svfloat32_t& value, const svfloat32_t& alpha)
        {
            const svbool_t body = svptrue_b32();
            svst1_f32(body, ptr, svmla_f32_x(body, svld1_f32(body, ptr), value, alpha));
        }

        static void Kernel1x1nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc, size_t n)
        {
            const size_t F = svcntw();
            const svbool_t mask = svwhilelt_b32((uint64_t)0, (uint64_t)n);
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(mask, B + 0 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_m(mask, c00, b0, a0);
                A += 1;
                B += ldb;
            }
            AddProduct(C + 0 * ldc + 0 * F, c00, svdup_n_f32(alpha), mask);
        }

        static void Kernel1x2nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c01 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(body, B + 0 * F);
                svfloat32_t b1 = svld1_f32(body, B + 1 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_x(body, c00, b0, a0);
                c01 = svmla_f32_x(body, c01, b1, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha);
            AddProduct(C + 0 * ldc + 1 * F, c01, _alpha);
        }

        static void Kernel1x3nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c01 = zero;
            svfloat32_t c02 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(body, B + 0 * F);
                svfloat32_t b1 = svld1_f32(body, B + 1 * F);
                svfloat32_t b2 = svld1_f32(body, B + 2 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_x(body, c00, b0, a0);
                c01 = svmla_f32_x(body, c01, b1, a0);
                c02 = svmla_f32_x(body, c02, b2, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha);
            AddProduct(C + 0 * ldc + 1 * F, c01, _alpha);
            AddProduct(C + 0 * ldc + 2 * F, c02, _alpha);
        }

        static void Kernel1x4nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c01 = zero;
            svfloat32_t c02 = zero;
            svfloat32_t c03 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(body, B + 0 * F);
                svfloat32_t b1 = svld1_f32(body, B + 1 * F);
                svfloat32_t b2 = svld1_f32(body, B + 2 * F);
                svfloat32_t b3 = svld1_f32(body, B + 3 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_x(body, c00, b0, a0);
                c01 = svmla_f32_x(body, c01, b1, a0);
                c02 = svmla_f32_x(body, c02, b2, a0);
                c03 = svmla_f32_x(body, c03, b3, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha);
            AddProduct(C + 0 * ldc + 1 * F, c01, _alpha);
            AddProduct(C + 0 * ldc + 2 * F, c02, _alpha);
            AddProduct(C + 0 * ldc + 3 * F, c03, _alpha);
        }

        static void Kernel2x1nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc, size_t n)
        {
            const size_t F = svcntw();
            const svbool_t mask = svwhilelt_b32((uint64_t)0, (uint64_t)n);
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c10 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(mask, B + 0 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_m(mask, c00, b0, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_m(mask, c10, b0, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha, mask);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha, mask);
        }

        static void Kernel2x2nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c01 = zero;
            svfloat32_t c10 = zero;
            svfloat32_t c11 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(body, B + 0 * F);
                svfloat32_t b1 = svld1_f32(body, B + 1 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_x(body, c00, b0, a0);
                c01 = svmla_f32_x(body, c01, b1, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_x(body, c10, b0, a0);
                c11 = svmla_f32_x(body, c11, b1, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha);
            AddProduct(C + 0 * ldc + 1 * F, c01, _alpha);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha);
            AddProduct(C + 1 * ldc + 1 * F, c11, _alpha);
        }

        static void Kernel2x3nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c01 = zero;
            svfloat32_t c02 = zero;
            svfloat32_t c10 = zero;
            svfloat32_t c11 = zero;
            svfloat32_t c12 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(body, B + 0 * F);
                svfloat32_t b1 = svld1_f32(body, B + 1 * F);
                svfloat32_t b2 = svld1_f32(body, B + 2 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_x(body, c00, b0, a0);
                c01 = svmla_f32_x(body, c01, b1, a0);
                c02 = svmla_f32_x(body, c02, b2, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_x(body, c10, b0, a0);
                c11 = svmla_f32_x(body, c11, b1, a0);
                c12 = svmla_f32_x(body, c12, b2, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha);
            AddProduct(C + 0 * ldc + 1 * F, c01, _alpha);
            AddProduct(C + 0 * ldc + 2 * F, c02, _alpha);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha);
            AddProduct(C + 1 * ldc + 1 * F, c11, _alpha);
            AddProduct(C + 1 * ldc + 2 * F, c12, _alpha);
        }

        static void Kernel2x4nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c01 = zero;
            svfloat32_t c02 = zero;
            svfloat32_t c03 = zero;
            svfloat32_t c10 = zero;
            svfloat32_t c11 = zero;
            svfloat32_t c12 = zero;
            svfloat32_t c13 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(body, B + 0 * F);
                svfloat32_t b1 = svld1_f32(body, B + 1 * F);
                svfloat32_t b2 = svld1_f32(body, B + 2 * F);
                svfloat32_t b3 = svld1_f32(body, B + 3 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_x(body, c00, b0, a0);
                c01 = svmla_f32_x(body, c01, b1, a0);
                c02 = svmla_f32_x(body, c02, b2, a0);
                c03 = svmla_f32_x(body, c03, b3, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_x(body, c10, b0, a0);
                c11 = svmla_f32_x(body, c11, b1, a0);
                c12 = svmla_f32_x(body, c12, b2, a0);
                c13 = svmla_f32_x(body, c13, b3, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha);
            AddProduct(C + 0 * ldc + 1 * F, c01, _alpha);
            AddProduct(C + 0 * ldc + 2 * F, c02, _alpha);
            AddProduct(C + 0 * ldc + 3 * F, c03, _alpha);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha);
            AddProduct(C + 1 * ldc + 1 * F, c11, _alpha);
            AddProduct(C + 1 * ldc + 2 * F, c12, _alpha);
            AddProduct(C + 1 * ldc + 3 * F, c13, _alpha);
        }

        static void Kernel3x1nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc, size_t n)
        {
            const size_t F = svcntw();
            const svbool_t mask = svwhilelt_b32((uint64_t)0, (uint64_t)n);
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c10 = zero;
            svfloat32_t c20 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(mask, B + 0 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_m(mask, c00, b0, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_m(mask, c10, b0, a0);
                a0 = svdup_n_f32(A[2 * lda]);
                c20 = svmla_f32_m(mask, c20, b0, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha, mask);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha, mask);
            AddProduct(C + 2 * ldc + 0 * F, c20, _alpha, mask);
        }

        static void Kernel3x2nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c01 = zero;
            svfloat32_t c10 = zero;
            svfloat32_t c11 = zero;
            svfloat32_t c20 = zero;
            svfloat32_t c21 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(body, B + 0 * F);
                svfloat32_t b1 = svld1_f32(body, B + 1 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_x(body, c00, b0, a0);
                c01 = svmla_f32_x(body, c01, b1, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_x(body, c10, b0, a0);
                c11 = svmla_f32_x(body, c11, b1, a0);
                a0 = svdup_n_f32(A[2 * lda]);
                c20 = svmla_f32_x(body, c20, b0, a0);
                c21 = svmla_f32_x(body, c21, b1, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha);
            AddProduct(C + 0 * ldc + 1 * F, c01, _alpha);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha);
            AddProduct(C + 1 * ldc + 1 * F, c11, _alpha);
            AddProduct(C + 2 * ldc + 0 * F, c20, _alpha);
            AddProduct(C + 2 * ldc + 1 * F, c21, _alpha);
        }

        static void Kernel3x3nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c01 = zero;
            svfloat32_t c02 = zero;
            svfloat32_t c10 = zero;
            svfloat32_t c11 = zero;
            svfloat32_t c12 = zero;
            svfloat32_t c20 = zero;
            svfloat32_t c21 = zero;
            svfloat32_t c22 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(body, B + 0 * F);
                svfloat32_t b1 = svld1_f32(body, B + 1 * F);
                svfloat32_t b2 = svld1_f32(body, B + 2 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_x(body, c00, b0, a0);
                c01 = svmla_f32_x(body, c01, b1, a0);
                c02 = svmla_f32_x(body, c02, b2, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_x(body, c10, b0, a0);
                c11 = svmla_f32_x(body, c11, b1, a0);
                c12 = svmla_f32_x(body, c12, b2, a0);
                a0 = svdup_n_f32(A[2 * lda]);
                c20 = svmla_f32_x(body, c20, b0, a0);
                c21 = svmla_f32_x(body, c21, b1, a0);
                c22 = svmla_f32_x(body, c22, b2, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha);
            AddProduct(C + 0 * ldc + 1 * F, c01, _alpha);
            AddProduct(C + 0 * ldc + 2 * F, c02, _alpha);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha);
            AddProduct(C + 1 * ldc + 1 * F, c11, _alpha);
            AddProduct(C + 1 * ldc + 2 * F, c12, _alpha);
            AddProduct(C + 2 * ldc + 0 * F, c20, _alpha);
            AddProduct(C + 2 * ldc + 1 * F, c21, _alpha);
            AddProduct(C + 2 * ldc + 2 * F, c22, _alpha);
        }

        static void Kernel3x4nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c01 = zero;
            svfloat32_t c02 = zero;
            svfloat32_t c03 = zero;
            svfloat32_t c10 = zero;
            svfloat32_t c11 = zero;
            svfloat32_t c12 = zero;
            svfloat32_t c13 = zero;
            svfloat32_t c20 = zero;
            svfloat32_t c21 = zero;
            svfloat32_t c22 = zero;
            svfloat32_t c23 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(body, B + 0 * F);
                svfloat32_t b1 = svld1_f32(body, B + 1 * F);
                svfloat32_t b2 = svld1_f32(body, B + 2 * F);
                svfloat32_t b3 = svld1_f32(body, B + 3 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_x(body, c00, b0, a0);
                c01 = svmla_f32_x(body, c01, b1, a0);
                c02 = svmla_f32_x(body, c02, b2, a0);
                c03 = svmla_f32_x(body, c03, b3, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_x(body, c10, b0, a0);
                c11 = svmla_f32_x(body, c11, b1, a0);
                c12 = svmla_f32_x(body, c12, b2, a0);
                c13 = svmla_f32_x(body, c13, b3, a0);
                a0 = svdup_n_f32(A[2 * lda]);
                c20 = svmla_f32_x(body, c20, b0, a0);
                c21 = svmla_f32_x(body, c21, b1, a0);
                c22 = svmla_f32_x(body, c22, b2, a0);
                c23 = svmla_f32_x(body, c23, b3, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha);
            AddProduct(C + 0 * ldc + 1 * F, c01, _alpha);
            AddProduct(C + 0 * ldc + 2 * F, c02, _alpha);
            AddProduct(C + 0 * ldc + 3 * F, c03, _alpha);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha);
            AddProduct(C + 1 * ldc + 1 * F, c11, _alpha);
            AddProduct(C + 1 * ldc + 2 * F, c12, _alpha);
            AddProduct(C + 1 * ldc + 3 * F, c13, _alpha);
            AddProduct(C + 2 * ldc + 0 * F, c20, _alpha);
            AddProduct(C + 2 * ldc + 1 * F, c21, _alpha);
            AddProduct(C + 2 * ldc + 2 * F, c22, _alpha);
            AddProduct(C + 2 * ldc + 3 * F, c23, _alpha);
        }

        static void Kernel4x1nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc, size_t n)
        {
            const size_t F = svcntw();
            const svbool_t mask = svwhilelt_b32((uint64_t)0, (uint64_t)n);
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c10 = zero;
            svfloat32_t c20 = zero;
            svfloat32_t c30 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(mask, B + 0 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_m(mask, c00, b0, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_m(mask, c10, b0, a0);
                a0 = svdup_n_f32(A[2 * lda]);
                c20 = svmla_f32_m(mask, c20, b0, a0);
                a0 = svdup_n_f32(A[3 * lda]);
                c30 = svmla_f32_m(mask, c30, b0, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha, mask);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha, mask);
            AddProduct(C + 2 * ldc + 0 * F, c20, _alpha, mask);
            AddProduct(C + 3 * ldc + 0 * F, c30, _alpha, mask);
        }

        static void Kernel4x4nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c01 = zero;
            svfloat32_t c02 = zero;
            svfloat32_t c03 = zero;
            svfloat32_t c10 = zero;
            svfloat32_t c11 = zero;
            svfloat32_t c12 = zero;
            svfloat32_t c13 = zero;
            svfloat32_t c20 = zero;
            svfloat32_t c21 = zero;
            svfloat32_t c22 = zero;
            svfloat32_t c23 = zero;
            svfloat32_t c30 = zero;
            svfloat32_t c31 = zero;
            svfloat32_t c32 = zero;
            svfloat32_t c33 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(body, B + 0 * F);
                svfloat32_t b1 = svld1_f32(body, B + 1 * F);
                svfloat32_t b2 = svld1_f32(body, B + 2 * F);
                svfloat32_t b3 = svld1_f32(body, B + 3 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_x(body, c00, b0, a0);
                c01 = svmla_f32_x(body, c01, b1, a0);
                c02 = svmla_f32_x(body, c02, b2, a0);
                c03 = svmla_f32_x(body, c03, b3, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_x(body, c10, b0, a0);
                c11 = svmla_f32_x(body, c11, b1, a0);
                c12 = svmla_f32_x(body, c12, b2, a0);
                c13 = svmla_f32_x(body, c13, b3, a0);
                a0 = svdup_n_f32(A[2 * lda]);
                c20 = svmla_f32_x(body, c20, b0, a0);
                c21 = svmla_f32_x(body, c21, b1, a0);
                c22 = svmla_f32_x(body, c22, b2, a0);
                c23 = svmla_f32_x(body, c23, b3, a0);
                a0 = svdup_n_f32(A[3 * lda]);
                c30 = svmla_f32_x(body, c30, b0, a0);
                c31 = svmla_f32_x(body, c31, b1, a0);
                c32 = svmla_f32_x(body, c32, b2, a0);
                c33 = svmla_f32_x(body, c33, b3, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha);
            AddProduct(C + 0 * ldc + 1 * F, c01, _alpha);
            AddProduct(C + 0 * ldc + 2 * F, c02, _alpha);
            AddProduct(C + 0 * ldc + 3 * F, c03, _alpha);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha);
            AddProduct(C + 1 * ldc + 1 * F, c11, _alpha);
            AddProduct(C + 1 * ldc + 2 * F, c12, _alpha);
            AddProduct(C + 1 * ldc + 3 * F, c13, _alpha);
            AddProduct(C + 2 * ldc + 0 * F, c20, _alpha);
            AddProduct(C + 2 * ldc + 1 * F, c21, _alpha);
            AddProduct(C + 2 * ldc + 2 * F, c22, _alpha);
            AddProduct(C + 2 * ldc + 3 * F, c23, _alpha);
            AddProduct(C + 3 * ldc + 0 * F, c30, _alpha);
            AddProduct(C + 3 * ldc + 1 * F, c31, _alpha);
            AddProduct(C + 3 * ldc + 2 * F, c32, _alpha);
            AddProduct(C + 3 * ldc + 3 * F, c33, _alpha);
        }

        static void Kernel5x1nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc, size_t n)
        {
            const size_t F = svcntw();
            const svbool_t mask = svwhilelt_b32((uint64_t)0, (uint64_t)n);
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c10 = zero;
            svfloat32_t c20 = zero;
            svfloat32_t c30 = zero;
            svfloat32_t c40 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(mask, B + 0 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_m(mask, c00, b0, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_m(mask, c10, b0, a0);
                a0 = svdup_n_f32(A[2 * lda]);
                c20 = svmla_f32_m(mask, c20, b0, a0);
                a0 = svdup_n_f32(A[3 * lda]);
                c30 = svmla_f32_m(mask, c30, b0, a0);
                a0 = svdup_n_f32(A[4 * lda]);
                c40 = svmla_f32_m(mask, c40, b0, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha, mask);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha, mask);
            AddProduct(C + 2 * ldc + 0 * F, c20, _alpha, mask);
            AddProduct(C + 3 * ldc + 0 * F, c30, _alpha, mask);
            AddProduct(C + 4 * ldc + 0 * F, c40, _alpha, mask);
        }

        static void Kernel5x4nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c01 = zero;
            svfloat32_t c02 = zero;
            svfloat32_t c03 = zero;
            svfloat32_t c10 = zero;
            svfloat32_t c11 = zero;
            svfloat32_t c12 = zero;
            svfloat32_t c13 = zero;
            svfloat32_t c20 = zero;
            svfloat32_t c21 = zero;
            svfloat32_t c22 = zero;
            svfloat32_t c23 = zero;
            svfloat32_t c30 = zero;
            svfloat32_t c31 = zero;
            svfloat32_t c32 = zero;
            svfloat32_t c33 = zero;
            svfloat32_t c40 = zero;
            svfloat32_t c41 = zero;
            svfloat32_t c42 = zero;
            svfloat32_t c43 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(body, B + 0 * F);
                svfloat32_t b1 = svld1_f32(body, B + 1 * F);
                svfloat32_t b2 = svld1_f32(body, B + 2 * F);
                svfloat32_t b3 = svld1_f32(body, B + 3 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_x(body, c00, b0, a0);
                c01 = svmla_f32_x(body, c01, b1, a0);
                c02 = svmla_f32_x(body, c02, b2, a0);
                c03 = svmla_f32_x(body, c03, b3, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_x(body, c10, b0, a0);
                c11 = svmla_f32_x(body, c11, b1, a0);
                c12 = svmla_f32_x(body, c12, b2, a0);
                c13 = svmla_f32_x(body, c13, b3, a0);
                a0 = svdup_n_f32(A[2 * lda]);
                c20 = svmla_f32_x(body, c20, b0, a0);
                c21 = svmla_f32_x(body, c21, b1, a0);
                c22 = svmla_f32_x(body, c22, b2, a0);
                c23 = svmla_f32_x(body, c23, b3, a0);
                a0 = svdup_n_f32(A[3 * lda]);
                c30 = svmla_f32_x(body, c30, b0, a0);
                c31 = svmla_f32_x(body, c31, b1, a0);
                c32 = svmla_f32_x(body, c32, b2, a0);
                c33 = svmla_f32_x(body, c33, b3, a0);
                a0 = svdup_n_f32(A[4 * lda]);
                c40 = svmla_f32_x(body, c40, b0, a0);
                c41 = svmla_f32_x(body, c41, b1, a0);
                c42 = svmla_f32_x(body, c42, b2, a0);
                c43 = svmla_f32_x(body, c43, b3, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha);
            AddProduct(C + 0 * ldc + 1 * F, c01, _alpha);
            AddProduct(C + 0 * ldc + 2 * F, c02, _alpha);
            AddProduct(C + 0 * ldc + 3 * F, c03, _alpha);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha);
            AddProduct(C + 1 * ldc + 1 * F, c11, _alpha);
            AddProduct(C + 1 * ldc + 2 * F, c12, _alpha);
            AddProduct(C + 1 * ldc + 3 * F, c13, _alpha);
            AddProduct(C + 2 * ldc + 0 * F, c20, _alpha);
            AddProduct(C + 2 * ldc + 1 * F, c21, _alpha);
            AddProduct(C + 2 * ldc + 2 * F, c22, _alpha);
            AddProduct(C + 2 * ldc + 3 * F, c23, _alpha);
            AddProduct(C + 3 * ldc + 0 * F, c30, _alpha);
            AddProduct(C + 3 * ldc + 1 * F, c31, _alpha);
            AddProduct(C + 3 * ldc + 2 * F, c32, _alpha);
            AddProduct(C + 3 * ldc + 3 * F, c33, _alpha);
            AddProduct(C + 4 * ldc + 0 * F, c40, _alpha);
            AddProduct(C + 4 * ldc + 1 * F, c41, _alpha);
            AddProduct(C + 4 * ldc + 2 * F, c42, _alpha);
            AddProduct(C + 4 * ldc + 3 * F, c43, _alpha);
        }

        static void Kernel6x1nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc, size_t n)
        {
            const size_t F = svcntw();
            const svbool_t mask = svwhilelt_b32((uint64_t)0, (uint64_t)n);
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c10 = zero;
            svfloat32_t c20 = zero;
            svfloat32_t c30 = zero;
            svfloat32_t c40 = zero;
            svfloat32_t c50 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(mask, B + 0 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_m(mask, c00, b0, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_m(mask, c10, b0, a0);
                a0 = svdup_n_f32(A[2 * lda]);
                c20 = svmla_f32_m(mask, c20, b0, a0);
                a0 = svdup_n_f32(A[3 * lda]);
                c30 = svmla_f32_m(mask, c30, b0, a0);
                a0 = svdup_n_f32(A[4 * lda]);
                c40 = svmla_f32_m(mask, c40, b0, a0);
                a0 = svdup_n_f32(A[5 * lda]);
                c50 = svmla_f32_m(mask, c50, b0, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha, mask);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha, mask);
            AddProduct(C + 2 * ldc + 0 * F, c20, _alpha, mask);
            AddProduct(C + 3 * ldc + 0 * F, c30, _alpha, mask);
            AddProduct(C + 4 * ldc + 0 * F, c40, _alpha, mask);
            AddProduct(C + 5 * ldc + 0 * F, c50, _alpha, mask);
        }

        static void Kernel6x2nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c01 = zero;
            svfloat32_t c10 = zero;
            svfloat32_t c11 = zero;
            svfloat32_t c20 = zero;
            svfloat32_t c21 = zero;
            svfloat32_t c30 = zero;
            svfloat32_t c31 = zero;
            svfloat32_t c40 = zero;
            svfloat32_t c41 = zero;
            svfloat32_t c50 = zero;
            svfloat32_t c51 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(body, B + 0 * F);
                svfloat32_t b1 = svld1_f32(body, B + 1 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_x(body, c00, b0, a0);
                c01 = svmla_f32_x(body, c01, b1, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_x(body, c10, b0, a0);
                c11 = svmla_f32_x(body, c11, b1, a0);
                a0 = svdup_n_f32(A[2 * lda]);
                c20 = svmla_f32_x(body, c20, b0, a0);
                c21 = svmla_f32_x(body, c21, b1, a0);
                a0 = svdup_n_f32(A[3 * lda]);
                c30 = svmla_f32_x(body, c30, b0, a0);
                c31 = svmla_f32_x(body, c31, b1, a0);
                a0 = svdup_n_f32(A[4 * lda]);
                c40 = svmla_f32_x(body, c40, b0, a0);
                c41 = svmla_f32_x(body, c41, b1, a0);
                a0 = svdup_n_f32(A[5 * lda]);
                c50 = svmla_f32_x(body, c50, b0, a0);
                c51 = svmla_f32_x(body, c51, b1, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha);
            AddProduct(C + 0 * ldc + 1 * F, c01, _alpha);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha);
            AddProduct(C + 1 * ldc + 1 * F, c11, _alpha);
            AddProduct(C + 2 * ldc + 0 * F, c20, _alpha);
            AddProduct(C + 2 * ldc + 1 * F, c21, _alpha);
            AddProduct(C + 3 * ldc + 0 * F, c30, _alpha);
            AddProduct(C + 3 * ldc + 1 * F, c31, _alpha);
            AddProduct(C + 4 * ldc + 0 * F, c40, _alpha);
            AddProduct(C + 4 * ldc + 1 * F, c41, _alpha);
            AddProduct(C + 5 * ldc + 0 * F, c50, _alpha);
            AddProduct(C + 5 * ldc + 1 * F, c51, _alpha);
        }

        static void Kernel6x3nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c01 = zero;
            svfloat32_t c02 = zero;
            svfloat32_t c10 = zero;
            svfloat32_t c11 = zero;
            svfloat32_t c12 = zero;
            svfloat32_t c20 = zero;
            svfloat32_t c21 = zero;
            svfloat32_t c22 = zero;
            svfloat32_t c30 = zero;
            svfloat32_t c31 = zero;
            svfloat32_t c32 = zero;
            svfloat32_t c40 = zero;
            svfloat32_t c41 = zero;
            svfloat32_t c42 = zero;
            svfloat32_t c50 = zero;
            svfloat32_t c51 = zero;
            svfloat32_t c52 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(body, B + 0 * F);
                svfloat32_t b1 = svld1_f32(body, B + 1 * F);
                svfloat32_t b2 = svld1_f32(body, B + 2 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_x(body, c00, b0, a0);
                c01 = svmla_f32_x(body, c01, b1, a0);
                c02 = svmla_f32_x(body, c02, b2, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_x(body, c10, b0, a0);
                c11 = svmla_f32_x(body, c11, b1, a0);
                c12 = svmla_f32_x(body, c12, b2, a0);
                a0 = svdup_n_f32(A[2 * lda]);
                c20 = svmla_f32_x(body, c20, b0, a0);
                c21 = svmla_f32_x(body, c21, b1, a0);
                c22 = svmla_f32_x(body, c22, b2, a0);
                a0 = svdup_n_f32(A[3 * lda]);
                c30 = svmla_f32_x(body, c30, b0, a0);
                c31 = svmla_f32_x(body, c31, b1, a0);
                c32 = svmla_f32_x(body, c32, b2, a0);
                a0 = svdup_n_f32(A[4 * lda]);
                c40 = svmla_f32_x(body, c40, b0, a0);
                c41 = svmla_f32_x(body, c41, b1, a0);
                c42 = svmla_f32_x(body, c42, b2, a0);
                a0 = svdup_n_f32(A[5 * lda]);
                c50 = svmla_f32_x(body, c50, b0, a0);
                c51 = svmla_f32_x(body, c51, b1, a0);
                c52 = svmla_f32_x(body, c52, b2, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha);
            AddProduct(C + 0 * ldc + 1 * F, c01, _alpha);
            AddProduct(C + 0 * ldc + 2 * F, c02, _alpha);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha);
            AddProduct(C + 1 * ldc + 1 * F, c11, _alpha);
            AddProduct(C + 1 * ldc + 2 * F, c12, _alpha);
            AddProduct(C + 2 * ldc + 0 * F, c20, _alpha);
            AddProduct(C + 2 * ldc + 1 * F, c21, _alpha);
            AddProduct(C + 2 * ldc + 2 * F, c22, _alpha);
            AddProduct(C + 3 * ldc + 0 * F, c30, _alpha);
            AddProduct(C + 3 * ldc + 1 * F, c31, _alpha);
            AddProduct(C + 3 * ldc + 2 * F, c32, _alpha);
            AddProduct(C + 4 * ldc + 0 * F, c40, _alpha);
            AddProduct(C + 4 * ldc + 1 * F, c41, _alpha);
            AddProduct(C + 4 * ldc + 2 * F, c42, _alpha);
            AddProduct(C + 5 * ldc + 0 * F, c50, _alpha);
            AddProduct(C + 5 * ldc + 1 * F, c51, _alpha);
            AddProduct(C + 5 * ldc + 2 * F, c52, _alpha);
        }

        static void Kernel6x4nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svfloat32_t zero = svdup_n_f32(0.0f);
            svfloat32_t c00 = zero;
            svfloat32_t c01 = zero;
            svfloat32_t c02 = zero;
            svfloat32_t c03 = zero;
            svfloat32_t c10 = zero;
            svfloat32_t c11 = zero;
            svfloat32_t c12 = zero;
            svfloat32_t c13 = zero;
            svfloat32_t c20 = zero;
            svfloat32_t c21 = zero;
            svfloat32_t c22 = zero;
            svfloat32_t c23 = zero;
            svfloat32_t c30 = zero;
            svfloat32_t c31 = zero;
            svfloat32_t c32 = zero;
            svfloat32_t c33 = zero;
            svfloat32_t c40 = zero;
            svfloat32_t c41 = zero;
            svfloat32_t c42 = zero;
            svfloat32_t c43 = zero;
            svfloat32_t c50 = zero;
            svfloat32_t c51 = zero;
            svfloat32_t c52 = zero;
            svfloat32_t c53 = zero;
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(body, B + 0 * F);
                svfloat32_t b1 = svld1_f32(body, B + 1 * F);
                svfloat32_t b2 = svld1_f32(body, B + 2 * F);
                svfloat32_t b3 = svld1_f32(body, B + 3 * F);
                svfloat32_t a0 = svdup_n_f32(A[0 * lda]);
                c00 = svmla_f32_x(body, c00, b0, a0);
                c01 = svmla_f32_x(body, c01, b1, a0);
                c02 = svmla_f32_x(body, c02, b2, a0);
                c03 = svmla_f32_x(body, c03, b3, a0);
                a0 = svdup_n_f32(A[1 * lda]);
                c10 = svmla_f32_x(body, c10, b0, a0);
                c11 = svmla_f32_x(body, c11, b1, a0);
                c12 = svmla_f32_x(body, c12, b2, a0);
                c13 = svmla_f32_x(body, c13, b3, a0);
                a0 = svdup_n_f32(A[2 * lda]);
                c20 = svmla_f32_x(body, c20, b0, a0);
                c21 = svmla_f32_x(body, c21, b1, a0);
                c22 = svmla_f32_x(body, c22, b2, a0);
                c23 = svmla_f32_x(body, c23, b3, a0);
                a0 = svdup_n_f32(A[3 * lda]);
                c30 = svmla_f32_x(body, c30, b0, a0);
                c31 = svmla_f32_x(body, c31, b1, a0);
                c32 = svmla_f32_x(body, c32, b2, a0);
                c33 = svmla_f32_x(body, c33, b3, a0);
                a0 = svdup_n_f32(A[4 * lda]);
                c40 = svmla_f32_x(body, c40, b0, a0);
                c41 = svmla_f32_x(body, c41, b1, a0);
                c42 = svmla_f32_x(body, c42, b2, a0);
                c43 = svmla_f32_x(body, c43, b3, a0);
                a0 = svdup_n_f32(A[5 * lda]);
                c50 = svmla_f32_x(body, c50, b0, a0);
                c51 = svmla_f32_x(body, c51, b1, a0);
                c52 = svmla_f32_x(body, c52, b2, a0);
                c53 = svmla_f32_x(body, c53, b3, a0);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            AddProduct(C + 0 * ldc + 0 * F, c00, _alpha);
            AddProduct(C + 0 * ldc + 1 * F, c01, _alpha);
            AddProduct(C + 0 * ldc + 2 * F, c02, _alpha);
            AddProduct(C + 0 * ldc + 3 * F, c03, _alpha);
            AddProduct(C + 1 * ldc + 0 * F, c10, _alpha);
            AddProduct(C + 1 * ldc + 1 * F, c11, _alpha);
            AddProduct(C + 1 * ldc + 2 * F, c12, _alpha);
            AddProduct(C + 1 * ldc + 3 * F, c13, _alpha);
            AddProduct(C + 2 * ldc + 0 * F, c20, _alpha);
            AddProduct(C + 2 * ldc + 1 * F, c21, _alpha);
            AddProduct(C + 2 * ldc + 2 * F, c22, _alpha);
            AddProduct(C + 2 * ldc + 3 * F, c23, _alpha);
            AddProduct(C + 3 * ldc + 0 * F, c30, _alpha);
            AddProduct(C + 3 * ldc + 1 * F, c31, _alpha);
            AddProduct(C + 3 * ldc + 2 * F, c32, _alpha);
            AddProduct(C + 3 * ldc + 3 * F, c33, _alpha);
            AddProduct(C + 4 * ldc + 0 * F, c40, _alpha);
            AddProduct(C + 4 * ldc + 1 * F, c41, _alpha);
            AddProduct(C + 4 * ldc + 2 * F, c42, _alpha);
            AddProduct(C + 4 * ldc + 3 * F, c43, _alpha);
            AddProduct(C + 5 * ldc + 0 * F, c50, _alpha);
            AddProduct(C + 5 * ldc + 1 * F, c51, _alpha);
            AddProduct(C + 5 * ldc + 2 * F, c52, _alpha);
            AddProduct(C + 5 * ldc + 3 * F, c53, _alpha);
        }

        SIMD_INLINE void KernelMx4nn(size_t M, size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            switch (M)
            {
            case 1: Kernel1x4nn(K, alpha, A, lda, B, ldb, C, ldc); break;
            case 2: Kernel2x4nn(K, alpha, A, lda, B, ldb, C, ldc); break;
            case 3: Kernel3x4nn(K, alpha, A, lda, B, ldb, C, ldc); break;
            case 4: Kernel4x4nn(K, alpha, A, lda, B, ldb, C, ldc); break;
            case 5: Kernel5x4nn(K, alpha, A, lda, B, ldb, C, ldc); break;
            case 6: Kernel6x4nn(K, alpha, A, lda, B, ldb, C, ldc); break;
            }
        }

        SIMD_INLINE void KernelMx3nn(size_t M, size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            switch (M)
            {
            case 1: Kernel1x3nn(K, alpha, A, lda, B, ldb, C, ldc); break;
            case 2: Kernel2x3nn(K, alpha, A, lda, B, ldb, C, ldc); break;
            case 3: Kernel3x3nn(K, alpha, A, lda, B, ldb, C, ldc); break;
            case 4: Kernel4x1nn(K, alpha, A, lda, B + 0 * F, ldb, C + 0 * F, ldc, F);
                    Kernel4x1nn(K, alpha, A, lda, B + 1 * F, ldb, C + 1 * F, ldc, F);
                    Kernel4x1nn(K, alpha, A, lda, B + 2 * F, ldb, C + 2 * F, ldc, F); break;
            case 5: Kernel5x1nn(K, alpha, A, lda, B + 0 * F, ldb, C + 0 * F, ldc, F);
                    Kernel5x1nn(K, alpha, A, lda, B + 1 * F, ldb, C + 1 * F, ldc, F);
                    Kernel5x1nn(K, alpha, A, lda, B + 2 * F, ldb, C + 2 * F, ldc, F); break;
            case 6: Kernel6x3nn(K, alpha, A, lda, B, ldb, C, ldc); break;
            }
        }

        SIMD_INLINE void KernelMx2nn(size_t M, size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            switch (M)
            {
            case 1: Kernel1x2nn(K, alpha, A, lda, B, ldb, C, ldc); break;
            case 2: Kernel2x2nn(K, alpha, A, lda, B, ldb, C, ldc); break;
            case 3: Kernel3x2nn(K, alpha, A, lda, B, ldb, C, ldc); break;
            case 4: Kernel4x1nn(K, alpha, A, lda, B + 0 * F, ldb, C + 0 * F, ldc, F);
                    Kernel4x1nn(K, alpha, A, lda, B + 1 * F, ldb, C + 1 * F, ldc, F); break;
            case 5: Kernel5x1nn(K, alpha, A, lda, B + 0 * F, ldb, C + 0 * F, ldc, F);
                    Kernel5x1nn(K, alpha, A, lda, B + 1 * F, ldb, C + 1 * F, ldc, F); break;
            case 6: Kernel6x2nn(K, alpha, A, lda, B, ldb, C, ldc); break;
            }
        }

        SIMD_INLINE void KernelMx1nn(size_t M, size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc, size_t n)
        {
            switch (M)
            {
            case 1: Kernel1x1nn(K, alpha, A, lda, B, ldb, C, ldc, n); break;
            case 2: Kernel2x1nn(K, alpha, A, lda, B, ldb, C, ldc, n); break;
            case 3: Kernel3x1nn(K, alpha, A, lda, B, ldb, C, ldc, n); break;
            case 4: Kernel4x1nn(K, alpha, A, lda, B, ldb, C, ldc, n); break;
            case 5: Kernel5x1nn(K, alpha, A, lda, B, ldb, C, ldc, n); break;
            case 6: Kernel6x1nn(K, alpha, A, lda, B, ldb, C, ldc, n); break;
            }
        }

        SIMD_INLINE void MicroKernel(size_t M, size_t N, size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const size_t microN = 4 * F;
            size_t j = 0;
            for (; j + microN <= N; j += microN)
                KernelMx4nn(M, K, alpha, A, lda, B + j, ldb, C + j, ldc);
            if (j + 3 * F <= N)
            {
                KernelMx3nn(M, K, alpha, A, lda, B + j, ldb, C + j, ldc);
                j += 3 * F;
            }
            if (j + 2 * F <= N)
            {
                KernelMx2nn(M, K, alpha, A, lda, B + j, ldb, C + j, ldc);
                j += 2 * F;
            }
            if (j + F <= N)
            {
                KernelMx1nn(M, K, alpha, A, lda, B + j, ldb, C + j, ldc, F);
                j += F;
            }
            if (j < N)
                KernelMx1nn(M, K, alpha, A, lda, B + j, ldb, C + j, ldc, N - j);
        }

        void Gemm32fNN(size_t M, size_t N, size_t K, const float* alpha, const float* A, size_t lda, const float* B, size_t ldb, const float* beta, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const size_t microM = 6;
            const size_t microN = 4 * F;
            const size_t L1 = Base::AlgCacheL1();
            const size_t L2 = Base::AlgCacheL2();
            const size_t L3 = Base::AlgCacheL3();
            size_t macroK = Simd::Min(L1 / sizeof(float) / microN, K);
            if (macroK == 0)
                macroK = K;
            size_t macroM = Simd::RestrictRange(AlignLoAny(L2 / sizeof(float) / macroK, microM), microM, AlignHiAny(M, microM));
            size_t macroN = Simd::RestrictRange(AlignLoAny(L3 / sizeof(float) / macroK, microN), microN, AlignHiAny(N, microN));

            for (size_t j = 0; j < N; j += macroN)
            {
                size_t currentN = Simd::Min(N, j + macroN) - j;
                for (size_t k = 0; k < K; k += macroK)
                {
                    size_t currentK = Simd::Min(K, k + macroK) - k;
                    for (size_t i = 0; i < M; i += macroM)
                    {
                        size_t currentM = Simd::Min(M, i + macroM) - i;
                        if (k == 0)
                            GemmScaleC(currentM, currentN, beta[0], C + i * ldc + j, ldc);
                        for (size_t ii = 0; ii < currentM; ii += microM)
                        {
                            size_t m = Simd::Min(microM, currentM - ii);
                            MicroKernel(m, currentN, currentK, alpha[0], A + (i + ii) * lda + k, lda, B + k * ldb + j, ldb, C + (i + ii) * ldc + j, ldc);
                        }
                    }
                }
            }
        }
    }
#endif
}
