/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_SSE_ENABLE    
    namespace Sse
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

        static void Kernel4x12nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
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
            const size_t o0 = lda * 0;
            const size_t o1 = lda * 1;
            const size_t o2 = lda * 2;
            const size_t o3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + 0 * F);
                b1 = _mm_loadu_ps(B + 1 * F);
                b2 = _mm_loadu_ps(B + 2 * F);
                a0 = _mm_set1_ps(A[o0]);
                c00 = _mm_add_ps(_mm_mul_ps(a0, b0), c00);
                c01 = _mm_add_ps(_mm_mul_ps(a0, b1), c01);
                c02 = _mm_add_ps(_mm_mul_ps(a0, b2), c02);
                a0 = _mm_set1_ps(A[o1]);
                c10 = _mm_add_ps(_mm_mul_ps(a0, b0), c10);
                c11 = _mm_add_ps(_mm_mul_ps(a0, b1), c11);
                c12 = _mm_add_ps(_mm_mul_ps(a0, b2), c12);
                a0 = _mm_set1_ps(A[o2]);
                c20 = _mm_add_ps(_mm_mul_ps(a0, b0), c20);
                c21 = _mm_add_ps(_mm_mul_ps(a0, b1), c21);
                c22 = _mm_add_ps(_mm_mul_ps(a0, b2), c22);
                a0 = _mm_set1_ps(A[o3]);
                c30 = _mm_add_ps(_mm_mul_ps(a0, b0), c30);
                c31 = _mm_add_ps(_mm_mul_ps(a0, b1), c31);
                c32 = _mm_add_ps(_mm_mul_ps(a0, b2), c32);
                B += ldb;
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

        static void Kernel4x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            __m128 c00 = _mm_setzero_ps();
            __m128 c10 = _mm_setzero_ps();
            __m128 c20 = _mm_setzero_ps();
            __m128 c30 = _mm_setzero_ps();
            __m128 c01 = _mm_setzero_ps();
            __m128 c11 = _mm_setzero_ps();
            __m128 c21 = _mm_setzero_ps();
            __m128 c31 = _mm_setzero_ps();
            const size_t o0 = lda * 0;
            const size_t o1 = lda * 1;
            const size_t o2 = lda * 2;
            const size_t o3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            __m128 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + 0 * F);
                b1 = _mm_loadu_ps(B + 1 * F);
                a0 = _mm_set1_ps(A[o0]);
                c00 = _mm_add_ps(_mm_mul_ps(a0, b0), c00);
                c01 = _mm_add_ps(_mm_mul_ps(a0, b1), c01);
                a0 = _mm_set1_ps(A[o1]);
                c10 = _mm_add_ps(_mm_mul_ps(a0, b0), c10);
                c11 = _mm_add_ps(_mm_mul_ps(a0, b1), c11);
                a0 = _mm_set1_ps(A[o2]);
                c20 = _mm_add_ps(_mm_mul_ps(a0, b0), c20);
                c21 = _mm_add_ps(_mm_mul_ps(a0, b1), c21);
                a0 = _mm_set1_ps(A[o3]);
                c30 = _mm_add_ps(_mm_mul_ps(a0, b0), c30);
                c31 = _mm_add_ps(_mm_mul_ps(a0, b1), c31);
                B += ldb;
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

        static void Kernel4x4nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            __m128 c0 = _mm_setzero_ps();
            __m128 c1 = _mm_setzero_ps();
            __m128 c2 = _mm_setzero_ps();
            __m128 c3 = _mm_setzero_ps();
            const size_t o0 = lda * 0;
            const size_t o1 = lda * 1;
            const size_t o2 = lda * 2;
            const size_t o3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            __m128 b0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B);
                c0 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[o0])), c0);
                c1 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[o1])), c1);
                c2 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[o2])), c2);
                c3 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[o3])), c3);
                B += ldb;
                A += sa;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            AddProduct(C + 0 * ldc, _alpha, c0, tail);
            AddProduct(C + 1 * ldc, _alpha, c1, tail);
            AddProduct(C + 2 * ldc, _alpha, c2, tail);
            AddProduct(C + 3 * ldc, _alpha, c3, tail);
        }

        static void Kernel6x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
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
            const size_t o0 = lda * 0;
            const size_t o1 = lda * 1;
            const size_t o2 = lda * 2;
            const size_t o3 = lda * 3;
            const size_t o4 = lda * 4;
            const size_t o5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            __m128 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + 0 * F);
                b1 = _mm_loadu_ps(B + 1 * F);
                a0 = _mm_set1_ps(A[o0]);
                c00 = _mm_add_ps(_mm_mul_ps(a0, b0), c00);
                c01 = _mm_add_ps(_mm_mul_ps(a0, b1), c01);
                a0 = _mm_set1_ps(A[o1]);
                c10 = _mm_add_ps(_mm_mul_ps(a0, b0), c10);
                c11 = _mm_add_ps(_mm_mul_ps(a0, b1), c11);
                a0 = _mm_set1_ps(A[o2]);
                c20 = _mm_add_ps(_mm_mul_ps(a0, b0), c20);
                c21 = _mm_add_ps(_mm_mul_ps(a0, b1), c21);
                a0 = _mm_set1_ps(A[o3]);
                c30 = _mm_add_ps(_mm_mul_ps(a0, b0), c30);
                c31 = _mm_add_ps(_mm_mul_ps(a0, b1), c31);
                a0 = _mm_set1_ps(A[o4]);
                c40 = _mm_add_ps(_mm_mul_ps(a0, b0), c40);
                c41 = _mm_add_ps(_mm_mul_ps(a0, b1), c41);
                a0 = _mm_set1_ps(A[o5]);
                c50 = _mm_add_ps(_mm_mul_ps(a0, b0), c50);
                c51 = _mm_add_ps(_mm_mul_ps(a0, b1), c51);
                B += ldb;
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

        static void Kernel6x4nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            __m128 c0 = _mm_setzero_ps();
            __m128 c1 = _mm_setzero_ps();
            __m128 c2 = _mm_setzero_ps();
            __m128 c3 = _mm_setzero_ps();
            __m128 c4 = _mm_setzero_ps();
            __m128 c5 = _mm_setzero_ps();
            const size_t o0 = lda * 0;
            const size_t o1 = lda * 1;
            const size_t o2 = lda * 2;
            const size_t o3 = lda * 3;
            const size_t o4 = lda * 4;
            const size_t o5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            __m128 b0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B);
                c0 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[o0])), c0);
                c1 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[o1])), c1);
                c2 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[o2])), c2);
                c3 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[o3])), c3);
                c4 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[o4])), c4);
                c5 = _mm_add_ps(_mm_mul_ps(b0, _mm_set1_ps(A[o5])), c5);
                B += ldb;
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

        static void KernelMx12nn(size_t M, size_t N, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            __m128 c[4][3];
            size_t o[4];
            const size_t sa = lda == 1 ? M : 1;
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = _mm_setzero_ps();
                c[i][1] = _mm_setzero_ps();
                c[i][2] = _mm_setzero_ps();
                o[i] = lda * i;
            }
            __m128 b0, b1, b2, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + 0 * F);
                b1 = _mm_loadu_ps(B + 1 * F);
                b2 = _mm_loadu_ps(B + 2 * F);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = _mm_set1_ps(A[o[i]]);
                    c[i][0] = _mm_add_ps(_mm_mul_ps(b0, a0), c[i][0]);
                    c[i][1] = _mm_add_ps(_mm_mul_ps(b1, a0), c[i][1]);
                    c[i][2] = _mm_add_ps(_mm_mul_ps(b2, a0), c[i][2]);
                }
                B += ldb;
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

        static void KernelMx8nn(size_t M, size_t N, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            __m128 c[6][2];
            size_t o[6];
            const size_t sa = lda == 1 ? M : 1;
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = _mm_setzero_ps();
                c[i][1] = _mm_setzero_ps();
                o[i] = lda * i;
            }
            __m128 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + 0 * F);
                b1 = _mm_loadu_ps(B + 1 * F);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = _mm_set1_ps(A[o[i]]);
                    c[i][0] = _mm_add_ps(_mm_mul_ps(b0, a0), c[i][0]);
                    c[i][1] = _mm_add_ps(_mm_mul_ps(b1, a0), c[i][1]);
                }
                B += ldb;
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

        static void KernelMx4nn(size_t M, size_t N, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
#ifdef SIMD_X64_ENABLE
            __m128 c[6];
            size_t o[6];
#else
            __m128 c[4];
            size_t o[4];
#endif
            const size_t sa = lda == 1 ? M : 1;
            for (size_t i = 0; i < M; ++i)
            {
                c[i] = _mm_setzero_ps();
                o[i] = lda * i;
            }
            __m128 b0, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm_loadu_ps(B + 0 * F);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = _mm_set1_ps(A[o[i]]);
                    c[i] = _mm_add_ps(_mm_mul_ps(b0, a0), c[i]);
                }
                B += ldb;
                A += sa;
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            for (size_t i = 0; i < M; ++i)
                AddProduct(C + i * ldc, _alpha, c[i], tail);
        }

        static void PackA(const float * src, size_t stride, size_t M, size_t K, size_t cell, float * dst)
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

        static void PackBnn(const float * B, size_t ldb, size_t K, size_t N, size_t microN, float * pB)
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
                        __m128 mask0 = Sse::LeftNotZero(n - 0 * F);
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
                        __m128 mask0 = Sse::LeftNotZero(n - 0 * F);
                        __m128 mask1 = Sse::LeftNotZero(n - 1 * F);
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
                        __m128 mask0 = Sse::LeftNotZero(n - 0 * F);
                        __m128 mask1 = Sse::LeftNotZero(n - 1 * F);
                        __m128 mask2 = Sse::LeftNotZero(n - 2 * F);
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
            const size_t CACHE_L1_SIZE = 32 * 1024;
            const size_t CACHE_L2_SIZE = 256 * 1024;
            const size_t CACHE_L3_SIZE = 2 * 1024 * 1024;
            typedef Simd::GemmNN<float, size_t> GemmNN;
            GemmNN::Main kernelMM, kernelMT;
            GemmNN::Tail kernelTM, kernelTT;
            size_t microM, microN, L1, L2;
#ifdef SIMD_X64_ENABLE
            if (N < K)
            {
                microM = 6;
                microN = 8;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Kernel6x8nn;
                kernelMT = tail > F ? Kernel6x8nn : Kernel6x4nn;
                kernelTM = KernelMx8nn;
                kernelTT = tail > F ? KernelMx8nn : KernelMx4nn;
            }
            else
            {
                microM = 4;
                microN = 12;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Kernel4x12nn;
                kernelMT = tail > DF ? Kernel4x12nn : (tail > F ? Kernel4x8nn : Kernel4x4nn);
                kernelTM = KernelMx12nn;
                kernelTT = tail > DF ? KernelMx12nn : (tail > F ? KernelMx8nn : KernelMx4nn);
            }
#else
            microM = 4;
            microN = 4;
            kernelMM = Kernel4x4nn;
            kernelMT = Kernel4x4nn;
            kernelTM = KernelMx4nn;
            kernelTT = KernelMx4nn;
#endif
            GemmNN::PackA packA = NULL;
            L1 = N > 4096 ? CACHE_L2_SIZE : CACHE_L1_SIZE;
            L2 = N > 4096 ? CACHE_L3_SIZE : CACHE_L2_SIZE;
            GemmNN gemmNN(M, N, K, microM, microN, L1, L2, CACHE_L3_SIZE, F,
                kernelMM, kernelMT, kernelTM, kernelTT, packA, PackBnn, GemmScaleC, NULL);
            gemmNN.Run(alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
#endif// SIMD_SSE_ENABLE
}
