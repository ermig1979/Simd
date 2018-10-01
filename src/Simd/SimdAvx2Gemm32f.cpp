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
#include "Simd/SimdExtract.h"
#include "Simd/SimdGemm.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE void AddProduct(float * ptr, __m256 value, __m256 alpha)
        {
            _mm256_storeu_ps(ptr, _mm256_fmadd_ps(value, alpha, _mm256_loadu_ps(ptr)));
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

        static void Kernel4x24nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
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
            const float * A0 = A + lda * 0;
            const float * A1 = A + lda * 1;
            const float * A2 = A + lda * 2;
            const float * A3 = A + lda * 3;
            __m256 b0, b1, b2, a0;
            for (size_t k = 0; k < K; ++k)
            {
                _mm_prefetch((char*)B + 384, _MM_HINT_T0);
                b0 = _mm256_loadu_ps(B + 0 * F);
                b1 = _mm256_loadu_ps(B + 1 * F);
                b2 = _mm256_loadu_ps(B + 2 * F);
                a0 = _mm256_set1_ps(*A0++);
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c01 = _mm256_fmadd_ps(a0, b1, c01);
                c02 = _mm256_fmadd_ps(a0, b2, c02);
                a0 = _mm256_set1_ps(*A1++);
                c10 = _mm256_fmadd_ps(a0, b0, c10);
                c11 = _mm256_fmadd_ps(a0, b1, c11);
                c12 = _mm256_fmadd_ps(a0, b2, c12);
                a0 = _mm256_set1_ps(*A2++);
                c20 = _mm256_fmadd_ps(a0, b0, c20);
                c21 = _mm256_fmadd_ps(a0, b1, c21);
                c22 = _mm256_fmadd_ps(a0, b2, c22);
                a0 = _mm256_set1_ps(*A3++);
                c30 = _mm256_fmadd_ps(a0, b0, c30);
                c31 = _mm256_fmadd_ps(a0, b1, c31);
                c32 = _mm256_fmadd_ps(a0, b2, c32);
                B += ldb;
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

        static void Kernel4x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            __m256 c00 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps();
            __m256 c30 = _mm256_setzero_ps();
            __m256 c01 = _mm256_setzero_ps();
            __m256 c11 = _mm256_setzero_ps();
            __m256 c21 = _mm256_setzero_ps();
            __m256 c31 = _mm256_setzero_ps();
            const float * A0 = A + lda * 0;
            const float * A1 = A + lda * 1;
            const float * A2 = A + lda * 2;
            const float * A3 = A + lda * 3;
            __m256 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm256_loadu_ps(B + 0 * F);
                b1 = _mm256_loadu_ps(B + 1 * F);
                a0 = _mm256_set1_ps(*A0++);
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c01 = _mm256_fmadd_ps(a0, b1, c01);
                a0 = _mm256_set1_ps(*A1++);
                c10 = _mm256_fmadd_ps(a0, b0, c10);
                c11 = _mm256_fmadd_ps(a0, b1, c11);
                a0 = _mm256_set1_ps(*A2++);
                c20 = _mm256_fmadd_ps(a0, b0, c20);
                c21 = _mm256_fmadd_ps(a0, b1, c21);
                a0 = _mm256_set1_ps(*A3++);
                c30 = _mm256_fmadd_ps(a0, b0, c30);
                c31 = _mm256_fmadd_ps(a0, b1, c31);
                B += ldb;
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

        static void Kernel4x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            const float * a0 = A + lda * 0;
            const float * a1 = A + lda * 1;
            const float * a2 = A + lda * 2;
            const float * a3 = A + lda * 3;
            __m256 b0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm256_loadu_ps(B);
                c0 = _mm256_fmadd_ps(b0, _mm256_set1_ps(*a0++), c0);
                c1 = _mm256_fmadd_ps(b0, _mm256_set1_ps(*a1++), c1);
                c2 = _mm256_fmadd_ps(b0, _mm256_set1_ps(*a2++), c2);
                c3 = _mm256_fmadd_ps(b0, _mm256_set1_ps(*a3++), c3);
                B += ldb;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
            AddProduct(C + 0 * ldc, _alpha, c0, tail);
            AddProduct(C + 1 * ldc, _alpha, c1, tail);
            AddProduct(C + 2 * ldc, _alpha, c2, tail);
            AddProduct(C + 3 * ldc, _alpha, c3, tail);
        }

        static void Kernel6x16nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
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
            const float * A0 = A + lda * 0;
            const float * A1 = A + lda * 1;
            const float * A2 = A + lda * 2;
            const float * A3 = A + lda * 3;
            const float * A4 = A + lda * 4;
            const float * A5 = A + lda * 5;
            __m256 b0, b1, a0, a1;
            for (size_t k = 0; k < K; k++)
            {
                _mm_prefetch((char*)B + 512, _MM_HINT_T0);
                b0 = _mm256_loadu_ps(B + 0 * F);
                b1 = _mm256_loadu_ps(B + 1 * F);
                a0 = _mm256_set1_ps(*A0++);
                a1 = _mm256_set1_ps(*A1++);
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c01 = _mm256_fmadd_ps(a0, b1, c01);
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c11 = _mm256_fmadd_ps(a1, b1, c11);
                a0 = _mm256_set1_ps(*A2++);
                a1 = _mm256_set1_ps(*A3++);
                c20 = _mm256_fmadd_ps(a0, b0, c20);
                c21 = _mm256_fmadd_ps(a0, b1, c21);
                c30 = _mm256_fmadd_ps(a1, b0, c30);
                c31 = _mm256_fmadd_ps(a1, b1, c31);
                a0 = _mm256_set1_ps(*A4++);
                a1 = _mm256_set1_ps(*A5++);
                c40 = _mm256_fmadd_ps(a0, b0, c40);
                c41 = _mm256_fmadd_ps(a0, b1, c41);
                c50 = _mm256_fmadd_ps(a1, b0, c50);
                c51 = _mm256_fmadd_ps(a1, b1, c51);
                B += ldb;
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

        static void Kernel6x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            __m256 c4 = _mm256_setzero_ps();
            __m256 c5 = _mm256_setzero_ps();
            const float * a0 = A + lda * 0;
            const float * a1 = A + lda * 1;
            const float * a2 = A + lda * 2;
            const float * a3 = A + lda * 3;
            const float * a4 = A + lda * 4;
            const float * a5 = A + lda * 5;
            __m256 b0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm256_loadu_ps(B);
                c0 = _mm256_fmadd_ps(b0, _mm256_set1_ps(*a0++), c0);
                c1 = _mm256_fmadd_ps(b0, _mm256_set1_ps(*a1++), c1);
                c2 = _mm256_fmadd_ps(b0, _mm256_set1_ps(*a2++), c2);
                c3 = _mm256_fmadd_ps(b0, _mm256_set1_ps(*a3++), c3);
                c4 = _mm256_fmadd_ps(b0, _mm256_set1_ps(*a4++), c4);
                c5 = _mm256_fmadd_ps(b0, _mm256_set1_ps(*a5++), c5);
                B += ldb;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
            AddProduct(C + 0 * ldc, _alpha, c0, tail);
            AddProduct(C + 1 * ldc, _alpha, c1, tail);
            AddProduct(C + 2 * ldc, _alpha, c2, tail);
            AddProduct(C + 3 * ldc, _alpha, c3, tail);
            AddProduct(C + 4 * ldc, _alpha, c4, tail);
            AddProduct(C + 5 * ldc, _alpha, c5, tail);
        }

        static void KernelMx24nn(size_t M, size_t N, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            __m256 c[4][3];
            const float * a[4];
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = _mm256_setzero_ps();
                c[i][1] = _mm256_setzero_ps();
                c[i][2] = _mm256_setzero_ps();
                a[i] = A + lda * i;
            }
            __m256 b0, b1, b2, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm256_loadu_ps(B + 0 * F);
                b1 = _mm256_loadu_ps(B + 1 * F);
                b2 = _mm256_loadu_ps(B + 2 * F);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = _mm256_set1_ps(*a[i]++);
                    c[i][0] = _mm256_add_ps(_mm256_mul_ps(b0, a0), c[i][0]);
                    c[i][1] = _mm256_add_ps(_mm256_mul_ps(b1, a0), c[i][1]);
                    c[i][2] = _mm256_add_ps(_mm256_mul_ps(b2, a0), c[i][2]);
                }
                B += ldb;
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

        static void KernelMx16nn(size_t M, size_t N, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            __m256 c[6][2];
            const float * a[6];
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = _mm256_setzero_ps();
                c[i][1] = _mm256_setzero_ps();
                a[i] = A + lda * i;
            }
            __m256 b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm256_loadu_ps(B + 0 * F);
                b1 = _mm256_loadu_ps(B + 1 * F);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = _mm256_set1_ps(*a[i]++);
                    c[i][0] = _mm256_fmadd_ps(b0, a0, c[i][0]);
                    c[i][1] = _mm256_fmadd_ps(b1, a0, c[i][1]);
                }
                B += ldb;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
            for (size_t i = 0; i < M; ++i)
            {
                AddProduct(C + 0 * F, _alpha, c[i][0]);
                AddProduct(C + 1 * F, _alpha, c[i][1], tail);
                C += ldc;
            }
        }

        static void KernelMx8nn(size_t M, size_t N, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            __m256 c[4];
            const float * a[4];
            for (size_t i = 0; i < M; ++i)
            {
                c[i] = _mm256_setzero_ps();
                a[i] = A + lda * i;
            }
            __m256 b0, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = _mm256_loadu_ps(B + 0 * F);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = _mm256_set1_ps(*a[i]++);
                    c[i] = _mm256_fmadd_ps(b0, a0, c[i]);
                }
                B += ldb;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
            for (size_t i = 0; i < M; ++i)
                AddProduct(C + i * ldc, _alpha, c[i], tail);
        }

        static void PackA(const float * src, size_t stride, size_t M, size_t K, size_t cell, float * dst)
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
            if (K > 4096)
            {
                microM = 6;
                microN = 16;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Kernel6x16nn;
                kernelMT = tail > F ? Kernel6x16nn : Kernel6x8nn;
                kernelTM = KernelMx16nn;
                kernelTT = tail > F ? KernelMx16nn : KernelMx8nn;
            }
            else
            {
                microM = 4;
                microN = 24;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Kernel4x24nn;
                kernelMT = tail > DF ? Kernel4x24nn : (tail > F ? Kernel4x16nn : Kernel4x8nn);
                kernelTM = KernelMx24nn;
                kernelTT = tail > DF ? KernelMx24nn : (tail > F ? KernelMx16nn : KernelMx8nn);
            }
#else
            microM = 4;
            microN = 8;
            kernelMM = Kernel4x8;
            kernelMT = Kernel4x8;
            kernelTM = KernelMx8;
            kernelTT = KernelMx8;
#endif
            L1 = N > 4096 ? CACHE_L2_SIZE : CACHE_L1_SIZE;
            L2 = N > 4096 ? CACHE_L3_SIZE : CACHE_L2_SIZE;
            GemmNN gemmNN(M, N, K, microM, microN, L1, L2, CACHE_L3_SIZE, F,
                kernelMM, kernelMT, kernelTM, kernelTT, Avx::GemmScaleC, Avx::GemmPackB, NULL);
            gemmNN.Run(alpha, A, lda, B, ldb, beta, C, ldc);
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
            _mm_storeu_ps(dst, _mm_fmadd_ps(alpha, sum128, _mm_loadu_ps(dst)));
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
                c00 = _mm256_fmadd_ps(a0, b0, c00);
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                a0 = _mm256_and_ps(tail, _mm256_loadu_ps(A0 + k));
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_fmadd_ps(a0, b0, c00);
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
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                b0 = _mm256_loadu_ps(B1 + k);
                c01 = _mm256_fmadd_ps(a0, b0, c01);
                b0 = _mm256_loadu_ps(B2 + k);
                c02 = _mm256_fmadd_ps(a0, b0, c02);
                b0 = _mm256_loadu_ps(B3 + k);
                c03 = _mm256_fmadd_ps(a0, b0, c03);
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                a0 = _mm256_and_ps(tail, _mm256_loadu_ps(A0 + k));
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                b0 = _mm256_loadu_ps(B1 + k);
                c01 = _mm256_fmadd_ps(a0, b0, c01);
                b0 = _mm256_loadu_ps(B2 + k);
                c02 = _mm256_fmadd_ps(a0, b0, c02);
                b0 = _mm256_loadu_ps(B3 + k);
                c03 = _mm256_fmadd_ps(a0, b0, c03);
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
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c10 = _mm256_fmadd_ps(a1, b0, c10);
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                a0 = _mm256_and_ps(tail, _mm256_loadu_ps(A0 + k));
                a1 = _mm256_and_ps(tail, _mm256_loadu_ps(A1 + k));
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c10 = _mm256_fmadd_ps(a1, b0, c10);
            }
            C[0 * ldc] += alpha * Avx::ExtractSum(c00);
            C[1 * ldc] += alpha * Avx::ExtractSum(c10);
        }

        static void Kernel2x4x8nt(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            size_t K8 = K & (~8);
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
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                b0 = _mm256_loadu_ps(B1 + k);
                c01 = _mm256_fmadd_ps(a0, b0, c01);
                c11 = _mm256_fmadd_ps(a1, b0, c11);
                b0 = _mm256_loadu_ps(B2 + k);
                c02 = _mm256_fmadd_ps(a0, b0, c02);
                c12 = _mm256_fmadd_ps(a1, b0, c12);
                b0 = _mm256_loadu_ps(B3 + k);
                c03 = _mm256_fmadd_ps(a0, b0, c03);
                c13 = _mm256_fmadd_ps(a1, b0, c13);
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                a0 = _mm256_and_ps(tail, _mm256_loadu_ps(A0 + k));
                a1 = _mm256_and_ps(tail, _mm256_loadu_ps(A1 + k));
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                b0 = _mm256_loadu_ps(B1 + k);
                c01 = _mm256_fmadd_ps(a0, b0, c01);
                c11 = _mm256_fmadd_ps(a1, b0, c11);
                b0 = _mm256_loadu_ps(B2 + k);
                c02 = _mm256_fmadd_ps(a0, b0, c02);
                c12 = _mm256_fmadd_ps(a1, b0, c12);
                b0 = _mm256_loadu_ps(B3 + k);
                c03 = _mm256_fmadd_ps(a0, b0, c03);
                c13 = _mm256_fmadd_ps(a1, b0, c13);
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
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c20 = _mm256_fmadd_ps(a2, b0, c20);
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                a0 = _mm256_and_ps(tail, _mm256_loadu_ps(A0 + k));
                a1 = _mm256_and_ps(tail, _mm256_loadu_ps(A1 + k));
                a2 = _mm256_and_ps(tail, _mm256_loadu_ps(A2 + k));
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c20 = _mm256_fmadd_ps(a2, b0, c20);
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
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c20 = _mm256_fmadd_ps(a2, b0, c20);
                b0 = _mm256_loadu_ps(B1 + k);
                c01 = _mm256_fmadd_ps(a0, b0, c01);
                c11 = _mm256_fmadd_ps(a1, b0, c11);
                c21 = _mm256_fmadd_ps(a2, b0, c21);
                b0 = _mm256_loadu_ps(B2 + k);
                c02 = _mm256_fmadd_ps(a0, b0, c02);
                c12 = _mm256_fmadd_ps(a1, b0, c12);
                c22 = _mm256_fmadd_ps(a2, b0, c22);
                b0 = _mm256_loadu_ps(B3 + k);
                c03 = _mm256_fmadd_ps(a0, b0, c03);
                c13 = _mm256_fmadd_ps(a1, b0, c13);
                c23 = _mm256_fmadd_ps(a2, b0, c23);
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                a0 = _mm256_and_ps(tail, _mm256_loadu_ps(A0 + k));
                a1 = _mm256_and_ps(tail, _mm256_loadu_ps(A1 + k));
                a2 = _mm256_and_ps(tail, _mm256_loadu_ps(A2 + k));
                b0 = _mm256_loadu_ps(B0 + k);
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c20 = _mm256_fmadd_ps(a2, b0, c20);
                b0 = _mm256_loadu_ps(B1 + k);
                c01 = _mm256_fmadd_ps(a0, b0, c01);
                c11 = _mm256_fmadd_ps(a1, b0, c11);
                c21 = _mm256_fmadd_ps(a2, b0, c21);
                b0 = _mm256_loadu_ps(B2 + k);
                c02 = _mm256_fmadd_ps(a0, b0, c02);
                c12 = _mm256_fmadd_ps(a1, b0, c12);
                c22 = _mm256_fmadd_ps(a2, b0, c22);
                b0 = _mm256_loadu_ps(B3 + k);
                c03 = _mm256_fmadd_ps(a0, b0, c03);
                c13 = _mm256_fmadd_ps(a1, b0, c13);
                c23 = _mm256_fmadd_ps(a2, b0, c23);
            }
            __m128 _alpha = _mm_set1_ps(alpha);
            Add4ExtractedSums(c00, c01, c02, c03, _alpha, C + 0 * ldc);
            Add4ExtractedSums(c10, c11, c12, c13, _alpha, C + 1 * ldc);
            Add4ExtractedSums(c20, c21, c22, c23, _alpha, C + 2 * ldc);
        }

        void Gemm32fNT(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            const size_t CACHE_L1_SIZE = 32 * 1024;
            const size_t CACHE_L2_SIZE = 256 * 1024;
            const size_t CACHE_L3_SIZE = 2 * 1024 * 1024;
            typedef Simd::GemmNT<float> GemmNT;
#ifdef SIMD_X64_ENABLE
            GemmNT gemmNT(M, N, K, CACHE_L1_SIZE, CACHE_L2_SIZE, CACHE_L3_SIZE, F, Avx::GemmScaleC,
                Kernel1x1x8nt, Kernel1x4x8nt, Kernel2x1x8nt, Kernel2x4x8nt, Kernel3x1x8nt, Kernel3x4x8nt, NULL, NULL);
#else
            GemmNT gemmNT(M, N, K, CACHE_L1_SIZE, CACHE_L2_SIZE, CACHE_L3_SIZE, F, Sse::GemmScaleC,
                Kernel1x1x8nt, Kernel1x4x8nt, NULL, NULL, NULL, NULL, NULL, NULL);
#endif
            gemmNT.Run(alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
