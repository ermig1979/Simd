/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdArray.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE void MulBy(float * ptr, __m256 value)
        {
            _mm256_storeu_ps(ptr, _mm256_mul_ps(_mm256_loadu_ps(ptr), value));
        }

        SIMD_INLINE void AddTo(float * ptr, __m256 value)
        {
            _mm256_storeu_ps(ptr, _mm256_add_ps(_mm256_loadu_ps(ptr), value));
        }

        SIMD_INLINE void AddProduct(float * ptr, __m256 value, __m256 alpha)
        {
            _mm256_storeu_ps(ptr, _mm256_fmadd_ps(value, alpha, _mm256_loadu_ps(ptr)));
        }

        static void Kernel4x8(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register __m256 c0 = _mm256_setzero_ps();
            register __m256 c1 = _mm256_setzero_ps();
            register __m256 c2 = _mm256_setzero_ps();
            register __m256 c3 = _mm256_setzero_ps();
            const float * a0 = A + lda * 0;
            const float * a1 = A + lda * 1;
            const float * a2 = A + lda * 2;
            const float * a3 = A + lda * 3;
            register __m256 b0;
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
            AddProduct(C + 0 * ldc, _alpha, c0);
            AddProduct(C + 1 * ldc, _alpha, c1);
            AddProduct(C + 2 * ldc, _alpha, c2);
            AddProduct(C + 3 * ldc, _alpha, c3);
        }

        static void KernelMx8(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register __m256 c[4];
            register const float * a[4];
            for (size_t i = 0; i < M; ++i)
            {
                c[i] = _mm256_setzero_ps();
                a[i] = A + lda * i;
            }
            register __m256 b0, a0;
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
                AddProduct(C + i*ldc, _alpha, c[i]);
        }

        static void Kernel4x24(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register __m256 c00 = _mm256_setzero_ps();
            register __m256 c10 = _mm256_setzero_ps();
            register __m256 c20 = _mm256_setzero_ps();
            register __m256 c30 = _mm256_setzero_ps();
            register __m256 c01 = _mm256_setzero_ps();
            register __m256 c11 = _mm256_setzero_ps();
            register __m256 c21 = _mm256_setzero_ps();
            register __m256 c31 = _mm256_setzero_ps();
            register __m256 c02 = _mm256_setzero_ps();
            register __m256 c12 = _mm256_setzero_ps();
            register __m256 c22 = _mm256_setzero_ps();
            register __m256 c32 = _mm256_setzero_ps();
            const float * A0 = A + lda * 0;
            const float * A1 = A + lda * 1;
            const float * A2 = A + lda * 2;
            const float * A3 = A + lda * 3;
            register __m256 b0, b1, b2, a0;
            for (size_t k = 0; k < K; k++)
            {
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
            AddProduct(C + 2 * F, _alpha, c02);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11);
            AddProduct(C + 2 * F, _alpha, c12);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21);
            AddProduct(C + 2 * F, _alpha, c22);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31);
            AddProduct(C + 2 * F, _alpha, c32);
        }

        static void KernelMx24(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register __m256 c[4][3];
            register const float * a[4];
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = _mm256_setzero_ps();
                c[i][1] = _mm256_setzero_ps();
                c[i][2] = _mm256_setzero_ps();
                a[i] = A + lda * i;
            }
            register __m256 b0, b1, b2, a0;
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
                AddProduct(C + 2 * F, _alpha, c[i][2]);
                C += ldc;
            }
        }

        static void Kernel6x16(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register __m256 c00 = _mm256_setzero_ps();
            register __m256 c10 = _mm256_setzero_ps();
            register __m256 c20 = _mm256_setzero_ps();
            register __m256 c30 = _mm256_setzero_ps();
            register __m256 c40 = _mm256_setzero_ps();
            register __m256 c50 = _mm256_setzero_ps();
            register __m256 c01 = _mm256_setzero_ps();
            register __m256 c11 = _mm256_setzero_ps();
            register __m256 c21 = _mm256_setzero_ps();
            register __m256 c31 = _mm256_setzero_ps();
            register __m256 c41 = _mm256_setzero_ps();
            register __m256 c51 = _mm256_setzero_ps();
            const float * A0 = A + lda * 0;
            const float * A1 = A + lda * 1;
            const float * A2 = A + lda * 2;
            const float * A3 = A + lda * 3;
            const float * A4 = A + lda * 4;
            const float * A5 = A + lda * 5;
            register __m256 b0, b1, a0;
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
                a0 = _mm256_set1_ps(*A4++);
                c40 = _mm256_fmadd_ps(a0, b0, c40);
                c41 = _mm256_fmadd_ps(a0, b1, c41);
                a0 = _mm256_set1_ps(*A5++);
                c50 = _mm256_fmadd_ps(a0, b0, c50);
                c51 = _mm256_fmadd_ps(a0, b1, c51);
                B += ldb;
            }
            __m256 _alpha = _mm256_set1_ps(alpha);
            AddProduct(C + 0 * F, _alpha, c00);
            AddProduct(C + 1 * F, _alpha, c01);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c10);
            AddProduct(C + 1 * F, _alpha, c11);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c20);
            AddProduct(C + 1 * F, _alpha, c21);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c30);
            AddProduct(C + 1 * F, _alpha, c31);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c40);
            AddProduct(C + 1 * F, _alpha, c41);
            C += ldc;
            AddProduct(C + 0 * F, _alpha, c50);
            AddProduct(C + 1 * F, _alpha, c51);
        }

        static void KernelMx16(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc)
        {
            register __m256 c[6][2];
            register const float * a[6];
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = _mm256_setzero_ps();
                c[i][1] = _mm256_setzero_ps();
                a[i] = A + lda * i;
            }
            register __m256 b0, b1, a0;
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
                AddProduct(C + 1 * F, _alpha, c[i][1]);
                C += ldc;
            }
        }

        static void MulBy(float * ptr, size_t stride, size_t height, size_t width, float value)
        {
            size_t aligned = AlignLo(width, QF);
            size_t partial = AlignLo(width, F);
            __m256 _value = _mm256_set1_ps(value);
            for (size_t i = 0; i < height; ++i)
            {
                size_t j = 0;
                for (; j < aligned; j += QF)
                {
                    MulBy(ptr + j + F * 0, _value);
                    MulBy(ptr + j + F * 1, _value);
                    MulBy(ptr + j + F * 2, _value);
                    MulBy(ptr + j + F * 3, _value);
                }
                for (; j < partial; j += F)
                    MulBy(ptr + j, _value);
                for (; j < width; ++j)
                    ptr[j] *= value;
                ptr += stride;
            }
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

        static void PackB(const float * src, size_t srcStride, size_t K, size_t N, size_t cell, float * dst)
        {
            for (size_t j = 0; j < N; j += cell)
            {
                size_t n = Simd::Min(cell, N - j);
                if (n == cell && cell == 8)
                {
                    for (size_t k = 0; k < K; ++k)
                    {
                        const float * psrc = src + k * srcStride;
                        _mm256_storeu_ps(dst + 0 * F, _mm256_loadu_ps(psrc + 0 * F));
                        dst += 8;
                    }
                }
                else if (n == cell && cell == 16)
                {
                    for (size_t k = 0; k < K; ++k)
                    {
                        const float * psrc = src + k * srcStride;
                        _mm256_storeu_ps(dst + 0*F, _mm256_loadu_ps(psrc + 0 * F));
                        _mm256_storeu_ps(dst + 1*F, _mm256_loadu_ps(psrc + 1 * F));
                        dst += 16;
                    }
                }
                else if (n == cell && cell == 24)
                {
                    for (size_t k = 0; k < K; ++k)
                    {
                        const float * psrc = src + k * srcStride;
                        _mm256_storeu_ps(dst + 0 * F, _mm256_loadu_ps(psrc + 0 * F));
                        _mm256_storeu_ps(dst + 1 * F, _mm256_loadu_ps(psrc + 1 * F));
                        _mm256_storeu_ps(dst + 2 * F, _mm256_loadu_ps(psrc + 2 * F));
                        dst += 24;
                    }
                }
                else
                {
                    for (size_t k = 0; k < K; ++k)
                    {
                        const float * psrc = src + k * srcStride;
                        size_t c = 0;
                        for (; c < n; ++c)
                            *(dst++) = *(psrc++);
                        for (; c < cell; ++c)
                            *(dst++) = 0;
                    }
                }
                src += cell;
            }
        }

        class Gemm32fAlg
        {
            typedef void (*MicroKernelPtr)(size_t M, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc);
            Array<float> _A, _B;
            size_t _lda, _ldb, _microM, _microN, _macroM, _macroN;
            MicroKernelPtr _microKernelMain, _microKernelEdge;
 
            void Init(size_t M, size_t N, size_t K)
            {
#ifdef SIMD_X64_ENABLE
                if (true)
                {
                    _microM = 6;
                    _microN = 16;
                    _microKernelMain = Kernel6x16;
                    _microKernelEdge = KernelMx16;
                }
                else
                {
                    _microM = 4;
                    _microN = 24;
                    _microKernelMain = Kernel4x24;
                    _microKernelEdge = KernelMx24;
                }
#else
                _microM = 4;
                _microN = 8;
                _microKernelMain = Kernel4x8;
                _microKernelEdge = KernelMx8;
#endif
                _macroM = 256 / _microM * _microM;
                _macroN = 128 / _microN * _microN;
                _lda = AlignHi(K, F);
                _ldb = AlignHiAny(N, _microN);

                _A.Resize(_lda*_macroM);
                _B.Resize(_ldb*K);
            }

            void MacroKernel(size_t M, size_t N, size_t K, float alpha, const float * A, size_t lda, const float * Ap, const float * B, size_t ldb, const float * Bp, float beta, float * C, size_t ldc)
            {
                MulBy(C, ldc, M, N, beta);

                size_t MA = AlignLoAny(M, _microM);
                size_t i = 0;
                for (; i < MA; i += _microM)
                    for (size_t j = 0; j < N; j += _microN)
                        _microKernelMain(M, K, alpha, A + i * lda, lda, Bp + j * K, _microN, C + i * ldc + j, ldc);
                if (i < M)
                    for (size_t j = 0; j < N; j += _microN)
                        _microKernelEdge(M - MA, K, alpha, A + i * lda, lda, Bp + j * K, _microN, C + i * ldc + j, ldc);
            }

        public:
            void Run(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
            {
                Init(M, N, K);

                PackB(B, ldb, K, N, _microN, _B.data);
                for (size_t i = 0; i < M; i += _macroM)
                {
                    size_t macroM = Simd::Min(M, i + _macroM) - i;
                    //PackA(A + i * lda, lda, macroM, K, _microM, _A.data);
                    for (size_t j = 0; j < N; j += _macroN)
                    {
                        size_t macroN = Simd::Min(N, j + _macroN) - j;
                        MacroKernel(macroM, macroN, K, *alpha, A + i * lda, lda, _A.data, B + j, ldb, _B.data + j*K, *beta, C + i * ldc + j, ldc);
                    }
                }
            }
        };

        void Gemm32fNN(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            Gemm32fAlg alg;
            alg.Run(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
