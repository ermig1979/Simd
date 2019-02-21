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

        static void Kernel4x12nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
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
            const size_t o0 = lda * 0;
            const size_t o1 = lda * 1;
            const size_t o2 = lda * 2;
            const size_t o3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + 0 * F);
                b1 = Load<false>(B + 1 * F);
                b2 = Load<false>(B + 2 * F);
                a0 = vdupq_n_f32(A[o0]);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                c02 = vmlaq_f32(c02, a0, b2);
                a0 = vdupq_n_f32(A[o1]);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                c12 = vmlaq_f32(c12, a0, b2);
                a0 = vdupq_n_f32(A[o2]);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                c22 = vmlaq_f32(c22, a0, b2);
                a0 = vdupq_n_f32(A[o3]);
                c30 = vmlaq_f32(c30, a0, b0);
                c31 = vmlaq_f32(c31, a0, b1);
                c32 = vmlaq_f32(c32, a0, b2);
                B += ldb;
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

        static void Kernel4x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c00 = vdupq_n_f32(0);
            float32x4_t c10 = vdupq_n_f32(0);
            float32x4_t c20 = vdupq_n_f32(0);
            float32x4_t c30 = vdupq_n_f32(0);
            float32x4_t c01 = vdupq_n_f32(0);
            float32x4_t c11 = vdupq_n_f32(0);
            float32x4_t c21 = vdupq_n_f32(0);
            float32x4_t c31 = vdupq_n_f32(0);
            const size_t o0 = lda * 0;
            const size_t o1 = lda * 1;
            const size_t o2 = lda * 2;
            const size_t o3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            float32x4_t b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + 0 * F);
                b1 = Load<false>(B + 1 * F);
                a0 = vdupq_n_f32(A[o0]);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                a0 = vdupq_n_f32(A[o1]);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                a0 = vdupq_n_f32(A[o2]);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                a0 = vdupq_n_f32(A[o3]);
                c30 = vmlaq_f32(c30, a0, b0);
                c31 = vmlaq_f32(c31, a0, b1);
                B += ldb;
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

        static void Kernel4x4nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c0 = vdupq_n_f32(0);
            float32x4_t c1 = vdupq_n_f32(0);
            float32x4_t c2 = vdupq_n_f32(0);
            float32x4_t c3 = vdupq_n_f32(0);
            const size_t o0 = lda * 0;
            const size_t o1 = lda * 1;
            const size_t o2 = lda * 2;
            const size_t o3 = lda * 3;
            const size_t sa = lda == 1 ? 4 : 1;
            float32x4_t b0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B);
                c0 = vmlaq_f32(c0, b0, vdupq_n_f32(A[o0]));
                c1 = vmlaq_f32(c1, b0, vdupq_n_f32(A[o1]));
                c2 = vmlaq_f32(c2, b0, vdupq_n_f32(A[o2]));
                c3 = vmlaq_f32(c3, b0, vdupq_n_f32(A[o3]));
                B += ldb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
            AddProduct(C + 0 * ldc, _alpha, c0, tail);
            AddProduct(C + 1 * ldc, _alpha, c1, tail);
            AddProduct(C + 2 * ldc, _alpha, c2, tail);
            AddProduct(C + 3 * ldc, _alpha, c3, tail);
        }

        static void Kernel6x8nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
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
            const size_t o0 = lda * 0;
            const size_t o1 = lda * 1;
            const size_t o2 = lda * 2;
            const size_t o3 = lda * 3;
            const size_t o4 = lda * 4;
            const size_t o5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            float32x4_t b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + 0 * F);
                b1 = Load<false>(B + 1 * F);
                a0 = vdupq_n_f32(A[o0]);
                c00 = vmlaq_f32(c00, a0, b0);
                c01 = vmlaq_f32(c01, a0, b1);
                a0 = vdupq_n_f32(A[o1]);
                c10 = vmlaq_f32(c10, a0, b0);
                c11 = vmlaq_f32(c11, a0, b1);
                a0 = vdupq_n_f32(A[o2]);
                c20 = vmlaq_f32(c20, a0, b0);
                c21 = vmlaq_f32(c21, a0, b1);
                a0 = vdupq_n_f32(A[o3]);
                c30 = vmlaq_f32(c30, a0, b0);
                c31 = vmlaq_f32(c31, a0, b1);
                a0 = vdupq_n_f32(A[o4]);
                c40 = vmlaq_f32(c40, a0, b0);
                c41 = vmlaq_f32(c41, a0, b1);
                a0 = vdupq_n_f32(A[o5]);
                c50 = vmlaq_f32(c50, a0, b0);
                c51 = vmlaq_f32(c51, a0, b1);
                B += ldb;
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

        static void Kernel6x4nn(size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c0 = vdupq_n_f32(0);
            float32x4_t c1 = vdupq_n_f32(0);
            float32x4_t c2 = vdupq_n_f32(0);
            float32x4_t c3 = vdupq_n_f32(0);
            float32x4_t c4 = vdupq_n_f32(0);
            float32x4_t c5 = vdupq_n_f32(0);
            const size_t o0 = lda * 0;
            const size_t o1 = lda * 1;
            const size_t o2 = lda * 2;
            const size_t o3 = lda * 3;
            const size_t o4 = lda * 4;
            const size_t o5 = lda * 5;
            const size_t sa = lda == 1 ? 6 : 1;
            float32x4_t b0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B);
                c0 = vmlaq_f32(c0, b0, vdupq_n_f32(A[o0]));
                c1 = vmlaq_f32(c1, b0, vdupq_n_f32(A[o1]));
                c2 = vmlaq_f32(c2, b0, vdupq_n_f32(A[o2]));
                c3 = vmlaq_f32(c3, b0, vdupq_n_f32(A[o3]));
                c4 = vmlaq_f32(c4, b0, vdupq_n_f32(A[o4]));
                c5 = vmlaq_f32(c5, b0, vdupq_n_f32(A[o5]));
                B += ldb;
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

        static void KernelMx12nn(size_t M, size_t N, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c[4][3];
            size_t o[4];
            const size_t sa = lda == 1 ? M : 1;
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = vdupq_n_f32(0);
                c[i][1] = vdupq_n_f32(0);
                c[i][2] = vdupq_n_f32(0);
                o[i] = lda * i;
            }
            float32x4_t b0, b1, b2, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + 0 * F);
                b1 = Load<false>(B + 1 * F);
                b2 = Load<false>(B + 2 * F);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = vdupq_n_f32(A[o[i]]);
                    c[i][0] = vmlaq_f32(c[i][0], b0, a0);
                    c[i][1] = vmlaq_f32(c[i][1], b1, a0);
                    c[i][2] = vmlaq_f32(c[i][2], b2, a0);
                }
                B += ldb;
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

        static void KernelMx8nn(size_t M, size_t N, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
            float32x4_t c[6][2];
            size_t o[6];
            const size_t sa = lda == 1 ? M : 1;
            for (size_t i = 0; i < M; ++i)
            {
                c[i][0] = vdupq_n_f32(0);
                c[i][1] = vdupq_n_f32(0);
                o[i] = lda * i;
            }
            float32x4_t b0, b1, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + 0 * F);
                b1 = Load<false>(B + 1 * F);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = vdupq_n_f32(A[o[i]]);
                    c[i][0] = vmlaq_f32(c[i][0], b0, a0);
                    c[i][1] = vmlaq_f32(c[i][1], b1, a0);
                }
                B += ldb;
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

        static void KernelMx4nn(size_t M, size_t N, size_t K, float alpha, const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, size_t tail)
        {
#ifdef SIMD_X64_ENABLE
            float32x4_t c[6];
            size_t o[6];
#else
            float32x4_t c[4];
            size_t o[4];
#endif
            const size_t sa = lda == 1 ? M : 1;
            for (size_t i = 0; i < M; ++i)
            {
                c[i] = vdupq_n_f32(0);
                o[i] = lda * i;
            }
            float32x4_t b0, a0;
            for (size_t k = 0; k < K; k++)
            {
                b0 = Load<false>(B + 0 * F);
                for (size_t i = 0; i < M; ++i)
                {
                    a0 = vdupq_n_f32(A[o[i]]);
                    c[i] = vmlaq_f32(c[i], b0, a0);
                }
                B += ldb;
                A += sa;
            }
            float32x4_t _alpha = vdupq_n_f32(alpha);
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
            GemmNN::PackA packA = PackA;
            L1 = N > 4096 ? CACHE_L2_SIZE : CACHE_L1_SIZE;
            L2 = N > 4096 ? CACHE_L3_SIZE : CACHE_L2_SIZE;
            GemmNN gemmNN(M, N, K, microM, microN, L1, L2, CACHE_L3_SIZE, F,
                kernelMM, kernelMT, kernelTM, kernelTT, packA, PackBnn, GemmScaleC, NULL);
            gemmNN.Run(alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
#endif// SIMD_NEON_ENABLE
}
