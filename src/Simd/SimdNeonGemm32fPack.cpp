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
#ifdef SIMD_NEON_ENABLE 
    namespace Neon
    {
        SIMD_INLINE void GemmPackA_4x4(const float* src, size_t stride, float* dst)
        {
            float32x4x4_t dst0;
            dst0.val[0] = Load<false>(src + 0 * stride);
            dst0.val[1] = Load<false>(src + 1 * stride);
            dst0.val[2] = Load<false>(src + 2 * stride);
            dst0.val[3] = Load<false>(src + 3 * stride);
            Store4<false>(dst, dst0);
        }

        SIMD_INLINE void GemmPackA_6x4(const float* src, size_t stride, float* dst)
        {
            float32x4_t src0 = Load<false>(src + 0 * stride);
            float32x4_t src1 = Load<false>(src + 1 * stride);
            float32x4_t src2 = Load<false>(src + 2 * stride);
            float32x4_t src3 = Load<false>(src + 3 * stride);
            float32x4_t src4 = Load<false>(src + 4 * stride);
            float32x4_t src5 = Load<false>(src + 5 * stride);
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
        }

        SIMD_INLINE void GemmPackA_8x4(const float* src, size_t stride, float* dst)
        {
            float32x4x2_t src04 = vzipq_f32(Load<false>(src + 0 * stride), Load<false>(src + 4 * stride));
            float32x4x2_t src15 = vzipq_f32(Load<false>(src + 1 * stride), Load<false>(src + 5 * stride));
            float32x4x2_t src26 = vzipq_f32(Load<false>(src + 2 * stride), Load<false>(src + 6 * stride));
            float32x4x2_t src37 = vzipq_f32(Load<false>(src + 3 * stride), Load<false>(src + 7 * stride));
            float32x4x4_t dst0;
            dst0.val[0] = src04.val[0];
            dst0.val[1] = src15.val[0];
            dst0.val[2] = src26.val[0];
            dst0.val[3] = src37.val[0];
            Store4<false>(dst, dst0);
            float32x4x4_t dst1;
            dst1.val[0] = src04.val[1];
            dst1.val[1] = src15.val[1];
            dst1.val[2] = src26.val[1];
            dst1.val[3] = src37.val[1];
            Store4<false>(dst + 16, dst1);
        }

        SIMD_INLINE void GemmPackA_12x4(const float* src, size_t stride, float* dst)
        {
            float32x4x2_t b[6];
            b[0] = vzipq_f32(Load<false>(src + 0 * stride), Load<false>(src + 6 * stride));
            b[1] = vzipq_f32(Load<false>(src + 1 * stride), Load<false>(src + 7 * stride));
            b[2] = vzipq_f32(Load<false>(src + 2 * stride), Load<false>(src + 8 * stride));
            b[3] = vzipq_f32(Load<false>(src + 3 * stride), Load<false>(src + 9 * stride));
            b[4] = vzipq_f32(Load<false>(src + 4 * stride), Load<false>(src + 10 * stride));
            b[5] = vzipq_f32(Load<false>(src + 5 * stride), Load<false>(src + 11 * stride));

            float32x4x2_t c[3];
            c[0] = vzipq_f32(b[0].val[0], b[3].val[0]);
            c[1] = vzipq_f32(b[1].val[0], b[4].val[0]);
            c[2] = vzipq_f32(b[2].val[0], b[5].val[0]);

            float32x4x3_t d;
            d.val[0] = c[0].val[0];
            d.val[1] = c[1].val[0];
            d.val[2] = c[2].val[0];
            Store3<false>(dst + 0, d);
            d.val[0] = c[0].val[1];
            d.val[1] = c[1].val[1];
            d.val[2] = c[2].val[1];
            Store3<false>(dst + 12, d);

            c[0] = vzipq_f32(b[0].val[1], b[3].val[1]);
            c[1] = vzipq_f32(b[1].val[1], b[4].val[1]);
            c[2] = vzipq_f32(b[2].val[1], b[5].val[1]);

            d.val[0] = c[0].val[0];
            d.val[1] = c[1].val[0];
            d.val[2] = c[2].val[0];
            Store3<false>(dst + 24, d);
            d.val[0] = c[0].val[1];
            d.val[1] = c[1].val[1];
            d.val[2] = c[2].val[1];
            Store3<false>(dst + 36, d);
        }

        void GemmPackA(const float * src, size_t stride, size_t M, size_t K, size_t cell, float * dst)
        {
            size_t K4 = AlignLo(K, 4);
            for (size_t i = 0; i < M; i += cell)
            {
                size_t m = Simd::Min(cell, M - i), k = 0;
                if (cell == 4 && m == 4)
                {
                    for (; k < K4; k += 4, dst += 16)
                        GemmPackA_4x4(src + k, stride, dst);
                }
                else if (cell == 6 && m == 6)
                {
                    for (; k < K4; k += 4, dst += 24)
                        GemmPackA_6x4(src + k, stride, dst);
                }
                else if (cell == 8 && m == 8)
                {
                    for (; k < K4; k += 4, dst += 32)
                        GemmPackA_8x4(src + k, stride, dst);
                }
                else if (cell == 12 && m == 12)
                {
                    for (; k < K4; k += 4, dst += 48)
                        GemmPackA_12x4(src + k, stride, dst);
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
                        float32x4_t mask0 = LeftNotZero32f(n - 0 * F);
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
                        float32x4_t mask0 = LeftNotZero32f(n - 0 * F);
                        float32x4_t mask1 = LeftNotZero32f(n - 1 * F);
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
                        float32x4_t mask0 = LeftNotZero32f(n - 0 * F);
                        float32x4_t mask1 = LeftNotZero32f(n - 1 * F);
                        float32x4_t mask2 = LeftNotZero32f(n - 2 * F);
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
                else if (microN == 4 * F)
                {
                    if (n == microN)
                    {
                        for (; k < K; ++k)
                        {
                            const float* b = B + k * ldb;
                            Store<false>(pB + 0 * F, Load<false>(b + 0 * F));
                            Store<false>(pB + 1 * F, Load<false>(b + 1 * F));
                            Store<false>(pB + 2 * F, Load<false>(b + 2 * F));
                            Store<false>(pB + 3 * F, Load<false>(b + 3 * F));
                            pB += microN;
                        }
                    }
                    else
                    {
                        float32x4_t mask0 = LeftNotZero32f(n - 0 * F);
                        float32x4_t mask1 = LeftNotZero32f(n - 1 * F);
                        float32x4_t mask2 = LeftNotZero32f(n - 2 * F);
                        float32x4_t mask3 = LeftNotZero32f(n - 3 * F);
                        for (; k < K - 1; ++k)
                        {
                            const float* b = B + k * ldb;
                            Store<false>(pB + 0 * F, And(mask0, Load<false>(b + 0 * F)));
                            Store<false>(pB + 1 * F, And(mask1, Load<false>(b + 1 * F)));
                            Store<false>(pB + 2 * F, And(mask2, Load<false>(b + 2 * F)));
                            Store<false>(pB + 3 * F, And(mask2, Load<false>(b + 3 * F)));
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
    }
#endif// SIMD_NEON_ENABLE
}
