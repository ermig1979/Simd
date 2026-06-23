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

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE void AddProduct(float* ptr, const svfloat32_t& value, const svfloat32_t& alpha, const svbool_t& mask)
        {
            svst1_f32(mask, ptr, svmla_f32_m(mask, svld1_f32(mask, ptr), value, alpha));
        }

#define SIMD_SVE2_GEMM_INIT(row) \
        svfloat32_t c##row##0, c##row##1, c##row##2, c##row##3; \
        if (M > row) c##row##0 = zero, c##row##1 = zero, c##row##2 = zero, c##row##3 = zero;

#define SIMD_SVE2_GEMM_ROW(row) \
        if (M > row) \
        { \
            svfloat32_t a = svdup_n_f32(A[(row) * lda]); \
            c##row##0 = svmla_f32_m(mask0, c##row##0, b0, a); \
            c##row##1 = svmla_f32_m(mask1, c##row##1, b1, a); \
            c##row##2 = svmla_f32_m(mask2, c##row##2, b2, a); \
            c##row##3 = svmla_f32_m(mask3, c##row##3, b3, a); \
        }

#define SIMD_SVE2_GEMM_SAVE(row) \
        if (M > row) \
        { \
            AddProduct(C + (row) * ldc + 0 * F, c##row##0, _alpha, mask0); \
            AddProduct(C + (row) * ldc + 1 * F, c##row##1, _alpha, mask1); \
            AddProduct(C + (row) * ldc + 2 * F, c##row##2, _alpha, mask2); \
            AddProduct(C + (row) * ldc + 3 * F, c##row##3, _alpha, mask3); \
        }

        template<int M> void GemmKernelMx4nn(size_t K, float alpha, const float* A, size_t lda, const float* B, size_t ldb, size_t F, float* C, size_t ldc, size_t tail)
        {
            const svbool_t mask0 = svwhilelt_b32(0 * F, tail);
            const svbool_t mask1 = svwhilelt_b32(1 * F, tail);
            const svbool_t mask2 = svwhilelt_b32(2 * F, tail);
            const svbool_t mask3 = svwhilelt_b32(3 * F, tail);
            const svfloat32_t zero = svdup_n_f32(0.0f);
            SIMD_SVE2_GEMM_INIT(0);
            SIMD_SVE2_GEMM_INIT(1);
            SIMD_SVE2_GEMM_INIT(2);
            SIMD_SVE2_GEMM_INIT(3);
            SIMD_SVE2_GEMM_INIT(4);
            SIMD_SVE2_GEMM_INIT(5);
            for (size_t k = 0; k < K; ++k)
            {
                svfloat32_t b0 = svld1_f32(mask0, B + 0 * F);
                svfloat32_t b1 = svld1_f32(mask1, B + 1 * F);
                svfloat32_t b2 = svld1_f32(mask2, B + 2 * F);
                svfloat32_t b3 = svld1_f32(mask3, B + 3 * F);
                SIMD_SVE2_GEMM_ROW(0);
                SIMD_SVE2_GEMM_ROW(1);
                SIMD_SVE2_GEMM_ROW(2);
                SIMD_SVE2_GEMM_ROW(3);
                SIMD_SVE2_GEMM_ROW(4);
                SIMD_SVE2_GEMM_ROW(5);
                A += 1;
                B += ldb;
            }
            svfloat32_t _alpha = svdup_n_f32(alpha);
            SIMD_SVE2_GEMM_SAVE(0);
            SIMD_SVE2_GEMM_SAVE(1);
            SIMD_SVE2_GEMM_SAVE(2);
            SIMD_SVE2_GEMM_SAVE(3);
            SIMD_SVE2_GEMM_SAVE(4);
            SIMD_SVE2_GEMM_SAVE(5);
        }

        void Gemm32fNN(size_t M, size_t N, size_t K, const float* alpha, const float* A, size_t lda, const float* B, size_t ldb, const float* beta, float* C, size_t ldc)
        {
            const size_t F = svcntw();
            const size_t microM = 6;
            const size_t microN = 4 * F;
            GemmScaleC(M, N, beta[0], C, ldc);
            for (size_t i = 0; i < M; i += microM)
            {
                size_t m = Simd::Min(microM, M - i);
                for (size_t j = 0; j < N; j += microN)
                {
                    size_t tail = Simd::Min(microN, N - j);
                    switch (m)
                    {
                    case 1: GemmKernelMx4nn<1>(K, alpha[0], A + i * lda, lda, B + j, ldb, F, C + i * ldc + j, ldc, tail); break;
                    case 2: GemmKernelMx4nn<2>(K, alpha[0], A + i * lda, lda, B + j, ldb, F, C + i * ldc + j, ldc, tail); break;
                    case 3: GemmKernelMx4nn<3>(K, alpha[0], A + i * lda, lda, B + j, ldb, F, C + i * ldc + j, ldc, tail); break;
                    case 4: GemmKernelMx4nn<4>(K, alpha[0], A + i * lda, lda, B + j, ldb, F, C + i * ldc + j, ldc, tail); break;
                    case 5: GemmKernelMx4nn<5>(K, alpha[0], A + i * lda, lda, B + j, ldb, F, C + i * ldc + j, ldc, tail); break;
                    case 6: GemmKernelMx4nn<6>(K, alpha[0], A + i * lda, lda, B + j, ldb, F, C + i * ldc + j, ldc, tail); break;
                    }
                }
            }
        }

#undef SIMD_SVE2_GEMM_INIT
#undef SIMD_SVE2_GEMM_ROW
#undef SIMD_SVE2_GEMM_SAVE
    }
#endif
}
