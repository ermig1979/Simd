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
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE void ScaleC(float* C, const svbool_t& mask, const svfloat32_t& beta)
        {
            svst1_f32(mask, C, svmul_f32_x(mask, svld1_f32(mask, C), beta));
        }

        void GemmScaleC(size_t M, size_t N, float beta, float* C, size_t ldc)
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
                size_t F = svcntw(), QF = 4 * F, NQF = AlignLoAny(N, QF), NF = AlignLoAny(N, F);
                const svbool_t body = svptrue_b32();
                const svfloat32_t _beta = svdup_n_f32(beta);
                for (size_t i = 0; i < M; ++i)
                {
                    size_t j = 0;
                    for (; j < NQF; j += QF)
                    {
                        ScaleC(C + j + 0 * F, body, _beta);
                        ScaleC(C + j + 1 * F, body, _beta);
                        ScaleC(C + j + 2 * F, body, _beta);
                        ScaleC(C + j + 3 * F, body, _beta);
                    }
                    for (; j < NF; j += F)
                        ScaleC(C + j, body, _beta);
                    if (j < N)
                        ScaleC(C + j, svwhilelt_b32(j, N), _beta);
                    C += ldc;
                }
            }
        }
    }
#endif
}
