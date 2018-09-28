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
#include "Simd/SimdDefs.h"

namespace Simd
{
    namespace Base
    {
        void Gemm32fNN(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            float b = beta[0];
            for (size_t i = 0; i < M; ++i)
            {
                float * pC = C + i * ldc;
                for (size_t j = 0; j < N; ++j)
                    pC[j] = b * pC[j];
                for (size_t k = 0; k < K; ++k)
                {
                    const float * pB = B + k * ldb;
                    float a = alpha[0] * A[i*lda + k];
                    for (size_t j = 0; j < N; ++j)
                        pC[j] = a * pB[j] + pC[j];
                }
            }
        }

        void Gemm32fNT(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
        {
            float b = beta[0];
            for (size_t i = 0; i < M; ++i)
            {
                float * pC = C + i * ldc;
                for (size_t j = 0; j < N; ++j)
                    pC[j] = b * pC[j];
                for (size_t j = 0; j < N; ++j)
                {
                    const float * pA = A + i * K;
                    const float * pB = B + j * K;
                    float sum = 0;
                    for (size_t k = 0; k < K; ++k)
                        sum += pA[k] * pB[k];
                    pC[j] += sum*alpha[0];
                }
            }
        }
    }
}
