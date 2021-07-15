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
#include "Simd/SimdMath.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdFloat16.h"

namespace Simd
{
    namespace Base
    {
        void Float32ToFloat16(const float * src, size_t size, uint16_t * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < alignedSize; i += 4)
            {
                dst[i + 0] = Float32ToFloat16(src[i + 0]);
                dst[i + 1] = Float32ToFloat16(src[i + 1]);
                dst[i + 2] = Float32ToFloat16(src[i + 2]);
                dst[i + 3] = Float32ToFloat16(src[i + 3]);
            }
            for (; i < size; ++i)
                dst[i] = Float32ToFloat16(src[i]);
        }

        void Float16ToFloat32(const uint16_t * src, size_t size, float * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < alignedSize; i += 4)
            {
                dst[i + 0] = Float16ToFloat32(src[i + 0]);
                dst[i + 1] = Float16ToFloat32(src[i + 1]);
                dst[i + 2] = Float16ToFloat32(src[i + 2]);
                dst[i + 3] = Float16ToFloat32(src[i + 3]);
            }
            for (; i < size; ++i)
                dst[i] = Float16ToFloat32(src[i]);
        }

        SIMD_INLINE float SquaredDifference16f(uint16_t a, uint16_t b)
        {
            return Simd::Square(Float16ToFloat32(a) - Float16ToFloat32(b));
        }

        void SquaredDifferenceSum16f(const uint16_t * a, const uint16_t * b, size_t size, float * sum)
        {
            size_t alignedSize = Simd::AlignLo(size, 4);
            float sums[4] = { 0, 0, 0, 0 };
            size_t i = 0;
            for (; i < alignedSize; i += 4)
            {
                sums[0] += SquaredDifference16f(a[i + 0], b[i + 0]);
                sums[1] += SquaredDifference16f(a[i + 1], b[i + 1]);
                sums[2] += SquaredDifference16f(a[i + 2], b[i + 2]);
                sums[3] += SquaredDifference16f(a[i + 3], b[i + 3]);
            }
            for (; i < size; ++i)
                sums[0] += SquaredDifference16f(a[i], b[i]);
            *sum = sums[0] + sums[1] + sums[2] + sums[3];
        }

        void CosineDistance16f(const uint16_t * a, const uint16_t * b, size_t size, float * distance)
        {
            float aa = 0, ab = 0, bb = 0;
            for (size_t i = 0; i < size; ++i)
            {
                float _a = Float16ToFloat32(a[i]);
                float _b = Float16ToFloat32(b[i]);
                aa += _a * _a;
                ab += _a * _b;
                bb += _b * _b;
            }
            *distance = 1.0f - ab / ::sqrt(aa*bb);
        }

        void CosineDistancesMxNa16f(size_t M, size_t N, size_t K, const uint16_t * const * A, const uint16_t * const * B, float * distances)
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    CosineDistance16f(A[i], B[j], K, distances + i * N + j);
        }

        void CosineDistancesMxNp16f(size_t M, size_t N, size_t K, const uint16_t* A, const uint16_t* B, float* distances)
        {
            for (size_t i = 0; i < M; ++i)
                for (size_t j = 0; j < N; ++j)
                    CosineDistance16f(A + i * K, B + j * K, K, distances + i * N + j);
        }

        void VectorNorm16f(const uint16_t* data, size_t size, float* norm)
        {
            float sum = 0;
            for (size_t i = 0; i < size; ++i)
            {
                float val = Float16ToFloat32(data[i]);
                sum += val * val;
            }
            *norm = ::sqrt(sum);
        }

        void VectorNormNa16f(size_t N, size_t K, const uint16_t* const* A, float* norms)
        {
            for (size_t j = 0; j < N; ++j)
                VectorNorm16f(A[j], K, norms + j);
        }

        void VectorNormNp16f(size_t N, size_t K, const uint16_t* A, float* norms)
        {
            for (size_t j = 0; j < N; ++j)
                VectorNorm16f(A + j * K, K, norms + j);
        }
    }
}
