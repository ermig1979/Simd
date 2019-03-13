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
#include "Simd/SimdMath.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
    namespace Base
    {
        union Bits
        {
            float f;
            int32_t si;
            uint32_t ui;
        };

        const int SHIFT = 13;
        const int SHIFT_SIGN = 16;

        const int32_t INF_N = 0x7F800000; // flt32 infinity
        const int32_t MAX_N = 0x477FE000; // max flt16 normal as a flt32
        const int32_t MIN_N = 0x38800000; // min flt16 normal as a flt32
        const int32_t SIGN_N = 0x80000000; // flt32 sign bit

        const int32_t INF_C = INF_N >> SHIFT;
        const int32_t NAN_N = (INF_C + 1) << SHIFT; // minimum flt16 nan as a flt32
        const int32_t MAX_C = MAX_N >> SHIFT;
        const int32_t MIN_C = MIN_N >> SHIFT;
        const int32_t SIGN_C = SIGN_N >> SHIFT_SIGN; // flt16 sign bit

        const int32_t MUL_N = 0x52000000; // (1 << 23) / MIN_N
        const int32_t MUL_C = 0x33800000; // MIN_N / (1 << (23 - shift))

        const int32_t SUB_C = 0x003FF; // max flt32 subnormal down shifted
        const int32_t NOR_C = 0x00400; // min flt32 normal down shifted

        const int32_t MAX_D = INF_C - MAX_C - 1;
        const int32_t MIN_D = MIN_C - SUB_C - 1;

        SIMD_INLINE uint16_t Float32ToFloat16(float value)
        {
            Bits v, s;
            v.f = value;
            uint32_t sign = v.si & SIGN_N;
            v.si ^= sign;
            sign >>= SHIFT_SIGN; // logical shift
            s.si = MUL_N;
            s.si = int32_t(s.f * v.f); // correct subnormals
            v.si ^= (s.si ^ v.si) & -(MIN_N > v.si);
            v.si ^= (INF_N ^ v.si) & -((INF_N > v.si) & (v.si > MAX_N));
            v.si ^= (NAN_N ^ v.si) & -((NAN_N > v.si) & (v.si > INF_N));
            v.ui >>= SHIFT; // logical shift
            v.si ^= ((v.si - MAX_D) ^ v.si) & -(v.si > MAX_C);
            v.si ^= ((v.si - MIN_D) ^ v.si) & -(v.si > SUB_C);
            return v.ui | sign;
        }

        SIMD_INLINE float Float16ToFloat32(uint16_t value)
        {
            Bits v;
            v.ui = value;
            int32_t sign = v.si & SIGN_C;
            v.si ^= sign;
            sign <<= SHIFT_SIGN;
            v.si ^= ((v.si + MIN_D) ^ v.si) & -(v.si > SUB_C);
            v.si ^= ((v.si + MAX_D) ^ v.si) & -(v.si > MAX_C);
            Bits s;
            s.si = MUL_C;
            s.f *= v.si;
            int32_t mask = -(NOR_C > v.si);
            v.si <<= SHIFT;
            v.si ^= (s.si ^ v.si) & mask;
            v.si |= sign;
            return v.f;
        }

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
    }
}
