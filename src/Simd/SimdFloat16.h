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
#ifndef __SimdFloat16_h__
#define __SimdFloat16_h__

#include "Simd/SimdInit.h"

namespace Simd
{
    namespace Base
    {
        namespace Fp16
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
        }

        SIMD_INLINE uint16_t Float32ToFloat16(float value)
        {
            Fp16::Bits v, s;
            v.f = value;
            uint32_t sign = v.si & Fp16::SIGN_N;
            v.si ^= sign;
            sign >>= Fp16::SHIFT_SIGN; // logical shift
            s.si = Fp16::MUL_N;
            s.si = int32_t(s.f * v.f); // correct subnormals
            v.si ^= (s.si ^ v.si) & -(Fp16::MIN_N > v.si);
            v.si ^= (Fp16::INF_N ^ v.si) & -((Fp16::INF_N > v.si) & (v.si > Fp16::MAX_N));
            v.si ^= (Fp16::NAN_N ^ v.si) & -((Fp16::NAN_N > v.si) & (v.si > Fp16::INF_N));
            v.ui >>= Fp16::SHIFT; // logical shift
            v.si ^= ((v.si - Fp16::MAX_D) ^ v.si) & -(v.si > Fp16::MAX_C);
            v.si ^= ((v.si - Fp16::MIN_D) ^ v.si) & -(v.si > Fp16::SUB_C);
            return v.ui | sign;
        }

        SIMD_INLINE float Float16ToFloat32(uint16_t value)
        {
            Fp16::Bits v;
            v.ui = value;
            int32_t sign = v.si & Fp16::SIGN_C;
            v.si ^= sign;
            sign <<= Fp16::SHIFT_SIGN;
            v.si ^= ((v.si + Fp16::MIN_D) ^ v.si) & -(v.si > Fp16::SUB_C);
            v.si ^= ((v.si + Fp16::MAX_D) ^ v.si) & -(v.si > Fp16::MAX_C);
            Fp16::Bits s;
            s.si = Fp16::MUL_C;
            s.f *= v.si;
            int32_t mask = -(Fp16::NOR_C > v.si);
            v.si <<= Fp16::SHIFT;
            v.si ^= (s.si ^ v.si) & mask;
            v.si |= sign;
            return v.f;
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        namespace Fp16
        {
            union Bits
            {
                __m128 f;
                __m128i i;
            };

            const __m128i INF_N = SIMD_MM_SET1_EPI32(Base::Fp16::INF_N);
            const __m128i MAX_N = SIMD_MM_SET1_EPI32(Base::Fp16::MAX_N);
            const __m128i MIN_N = SIMD_MM_SET1_EPI32(Base::Fp16::MIN_N);
            const __m128i SIGN_N = SIMD_MM_SET1_EPI32(Base::Fp16::SIGN_N);

            const __m128i INF_C = SIMD_MM_SET1_EPI32(Base::Fp16::INF_C);
            const __m128i NAN_N = SIMD_MM_SET1_EPI32(Base::Fp16::NAN_N);
            const __m128i MAX_C = SIMD_MM_SET1_EPI32(Base::Fp16::MAX_C);
            const __m128i MIN_C = SIMD_MM_SET1_EPI32(Base::Fp16::MIN_C);
            const __m128i SIGN_C = SIMD_MM_SET1_EPI32(Base::Fp16::SIGN_C);

            const __m128i MUL_N = SIMD_MM_SET1_EPI32(Base::Fp16::MUL_N);
            const __m128i MUL_C = SIMD_MM_SET1_EPI32(Base::Fp16::MUL_C);

            const __m128i SUB_C = SIMD_MM_SET1_EPI32(Base::Fp16::SUB_C);
            const __m128i NOR_C = SIMD_MM_SET1_EPI32(Base::Fp16::NOR_C);

            const __m128i MAX_D = SIMD_MM_SET1_EPI32(Base::Fp16::MAX_D);
            const __m128i MIN_D = SIMD_MM_SET1_EPI32(Base::Fp16::MIN_D);
        }

        SIMD_INLINE __m128i Float32ToFloat16(__m128 value)
        {
            Fp16::Bits v, s;
            v.f = value;
            __m128i sign = _mm_and_si128(v.i, Fp16::SIGN_N);
            v.i = _mm_xor_si128(v.i, sign);
            sign = _mm_srli_epi32(sign, Base::Fp16::SHIFT_SIGN);
            s.i = Fp16::MUL_N;
            s.i = _mm_cvtps_epi32(_mm_floor_ps(_mm_mul_ps(s.f, v.f))); 
            v.i = _mm_xor_si128(v.i, _mm_and_si128(_mm_xor_si128(s.i, v.i), _mm_cmpgt_epi32(Fp16::MIN_N, v.i)));
            v.i = _mm_xor_si128(v.i, _mm_and_si128(_mm_xor_si128(Fp16::INF_N, v.i), _mm_and_si128(_mm_cmpgt_epi32(Fp16::INF_N, v.i), _mm_cmpgt_epi32(v.i, Fp16::MAX_N))));
            v.i = _mm_xor_si128(v.i, _mm_and_si128(_mm_xor_si128(Fp16::NAN_N, v.i), _mm_and_si128(_mm_cmpgt_epi32(Fp16::NAN_N, v.i), _mm_cmpgt_epi32(v.i, Fp16::INF_N))));
            v.i = _mm_srli_epi32(v.i, Base::Fp16::SHIFT); 
            v.i = _mm_xor_si128(v.i, _mm_and_si128(_mm_xor_si128(_mm_sub_epi32(v.i, Fp16::MAX_D), v.i), _mm_cmpgt_epi32(v.i, Fp16::MAX_C)));
            v.i = _mm_xor_si128(v.i, _mm_and_si128(_mm_xor_si128(_mm_sub_epi32(v.i, Fp16::MIN_D), v.i), _mm_cmpgt_epi32(v.i, Fp16::SUB_C)));
            return _mm_or_si128(v.i, sign);
        }

        SIMD_INLINE __m128 Float16ToFloat32(__m128i value)
        {
            Fp16::Bits v;
            v.i = value;
            __m128i sign = _mm_and_si128(v.i, Fp16::SIGN_C);
            v.i = _mm_xor_si128(v.i, sign);
            sign = _mm_slli_epi32(sign, Base::Fp16::SHIFT_SIGN);
            v.i = _mm_xor_si128(v.i, _mm_and_si128(_mm_xor_si128(_mm_add_epi32(v.i, Fp16::MIN_D), v.i), _mm_cmpgt_epi32(v.i, Fp16::SUB_C)));
            v.i = _mm_xor_si128(v.i, _mm_and_si128(_mm_xor_si128(_mm_add_epi32(v.i, Fp16::MAX_D), v.i), _mm_cmpgt_epi32(v.i, Fp16::MAX_C)));
            Fp16::Bits s;
            s.i = Fp16::MUL_C;
            s.f = _mm_mul_ps(s.f, _mm_cvtepi32_ps(v.i));
            __m128i mask = _mm_cmpgt_epi32(Fp16::NOR_C, v.i);
            v.i = _mm_slli_epi32(v.i, Base::Fp16::SHIFT);
            v.i = _mm_xor_si128(v.i, _mm_and_si128(_mm_xor_si128(s.i, v.i), mask));
            v.i = _mm_or_si128(v.i, sign);
            return v.f;
        }
    }
#endif   
}

#endif//__SimdFloat16_h__
