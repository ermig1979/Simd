/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#ifndef __SimdBFloat16_h__
#define __SimdBFloat16_h__

#include "Simd/SimdStore.h"

namespace Simd
{
    namespace Base
    {
        namespace Bf16
        {
            union Bits
            {
                float f32;
                uint32_t u32;

                SIMD_INLINE Bits(float val) : f32(val) { }
                SIMD_INLINE Bits(uint32_t val) : u32(val) { }
            };

            const int SHIFT = 16;
            const uint32_t ROUND = 0x00008000;
            const uint32_t MASK = 0xFFFF0000;
        }

        SIMD_INLINE float RoundToBFloat16(float value)
        {
            return Bf16::Bits((Bf16::Bits(value).u32 + Bf16::ROUND) & Bf16::MASK).f32;
        }

        SIMD_INLINE uint16_t Float32ToBFloat16(float value)
        {
            return uint16_t((Bf16::Bits(value).u32 + Bf16::ROUND) >> Bf16::SHIFT);
        }

        SIMD_INLINE float BFloat16ToFloat32(uint16_t value)
        {
            return Bf16::Bits(uint32_t(value) << Bf16::SHIFT).f32;
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        namespace Bf16
        {
            const __m128i ROUND = SIMD_MM_SET1_EPI32(Base::Bf16::ROUND);
            const __m128i MASK = SIMD_MM_SET1_EPI32(Base::Bf16::MASK);
        }

        SIMD_INLINE __m128i Float32ToBFloat16(__m128 value)
        {
            return _mm_srli_epi32(_mm_add_epi32(_mm_castps_si128(value), Bf16::ROUND), Base::Bf16::SHIFT);
        }

        SIMD_INLINE __m128 BFloat16ToFloat32(__m128i value)
        {
            return _mm_castsi128_ps(_mm_slli_epi32(value, Base::Bf16::SHIFT));
        }
    }
#endif   

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        namespace Bf16
        {
            const __m256i ROUND = SIMD_MM256_SET1_EPI32(Base::Bf16::ROUND);
            const __m256i MASK = SIMD_MM256_SET1_EPI32(Base::Bf16::MASK);
        }

        SIMD_INLINE __m256i Float32ToBFloat16(__m256 value)
        {
            return _mm256_srli_epi32(_mm256_add_epi32(_mm256_castps_si256(value), Bf16::ROUND), Base::Bf16::SHIFT);
        }

        SIMD_INLINE __m256 BFloat16ToFloat32(__m256i value)
        {
            return _mm256_castsi256_ps(_mm256_slli_epi32(value, Base::Bf16::SHIFT));
        }
    }
#endif  

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        namespace Bf16
        {
            const __m512i ROUND = SIMD_MM512_SET1_EPI32(Base::Bf16::ROUND);
            const __m512i MASK = SIMD_MM512_SET1_EPI32(Base::Bf16::MASK);
        }

        SIMD_INLINE __m512i Float32ToBFloat16(__m512 value)
        {
            return _mm512_srli_epi32(_mm512_add_epi32(_mm512_castps_si512(value), Bf16::ROUND), Base::Bf16::SHIFT);
        }

        SIMD_INLINE __m512 BFloat16ToFloat32(__m512i value)
        {
            return _mm512_castsi512_ps(_mm512_slli_epi32(value, Base::Bf16::SHIFT));
        }

        template <bool align, bool mask> SIMD_INLINE void Float32ToBFloat16(const float* src, uint16_t* dst, __mmask16 srcMask[2], __mmask32 dstMask[1])
        {
            __m512 s0 = Load<align, mask>(src + 0 * F, srcMask[0]);
            __m512 s1 = Load<align, mask>(src + 1 * F, srcMask[1]);
            __m512i d0 = Float32ToBFloat16(s0);
            __m512i d1 = Float32ToBFloat16(s1);
            Store<align, mask>(dst, _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi32(d0, d1)), dstMask[0]);
        }
    }
#endif 

#ifdef SIMD_AVX512BF16_ENABLE    
    namespace Avx512bf16
    {
        template <bool align, bool mask> SIMD_INLINE void Float32ToBFloat16(const float* src, uint16_t * dst, __mmask16 srcMask[2], __mmask32 dstMask[1])
        {
            __m512 s0 = Avx512bw::Load<align, mask>(src + 0 * F, srcMask[0]);
            __m512 s1 = Avx512bw::Load<align, mask>(src + 1 * F, srcMask[1]);
            Avx512bw::Store<align, mask>(dst, (__m512i)_mm512_cvtne2ps_pbh(s0, s1), dstMask[0]);
        }
    }
#endif 
}

#endif//__SimdBFloat16_h__
