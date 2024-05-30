/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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

#ifdef SIMD_LOG_ENABLE
        SIMD_INLINE void Log16b(const uint16_t* data, size_t size, const std::string& name)
        {
            std::cout << name.c_str() << " = { " << std::setprecision(3) << std::fixed;
            for (size_t i = 0; i < size; i++)
            {
                std::cout << Base::BFloat16ToFloat32(data[i]) << " ";
            }
            std::cout << "} " << std::endl << std::flush;
        }
#endif

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

        SIMD_INLINE void Float32ToBFloat16(const float * src, uint16_t * dst)
        {
            __m128i d0 = Float32ToBFloat16(_mm_loadu_ps(src + 0));
            __m128i d1 = Float32ToBFloat16(_mm_loadu_ps(src + F));
            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi32(d0, d1));
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

        SIMD_INLINE void Float32ToBFloat16(const float* src, uint16_t* dst)
        {
            __m256i d0 = Float32ToBFloat16(_mm256_loadu_ps(src + 0));
            __m256i d1 = Float32ToBFloat16(_mm256_loadu_ps(src + F));
            _mm256_storeu_si256((__m256i*)dst, _mm256_permute4x64_epi64(_mm256_packus_epi32(d0, d1), 0xD8));
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

        SIMD_INLINE void Float32ToBFloat16(const float* src, uint16_t* dst)
        {
            __m512 s0 = _mm512_loadu_ps(src + 0 * F);
            __m512 s1 = _mm512_loadu_ps(src + 1 * F);
            __m512i d0 = Float32ToBFloat16(s0);
            __m512i d1 = Float32ToBFloat16(s1);
            _mm512_storeu_si512(dst, _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi32(d0, d1)));
        }        
        
        SIMD_INLINE void Float32ToBFloat16(const float* src, uint16_t* dst, __mmask32 loadMask, __mmask32 saveMask = __mmask32(-1))
        {
            __m512 s0 = _mm512_maskz_loadu_ps(__mmask16(loadMask >> 0 * 16), src + 0 * F);
            __m512 s1 = _mm512_maskz_loadu_ps(__mmask16(loadMask >> 1 * 16), src + 1 * F);
            __m512i d0 = Float32ToBFloat16(s0);
            __m512i d1 = Float32ToBFloat16(s1);
            _mm512_mask_storeu_epi16(dst, saveMask, _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi32(d0, d1)));
        }
    }
#endif 

#ifdef SIMD_AMXBF16_ENABLE    
    namespace AmxBf16
    {
        template <bool align, bool mask> SIMD_INLINE void Float32ToBFloat16(const float* src, uint16_t* dst, __mmask16 srcMask[2], __mmask32 dstMask[1])
        {
            __m512 s0 = Avx512bw::Load<align, mask>(src + 0 * F, srcMask[0]);
            __m512 s1 = Avx512bw::Load<align, mask>(src + 1 * F, srcMask[1]);
            Avx512bw::Store<align, mask>(dst, (__m512i)_mm512_cvtne2ps_pbh(s1, s0), dstMask[0]);
        }

        SIMD_INLINE void Float32ToBFloat16(const float* src, uint16_t* dst)
        {
            __m512 s0 = _mm512_loadu_ps(src + 0 * F);
            __m512 s1 = _mm512_loadu_ps(src + 1 * F);
            _mm512_storeu_si512(dst, (__m512i)_mm512_cvtne2ps_pbh(s1, s0));
        }

        SIMD_INLINE void Float32ToBFloat16(const float* src, uint16_t* dst, __mmask32 loadMask, __mmask32 saveMask = __mmask32(-1))
        {
            __m512 s0 = _mm512_maskz_loadu_ps(__mmask16(loadMask >> 0 * 16), src + 0 * F);
            __m512 s1 = _mm512_maskz_loadu_ps(__mmask16(loadMask >> 1 * 16), src + 1 * F);
            _mm512_mask_storeu_epi16(dst, saveMask, (__m512i)_mm512_cvtne2ps_pbh(s1, s0));
        }

        SIMD_INLINE __m512 BFloat16ToFloat32(__m256i value)
        {
            static const __m512i K16_PERM = SIMD_MM512_SETR_EPI16(
                0x10, 0x00, 0x10, 0x01, 0x10, 0x02, 0x10, 0x03, 0x10, 0x04, 0x10, 0x05, 0x10, 0x06, 0x10, 0x07,
                0x10, 0x08, 0x10, 0x09, 0x10, 0x0A, 0x10, 0x0B, 0x10, 0x0C, 0x10, 0x0D, 0x10, 0x0E, 0x10, 0x0F);
            return _mm512_castsi512_ps(_mm512_permutexvar_epi16(K16_PERM, _mm512_castsi256_si512(value)));
        }
    }
#endif

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        namespace Bf16
        {
            const uint32x4_t ROUND = SIMD_VEC_SET1_EPI32(Base::Bf16::ROUND);
            const uint32x4_t MASK = SIMD_VEC_SET1_EPI32(Base::Bf16::MASK);
        }

        SIMD_INLINE uint32x4_t Float32ToBFloat16(float32x4_t value)
        {
            return vshrq_n_u32(vaddq_u32(vreinterpretq_u32_f32(value), Bf16::ROUND), Base::Bf16::SHIFT);
        }

        SIMD_INLINE float32x4_t BFloat16ToFloat32(uint32x4_t value)
        {
            return vreinterpretq_f32_u32(vshlq_n_u32(value, Base::Bf16::SHIFT));
        }
    }
#endif 
}

#endif
