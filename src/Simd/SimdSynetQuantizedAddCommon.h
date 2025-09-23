/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#ifndef __SimdSynetQuantizedAddCommon_h__
#define __SimdSynetQuantizedAddCommon_h__

#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdFmadd.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void QuantizedAdd(int a, float adScale, int b, float bdScale, float term, uint8_t& dst)
        {
            float val = Fmadd<false>(float(a), adScale, Fmadd<false>(float(b), bdScale, term));
            dst = (uint8_t)RestrictRange(NearByInt(val), 0, 255);
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE __m128i QuantizedAdd(const __m128i& a, const __m128& adScale, const __m128i& b, const __m128& bdScale, const __m128& term)
        {
            return _mm_cvtps_epi32(Fmadd<false>(_mm_cvtepi32_ps(a), adScale, Fmadd<false>(_mm_cvtepi32_ps(b), bdScale, term)));
        }

        SIMD_INLINE void QuantizedAdd8u8u8u1(const uint8_t* a, const __m128& adScale, const uint8_t* b, const __m128& bdScale, const __m128& term, uint8_t* dst)
        {
            __m128i d0 = QuantizedAdd(_mm_set1_epi32(a[0]), adScale, _mm_set1_epi32(b[0]), bdScale, term);
            dst[0] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void QuantizedAdd8u8u8u4(const uint8_t* a, const __m128& adScale, const uint8_t* b, const __m128& bdScale, const __m128& term, uint8_t* dst)
        {
            __m128i a0 = _mm_cvtepu8_epi32(_mm_set1_epi32(((int32_t*)a)[0]));
            __m128i b0 = _mm_cvtepu8_epi32(_mm_set1_epi32(((int32_t*)b)[0]));
            __m128i d0 = QuantizedAdd(a0, adScale, b0, bdScale, term);
            ((uint32_t*)dst)[0] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void QuantizedAdd8u8u8u16(const uint8_t* a, const __m128& adScale, const uint8_t* b, const __m128& bdScale, const __m128& term, uint8_t* dst)
        {
            __m128i _a = _mm_loadu_si128((__m128i*)a);
            __m128i _b = _mm_loadu_si128((__m128i*)b);
            __m128i d0 = QuantizedAdd(_mm_cvtepu8_epi32(_mm_srli_si128(_a, 0 * 4)), adScale, _mm_cvtepu8_epi32(_mm_srli_si128(_b, 0 * 4)), bdScale, term);
            __m128i d1 = QuantizedAdd(_mm_cvtepu8_epi32(_mm_srli_si128(_a, 1 * 4)), adScale, _mm_cvtepu8_epi32(_mm_srli_si128(_b, 1 * 4)), bdScale, term);
            __m128i d2 = QuantizedAdd(_mm_cvtepu8_epi32(_mm_srli_si128(_a, 2 * 4)), adScale, _mm_cvtepu8_epi32(_mm_srli_si128(_b, 2 * 4)), bdScale, term);
            __m128i d3 = QuantizedAdd(_mm_cvtepu8_epi32(_mm_srli_si128(_a, 3 * 4)), adScale, _mm_cvtepu8_epi32(_mm_srli_si128(_b, 3 * 4)), bdScale, term);
            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE __m256i QuantizedAdd(const __m256i& a, const __m256& adScale, const __m256i& b, const __m256& bdScale, const __m256& term)
        {
            return _mm256_cvtps_epi32(Fmadd<false>(_mm256_cvtepi32_ps(a), adScale, Fmadd<false>(_mm256_cvtepi32_ps(b), bdScale, term)));
        }

        SIMD_INLINE void QuantizedAdd8u8u8u1(const uint8_t* a, const __m256& adScale, const uint8_t* b, const __m256& bdScale, const __m256& term, uint8_t* dst)
        {
            __m256i d0 = QuantizedAdd(_mm256_set1_epi32(a[0]), adScale, _mm256_set1_epi32(b[0]), bdScale, term);
            dst[0] = _mm_cvtsi128_si32(_mm256_castsi256_si128(_mm256_packus_epi16(_mm256_packs_epi32(d0, K_ZERO), K_ZERO)));
        }

        SIMD_INLINE void QuantizedAdd8u8u8u4(const uint8_t* a, const __m256& adScale, const uint8_t* b, const __m256& bdScale, const __m256& term, uint8_t* dst)
        {
            __m256i a0 = _mm256_cvtepu8_epi32(_mm_set1_epi32(((int32_t*)a)[0]));
            __m256i b0 = _mm256_cvtepu8_epi32(_mm_set1_epi32(((int32_t*)b)[0]));
            __m256i d0 = QuantizedAdd(a0, adScale, b0, bdScale, term);
            ((uint32_t*)dst)[0] = _mm_cvtsi128_si32(_mm256_castsi256_si128(_mm256_packus_epi16(_mm256_packs_epi32(d0, K_ZERO), K_ZERO)));
        }

        SIMD_INLINE void QuantizedAdd8u8u8u16(const uint8_t* a, const __m256& adScale, const uint8_t* b, const __m256& bdScale, const __m256& term, uint8_t* dst)
        {
            __m128i _a = _mm_loadu_si128((__m128i*)a);
            __m128i _b = _mm_loadu_si128((__m128i*)b);
            __m256i d0 = QuantizedAdd(_mm256_cvtepu8_epi32(_mm_srli_si128(_a, 0 * 8)), adScale, _mm256_cvtepu8_epi32(_mm_srli_si128(_b, 0 * 8)), bdScale, term);
            __m256i d1 = QuantizedAdd(_mm256_cvtepu8_epi32(_mm_srli_si128(_a, 1 * 8)), adScale, _mm256_cvtepu8_epi32(_mm_srli_si128(_b, 1 * 8)), bdScale, term);
            _mm_storeu_si128((__m128i*)dst, _mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO)));
        }

        SIMD_INLINE void QuantizedAdd8u8u8u32(const uint8_t* a, const __m256& adScale, const uint8_t* b, const __m256& bdScale, const __m256& term, uint8_t* dst)
        {
            __m128i a0 = _mm_loadu_si128((__m128i*)a + 0), b0 = _mm_loadu_si128((__m128i*)b + 0);
            __m256i d0 = QuantizedAdd(_mm256_cvtepu8_epi32(_mm_srli_si128(a0, 0 * 8)), adScale, _mm256_cvtepu8_epi32(_mm_srli_si128(b0, 0 * 8)), bdScale, term);
            __m256i d1 = QuantizedAdd(_mm256_cvtepu8_epi32(_mm_srli_si128(a0, 1 * 8)), adScale, _mm256_cvtepu8_epi32(_mm_srli_si128(b0, 1 * 8)), bdScale, term);
            __m128i a1 = _mm_loadu_si128((__m128i*)a + 1), b1 = _mm_loadu_si128((__m128i*)b + 1);
            __m256i d2 = QuantizedAdd(_mm256_cvtepu8_epi32(_mm_srli_si128(a1, 0 * 8)), adScale, _mm256_cvtepu8_epi32(_mm_srli_si128(b1, 0 * 8)), bdScale, term);
            __m256i d3 = QuantizedAdd(_mm256_cvtepu8_epi32(_mm_srli_si128(a1, 1 * 8)), adScale, _mm256_cvtepu8_epi32(_mm_srli_si128(b1, 1 * 8)), bdScale, term);
            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE __m512i QuantizedAdd(const __m512i& a, const __m512& adScale, const __m512i& b, const __m512& bdScale, const __m512& term)
        {
            return _mm512_cvtps_epi32(Fmadd<false>(_mm512_cvtepi32_ps(a), adScale, Fmadd<false>(_mm512_cvtepi32_ps(b), bdScale, term)));
        }

        SIMD_INLINE void QuantizedAdd8u8u8u16(const uint8_t* a, const __m512& adScale, const uint8_t* b, const __m512& bdScale, const __m512& term, uint8_t* dst, __mmask16 tail = -1)
        {
            __m512i d0 = QuantizedAdd(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, a)), adScale, _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, b)), bdScale, term);
            _mm_mask_storeu_epi8(dst, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0), K_ZERO)));
        }

        SIMD_INLINE void QuantizedAdd8u8u8u64(const uint8_t* a, const __m512& adScale, const uint8_t* b, const __m512& bdScale, const __m512& term, uint8_t* dst)
        {
            __m512i d0 = QuantizedAdd(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 0)), adScale, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 0)), bdScale, term);
            __m512i d1 = QuantizedAdd(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 1)), adScale, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 1)), bdScale, term);
            __m512i d2 = QuantizedAdd(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 2)), adScale, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 2)), bdScale, term);
            __m512i d3 = QuantizedAdd(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 3)), adScale, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 3)), bdScale, term);
            _mm512_storeu_si512((__m512i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }
    }
#endif
}

#endif
