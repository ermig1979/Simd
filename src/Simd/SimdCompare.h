/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar,
*               2018-2019 Radchenko Andrey.
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
#ifndef __SimdCompare_h__
#define __SimdCompare_h__

#include "Simd/SimdConst.h"

namespace Simd
{
    namespace Base
    {
        template <SimdCompareType type> SIMD_INLINE bool Compare8u(const uint8_t & src, const uint8_t & b);

        template <> SIMD_INLINE bool Compare8u<SimdCompareEqual>(const uint8_t & a, const uint8_t & b)
        {
            return a == b;
        }

        template <> SIMD_INLINE bool Compare8u<SimdCompareNotEqual>(const uint8_t & a, const uint8_t & b)
        {
            return a != b;
        }

        template <> SIMD_INLINE bool Compare8u<SimdCompareGreater>(const uint8_t & a, const uint8_t & b)
        {
            return a > b;
        }

        template <> SIMD_INLINE bool Compare8u<SimdCompareGreaterOrEqual>(const uint8_t & a, const uint8_t & b)
        {
            return a >= b;
        }

        template <> SIMD_INLINE bool Compare8u<SimdCompareLesser>(const uint8_t & a, const uint8_t & b)
        {
            return a < b;
        }

        template <> SIMD_INLINE bool Compare8u<SimdCompareLesserOrEqual>(const uint8_t & a, const uint8_t & b)
        {
            return a <= b;
        }

        template <SimdCompareType type> SIMD_INLINE bool Compare16i(const int16_t & src, const int16_t & b);

        template <> SIMD_INLINE bool Compare16i<SimdCompareEqual>(const int16_t & a, const int16_t & b)
        {
            return a == b;
        }

        template <> SIMD_INLINE bool Compare16i<SimdCompareNotEqual>(const int16_t & a, const int16_t & b)
        {
            return a != b;
        }

        template <> SIMD_INLINE bool Compare16i<SimdCompareGreater>(const int16_t & a, const int16_t & b)
        {
            return a > b;
        }

        template <> SIMD_INLINE bool Compare16i<SimdCompareGreaterOrEqual>(const int16_t & a, const int16_t & b)
        {
            return a >= b;
        }

        template <> SIMD_INLINE bool Compare16i<SimdCompareLesser>(const int16_t & a, const int16_t & b)
        {
            return a < b;
        }

        template <> SIMD_INLINE bool Compare16i<SimdCompareLesserOrEqual>(const int16_t & a, const int16_t & b)
        {
            return a <= b;
        }
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        SIMD_INLINE __m128i NotEqual8u(__m128i a, __m128i b)
        {
            return _mm_andnot_si128(_mm_cmpeq_epi8(a, b), K_INV_ZERO);
        }

        SIMD_INLINE __m128i Greater8u(__m128i a, __m128i b)
        {
            return _mm_andnot_si128(_mm_cmpeq_epi8(_mm_min_epu8(a, b), a), K_INV_ZERO);
        }

        SIMD_INLINE __m128i GreaterOrEqual8u(__m128i a, __m128i b)
        {
            return _mm_cmpeq_epi8(_mm_max_epu8(a, b), a);
        }

        SIMD_INLINE __m128i Lesser8u(__m128i a, __m128i b)
        {
            return _mm_andnot_si128(_mm_cmpeq_epi8(_mm_max_epu8(a, b), a), K_INV_ZERO);
        }

        SIMD_INLINE __m128i LesserOrEqual8u(__m128i a, __m128i b)
        {
            return _mm_cmpeq_epi8(_mm_min_epu8(a, b), a);
        }

        template<SimdCompareType compareType> SIMD_INLINE __m128i Compare8u(__m128i a, __m128i b);

        template<> SIMD_INLINE __m128i Compare8u<SimdCompareEqual>(__m128i a, __m128i b)
        {
            return _mm_cmpeq_epi8(a, b);
        }

        template<> SIMD_INLINE __m128i Compare8u<SimdCompareNotEqual>(__m128i a, __m128i b)
        {
            return NotEqual8u(a, b);
        }

        template<> SIMD_INLINE __m128i Compare8u<SimdCompareGreater>(__m128i a, __m128i b)
        {
            return Greater8u(a, b);
        }

        template<> SIMD_INLINE __m128i Compare8u<SimdCompareGreaterOrEqual>(__m128i a, __m128i b)
        {
            return GreaterOrEqual8u(a, b);
        }

        template<> SIMD_INLINE __m128i Compare8u<SimdCompareLesser>(__m128i a, __m128i b)
        {
            return Lesser8u(a, b);
        }

        template<> SIMD_INLINE __m128i Compare8u<SimdCompareLesserOrEqual>(__m128i a, __m128i b)
        {
            return LesserOrEqual8u(a, b);
        }

        SIMD_INLINE __m128i NotEqual16i(__m128i a, __m128i b)
        {
            return _mm_andnot_si128(_mm_cmpeq_epi16(a, b), K_INV_ZERO);
        }

        SIMD_INLINE __m128i GreaterOrEqual16i_m128(__m128i a, __m128i b)
        {
            return _mm_andnot_si128(_mm_cmplt_epi16(a, b), K_INV_ZERO);
        }

        SIMD_INLINE __m128i LesserOrEqual16i(__m128i a, __m128i b)
        {
            return _mm_andnot_si128(_mm_cmpgt_epi16(a, b), K_INV_ZERO);
        }

        template<SimdCompareType compareType> SIMD_INLINE __m128i Compare16i(__m128i a, __m128i b);

        template<> SIMD_INLINE __m128i Compare16i<SimdCompareEqual>(__m128i a, __m128i b)
        {
            return _mm_cmpeq_epi16(a, b);
        }

        template<> SIMD_INLINE __m128i Compare16i<SimdCompareNotEqual>(__m128i a, __m128i b)
        {
            return NotEqual16i(a, b);
        }

        template<> SIMD_INLINE __m128i Compare16i<SimdCompareGreater>(__m128i a, __m128i b)
        {
            return _mm_cmpgt_epi16(a, b);
        }

        template<> SIMD_INLINE __m128i Compare16i<SimdCompareGreaterOrEqual>(__m128i a, __m128i b)
        {
            return GreaterOrEqual16i_m128(a, b);
        }

        template<> SIMD_INLINE __m128i Compare16i<SimdCompareLesser>(__m128i a, __m128i b)
        {
            return _mm_cmplt_epi16(a, b);
        }

        template<> SIMD_INLINE __m128i Compare16i<SimdCompareLesserOrEqual>(__m128i a, __m128i b)
        {
            return LesserOrEqual16i(a, b);
        }
    }
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE __m256i NotEqual8u(__m256i a, __m256i b)
        {
            return _mm256_andnot_si256(_mm256_cmpeq_epi8(a, b), K_INV_ZERO);
        }

        SIMD_INLINE __m256i Greater8u(__m256i a, __m256i b)
        {
            return _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_min_epu8(a, b), a), K_INV_ZERO);
        }

        SIMD_INLINE __m256i GreaterOrEqual8u(__m256i a, __m256i b)
        {
            return _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a);
        }

        SIMD_INLINE __m256i Lesser8u(__m256i a, __m256i b)
        {
            return _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a), K_INV_ZERO);
        }

        SIMD_INLINE __m256i LesserOrEqual8u(__m256i a, __m256i b)
        {
            return _mm256_cmpeq_epi8(_mm256_min_epu8(a, b), a);
        }

        template<SimdCompareType compareType> SIMD_INLINE __m256i Compare8u(__m256i a, __m256i b);

        template<> SIMD_INLINE __m256i Compare8u<SimdCompareEqual>(__m256i a, __m256i b)
        {
            return _mm256_cmpeq_epi8(a, b);
        }

        template<> SIMD_INLINE __m256i Compare8u<SimdCompareNotEqual>(__m256i a, __m256i b)
        {
            return NotEqual8u(a, b);
        }

        template<> SIMD_INLINE __m256i Compare8u<SimdCompareGreater>(__m256i a, __m256i b)
        {
            return Greater8u(a, b);
        }

        template<> SIMD_INLINE __m256i Compare8u<SimdCompareGreaterOrEqual>(__m256i a, __m256i b)
        {
            return GreaterOrEqual8u(a, b);
        }

        template<> SIMD_INLINE __m256i Compare8u<SimdCompareLesser>(__m256i a, __m256i b)
        {
            return Lesser8u(a, b);
        }

        template<> SIMD_INLINE __m256i Compare8u<SimdCompareLesserOrEqual>(__m256i a, __m256i b)
        {
            return LesserOrEqual8u(a, b);
        }

        SIMD_INLINE __m256i NotEqual16i(__m256i a, __m256i b)
        {
            return _mm256_andnot_si256(_mm256_cmpeq_epi16(a, b), K_INV_ZERO);
        }

        SIMD_INLINE __m256i GreaterOrEqual16i_m256(__m256i a, __m256i b)
        {
            return _mm256_andnot_si256(_mm256_cmpgt_epi16(b, a), K_INV_ZERO);
        }

        SIMD_INLINE __m256i LesserOrEqual16i(__m256i a, __m256i b)
        {
            return _mm256_andnot_si256(_mm256_cmpgt_epi16(a, b), K_INV_ZERO);
        }

        template<SimdCompareType compareType> SIMD_INLINE __m256i Compare16i(__m256i a, __m256i b);

        template<> SIMD_INLINE __m256i Compare16i<SimdCompareEqual>(__m256i a, __m256i b)
        {
            return _mm256_cmpeq_epi16(a, b);
        }

        template<> SIMD_INLINE __m256i Compare16i<SimdCompareNotEqual>(__m256i a, __m256i b)
        {
            return NotEqual16i(a, b);
        }

        template<> SIMD_INLINE __m256i Compare16i<SimdCompareGreater>(__m256i a, __m256i b)
        {
            return _mm256_cmpgt_epi16(a, b);
        }

        template<> SIMD_INLINE __m256i Compare16i<SimdCompareGreaterOrEqual>(__m256i a, __m256i b)
        {
            return GreaterOrEqual16i_m256(a, b);
        }

        template<> SIMD_INLINE __m256i Compare16i<SimdCompareLesser>(__m256i a, __m256i b)
        {
            return _mm256_cmpgt_epi16(b, a);
        }

        template<> SIMD_INLINE __m256i Compare16i<SimdCompareLesserOrEqual>(__m256i a, __m256i b)
        {
            return LesserOrEqual16i(a, b);
        }
    }
#endif// SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template<SimdCompareType compareType> SIMD_INLINE __mmask64 Compare8u(__m512i a, __m512i b);

        template<> SIMD_INLINE __mmask64 Compare8u<SimdCompareEqual>(__m512i a, __m512i b)
        {
            return _mm512_cmpeq_epu8_mask(a, b);
        }

        template<> SIMD_INLINE __mmask64 Compare8u<SimdCompareNotEqual>(__m512i a, __m512i b)
        {
            return _mm512_cmpneq_epu8_mask(a, b);
        }

        template<> SIMD_INLINE __mmask64 Compare8u<SimdCompareGreater>(__m512i a, __m512i b)
        {
            return _mm512_cmpgt_epu8_mask(a, b);
        }

        template<> SIMD_INLINE __mmask64 Compare8u<SimdCompareGreaterOrEqual>(__m512i a, __m512i b)
        {
            return _mm512_cmpge_epu8_mask(a, b);
        }

        template<> SIMD_INLINE __mmask64 Compare8u<SimdCompareLesser>(__m512i a, __m512i b)
        {
            return _mm512_cmplt_epu8_mask(a, b);
        }

        template<> SIMD_INLINE __mmask64 Compare8u<SimdCompareLesserOrEqual>(__m512i a, __m512i b)
        {
            return _mm512_cmple_epu8_mask(a, b);
        }

        template<SimdCompareType compareType> SIMD_INLINE __mmask32 Compare16i(__m512i a, __m512i b);

        template<> SIMD_INLINE __mmask32 Compare16i<SimdCompareEqual>(__m512i a, __m512i b)
        {
            return _mm512_cmpeq_epi16_mask(a, b);
        }

        template<> SIMD_INLINE __mmask32 Compare16i<SimdCompareNotEqual>(__m512i a, __m512i b)
        {
            return _mm512_cmpneq_epi16_mask(a, b);
        }

        template<> SIMD_INLINE __mmask32 Compare16i<SimdCompareGreater>(__m512i a, __m512i b)
        {
            return _mm512_cmpgt_epi16_mask(a, b);
        }

        template<> SIMD_INLINE __mmask32 Compare16i<SimdCompareGreaterOrEqual>(__m512i a, __m512i b)
        {
            return _mm512_cmpge_epi16_mask(a, b);
        }

        template<> SIMD_INLINE __mmask32 Compare16i<SimdCompareLesser>(__m512i a, __m512i b)
        {
            return _mm512_cmplt_epi16_mask(a, b);
        }

        template<> SIMD_INLINE __mmask32 Compare16i<SimdCompareLesserOrEqual>(__m512i a, __m512i b)
        {
            return _mm512_cmple_epi16_mask(a, b);
        }
    }
#endif// SIMD_AVX512BW_ENABLE

#ifdef SIMD_VMX_ENABLE    
    namespace Vmx
    {
        SIMD_INLINE v128_u8 GreaterOrEqual(v128_u8 a, v128_u8 b)
        {
            return (v128_u8)vec_cmpeq(vec_max(a, b), a);
        }

        template<SimdCompareType compareType> SIMD_INLINE v128_u8 Compare8u(v128_u8 a, v128_u8 b);

        template<> SIMD_INLINE v128_u8 Compare8u<SimdCompareEqual>(v128_u8 a, v128_u8 b)
        {
            return (v128_u8)vec_cmpeq(a, b);
        }

        template<> SIMD_INLINE v128_u8 Compare8u<SimdCompareNotEqual>(v128_u8 a, v128_u8 b)
        {
            return vec_xor((v128_u8)vec_cmpeq(a, b), K8_FF);
        }

        template<> SIMD_INLINE v128_u8 Compare8u<SimdCompareGreater>(v128_u8 a, v128_u8 b)
        {
            return (v128_u8)vec_cmpgt(a, b);
        }

        template<> SIMD_INLINE v128_u8 Compare8u<SimdCompareGreaterOrEqual>(v128_u8 a, v128_u8 b)
        {
            return GreaterOrEqual(a, b);
        }

        template<> SIMD_INLINE v128_u8 Compare8u<SimdCompareLesser>(v128_u8 a, v128_u8 b)
        {
            return (v128_u8)vec_cmplt(a, b);
        }

        template<> SIMD_INLINE v128_u8 Compare8u<SimdCompareLesserOrEqual>(v128_u8 a, v128_u8 b)
        {
            return vec_xor((v128_u8)vec_cmpgt(a, b), K8_FF);
        }

        template<SimdCompareType compareType> SIMD_INLINE v128_s16 Compare16i(v128_s16 a, v128_s16 b);

        template<> SIMD_INLINE v128_s16 Compare16i<SimdCompareEqual>(v128_s16 a, v128_s16 b)
        {
            return (v128_s16)vec_cmpeq(a, b);
        }

        template<> SIMD_INLINE v128_s16 Compare16i<SimdCompareNotEqual>(v128_s16 a, v128_s16 b)
        {
            return (v128_s16)vec_xor((v128_u16)vec_cmpeq(a, b), K16_FFFF);
        }

        template<> SIMD_INLINE v128_s16 Compare16i<SimdCompareGreater>(v128_s16 a, v128_s16 b)
        {
            return (v128_s16)vec_cmpgt(a, b);
        }

        template<> SIMD_INLINE v128_s16 Compare16i<SimdCompareGreaterOrEqual>(v128_s16 a, v128_s16 b)
        {
            return (v128_s16)vec_cmpeq(vec_max(a, b), a);
        }

        template<> SIMD_INLINE v128_s16 Compare16i<SimdCompareLesser>(v128_s16 a, v128_s16 b)
        {
            return (v128_s16)vec_cmplt(a, b);
        }

        template<> SIMD_INLINE v128_s16 Compare16i<SimdCompareLesserOrEqual>(v128_s16 a, v128_s16 b)
        {
            return (v128_s16)vec_xor((v128_u16)vec_cmpgt(a, b), K16_FFFF);
        }
    }
#endif// SIMD_VMX_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template<SimdCompareType compareType> SIMD_INLINE uint8x16_t Compare8u(const uint8x16_t & a, const uint8x16_t & b);

        template<> SIMD_INLINE uint8x16_t Compare8u<SimdCompareEqual>(const uint8x16_t & a, const uint8x16_t & b)
        {
            return vceqq_u8(a, b);
        }

        template<> SIMD_INLINE uint8x16_t Compare8u<SimdCompareNotEqual>(const uint8x16_t & a, const uint8x16_t & b)
        {
            return vmvnq_u8(vceqq_u8(a, b));
        }

        template<> SIMD_INLINE uint8x16_t Compare8u<SimdCompareGreater>(const uint8x16_t & a, const uint8x16_t & b)
        {
            return vcgtq_u8(a, b);
        }

        template<> SIMD_INLINE uint8x16_t Compare8u<SimdCompareGreaterOrEqual>(const uint8x16_t & a, const uint8x16_t & b)
        {
            return vcgeq_u8(a, b);
        }

        template<> SIMD_INLINE uint8x16_t Compare8u<SimdCompareLesser>(const uint8x16_t & a, const uint8x16_t & b)
        {
            return vcltq_u8(a, b);
        }

        template<> SIMD_INLINE uint8x16_t Compare8u<SimdCompareLesserOrEqual>(const uint8x16_t & a, const uint8x16_t & b)
        {
            return vcleq_u8(a, b);
        }

        template<SimdCompareType compareType> SIMD_INLINE uint16x8_t Compare16i(const int16x8_t & a, const int16x8_t & b);

        template<> SIMD_INLINE uint16x8_t Compare16i<SimdCompareEqual>(const int16x8_t & a, const int16x8_t & b)
        {
            return vceqq_s16(a, b);
        }

        template<> SIMD_INLINE uint16x8_t Compare16i<SimdCompareNotEqual>(const int16x8_t & a, const int16x8_t & b)
        {
            return vmvnq_u16(vceqq_s16(a, b));
        }

        template<> SIMD_INLINE uint16x8_t Compare16i<SimdCompareGreater>(const int16x8_t & a, const int16x8_t & b)
        {
            return vcgtq_s16(a, b);
        }

        template<> SIMD_INLINE uint16x8_t Compare16i<SimdCompareGreaterOrEqual>(const int16x8_t & a, const int16x8_t & b)
        {
            return vcgeq_s16(a, b);
        }

        template<> SIMD_INLINE uint16x8_t Compare16i<SimdCompareLesser>(const int16x8_t & a, const int16x8_t & b)
        {
            return vcltq_s16(a, b);
        }

        template<> SIMD_INLINE uint16x8_t Compare16i<SimdCompareLesserOrEqual>(const int16x8_t & a, const int16x8_t & b)
        {
            return vcleq_s16(a, b);
        }
    }
#endif// SIMD_NEON_ENABLE
}
#endif//__SimdCompare_h__
