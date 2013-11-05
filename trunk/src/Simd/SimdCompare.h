/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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

#include "Simd/SimdLib.h"
#include "Simd/SimdConst.h"

namespace Simd
{
    namespace Base
    {
        template <SimdCompareType type> SIMD_INLINE bool Compare(const uint8_t & src, const uint8_t & b);

        template <> SIMD_INLINE bool Compare<SimdCompareEqual>(const uint8_t & a, const uint8_t & b)
        {
            return a == b;
        }

        template <> SIMD_INLINE bool Compare<SimdCompareNotEqual>(const uint8_t & a, const uint8_t & b)
        {
            return a != b;
        }

        template <> SIMD_INLINE bool Compare<SimdCompareGreater>(const uint8_t & a, const uint8_t & b)
        {
            return a > b;
        }

        template <> SIMD_INLINE bool Compare<SimdCompareGreaterOrEqual>(const uint8_t & a, const uint8_t & b)
        {
            return a >= b;
        }

        template <> SIMD_INLINE bool Compare<SimdCompareLesser>(const uint8_t & a, const uint8_t & b)
        {
            return a < b;
        }

        template <> SIMD_INLINE bool Compare<SimdCompareLesserOrEqual>(const uint8_t & a, const uint8_t & b)
        {
            return a <= b;
        }
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        SIMD_INLINE __m128i NotEqualU8(__m128i a, __m128i b)
        {
            return _mm_andnot_si128(_mm_cmpeq_epi8(a, b), K_INV_ZERO);
        }

        SIMD_INLINE __m128i GreaterU8(__m128i a, __m128i b)
        {
            return _mm_andnot_si128(_mm_cmpeq_epi8(_mm_min_epu8(a, b), a), K_INV_ZERO);
        }

        SIMD_INLINE __m128i GreaterOrEqualU8(__m128i a, __m128i b)
        {
            return _mm_cmpeq_epi8(_mm_max_epu8(a, b), a);
        }

        SIMD_INLINE __m128i LesserU8(__m128i a, __m128i b)
        {
            return _mm_andnot_si128(_mm_cmpeq_epi8(_mm_max_epu8(a, b), a), K_INV_ZERO);
        }

        SIMD_INLINE __m128i LesserOrEqualU8(__m128i a, __m128i b)
        {
            return _mm_cmpeq_epi8(_mm_min_epu8(a, b), a);
        }

        template<SimdCompareType compareType> SIMD_INLINE __m128i Compare(__m128i a, __m128i b);

        template<> SIMD_INLINE __m128i Compare<SimdCompareEqual>(__m128i a, __m128i b)
        {
            return _mm_cmpeq_epi8(a, b);
        }

        template<> SIMD_INLINE __m128i Compare<SimdCompareNotEqual>(__m128i a, __m128i b)
        {
            return NotEqualU8(a, b);
        }

        template<> SIMD_INLINE __m128i Compare<SimdCompareGreater>(__m128i a, __m128i b)
        {
            return GreaterU8(a, b);
        }

        template<> SIMD_INLINE __m128i Compare<SimdCompareGreaterOrEqual>(__m128i a, __m128i b)
        {
            return GreaterOrEqualU8(a, b);
        }

        template<> SIMD_INLINE __m128i Compare<SimdCompareLesser>(__m128i a, __m128i b)
        {
            return LesserU8(a, b);
        }

        template<> SIMD_INLINE __m128i Compare<SimdCompareLesserOrEqual>(__m128i a, __m128i b)
        {
            return LesserOrEqualU8(a, b);
        }   
    }
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE __m256i NotEqualU8(__m256i a, __m256i b)
        {
            return _mm256_andnot_si256(_mm256_cmpeq_epi8(a, b), K_INV_ZERO);
        }

        SIMD_INLINE __m256i GreaterU8(__m256i a, __m256i b)
        {
            return _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_min_epu8(a, b), a), K_INV_ZERO);
        }

        SIMD_INLINE __m256i GreaterOrEqualU8(__m256i a, __m256i b)
        {
            return _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a);
        }

        SIMD_INLINE __m256i LesserU8(__m256i a, __m256i b)
        {
            return _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a), K_INV_ZERO);
        }

        SIMD_INLINE __m256i LesserOrEqualU8(__m256i a, __m256i b)
        {
            return _mm256_cmpeq_epi8(_mm256_min_epu8(a, b), a);
        }

        template<SimdCompareType compareType> SIMD_INLINE __m256i Compare(__m256i a, __m256i b);

        template<> SIMD_INLINE __m256i Compare<SimdCompareEqual>(__m256i a, __m256i b)
        {
            return _mm256_cmpeq_epi8(a, b);
        }

        template<> SIMD_INLINE __m256i Compare<SimdCompareNotEqual>(__m256i a, __m256i b)
        {
            return NotEqualU8(a, b);
        }

        template<> SIMD_INLINE __m256i Compare<SimdCompareGreater>(__m256i a, __m256i b)
        {
            return GreaterU8(a, b);
        }

        template<> SIMD_INLINE __m256i Compare<SimdCompareGreaterOrEqual>(__m256i a, __m256i b)
        {
            return GreaterOrEqualU8(a, b);
        }

        template<> SIMD_INLINE __m256i Compare<SimdCompareLesser>(__m256i a, __m256i b)
        {
            return LesserU8(a, b);
        }

        template<> SIMD_INLINE __m256i Compare<SimdCompareLesserOrEqual>(__m256i a, __m256i b)
        {
            return LesserOrEqualU8(a, b);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
#endif//__SimdCompare_h__