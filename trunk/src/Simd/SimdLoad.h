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
#ifndef __SimdLoad_h__
#define __SimdLoad_h__

#include "Simd/SimdConst.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE
	namespace Sse2
	{
		template <bool align> SIMD_INLINE __m128i Load(const __m128i * p);

		template <> SIMD_INLINE __m128i Load<false>(const __m128i * p)
		{
			return _mm_loadu_si128(p); 
		}

		template <> SIMD_INLINE __m128i Load<true>(const __m128i * p)
		{
			return _mm_load_si128(p); 
		}

		template <bool align> SIMD_INLINE __m128i LoadMaskI8(const __m128i * p, __m128i index)
		{
			return _mm_cmpeq_epi8(Load<align>(p), index);
		}

		template <size_t count> SIMD_INLINE __m128i LoadBeforeFirst(__m128i first)
		{
			return _mm_or_si128(_mm_slli_si128(first, count), _mm_and_si128(first, _mm_srli_si128(K_INV_ZERO, A - count)));
		}

		template <size_t count> SIMD_INLINE __m128i LoadAfterLast(__m128i last)
		{
			return _mm_or_si128(_mm_srli_si128(last, count), _mm_and_si128(last, _mm_slli_si128(K_INV_ZERO, A - count)));
		}

		template <bool align, size_t step> SIMD_INLINE void LoadNose3(const uint8_t * p, __m128i a[3])
		{
			a[1] = Load<align>((__m128i*)p);
			a[0] = LoadBeforeFirst<step>(a[1]);
			a[2] = _mm_loadu_si128((__m128i*)(p + step));
		}

		template <bool align, size_t step> SIMD_INLINE void LoadBody3(const uint8_t * p, __m128i a[3])
		{
			a[0] = _mm_loadu_si128((__m128i*)(p - step));
			a[1] = Load<align>((__m128i*)p);
			a[2] = _mm_loadu_si128((__m128i*)(p + step));
		}

		template <bool align, size_t step> SIMD_INLINE void LoadTail3(const uint8_t * p, __m128i a[3])
		{
			a[0] = _mm_loadu_si128((__m128i*)(p - step));
			a[1] = Load<align>((__m128i*)p);
			a[2] = LoadAfterLast<step>(a[1]);
		}

		template <bool align, size_t step> SIMD_INLINE void LoadNose5(const uint8_t * p, __m128i a[5])
		{
			a[2] = Load<align>((__m128i*)p);
			a[1] = LoadBeforeFirst<step>(a[2]);
			a[0] = LoadBeforeFirst<step>(a[1]);
			a[3] = _mm_loadu_si128((__m128i*)(p + step));
			a[4] = _mm_loadu_si128((__m128i*)(p + 2*step));
		}

		template <bool align, size_t step> SIMD_INLINE void LoadBody5(const uint8_t * p, __m128i a[5])
		{
			a[0] = _mm_loadu_si128((__m128i*)(p - 2*step));
			a[1] = _mm_loadu_si128((__m128i*)(p - step));
			a[2] = Load<align>((__m128i*)p);
			a[3] = _mm_loadu_si128((__m128i*)(p + step));
			a[4] = _mm_loadu_si128((__m128i*)(p + 2*step));
		}

		template <bool align, size_t step> SIMD_INLINE void LoadTail5(const uint8_t * p, __m128i a[5])
		{
			a[0] = _mm_loadu_si128((__m128i*)(p - 2*step));
			a[1] = _mm_loadu_si128((__m128i*)(p - step));
			a[2] = Load<align>((__m128i*)p);
			a[3] = LoadAfterLast<step>(a[2]);
			a[4] = LoadAfterLast<step>(a[3]);
		}
	}
#endif//SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE
	namespace Avx2
	{
		template <bool align> SIMD_INLINE __m256i Load(const __m256i * p);

		template <> SIMD_INLINE __m256i Load<false>(const __m256i * p)
		{
			return _mm256_loadu_si256(p); 
		}

		template <> SIMD_INLINE __m256i Load<true>(const __m256i * p)
		{
			return _mm256_load_si256(p); 
		}

        template <bool align> SIMD_INLINE __m128i LoadHalf(const __m128i * p);

        template <> SIMD_INLINE __m128i LoadHalf<false>(const __m128i * p)
        {
            return _mm_loadu_si128(p); 
        }

        template <> SIMD_INLINE __m128i LoadHalf<true>(const __m128i * p)
        {
            return _mm_load_si128(p); 
        }

        template <size_t count> SIMD_INLINE __m128i LoadHalfBeforeFirst(__m128i first)
        {
            return _mm_or_si128(_mm_slli_si128(first, count), _mm_and_si128(first, _mm_srli_si128(Sse2::K_INV_ZERO, HA - count)));
        }

        template <size_t count> SIMD_INLINE __m128i LoadHalfAfterLast(__m128i last)
        {
            return _mm_or_si128(_mm_srli_si128(last, count), _mm_and_si128(last, _mm_slli_si128(Sse2::K_INV_ZERO, HA - count)));
        }

        template <bool align> SIMD_INLINE __m256i LoadPermuted(const __m256i * p)
        {
            return _mm256_permute4x64_epi64(Load<align>(p), 0xD8); 
        }

        template <bool align> SIMD_INLINE __m256i LoadMaskI8(const __m256i * p, __m256i index)
        {
            return _mm256_cmpeq_epi8(Load<align>(p), index);
        }

        template <bool align, size_t step> SIMD_INLINE __m256i LoadBeforeFirst(const uint8_t * p)
        {
            __m128i lo = LoadHalfBeforeFirst<step>(LoadHalf<align>((__m128i*)p));
            __m128i hi = _mm_loadu_si128((__m128i*)(p + HA - step));
            return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 0x1);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBeforeFirst(const uint8_t * p, __m256i & first, __m256i & second)
        {
            __m128i firstLo = LoadHalfBeforeFirst<step>(LoadHalf<align>((__m128i*)p));
            __m128i firstHi= _mm_loadu_si128((__m128i*)(p + HA - step));
            first = _mm256_inserti128_si256(_mm256_castsi128_si256(firstLo), firstHi, 0x1);

            __m128i secondLo = LoadHalfBeforeFirst<step>(firstLo);
            __m128i secondHi= _mm_loadu_si128((__m128i*)(p + HA - 2*step));
            second = _mm256_inserti128_si256(_mm256_castsi128_si256(secondLo), secondHi, 0x1);
        }

        template <bool align, size_t step> SIMD_INLINE __m256i LoadAfterLast(const uint8_t * p)
        {
            __m128i lo = _mm_loadu_si128((__m128i*)(p + step)); 
            __m128i hi = LoadHalfAfterLast<step>(LoadHalf<align>((__m128i*)(p + HA)));
            return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 0x1);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadAfterLast(const uint8_t * p, __m256i & first, __m256i & second)
        {
            __m128i firstLo = _mm_loadu_si128((__m128i*)(p + step)); 
            __m128i firstHi = LoadHalfAfterLast<step>(LoadHalf<align>((__m128i*)(p + HA)));
            first = _mm256_inserti128_si256(_mm256_castsi128_si256(firstLo), firstHi, 0x1);

            __m128i secondLo = _mm_loadu_si128((__m128i*)(p + 2*step)); 
            __m128i secondHi = LoadHalfAfterLast<step>(firstHi);
            second = _mm256_inserti128_si256(_mm256_castsi128_si256(secondLo), secondHi, 0x1);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadNose3(const uint8_t * p, __m256i a[3])
        {
            a[0] = LoadBeforeFirst<align, step>(p);
            a[1] = Load<align>((__m256i*)p);
            a[2] = _mm256_loadu_si256((__m256i*)(p + step));
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBody3(const uint8_t * p, __m256i a[3])
        {
            a[0] = _mm256_loadu_si256((__m256i*)(p - step));
            a[1] = Load<align>((__m256i*)p);
            a[2] = _mm256_loadu_si256((__m256i*)(p + step));
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTail3(const uint8_t * p, __m256i a[3])
        {
            a[0] = _mm256_loadu_si256((__m256i*)(p - step));
            a[1] = Load<align>((__m256i*)p);
            a[2] = LoadAfterLast<align, step>(p);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadNose5(const uint8_t * p, __m256i a[5])
        {
            LoadBeforeFirst<align, step>(p, a[1], a[0]);
            a[2] = Load<align>((__m256i*)p);
            a[3] = _mm256_loadu_si256((__m256i*)(p + step));
            a[4] = _mm256_loadu_si256((__m256i*)(p + 2*step));
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBody5(const uint8_t * p, __m256i a[5])
        {
            a[0] = _mm256_loadu_si256((__m256i*)(p - 2*step));
            a[1] = _mm256_loadu_si256((__m256i*)(p - step));
            a[2] = Load<align>((__m256i*)p);
            a[3] = _mm256_loadu_si256((__m256i*)(p + step));
            a[4] = _mm256_loadu_si256((__m256i*)(p + 2*step));
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTail5(const uint8_t * p, __m256i a[5])
        {
            a[0] = _mm256_loadu_si256((__m256i*)(p - 2*step));
            a[1] = _mm256_loadu_si256((__m256i*)(p - step));
            a[2] = Load<align>((__m256i*)p);
            LoadAfterLast<align, step>(p, a[3], a[4]);
        }
    }
#endif//SIMD_AVX2_ENABLE
}
#endif//__SimdLoad_h__
