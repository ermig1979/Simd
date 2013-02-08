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

		template <size_t count> SIMD_INLINE __m128i LoadBeforeFirst(__m128i first)
		{
			return _mm_or_si128(_mm_slli_si128(first, count), _mm_and_si128(first, _mm_srli_si128(K_INV_ZERO, A - count)));
		}

		template <size_t count> SIMD_INLINE __m128i LoadAfterLast(__m128i last)
		{
			return _mm_or_si128(_mm_srli_si128(last, count), _mm_and_si128(last, _mm_slli_si128(K_INV_ZERO, A - count)));
		}

		template <bool align, size_t step> SIMD_INLINE void LoadNose3(const uchar * p, __m128i a[3])
		{
			a[1] = Load<align>((__m128i*)p);
			a[0] = LoadBeforeFirst<step>(a[1]);
			a[2] = _mm_loadu_si128((__m128i*)(p + step));
		}

		template <bool align, size_t step> SIMD_INLINE void LoadBody3(const uchar * p, __m128i a[3])
		{
			a[0] = _mm_loadu_si128((__m128i*)(p - step));
			a[1] = Load<align>((__m128i*)p);
			a[2] = _mm_loadu_si128((__m128i*)(p + step));
		}

		template <bool align, size_t step> SIMD_INLINE void LoadTail3(const uchar * p, __m128i a[3])
		{
			a[0] = _mm_loadu_si128((__m128i*)(p - step));
			a[1] = Load<align>((__m128i*)p);
			a[2] = LoadAfterLast<step>(a[1]);
		}

		template <bool align, size_t step> SIMD_INLINE void LoadNose5(const uchar * p, __m128i a[5])
		{
			a[2] = Load<align>((__m128i*)p);
			a[1] = LoadBeforeFirst<step>(a[2]);
			a[0] = LoadBeforeFirst<step>(a[1]);
			a[3] = _mm_loadu_si128((__m128i*)(p + step));
			a[4] = _mm_loadu_si128((__m128i*)(p + 2*step));
		}

		template <bool align, size_t step> SIMD_INLINE void LoadBody5(const uchar * p, __m128i a[5])
		{
			a[0] = _mm_loadu_si128((__m128i*)(p - 2*step));
			a[1] = _mm_loadu_si128((__m128i*)(p - step));
			a[2] = Load<align>((__m128i*)p);
			a[3] = _mm_loadu_si128((__m128i*)(p + step));
			a[4] = _mm_loadu_si128((__m128i*)(p + 2*step));
		}

		template <bool align, size_t step> SIMD_INLINE void LoadTail5(const uchar * p, __m128i a[5])
		{
			a[0] = _mm_loadu_si128((__m128i*)(p - 2*step));
			a[1] = _mm_loadu_si128((__m128i*)(p - step));
			a[2] = Load<align>((__m128i*)p);
			a[3] = LoadAfterLast<step>(a[2]);
			a[4] = LoadAfterLast<step>(a[3]);
		}
	}
#endif//SIMD_SSE2_ENABLE
}
#endif//__SimdLoad_h__
