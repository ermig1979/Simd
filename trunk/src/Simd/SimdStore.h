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
#ifndef __SimdStore_h__
#define __SimdStore_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE
	namespace Sse2
	{
		template <bool align> SIMD_INLINE void Store(__m128i * p, __m128i a);

		template <> SIMD_INLINE void Store<false>(__m128i * p, __m128i a)
		{
			return _mm_storeu_si128(p, a); 
		}

		template <> SIMD_INLINE void Store<true>(__m128i * p, __m128i a)
		{
			return _mm_store_si128(p, a); 
		}
	}
#endif//SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE
	namespace Avx2
	{
		template <bool align> SIMD_INLINE void Store(__m256i * p, __m256i a);

		template <> SIMD_INLINE void Store<false>(__m256i * p, __m256i a)
		{
			return _mm256_storeu_si256(p, a); 
		}

		template <> SIMD_INLINE void Store<true>(__m256i * p, __m256i a)
		{
			return _mm256_store_si256(p, a); 
		}
	}
#endif//SIMD_SAVX2_ENABLE
}
#endif//__SimdStore_h__
