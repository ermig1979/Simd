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
#ifndef __SimdExtract_h__
#define __SimdExtract_h__

#include "Simd/SimdTypes.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE
	namespace Sse2
	{
		template <int index> SIMD_INLINE int ExtractInt8(__m128i a)
		{
			return _mm_extract_epi16(_mm_srli_si128(a, index & 0x1), index >> 1) & 0xFF;
		}

		template <int index> SIMD_INLINE int ExtractInt16(__m128i a)
		{
			return _mm_extract_epi16(a, index);
		}

		template <int index> SIMD_INLINE int ExtractInt32(__m128i a)
		{
			return _mm_cvtsi128_si32(_mm_srli_si128(a, 4 * index));
		}

		SIMD_INLINE int ExtractInt32Sum(__m128i a)
		{
			return ExtractInt32<0>(a) + ExtractInt32<1>(a) + ExtractInt32<2>(a) + ExtractInt32<3>(a);
		}

		template <int index> SIMD_INLINE int64_t ExtractInt64(__m128i a)
		{
#if defined(_M_X64) && (!defined(_MSC_VER) || (defined(_MSC_VER) && _MSC_VER >= 1600))
			return _mm_cvtsi128_si64(_mm_srli_si128(a, 8 * index));
#else
			return (int64_t)ExtractInt32<2*index + 1>(a)*0x100000000 + (uint32_t)ExtractInt32<2*index>(a);
#endif
		}

		SIMD_INLINE int64_t ExtractInt64Sum(__m128i a)
		{
			return ExtractInt64<0>(a) + ExtractInt64<1>(a);
		}
	}
#endif// SIMD_SSE2_ENABLE
}

#endif//__SimdExtract_h__