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
#ifndef __SimdSet_h__
#define __SimdSet_h__

#include "Simd/SimdTypes.h"
#include "Simd/SimdConst.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        SIMD_INLINE __m128i SetInt8(char a0, char a1)
        {
            return _mm_unpacklo_epi8(_mm_set1_epi8(a0), _mm_set1_epi8(a1));
        }

        SIMD_INLINE __m128i SetInt16(short a0, short a1)
        {
            return _mm_unpacklo_epi16(_mm_set1_epi16(a0), _mm_set1_epi16(a1));
        }

        SIMD_INLINE __m128i SetInt32(int a0, int a1)
        {
            return _mm_unpacklo_epi32(_mm_set1_epi32(a0), _mm_set1_epi32(a1));
        }

		SIMD_INLINE __m128 SetFloat(float a0, float a1)
		{
			return _mm_unpacklo_ps(_mm_set_ps1(a0), _mm_set_ps1(a1));
		}
	}
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE
	namespace Avx2
	{
		SIMD_INLINE __m256i SetInt8(char a0, char a1)
		{
			return _mm256_unpacklo_epi8(_mm256_set1_epi8(a0), _mm256_set1_epi8(a1));
		}

		SIMD_INLINE __m256i SetInt16(short a0, short a1)
		{
			return _mm256_unpacklo_epi16(_mm256_set1_epi16(a0), _mm256_set1_epi16(a1));
		}

		SIMD_INLINE __m256i SetInt32(int a0, int a1)
		{
			return _mm256_unpacklo_epi32(_mm256_set1_epi32(a0), _mm256_set1_epi32(a1));
		}

		SIMD_INLINE __m256 SetFloat(float a0, float a1)
		{
			return _mm256_unpacklo_ps(_mm256_set1_ps(a0), _mm256_set1_ps(a1));
		}

        template <class T> SIMD_INLINE __m256i SetMask(T first, size_t position, T second)
        {
            const size_t size = A/sizeof(T);
            assert(position <= size);
            T mask[size];
            for(size_t i = 0; i < position; ++i)
                mask[i] = first;
            for(size_t i = position; i < size; ++i)
                mask[i] = second;
            return _mm256_loadu_si256((__m256i*)mask);
        }
	}
#endif// SIMD_AVX2_ENABLE
}

#endif//__SimdSet_h__
