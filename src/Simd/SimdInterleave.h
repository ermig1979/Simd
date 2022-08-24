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
#ifndef __SimdInterleave_h__
#define __SimdInterleave_h__

#include "Simd/SimdConst.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        template <int index> __m128i InterleaveBgr(__m128i blue, __m128i green, __m128i red);

        template<> SIMD_INLINE __m128i InterleaveBgr<0>(__m128i blue, __m128i green, __m128i red)
        {
            return
                _mm_or_si128(_mm_shuffle_epi8(blue, K8_SHUFFLE_BLUE_TO_BGR0),
                    _mm_or_si128(_mm_shuffle_epi8(green, K8_SHUFFLE_GREEN_TO_BGR0),
                        _mm_shuffle_epi8(red, K8_SHUFFLE_RED_TO_BGR0)));
        }

        template<> SIMD_INLINE __m128i InterleaveBgr<1>(__m128i blue, __m128i green, __m128i red)
        {
            return
                _mm_or_si128(_mm_shuffle_epi8(blue, K8_SHUFFLE_BLUE_TO_BGR1),
                    _mm_or_si128(_mm_shuffle_epi8(green, K8_SHUFFLE_GREEN_TO_BGR1),
                        _mm_shuffle_epi8(red, K8_SHUFFLE_RED_TO_BGR1)));
        }

        template<> SIMD_INLINE __m128i InterleaveBgr<2>(__m128i blue, __m128i green, __m128i red)
        {
            return
                _mm_or_si128(_mm_shuffle_epi8(blue, K8_SHUFFLE_BLUE_TO_BGR2),
                    _mm_or_si128(_mm_shuffle_epi8(green, K8_SHUFFLE_GREEN_TO_BGR2),
                        _mm_shuffle_epi8(red, K8_SHUFFLE_RED_TO_BGR2)));
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        template <int index> __m256i InterleaveBgr(__m256i blue, __m256i green, __m256i red);

        template<> SIMD_INLINE __m256i InterleaveBgr<0>(__m256i blue, __m256i green, __m256i red)
        {
            return
                _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(blue, 0x44), K8_SHUFFLE_PERMUTED_BLUE_TO_BGR0),
                    _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(green, 0x44), K8_SHUFFLE_PERMUTED_GREEN_TO_BGR0),
                        _mm256_shuffle_epi8(_mm256_permute4x64_epi64(red, 0x44), K8_SHUFFLE_PERMUTED_RED_TO_BGR0)));
        }

        template<> SIMD_INLINE __m256i InterleaveBgr<1>(__m256i blue, __m256i green, __m256i red)
        {
            return
                _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(blue, 0x99), K8_SHUFFLE_PERMUTED_BLUE_TO_BGR1),
                    _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(green, 0x99), K8_SHUFFLE_PERMUTED_GREEN_TO_BGR1),
                        _mm256_shuffle_epi8(_mm256_permute4x64_epi64(red, 0x99), K8_SHUFFLE_PERMUTED_RED_TO_BGR1)));
        }

        template<> SIMD_INLINE __m256i InterleaveBgr<2>(__m256i blue, __m256i green, __m256i red)
        {
            return
                _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(blue, 0xEE), K8_SHUFFLE_PERMUTED_BLUE_TO_BGR2),
                    _mm256_or_si256(_mm256_shuffle_epi8(_mm256_permute4x64_epi64(green, 0xEE), K8_SHUFFLE_PERMUTED_GREEN_TO_BGR2),
                        _mm256_shuffle_epi8(_mm256_permute4x64_epi64(red, 0xEE), K8_SHUFFLE_PERMUTED_RED_TO_BGR2)));
        }
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <int index> __m512i InterleaveBgr(__m512i blue, __m512i green, __m512i red);

        template<> SIMD_INLINE __m512i InterleaveBgr<0>(__m512i blue, __m512i green, __m512i red)
        {
            return
                _mm512_or_si512(_mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR0, blue), K8_SHUFFLE_BLUE_TO_BGR0),
                    _mm512_or_si512(_mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR0, green), K8_SHUFFLE_GREEN_TO_BGR0),
                        _mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR0, red), K8_SHUFFLE_RED_TO_BGR0)));
        }

        template<> SIMD_INLINE __m512i InterleaveBgr<1>(__m512i blue, __m512i green, __m512i red)
        {
            return
                _mm512_or_si512(_mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR1, blue), K8_SHUFFLE_BLUE_TO_BGR1),
                    _mm512_or_si512(_mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR1, green), K8_SHUFFLE_GREEN_TO_BGR1),
                        _mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR1, red), K8_SHUFFLE_RED_TO_BGR1)));
        }

        template<> SIMD_INLINE __m512i InterleaveBgr<2>(__m512i blue, __m512i green, __m512i red)
        {
            return
                _mm512_or_si512(_mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR2, blue), K8_SHUFFLE_BLUE_TO_BGR2),
                    _mm512_or_si512(_mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR2, green), K8_SHUFFLE_GREEN_TO_BGR2),
                        _mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_COLOR_TO_BGR2, red), K8_SHUFFLE_RED_TO_BGR2)));
        }
    }
#endif
}

#endif
