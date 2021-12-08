/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#ifndef __SimdDeinterleave_h__
#define __SimdDeinterleave_h__

#include "Simd/SimdConst.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        template<int part> SIMD_INLINE __m128i Deinterleave8(__m128i ab0, __m128i ab1);

        template<> SIMD_INLINE __m128i Deinterleave8<0>(__m128i ab0, __m128i ab1)
        {
            return _mm_packus_epi16(_mm_and_si128(ab0, K16_00FF), _mm_and_si128(ab1, K16_00FF));
        }

        template<> SIMD_INLINE __m128i Deinterleave8<1>(__m128i ab0, __m128i ab1)
        {
            return _mm_packus_epi16(
                _mm_and_si128(_mm_srli_si128(ab0, 1), K16_00FF), 
                _mm_and_si128(_mm_srli_si128(ab1, 1), K16_00FF));
        }
    }
#endif

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        const __m128i K8_SHUFFLE_DEINTERLEAVE_8_TO_64 = SIMD_MM_SETR_EPI8(0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF);

        SIMD_INLINE __m128i Deinterleave8To64(__m128i ab8)
        {
            return _mm_shuffle_epi8(ab8, K8_SHUFFLE_DEINTERLEAVE_8_TO_64);
        }

        template<int part> SIMD_INLINE __m128i Deinterleave8(__m128i ab0, __m128i ab1);

        template<> SIMD_INLINE __m128i Deinterleave8<0>(__m128i ab0, __m128i ab1)
        {
            return _mm_packus_epi16(_mm_and_si128(ab0, K16_00FF), _mm_and_si128(ab1, K16_00FF));
        }

        template<> SIMD_INLINE __m128i Deinterleave8<1>(__m128i ab0, __m128i ab1)
        {
            return _mm_packus_epi16(
                _mm_and_si128(_mm_srli_si128(ab0, 1), K16_00FF),
                _mm_and_si128(_mm_srli_si128(ab1, 1), K16_00FF));
        }

        template<int part> SIMD_INLINE __m128i Deinterleave64(__m128i ab0, __m128i ab1);

        template<> SIMD_INLINE __m128i Deinterleave64<0>(__m128i ab0, __m128i ab1)
        {
            return _mm_unpacklo_epi64(ab0, ab1);
        }

        template<> SIMD_INLINE __m128i Deinterleave64<1>(__m128i ab0, __m128i ab1)
        {
            return _mm_unpackhi_epi64(ab0, ab1);
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        const __m256i K8_SHUFFLE_DEINTERLEAVE_8_TO_64 = SIMD_MM256_SETR_EPI8(
            0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF,
            0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF);

        SIMD_INLINE __m256i Deinterleave8To64(__m256i ab8)
        {
            return _mm256_shuffle_epi8(ab8, K8_SHUFFLE_DEINTERLEAVE_8_TO_64);
        }

        SIMD_INLINE __m256i Deinterleave32To64(__m256i ab32)
        {
            return _mm256_shuffle_epi32(ab32, 0xD8);
        }

        template<int part> SIMD_INLINE __m256i Deinterleave8(__m256i ab0, __m256i ab1);
#if 1
        template<> SIMD_INLINE __m256i Deinterleave8<0>(__m256i ab0, __m256i ab1)
        {
            return PackI16ToU8(_mm256_and_si256(ab0, K16_00FF), _mm256_and_si256(ab1, K16_00FF));
        }

        template<> SIMD_INLINE __m256i Deinterleave8<1>(__m256i ab0, __m256i ab1)
        {
            return PackI16ToU8(
                _mm256_and_si256(_mm256_srli_si256(ab0, 1), K16_00FF),
                _mm256_and_si256(_mm256_srli_si256(ab1, 1), K16_00FF));
        }
#else
        template<> SIMD_INLINE __m256i Deinterleave8<0>(__m256i ab0, __m256i ab1)
        {
            __m256i _ab0 = Deinterleave8To64(ab0);
            __m256i _ab1 = Deinterleave8To64(ab1);
            return _mm256_permute4x64_epi64(_mm256_unpacklo_epi64(_ab0, _ab1), 0xD8);
        }

        template<> SIMD_INLINE __m256i Deinterleave8<1>(__m256i ab0, __m256i ab1)
        {
            __m256i _ab0 = Deinterleave8To64(ab0);
            __m256i _ab1 = Deinterleave8To64(ab1);
            return _mm256_permute4x64_epi64(_mm256_unpackhi_epi64(_ab0, _ab1), 0xD8);
        }
#endif

        template<int part> SIMD_INLINE __m256i Deinterleave64(__m256i ab0, __m256i ab1);

        template<> SIMD_INLINE __m256i Deinterleave64<0>(__m256i ab0, __m256i ab1)
        {
            return _mm256_permute4x64_epi64(_mm256_unpacklo_epi64(ab0, ab1), 0xD8);
        }

        template<> SIMD_INLINE __m256i Deinterleave64<1>(__m256i ab0, __m256i ab1)
        {
            return _mm256_permute4x64_epi64(_mm256_unpackhi_epi64(ab0, ab1), 0xD8);
        }
    }
#endif
}

#endif//__SimdDeinterleave_h__
