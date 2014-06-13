/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
#include "Simd/SimdLoad.h"
#include "Simd/SimdMath.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE
	namespace Sse2
	{
		template <bool align> SIMD_INLINE void Store(__m128i * p, __m128i a);

		template <> SIMD_INLINE void Store<false>(__m128i * p, __m128i a)
		{
			_mm_storeu_si128(p, a);
		}

		template <> SIMD_INLINE void Store<true>(__m128i * p, __m128i a)
		{
			_mm_store_si128(p, a);
		}

        template <bool align> SIMD_INLINE void StoreMasked(__m128i * p, __m128i value, __m128i mask)
        {
            __m128i old = Load<align>(p);
            Store<align>(p, Combine(mask, value, old));
        }

        template <bool align> SIMD_INLINE void AlphaBlending(const __m128i * src, __m128i * dst, __m128i alpha)
        {
            __m128i _src = Load<align>(src);
            __m128i _dst = Load<align>(dst);
            __m128i lo = AlphaBlendingI16(_mm_unpacklo_epi8(_src, K_ZERO), _mm_unpacklo_epi8(_dst, K_ZERO), _mm_unpacklo_epi8(alpha, K_ZERO));
            __m128i hi = AlphaBlendingI16(_mm_unpackhi_epi8(_src, K_ZERO), _mm_unpackhi_epi8(_dst, K_ZERO), _mm_unpackhi_epi8(alpha, K_ZERO));
            Store<align>(dst, _mm_packus_epi16(lo, hi));
        }
	}
#endif//SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE
	namespace Avx2
	{
		template <bool align> SIMD_INLINE void Store(__m256i * p, __m256i a);

		template <> SIMD_INLINE void Store<false>(__m256i * p, __m256i a)
		{
			_mm256_storeu_si256(p, a);
		}

		template <> SIMD_INLINE void Store<true>(__m256i * p, __m256i a)
		{
			_mm256_store_si256(p, a);
		}

        template <bool align> SIMD_INLINE void StoreMasked(__m256i * p, __m256i value, __m256i mask)
        {
            __m256i old = Load<align>(p);
            Store<align>(p, Combine(mask, value, old));
        }

        SIMD_INLINE __m256i PackI16ToI8(__m256i lo, __m256i hi)
        {
            return _mm256_permute4x64_epi64(_mm256_packs_epi16(lo, hi), 0xD8);
        }

        SIMD_INLINE __m256i PackU16ToU8(__m256i lo, __m256i hi)
        {
            return _mm256_permute4x64_epi64(_mm256_packus_epi16(lo, hi), 0xD8);
        }

        SIMD_INLINE __m256i PackI32ToI16(__m256i lo, __m256i hi)
        {
            return _mm256_permute4x64_epi64(_mm256_packs_epi32(lo, hi), 0xD8);
        }

        SIMD_INLINE __m256i PackU32ToI16(__m256i lo, __m256i hi)
        {
            return _mm256_permute4x64_epi64(_mm256_packus_epi32(lo, hi), 0xD8);
        }

        SIMD_INLINE void Permute2x128(__m256i & lo, __m256i & hi)
        {
            __m256i _lo = lo;
            lo = _mm256_permute2x128_si256(lo, hi, 0x20);
            hi = _mm256_permute2x128_si256(_lo, hi, 0x31);
        }
    }
#endif//SIMD_SAVX2_ENABLE

#ifdef SIMD_VSX_ENABLE
    namespace Vsx
    {
        template <bool align> SIMD_INLINE void Store(uint8_t * p, v128_u8 a);

        template <> SIMD_INLINE void Store<false>(uint8_t * p, v128_u8 a)
        {
            //vec_vsx_st(a, 0, p);
            v128_u8 lo = vec_ld(0, p);
            v128_u8 hi = vec_ld(A, p);
            v128_u8 perm = vec_lvsr(0, p);
            v128_u8 mask = vec_perm(K8_00, K8_FF, perm);
            v128_u8 value = vec_perm(a, a, perm);
            vec_st(vec_sel(lo, value, mask), 0, p);
            vec_st(vec_sel(value, hi, mask), A, p);
        }

        template <> SIMD_INLINE void Store<true>(uint8_t * p, v128_u8 a)
        {
            vec_st(a, 0, p);
        }

        template <bool align> SIMD_INLINE void Store(uint16_t * p, v128_u16 a)
        {
            Store<align>((uint8_t*)p, (v128_u8)a);
        }
    }
#endif//SIMD_VSX_ENABLE
}
#endif//__SimdStore_h__
