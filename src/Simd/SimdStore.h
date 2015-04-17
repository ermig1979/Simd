/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_SSE_ENABLE
    namespace Sse
    {
        template <bool align> SIMD_INLINE void Store(float  * p, __m128 a);

        template <> SIMD_INLINE void Store<false>(float  * p, __m128 a)
        {
            _mm_storeu_ps(p, a);
        }

        template <> SIMD_INLINE void Store<true>(float  * p, __m128 a)
        {
            _mm_store_ps(p, a);
        }
    }
#endif//SIMD_SSE_ENABLE

#ifdef SIMD_SSE2_ENABLE
	namespace Sse2
	{
        using namespace Sse;

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

#ifdef SIMD_AVX_ENABLE
    namespace Avx
    {
        template <bool align> SIMD_INLINE void Store(float * p, __m256 a);

        template <> SIMD_INLINE void Store<false>(float * p, __m256 a)
        {
            _mm256_storeu_ps(p, a);
        }

        template <> SIMD_INLINE void Store<true>(float * p, __m256 a)
        {
            _mm256_store_ps(p, a);
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE
	namespace Avx2
	{
        using namespace Avx;

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

#ifdef SIMD_VMX_ENABLE
    namespace Vmx
    {
        template <bool align> SIMD_INLINE void Store(uint8_t * p, v128_u8 a);

        template <> SIMD_INLINE void Store<false>(uint8_t * p, v128_u8 a)
        {
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

        template <bool align> SIMD_INLINE void Store(uint32_t * p, v128_u32 a)
        {
            Store<align>((uint8_t*)p, (v128_u8)a);
        }

        template <bool align> SIMD_INLINE void Store(int32_t * p, v128_s32 a)
        {
            Store<align>((uint8_t*)p, (v128_u8)a);
        }

        template <bool align> SIMD_INLINE void Store(float * p, v128_f32 a)
        {
            Store<align>((uint8_t*)p, (v128_u8)a);
        }

        template <bool align> struct Storer;

        template <> struct Storer<true>
        {
            template <class T> Storer(T * ptr)
                :_ptr((uint8_t*)ptr)
            {
            }

            template <class T> SIMD_INLINE void First(T value)
            {
                vec_st((v128_u8)value, 0, _ptr);
            }

            template <class T> SIMD_INLINE void Next(T value)
            {
                _ptr += A;
                vec_st((v128_u8)value, 0, _ptr);
            }

            SIMD_INLINE void Flush()
            {
            }

        private:
            uint8_t * _ptr;
        };

        template <> struct Storer<false>
        {
            template <class T> SIMD_INLINE Storer(T * ptr)
                :_ptr((uint8_t*)ptr)
            {
                _perm = vec_lvsr(0, _ptr);
                _mask = vec_perm(K8_00, K8_FF, _perm);
            }

            template <class T> SIMD_INLINE void First(T value)
            {
                _last = (v128_u8)value;
                v128_u8 background = vec_ld(0, _ptr);
                v128_u8 foreground = vec_perm(_last, _last, _perm);
                vec_st(vec_sel(background, foreground, _mask), 0, _ptr);
            }

            template <class T> SIMD_INLINE void Next(T value)
            {
                _ptr += A;
                vec_st(vec_perm(_last, (v128_u8)value, _perm), 0, _ptr);
                _last = (v128_u8)value;
            }

            SIMD_INLINE void Flush()
            {
                v128_u8 background = vec_ld(A, _ptr);
                v128_u8 foreground = vec_perm(_last, _last, _perm); 
                vec_st(vec_sel(foreground, background, _mask), A, _ptr);
            }

        private:
            uint8_t * _ptr;
            v128_u8 _perm;
            v128_u8 _mask;
            v128_u8 _last;
        };

        template <bool align, bool first> void Store(Storer<align> & storer, v128_u8 value);

        template <> SIMD_INLINE void Store<true, true>(Storer<true> & storer, v128_u8 value)
        {
            storer.First(value);
        }

        template <> SIMD_INLINE void Store<false, true>(Storer<false> & storer, v128_u8 value)
        {
            storer.First(value);
        }

        template <> SIMD_INLINE void Store<true, false>(Storer<true> & storer, v128_u8 value)
        {
            storer.Next(value);
        }

        template <> SIMD_INLINE void Store<false, false>(Storer<false> & storer, v128_u8 value)
        {
            storer.Next(value);
        }

        template <bool align, bool first> void Store(Storer<align> & storer, v128_u16 value)
        {
            Store<align, first>(storer, (v128_u8)value);
        }

        template <bool align, bool first> void Store(Storer<align> & storer, v128_s16 value)
        {
            Store<align, first>(storer, (v128_u8)value);
        }

        template <bool align> SIMD_INLINE void Flush(Storer<align> & s0)
        {
            s0.Flush();
        }

        template <bool align> SIMD_INLINE void Flush(Storer<align> & s0, Storer<align> & s1)
        {
            s0.Flush(); s1.Flush();
        }

        template <bool align> SIMD_INLINE void Flush(Storer<align> & s0, Storer<align> & s1, Storer<align> & s2)
        {
            s0.Flush(); s1.Flush(); s2.Flush();
        }

        template <bool align> SIMD_INLINE void Flush(Storer<align> & s0, Storer<align> & s1, Storer<align> & s2, Storer<align> & s3)
        {
            s0.Flush(); s1.Flush(); s2.Flush(); s3.Flush();
        }
    }
#endif//SIMD_VMX_ENABLE
}
#endif//__SimdStore_h__
