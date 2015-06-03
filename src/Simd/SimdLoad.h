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
#ifndef __SimdLoad_h__
#define __SimdLoad_h__

#include "Simd/SimdConst.h"

namespace Simd
{
#ifdef SIMD_SSE_ENABLE
    namespace Sse
    {
        template <bool align> SIMD_INLINE __m128 Load(const float * p);

        template <> SIMD_INLINE __m128 Load<false>(const float * p)
        {
            return _mm_loadu_ps(p); 
        }

        template <> SIMD_INLINE __m128 Load<true>(const float * p)
        {
            return _mm_load_ps(p); 
        }
    }
#endif//SIMD_SSE_ENABLE

#ifdef SIMD_SSE2_ENABLE
	namespace Sse2
	{
        using namespace Sse;

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

        SIMD_INLINE void LoadNoseDx(const uint8_t * p, __m128i a[3])
        {
            a[0] = LoadBeforeFirst<1>(_mm_loadu_si128((__m128i*)p));
            a[2] = _mm_loadu_si128((__m128i*)(p + 1));
        }

        SIMD_INLINE void LoadBodyDx(const uint8_t * p, __m128i a[3])
        {
            a[0] = _mm_loadu_si128((__m128i*)(p - 1));
            a[2] = _mm_loadu_si128((__m128i*)(p + 1));
        }

        SIMD_INLINE void LoadTailDx(const uint8_t * p, __m128i a[3])
        {
            a[0] = _mm_loadu_si128((__m128i*)(p - 1));
            a[2] = LoadAfterLast<1>(_mm_loadu_si128((__m128i*)p));
        }
	}
#endif//SIMD_SSE2_ENABLE

#ifdef SIMD_AVX_ENABLE
    namespace Avx
    {
        template <bool align> SIMD_INLINE __m256 Load(const float * p);

        template <> SIMD_INLINE __m256 Load<false>(const float * p)
        {
            return _mm256_loadu_ps(p); 
        }

        template <> SIMD_INLINE __m256 Load<true>(const float * p)
        {
            return _mm256_load_ps(p); 
        }
    }
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE
	namespace Avx2
	{
        using namespace Avx;

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

        SIMD_INLINE __m256i PermutedUnpackLoU8(__m256i a, __m256i b = K_ZERO)
        {
            return _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(a, b), 0xD8);
        }

        SIMD_INLINE __m256i PermutedUnpackHiU8(__m256i a, __m256i b = K_ZERO)
        {
            return _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(a, b), 0xD8);
        }

        SIMD_INLINE __m256i PermutedUnpackLoU16(__m256i a, __m256i b = K_ZERO)
        {
            return _mm256_permute4x64_epi64(_mm256_unpacklo_epi16(a, b), 0xD8);
        }

        SIMD_INLINE __m256i PermutedUnpackHiU16(__m256i a, __m256i b = K_ZERO)
        {
            return _mm256_permute4x64_epi64(_mm256_unpackhi_epi16(a, b), 0xD8);
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

        SIMD_INLINE void LoadNoseDx(const uint8_t * p, __m256i a[3])
        {
            a[0] = LoadBeforeFirst<false, 1>(p);
            a[2] = _mm256_loadu_si256((__m256i*)(p + 1));
        }

        SIMD_INLINE void LoadBodyDx(const uint8_t * p, __m256i a[3])
        {
            a[0] = _mm256_loadu_si256((__m256i*)(p - 1));
            a[2] = _mm256_loadu_si256((__m256i*)(p + 1));
        }

        SIMD_INLINE void LoadTailDx(const uint8_t * p, __m256i a[3])
        {
            a[0] = _mm256_loadu_si256((__m256i*)(p - 1));
            a[2] = LoadAfterLast<false, 1>(p);
        }
    }
#endif//SIMD_AVX2_ENABLE

#ifdef SIMD_VMX_ENABLE
    namespace Vmx
    {
        template <bool align> SIMD_INLINE v128_u8 Load(const uint8_t * p);

        template <> SIMD_INLINE v128_u8 Load<false>(const uint8_t * p)
        {
            v128_u8 lo = vec_ld(0, p);
            v128_u8 hi = vec_ld(A, p);
            return vec_perm(lo, hi, vec_lvsl(0, p));        
        }        
        
        template <> SIMD_INLINE v128_u8 Load<true>(const uint8_t * p)
        {
            return vec_ld(0, p); 
        }

        template <bool align> SIMD_INLINE v128_u16 Load(const uint16_t * p)
        {
            return (v128_u16)Load<align>((const uint8_t*)p);
        }

        template <bool align> SIMD_INLINE v128_s16 Load(const int16_t * p)
        {
            return (v128_s16)Load<align>((const uint8_t*)p);
        }

        template <bool align> SIMD_INLINE v128_u32 Load(const uint32_t * p)
        {
            return (v128_u32)Load<align>((const uint8_t*)p);
        }

        template <bool align> SIMD_INLINE v128_f32 Load(const float * p)
        {
            return (v128_f32)Load<align>((const uint8_t*)p);
        }

        template <bool align> SIMD_INLINE v128_u8 LoadMaskU8(const uint8_t * p, v128_u8 index)
        {
            return (v128_u8)vec_cmpeq(Load<align>(p), index);
        }

        template <size_t count> SIMD_INLINE v128_u8 LoadBeforeFirst(v128_u8 first);

        template <> SIMD_INLINE v128_u8 LoadBeforeFirst<1>(v128_u8 first)
        {
            return vec_perm(first, first, K8_PERM_LOAD_BEFORE_FIRST_1);
        }

        template <> SIMD_INLINE v128_u8 LoadBeforeFirst<2>(v128_u8 first)
        {
            return vec_perm(first, first, K8_PERM_LOAD_BEFORE_FIRST_2);
        }

        template <> SIMD_INLINE v128_u8 LoadBeforeFirst<3>(v128_u8 first)
        {
            return vec_perm(first, first, K8_PERM_LOAD_BEFORE_FIRST_3);
        }

        template <> SIMD_INLINE v128_u8 LoadBeforeFirst<4>(v128_u8 first)
        {
            return vec_perm(first, first, K8_PERM_LOAD_BEFORE_FIRST_4);
        }

        template <size_t count> SIMD_INLINE v128_u8 LoadAfterLast(v128_u8 last);

        template <> SIMD_INLINE v128_u8 LoadAfterLast<1>(v128_u8 last)
        {
            return vec_perm(last, last, K8_PERM_LOAD_AFTER_LAST_1);
        }

        template <> SIMD_INLINE v128_u8 LoadAfterLast<2>(v128_u8 last)
        {
            return vec_perm(last, last, K8_PERM_LOAD_AFTER_LAST_2);
        }

        template <> SIMD_INLINE v128_u8 LoadAfterLast<3>(v128_u8 last)
        {
            return vec_perm(last, last, K8_PERM_LOAD_AFTER_LAST_3);
        }

        template <> SIMD_INLINE v128_u8 LoadAfterLast<4>(v128_u8 last)
        {
            return vec_perm(last, last, K8_PERM_LOAD_AFTER_LAST_4);
        }

        template <bool align> struct Loader;

        template <> struct Loader<true>
        {
            template <class T> Loader(const T * ptr)
                :_ptr((const uint8_t*)ptr)
            {
            }

            SIMD_INLINE v128_u8 First() const
            {
                return vec_ld(0, _ptr);
            }

            SIMD_INLINE v128_u8 Next() const
            {
                _ptr += A;
                return vec_ld(0, _ptr);
            }

        private:
            mutable const uint8_t * _ptr;
        };

        template <> struct Loader<false>
        {
            template <class T> SIMD_INLINE Loader(const T * ptr)
                :_ptr((const uint8_t*)ptr)
            {
                _perm = vec_lvsl(0, _ptr);
            }

            SIMD_INLINE v128_u8 First() const
            {
                return vec_perm(vec_ld(0, _ptr), vec_ld(A, _ptr), _perm);
            }

            SIMD_INLINE v128_u8 Next() const
            {
                _ptr += A;
                return vec_perm(vec_ld(0, _ptr), vec_ld(A, _ptr), _perm);
            }

        private:
            mutable const uint8_t * _ptr;
            v128_u8 _perm;
        };

        template <bool align, bool first> v128_u8 Load(const Loader<align> & loader);

        template <> SIMD_INLINE v128_u8 Load<true, true>(const Loader<true> & loader)
        {
            return loader.First();
        }

        template <> SIMD_INLINE v128_u8 Load<false, true>(const Loader<false> & loader)
        {
            return loader.First();
        }

        template <> SIMD_INLINE v128_u8 Load<true, false>(const Loader<true> & loader)
        {
            return loader.Next();
        }

        template <> SIMD_INLINE v128_u8 Load<false, false>(const Loader<false> & loader)
        {
            return loader.Next();
        }

        template <bool align, size_t step> SIMD_INLINE void LoadNose3(const uint8_t * p, v128_u8 a[3])
        {
            a[1] = Load<align>(p);
            a[0] = LoadBeforeFirst<step>(a[1]);
            a[2] = Load<false>(p + step);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBody3(const uint8_t * p, v128_u8 a[3])
        {
            a[0] = Load<false>(p - step);
            a[1] = Load<align>(p);
            a[2] = Load<false>(p + step);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTail3(const uint8_t * p, v128_u8 a[3])
        {
            a[0] = Load<false>(p - step);
            a[1] = Load<align>(p);
            a[2] = LoadAfterLast<step>(a[1]);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadNose5(const uint8_t * p, v128_u8 a[5])
        {
            a[2] = Load<align>(p);
            a[1] = LoadBeforeFirst<step>(a[2]);
            a[0] = LoadBeforeFirst<step>(a[1]);
            a[3] = Load<false>(p + step);
            a[4] = Load<false>(p + 2*step);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBody5(const uint8_t * p, v128_u8 a[5])
        {
            a[0] = Load<false>(p - 2*step);
            a[1] = Load<false>(p - step);
            a[2] = Load<align>(p);
            a[3] = Load<false>(p + step);
            a[4] = Load<false>(p + 2*step);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTail5(const uint8_t * p, v128_u8 a[5])
        {
            a[0] = Load<false>(p - 2*step);
            a[1] = Load<false>(p - step);
            a[2] = Load<align>(p);
            a[3] = LoadAfterLast<step>(a[2]);
            a[4] = LoadAfterLast<step>(a[3]);
        }

        template <int part> v128_u16 UnpackU8(v128_u8 a, v128_u8 b = K8_00);

        template <> SIMD_INLINE v128_u16 UnpackU8<0>(v128_u8 a, v128_u8 b)
        {
            return (v128_u16)vec_perm(a, b, K8_PERM_UNPACK_LO_U8);
        }

        template <> SIMD_INLINE v128_u16 UnpackU8<1>(v128_u8 a, v128_u8 b)
        {
            return (v128_u16)vec_perm(a, b, K8_PERM_UNPACK_HI_U8);
        }

        SIMD_INLINE v128_u16 UnpackLoU8(v128_u8 a, v128_u8 b = K8_00)
        {
            return (v128_u16)vec_perm(a, b, K8_PERM_UNPACK_LO_U8);
        }

        SIMD_INLINE v128_u16 UnpackHiU8(v128_u8 a, v128_u8 b = K8_00)
        {
            return (v128_u16)vec_perm(a, b, K8_PERM_UNPACK_HI_U8);
        }

        SIMD_INLINE v128_u32 UnpackLoU16(v128_u16 a, v128_u16 b = K16_0000)
        {
            return (v128_u32)vec_perm(a, b, K8_PERM_UNPACK_LO_U16);
        }

        SIMD_INLINE v128_u32 UnpackHiU16(v128_u16 a, v128_u16 b = K16_0000)
        {
            return (v128_u32)vec_perm(a, b, K8_PERM_UNPACK_HI_U16);
        }
    }
#endif//SIMD_VMX_ENABLE
}
#endif//__SimdLoad_h__
