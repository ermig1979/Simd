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
#ifndef __SimdLoadBlock_h__
#define __SimdLoadBlock_h__

#include "Simd/SimdLoad.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
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
            a[4] = _mm_loadu_si128((__m128i*)(p + 2 * step));
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBody5(const uint8_t * p, __m128i a[5])
        {
            a[0] = _mm_loadu_si128((__m128i*)(p - 2 * step));
            a[1] = _mm_loadu_si128((__m128i*)(p - step));
            a[2] = Load<align>((__m128i*)p);
            a[3] = _mm_loadu_si128((__m128i*)(p + step));
            a[4] = _mm_loadu_si128((__m128i*)(p + 2 * step));
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTail5(const uint8_t * p, __m128i a[5])
        {
            a[0] = _mm_loadu_si128((__m128i*)(p - 2 * step));
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

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
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
            a[4] = _mm256_loadu_si256((__m256i*)(p + 2 * step));
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBody5(const uint8_t * p, __m256i a[5])
        {
            a[0] = _mm256_loadu_si256((__m256i*)(p - 2 * step));
            a[1] = _mm256_loadu_si256((__m256i*)(p - step));
            a[2] = Load<align>((__m256i*)p);
            a[3] = _mm256_loadu_si256((__m256i*)(p + step));
            a[4] = _mm256_loadu_si256((__m256i*)(p + 2 * step));
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTail5(const uint8_t * p, __m256i a[5])
        {
            a[0] = _mm256_loadu_si256((__m256i*)(p - 2 * step));
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

#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        template <bool align, size_t step> SIMD_INLINE void LoadNose3(const uint8_t * p, __m512i a[3])
        {
            a[0] = LoadBeforeFirst<step>(p);
            a[1] = Load<align>(p);
            a[2] = Load<false>(p + step);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBody3(const uint8_t * p, __m512i a[3])
        {
            a[0] = Load<false>(p - step);
            a[1] = Load<align>(p);
            a[2] = Load<false>(p + step);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTail3(const uint8_t * p, __m512i a[3])
        {
            a[0] = Load<false>(p - step);
            a[1] = Load<align>(p);
            a[2] = LoadAfterLast<step>(p);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadNose5(const uint8_t * p, __m512i a[5])
        {
            a[0] = LoadBeforeFirst2<step>(p);
            a[1] = LoadBeforeFirst<step>(p);
            a[2] = Load<align>(p);
            a[3] = Load<false>(p + step);
            a[4] = Load<false>(p + 2 * step);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBody5(const uint8_t * p, __m512i a[5])
        {
            a[0] = Load<false>(p - 2 * step);
            a[1] = Load<false>(p - step);
            a[2] = Load<align>(p);
            a[3] = Load<false>(p + step);
            a[4] = Load<false>(p + 2 * step);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTail5(const uint8_t * p, __m512i a[5])
        {
            a[0] = Load<false>(p - 2 * step);
            a[1] = Load<false>(p - step);
            a[2] = Load<align>(p);
            a[3] = LoadAfterLast<step>(p);
            a[4] = LoadAfterLast2<step>(p);
        }

        SIMD_INLINE void LoadNoseDx(const uint8_t * p, __m512i a[3])
        {
            a[0] = LoadBeforeFirst<1>(p);
            a[2] = Load<false>(p + 1);
        }

        SIMD_INLINE void LoadBodyDx(const uint8_t * p, __m512i a[3])
        {
            a[0] = Load<false>(p - 1);
            a[2] = Load<false>(p + 1);
        }

        SIMD_INLINE void LoadTailDx(const uint8_t * p, __m512i a[3])
        {
            a[0] = Load<false>(p - 1);
            a[2] = LoadAfterLast<1>(p);
        }
    }
#endif//SIMD_AVX512BW_ENABLE

#ifdef SIMD_VMX_ENABLE
    namespace Vmx
    {
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
            a[4] = Load<false>(p + 2 * step);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBody5(const uint8_t * p, v128_u8 a[5])
        {
            a[0] = Load<false>(p - 2 * step);
            a[1] = Load<false>(p - step);
            a[2] = Load<align>(p);
            a[3] = Load<false>(p + step);
            a[4] = Load<false>(p + 2 * step);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTail5(const uint8_t * p, v128_u8 a[5])
        {
            a[0] = Load<false>(p - 2 * step);
            a[1] = Load<false>(p - step);
            a[2] = Load<align>(p);
            a[3] = LoadAfterLast<step>(a[2]);
            a[4] = LoadAfterLast<step>(a[3]);
        }
    }
#endif//SIMD_VMX_ENABLE

#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        template <bool align, size_t step> SIMD_INLINE void LoadNose3(const uint8_t * p, uint8x16_t a[3])
        {
            a[1] = Load<align>(p);
            a[0] = LoadBeforeFirst<step>(a[1]);
            a[2] = vld1q_u8(p + step);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBody3(const uint8_t * p, uint8x16_t a[3])
        {
#if defined(__GNUC__) && SIMD_NEON_PREFECH_SIZE
            __builtin_prefetch(p + SIMD_NEON_PREFECH_SIZE);
#endif
            a[0] = vld1q_u8(p - step);
            a[1] = Load<align>(p);
            a[2] = vld1q_u8(p + step);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTail3(const uint8_t * p, uint8x16_t a[3])
        {
            a[0] = vld1q_u8(p - step);
            a[1] = Load<align>(p);
            a[2] = LoadAfterLast<step>(a[1]);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadNose5(const uint8_t * p, uint8x16_t a[5])
        {
            a[2] = Load<align>(p);
            a[1] = LoadBeforeFirst<step>(a[2]);
            a[0] = LoadBeforeFirst<step>(a[1]);
            a[3] = vld1q_u8(p + step);
            a[4] = vld1q_u8(p + 2 * step);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadBody5(const uint8_t * p, uint8x16_t a[5])
        {
#if defined(__GNUC__) && SIMD_NEON_PREFECH_SIZE
            __builtin_prefetch(p + SIMD_NEON_PREFECH_SIZE);
#endif
            a[0] = vld1q_u8(p - 2 * step);
            a[1] = vld1q_u8(p - step);
            a[2] = Load<align>(p);
            a[3] = vld1q_u8(p + step);
            a[4] = vld1q_u8(p + 2 * step);
        }

        template <bool align, size_t step> SIMD_INLINE void LoadTail5(const uint8_t * p, uint8x16_t a[5])
        {
            a[0] = vld1q_u8(p - 2 * step);
            a[1] = vld1q_u8(p - step);
            a[2] = Load<align>(p);
            a[3] = LoadAfterLast<step>(a[2]);
            a[4] = LoadAfterLast<step>(a[3]);
        }

        SIMD_INLINE void LoadNoseDx(const uint8_t * p, uint8x16_t a[3])
        {
            a[0] = LoadBeforeFirst<1>(vld1q_u8(p));
            a[2] = vld1q_u8(p + 1);
        }

        SIMD_INLINE void LoadBodyDx(const uint8_t * p, uint8x16_t a[3])
        {
#if defined(__GNUC__) && SIMD_NEON_PREFECH_SIZE
            __builtin_prefetch(p + SIMD_NEON_PREFECH_SIZE);
#endif
            a[0] = vld1q_u8(p - 1);
            a[2] = vld1q_u8(p + 1);
        }

        SIMD_INLINE void LoadTailDx(const uint8_t * p, uint8x16_t a[3])
        {
            a[0] = vld1q_u8(p - 1);
            a[2] = LoadAfterLast<1>(vld1q_u8(p));
        }
    }
#endif//SIMD_NEON_ENABLE
}
#endif//__SimdLoadBlock_h__
