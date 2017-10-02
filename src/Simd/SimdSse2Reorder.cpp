/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        SIMD_INLINE __m128i Swap8(__m128i value)
        {
            return _mm_or_si128(_mm_srli_epi16(value, 8), _mm_slli_epi16(value, 8));
        }

        template <bool align> SIMD_INLINE void Reorder16bit(const uint8_t * src, uint8_t * dst)
        {
            __m128i _src = Load<align>((__m128i*)src);
            Store<align>((__m128i*)dst, Swap8(_src));
        }

        template <bool align> void Reorder16bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size >= A && size % 2 == 0);

            size_t alignedSize = AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                Reorder16bit<align>(src + i, dst + i);
            for (size_t i = alignedSize; i < size; i += 2)
                Base::Reorder16bit(src + i, dst + i);
        }

        void Reorder16bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Reorder16bit<true>(src, size, dst);
            else
                Reorder16bit<false>(src, size, dst);
        }

        SIMD_INLINE __m128i Swap16(__m128i value)
        {
            return _mm_or_si128(_mm_srli_epi32(value, 16), _mm_slli_epi32(value, 16));
        }

        template <bool align> SIMD_INLINE void Reorder32bit(const uint8_t * src, uint8_t * dst)
        {
            __m128i _src = Load<align>((__m128i*)src);
            Store<align>((__m128i*)dst, Swap16(Swap8(_src)));
        }

        template <bool align> void Reorder32bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size >= A && size % 4 == 0);

            size_t alignedSize = AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                Reorder32bit<align>(src + i, dst + i);
            for (size_t i = alignedSize; i < size; i += 4)
                Base::Reorder32bit(src + i, dst + i);
        }

        void Reorder32bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Reorder32bit<true>(src, size, dst);
            else
                Reorder32bit<false>(src, size, dst);
        }

        SIMD_INLINE __m128i Swap32(__m128i value)
        {
            return _mm_shuffle_epi32(value, 0xB1);
        }

        template <bool align> SIMD_INLINE void Reorder64bit(const uint8_t * src, uint8_t * dst)
        {
            __m128i _src = Load<align>((__m128i*)src);
            Store<align>((__m128i*)dst, Swap32(Swap16(Swap8(_src))));
        }

        template <bool align> void Reorder64bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size >= A && size % 8 == 0);

            size_t alignedSize = AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                Reorder64bit<align>(src + i, dst + i);
            for (size_t i = alignedSize; i < size; i += 8)
                Base::Reorder64bit(src + i, dst + i);
        }

        void Reorder64bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Reorder64bit<true>(src, size, dst);
            else
                Reorder64bit<false>(src, size, dst);
        }
    }
#endif// SIMD_SSE2_ENABLE
}
