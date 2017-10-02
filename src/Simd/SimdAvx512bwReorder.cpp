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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        const __m512i K8_SHUFFLE_REORDER_16 = SIMD_MM512_SETR_EPI8(
            0x1, 0x0, 0x3, 0x2, 0x5, 0x4, 0x7, 0x6, 0x9, 0x8, 0xB, 0xA, 0xD, 0xC, 0xF, 0xE,
            0x1, 0x0, 0x3, 0x2, 0x5, 0x4, 0x7, 0x6, 0x9, 0x8, 0xB, 0xA, 0xD, 0xC, 0xF, 0xE,
            0x1, 0x0, 0x3, 0x2, 0x5, 0x4, 0x7, 0x6, 0x9, 0x8, 0xB, 0xA, 0xD, 0xC, 0xF, 0xE,
            0x1, 0x0, 0x3, 0x2, 0x5, 0x4, 0x7, 0x6, 0x9, 0x8, 0xB, 0xA, 0xD, 0xC, 0xF, 0xE);

        template <bool align, bool mask> SIMD_INLINE void Reorder16bit(const uint8_t * src, uint8_t * dst, __mmask64 tail = -1)
        {
            Store<align, mask>(dst, _mm512_shuffle_epi8((Load<align, mask>(src, tail)), K8_SHUFFLE_REORDER_16), tail);
        }

        template <bool align> SIMD_INLINE void Reorder16bit4(const uint8_t * src, uint8_t * dst)
        {
            Store<align>(dst + 0 * A, _mm512_shuffle_epi8(Load<align>(src + 0 * A), K8_SHUFFLE_REORDER_16));
            Store<align>(dst + 1 * A, _mm512_shuffle_epi8(Load<align>(src + 1 * A), K8_SHUFFLE_REORDER_16));
            Store<align>(dst + 2 * A, _mm512_shuffle_epi8(Load<align>(src + 2 * A), K8_SHUFFLE_REORDER_16));
            Store<align>(dst + 3 * A, _mm512_shuffle_epi8(Load<align>(src + 3 * A), K8_SHUFFLE_REORDER_16));
        }

        template <bool align> void Reorder16bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size % 2 == 0);

            size_t alignedSize = AlignLo(size, A);
            size_t fullAlignedSize = AlignLo(size, QA);
            __mmask64 tailMask = TailMask64(size - alignedSize);
            size_t i = 0;
            for (; i < fullAlignedSize; i += QA)
                Reorder16bit4<align>(src + i, dst + i);
            for (; i < alignedSize; i += A)
                Reorder16bit<align, false>(src + i, dst + i);
            if (i < size)
                Reorder16bit<align, true>(src + i, dst + i, tailMask);
        }

        void Reorder16bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Reorder16bit<true>(src, size, dst);
            else
                Reorder16bit<false>(src, size, dst);
        }

        const __m512i K8_SHUFFLE_REORDER_32 = SIMD_MM512_SETR_EPI8(
            0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4, 0xB, 0xA, 0x9, 0x8, 0xF, 0xE, 0xD, 0xC,
            0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4, 0xB, 0xA, 0x9, 0x8, 0xF, 0xE, 0xD, 0xC,
            0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4, 0xB, 0xA, 0x9, 0x8, 0xF, 0xE, 0xD, 0xC,
            0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4, 0xB, 0xA, 0x9, 0x8, 0xF, 0xE, 0xD, 0xC);

        template <bool align, bool mask> SIMD_INLINE void Reorder32bit(const uint8_t * src, uint8_t * dst, __mmask64 tail = -1)
        {
            Store<align, mask>(dst, _mm512_shuffle_epi8((Load<align, mask>(src, tail)), K8_SHUFFLE_REORDER_32), tail);
        }

        template <bool align> SIMD_INLINE void Reorder32bit4(const uint8_t * src, uint8_t * dst)
        {
            Store<align>(dst + 0 * A, _mm512_shuffle_epi8(Load<align>(src + 0 * A), K8_SHUFFLE_REORDER_32));
            Store<align>(dst + 1 * A, _mm512_shuffle_epi8(Load<align>(src + 1 * A), K8_SHUFFLE_REORDER_32));
            Store<align>(dst + 2 * A, _mm512_shuffle_epi8(Load<align>(src + 2 * A), K8_SHUFFLE_REORDER_32));
            Store<align>(dst + 3 * A, _mm512_shuffle_epi8(Load<align>(src + 3 * A), K8_SHUFFLE_REORDER_32));
        }

        template <bool align> void Reorder32bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size % 4 == 0);

            size_t alignedSize = AlignLo(size, A);
            size_t fullAlignedSize = AlignLo(size, QA);
            __mmask64 tailMask = TailMask64(size - alignedSize);
            size_t i = 0;
            for (; i < fullAlignedSize; i += QA)
                Reorder32bit4<align>(src + i, dst + i);
            for (; i < alignedSize; i += A)
                Reorder32bit<align, false>(src + i, dst + i);
            if (i < size)
                Reorder32bit<align, true>(src + i, dst + i, tailMask);
        }

        void Reorder32bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Reorder32bit<true>(src, size, dst);
            else
                Reorder32bit<false>(src, size, dst);
        }

        const __m512i K8_SHUFFLE_REORDER_64 = SIMD_MM512_SETR_EPI8(
            0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0, 0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8,
            0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0, 0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8,
            0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0, 0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8,
            0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0, 0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8);

        template <bool align, bool mask> SIMD_INLINE void Reorder64bit(const uint8_t * src, uint8_t * dst, __mmask64 tail = -1)
        {
            Store<align, mask>(dst, _mm512_shuffle_epi8((Load<align, mask>(src, tail)), K8_SHUFFLE_REORDER_64), tail);
        }

        template <bool align> SIMD_INLINE void Reorder64bit4(const uint8_t * src, uint8_t * dst)
        {
            Store<align>(dst + 0 * A, _mm512_shuffle_epi8(Load<align>(src + 0 * A), K8_SHUFFLE_REORDER_64));
            Store<align>(dst + 1 * A, _mm512_shuffle_epi8(Load<align>(src + 1 * A), K8_SHUFFLE_REORDER_64));
            Store<align>(dst + 2 * A, _mm512_shuffle_epi8(Load<align>(src + 2 * A), K8_SHUFFLE_REORDER_64));
            Store<align>(dst + 3 * A, _mm512_shuffle_epi8(Load<align>(src + 3 * A), K8_SHUFFLE_REORDER_64));
        }

        template <bool align> void Reorder64bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size % 8 == 0);

            size_t alignedSize = AlignLo(size, A);
            size_t fullAlignedSize = AlignLo(size, QA);
            __mmask64 tailMask = TailMask64(size - alignedSize);
            size_t i = 0;
            for (; i < fullAlignedSize; i += QA)
                Reorder64bit4<align>(src + i, dst + i);
            for (; i < alignedSize; i += A)
                Reorder64bit<align, false>(src + i, dst + i);
            if (i < size)
                Reorder64bit<align, true>(src + i, dst + i, tailMask);
        }

        void Reorder64bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Reorder64bit<true>(src, size, dst);
            else
                Reorder64bit<false>(src, size, dst);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
