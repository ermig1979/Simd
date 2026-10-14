/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdReorder.h"

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

        template <bool mask> SIMD_INLINE void Reorder16bit(const uint8_t * src, uint8_t * dst, __mmask64 tail = -1)
        {
            __m512i value = _mm512_shuffle_epi8(mask ? _mm512_maskz_loadu_epi8(tail, src) : _mm512_loadu_si512(src), K8_SHUFFLE_REORDER_16);
            if (mask)
                _mm512_mask_storeu_epi8(dst, tail, value);
            else
                _mm512_storeu_si512(dst, value);
        }

        SIMD_INLINE void Reorder16bit4(const uint8_t * src, uint8_t * dst)
        {
            _mm512_storeu_si512(dst + 0 * A, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 0 * A), K8_SHUFFLE_REORDER_16));
            _mm512_storeu_si512(dst + 1 * A, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 1 * A), K8_SHUFFLE_REORDER_16));
            _mm512_storeu_si512(dst + 2 * A, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 2 * A), K8_SHUFFLE_REORDER_16));
            _mm512_storeu_si512(dst + 3 * A, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 3 * A), K8_SHUFFLE_REORDER_16));
        }

        void Reorder16bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size % 2 == 0);

            size_t alignedSize = AlignLo(size, A);
            size_t fullAlignedSize = AlignLo(size, QA);
            __mmask64 tailMask = TailMask64(size - alignedSize);
            size_t i = 0;
            for (; i < fullAlignedSize; i += QA)
                Reorder16bit4(src + i, dst + i);
            for (; i < alignedSize; i += A)
                Reorder16bit<false>(src + i, dst + i);
            if (i < size)
                Reorder16bit<true>(src + i, dst + i, tailMask);
        }

        const __m512i K8_SHUFFLE_REORDER_32 = SIMD_MM512_SETR_EPI8(
            0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4, 0xB, 0xA, 0x9, 0x8, 0xF, 0xE, 0xD, 0xC,
            0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4, 0xB, 0xA, 0x9, 0x8, 0xF, 0xE, 0xD, 0xC,
            0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4, 0xB, 0xA, 0x9, 0x8, 0xF, 0xE, 0xD, 0xC,
            0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4, 0xB, 0xA, 0x9, 0x8, 0xF, 0xE, 0xD, 0xC);

        template <bool mask> SIMD_INLINE void Reorder32bit(const uint8_t * src, uint8_t * dst, __mmask64 tail = -1)
        {
            __m512i value = _mm512_shuffle_epi8(mask ? _mm512_maskz_loadu_epi8(tail, src) : _mm512_loadu_si512(src), K8_SHUFFLE_REORDER_32);
            if (mask)
                _mm512_mask_storeu_epi8(dst, tail, value);
            else
                _mm512_storeu_si512(dst, value);
        }

        SIMD_INLINE void Reorder32bit4(const uint8_t * src, uint8_t * dst)
        {
            _mm512_storeu_si512(dst + 0 * A, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 0 * A), K8_SHUFFLE_REORDER_32));
            _mm512_storeu_si512(dst + 1 * A, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 1 * A), K8_SHUFFLE_REORDER_32));
            _mm512_storeu_si512(dst + 2 * A, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 2 * A), K8_SHUFFLE_REORDER_32));
            _mm512_storeu_si512(dst + 3 * A, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 3 * A), K8_SHUFFLE_REORDER_32));
        }

        void Reorder32bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size % 4 == 0);

            size_t alignedSize = AlignLo(size, A);
            size_t fullAlignedSize = AlignLo(size, QA);
            __mmask64 tailMask = TailMask64(size - alignedSize);
            size_t i = 0;
            for (; i < fullAlignedSize; i += QA)
                Reorder32bit4(src + i, dst + i);
            for (; i < alignedSize; i += A)
                Reorder32bit<false>(src + i, dst + i);
            if (i < size)
                Reorder32bit<true>(src + i, dst + i, tailMask);
        }

        const __m512i K8_SHUFFLE_REORDER_64 = SIMD_MM512_SETR_EPI8(
            0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0, 0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8,
            0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0, 0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8,
            0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0, 0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8,
            0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0, 0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8);

        template <bool mask> SIMD_INLINE void Reorder64bit(const uint8_t * src, uint8_t * dst, __mmask64 tail = -1)
        {
            __m512i value = _mm512_shuffle_epi8(mask ? _mm512_maskz_loadu_epi8(tail, src) : _mm512_loadu_si512(src), K8_SHUFFLE_REORDER_64);
            if (mask)
                _mm512_mask_storeu_epi8(dst, tail, value);
            else
                _mm512_storeu_si512(dst, value);
        }

        SIMD_INLINE void Reorder64bit4(const uint8_t * src, uint8_t * dst)
        {
            _mm512_storeu_si512(dst + 0 * A, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 0 * A), K8_SHUFFLE_REORDER_64));
            _mm512_storeu_si512(dst + 1 * A, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 1 * A), K8_SHUFFLE_REORDER_64));
            _mm512_storeu_si512(dst + 2 * A, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 2 * A), K8_SHUFFLE_REORDER_64));
            _mm512_storeu_si512(dst + 3 * A, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 3 * A), K8_SHUFFLE_REORDER_64));
        }

        void Reorder64bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size % 8 == 0);

            size_t alignedSize = AlignLo(size, A);
            size_t fullAlignedSize = AlignLo(size, QA);
            __mmask64 tailMask = TailMask64(size - alignedSize);
            size_t i = 0;
            for (; i < fullAlignedSize; i += QA)
                Reorder64bit4(src + i, dst + i);
            for (; i < alignedSize; i += A)
                Reorder64bit<false>(src + i, dst + i);
            if (i < size)
                Reorder64bit<true>(src + i, dst + i, tailMask);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
