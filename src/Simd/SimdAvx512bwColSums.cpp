/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t)*width + sizeof(uint32_t)*width);
                    sums16 = (uint16_t*)_p;
                    sums32 = (uint32_t*)(sums16 + width);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t * sums16;
                uint32_t * sums32;
            private:
                void *_p;
            };
        }

        SIMD_INLINE void AddColSum16(const uint8_t * src, __m512i& lo, __m512i& hi, __mmask64 tail = -1)
        {
            __m512i _src = _mm512_maskz_loadu_epi8(tail, src);
            lo = _mm512_add_epi16(lo, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(_src, 0)));
            hi = _mm512_add_epi16(hi, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(_src, 1)));
        }

        template<bool align, bool masked> SIMD_INLINE void GetColSum16x1(const uint8_t * src, uint16_t * dst, __mmask64 tail = -1)
        {
            __m512i lo = Load<true>(dst + 00);
            __m512i hi = Load<true>(dst + HA);
            AddColSum16(src, lo, hi, tail);
            Store<true>(dst + 00, lo);
            Store<true>(dst + HA, hi);
        }

        template<bool align, bool masked> SIMD_INLINE void GetColSum16x4(const uint8_t * src, size_t stride, uint16_t * dst, __mmask64 tail = -1)
        {
            __m512i lo = Load<true>(dst + 00);
            __m512i hi = Load<true>(dst + HA);
            AddColSum16(src + 0 * stride, lo, hi, tail);
            AddColSum16(src + 1 * stride, lo, hi, tail);
            AddColSum16(src + 2 * stride, lo, hi, tail);
            AddColSum16(src + 3 * stride, lo, hi, tail);
            Store<true>(dst + 00, lo);
            Store<true>(dst + HA, hi);
        }

        template<bool align, bool masked> SIMD_INLINE void GetColSum16x8(const uint8_t * src, size_t stride, uint16_t * dst, __mmask64 tail = -1)
        {
            __m512i lo = Load<true>(dst + 00);
            __m512i hi = Load<true>(dst + HA);
            AddColSum16(src + 0 * stride, lo, hi, tail);
            AddColSum16(src + 1 * stride, lo, hi, tail);
            AddColSum16(src + 2 * stride, lo, hi, tail);
            AddColSum16(src + 3 * stride, lo, hi, tail);
            AddColSum16(src + 4 * stride, lo, hi, tail);
            AddColSum16(src + 5 * stride, lo, hi, tail);
            AddColSum16(src + 6 * stride, lo, hi, tail);
            AddColSum16(src + 7 * stride, lo, hi, tail);
            Store<true>(dst + 00, lo);
            Store<true>(dst + HA, hi);
        }

        SIMD_INLINE void Sum16To32(const uint16_t * src, uint32_t * dst)
        {
            __m512i _src = _mm512_loadu_si512((__m512i*)src);
            Store<true>(dst + 0 * F, _mm512_add_epi32(Load<true>(dst + 0 * F), _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(_src, 0))));
            Store<true>(dst + 1 * F, _mm512_add_epi32(Load<true>(dst + 1 * F), _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(_src, 1))));
        }

        template <bool align> void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedLoWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedLoWidth);
            size_t alignedHiWidth = AlignHi(width, A);
            size_t stepSize = 256;
            size_t stepCount = DivHi(height, 256);

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint32_t)*alignedHiWidth);
            for (size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);
                size_t rowEnd4 = AlignLo(rowEnd, 4);
                size_t rowEnd8 = AlignLo(rowEnd, 8);

                memset(buffer.sums16, 0, sizeof(uint16_t)*alignedHiWidth);
                size_t row = rowStart;
                for (; row < rowEnd8; row += 8)
                {
                    size_t col = 0;
                    for (; col < alignedLoWidth; col += A)
                        GetColSum16x8<align, false>(src + col, stride, buffer.sums16 + col);
                    if (col < width)
                        GetColSum16x8<align, true>(src + col, stride, buffer.sums16 + col, tailMask);
                    src += 8 * stride;
                }
                for (; row < rowEnd4; row += 4)
                {
                    size_t col = 0;
                    for (; col < alignedLoWidth; col += A)
                        GetColSum16x4<align, false>(src + col, stride, buffer.sums16 + col);
                    if (col < width)
                        GetColSum16x4<align, true>(src + col, stride, buffer.sums16 + col, tailMask);
                    src += 4 * stride;
                }
                for (; row < rowEnd; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedLoWidth; col += A)
                        GetColSum16x1<align, false>(src + col, buffer.sums16 + col);
                    if (col < width)
                        GetColSum16x1<align, true>(src + col, buffer.sums16 + col, tailMask);
                    src += stride;
                }
                for (size_t col = 0; col < alignedHiWidth; col += HA)
                    Sum16To32(buffer.sums16 + col, buffer.sums32 + col);
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t)*width);
        }

        void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetColSums<true>(src, stride, width, height, sums);
            else
                GetColSums<false>(src, stride, width, height, sums);
        }

        //-----------------------------------------------------------------------------------------

        const __m512i K32_PERMUTE_FOR_COL_SUMS = SIMD_MM512_SETR_EPI32(0x0, 0x8, 0x4, 0xC, 0x1, 0x9, 0x5, 0xD, 0x2, 0xA, 0x6, 0xE, 0x3, 0xB, 0x7, 0xF);

        template<bool align, bool masked> SIMD_INLINE void GetAbsDxColSum16(const uint8_t * src, uint16_t * dst, __mmask64 tail = -1)
        {
            __m512i src0 = Load<align, masked>(src + 0, tail);
            __m512i src1 = Load<false, masked>(src + 1, tail);
            __m512i absDiff = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_COL_SUMS, AbsDifferenceU8(src0, src1));
            Store<true>(dst + 00, _mm512_add_epi16(Load<true>(dst + 00), _mm512_unpacklo_epi8(absDiff, K_ZERO)));
            Store<true>(dst + HA, _mm512_add_epi16(Load<true>(dst + HA), _mm512_unpackhi_epi8(absDiff, K_ZERO)));
        }

        template <bool align> void GetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            width--;
            size_t alignedLoWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedLoWidth);
            size_t alignedHiWidth = AlignHi(width, A);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX) / stepSize;

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint32_t)*alignedHiWidth);
            for (size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);
                memset(buffer.sums16, 0, sizeof(uint16_t)*alignedHiWidth);
                for (size_t row = rowStart; row < rowEnd; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedLoWidth; col += A)
                        GetAbsDxColSum16<align, false>(src + col, buffer.sums16 + col);
                    if (col < width)
                        GetAbsDxColSum16<align, true>(src + col, buffer.sums16 + col, tailMask);
                    src += stride;
                }
                for (size_t col = 0; col < alignedHiWidth; col += A)
                    Sum16To32(buffer.sums16 + col, buffer.sums32 + col);
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t)*width);
            sums[width] = 0;
        }

        void GetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetAbsDxColSums<true>(src, stride, width, height, sums);
            else
                GetAbsDxColSums<false>(src, stride, width, height, sums);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
