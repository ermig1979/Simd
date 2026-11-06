/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdExtract.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <int bits> SIMD_INLINE void Sum(__m512i & sum, const __m512i & value);

        template <> SIMD_INLINE void Sum<32>(__m512i & sum, const __m512i & value)
        {
            sum = _mm512_add_epi32(sum, value);
        }

        template <> SIMD_INLINE void Sum<64>(__m512i & sum, const __m512i & value)
        {
            sum = _mm512_add_epi64(sum, value);
        }

        template <int bits> void AbsDifferenceSum4(const uint8_t * a, const uint8_t * b, __m512i * sums)
        {
            Sum<bits>(sums[0], _mm512_sad_epu8(_mm512_loadu_si512(a + 0 * A), _mm512_loadu_si512(b + 0 * A)));
            Sum<bits>(sums[1], _mm512_sad_epu8(_mm512_loadu_si512(a + 1 * A), _mm512_loadu_si512(b + 1 * A)));
            Sum<bits>(sums[0], _mm512_sad_epu8(_mm512_loadu_si512(a + 2 * A), _mm512_loadu_si512(b + 2 * A)));
            Sum<bits>(sums[1], _mm512_sad_epu8(_mm512_loadu_si512(a + 3 * A), _mm512_loadu_si512(b + 3 * A)));
        }

        template <int bits, bool mask> void AbsDifferenceSum1(const uint8_t * a, const uint8_t * b, __m512i * sums, __mmask64 tail = -1)
        {
            const __m512i a0 = mask ? _mm512_maskz_loadu_epi8(tail, a) : _mm512_loadu_si512(a);
            const __m512i b0 = mask ? _mm512_maskz_loadu_epi8(tail, b) : _mm512_loadu_si512(b);
            Sum<bits>(sums[0], _mm512_sad_epu8(a0, b0));
        }

        template <int bits> void AbsDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, size_t width, size_t height, uint64_t * sum)
        {
            size_t fullAlignedWidth = AlignLo(width, QA);
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            __m512i sums[2] = { _mm512_setzero_si512(), _mm512_setzero_si512() };
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += QA)
                    AbsDifferenceSum4<bits>(a + col, b + col, sums);
                for (; col < alignedWidth; col += A)
                    AbsDifferenceSum1<bits, false>(a + col, b + col, sums);
                if (col < width)
                    AbsDifferenceSum1<bits, true>(a + col, b + col, sums, tailMask);
                a += aStride;
                b += bStride;
            }
            *sum = ExtractSum<uint64_t>(_mm512_add_epi64(sums[0], sums[1]));
        }

        void AbsDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            size_t width, size_t height, uint64_t * sum)
        {
            if (width*height >= 256 * 256 * 256 * 8)
                AbsDifferenceSum<64>(a, aStride, b, bStride, width, height, sum);
            else
                AbsDifferenceSum<32>(a, aStride, b, bStride, width, height, sum);
        }

        template <int bits> void AbsDifferenceSumMasked(const uint8_t * a, const uint8_t * b, const uint8_t * m, const __m512i & index, __m512i * sums)
        {
            __mmask64 m0 = _mm512_cmpeq_epu8_mask(_mm512_loadu_si512(m), index);
            __m512i a0 = _mm512_maskz_loadu_epi8(m0, a);
            __m512i b0 = _mm512_maskz_loadu_epi8(m0, b);
            Sum<bits>(sums[0], _mm512_sad_epu8(a0, b0));
        }

        template <int bits> void AbsDifferenceSumMasked4(const uint8_t * a, const uint8_t * b, const uint8_t * m, const __m512i & index, __m512i * sums)
        {
            AbsDifferenceSumMasked<bits>(a + 0 * A, b + 0 * A, m + 0 * A, index, sums + 0);
            AbsDifferenceSumMasked<bits>(a + 1 * A, b + 1 * A, m + 1 * A, index, sums + 1);
            AbsDifferenceSumMasked<bits>(a + 2 * A, b + 2 * A, m + 2 * A, index, sums + 0);
            AbsDifferenceSumMasked<bits>(a + 3 * A, b + 3 * A, m + 3 * A, index, sums + 1);
        }

        template <int bits, bool mask> void AbsDifferenceSumMasked1(const uint8_t * a, const uint8_t * b, const uint8_t * m, __m512i & index, __m512i * sums, __mmask64 mm = -1)
        {
            __mmask64 m0 = _mm512_cmpeq_epu8_mask(mask ? _mm512_maskz_loadu_epi8(mm, m) : _mm512_loadu_si512(m), index) & mm;
            const __m512i a0 = _mm512_maskz_loadu_epi8(m0, a);
            const __m512i b0 = _mm512_maskz_loadu_epi8(m0, b);
            Sum<bits>(sums[0], _mm512_sad_epu8(a0, b0));
        }

        template <int bits> void AbsDifferenceSumMasked(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            const uint8_t * mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            __m512i _index = _mm512_set1_epi8(index);
            size_t fullAlignedWidth = AlignLo(width, QA);
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            __m512i sums[2] = { _mm512_setzero_si512(), _mm512_setzero_si512() };
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += QA)
                    AbsDifferenceSumMasked4<bits>(a + col, b + col, mask + col, _index, sums);
                for (; col < alignedWidth; col += A)
                    AbsDifferenceSumMasked1<bits, false>(a + col, b + col, mask + col, _index, sums);
                if (col < width)
                    AbsDifferenceSumMasked1<bits, true>(a + col, b + col, mask + col, _index, sums, tailMask);
                a += aStride;
                b += bStride;
                mask += maskStride;
            }
            sums[0] = _mm512_add_epi64(sums[0], sums[1]);
            *sum = ExtractSum<uint64_t>(sums[0]);
        }

        void AbsDifferenceSumMasked(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            const uint8_t * mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            if (width*height >= 256 * 256 * 256 * 8)
                AbsDifferenceSumMasked<64>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
            else
                AbsDifferenceSumMasked<32>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
        }

        template <int bits, bool mask> void AbsDifferenceSums3(__m512i current, const uint8_t * background, __m512i sums[3], __mmask64 m = -1)
        {
            Sum<bits>(sums[0], _mm512_sad_epu8(current, mask ? _mm512_maskz_loadu_epi8(m, background - 1) : _mm512_loadu_si512(background - 1)));
            Sum<bits>(sums[1], _mm512_sad_epu8(current, mask ? _mm512_maskz_loadu_epi8(m, background) : _mm512_loadu_si512(background)));
            Sum<bits>(sums[2], _mm512_sad_epu8(current, mask ? _mm512_maskz_loadu_epi8(m, background + 1) : _mm512_loadu_si512(background + 1)));
        }

        template <int bits, bool mask> void AbsDifferenceSums3x3(const uint8_t * current, const uint8_t * background, size_t backgroundStride, __m512i sums[9], __mmask64 m = -1)
        {
            const __m512i _current = mask ? _mm512_maskz_loadu_epi8(m, current) : _mm512_loadu_si512(current);
            AbsDifferenceSums3<bits, mask>(_current, background - backgroundStride, sums + 0, m);
            AbsDifferenceSums3<bits, mask>(_current, background, sums + 3, m);
            AbsDifferenceSums3<bits, mask>(_current, background + backgroundStride, sums + 6, m);
        }

        template <int bits, bool mask> void AbsDifferenceSums3x3x2(const uint8_t * current0, size_t currentStride, const uint8_t * background1, size_t backgroundStride, __m512i sums[9], __mmask64 m = -1)
        {
            const __m512i current00 = mask ? _mm512_maskz_loadu_epi8(m, current0) : _mm512_loadu_si512(current0);
            const uint8_t * background0 = background1 - backgroundStride;
            const __m512i background00 = mask ? _mm512_maskz_loadu_epi8(m, background0 - 1) : _mm512_loadu_si512(background0 - 1);
            const __m512i background01 = mask ? _mm512_maskz_loadu_epi8(m, background0) : _mm512_loadu_si512(background0);
            const __m512i background02 = mask ? _mm512_maskz_loadu_epi8(m, background0 + 1) : _mm512_loadu_si512(background0 + 1);
            Sum<bits>(sums[0], _mm512_sad_epu8(current00, background00));
            Sum<bits>(sums[1], _mm512_sad_epu8(current00, background01));
            Sum<bits>(sums[2], _mm512_sad_epu8(current00, background02));
            const uint8_t * current1 = current0 + currentStride;
            const __m512i current10 = mask ? _mm512_maskz_loadu_epi8(m, current1) : _mm512_loadu_si512(current1);
            const __m512i background10 = mask ? _mm512_maskz_loadu_epi8(m, background1 - 1) : _mm512_loadu_si512(background1 - 1);
            const __m512i background11 = mask ? _mm512_maskz_loadu_epi8(m, background1) : _mm512_loadu_si512(background1);
            const __m512i background12 = mask ? _mm512_maskz_loadu_epi8(m, background1 + 1) : _mm512_loadu_si512(background1 + 1);
            Sum<bits>(sums[0], _mm512_sad_epu8(current10, background10));
            Sum<bits>(sums[1], _mm512_sad_epu8(current10, background11));
            Sum<bits>(sums[2], _mm512_sad_epu8(current10, background12));
            Sum<bits>(sums[3], _mm512_sad_epu8(current00, background10));
            Sum<bits>(sums[4], _mm512_sad_epu8(current00, background11));
            Sum<bits>(sums[5], _mm512_sad_epu8(current00, background12));
            const uint8_t * background2 = background1 + backgroundStride;
            const __m512i background20 = mask ? _mm512_maskz_loadu_epi8(m, background2 - 1) : _mm512_loadu_si512(background2 - 1);
            const __m512i background21 = mask ? _mm512_maskz_loadu_epi8(m, background2) : _mm512_loadu_si512(background2);
            const __m512i background22 = mask ? _mm512_maskz_loadu_epi8(m, background2 + 1) : _mm512_loadu_si512(background2 + 1);
            Sum<bits>(sums[3], _mm512_sad_epu8(current10, background20));
            Sum<bits>(sums[4], _mm512_sad_epu8(current10, background21));
            Sum<bits>(sums[5], _mm512_sad_epu8(current10, background22));
            Sum<bits>(sums[6], _mm512_sad_epu8(current00, background20));
            Sum<bits>(sums[7], _mm512_sad_epu8(current00, background21));
            Sum<bits>(sums[8], _mm512_sad_epu8(current00, background22));
            const uint8_t * background3 = background2 + backgroundStride;
            const __m512i background30 = mask ? _mm512_maskz_loadu_epi8(m, background3 - 1) : _mm512_loadu_si512(background3 - 1);
            const __m512i background31 = mask ? _mm512_maskz_loadu_epi8(m, background3) : _mm512_loadu_si512(background3);
            const __m512i background32 = mask ? _mm512_maskz_loadu_epi8(m, background3 + 1) : _mm512_loadu_si512(background3 + 1);
            Sum<bits>(sums[6], _mm512_sad_epu8(current10, background30));
            Sum<bits>(sums[7], _mm512_sad_epu8(current10, background31));
            Sum<bits>(sums[8], _mm512_sad_epu8(current10, background32));
        }

        template <int bits> void AbsDifferenceSums3x3(const uint8_t * current, size_t currentStride,
            const uint8_t * background, size_t backgroundStride, size_t width, size_t height, uint64_t * sums)
        {
            width -= 2;
            height -= 2;
            current += 1 + currentStride;
            background += 1 + backgroundStride;

            size_t alignedHeight = AlignLo(height, 2);
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = __mmask64(-1) >> (A + alignedWidth - width);
            __m512i _sums[9];
            for (size_t i = 0; i < 9; ++i)
                _sums[i] = _mm512_setzero_si512();

            size_t row = 0;
#if SIMD_ZMM_COUNT == 32
            for (; row < alignedHeight; row += 2)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    AbsDifferenceSums3x3x2<bits, false>(current + col, currentStride, background + col, backgroundStride, _sums);
                if (col < width)
                    AbsDifferenceSums3x3x2<bits, true>(current + col, currentStride, background + col, backgroundStride, _sums, tailMask);
                current += 2 * currentStride;
                background += 2 * backgroundStride;
            }
#endif
            for (; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    AbsDifferenceSums3x3<bits, false>(current + col, background + col, backgroundStride, _sums);
                if (col < width)
                    AbsDifferenceSums3x3<bits, true>(current + col, background + col, backgroundStride, _sums, tailMask);
                current += currentStride;
                background += backgroundStride;
            }

            for (size_t i = 0; i < 9; ++i)
                sums[i] = ExtractSum<uint64_t>(_sums[i]);
        }

        void AbsDifferenceSums3x3(const uint8_t * current, size_t currentStride, const uint8_t * background, size_t backgroundStride,
            size_t width, size_t height, uint64_t * sums)
        {
            if (width*height >= 256 * 256 * 256 * 8)
                AbsDifferenceSums3x3<64>(current, currentStride, background, backgroundStride, width, height, sums);
            else
                AbsDifferenceSums3x3<32>(current, currentStride, background, backgroundStride, width, height, sums);
        }

        template <int bits, bool mask> void AbsDifferenceSums3x3Masked(const uint8_t * current, const uint8_t * background, size_t backgroundStride, const uint8_t * m, const __m512i & index, __m512i sums[9], __mmask64 mm = -1)
        {
            __mmask64 m0 = _mm512_cmpeq_epu8_mask(mask ? _mm512_maskz_loadu_epi8(mm, m) : _mm512_loadu_si512(m), index) & mm;
            const __m512i _current = _mm512_maskz_loadu_epi8(m0, current);
            AbsDifferenceSums3<bits, true>(_current, background - backgroundStride, sums + 0, m0);
            AbsDifferenceSums3<bits, true>(_current, background, sums + 3, m0);
            AbsDifferenceSums3<bits, true>(_current, background + backgroundStride, sums + 6, m0);
        }

        template <int bits, bool mask> void AbsDifferenceSums3x3x2(const uint8_t * current0, size_t currentStride, const uint8_t * background1, size_t backgroundStride,
            const uint8_t * mask0, size_t maskStride, const __m512i & index, __m512i sums[9], __mmask64 mm = -1)
        {
            __mmask64 m0 = mm & _mm512_cmpeq_epu8_mask(mask ? _mm512_maskz_loadu_epi8(mm, mask0) : _mm512_loadu_si512(mask0), index);
            __m512i mask00 = _mm512_maskz_set1_epi8(m0, -1);
            const __m512i current00 = _mm512_maskz_loadu_epi8(m0, current0);
            const uint8_t * background0 = background1 - backgroundStride;
            const __m512i background00 = _mm512_maskz_loadu_epi8(m0, background0 - 1);
            const __m512i background01 = _mm512_maskz_loadu_epi8(m0, background0);
            const __m512i background02 = _mm512_maskz_loadu_epi8(m0, background0 + 1);
            Sum<bits>(sums[0], _mm512_sad_epu8(current00, _mm512_and_si512(mask00, background00)));
            Sum<bits>(sums[1], _mm512_sad_epu8(current00, _mm512_and_si512(mask00, background01)));
            Sum<bits>(sums[2], _mm512_sad_epu8(current00, _mm512_and_si512(mask00, background02)));
            const uint8_t * mask1 = mask0 + maskStride;
            __mmask64 m1 = mm & _mm512_cmpeq_epu8_mask(mask ? _mm512_maskz_loadu_epi8(mm, mask1) : _mm512_loadu_si512(mask1), index);
            __m512i mask10 = _mm512_maskz_set1_epi8(m1, -1);
            const uint8_t * current1 = current0 + currentStride;
            const __m512i current10 = _mm512_maskz_loadu_epi8(m1, current1);
            const __m512i background10 = _mm512_loadu_si512(background1 - 1);
            const __m512i background11 = _mm512_loadu_si512(background1);
            const __m512i background12 = _mm512_loadu_si512(background1 + 1);
            Sum<bits>(sums[0], _mm512_sad_epu8(current10, _mm512_and_si512(mask10, background10)));
            Sum<bits>(sums[1], _mm512_sad_epu8(current10, _mm512_and_si512(mask10, background11)));
            Sum<bits>(sums[2], _mm512_sad_epu8(current10, _mm512_and_si512(mask10, background12)));
            Sum<bits>(sums[3], _mm512_sad_epu8(current00, _mm512_and_si512(mask00, background10)));
            Sum<bits>(sums[4], _mm512_sad_epu8(current00, _mm512_and_si512(mask00, background11)));
            Sum<bits>(sums[5], _mm512_sad_epu8(current00, _mm512_and_si512(mask00, background12)));
            const uint8_t * background2 = background1 + backgroundStride;
            const __m512i background20 = _mm512_loadu_si512(background2 - 1);
            const __m512i background21 = _mm512_loadu_si512(background2);
            const __m512i background22 = _mm512_loadu_si512(background2 + 1);
            Sum<bits>(sums[3], _mm512_sad_epu8(current10, _mm512_and_si512(mask10, background20)));
            Sum<bits>(sums[4], _mm512_sad_epu8(current10, _mm512_and_si512(mask10, background21)));
            Sum<bits>(sums[5], _mm512_sad_epu8(current10, _mm512_and_si512(mask10, background22)));
            Sum<bits>(sums[6], _mm512_sad_epu8(current00, _mm512_and_si512(mask00, background20)));
            Sum<bits>(sums[7], _mm512_sad_epu8(current00, _mm512_and_si512(mask00, background21)));
            Sum<bits>(sums[8], _mm512_sad_epu8(current00, _mm512_and_si512(mask00, background22)));
            const uint8_t * background3 = background2 + backgroundStride;
            const __m512i background30 = _mm512_maskz_loadu_epi8(m1, background3 - 1);
            const __m512i background31 = _mm512_maskz_loadu_epi8(m1, background3);
            const __m512i background32 = _mm512_maskz_loadu_epi8(m1, background3 + 1);
            Sum<bits>(sums[6], _mm512_sad_epu8(current10, background30));
            Sum<bits>(sums[7], _mm512_sad_epu8(current10, background31));
            Sum<bits>(sums[8], _mm512_sad_epu8(current10, background32));
        }

        template <int bits> void AbsDifferenceSums3x3Masked(const uint8_t * current, size_t currentStride, const uint8_t * background, size_t backgroundStride,
            const uint8_t * mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sums)
        {
            width -= 2;
            height -= 2;
            current += 1 + currentStride;
            background += 1 + backgroundStride;
            mask += 1 + maskStride;

            __m512i _index = _mm512_set1_epi8(index);
            size_t alignedHeight = AlignLo(height, 2);
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = __mmask64(-1) >> (A + alignedWidth - width);
            __m512i _sums[9];
            for (size_t i = 0; i < 9; ++i)
                _sums[i] = _mm512_setzero_si512();

            size_t row = 0;
#if SIMD_ZMM_COUNT == 32
            for (; row < alignedHeight; row += 2)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    AbsDifferenceSums3x3x2<bits, false>(current + col, currentStride, background + col, backgroundStride, mask + col, maskStride, _index, _sums);
                if (col < width)
                    AbsDifferenceSums3x3x2<bits, true>(current + col, currentStride, background + col, backgroundStride, mask + col, maskStride, _index, _sums, tailMask);
                current += 2 * currentStride;
                background += 2 * backgroundStride;
                mask += 2 * maskStride;
            }
#endif
            for (; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    AbsDifferenceSums3x3Masked<bits, false>(current + col, background + col, backgroundStride, mask + col, _index, _sums);
                if (col < width)
                    AbsDifferenceSums3x3Masked<bits, true>(current + col, background + col, backgroundStride, mask + col, _index, _sums, tailMask);
                current += currentStride;
                background += backgroundStride;
                mask += maskStride;
            }

            for (size_t i = 0; i < 9; ++i)
                sums[i] = ExtractSum<uint64_t>(_sums[i]);
        }

        void AbsDifferenceSums3x3Masked(const uint8_t * current, size_t currentStride, const uint8_t * background, size_t backgroundStride,
            const uint8_t * mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sums)
        {
            if (width*height >= 256 * 256 * 256 * 8)
                AbsDifferenceSums3x3Masked<64>(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
            else
                AbsDifferenceSums3x3Masked<32>(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
