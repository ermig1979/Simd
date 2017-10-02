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
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE __m512i SquaredDifference(const __m512i & a, const __m512i & b)
        {
            const __m512i lo = SubUnpackedU8<0>(a, b);
            const __m512i hi = SubUnpackedU8<1>(a, b);
            return _mm512_add_epi32(_mm512_madd_epi16(lo, lo), _mm512_madd_epi16(hi, hi));
        }

        template <bool align, bool masked> SIMD_INLINE void SquaredDifferenceSum(const uint8_t * a, const uint8_t * b, __m512i * sums, __mmask64 tail = -1)
        {
            const __m512i _a = Load<align, masked>(a, tail);
            const __m512i _b = Load<align, masked>(b, tail);
            sums[0] = _mm512_add_epi32(sums[0], SquaredDifference(_a, _b));
        }

        template <bool align> SIMD_INLINE void SquaredDifferenceSum4(const uint8_t * a, const uint8_t * b, __m512i * sums)
        {
            sums[0] = _mm512_add_epi32(sums[0], SquaredDifference(Load<align>(a + A * 0), Load<align>(b + A * 0)));
            sums[1] = _mm512_add_epi32(sums[1], SquaredDifference(Load<align>(a + A * 1), Load<align>(b + A * 1)));
            sums[2] = _mm512_add_epi32(sums[2], SquaredDifference(Load<align>(a + A * 2), Load<align>(b + A * 2)));
            sums[3] = _mm512_add_epi32(sums[3], SquaredDifference(Load<align>(a + A * 3), Load<align>(b + A * 3)));
        }

        template <bool align> void SquaredDifferenceSum(
            const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            size_t width, size_t height, uint64_t * sum)
        {
            assert(width < 256 * 256 * F);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            size_t fullAlignedWidth = Simd::AlignLo(width, QA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);

            size_t blockSize = (256 * 256 * F) / width;
            size_t blockCount = height / blockSize + 1;
            __m512i _sum = _mm512_setzero_si512();
            for (size_t block = 0; block < blockCount; ++block)
            {
                __m512i sums[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512() };
                for (size_t row = block*blockSize, endRow = Simd::Min(row + blockSize, height); row < endRow; ++row)
                {
                    size_t col = 0;
                    for (; col < fullAlignedWidth; col += QA)
                        SquaredDifferenceSum4<align>(a + col, b + col, sums);
                    for (; col < alignedWidth; col += A)
                        SquaredDifferenceSum<align, false>(a + col, b + col, sums);
                    if (col < width)
                        SquaredDifferenceSum<align, true>(a + col, b + col, sums, tailMask);
                    a += aStride;
                    b += bStride;
                }
                sums[0] = _mm512_add_epi32(_mm512_add_epi32(sums[0], sums[1]), _mm512_add_epi32(sums[2], sums[3]));
                _sum = _mm512_add_epi64(_sum, _mm512_add_epi64(_mm512_unpacklo_epi32(sums[0], K_ZERO), _mm512_unpackhi_epi32(sums[0], K_ZERO)));
            }
            *sum = ExtractSum<uint64_t>(_sum);
        }

        void SquaredDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
                SquaredDifferenceSum<true>(a, aStride, b, bStride, width, height, sum);
            else
                SquaredDifferenceSum<false>(a, aStride, b, bStride, width, height, sum);
        }

        template <bool align, bool masked> SIMD_INLINE void SquaredDifferenceSumMasked(const uint8_t * a, const uint8_t * b, const uint8_t * m, const __m512i & index, __m512i * sums, __mmask64 tail)
        {
            const __mmask64 mask = _mm512_cmpeq_epi8_mask((Load<align, masked>(m, tail)), index) & tail;
            const __m512i _a = Load<align, true>(a, mask);
            const __m512i _b = Load<align, true>(b, mask);
            sums[0] = _mm512_add_epi32(sums[0], SquaredDifference(_a, _b));
        }

        template <bool align, int idx> SIMD_INLINE void SquaredDifferenceSumMasked(const uint8_t * a, const uint8_t * b, const uint8_t * m, const __m512i & index, __m512i * sums)
        {
            const __mmask64 mask = _mm512_cmpeq_epi8_mask((Load<align>(m + A * idx)), index);
            const __m512i _a = Load<align, true>(a + A * idx, mask);
            const __m512i _b = Load<align, true>(b + A * idx, mask);
            sums[idx] = _mm512_add_epi32(sums[idx], SquaredDifference(_b, _a));
        }

        template <bool align> void SquaredDifferenceSumMasked(
            const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            const uint8_t * mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            assert(width < 256 * 256 * F);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(mask) && Aligned(maskStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            size_t fullAlignedWidth = Simd::AlignLo(width, QA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            __m512i _index = _mm512_set1_epi8(index);
            size_t blockSize = (256 * 256 * F) / width;
            size_t blockCount = height / blockSize + 1;
            __m512i _sum = _mm512_setzero_si512();
            for (size_t block = 0; block < blockCount; ++block)
            {
                __m512i sums[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512() };
                for (size_t row = block*blockSize, endRow = Simd::Min(row + blockSize, height); row < endRow; ++row)
                {
                    size_t col = 0;
                    for (; col < fullAlignedWidth; col += QA)
                    {

                        SquaredDifferenceSumMasked<align, 0>(a + col, b + col, mask + col, _index, sums);
                        SquaredDifferenceSumMasked<align, 1>(a + col, b + col, mask + col, _index, sums);
                        SquaredDifferenceSumMasked<align, 2>(a + col, b + col, mask + col, _index, sums);
                        SquaredDifferenceSumMasked<align, 3>(a + col, b + col, mask + col, _index, sums);
                    }
                    for (; col < alignedWidth; col += A)
                        SquaredDifferenceSumMasked<align, false>(a + col, b + col, mask + col, _index, sums, -1);
                    if (col < width)
                        SquaredDifferenceSumMasked<align, true>(a + col, b + col, mask + col, _index, sums, tailMask);
                    a += aStride;
                    b += bStride;
                    mask += maskStride;
                }
                sums[0] = _mm512_add_epi32(_mm512_add_epi32(sums[0], sums[1]), _mm512_add_epi32(sums[2], sums[3]));
                _sum = _mm512_add_epi64(_sum, _mm512_add_epi64(_mm512_unpacklo_epi32(sums[0], K_ZERO), _mm512_unpackhi_epi32(sums[0], K_ZERO)));
            }
            *sum = ExtractSum<uint64_t>(_sum);
        }

        void SquaredDifferenceSumMasked(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(mask) && Aligned(maskStride))
                SquaredDifferenceSumMasked<true>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
            else
                SquaredDifferenceSumMasked<false>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
