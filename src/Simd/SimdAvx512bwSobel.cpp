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
#include "Simd/SimdMemory.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdLoadBlock.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        const __m512i K64_PERMUTE_0 = SIMD_MM512_SETR_EPI64(0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB);
        const __m512i K64_PERMUTE_1 = SIMD_MM512_SETR_EPI64(0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF);

        template<bool abs> SIMD_INLINE void SobelDx(__m512i a[3][3], __m512i & lo, __m512i & hi)
        {
            lo = ConditionalAbs<abs>(BinomialSum16(SubUnpackedU8<0>(a[0][2], a[0][0]), SubUnpackedU8<0>(a[1][2], a[1][0]), SubUnpackedU8<0>(a[2][2], a[2][0])));
            hi = ConditionalAbs<abs>(BinomialSum16(SubUnpackedU8<1>(a[0][2], a[0][0]), SubUnpackedU8<1>(a[1][2], a[1][0]), SubUnpackedU8<1>(a[2][2], a[2][0])));
        }

        template<bool align, bool abs> SIMD_INLINE void SobelDx(__m512i a[3][3], int16_t * dst)
        {
            __m512i lo, hi;
            SobelDx<abs>(a, lo, hi);
            Store<align>(dst + 00, _mm512_permutex2var_epi64(lo, K64_PERMUTE_0, hi));
            Store<align>(dst + HA, _mm512_permutex2var_epi64(lo, K64_PERMUTE_1, hi));
        }

        template <bool align, bool abs> void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(dst) && Aligned(dstStride, HA));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            __m512i a[3][3];

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                LoadNoseDx(src0 + 0, a[0]);
                LoadNoseDx(src1 + 0, a[1]);
                LoadNoseDx(src2 + 0, a[2]);
                SobelDx<align, abs>(a, dst + 0);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBodyDx(src0 + col, a[0]);
                    LoadBodyDx(src1 + col, a[1]);
                    LoadBodyDx(src2 + col, a[2]);
                    SobelDx<align, abs>(a, dst + col);
                }
                LoadTailDx(src0 + width - A, a[0]);
                LoadTailDx(src1 + width - A, a[1]);
                LoadTailDx(src2 + width - A, a[2]);
                SobelDx<false, abs>(a, dst + width - A);

                dst += dstStride;
            }
        }

        void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(dst) && Aligned(dstStride))
                SobelDx<true, false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                SobelDx<false, false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        void SobelDxAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(dst) && Aligned(dstStride))
                SobelDx<true, true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                SobelDx<false, true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE void SobelDxAbsSum(__m512i a[3][3], __m512i * sums)
        {
            __m512i lo, hi;
            SobelDx<true>(a, lo, hi);
            sums[0] = _mm512_add_epi32(sums[0], _mm512_madd_epi16(lo, K16_0001));
            sums[1] = _mm512_add_epi32(sums[1], _mm512_madd_epi16(hi, K16_0001));
        }

        SIMD_INLINE void SetMask3(__m512i a[3], __m512i mask)
        {
            a[0] = _mm512_and_si512(a[0], mask);
            a[1] = _mm512_and_si512(a[1], mask);
            a[2] = _mm512_and_si512(a[2], mask);
        }

        SIMD_INLINE void SetMask3x3(__m512i a[3][3], __m512i mask)
        {
            SetMask3(a[0], mask);
            SetMask3(a[1], mask);
            SetMask3(a[2], mask);
        }

        template <bool align> void SobelDxAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width > A && width < 256 * 256 * F);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;

            __m512i a[3][3];
            __m512i tailMask = _mm512_mask_set1_epi8(K_INV_ZERO, TailMask64(A - width + bodyWidth), 0);

            size_t blockSize = (256 * 256 * F) / width;
            size_t blockCount = height / blockSize + 1;
            __m512i _sum = _mm512_setzero_si512();
            for (size_t block = 0; block < blockCount; ++block)
            {
                __m512i sums[2] = { _mm512_setzero_si512(), _mm512_setzero_si512() };
                for (size_t row = block*blockSize, endRow = Simd::Min(row + blockSize, height); row < endRow; ++row)
                {
                    src0 = src + stride*(row - 1);
                    src1 = src0 + stride;
                    src2 = src1 + stride;
                    if (row == 0)
                        src0 = src1;
                    if (row == height - 1)
                        src2 = src1;

                    LoadNoseDx(src0 + 0, a[0]);
                    LoadNoseDx(src1 + 0, a[1]);
                    LoadNoseDx(src2 + 0, a[2]);
                    SobelDxAbsSum(a, sums);
                    for (size_t col = A; col < bodyWidth; col += A)
                    {
                        LoadBodyDx(src0 + col, a[0]);
                        LoadBodyDx(src1 + col, a[1]);
                        LoadBodyDx(src2 + col, a[2]);
                        SobelDxAbsSum(a, sums);
                    }
                    LoadTailDx(src0 + width - A, a[0]);
                    LoadTailDx(src1 + width - A, a[1]);
                    LoadTailDx(src2 + width - A, a[2]);
                    SetMask3x3(a, tailMask);
                    SobelDxAbsSum(a, sums);
                }
                sums[0] = _mm512_add_epi32(sums[0], sums[1]);
                _sum = _mm512_add_epi64(_sum, _mm512_add_epi64(_mm512_unpacklo_epi32(sums[0], K_ZERO), _mm512_unpackhi_epi32(sums[0], K_ZERO)));
            }

            *sum = ExtractSum<uint64_t>(_sum);
        }

        void SobelDxAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(src) && Aligned(stride))
                SobelDxAbsSum<true>(src, stride, width, height, sum);
            else
                SobelDxAbsSum<false>(src, stride, width, height, sum);
        }

        template<bool abs> SIMD_INLINE void SobelDy(__m512i a[3][3], __m512i & lo, __m512i & hi)
        {
            lo = ConditionalAbs<abs>(BinomialSum16(SubUnpackedU8<0>(a[2][0], a[0][0]), SubUnpackedU8<0>(a[2][1], a[0][1]), SubUnpackedU8<0>(a[2][2], a[0][2])));
            hi = ConditionalAbs<abs>(BinomialSum16(SubUnpackedU8<1>(a[2][0], a[0][0]), SubUnpackedU8<1>(a[2][1], a[0][1]), SubUnpackedU8<1>(a[2][2], a[0][2])));
        }

        template<bool align, bool abs> SIMD_INLINE void SobelDy(__m512i a[3][3], int16_t * dst)
        {
            __m512i lo, hi;
            SobelDy<abs>(a, lo, hi);
            Store<align>(dst + 00, _mm512_permutex2var_epi64(lo, K64_PERMUTE_0, hi));
            Store<align>(dst + HA, _mm512_permutex2var_epi64(lo, K64_PERMUTE_1, hi));
        }

        template <bool align, bool abs> void SobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(dst) && Aligned(dstStride, HA));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            __m512i a[3][3];

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                SobelDy<align, abs>(a, dst + 0);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    SobelDy<align, abs>(a, dst + col);
                }
                LoadTail3<false, 1>(src0 + width - A, a[0]);
                LoadTail3<false, 1>(src2 + width - A, a[2]);
                SobelDy<false, abs>(a, dst + width - A);

                dst += dstStride;
            }
        }

        void SobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                SobelDy<true, false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                SobelDy<false, false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        void SobelDyAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                SobelDy<true, true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                SobelDy<false, true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE void SobelDyAbsSum(__m512i a[3][3], __m512i * sums)
        {
            __m512i lo, hi;
            SobelDy<true>(a, lo, hi);
            sums[0] = _mm512_add_epi32(sums[0], _mm512_madd_epi16(lo, K16_0001));
            sums[1] = _mm512_add_epi32(sums[1], _mm512_madd_epi16(hi, K16_0001));
        }

        template <bool align> void SobelDyAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width > A);
            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;

            __m512i a[3][3];
            __m512i tailMask = _mm512_mask_set1_epi8(K_INV_ZERO, TailMask64(A - width + bodyWidth), 0);

            size_t blockSize = (256 * 256 * F) / width;
            size_t blockCount = height / blockSize + 1;
            __m512i _sum = _mm512_setzero_si512();
            for (size_t block = 0; block < blockCount; ++block)
            {
                __m512i sums[2] = { _mm512_setzero_si512(), _mm512_setzero_si512() };
                for (size_t row = block*blockSize, endRow = Simd::Min(row + blockSize, height); row < endRow; ++row)
                {
                    src0 = src + stride*(row - 1);
                    src1 = src0 + stride;
                    src2 = src1 + stride;
                    if (row == 0)
                        src0 = src1;
                    if (row == height - 1)
                        src2 = src1;

                    LoadNose3<align, 1>(src0 + 0, a[0]);
                    LoadNose3<align, 1>(src2 + 0, a[2]);
                    SobelDyAbsSum(a, sums);
                    for (size_t col = A; col < bodyWidth; col += A)
                    {
                        LoadBody3<align, 1>(src0 + col, a[0]);
                        LoadBody3<align, 1>(src2 + col, a[2]);
                        SobelDyAbsSum(a, sums);
                    }
                    LoadTail3<false, 1>(src0 + width - A, a[0]);
                    LoadTail3<false, 1>(src2 + width - A, a[2]);
                    SetMask3x3(a, tailMask);
                    SobelDyAbsSum(a, sums);
                }
                sums[0] = _mm512_add_epi32(sums[0], sums[1]);
                _sum = _mm512_add_epi64(_sum, _mm512_add_epi64(_mm512_unpacklo_epi32(sums[0], K_ZERO), _mm512_unpackhi_epi32(sums[0], K_ZERO)));
            }

            *sum = ExtractSum<uint64_t>(_sum);
        }

        void SobelDyAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(src) && Aligned(stride))
                SobelDyAbsSum<true>(src, stride, width, height, sum);
            else
                SobelDyAbsSum<false>(src, stride, width, height, sum);
        }

        SIMD_INLINE __m512i ContourMetrics(__m512i dx, __m512i dy)
        {
            return _mm512_add_epi16(_mm512_slli_epi16(_mm512_add_epi16(dx, dy), 1), _mm512_maskz_set1_epi16(_mm512_cmpgt_epi16_mask(dy, dx), 1));
        }

        SIMD_INLINE void ContourMetrics(__m512i a[3][3], __m512i & lo, __m512i & hi)
        {
            __m512i dxLo, dxHi, dyLo, dyHi;
            SobelDx<true>(a, dxLo, dxHi);
            SobelDy<true>(a, dyLo, dyHi);
            lo = ContourMetrics(dxLo, dyLo);
            hi = ContourMetrics(dxHi, dyHi);
        }

        template<bool align> SIMD_INLINE void ContourMetrics(__m512i a[3][3], int16_t * dst)
        {
            __m512i lo, hi;
            ContourMetrics(a, lo, hi);
            Store<align>(dst + 00, _mm512_permutex2var_epi64(lo, K64_PERMUTE_0, hi));
            Store<align>(dst + HA, _mm512_permutex2var_epi64(lo, K64_PERMUTE_1, hi));
        }

        template <bool align> void ContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(dst) && Aligned(dstStride, HA));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            __m512i a[3][3];

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src1 + 0, a[1]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                ContourMetrics<align>(a, dst + 0);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src1 + col, a[1]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    ContourMetrics<align>(a, dst + col);
                }
                LoadTail3<false, 1>(src0 + width - A, a[0]);
                LoadTail3<false, 1>(src1 + width - A, a[1]);
                LoadTail3<false, 1>(src2 + width - A, a[2]);
                ContourMetrics<false>(a, dst + width - A);

                dst += dstStride;
            }
        }

        void ContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                ContourMetrics<true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                ContourMetrics<false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        template<bool align> SIMD_INLINE void ContourMetricsMasked(__m512i a[3][3], const uint8_t * mask, const __m512i & indexMin, int16_t * dst)
        {
            __m512i m = _mm512_maskz_set1_epi8(_mm512_cmpge_epu8_mask(Load<align>((__m256i*)mask), indexMin), -1);
            __m512i lo, hi;
            ContourMetrics(a, lo, hi);
            lo = _mm512_and_si512(lo, _mm512_unpacklo_epi8(m, m));
            hi = _mm512_and_si512(hi, _mm512_unpackhi_epi8(m, m));
            Store<align>(dst + 00, _mm512_permutex2var_epi64(lo, K64_PERMUTE_0, hi));
            Store<align>(dst + HA, _mm512_permutex2var_epi64(lo, K64_PERMUTE_1, hi));
        }

        template <bool align> void ContourMetricsMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t indexMin, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride, HA) && Aligned(mask) && Aligned(maskStride));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            __m512i _indexMin = _mm512_set1_epi8(indexMin);
            __m512i a[3][3];

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src1 + 0, a[1]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                ContourMetricsMasked<align>(a, mask + 0, _indexMin, dst + 0);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src1 + col, a[1]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    ContourMetricsMasked<align>(a, mask + col, _indexMin, dst + col);
                }
                LoadTail3<false, 1>(src0 + width - A, a[0]);
                LoadTail3<false, 1>(src1 + width - A, a[1]);
                LoadTail3<false, 1>(src2 + width - A, a[2]);
                ContourMetricsMasked<false>(a, mask + width - A, _indexMin, dst + width - A);

                dst += dstStride;
                mask += maskStride;
            }
        }

        void ContourMetricsMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t indexMin, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride) && Aligned(mask) && Aligned(maskStride))
                ContourMetricsMasked<true>(src, srcStride, width, height, mask, maskStride, indexMin, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                ContourMetricsMasked<false>(src, srcStride, width, height, mask, maskStride, indexMin, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        template<bool align> SIMD_INLINE __mmask32 AnchorComponent(const int16_t * src, size_t step, const __m512i & current, const __m512i & threshold)
        {
            __m512i last = _mm512_srli_epi16(Load<align>(src - step), 1);
            __m512i next = _mm512_srli_epi16(Load<align>(src + step), 1);
            return _mm512_cmpge_epi16_mask(_mm512_sub_epi16(current, last), threshold) & _mm512_cmpge_epi16_mask(_mm512_sub_epi16(current, next), threshold);
        }

        template<bool align> SIMD_INLINE __mmask32 Anchor(const int16_t * src, size_t stride, const __m512i & threshold)
        {
            __m512i _src = Load<align>(src);
            __m512i magnitude = _mm512_srli_epi16(_src, 1);
            __mmask32 direction = _mm512_cmpeq_epi16_mask(_mm512_and_si512(_src, K16_0001), K16_0001);
            __mmask32 vertical = AnchorComponent<false>(src, 1, magnitude, threshold) & direction;
            __mmask32 horizontal = AnchorComponent<align>(src, stride, magnitude, threshold) & (~direction);
            return _mm512_cmpneq_epi16_mask(magnitude, K_ZERO) & (vertical | horizontal);
        }

        template<bool align> SIMD_INLINE void Anchor(const int16_t * src, size_t stride, const __m512i & threshold, uint8_t * dst)
        {
            __mmask32 lo = Anchor<align>(src + 00, stride, threshold);
            __mmask32 hi = Anchor<align>(src + HA, stride, threshold);
            Store<align>(dst, _mm512_maskz_set1_epi8(__mmask64(lo) | (__mmask64(hi) << 32), -1));
        }

        template <bool align> void ContourAnchors(const int16_t * src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride, HA) && Aligned(dst) && Aligned(dstStride));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            __m512i _threshold = _mm512_set1_epi16(threshold);
            memset(dst, 0, width);
            memset(dst + dstStride*(height - 1), 0, width);
            src += srcStride;
            dst += dstStride;
            for (size_t row = 1; row < height - 1; row += step)
            {
                dst[0] = 0;
                Anchor<false>(src + 1, srcStride, _threshold, dst + 1);
                for (size_t col = A; col < bodyWidth; col += A)
                    Anchor<align>(src + col, srcStride, _threshold, dst + col);
                Anchor<false>(src + width - A - 1, srcStride, _threshold, dst + width - A - 1);
                dst[width - 1] = 0;
                src += step*srcStride;
                dst += step*dstStride;
            }
        }

        void ContourAnchors(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t * dst, size_t dstStride)
        {
            assert(srcStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                ContourAnchors<true>((const int16_t *)src, srcStride / sizeof(int16_t), width, height, step, threshold, dst, dstStride);
            else
                ContourAnchors<false>((const int16_t *)src, srcStride / sizeof(int16_t), width, height, step, threshold, dst, dstStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
