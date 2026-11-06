/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdUnpack.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template<bool abs> SIMD_INLINE void SobelDx(__m256i a[3][3], __m256i & lo, __m256i & hi)
        {
            lo = ConditionalAbs<abs>(BinomialSum16(SubUnpackedU8<0>(a[0][2], a[0][0]), SubUnpackedU8<0>(a[1][2], a[1][0]), SubUnpackedU8<0>(a[2][2], a[2][0])));
            hi = ConditionalAbs<abs>(BinomialSum16(SubUnpackedU8<1>(a[0][2], a[0][0]), SubUnpackedU8<1>(a[1][2], a[1][0]), SubUnpackedU8<1>(a[2][2], a[2][0])));
        }

        template<bool abs> SIMD_INLINE void SobelDx(__m256i a[3][3], int16_t * dst)
        {
            __m256i lo, hi;
            SobelDx<abs>(a, lo, hi);
            _mm256_storeu_si256((__m256i*)dst + 0, _mm256_permute2x128_si256(lo, hi, 0x20));
            _mm256_storeu_si256((__m256i*)dst + 1, _mm256_permute2x128_si256(lo, hi, 0x31));
        }

        template <bool abs> void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            __m256i a[3][3];

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
                SobelDx<abs>(a, dst + 0);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBodyDx(src0 + col, a[0]);
                    LoadBodyDx(src1 + col, a[1]);
                    LoadBodyDx(src2 + col, a[2]);
                    SobelDx<abs>(a, dst + col);
                }
                LoadTailDx(src0 + width - A, a[0]);
                LoadTailDx(src1 + width - A, a[1]);
                LoadTailDx(src2 + width - A, a[2]);
                SobelDx<abs>(a, dst + width - A);

                dst += dstStride;
            }
        }

        void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            SobelDx<false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        void SobelDxAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            SobelDx<true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE void SobelDxAbsSum(__m256i a[3][3], __m256i & sum)
        {
            __m256i lo, hi;
            SobelDx<true>(a, lo, hi);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(lo, K16_0001));
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(hi, K16_0001));
        }

        SIMD_INLINE void SetMask3(__m256i a[3], __m256i mask)
        {
            a[0] = _mm256_and_si256(a[0], mask);
            a[1] = _mm256_and_si256(a[1], mask);
            a[2] = _mm256_and_si256(a[2], mask);
        }

        SIMD_INLINE void SetMask3x3(__m256i a[3][3], __m256i mask)
        {
            SetMask3(a[0], mask);
            SetMask3(a[1], mask);
            SetMask3(a[2], mask);
        }

        void SobelDxAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width > A);

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;

            __m256i a[3][3];
            __m256i tailMask = SetMask<uint8_t>(0, A - width + bodyWidth, 0xFF);
            __m256i fullSum = _mm256_setzero_si256();

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + stride*(row - 1);
                src1 = src0 + stride;
                src2 = src1 + stride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                __m256i rowSum = _mm256_setzero_si256();

                LoadNoseDx(src0 + 0, a[0]);
                LoadNoseDx(src1 + 0, a[1]);
                LoadNoseDx(src2 + 0, a[2]);
                SobelDxAbsSum(a, rowSum);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBodyDx(src0 + col, a[0]);
                    LoadBodyDx(src1 + col, a[1]);
                    LoadBodyDx(src2 + col, a[2]);
                    SobelDxAbsSum(a, rowSum);
                }
                LoadTailDx(src0 + width - A, a[0]);
                LoadTailDx(src1 + width - A, a[1]);
                LoadTailDx(src2 + width - A, a[2]);
                SetMask3x3(a, tailMask);
                SobelDxAbsSum(a, rowSum);

                fullSum = _mm256_add_epi64(fullSum, HorizontalSum32(rowSum));
            }
            *sum = ExtractSum<uint64_t>(fullSum);
        }

        template<bool abs> SIMD_INLINE void SobelDy(__m256i a[3][3], __m256i & lo, __m256i & hi)
        {
            lo = ConditionalAbs<abs>(BinomialSum16(SubUnpackedU8<0>(a[2][0], a[0][0]), SubUnpackedU8<0>(a[2][1], a[0][1]), SubUnpackedU8<0>(a[2][2], a[0][2])));
            hi = ConditionalAbs<abs>(BinomialSum16(SubUnpackedU8<1>(a[2][0], a[0][0]), SubUnpackedU8<1>(a[2][1], a[0][1]), SubUnpackedU8<1>(a[2][2], a[0][2])));
        }

        template<bool abs> SIMD_INLINE void SobelDy(__m256i a[3][3], int16_t * dst)
        {
            __m256i lo, hi;
            SobelDy<abs>(a, lo, hi);
            _mm256_storeu_si256((__m256i*)dst + 0, _mm256_permute2x128_si256(lo, hi, 0x20));
            _mm256_storeu_si256((__m256i*)dst + 1, _mm256_permute2x128_si256(lo, hi, 0x31));
        }

        template <bool abs> void SobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            __m256i a[3][3];

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                LoadNose3<1>(src0 + 0, a[0]);
                LoadNose3<1>(src2 + 0, a[2]);
                SobelDy<abs>(a, dst + 0);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<1>(src0 + col, a[0]);
                    LoadBody3<1>(src2 + col, a[2]);
                    SobelDy<abs>(a, dst + col);
                }
                LoadTail3<1>(src0 + width - A, a[0]);
                LoadTail3<1>(src2 + width - A, a[2]);
                SobelDy<abs>(a, dst + width - A);

                dst += dstStride;
            }
        }

        void SobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            SobelDy<false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        void SobelDyAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            SobelDy<true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE void SobelDyAbsSum(__m256i a[3][3], __m256i & sum)
        {
            __m256i lo, hi;
            SobelDy<true>(a, lo, hi);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(lo, K16_0001));
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(hi, K16_0001));
        }

        void SobelDyAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width > A);

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;

            __m256i a[3][3];
            __m256i tailMask = SetMask<uint8_t>(0, A - width + bodyWidth, 0xFF);
            __m256i fullSum = _mm256_setzero_si256();

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + stride*(row - 1);
                src1 = src0 + stride;
                src2 = src1 + stride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                __m256i rowSum = _mm256_setzero_si256();

                LoadNose3<1>(src0 + 0, a[0]);
                LoadNose3<1>(src2 + 0, a[2]);
                SobelDyAbsSum(a, rowSum);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<1>(src0 + col, a[0]);
                    LoadBody3<1>(src2 + col, a[2]);
                    SobelDyAbsSum(a, rowSum);
                }
                LoadTail3<1>(src0 + width - A, a[0]);
                LoadTail3<1>(src2 + width - A, a[2]);
                SetMask3x3(a, tailMask);
                SobelDyAbsSum(a, rowSum);

                fullSum = _mm256_add_epi64(fullSum, HorizontalSum32(rowSum));
            }
            *sum = ExtractSum<uint64_t>(fullSum);
        }

        SIMD_INLINE __m256i ContourMetrics(__m256i dx, __m256i dy)
        {
            return _mm256_add_epi16(_mm256_slli_epi16(_mm256_add_epi16(dx, dy), 1), _mm256_and_si256(_mm256_cmpgt_epi16(dy, dx), K16_0001));
        }

        SIMD_INLINE void ContourMetrics(__m256i a[3][3], __m256i & lo, __m256i & hi)
        {
            __m256i dxLo, dxHi, dyLo, dyHi;
            SobelDx<true>(a, dxLo, dxHi);
            SobelDy<true>(a, dyLo, dyHi);
            lo = ContourMetrics(dxLo, dyLo);
            hi = ContourMetrics(dxHi, dyHi);
        }

        SIMD_INLINE void ContourMetrics(__m256i a[3][3], int16_t * dst)
        {
            __m256i lo, hi;
            ContourMetrics(a, lo, hi);
            _mm256_storeu_si256((__m256i*)dst + 0, _mm256_permute2x128_si256(lo, hi, 0x20));
            _mm256_storeu_si256((__m256i*)dst + 1, _mm256_permute2x128_si256(lo, hi, 0x31));
        }

        void ContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            __m256i a[3][3];

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                LoadNose3<1>(src0 + 0, a[0]);
                LoadNose3<1>(src1 + 0, a[1]);
                LoadNose3<1>(src2 + 0, a[2]);
                ContourMetrics(a, dst + 0);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<1>(src0 + col, a[0]);
                    LoadBody3<1>(src1 + col, a[1]);
                    LoadBody3<1>(src2 + col, a[2]);
                    ContourMetrics(a, dst + col);
                }
                LoadTail3<1>(src0 + width - A, a[0]);
                LoadTail3<1>(src1 + width - A, a[1]);
                LoadTail3<1>(src2 + width - A, a[2]);
                ContourMetrics(a, dst + width - A);

                dst += dstStride;
            }
        }

        void ContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            ContourMetrics(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE void ContourMetricsMasked(__m256i a[3][3], const uint8_t * mask, const __m256i & indexMin, int16_t * dst)
        {
            __m256i m = GreaterOrEqual8u(_mm256_loadu_si256((__m256i*)mask), indexMin);
            __m256i lo, hi;
            ContourMetrics(a, lo, hi);
            lo = _mm256_and_si256(lo, _mm256_unpacklo_epi8(m, m));
            hi = _mm256_and_si256(hi, _mm256_unpackhi_epi8(m, m));
            _mm256_storeu_si256((__m256i*)dst + 0, _mm256_permute2x128_si256(lo, hi, 0x20));
            _mm256_storeu_si256((__m256i*)dst + 1, _mm256_permute2x128_si256(lo, hi, 0x31));
        }

        void ContourMetricsMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t indexMin, int16_t * dst, size_t dstStride)
        {
            assert(width > A);

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            __m256i _indexMin = _mm256_set1_epi8(indexMin);
            __m256i a[3][3];

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                LoadNose3<1>(src0 + 0, a[0]);
                LoadNose3<1>(src1 + 0, a[1]);
                LoadNose3<1>(src2 + 0, a[2]);
                ContourMetricsMasked(a, mask + 0, _indexMin, dst + 0);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<1>(src0 + col, a[0]);
                    LoadBody3<1>(src1 + col, a[1]);
                    LoadBody3<1>(src2 + col, a[2]);
                    ContourMetricsMasked(a, mask + col, _indexMin, dst + col);
                }
                LoadTail3<1>(src0 + width - A, a[0]);
                LoadTail3<1>(src1 + width - A, a[1]);
                LoadTail3<1>(src2 + width - A, a[2]);
                ContourMetricsMasked(a, mask + width - A, _indexMin, dst + width - A);

                dst += dstStride;
                mask += maskStride;
            }
        }

        void ContourMetricsMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t indexMin, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            ContourMetricsMasked(src, srcStride, width, height, mask, maskStride, indexMin, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE __m256i AnchorComponent(const int16_t * src, size_t step, const __m256i & current, const __m256i & threshold, const __m256i & mask)
        {
            __m256i last = _mm256_srli_epi16(_mm256_loadu_si256((__m256i*)(src - step)), 1);
            __m256i next = _mm256_srli_epi16(_mm256_loadu_si256((__m256i*)(src + step)), 1);
            return _mm256_andnot_si256(_mm256_or_si256(_mm256_cmpgt_epi16(threshold, _mm256_sub_epi16(current, last)),
                _mm256_cmpgt_epi16(threshold, _mm256_sub_epi16(current, next))), mask);
        }

        SIMD_INLINE __m256i Anchor(const int16_t * src, size_t stride, const __m256i & threshold)
        {
            __m256i _src = _mm256_loadu_si256((__m256i*)src);
            __m256i direction = _mm256_and_si256(_src, K16_0001);
            __m256i magnitude = _mm256_srli_epi16(_src, 1);
            __m256i vertical = AnchorComponent(src, 1, magnitude, threshold, _mm256_cmpeq_epi16(direction, K16_0001));
            __m256i horizontal = AnchorComponent(src, stride, magnitude, threshold, _mm256_cmpeq_epi16(direction, K_ZERO));
            return _mm256_andnot_si256(_mm256_cmpeq_epi16(magnitude, K_ZERO), _mm256_and_si256(_mm256_or_si256(vertical, horizontal), K16_00FF));
        }

        SIMD_INLINE void Anchor(const int16_t * src, size_t stride, const __m256i & threshold, uint8_t * dst)
        {
            __m256i lo = Anchor(src, stride, threshold);
            __m256i hi = Anchor(src + HA, stride, threshold);
            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(lo, hi));
        }

        void ContourAnchors(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t * dst, size_t dstStride)
        {
            assert(srcStride % sizeof(int16_t) == 0);

            const int16_t * src16 = (const int16_t *)src;
            size_t srcStride16 = srcStride / sizeof(int16_t);

            assert(width > A);

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            __m256i _threshold = _mm256_set1_epi16(threshold);
            memset(dst, 0, width);
            memset(dst + dstStride*(height - 1), 0, width);
            src16 += srcStride16;
            dst += dstStride;
            for (size_t row = 1; row < height - 1; row += step)
            {
                dst[0] = 0;
                Anchor(src16 + 1, srcStride16, _threshold, dst + 1);
                for (size_t col = A; col < bodyWidth; col += A)
                    Anchor(src16 + col, srcStride16, _threshold, dst + col);
                Anchor(src16 + width - A - 1, srcStride16, _threshold, dst + width - A - 1);
                dst[width - 1] = 0;
                src16 += step*srcStride16;
                dst += step*dstStride;
            }
        }
    }
#endif// SIMD_AVX2_ENABLE
}
