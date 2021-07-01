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
#include "Simd/SimdLoadBlock.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template<bool abs, int part> SIMD_INLINE int16x8_t SobelDx(uint8x16_t a[3][3])
        {
            return ConditionalAbs<abs>(BinomialSum(Sub<part>(a[0][2], a[0][0]), Sub<part>(a[1][2], a[1][0]), Sub<part>(a[2][2], a[2][0])));
        }

        template<bool align, bool abs> SIMD_INLINE void SobelDx(uint8x16_t a[3][3], int16_t * dst)
        {
            Store<align>(dst, SobelDx<abs, 0>(a));
            Store<align>(dst + HA, SobelDx<abs, 1>(a));
        }

        template <bool align, bool abs> void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(dst) && Aligned(dstStride, HA));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            uint8x16_t a[3][3];

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

        SIMD_INLINE void SobelDxAbsSum(uint8x16_t a[3][3], uint32x4_t & sum)
        {
            sum = vaddq_u32(sum, vpaddlq_u16((uint16x8_t)vaddq_s16(SobelDx<true, 0>(a), SobelDx<true, 1>(a))));
        }

        SIMD_INLINE void SetMask3(uint8x16_t a[3], uint8x16_t mask)
        {
            a[0] = vandq_u8(a[0], mask);
            a[1] = vandq_u8(a[1], mask);
            a[2] = vandq_u8(a[2], mask);
        }

        SIMD_INLINE void SetMask3x3(uint8x16_t a[3][3], uint8x16_t mask)
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

            uint8x16_t a[3][3];
            uint64x2_t fullSum = K64_0000000000000000;
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + bodyWidth);

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + stride*(row - 1);
                src1 = src0 + stride;
                src2 = src1 + stride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                uint32x4_t rowSum = K32_00000000;

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

                fullSum = vaddq_u64(fullSum, vpaddlq_u32(rowSum));
            }
            *sum = ExtractSum64u(fullSum);
        }

        template<bool abs, int part> SIMD_INLINE int16x8_t SobelDy(uint8x16_t a[3][3])
        {
            return ConditionalAbs<abs>(BinomialSum(Sub<part>(a[2][0], a[0][0]), Sub<part>(a[2][1], a[0][1]), Sub<part>(a[2][2], a[0][2])));
        }

        template<bool align, bool abs> SIMD_INLINE void SobelDy(uint8x16_t a[3][3], int16_t * dst)
        {
            Store<align>(dst, SobelDy<abs, 0>(a));
            Store<align>(dst + HA, SobelDy<abs, 1>(a));
        }

        template <bool align, bool abs> void SobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(dst) && Aligned(dstStride, HA));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            uint8x16_t a[3][3];

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

        SIMD_INLINE void SobelDyAbsSum(uint8x16_t a[3][3], uint32x4_t & sum)
        {
            sum = vaddq_u32(sum, vpaddlq_u16((uint16x8_t)vaddq_s16(SobelDy<true, 0>(a), SobelDy<true, 1>(a))));
        }

        template <bool align> void SobelDyAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width > A);

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;

            uint8x16_t a[3][3];
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + bodyWidth);
            uint64x2_t fullSum = K64_0000000000000000;

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + stride*(row - 1);
                src1 = src0 + stride;
                src2 = src1 + stride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                uint32x4_t rowSum = K32_00000000;

                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                SobelDyAbsSum(a, rowSum);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    SobelDyAbsSum(a, rowSum);
                }
                LoadTail3<false, 1>(src0 + width - A, a[0]);
                LoadTail3<false, 1>(src2 + width - A, a[2]);
                SetMask3x3(a, tailMask);
                SobelDyAbsSum(a, rowSum);

                fullSum = vaddq_u64(fullSum, vpaddlq_u32(rowSum));
            }
            *sum = ExtractSum64u(fullSum);
        }

        void SobelDyAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(src) && Aligned(stride))
                SobelDyAbsSum<true>(src, stride, width, height, sum);
            else
                SobelDyAbsSum<false>(src, stride, width, height, sum);
        }

        template<int part> SIMD_INLINE int16x8_t ContourMetrics(uint8x16_t a[3][3])
        {
            int16x8_t dx = SobelDx<true, part>(a);
            int16x8_t dy = SobelDy<true, part>(a);
            return vaddq_s16(vshlq_n_s16(vaddq_s16(dx, dy), 1), (int16x8_t)vandq_u16(vcltq_s16(dx, dy), K16_0001));
        }

        template<bool align> SIMD_INLINE void ContourMetrics(uint8x16_t a[3][3], int16_t * dst)
        {
            Store<align>(dst, ContourMetrics<0>(a));
            Store<align>(dst + HA, ContourMetrics<1>(a));
        }

        template <bool align> void ContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride, HA));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            uint8x16_t a[3][3];

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

        template<bool align> SIMD_INLINE void ContourMetricsMasked(uint8x16_t a[3][3], const uint8_t * mask, const uint8x16_t & indexMin, int16_t * dst)
        {
            uint8x16_t _mask = vcgeq_u8(Load<align>(mask), indexMin);
            Store<align>(dst, vandq_s16(ContourMetrics<0>(a), (int16x8_t)Stretch2<0>(_mask)));
            Store<align>(dst + HA, vandq_s16(ContourMetrics<1>(a), (int16x8_t)Stretch2<1>(_mask)));
        }

        template <bool align> void ContourMetricsMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t indexMin, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride, HA) && Aligned(mask) && Aligned(maskStride));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            uint8x16_t _indexMin = vdupq_n_u8(indexMin);
            uint8x16_t a[3][3];

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

        template<bool align> SIMD_INLINE uint16x8_t AnchorComponent(const int16_t * src, size_t step, const int16x8_t & value, const uint16x8_t & mask)
        {
            int16x8_t last = (int16x8_t)vshrq_n_u16((uint16x8_t)Load<align>(src - step), 1);
            int16x8_t next = (int16x8_t)vshrq_n_u16((uint16x8_t)Load<align>(src + step), 1);
            return vandq_u16(vandq_u16(vcgeq_s16(value, last), vcgeq_s16(value, next)), mask);
        }

        template<bool align> SIMD_INLINE uint16x8_t Anchor(const int16_t * src, size_t stride, const int16x8_t & threshold)
        {
            int16x8_t _src = Load<align>(src);
            uint16x8_t direction = vandq_u16((uint16x8_t)_src, K16_0001);
            int16x8_t magnitude = (int16x8_t)vshrq_n_u16((uint16x8_t)_src, 1);
            int16x8_t value = vsubq_s16(magnitude, threshold);
            uint16x8_t vertical = AnchorComponent<false>(src, 1, value, vceqq_u16(direction, K16_0001));
            uint16x8_t horizontal = AnchorComponent<align>(src, stride, value, vceqq_u16(direction, K16_0000));
            return vandq_u16(vcgtq_u16((uint16x8_t)magnitude, K16_0000), vandq_u16(vorrq_u16(vertical, horizontal), K16_00FF));
        }

        template<bool align> SIMD_INLINE void Anchor(const int16_t * src, size_t stride, const int16x8_t & threshold, uint8_t * dst)
        {
            uint16x8_t lo = Anchor<align>(src, stride, threshold);
            uint16x8_t hi = Anchor<align>(src + HA, stride, threshold);
            Store<align>(dst, PackU16(lo, hi));
        }

        template <bool align> void ContourAnchors(const int16_t * src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride, HA) && Aligned(dst) && Aligned(dstStride));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            int16x8_t _threshold = vdupq_n_s16(threshold);
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
#endif// SIMD_NEON_ENABLE
}
