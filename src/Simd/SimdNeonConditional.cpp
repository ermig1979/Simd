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
#include "Simd/SimdExtract.h"
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align, SimdCompareType compareType>
        void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, uint32_t * count)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t blockSize = A << 8;
            size_t blockCount = (alignedWidth >> 8) + 1;

            uint8x16_t _value = vdupq_n_u8(value);

            uint32x4_t _count = K32_00000000;
            for (size_t row = 0; row < height; ++row)
            {
                uint16x8_t rowSum = K16_0000;
                for (size_t block = 0; block < blockCount; ++block)
                {
                    uint8x16_t blockSum = K8_00;
                    for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
                    {
                        const uint8x16_t mask = Compare8u<compareType>(Load<align>(src + col), _value);
                        blockSum = vaddq_u8(blockSum, vandq_u8(mask, K8_01));
                    }
                    rowSum = vaddq_u16(rowSum, vpaddlq_u8(blockSum));
                }
                if (alignedWidth != width)
                {
                    const uint8x16_t mask = vandq_u8(Compare8u<compareType>(Load<false>(src + width - A), _value), tailMask);
                    rowSum = vaddq_u16(rowSum, vpaddlq_u8(vandq_u8(mask, K8_01)));
                }
                _count = vaddq_u32(_count, vpaddlq_u16(rowSum));
                src += stride;
            }
            *count = ExtractSum32u(_count);
        }

        template <SimdCompareType compareType>
        void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, uint32_t * count)
        {
            if (Aligned(src) && Aligned(stride))
                ConditionalCount8u<true, compareType>(src, stride, width, height, value, count);
            else
                ConditionalCount8u<false, compareType>(src, stride, width, height, value, count);
        }

        void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height,
            uint8_t value, SimdCompareType compareType, uint32_t * count)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalCount8u<SimdCompareEqual>(src, stride, width, height, value, count);
            case SimdCompareNotEqual:
                return ConditionalCount8u<SimdCompareNotEqual>(src, stride, width, height, value, count);
            case SimdCompareGreater:
                return ConditionalCount8u<SimdCompareGreater>(src, stride, width, height, value, count);
            case SimdCompareGreaterOrEqual:
                return ConditionalCount8u<SimdCompareGreaterOrEqual>(src, stride, width, height, value, count);
            case SimdCompareLesser:
                return ConditionalCount8u<SimdCompareLesser>(src, stride, width, height, value, count);
            case SimdCompareLesserOrEqual:
                return ConditionalCount8u<SimdCompareLesserOrEqual>(src, stride, width, height, value, count);
            default:
                assert(0);
            }
        }

        template <bool align, SimdCompareType compareType>
        void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, int16_t value, uint32_t * count)
        {
            assert(width >= HA);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, HA);
            uint16x8_t tailMask = (uint16x8_t)ShiftLeft(K8_FF, 2 * (HA - width + alignedWidth));

            int16x8_t _value = vdupq_n_s16(value);

            uint32x4_t _count = K32_00000000;
            for (size_t row = 0; row < height; ++row)
            {
                const int16_t * s = (const int16_t *)src;
                uint16x8_t rowSum = K16_0000;
                for (size_t col = 0; col < alignedWidth; col += HA)
                {
                    const uint16x8_t mask = Compare16i<compareType>(Load<align>(s + col), _value);
                    rowSum = vaddq_u16(rowSum, vandq_u16(mask, K16_0001));
                }
                if (alignedWidth != width)
                {
                    const uint16x8_t mask = vandq_u16(Compare16i<compareType>(Load<false>(s + width - HA), _value), tailMask);
                    rowSum = vaddq_u16(rowSum, vandq_u16(mask, K16_0001));
                }
                _count = vaddq_u32(_count, vpaddlq_u16(rowSum));
                src += stride;
            }
            *count = ExtractSum32u(_count);
        }

        template <SimdCompareType compareType>
        void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, int16_t value, uint32_t * count)
        {
            if (Aligned(src) && Aligned(stride))
                ConditionalCount16i<true, compareType>(src, stride, width, height, value, count);
            else
                ConditionalCount16i<false, compareType>(src, stride, width, height, value, count);
        }

        void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height,
            int16_t value, SimdCompareType compareType, uint32_t * count)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalCount16i<SimdCompareEqual>(src, stride, width, height, value, count);
            case SimdCompareNotEqual:
                return ConditionalCount16i<SimdCompareNotEqual>(src, stride, width, height, value, count);
            case SimdCompareGreater:
                return ConditionalCount16i<SimdCompareGreater>(src, stride, width, height, value, count);
            case SimdCompareGreaterOrEqual:
                return ConditionalCount16i<SimdCompareGreaterOrEqual>(src, stride, width, height, value, count);
            case SimdCompareLesser:
                return ConditionalCount16i<SimdCompareLesser>(src, stride, width, height, value, count);
            case SimdCompareLesserOrEqual:
                return ConditionalCount16i<SimdCompareLesserOrEqual>(src, stride, width, height, value, count);
            default:
                assert(0);
            }
        }

        template <bool align, SimdCompareType compareType>
        void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t blockSize = A << 8;
            size_t blockCount = (alignedWidth >> 8) + 1;

            uint8x16_t _value = vdupq_n_u8(value);

            uint64x2_t _sum = K64_0000000000000000;
            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowSum = K32_00000000;
                for (size_t block = 0; block < blockCount; ++block)
                {
                    uint16x8_t blockSum = K16_0000;
                    for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
                    {
                        const uint8x16_t _src = Load<align>(src + col);
                        const uint8x16_t _mask = Compare8u<compareType>(Load<align>(mask + col), _value);
                        blockSum = vaddq_u16(blockSum, vpaddlq_u8(vandq_u8(_mask, _src)));
                    }
                    rowSum = vaddq_u32(rowSum, vpaddlq_u16(blockSum));
                }
                if (alignedWidth != width)
                {
                    const uint8x16_t _src = Load<false>(src + width - A);
                    const uint8x16_t _mask = vandq_u8(Compare8u<compareType>(Load<false>(mask + width - A), _value), tailMask);
                    rowSum = vaddq_u32(rowSum, vpaddlq_u16(vpaddlq_u8(vandq_u8(_mask, _src))));
                }
                _sum = vaddq_u64(_sum, vpaddlq_u32(rowSum));
                src += srcStride;
                mask += maskStride;
            }
            *sum = ExtractSum64u(_sum);
        }

        template <SimdCompareType compareType>
        void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride))
                ConditionalSum<true, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
            else
                ConditionalSum<false, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
        }

        void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalSum<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareNotEqual:
                return ConditionalSum<SimdCompareNotEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreater:
                return ConditionalSum<SimdCompareGreater>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreaterOrEqual:
                return ConditionalSum<SimdCompareGreaterOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesser:
                return ConditionalSum<SimdCompareLesser>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesserOrEqual:
                return ConditionalSum<SimdCompareLesserOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            default:
                assert(0);
            }
        }

        SIMD_INLINE uint16x8_t Square(uint8x8_t value)
        {
            return vmull_u8(value, value);
        }

        SIMD_INLINE uint32x4_t Square(uint8x16_t value)
        {
            uint16x8_t lo = Square(vget_low_u8(value));
            uint16x8_t hi = Square(vget_high_u8(value));
            return vaddq_u32(vpaddlq_u16(lo), vpaddlq_u16(hi));
        }

        template <bool align, SimdCompareType compareType>
        void ConditionalSquareSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);

            uint8x16_t _value = vdupq_n_u8(value);

            uint64x2_t _sum = K64_0000000000000000;
            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowSum = K32_00000000;
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const uint8x16_t _mask = Compare8u<compareType>(Load<align>(mask + col), _value);
                    const uint8x16_t _src = vandq_u8(_mask, Load<align>(src + col));
                    rowSum = vaddq_u32(rowSum, Square(_src));
                }
                if (alignedWidth != width)
                {
                    const uint8x16_t _mask = vandq_u8(Compare8u<compareType>(Load<false>(mask + width - A), _value), tailMask);
                    const uint8x16_t _src = vandq_u8(_mask, Load<false>(src + width - A));
                    rowSum = vaddq_u32(rowSum, Square(_src));
                }
                _sum = vaddq_u64(_sum, vpaddlq_u32(rowSum));
                src += srcStride;
                mask += maskStride;
            }
            *sum = ExtractSum64u(_sum);
        }

        template <SimdCompareType compareType>
        void ConditionalSquareSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride))
                ConditionalSquareSum<true, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
            else
                ConditionalSquareSum<false, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
        }

        void ConditionalSquareSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalSquareSum<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareNotEqual:
                return ConditionalSquareSum<SimdCompareNotEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreater:
                return ConditionalSquareSum<SimdCompareGreater>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreaterOrEqual:
                return ConditionalSquareSum<SimdCompareGreaterOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesser:
                return ConditionalSquareSum<SimdCompareLesser>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesserOrEqual:
                return ConditionalSquareSum<SimdCompareLesserOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            default:
                assert(0);
            }
        }

        template <bool align>
        SIMD_INLINE uint32x4_t SquaredDifference(const uint8_t * src, ptrdiff_t step, uint8x16_t mask)
        {
            const uint8x16_t a = vandq_u8(Load<align>(src - step), mask);
            const uint8x16_t b = vandq_u8(Load<align>(src + step), mask);
            return Square(vabdq_u8(a, b));
        }

        template <bool align, SimdCompareType compareType>
        void ConditionalSquareGradientSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            assert(width >= A + 2 && height >= 3);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride));

            src += srcStride;
            mask += maskStride;
            height -= 2;

            size_t alignedWidth = Simd::AlignLo(width - 1, A);
            uint8x16_t noseMask = ShiftRight(K8_FF, 1);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + 1 + alignedWidth);

            uint8x16_t _value = vdupq_n_u8(value);

            uint64x2_t _sum = K64_0000000000000000;
            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowSum = K32_00000000;
                {
                    const uint8x16_t _mask = vandq_u8(Compare8u<compareType>(Load<false>(mask + 1), _value), noseMask);
                    rowSum = vaddq_u32(rowSum, SquaredDifference<false>(src + 1, 1, _mask));
                    rowSum = vaddq_u32(rowSum, SquaredDifference<false>(src + 1, srcStride, _mask));
                }
                for (size_t col = A; col < alignedWidth; col += A)
                {
                    const uint8x16_t _mask = Compare8u<compareType>(Load<false>(mask + col), _value);
                    rowSum = vaddq_u32(rowSum, SquaredDifference<false>(src + col, 1, _mask));
                    rowSum = vaddq_u32(rowSum, SquaredDifference<align>(src + col, srcStride, _mask));
                }
                if (alignedWidth != width - 1)
                {
                    size_t offset = width - A - 1;
                    const uint8x16_t _mask = vandq_u8(Compare8u<compareType>(Load<false>(mask + offset), _value), tailMask);
                    rowSum = vaddq_u32(rowSum, SquaredDifference<false>(src + offset, 1, _mask));
                    rowSum = vaddq_u32(rowSum, SquaredDifference<false>(src + offset, srcStride, _mask));
                }
                _sum = vaddq_u64(_sum, vpaddlq_u32(rowSum));
                src += srcStride;
                mask += maskStride;
            }
            *sum = ExtractSum64u(_sum);
        }

        template <SimdCompareType compareType>
        void ConditionalSquareGradientSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride))
                ConditionalSquareGradientSum<true, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
            else
                ConditionalSquareGradientSum<false, compareType>(src, srcStride, width, height, mask, maskStride, value, sum);
        }

        void ConditionalSquareGradientSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalSquareGradientSum<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareNotEqual:
                return ConditionalSquareGradientSum<SimdCompareNotEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreater:
                return ConditionalSquareGradientSum<SimdCompareGreater>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreaterOrEqual:
                return ConditionalSquareGradientSum<SimdCompareGreaterOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesser:
                return ConditionalSquareGradientSum<SimdCompareLesser>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesserOrEqual:
                return ConditionalSquareGradientSum<SimdCompareLesserOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            default:
                assert(0);
            }
        }

        template <bool align, SimdCompareType compareType>
        SIMD_INLINE void ConditionalFill(const uint8_t * src, size_t offset, const uint8x16_t & threshold, const uint8x16_t & value, uint8_t * dst)
        {
            const uint8x16_t _src = Load<align>(src + offset);
            const uint8x16_t _dst = Load<align>(dst + offset);
            Store<align>(dst + offset, vbslq_u8(Compare8u<compareType>(_src, threshold), value, _dst));
        }

        template <bool align, SimdCompareType compareType>
        void ConditionalFill(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t threshold, uint8_t value, uint8_t * dst, size_t dstStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = Simd::AlignLo(width, A);

            uint8x16_t _value = vdupq_n_u8(value);
            uint8x16_t _threshold = vdupq_n_u8(threshold);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    ConditionalFill<align, compareType>(src, col, _threshold, _value, dst);
                if (alignedWidth != width)
                    ConditionalFill<false, compareType>(src, width - A, _threshold, _value, dst);
                src += srcStride;
                dst += dstStride;
            }
        }

        template <SimdCompareType compareType>
        void ConditionalFill(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t threshold, uint8_t value, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                ConditionalFill<true, compareType>(src, srcStride, width, height, threshold, value, dst, dstStride);
            else
                ConditionalFill<false, compareType>(src, srcStride, width, height, threshold, value, dst, dstStride);
        }

        void ConditionalFill(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t threshold, SimdCompareType compareType, uint8_t value, uint8_t * dst, size_t dstStride)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalFill<SimdCompareEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
            case SimdCompareNotEqual:
                return ConditionalFill<SimdCompareNotEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
            case SimdCompareGreater:
                return ConditionalFill<SimdCompareGreater>(src, srcStride, width, height, threshold, value, dst, dstStride);
            case SimdCompareGreaterOrEqual:
                return ConditionalFill<SimdCompareGreaterOrEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
            case SimdCompareLesser:
                return ConditionalFill<SimdCompareLesser>(src, srcStride, width, height, threshold, value, dst, dstStride);
            case SimdCompareLesserOrEqual:
                return ConditionalFill<SimdCompareLesserOrEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
            default:
                assert(0);
            }
        }
    }
#endif// SIMD_NEON_ENABLE
}
