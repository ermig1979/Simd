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
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        SIMD_INLINE uint16x8_t Square(const uint8x8_t & value)
        {
            return vmull_u8(value, value);
        }

        SIMD_INLINE uint32x4_t SquaredDifferenceSum(const uint8x16_t & a, const uint8x16_t & b)
        {
            uint8x16_t ad = vabdq_u8(a, b);
            uint16x8_t lo = Square(vget_low_u8(ad));
            uint16x8_t hi = Square(vget_high_u8(ad));
            return vaddq_u32(vpaddlq_u16(lo), vpaddlq_u16(hi));
        }

        SIMD_INLINE uint32x4_t SquaredDifferenceSumMasked(const uint8x16_t & a, const uint8x16_t & b, const uint8x16_t & mask)
        {
            uint8x16_t ad = vandq_u8(vabdq_u8(a, b), mask);
            uint16x8_t lo = Square(vget_low_u8(ad));
            uint16x8_t hi = Square(vget_high_u8(ad));
            return vaddq_u32(vpaddlq_u16(lo), vpaddlq_u16(hi));
        }

        template <bool align> void SquaredDifferenceSum(
            const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, uint64_t * sum)
        {
            assert(width < 0x10000);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);

            uint64x2_t _sum = K64_0000000000000000;
            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowSum = K32_00000000;
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    uint8x16_t _a = Load<align>(a + col);
                    uint8x16_t _b = Load<align>(b + col);
                    rowSum = vaddq_u32(rowSum, SquaredDifferenceSum(_a, _b));
                }
                if (width - alignedWidth)
                {
                    uint8x16_t _a = Load<align>(a + width - A);
                    uint8x16_t _b = Load<align>(b + width - A);
                    rowSum = vaddq_u32(rowSum, SquaredDifferenceSumMasked(_a, _b, tailMask));
                }
                _sum = vaddq_u64(_sum, vpaddlq_u32(rowSum));
                a += aStride;
                b += bStride;
            }
            *sum = ExtractSum64u(_sum);
        }

        void SquaredDifferenceSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
                SquaredDifferenceSum<true>(a, aStride, b, bStride, width, height, sum);
            else
                SquaredDifferenceSum<false>(a, aStride, b, bStride, width, height, sum);
        }

        template <bool align> void SquaredDifferenceSumMasked(
            const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            assert(width < 0x10000);
            if (align)
            {
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));
                assert(Aligned(mask) && Aligned(maskStride));
            }

            size_t alignedWidth = Simd::AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            uint8x16_t _index = vdupq_n_u8(index);
            uint64x2_t _sum = K64_0000000000000000;

            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowSum = K32_00000000;
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    uint8x16_t _mask = vceqq_u8(Load<align>(mask + col), _index);
                    uint8x16_t _a = Load<align>(a + col);
                    uint8x16_t _b = Load<align>(b + col);
                    rowSum = vaddq_u32(rowSum, SquaredDifferenceSumMasked(_a, _b, _mask));
                }
                if (width - alignedWidth)
                {
                    uint8x16_t _mask = vandq_u8(tailMask, vceqq_u8(Load<align>(mask + width - A), _index));
                    uint8x16_t _a = Load<align>(a + width - A);
                    uint8x16_t _b = Load<align>(b + width - A);
                    rowSum = vaddq_u32(rowSum, SquaredDifferenceSumMasked(_a, _b, _mask));
                }
                _sum = vaddq_u64(_sum, vpaddlq_u32(rowSum));
                a += aStride;
                b += bStride;
                mask += maskStride;
            }
            *sum = ExtractSum64u(_sum);
        }

        void SquaredDifferenceSumMasked(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(mask) && Aligned(maskStride))
                SquaredDifferenceSumMasked<true>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
            else
                SquaredDifferenceSumMasked<false>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
        }

        template <bool align> SIMD_INLINE void SquaredDifferenceSum32f(const float * a, const float * b, size_t offset, float32x4_t & sum)
        {
            float32x4_t _a = Load<align>(a + offset);
            float32x4_t _b = Load<align>(b + offset);
            float32x4_t _d = vsubq_f32(_a, _b);
            sum = vmlaq_f32(sum, _d, _d);
        }

        template <bool align> SIMD_INLINE void SquaredDifferenceSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            if (align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t alignedSize = AlignLo(size, 4);
            size_t i = 0;
            if (alignedSize)
            {
                float32x4_t sums = vdupq_n_f32(0);
                for (; i < alignedSize; i += 4)
                    SquaredDifferenceSum32f<align>(a, b, i, sums);
                *sum += ExtractSum32f(sums);
            }
            for (; i < size; ++i)
                *sum += Simd::Square(a[i] - b[i]);
        }

        void SquaredDifferenceSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            if (Aligned(a) && Aligned(b))
                SquaredDifferenceSum32f<true>(a, b, size, sum);
            else
                SquaredDifferenceSum32f<false>(a, b, size, sum);
        }

        template <bool align> SIMD_INLINE void SquaredDifferenceKahanSum32f(const float * a, const float * b, size_t offset, float32x4_t & sum, float32x4_t & correction)
        {
            float32x4_t _a = Load<align>(a + offset);
            float32x4_t _b = Load<align>(b + offset);
            float32x4_t _d = vsubq_f32(_a, _b);
            float32x4_t term = vmlaq_f32(correction, _d, _d);
            float32x4_t temp = vaddq_f32(sum, term);
            correction = vsubq_f32(vmulq_f32(temp, sum), term);
            sum = temp;
        }

        template <bool align> SIMD_INLINE void SquaredDifferenceKahanSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            if (align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t alignedSize = AlignLo(size, 4);
            size_t i = 0;
            if (alignedSize)
            {
                float32x4_t sums = vdupq_n_f32(0);
                float32x4_t corrections = vdupq_n_f32(0);
                for (; i < alignedSize; i += 4)
                    SquaredDifferenceKahanSum32f<align>(a, b, i, sums, corrections);
                *sum += ExtractSum32f(sums);
            }
            for (; i < size; ++i)
                *sum += Simd::Square(a[i] - b[i]);
        }

        void SquaredDifferenceKahanSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            if (Aligned(a) && Aligned(b))
                SquaredDifferenceKahanSum32f<true>(a, b, size, sum);
            else
                SquaredDifferenceKahanSum32f<false>(a, b, size, sum);
        }
    }
#endif// SIMD_NEON_ENABLE
}
