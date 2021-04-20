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
#include "Simd/SimdLoad.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> void AbsDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t blockSize = A << 7;
            size_t blockCount = (alignedWidth >> 7) + 1;

            uint64x2_t _sum = K64_0000000000000000;
            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowSum = K32_00000000;
                for (size_t block = 0; block < blockCount; ++block)
                {
                    uint16x8_t blockSum = K16_0000;
                    for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
                    {
                        const uint8x16_t ad = vabdq_u8(Load<align>(a + col), Load<align>(b + col));
                        blockSum = vaddq_u16(blockSum, vpaddlq_u8(ad));
                    }
                    rowSum = vaddq_u32(rowSum, vpaddlq_u16(blockSum));
                }
                if (alignedWidth != width)
                {
                    const uint8x16_t ad = vabdq_u8(Load<false>(a + width - A), Load<false>(b + width - A));
                    rowSum = vaddq_u32(rowSum, vpaddlq_u16(vpaddlq_u8(vandq_u8(tailMask, ad))));
                }
                _sum = vaddq_u64(_sum, vpaddlq_u32(rowSum));
                a += aStride;
                b += bStride;
            }
            *sum = ExtractSum64u(_sum);
        }

        void AbsDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
                AbsDifferenceSum<true>(a, aStride, b, bStride, width, height, sum);
            else
                AbsDifferenceSum<false>(a, aStride, b, bStride, width, height, sum);
        }

        template <bool align> void AbsDifferenceSumMasked(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));
                assert(Aligned(mask) && Aligned(maskStride));
            }

            size_t alignedWidth = Simd::AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t blockSize = A << 7;
            size_t blockCount = (alignedWidth >> 7) + 1;

            uint8x16_t _index = vdupq_n_u8(index);
            uint64x2_t _sum = K64_0000000000000000;

            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowSum = K32_00000000;
                for (size_t block = 0; block < blockCount; ++block)
                {
                    uint16x8_t blockSum = K16_0000;
                    for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
                    {
                        const uint8x16_t ad = vabdq_u8(Load<align>(a + col), Load<align>(b + col));
                        const uint8x16_t _mask = vceqq_u8(Load<align>(mask + col), _index);
                        blockSum = vaddq_u16(blockSum, vpaddlq_u8(vandq_u8(_mask, ad)));
                    }
                    rowSum = vaddq_u32(rowSum, vpaddlq_u16(blockSum));
                }
                if (alignedWidth != width)
                {
                    size_t col = width - A;
                    const uint8x16_t ad = vabdq_u8(Load<false>(a + col), Load<false>(b + col));
                    const uint8x16_t _mask = vandq_u8(vceqq_u8(Load<false>(mask + col), _index), tailMask);
                    rowSum = vaddq_u32(rowSum, vpaddlq_u16(vpaddlq_u8(vandq_u8(_mask, ad))));
                }
                _sum = vaddq_u64(_sum, vpaddlq_u32(rowSum));
                a += aStride;
                b += bStride;
                mask += maskStride;
            }
            *sum = ExtractSum64u(_sum);
        }

        void AbsDifferenceSumMasked(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(mask) && Aligned(maskStride))
                AbsDifferenceSumMasked<true>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
            else
                AbsDifferenceSumMasked<false>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
        }

        template <bool align> void AbsDifferenceSums3(uint8x16_t current, const uint8_t * background, uint16x8_t sums[3])
        {
            sums[0] = vaddq_u16(sums[0], vpaddlq_u8(vabdq_u8(current, Load<align>(background - 1))));
            sums[1] = vaddq_u16(sums[1], vpaddlq_u8(vabdq_u8(current, Load<false>(background))));
            sums[2] = vaddq_u16(sums[2], vpaddlq_u8(vabdq_u8(current, Load<false>(background + 1))));
        }

        template <bool align> void AbsDifferenceSums3x3(uint8x16_t current, const uint8_t * background, size_t stride, uint16x8_t sums[9])
        {
            AbsDifferenceSums3<align>(current, background - stride, sums + 0);
            AbsDifferenceSums3<align>(current, background, sums + 3);
            AbsDifferenceSums3<align>(current, background + stride, sums + 6);
        }

        template <bool align> void AbsDifferenceSums3Masked(uint8x16_t current, const uint8_t * background, uint8x16_t mask, uint32x4_t sums[3])
        {
            sums[0] = vaddq_u32(sums[0], vpaddlq_u16(vpaddlq_u8(vabdq_u8(current, vandq_u8(mask, Load<align>(background - 1))))));
            sums[1] = vaddq_u32(sums[1], vpaddlq_u16(vpaddlq_u8(vabdq_u8(current, vandq_u8(mask, Load<false>(background))))));
            sums[2] = vaddq_u32(sums[2], vpaddlq_u16(vpaddlq_u8(vabdq_u8(current, vandq_u8(mask, Load<false>(background + 1))))));
        }

        template <bool align> void AbsDifferenceSums3x3Masked(uint8x16_t current, const uint8_t * background, size_t stride, uint8x16_t mask, uint32x4_t sums[9])
        {
            AbsDifferenceSums3Masked<align>(current, background - stride, mask, sums + 0);
            AbsDifferenceSums3Masked<align>(current, background, mask, sums + 3);
            AbsDifferenceSums3Masked<align>(current, background + stride, mask, sums + 6);
        }

        template <bool align> void AbsDifferenceSums3x3(const uint8_t * current, size_t currentStride,
            const uint8_t * background, size_t backgroundStride, size_t width, size_t height, uint64_t * sums)
        {
            assert(height > 2 && width >= A + 2);
            if (align)
                assert(Aligned(background) && Aligned(backgroundStride));

            width -= 2;
            height -= 2;
            current += 1 + currentStride;
            background += 1 + backgroundStride;

            size_t alignedWidth = Simd::AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t blockSize = A << 7;
            size_t blockCount = (alignedWidth >> 7) + 1;

            uint64x2_t _sums[9];
            for (size_t i = 0; i < 9; ++i)
                _sums[i] = K64_0000000000000000;

            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowSums[9];
                for (size_t i = 0; i < 9; ++i)
                    rowSums[i] = K32_00000000;

                for (size_t block = 0; block < blockCount; ++block)
                {
                    uint16x8_t blockSums[9];
                    for (size_t i = 0; i < 9; ++i)
                        blockSums[i] = K16_0000;

                    for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
                    {
                        const uint8x16_t _current = Load<false>(current + col);
                        AbsDifferenceSums3x3<align>(_current, background + col, backgroundStride, blockSums);
                    }

                    for (size_t i = 0; i < 9; ++i)
                        rowSums[i] = vaddq_u32(rowSums[i], vpaddlq_u16(blockSums[i]));
                }

                if (alignedWidth != width)
                {
                    const uint8x16_t _current = vandq_u8(tailMask, Load<false>(current + width - A));
                    AbsDifferenceSums3x3Masked<false>(_current, background + width - A, backgroundStride, tailMask, rowSums);
                }

                for (size_t i = 0; i < 9; ++i)
                    _sums[i] = vaddq_u64(_sums[i], vpaddlq_u32(rowSums[i]));

                current += currentStride;
                background += backgroundStride;
            }

            for (size_t i = 0; i < 9; ++i)
                sums[i] = ExtractSum64u(_sums[i]);
        }

        void AbsDifferenceSums3x3(const uint8_t * current, size_t currentStride, const uint8_t * background, size_t backgroundStride,
            size_t width, size_t height, uint64_t * sums)
        {
            if (Aligned(background) && Aligned(backgroundStride))
                AbsDifferenceSums3x3<true>(current, currentStride, background, backgroundStride, width, height, sums);
            else
                AbsDifferenceSums3x3<false>(current, currentStride, background, backgroundStride, width, height, sums);
        }

        template <bool align> void AbsDifferenceSums3Masked(uint8x16_t current, const uint8_t * background, uint8x16_t mask, uint16x8_t sums[3])
        {
            sums[0] = vaddq_u16(sums[0], vpaddlq_u8(vabdq_u8(current, vandq_u8(mask, Load<align>(background - 1)))));
            sums[1] = vaddq_u16(sums[1], vpaddlq_u8(vabdq_u8(current, vandq_u8(mask, Load<false>(background)))));
            sums[2] = vaddq_u16(sums[2], vpaddlq_u8(vabdq_u8(current, vandq_u8(mask, Load<false>(background + 1)))));
        }

        template <bool align> void AbsDifferenceSums3x3Masked(uint8x16_t current, const uint8_t * background, size_t stride, uint8x16_t mask, uint16x8_t sums[9])
        {
            AbsDifferenceSums3Masked<align>(current, background - stride, mask, sums + 0);
            AbsDifferenceSums3Masked<align>(current, background, mask, sums + 3);
            AbsDifferenceSums3Masked<align>(current, background + stride, mask, sums + 6);
        }

        template <bool align> void AbsDifferenceSums3x3Masked(const uint8_t *current, size_t currentStride, const uint8_t *background, size_t backgroundStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sums)
        {
            assert(height > 2 && width >= A + 2);
            if (align)
                assert(Aligned(background) && Aligned(backgroundStride));

            width -= 2;
            height -= 2;
            current += 1 + currentStride;
            background += 1 + backgroundStride;
            mask += 1 + maskStride;

            size_t alignedWidth = Simd::AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t blockSize = A << 7;
            size_t blockCount = (alignedWidth >> 7) + 1;

            uint8x16_t _index = vdupq_n_u8(index);

            uint64x2_t _sums[9];
            for (size_t i = 0; i < 9; ++i)
                _sums[i] = K64_0000000000000000;

            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowSums[9];
                for (size_t i = 0; i < 9; ++i)
                    rowSums[i] = K32_00000000;

                for (size_t block = 0; block < blockCount; ++block)
                {
                    uint16x8_t blockSums[9];
                    for (size_t i = 0; i < 9; ++i)
                        blockSums[i] = K16_0000;

                    for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
                    {
                        const uint8x16_t _mask = vceqq_u8(Load<false>(mask + col), _index);
                        const uint8x16_t _current = vandq_u8(Load<false>(current + col), _mask);
                        AbsDifferenceSums3x3Masked<align>(_current, background + col, backgroundStride, _mask, blockSums);
                    }

                    for (size_t i = 0; i < 9; ++i)
                        rowSums[i] = vaddq_u32(rowSums[i], vpaddlq_u16(blockSums[i]));
                }

                if (alignedWidth != width)
                {
                    size_t col = width - A;
                    const uint8x16_t _mask = vandq_u8(tailMask, vceqq_u8(Load<false>(mask + col), _index));
                    const uint8x16_t _current = vandq_u8(_mask, Load<false>(current + col));
                    AbsDifferenceSums3x3Masked<false>(_current, background + col, backgroundStride, _mask, rowSums);
                }

                for (size_t i = 0; i < 9; ++i)
                    _sums[i] = vaddq_u64(_sums[i], vpaddlq_u32(rowSums[i]));

                current += currentStride;
                background += backgroundStride;
                mask += maskStride;
            }

            for (size_t i = 0; i < 9; ++i)
                sums[i] = ExtractSum64u(_sums[i]);
        }

        void AbsDifferenceSums3x3Masked(const uint8_t *current, size_t currentStride, const uint8_t *background, size_t backgroundStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sums)
        {
            if (Aligned(background) && Aligned(backgroundStride))
                AbsDifferenceSums3x3Masked<true>(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
            else
                AbsDifferenceSums3x3Masked<false>(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
        }
    }
#endif// SIMD_NEON_ENABLE
}
