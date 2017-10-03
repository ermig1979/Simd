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
#include "Simd/SimdLoad.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdExtract.h"

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        template <bool align> void SquaredDifferenceSum(const uint8_t * a, const uint8_t *b, size_t offset, v128_u32 & sum)
        {
            const v128_u8 _a = Load<align>(a + offset);
            const v128_u8 _b = Load<align>(b + offset);
            const v128_u8 d = AbsDifferenceU8(_a, _b);
            sum = vec_msum(d, d, sum);
        }

        template <bool align> void SquaredDifferenceSumMasked(const uint8_t * a, const uint8_t *b, size_t offset, const v128_u8 & mask, v128_u32 & sum)
        {
            const v128_u8 _a = vec_and(Load<align>(a + offset), mask);
            const v128_u8 _b = vec_and(Load<align>(b + offset), mask);
            const v128_u8 d = AbsDifferenceU8(_a, _b);
            sum = vec_msum(d, d, sum);
        }

        template <bool align> void SquaredDifferenceSum(
            const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

            size_t alignedWidth = AlignLo(width, QA);
            size_t bodyWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + bodyWidth);
            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                v128_u32 sums[4] = { K32_00000000, K32_00000000, K32_00000000, K32_00000000 };
                for (; col < alignedWidth; col += QA)
                {
                    SquaredDifferenceSum<align>(a, b, col, sums[0]);
                    SquaredDifferenceSum<align>(a, b, col + A, sums[1]);
                    SquaredDifferenceSum<align>(a, b, col + 2 * A, sums[2]);
                    SquaredDifferenceSum<align>(a, b, col + 3 * A, sums[3]);
                }
                sums[0] = vec_add(vec_add(sums[0], sums[1]), vec_add(sums[2], sums[3]));
                for (; col < bodyWidth; col += A)
                    SquaredDifferenceSum<align>(a, b, col, sums[0]);
                if (width - bodyWidth)
                    SquaredDifferenceSumMasked<false>(a, b, width - A, tailMask, sums[0]);
                *sum += ExtractSum(sums[0]);
                a += aStride;
                b += bStride;
            }
        }

        void SquaredDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
                SquaredDifferenceSum<true>(a, aStride, b, bStride, width, height, sum);
            else
                SquaredDifferenceSum<false>(a, aStride, b, bStride, width, height, sum);
        }

        template <bool align> void SquaredDifferenceSumMasked(const uint8_t * a, const uint8_t *b, const uint8_t * mask, size_t offset, const v128_u8 & index, v128_u32 & sum)
        {
            const v128_u8 _mask = LoadMaskU8<align>(mask + offset, index);
            SquaredDifferenceSumMasked<align>(a, b, offset, _mask, sum);
        }

        template <bool align> void SquaredDifferenceSumMasked(
            const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));
                assert(Aligned(mask) && Aligned(maskStride));
            }

            size_t alignedWidth = AlignLo(width, QA);
            size_t bodyWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + bodyWidth);
            v128_u8 _index = SetU8(index);
            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                v128_u32 sums[4] = { K32_00000000, K32_00000000, K32_00000000, K32_00000000 };
                for (; col < alignedWidth; col += QA)
                {
                    SquaredDifferenceSumMasked<align>(a, b, mask, col, _index, sums[0]);
                    SquaredDifferenceSumMasked<align>(a, b, mask, col + A, _index, sums[1]);
                    SquaredDifferenceSumMasked<align>(a, b, mask, col + 2 * A, _index, sums[2]);
                    SquaredDifferenceSumMasked<align>(a, b, mask, col + 3 * A, _index, sums[3]);
                }
                sums[0] = vec_add(vec_add(sums[0], sums[1]), vec_add(sums[2], sums[3]));
                for (; col < bodyWidth; col += A)
                    SquaredDifferenceSumMasked<align>(a, b, mask, col, _index, sums[0]);
                if (width - bodyWidth)
                {
                    const v128_u8 _mask = vec_and(tailMask, LoadMaskU8<false>(mask + width - A, _index));
                    SquaredDifferenceSumMasked<false>(a, b, width - A, _mask, sums[0]);
                }
                *sum += ExtractSum(sums[0]);
                a += aStride;
                b += bStride;
                mask += maskStride;
            }
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
#endif// SIMD_VMX_ENABLE
}
