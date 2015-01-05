/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#include "Simd/SimdVsx.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
    {
        template <bool align> void AbsDifferenceSum(
            const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, 
            size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

            size_t bodyWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + bodyWidth);
            *sum = 0;
            for(size_t row = 0; row < height; ++row)
            {
                v128_u32 rowSum = K32_00000000;
                for(size_t col = 0; col < bodyWidth; col += A)
                {
                    const v128_u8 _a = Load<align>(a + col);
                    const v128_u8 _b = Load<align>(b + col);
                    rowSum = vec_msum(AbsDifferenceU8(_a, _b), K8_01, rowSum);
                }
                if(width - bodyWidth)
                {
                    const v128_u8 _a = vec_and(tailMask, Load<false>(a + width - A));
                    const v128_u8 _b = vec_and(tailMask, Load<false>(b + width - A)); 
                    rowSum = vec_msum(AbsDifferenceU8(_a, _b), K8_01, rowSum);
                }
                *sum += ExtractSum(rowSum);
                a += aStride;
                b += bStride;
            }
        }

        void AbsDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, 
            size_t width, size_t height, uint64_t * sum)
        {
            if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
                AbsDifferenceSum<true>(a, aStride, b, bStride, width, height, sum);
            else
                AbsDifferenceSum<false>(a, aStride, b, bStride, width, height, sum);
        }

        template <bool align> void AbsDifferenceSumMasked(
            const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, 
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if(align)
            {
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));
                assert(Aligned(mask) && Aligned(maskStride));
            }

            size_t bodyWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + bodyWidth);
            v128_u8 _index = SetU8(index);
            *sum = 0;
            for(size_t row = 0; row < height; ++row)
            {
                v128_u32 rowSum = K32_00000000;
                for(size_t col = 0; col < bodyWidth; col += A)
                {
                    const v128_u8 _mask = LoadMaskU8<align>(mask + col, _index);
                    const v128_u8 _a = vec_and(_mask, Load<align>(a + col));
                    const v128_u8 _b = vec_and(_mask, Load<align>(b + col)); 
                    rowSum = vec_msum(AbsDifferenceU8(_a, _b), K8_01, rowSum);
                }
                if(width - bodyWidth)
                {
                    const v128_u8 _mask = vec_and(tailMask, LoadMaskU8<false>(mask + width - A, _index));
                    const v128_u8 _a = vec_and(_mask, Load<false>(a + width - A));
                    const v128_u8 _b = vec_and(_mask, Load<false>(b + width - A)); 
                    rowSum = vec_msum(AbsDifferenceU8(_a, _b), K8_01, rowSum);
                }
                *sum += ExtractSum(rowSum);
                a += aStride;
                b += bStride;
                mask += maskStride;
            }
        }

        void AbsDifferenceSumMasked(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, 
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(mask) && Aligned(maskStride))
                AbsDifferenceSumMasked<true>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
            else
                AbsDifferenceSumMasked<false>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
        }

        template <bool align> void AbsDifferenceSums3(const v128_u8 & current, const uint8_t * background, v128_u32 sums[3])
        {
            sums[0] = vec_msum(AbsDifferenceU8(current, Load<align>(background - 1)), K8_01, sums[0]);
            sums[1] = vec_msum(AbsDifferenceU8(current, Load<false>(background)), K8_01, sums[1]);
            sums[2] = vec_msum(AbsDifferenceU8(current, Load<false>(background + 1)), K8_01, sums[2]);
        }

        template <bool align> void AbsDifferenceSums3x3(const v128_u8 & current, const uint8_t * background, size_t stride, v128_u32 sums[9])
        {
            AbsDifferenceSums3<align>(current, background - stride, sums + 0);
            AbsDifferenceSums3<align>(current, background, sums + 3);
            AbsDifferenceSums3<align>(current, background + stride, sums + 6);
        }

        template <bool align> void AbsDifferenceSums3Masked(const v128_u8 & current, const uint8_t * background, const v128_u8 & mask, v128_u32 sums[3])
        {
            sums[0] = vec_msum(AbsDifferenceU8(current, vec_and(mask, Load<align>(background - 1))), K8_01, sums[0]);
            sums[1] = vec_msum(AbsDifferenceU8(current, vec_and(mask, Load<false>(background))), K8_01, sums[1]);
            sums[2] = vec_msum(AbsDifferenceU8(current, vec_and(mask, Load<false>(background + 1))), K8_01, sums[2]);
        }

        template <bool align> void AbsDifferenceSums3x3Masked(const v128_u8 & current, const uint8_t * background, size_t stride, const v128_u8 & mask, v128_u32 sums[9])
        {
            AbsDifferenceSums3Masked<align>(current, background - stride, mask, sums + 0);
            AbsDifferenceSums3Masked<align>(current, background, mask, sums + 3);
            AbsDifferenceSums3Masked<align>(current, background + stride, mask, sums + 6);
        }

        template <bool align> void AbsDifferenceSums3x3(const uint8_t * current, size_t currentStride, 
            const uint8_t * background, size_t backgroundStride, size_t width, size_t height, uint64_t * sums)
        {
            assert(height > 2 && width > A + 2);
            if(align)
                assert(Aligned(background) && Aligned(backgroundStride));

            width -= 2;
            height -= 2;
            current += 1 + currentStride;
            background += 1 + backgroundStride;

            size_t bodyWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + bodyWidth);

            for(size_t i = 0; i < 9; ++i)
                sums[i] = 0;

            for(size_t row = 0; row < height; ++row)
            {
                v128_u32 _sums[9];
                for(size_t i = 0; i < 9; ++i)
                    _sums[i] = K32_00000000;

                for(size_t col = 0; col < bodyWidth; col += A)
                {
                    const v128_u8 _current = Load<false>(current + col);
                    AbsDifferenceSums3x3<align>(_current, background + col, backgroundStride, _sums);
                }
                if(width - bodyWidth)
                {
                    const v128_u8 _current = vec_and(tailMask, Load<false>(current + width - A));
                    AbsDifferenceSums3x3Masked<false>(_current, background + width - A, backgroundStride, tailMask, _sums);
                }

                for(size_t i = 0; i < 9; ++i)
                    sums[i] += ExtractSum(_sums[i]);

                current += currentStride;
                background += backgroundStride;
            }
        }

        void AbsDifferenceSums3x3(const uint8_t * current, size_t currentStride, const uint8_t * background, size_t backgroundStride,
            size_t width, size_t height, uint64_t * sums)
        {
            if(Aligned(background) && Aligned(backgroundStride))
                AbsDifferenceSums3x3<true>(current, currentStride, background, backgroundStride, width, height, sums);
            else
                AbsDifferenceSums3x3<false>(current, currentStride, background, backgroundStride, width, height, sums);
        }

        template <bool align> void AbsDifferenceSums3x3Masked(const uint8_t *current, size_t currentStride, const uint8_t *background, size_t backgroundStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sums)
        {
            assert(height > 2 && width > A + 2);
            if(align)
                assert(Aligned(background) && Aligned(backgroundStride));

            width -= 2;
            height -= 2;
            current += 1 + currentStride;
            background += 1 + backgroundStride;
            mask += 1 + maskStride;

            size_t bodyWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + bodyWidth);
            v128_u8 _index = SetU8(index);

            for(size_t i = 0; i < 9; ++i)
                sums[i] = 0;

            for(size_t row = 0; row < height; ++row)
            {
                v128_u32 _sums[9];
                for(size_t i = 0; i < 9; ++i)
                    _sums[i] = K32_00000000;

                for(size_t col = 0; col < bodyWidth; col += A)
                {
                    const v128_u8 _mask = LoadMaskU8<false>(mask + col, _index);
                    const v128_u8 _current = vec_and(Load<false>(current + col), _mask);
                    AbsDifferenceSums3x3Masked<align>(_current, background + col, backgroundStride, _mask, _sums);
                }
                if(width - bodyWidth)
                {
                    const v128_u8 _mask = vec_and(LoadMaskU8<false>(mask + width - A, _index), tailMask);
                    const v128_u8 _current = vec_and(Load<false>(current + width - A), _mask);
                    AbsDifferenceSums3x3Masked<false>(_current, background + width - A, backgroundStride, _mask, _sums);
                }

                for(size_t i = 0; i < 9; ++i)
                    sums[i] += ExtractSum(_sums[i]);

                current += currentStride;
                background += backgroundStride;
                mask += maskStride;
            }
        }

        void AbsDifferenceSums3x3Masked(const uint8_t *current, size_t currentStride, const uint8_t *background, size_t backgroundStride,
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sums)
        {
            if(Aligned(background) && Aligned(backgroundStride))
                AbsDifferenceSums3x3Masked<true>(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
            else
                AbsDifferenceSums3x3Masked<false>(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
        }
    }
#endif// SIMD_VSX_ENABLE
}