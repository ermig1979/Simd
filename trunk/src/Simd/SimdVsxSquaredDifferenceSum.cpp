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
        template <bool align> void SquaredDifferenceSum(
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
                    const v128_u8 d = AbsDifferenceU8(_a, _b);
                    rowSum = vec_msum(d, d, rowSum);
                }
                if(width - bodyWidth)
                {
                    const v128_u8 _a = vec_and(tailMask, Load<false>(a + width - A));
                    const v128_u8 _b = vec_and(tailMask, Load<false>(b + width - A)); 
                    const v128_u8 d = AbsDifferenceU8(_a, _b);
                    rowSum = vec_msum(d, d, rowSum);
                }
                *sum += ExtractSum(rowSum);
                a += aStride;
                b += bStride;
            }
        }

        void SquaredDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, 
            size_t width, size_t height, uint64_t * sum)
        {
            if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
                SquaredDifferenceSum<true>(a, aStride, b, bStride, width, height, sum);
            else
                SquaredDifferenceSum<false>(a, aStride, b, bStride, width, height, sum);
        }

        template <bool align> void SquaredDifferenceSumMasked(
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
                    const v128_u8 d = AbsDifferenceU8(_a, _b);
                    rowSum = vec_msum(d, d, rowSum);
                }
                if(width - bodyWidth)
                {
                    const v128_u8 _mask = vec_and(tailMask, LoadMaskU8<false>(mask + width - A, _index));
                    const v128_u8 _a = vec_and(_mask, Load<false>(a + width - A));
                    const v128_u8 _b = vec_and(_mask, Load<false>(b + width - A)); 
                    const v128_u8 d = AbsDifferenceU8(_a, _b);
                    rowSum = vec_msum(d, d, rowSum);
                }
                *sum += ExtractSum(rowSum);
                a += aStride;
                b += bStride;
                mask += maskStride;
            }
        }

        void SquaredDifferenceSumMasked(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride, 
            const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
        {
            if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(mask) && Aligned(maskStride))
                SquaredDifferenceSumMasked<true>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
            else
                SquaredDifferenceSumMasked<false>(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
        }

        template <bool align> SIMD_INLINE float SquaredDifferenceSum32f(const float * a, const float * b, size_t size)
        {
            if(align)
                assert(Aligned(a) && Aligned(b));

            float sum = 0;
            size_t i = 0;
            size_t alignedSize = AlignLo(size, 4);
            if(alignedSize)
            {
                v128_f32 _sum = K_0_0f;
                for(; i < alignedSize; i += 4)
                {
                    v128_f32 _a = Load<align>(a + i);
                    v128_f32 _b = Load<align>(b + i);
                    v128_f32 _d = vec_sub(_a, _b);
                    _sum = vec_add(_sum, vec_mul(_d, _d));
                }
                sum += ExtractSum(_sum);
            }
            for(; i < size; ++i)
                sum += Simd::Square(a[i] - b[i]);
            return sum;
        }

        float SquaredDifferenceSum32f(const float * a, const float * b, size_t size)
        {
            if(Aligned(a) && Aligned(b))
                return SquaredDifferenceSum32f<true>(a, b, size);
            else
                return SquaredDifferenceSum32f<false>(a, b, size);
        }
    }
#endif// SIMD_VSX_ENABLE
}