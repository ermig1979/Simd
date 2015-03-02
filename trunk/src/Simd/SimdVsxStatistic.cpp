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
#include "Simd/SimdCompare.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
    {
        template <bool align> void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height, 
            uint8_t * min, uint8_t * max, uint8_t * average)
        {
            assert(width*height && width >= A);
            if(align)
                assert(Aligned(src) && Aligned(stride));

            size_t bodyWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + bodyWidth);
            uint64_t sum = 0;
            v128_u8 min_ = K8_FF;
            v128_u8 max_ = K8_00;
            for(size_t row = 0; row < height; ++row)
            {
                v128_u32 rowSum = K32_00000000;
                for(size_t col = 0; col < bodyWidth; col += A)
                {
                    const v128_u8 value = Load<align>(src + col);
                    min_ = vec_min(min_, value);
                    max_ = vec_max(max_, value);
                    rowSum = vec_msum(value, K8_01, rowSum);
                }
                if(width - bodyWidth)
                {
                    const v128_u8 value = Load<false>(src + width - A);
                    min_ = vec_min(min_, value);
                    max_ = vec_max(max_, value);
                    rowSum = vec_msum(vec_and(value, tailMask), K8_01, rowSum);
                }
                sum += ExtractSum(rowSum);
                src += stride;
            }

            *min = UCHAR_MAX;
            *max = 0;
            for (size_t i = 0; i < A; ++i)
            {
                *min = Base::MinU8(vec_extract(min_, i), *min);
                *max = Base::MaxU8(vec_extract(max_, i), *max);
            }
            *average = (uint8_t)((sum + width*height/2)/(width*height));
        }

        void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height, 
            uint8_t * min, uint8_t * max, uint8_t * average)
        {
            if(Aligned(src) && Aligned(stride))
                GetStatistic<true>(src, stride, width, height, min, max, average);
            else
                GetStatistic<false>(src, stride, width, height, min, max, average);
        }

        SIMD_INLINE void GetMoments(const v128_u16 & row, const v128_u16 & col, 
            v128_u32 & x, v128_u32 & y, v128_u32 & xx, v128_u32 & xy, v128_u32 & yy)
        {
            x = vec_msum(col, K16_0001, x);
            y = vec_msum(row, K16_0001, y);
            xx = vec_msum(col, col, xx);
            xy = vec_msum(col, row, xy);
            yy = vec_msum(row, row, yy);
        }

        SIMD_INLINE void SumTo(const v128_u32 & value, uint64_t * sum)
        {
            *sum += vec_extract(value, 0);
            *sum += vec_extract(value, 1);
            *sum += vec_extract(value, 2);
            *sum += vec_extract(value, 3);
        }

        SIMD_INLINE void GetMoments(const v128_u8 & mask, v128_u16 & row, v128_u16 & col, 
            v128_u32 & area, v128_u32 & x, v128_u32 & y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            area = vec_msum(vec_and(K8_01, mask), K8_01, area);

            v128_u32 _xx = K32_00000000;
            v128_u32 _xy = K32_00000000;
            v128_u32 _yy = K32_00000000;

            const v128_u16 lo = (v128_u16)vec_unpackh((v128_s8)mask);
            GetMoments(vec_and(lo, row), vec_and(lo, col), x, y, _xx, _xy, _yy);
            col = vec_add(col, K16_0008);

            const v128_u16 hi = (v128_u16)vec_unpackl((v128_s8)mask);
            GetMoments(vec_and(hi, row), vec_and(hi, col), x, y, _xx, _xy, _yy);
            col = vec_add(col, K16_0008);

            SumTo(_xx, xx);
            SumTo(_xy, xy);
            SumTo(_yy, yy);
        }

        template <bool align> void GetMoments(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            assert(width >= A && width < SHRT_MAX && height < SHRT_MAX);
            if(align)
                assert(Aligned(mask) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);

            const v128_u16 K16_I = SIMD_VEC_SETR_EPI16(0, 1, 2, 3, 4, 5, 6, 7);
            const v128_u8 _index = SetU8(index);
            const v128_u16 tailCol = vec_add(K16_I, SetU16(width - A));

            *x = 0;
            *y = 0;
            *xx = 0;
            *xy = 0;
            *yy = 0;

            v128_u32 _area = K32_00000000;
            for(size_t row = 0; row < height; ++row)
            {
                v128_u16 _col = K16_I;
                v128_u16 _row = SetU16(row);

                v128_u32 _x = K32_00000000;
                v128_u32 _y = K32_00000000;

                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    v128_u8 _mask = (v128_u8 )vec_cmpeq(Load<align>(mask + col), _index);
                    GetMoments(_mask, _row, _col, _area, _x, _y, xx, xy, yy);
                }
                if(alignedWidth != width)
                {
                    v128_u8 _mask = vec_and(vec_cmpeq(Load<false>(mask + width - A), _index), tailMask);
                    _col = tailCol;
                    GetMoments(_mask, _row, _col, _area, _x, _y, xx, xy, yy);
                }

                *x += ExtractSum(_x);
                *y += ExtractSum(_y);

                mask += stride;
            }
            *area = ExtractSum(_area);
        }

        void GetMoments(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index, 
            uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            if(Aligned(mask) && Aligned(stride))
                GetMoments<true>(mask, stride, width, height, index, area, x, y, xx, xy, yy);
            else
                GetMoments<false>(mask, stride, width, height, index, area, x, y, xx, xy, yy);
        }

        template <bool align> void GetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_01, A - width + alignedWidth);

            memset(sums, 0, sizeof(uint32_t)*height);
            for(size_t row = 0; row < height; ++row)
            {
                v128_u32 sum = K32_00000000;
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    v128_u8 _src = Load<align>(src + col);
                    sum = vec_msum(_src, K8_01, sum);
                }
                if(alignedWidth != width)
                {
                    v128_u8 _src = Load<false>(src + width - A);
                    sum = vec_msum(_src, tailMask, sum);
                }
                sums[row] = ExtractSum(sum);
                src += stride;
            }
        }

        void GetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if(Aligned(src) && Aligned(stride))
                GetRowSums<true>(src, stride, width, height, sums);
            else
                GetRowSums<false>(src, stride, width, height, sums);
        }

        template <bool align> void GetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_01, A - width + alignedWidth);

            memset(sums, 0, sizeof(uint32_t)*height);
            const uint8_t * src0 = src;
            const uint8_t * src1 = src + stride;
            height--;
            for(size_t row = 0; row < height; ++row)
            {
                v128_u32 sum = K32_00000000;
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    v128_u8 _src0 = Load<align>(src0 + col);
                    v128_u8 _src1 = Load<align>(src1 + col);
                    sum = vec_msum(AbsDifferenceU8(_src0, _src1), K8_01, sum);
                }
                if(alignedWidth != width)
                {
                    v128_u8 _src0 = Load<false>(src0 + width - A);
                    v128_u8 _src1 = Load<false>(src1 + width - A);
                    sum = vec_msum(AbsDifferenceU8(_src0, _src1), tailMask, sum);
                }
                sums[row] = ExtractSum(sum);
                src0 += stride;
                src1 += stride;
            }
        }

        void GetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if(Aligned(src) && Aligned(stride))
                GetAbsDyRowSums<true>(src, stride, width, height, sums);
            else
                GetAbsDyRowSums<false>(src, stride, width, height, sums);
        }

        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t)*width + sizeof(uint32_t)*width);
                    sums16 = (uint16_t*)_p;
                    sums32 = (uint32_t*)(sums16 + width);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t * sums16;
                uint32_t * sums32;
            private:
                void *_p;
            };
        }

        template <bool align> SIMD_INLINE void Sum16(v128_u8 src8, uint16_t * sums16)
        {
            Store<align>(sums16, vec_add(Load<align>(sums16), UnpackLoU8(src8)));
            Store<align>(sums16 + HA, vec_add(Load<align>(sums16 + HA), UnpackHiU8(src8)));
        }

        template <bool align> SIMD_INLINE void Sum32(v128_u16 src16, uint32_t * sums32)
        {
            Store<align>(sums32, vec_add(Load<align>(sums32), UnpackLoU16(src16)));
            Store<align>(sums32 + 4, vec_add(Load<align>(sums32 + 4), UnpackHiU16(src16)));
        }

        template <bool align> void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedLoWidth);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX)/stepSize;

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint32_t)*alignedHiWidth);

            for(size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Simd::Min(rowStart + stepSize, height);

                memset(buffer.sums16, 0, sizeof(uint16_t)*width);
                for(size_t row = rowStart; row < rowEnd; ++row)
                {
                    for(size_t col = 0; col < alignedLoWidth; col += A)
                    {
                        v128_u8 _src = Load<align>(src + col);
                        Sum16<true>(_src, buffer.sums16 + col);
                    }
                    if(alignedLoWidth != width)
                    {
                        v128_u8 _src = Load<false>(src + width - A);
                        Sum16<false>(vec_and(_src, tailMask), buffer.sums16 + width - A);
                    }
                    src += stride;
                }

                for(size_t col = 0; col < alignedHiWidth; col += HA)
                {
                    v128_u16 src16 = Load<true>(buffer.sums16 + col);
                    Sum32<true>(src16, buffer.sums32 + col);
                }
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t)*width);
        }

        void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if(Aligned(src) && Aligned(stride))
                GetColSums<true>(src, stride, width, height, sums);
            else
                GetColSums<false>(src, stride, width, height, sums);
        }

        template <bool align> void GetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            width--;
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedLoWidth);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX)/stepSize;

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint32_t)*alignedHiWidth);

            for(size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Simd::Min(rowStart + stepSize, height);

                memset(buffer.sums16, 0, sizeof(uint16_t)*width);
                for(size_t row = rowStart; row < rowEnd; ++row)
                {
                    for(size_t col = 0; col < alignedLoWidth; col += A)
                    {
                        v128_u8 _src0 = Load<align>(src + col + 0);
                        v128_u8 _src1 = Load<false>(src + col + 1);
                        Sum16<true>(AbsDifferenceU8(_src0, _src1), buffer.sums16 + col);
                    }
                    if(alignedLoWidth != width)
                    {
                        v128_u8 _src0 = Load<false>(src + width - A + 0);
                        v128_u8 _src1 = Load<false>(src + width - A + 1);
                        Sum16<false>(vec_and(AbsDifferenceU8(_src0, _src1), tailMask), buffer.sums16 + width - A);
                    }
                    src += stride;
                }

                for(size_t col = 0; col < alignedHiWidth; col += HA)
                {
                    v128_u16 src16 = Load<true>(buffer.sums16 + col);
                    Sum32<true>(src16, buffer.sums32 + col);
                }
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t)*width);
            sums[width] = 0;
        }

        void GetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if(Aligned(src) && Aligned(stride))
                GetAbsDxColSums<true>(src, stride, width, height, sums);
            else
                GetAbsDxColSums<false>(src, stride, width, height, sums);
        }

        template <bool align> void ValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            *sum = 0;
            for(size_t row = 0; row < height; ++row)
            {
                v128_u32 _sum = K32_00000000;
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    const v128_u8 _src = Load<align>(src + col);
                    _sum = vec_msum(_src, K8_01, _sum);
                }
                if(alignedWidth != width)
                {
                    const v128_u8 _src = vec_and(Load<false>(src + width - A), tailMask);
                    _sum = vec_msum(_src, K8_01, _sum);
                }
                *sum += ExtractSum(_sum);
                src += stride;
            }
        }

        void ValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            if(Aligned(src) && Aligned(stride))
                ValueSum<true>(src, stride, width, height, sum);
            else
                ValueSum<false>(src, stride, width, height, sum);
        }

        template <bool align> void SquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            *sum = 0;
            for(size_t row = 0; row < height; ++row)
            {
                v128_u32 _sum = K32_00000000;
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    const v128_u8 _src = Load<align>(src + col);
                    _sum = vec_msum(_src, _src, _sum);
                }
                if(alignedWidth != width)
                {
                    const v128_u8 _src = vec_and(Load<false>(src + width - A), tailMask);
                    _sum = vec_msum(_src, _src, _sum);
                }
                *sum += ExtractSum(_sum);
                src += stride;
            }
        }

        void SquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            if(Aligned(src) && Aligned(stride))
                SquareSum<true>(src, stride, width, height, sum);
            else
                SquareSum<false>(src, stride, width, height, sum);
        }

        template <bool align> void CorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            *sum = 0;
            for(size_t row = 0; row < height; ++row)
            {
                v128_u32 _sum = K32_00000000;
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    const v128_u8 _a = Load<align>(a + col);
                    const v128_u8 _b = Load<align>(b + col);
                    _sum = vec_msum(_a, _b, _sum);
                }
                if(alignedWidth != width)
                {
                    const v128_u8 _a = vec_and(Load<false>(a + width - A), tailMask);
                    const v128_u8 _b = vec_and(Load<false>(b + width - A), tailMask);
                    _sum = vec_msum(_a, _b, _sum);
                }
                *sum += ExtractSum(_sum);
                a += aStride;
                b += bStride;
            }
        }

        void CorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum)
        {
            if(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
                CorrelationSum<true>(a, aStride, b, bStride, width, height, sum);
            else
                CorrelationSum<false>(a, aStride, b, bStride, width, height, sum);
        }
    }
#endif// SIMD_VSX_ENABLE
}