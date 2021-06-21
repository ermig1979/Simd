/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar,
*               2018-2018 Radchenko Andrey.
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
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height,
            uint8_t * min, uint8_t * max, uint8_t * average)
        {
            assert(width*height && width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t blockSize = A << 8;
            size_t blockCount = (alignedWidth >> 8) + 1;
            uint64x2_t fullSum = K64_0000000000000000;
            uint8x16_t _min = K8_FF;
            uint8x16_t _max = K8_00;
            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowSum = K32_00000000;
                for (size_t block = 0; block < blockCount; ++block)
                {
                    uint16x8_t blockSum = K16_0000;
                    for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
                    {
                        const uint8x16_t _src = Load<align>(src + col);
                        _min = vminq_u8(_min, _src);
                        _max = vmaxq_u8(_max, _src);
                        blockSum = vaddq_u16(blockSum, vpaddlq_u8(_src));
                    }
                    rowSum = vaddq_u32(rowSum, vpaddlq_u16(blockSum));
                }
                if (width - alignedWidth)
                {
                    const uint8x16_t _src = Load<false>(src + width - A);
                    _min = vminq_u8(_min, _src);
                    _max = vmaxq_u8(_max, _src);
                    rowSum = vaddq_u32(rowSum, vpaddlq_u16(vpaddlq_u8(vandq_u8(_src, tailMask))));
                }
                fullSum = vaddq_u64(fullSum, vpaddlq_u32(rowSum));
                src += stride;
            }

            uint8_t min_buffer[A], max_buffer[A];
            Store<false>(min_buffer, _min);
            Store<false>(max_buffer, _max);
            *min = UCHAR_MAX;
            *max = 0;
            for (size_t i = 0; i < A; ++i)
            {
                *min = Base::MinU8(min_buffer[i], *min);
                *max = Base::MaxU8(max_buffer[i], *max);
            }
            *average = (uint8_t)((ExtractSum64u(fullSum) + width*height / 2) / (width*height));
        }

        void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height,
            uint8_t * min, uint8_t * max, uint8_t * average)
        {
            if (Aligned(src) && Aligned(stride))
                GetStatistic<true>(src, stride, width, height, min, max, average);
            else
                GetStatistic<false>(src, stride, width, height, min, max, average);
        }

        //-----------------------------------------------------------------------

        template <bool align> void GetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedWidth = AlignLo(width, A);
            const uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t blockSize = A << 8;
            size_t blockCount = (alignedWidth >> 8) + 1;

            memset(sums, 0, sizeof(uint32_t)*height);
            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowSum = K32_00000000;
                for (size_t block = 0; block < blockCount; ++block)
                {
                    uint16x8_t blockSum = K16_0000;
                    for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
                    {
                        const uint8x16_t _src = Load<align>(src + col);
                        blockSum = vaddq_u16(blockSum, vpaddlq_u8(_src));
                    }
                    rowSum = vaddq_u32(rowSum, vpaddlq_u16(blockSum));
                }
                if (alignedWidth != width)
                {
                    const uint8x16_t _src = vandq_u8(Load<false>(src + width - A), tailMask);
                    rowSum = vaddq_u32(rowSum, vpaddlq_u16(vpaddlq_u8(_src)));
                }
                sums[row] = ExtractSum32u(rowSum);
                src += stride;
            }
        }

        void GetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetRowSums<true>(src, stride, width, height, sums);
            else
                GetRowSums<false>(src, stride, width, height, sums);
        }

        //-----------------------------------------------------------------------

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

        template <bool align> SIMD_INLINE void Sum16(const uint8x16_t & src, uint16_t * dst)
        {
            Store<align>(dst + 0, vaddq_u16(Load<align>(dst + 0), UnpackU8<0>(src)));
            Store<align>(dst + 8, vaddq_u16(Load<align>(dst + 8), UnpackU8<1>(src)));
        }

        template <bool align> SIMD_INLINE void Sum32(const uint16x8_t & src, uint32_t * dst)
        {
            Store<align>(dst + 0, vaddq_u32(Load<align>(dst + 0), UnpackU16<0>(src)));
            Store<align>(dst + 4, vaddq_u32(Load<align>(dst + 4), UnpackU16<1>(src)));
        }

        template <bool align> void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            const uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedLoWidth);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX) / stepSize;

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint32_t)*alignedHiWidth);
            for (size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);

                memset(buffer.sums16, 0, sizeof(uint16_t)*width);
                for (size_t row = rowStart; row < rowEnd; ++row)
                {
                    for (size_t col = 0; col < alignedLoWidth; col += A)
                    {
                        const uint8x16_t _src = Load<align>(src + col);
                        Sum16<true>(_src, buffer.sums16 + col);
                    }
                    if (alignedLoWidth != width)
                    {
                        const uint8x16_t _src = vandq_u8(Load<false>(src + width - A), tailMask);
                        Sum16<false>(_src, buffer.sums16 + width - A);
                    }
                    src += stride;
                }

                for (size_t col = 0; col < alignedHiWidth; col += HA)
                    Sum32<true>(Load<true>(buffer.sums16 + col), buffer.sums32 + col);
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t)*width);
        }

        void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetColSums<true>(src, stride, width, height, sums);
            else
                GetColSums<false>(src, stride, width, height, sums);
        }

        template <bool align> void GetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedWidth = AlignLo(width, A);
            const uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t blockSize = A << 8;
            size_t blockCount = (alignedWidth >> 8) + 1;

            memset(sums, 0, sizeof(uint32_t)*height);
            const uint8_t * src0 = src;
            const uint8_t * src1 = src + stride;
            height--;
            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowSum = K32_00000000;
                for (size_t block = 0; block < blockCount; ++block)
                {
                    uint16x8_t blockSum = K16_0000;
                    for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
                    {
                        const uint8x16_t _src0 = Load<align>(src0 + col);
                        const uint8x16_t _src1 = Load<align>(src1 + col);
                        blockSum = vaddq_u16(blockSum, vpaddlq_u8(vabdq_u8(_src0, _src1)));
                    }
                    rowSum = vaddq_u32(rowSum, vpaddlq_u16(blockSum));
                }
                if (alignedWidth != width)
                {
                    const uint8x16_t _src0 = Load<false>(src0 + width - A);
                    const uint8x16_t _src1 = Load<false>(src1 + width - A);
                    rowSum = vaddq_u32(rowSum, vpaddlq_u16(vpaddlq_u8(vandq_u8(vabdq_u8(_src0, _src1), tailMask))));
                }
                sums[row] = ExtractSum32u(rowSum);
                src0 += stride;
                src1 += stride;
            }
        }

        void GetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetAbsDyRowSums<true>(src, stride, width, height, sums);
            else
                GetAbsDyRowSums<false>(src, stride, width, height, sums);
        }

        //-----------------------------------------------------------------------

        template <bool align> void GetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            width--;
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            const uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedLoWidth);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX) / stepSize;

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint32_t)*alignedHiWidth);
            for (size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);

                memset(buffer.sums16, 0, sizeof(uint16_t)*width);
                for (size_t row = rowStart; row < rowEnd; ++row)
                {
                    for (size_t col = 0; col < alignedLoWidth; col += A)
                    {
                        const uint8x16_t _src0 = Load<align>(src + col + 0);
                        const uint8x16_t _src1 = Load<false>(src + col + 1);
                        Sum16<true>(vabdq_u8(_src0, _src1), buffer.sums16 + col);
                    }
                    if (alignedLoWidth != width)
                    {
                        const uint8x16_t _src0 = Load<false>(src + width - A + 0);
                        const uint8x16_t _src1 = Load<false>(src + width - A + 1);
                        Sum16<false>(vandq_u8(vabdq_u8(_src0, _src1), tailMask), buffer.sums16 + width - A);
                    }
                    src += stride;
                }

                for (size_t col = 0; col < alignedHiWidth; col += HA)
                    Sum32<true>(Load<true>(buffer.sums16 + col), buffer.sums32 + col);
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t)*width);
            sums[width] = 0;
        }

        void GetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetAbsDxColSums<true>(src, stride, width, height, sums);
            else
                GetAbsDxColSums<false>(src, stride, width, height, sums);
        }

        //-----------------------------------------------------------------------

        template <bool align> void ValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t blockSize = A << 8;
            size_t blockCount = (alignedWidth >> 8) + 1;
            uint64x2_t fullSum = K64_0000000000000000;
            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowSum = K32_00000000;
                for (size_t block = 0; block < blockCount; ++block)
                {
                    uint16x8_t blockSum = K16_0000;
                    for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
                    {
                        const uint8x16_t _src = Load<align>(src + col);
                        blockSum = vaddq_u16(blockSum, vpaddlq_u8(_src));
                    }
                    rowSum = vaddq_u32(rowSum, vpaddlq_u16(blockSum));
                }
                if (width - alignedWidth)
                {
                    const uint8x16_t _src = vandq_u8(Load<false>(src + width - A), tailMask);
                    rowSum = vaddq_u32(rowSum, vpaddlq_u16(vpaddlq_u8(_src)));
                }
                fullSum = vaddq_u64(fullSum, vpaddlq_u32(rowSum));
                src += stride;
            }
            *sum = ExtractSum64u(fullSum);
        }

        void ValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(src) && Aligned(stride))
                ValueSum<true>(src, stride, width, height, sum);
            else
                ValueSum<false>(src, stride, width, height, sum);
        }

        //-----------------------------------------------------------------------

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

        template <bool align> void SquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);

            uint64x2_t fullSum = K64_0000000000000000;
            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowSum = K32_00000000;
                for (size_t col = 0; col < alignedWidth; col += A)
                    rowSum = vaddq_u32(rowSum, Square(Load<align>(src + col)));
                if (alignedWidth != width)
                    rowSum = vaddq_u32(rowSum, Square(vandq_u8(Load<false>(src + width - A), tailMask)));
                fullSum = vaddq_u64(fullSum, vpaddlq_u32(rowSum));
                src += stride;
            }
            *sum = ExtractSum64u(fullSum);
        }

        void SquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(src) && Aligned(stride))
                SquareSum<true>(src, stride, width, height, sum);
            else
                SquareSum<false>(src, stride, width, height, sum);
        }

        //-----------------------------------------------------------------------

        template <bool align> void ValueSquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * valueSum, uint64_t * squareSum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            uint64x2_t fullValueSum = K64_0000000000000000;
            uint64x2_t fullSquareSum = K64_0000000000000000;
            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowValueSum = K32_00000000;
                uint32x4_t rowSquareSum = K32_00000000;
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    uint8x16_t _src = Load<align>(src + col);
                    rowValueSum = vpadalq_u16(rowValueSum, vpaddlq_u8(_src));
                    rowSquareSum = vaddq_u32(rowSquareSum, Square(_src));
                }
                if (alignedWidth != width)
                {
                    uint8x16_t _src = vandq_u8(Load<false>(src + width - A), tailMask);
                    rowValueSum = vpadalq_u16(rowValueSum, vpaddlq_u8(_src));
                    rowSquareSum = vaddq_u32(rowSquareSum, Square(_src));
                }
                fullValueSum = vaddq_u64(fullValueSum, vpaddlq_u32(rowValueSum));
                fullSquareSum = vaddq_u64(fullSquareSum, vpaddlq_u32(rowSquareSum));
                src += stride;
            }
            *valueSum = ExtractSum64u(fullValueSum);
            *squareSum = ExtractSum64u(fullSquareSum);
        }

        void ValueSquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * valueSum, uint64_t * squareSum)
        {
            if (Aligned(src) && Aligned(stride))
                ValueSquareSum<true>(src, stride, width, height, valueSum, squareSum);
            else
                ValueSquareSum<false>(src, stride, width, height, valueSum, squareSum);
        }

        //-----------------------------------------------------------------------

        template <bool align> void ValueSquareSums2(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t widthA = Simd::AlignLo(width, A);
            uint8x16_t tail = ShiftLeft(K8_FF, A - width + widthA);
            uint64x2_t fullValueSums[2] = { K64_0000000000000000, K64_0000000000000000 };
            uint64x2_t fullSquareSums[2] = { K64_0000000000000000, K64_0000000000000000 };
            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowValueSums[2] = { K32_00000000, K32_00000000 };
                uint32x4_t rowSquareSums[2] = { K32_00000000, K32_00000000 };
                for (size_t col = 0, offs = 0; col < widthA; col += A, offs += A * 2)
                {
                    uint8x16x2_t _src = Load2<align>(src + offs);
                    rowValueSums[0] = vpadalq_u16(rowValueSums[0], vpaddlq_u8(_src.val[0]));
                    rowValueSums[1] = vpadalq_u16(rowValueSums[1], vpaddlq_u8(_src.val[1]));
                    rowSquareSums[0] = vaddq_u32(rowSquareSums[0], Square(_src.val[0]));
                    rowSquareSums[1] = vaddq_u32(rowSquareSums[1], Square(_src.val[1]));
                }
                if (widthA != width)
                {
                    size_t offs = (width - A) * 2;
                    uint8x16x2_t _src = Load2<align>(src + offs);
                    _src.val[0] = vandq_u8(_src.val[0], tail);
                    _src.val[1] = vandq_u8(_src.val[1], tail);
                    rowValueSums[0] = vpadalq_u16(rowValueSums[0], vpaddlq_u8(_src.val[0]));
                    rowValueSums[1] = vpadalq_u16(rowValueSums[1], vpaddlq_u8(_src.val[1]));
                    rowSquareSums[0] = vaddq_u32(rowSquareSums[0], Square(_src.val[0]));
                    rowSquareSums[1] = vaddq_u32(rowSquareSums[1], Square(_src.val[1]));
                }
                fullValueSums[0] = vaddq_u64(fullValueSums[0], vpaddlq_u32(rowValueSums[0]));
                fullValueSums[1] = vaddq_u64(fullValueSums[1], vpaddlq_u32(rowValueSums[1]));
                fullSquareSums[0] = vaddq_u64(fullSquareSums[0], vpaddlq_u32(rowSquareSums[0]));
                fullSquareSums[1] = vaddq_u64(fullSquareSums[1], vpaddlq_u32(rowSquareSums[1]));
                src += stride;
            }
            valueSums[0] = ExtractSum64u(fullValueSums[0]);
            valueSums[1] = ExtractSum64u(fullValueSums[1]);
            squareSums[0] = ExtractSum64u(fullSquareSums[0]);
            squareSums[1] = ExtractSum64u(fullSquareSums[1]);
        }

        void ValueSquareSums2(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            if (Aligned(src) && Aligned(stride))
                ValueSquareSums2<true>(src, stride, width, height, valueSums, squareSums);
            else
                ValueSquareSums2<false>(src, stride, width, height, valueSums, squareSums);
        }

        template <bool align> void ValueSquareSums3(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t widthA = Simd::AlignLo(width, A);
            uint8x16_t tail = ShiftLeft(K8_FF, A - width + widthA);
            uint64x2_t fullValueSums[3] = { K64_0000000000000000, K64_0000000000000000, K64_0000000000000000 };
            uint64x2_t fullSquareSums[3] = { K64_0000000000000000, K64_0000000000000000, K64_0000000000000000 };
            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowValueSums[3] = { K32_00000000, K32_00000000, K32_00000000 };
                uint32x4_t rowSquareSums[3] = { K32_00000000, K32_00000000, K32_00000000 };
                for (size_t col = 0, offs = 0; col < widthA; col += A, offs += A * 3)
                {
                    uint8x16x3_t _src = Load3<align>(src + offs);
                    rowValueSums[0] = vpadalq_u16(rowValueSums[0], vpaddlq_u8(_src.val[0]));
                    rowValueSums[1] = vpadalq_u16(rowValueSums[1], vpaddlq_u8(_src.val[1]));
                    rowValueSums[2] = vpadalq_u16(rowValueSums[2], vpaddlq_u8(_src.val[2]));
                    rowSquareSums[0] = vaddq_u32(rowSquareSums[0], Square(_src.val[0]));
                    rowSquareSums[1] = vaddq_u32(rowSquareSums[1], Square(_src.val[1]));
                    rowSquareSums[2] = vaddq_u32(rowSquareSums[2], Square(_src.val[2]));
                }
                if (widthA != width)
                {
                    size_t offs = (width - A) * 3;
                    uint8x16x3_t _src = Load3<align>(src + offs);
                    _src.val[0] = vandq_u8(_src.val[0], tail);
                    _src.val[1] = vandq_u8(_src.val[1], tail);
                    _src.val[2] = vandq_u8(_src.val[2], tail);
                    rowValueSums[0] = vpadalq_u16(rowValueSums[0], vpaddlq_u8(_src.val[0]));
                    rowValueSums[1] = vpadalq_u16(rowValueSums[1], vpaddlq_u8(_src.val[1]));
                    rowValueSums[2] = vpadalq_u16(rowValueSums[2], vpaddlq_u8(_src.val[2]));
                    rowSquareSums[0] = vaddq_u32(rowSquareSums[0], Square(_src.val[0]));
                    rowSquareSums[1] = vaddq_u32(rowSquareSums[1], Square(_src.val[1]));
                    rowSquareSums[2] = vaddq_u32(rowSquareSums[2], Square(_src.val[2]));
                }
                fullValueSums[0] = vaddq_u64(fullValueSums[0], vpaddlq_u32(rowValueSums[0]));
                fullValueSums[1] = vaddq_u64(fullValueSums[1], vpaddlq_u32(rowValueSums[1]));
                fullValueSums[2] = vaddq_u64(fullValueSums[2], vpaddlq_u32(rowValueSums[2]));
                fullSquareSums[0] = vaddq_u64(fullSquareSums[0], vpaddlq_u32(rowSquareSums[0]));
                fullSquareSums[1] = vaddq_u64(fullSquareSums[1], vpaddlq_u32(rowSquareSums[1]));
                fullSquareSums[2] = vaddq_u64(fullSquareSums[2], vpaddlq_u32(rowSquareSums[2]));
                src += stride;
            }
            valueSums[0] = ExtractSum64u(fullValueSums[0]);
            valueSums[1] = ExtractSum64u(fullValueSums[1]);
            valueSums[2] = ExtractSum64u(fullValueSums[2]);
            squareSums[0] = ExtractSum64u(fullSquareSums[0]);
            squareSums[1] = ExtractSum64u(fullSquareSums[1]);
            squareSums[2] = ExtractSum64u(fullSquareSums[2]);
        }

        void ValueSquareSums3(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            if (Aligned(src) && Aligned(stride))
                ValueSquareSums3<true>(src, stride, width, height, valueSums, squareSums);
            else
                ValueSquareSums3<false>(src, stride, width, height, valueSums, squareSums);
        }

        template <bool align> void ValueSquareSums4(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t widthA = Simd::AlignLo(width, A);
            uint8x16_t tail = ShiftLeft(K8_FF, A - width + widthA);
            uint64x2_t fullValueSums[4] = { K64_0000000000000000, K64_0000000000000000, K64_0000000000000000, K64_0000000000000000 };
            uint64x2_t fullSquareSums[4] = { K64_0000000000000000, K64_0000000000000000, K64_0000000000000000, K64_0000000000000000 };
            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowValueSums[4] = { K32_00000000, K32_00000000, K32_00000000, K32_00000000 };
                uint32x4_t rowSquareSums[4] = { K32_00000000, K32_00000000, K32_00000000, K32_00000000 };
                for (size_t col = 0, offs = 0; col < widthA; col += A, offs += A * 4)
                {
                    uint8x16x4_t _src = Load4<align>(src + offs);
                    rowValueSums[0] = vpadalq_u16(rowValueSums[0], vpaddlq_u8(_src.val[0]));
                    rowValueSums[1] = vpadalq_u16(rowValueSums[1], vpaddlq_u8(_src.val[1]));
                    rowValueSums[2] = vpadalq_u16(rowValueSums[2], vpaddlq_u8(_src.val[2]));
                    rowValueSums[3] = vpadalq_u16(rowValueSums[3], vpaddlq_u8(_src.val[3]));
                    rowSquareSums[0] = vaddq_u32(rowSquareSums[0], Square(_src.val[0]));
                    rowSquareSums[1] = vaddq_u32(rowSquareSums[1], Square(_src.val[1]));
                    rowSquareSums[2] = vaddq_u32(rowSquareSums[2], Square(_src.val[2]));
                    rowSquareSums[3] = vaddq_u32(rowSquareSums[3], Square(_src.val[3]));
                }
                if (widthA != width)
                {
                    size_t offs = (width - A) * 4;
                    uint8x16x4_t _src = Load4<align>(src + offs);
                    _src.val[0] = vandq_u8(_src.val[0], tail);
                    _src.val[1] = vandq_u8(_src.val[1], tail);
                    _src.val[2] = vandq_u8(_src.val[2], tail);
                    _src.val[3] = vandq_u8(_src.val[3], tail);
                    rowValueSums[0] = vpadalq_u16(rowValueSums[0], vpaddlq_u8(_src.val[0]));
                    rowValueSums[1] = vpadalq_u16(rowValueSums[1], vpaddlq_u8(_src.val[1]));
                    rowValueSums[2] = vpadalq_u16(rowValueSums[2], vpaddlq_u8(_src.val[2]));
                    rowValueSums[3] = vpadalq_u16(rowValueSums[3], vpaddlq_u8(_src.val[3]));
                    rowSquareSums[0] = vaddq_u32(rowSquareSums[0], Square(_src.val[0]));
                    rowSquareSums[1] = vaddq_u32(rowSquareSums[1], Square(_src.val[1]));
                    rowSquareSums[2] = vaddq_u32(rowSquareSums[2], Square(_src.val[2]));
                    rowSquareSums[3] = vaddq_u32(rowSquareSums[3], Square(_src.val[3]));
                }
                fullValueSums[0] = vaddq_u64(fullValueSums[0], vpaddlq_u32(rowValueSums[0]));
                fullValueSums[1] = vaddq_u64(fullValueSums[1], vpaddlq_u32(rowValueSums[1]));
                fullValueSums[2] = vaddq_u64(fullValueSums[2], vpaddlq_u32(rowValueSums[2]));
                fullValueSums[3] = vaddq_u64(fullValueSums[3], vpaddlq_u32(rowValueSums[3]));
                fullSquareSums[0] = vaddq_u64(fullSquareSums[0], vpaddlq_u32(rowSquareSums[0]));
                fullSquareSums[1] = vaddq_u64(fullSquareSums[1], vpaddlq_u32(rowSquareSums[1]));
                fullSquareSums[2] = vaddq_u64(fullSquareSums[2], vpaddlq_u32(rowSquareSums[2]));
                fullSquareSums[3] = vaddq_u64(fullSquareSums[3], vpaddlq_u32(rowSquareSums[3]));
                src += stride;
            }
            valueSums[0] = ExtractSum64u(fullValueSums[0]);
            valueSums[1] = ExtractSum64u(fullValueSums[1]);
            valueSums[2] = ExtractSum64u(fullValueSums[2]);
            valueSums[3] = ExtractSum64u(fullValueSums[3]);
            squareSums[0] = ExtractSum64u(fullSquareSums[0]);
            squareSums[1] = ExtractSum64u(fullSquareSums[1]);
            squareSums[2] = ExtractSum64u(fullSquareSums[2]);
            squareSums[3] = ExtractSum64u(fullSquareSums[3]);
        }

        void ValueSquareSums4(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            if (Aligned(src) && Aligned(stride))
                ValueSquareSums4<true>(src, stride, width, height, valueSums, squareSums);
            else
                ValueSquareSums4<false>(src, stride, width, height, valueSums, squareSums);
        }

        void ValueSquareSums(const uint8_t* src, size_t stride, size_t width, size_t height, size_t channels, uint64_t* valueSums, uint64_t* squareSums)
        {
            switch (channels)
            {
            case 1: ValueSquareSum(src, stride, width, height, valueSums, squareSums); break;
            case 2: ValueSquareSums2(src, stride, width, height, valueSums, squareSums); break;
            case 3: ValueSquareSums3(src, stride, width, height, valueSums, squareSums); break;
            case 4: ValueSquareSums4(src, stride, width, height, valueSums, squareSums); break;
            default:
                assert(0);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE uint32x4_t Correlation(const uint8x16_t & a, const uint8x16_t & b)
        {
            uint16x8_t lo = vmull_u8(Half<0>(a), Half<0>(b));
            uint16x8_t hi = vmull_u8(Half<1>(a), Half<1>(b));
            return vaddq_u32(vpaddlq_u16(lo), vpaddlq_u16(hi));
        }

        template <bool align> void CorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);

            uint64x2_t fullSum = K64_0000000000000000;
            for (size_t row = 0; row < height; ++row)
            {
                uint32x4_t rowSum = K32_00000000;
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    uint8x16_t _a = Load<align>(a + col);
                    uint8x16_t _b = Load<align>(b + col);
                    rowSum = vaddq_u32(rowSum, Correlation(_a, _b));
                }
                if (alignedWidth != width)
                {
                    uint8x16_t _a = vandq_u8(Load<align>(a + width - A), tailMask);
                    uint8x16_t _b = vandq_u8(Load<align>(b + width - A), tailMask);
                    rowSum = vaddq_u32(rowSum, Correlation(_a, _b));
                }
                fullSum = vaddq_u64(fullSum, vpaddlq_u32(rowSum));
                a += aStride;
                b += bStride;
            }
            *sum = ExtractSum64u(fullSum);
        }

        void CorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
                CorrelationSum<true>(a, aStride, b, bStride, width, height, sum);
            else
                CorrelationSum<false>(a, aStride, b, bStride, width, height, sum);
        }
    }
#endif// SIMD_NEON_ENABLE
}
