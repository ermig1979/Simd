/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar,
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

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
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

        SIMD_INLINE void AddSum16(const uint8x16_t & src, uint16x8_t & sum0, uint16x8_t & sum1)
        {
            sum0 = vaddw_u8(sum0, vget_low_u8(src));
            sum1 = vaddw_u8(sum1, vget_high_u8(src));
        }

        template <bool align> SIMD_INLINE void Sum16x1(const uint8_t * src, size_t, uint16_t * dst)
        {
            uint16x8_t sum0 = Load<true>(dst + 0);
            uint16x8_t sum1 = Load<true>(dst + HA);
            AddSum16(Load<align>(src), sum0, sum1);
            Store<true>(dst + 0, sum0);
            Store<true>(dst + HA, sum1);
        }

        template <bool align> SIMD_INLINE void Sum16x4(const uint8_t * src, size_t stride, uint16_t * dst)
        {
            uint16x8_t sum0 = Load<true>(dst + 0);
            uint16x8_t sum1 = Load<true>(dst + HA);
            AddSum16(Load<align>(src + 0 * stride), sum0, sum1);
            AddSum16(Load<align>(src + 1 * stride), sum0, sum1);
            AddSum16(Load<align>(src + 2 * stride), sum0, sum1);
            AddSum16(Load<align>(src + 3 * stride), sum0, sum1);
            Store<true>(dst + 0, sum0);
            Store<true>(dst + HA, sum1);
        }

        template <bool align> SIMD_INLINE void Sum16x8(const uint8_t * src, size_t stride, uint16_t * dst)
        {
            uint16x8_t sum0 = Load<true>(dst + 0);
            uint16x8_t sum1 = Load<true>(dst + HA);
            AddSum16(Load<align>(src + 0 * stride), sum0, sum1);
            AddSum16(Load<align>(src + 1 * stride), sum0, sum1);
            AddSum16(Load<align>(src + 2 * stride), sum0, sum1);
            AddSum16(Load<align>(src + 3 * stride), sum0, sum1);
            AddSum16(Load<align>(src + 4 * stride), sum0, sum1);
            AddSum16(Load<align>(src + 5 * stride), sum0, sum1);
            AddSum16(Load<align>(src + 6 * stride), sum0, sum1);
            AddSum16(Load<align>(src + 7 * stride), sum0, sum1);
            Store<true>(dst + 0, sum0);
            Store<true>(dst + HA, sum1);
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

        SIMD_INLINE void Sum16To32(const uint16_t * src, uint32_t * dst)
        {
            uint32x4_t sum0 = Load<true>(dst + 0);
            uint32x4_t sum1 = Load<true>(dst + 4);
            uint16x8_t src0 = Load<true>(src);
            Store<true>(dst + 0, vaddw_u16(sum0, vget_low_u16(src0)));
            Store<true>(dst + 4, vaddw_u16(sum1, vget_high_u16(src0)));
        }

        template <bool align> void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            size_t stepSize = 256;
            size_t stepCount = DivHi(height, stepSize);

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint32_t)*alignedHiWidth);
            for (size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);
                size_t rowEnd4 = AlignLo(rowEnd, 4);
                size_t rowEnd8 = AlignLo(rowEnd, 8);

                memset(buffer.sums16, 0, sizeof(uint16_t)*alignedHiWidth);
                size_t row = rowStart;
                for (; row < rowEnd8; row += 8)
                {
                    for (size_t col = 0; col < alignedLoWidth; col += A)
                        Sum16x8<align>(src + col, stride, buffer.sums16 + col);
                    if (alignedLoWidth != width)
                        Sum16x8<false>(src + width - A, stride, buffer.sums16 + alignedLoWidth);
                    src += 8 * stride;
                }
                for (; row < rowEnd4; row += 4)
                {
                    for (size_t col = 0; col < alignedLoWidth; col += A)
                        Sum16x4<align>(src + col, stride, buffer.sums16 + col);
                    if (alignedLoWidth != width)
                        Sum16x4<false>(src + width - A, stride, buffer.sums16 + alignedLoWidth);
                    src += 4 * stride;
                }
                for (; row < rowEnd; ++row)
                {
                    for (size_t col = 0; col < alignedLoWidth; col += A)
                        Sum16x1<align>(src + col, stride, buffer.sums16 + col);
                    if (alignedLoWidth != width)
                        Sum16x1<false>(src + width - A, stride, buffer.sums16 + alignedLoWidth);
                    src += stride;
                }

                for (size_t col = 0; col < alignedHiWidth; col += HA)
                    Sum16To32(buffer.sums16 + col, buffer.sums32 + col);
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t)*alignedLoWidth);
            if (alignedLoWidth != width)
                memcpy(sums + alignedLoWidth, buffer.sums32 + alignedLoWidth + alignedHiWidth - width, sizeof(uint32_t)*(width - alignedLoWidth));
        }

        void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetColSums<true>(src, stride, width, height, sums);
            else
                GetColSums<false>(src, stride, width, height, sums);
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
    }
#endif// SIMD_NEON_ENABLE
}
