/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
        SIMD_INLINE uint8x8_t Average(const uint8x16_t & s0, const uint8x16_t & s1)
        {
            return vshrn_n_u16(vaddq_u16(vaddq_u16(vpaddlq_u8(s0), vpaddlq_u8(s1)), vdupq_n_u16(2)), 2);
        }

        SIMD_INLINE uint8x16_t Average(const uint8x16_t & s00, const uint8x16_t & s01, const uint8x16_t & s10, const uint8x16_t & s11)
        {
            return vcombine_u8(Average(s00, s10), Average(s01, s11));
        }

        template <size_t channelCount> struct Color2x2
        {
            template <bool align> static SIMD_INLINE void Reduce(const uint8_t * src0, const uint8_t * src1, uint8_t * dst)
            {
                assert(0);
            }
        };

        template<> struct Color2x2<1>
        {
            template <bool align> static SIMD_INLINE void Reduce(const uint8_t * src0, const uint8_t * src1, uint8_t * dst)
            {
                Store<align>(dst, Average(Load<align>(src0 + 0), Load<align>(src0 + A), Load<align>(src1 + 0), Load<align>(src1 + A)));
            }
        };

        template<> struct Color2x2<2>
        {
            template <bool align> static SIMD_INLINE void Reduce(const uint8_t * src0, const uint8_t * src1, uint8_t * dst)
            {
                uint8x16x2_t s00 = Load2<align>(src0 + 0 * A);
                uint8x16x2_t s01 = Load2<align>(src0 + 2 * A);
                uint8x16x2_t s10 = Load2<align>(src1 + 0 * A);
                uint8x16x2_t s11 = Load2<align>(src1 + 2 * A);
                uint8x16x2_t d;
                d.val[0] = Average(s00.val[0], s01.val[0], s10.val[0], s11.val[0]);
                d.val[1] = Average(s00.val[1], s01.val[1], s10.val[1], s11.val[1]);
                Store2<align>(dst, d);
            }
        };

        template<> struct Color2x2<3>
        {
            template <bool align> static SIMD_INLINE void Reduce(const uint8_t * src0, const uint8_t * src1, uint8_t * dst)
            {
                uint8x16x3_t s0, s1;
                uint8x8x3_t d;
                s0 = Load3<align>(src0);
                s1 = Load3<align>(src1);
                d.val[0] = Average(s0.val[0], s1.val[0]);
                d.val[1] = Average(s0.val[1], s1.val[1]);
                d.val[2] = Average(s0.val[2], s1.val[2]);
                Store3<align>(dst, d);
                s0 = Load3<align>(src0 + 3*A);
                s1 = Load3<align>(src1 + 3*A);
                d.val[0] = Average(s0.val[0], s1.val[0]);
                d.val[1] = Average(s0.val[1], s1.val[1]);
                d.val[2] = Average(s0.val[2], s1.val[2]);
                Store3<align>(dst + 3*HA, d);
            }
        };

        template<> struct Color2x2<4>
        {
            template <bool align> static SIMD_INLINE void Reduce(const uint8_t * src0, const uint8_t * src1, uint8_t * dst)
            {
                uint8x16x4_t s0, s1;
                uint8x8x4_t d;
                s0 = Load4<align>(src0);
                s1 = Load4<align>(src1);
                d.val[0] = Average(s0.val[0], s1.val[0]);
                d.val[1] = Average(s0.val[1], s1.val[1]);
                d.val[2] = Average(s0.val[2], s1.val[2]);
                d.val[3] = Average(s0.val[3], s1.val[3]);
                Store4<align>(dst, d);
                s0 = Load4<align>(src0 + 4 * A);
                s1 = Load4<align>(src1 + 4 * A);
                d.val[0] = Average(s0.val[0], s1.val[0]);
                d.val[1] = Average(s0.val[1], s1.val[1]);
                d.val[2] = Average(s0.val[2], s1.val[2]);
                d.val[3] = Average(s0.val[3], s1.val[3]);
                Store4<align>(dst + 4 * HA, d);
            }
        };

        template <size_t channelCount, bool align> void ReduceColor2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t evenWidth = AlignLo(srcWidth, 2);
            size_t alignedWidth = AlignLo(srcWidth, A);
            size_t evenSize = evenWidth * channelCount;
            size_t alignedSize = alignedWidth * channelCount;
            size_t srcStep = DA * channelCount, dstStep = A * channelCount;
            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t *src0 = src;
                const uint8_t *src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                size_t srcOffset = 0, dstOffset = 0;
                for (; srcOffset < alignedSize; srcOffset += srcStep, dstOffset += dstStep)
                    Color2x2<channelCount>::template Reduce<align>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset);
                if (alignedSize != evenSize)
                {
                    srcOffset = evenSize - srcStep;
                    dstOffset = srcOffset / 2;
                    Color2x2<channelCount>::template Reduce<false>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset);
                }
                if (evenWidth != srcWidth)
                {
                    for (size_t c = 0; c < channelCount; ++c)
                        dst[evenSize / 2 + c] = Base::Average(src0[evenSize + c], src1[evenSize + c]);
                }
                src += 2 * srcStride;
                dst += dstStride;
            }
        }

        template <bool align> void ReduceColor2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth >= DA);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(dst) && Aligned(dstStride));
            }

            switch (channelCount)
            {
            case 1: ReduceColor2x2<1, align>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 2: ReduceColor2x2<2, align>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 3: ReduceColor2x2<3, align>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 4: ReduceColor2x2<4, align>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            default: assert(0);
            }
        }

        void ReduceColor2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                ReduceColor2x2<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
            else
                ReduceColor2x2<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
        }
    }
#endif// SIMD_NEON_ENABLE
}
