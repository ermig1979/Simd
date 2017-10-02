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
        SIMD_INLINE uint8x16_t TextureBoostedSaturatedGradient(const uint8x16_t & a, const uint8x16_t & b, const uint8x16_t & saturation, const uint8x16_t & boost)
        {
            uint8x16_t p = vminq_u8(vqsubq_u8(b, a), saturation);
            uint8x16_t n = vminq_u8(vqsubq_u8(a, b), saturation);
            return vmulq_u8(vsubq_u8(vaddq_u8(saturation, p), n), boost);
        }

        template<bool align> SIMD_INLINE void TextureBoostedSaturatedGradient(const uint8_t * src, size_t stride, uint8_t * dx, uint8_t * dy,
            const uint8x16_t & saturation, const uint8x16_t & boost)
        {
            Store<align>(dx, TextureBoostedSaturatedGradient(Load<false>(src - 1), Load<false>(src + 1), saturation, boost));
            Store<align>(dy, TextureBoostedSaturatedGradient(Load<align>(src - stride), Load<align>(src + stride), saturation, boost));
        }

        template<bool align> void TextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride)
        {
            assert(width >= A && int(2)*saturation*boost <= 0xFF);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dx) && Aligned(dxStride) && Aligned(dy) && Aligned(dyStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            uint8x16_t _saturation = vdupq_n_u8(saturation);
            uint8x16_t _boost = vdupq_n_u8(boost);

            memset(dx, 0, width);
            memset(dy, 0, width);
            src += srcStride;
            dx += dxStride;
            dy += dyStride;
            for (size_t row = 2; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    TextureBoostedSaturatedGradient<align>(src + col, srcStride, dx + col, dy + col, _saturation, _boost);
                if (width != alignedWidth)
                    TextureBoostedSaturatedGradient<false>(src + width - A, srcStride, dx + width - A, dy + width - A, _saturation, _boost);

                dx[0] = 0;
                dy[0] = 0;
                dx[width - 1] = 0;
                dy[width - 1] = 0;

                src += srcStride;
                dx += dxStride;
                dy += dyStride;
            }
            memset(dx, 0, width);
            memset(dy, 0, width);
        }

        void TextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dx) && Aligned(dxStride) && Aligned(dy) && Aligned(dyStride))
                TextureBoostedSaturatedGradient<true>(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
            else
                TextureBoostedSaturatedGradient<false>(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
        }

        template<bool align> SIMD_INLINE void TextureBoostedUv(const uint8_t * src, uint8_t * dst, uint8x16_t min, uint8x16_t max, uint8x16_t boost)
        {
            Store<align>(dst, vmulq_u8(vsubq_u8(vmaxq_u8(min, vminq_u8(max, Load<align>(src))), min), boost));
        }

        template<bool align> void TextureBoostedUv(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t boost, uint8_t * dst, size_t dstStride)
        {
            assert(width >= A && boost < 0x80);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = AlignLo(width, A);
            int min = 128 - (128 / boost);
            int max = 255 - min;

            uint8x16_t _min = vdupq_n_u8(min);
            uint8x16_t _max = vdupq_n_u8(max);
            uint8x16_t _boost = vdupq_n_u8(boost);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    TextureBoostedUv<align>(src + col, dst + col, _min, _max, _boost);
                if (width != alignedWidth)
                    TextureBoostedUv<false>(src + width - A, dst + width - A, _min, _max, _boost);
                src += srcStride;
                dst += dstStride;
            }
        }

        void TextureBoostedUv(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t boost, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                TextureBoostedUv<true>(src, srcStride, width, height, boost, dst, dstStride);
            else
                TextureBoostedUv<false>(src, srcStride, width, height, boost, dst, dstStride);
        }

        template <bool align> SIMD_INLINE int16x8_t TextureGetDifferenceSum(const uint8_t * src, const uint8_t * lo, const uint8_t * hi, size_t offset)
        {
            uint8x16_t _src = Load<align>(src + offset);
            uint8x16_t avg = vrhaddq_u8(Load<align>(lo + offset), Load<align>(hi + offset));
            return (int16x8_t)vsubq_u16(vpaddlq_u8(_src), vpaddlq_u8(avg));
        }

        template <bool align> SIMD_INLINE int16x8_t TextureGetDifferenceSum(const uint8_t * src, const uint8_t * lo, const uint8_t * hi, size_t offset, const uint8x16_t & mask)
        {
            uint8x16_t _src = vandq_u8(Load<align>(src + offset), mask);
            uint8x16_t avg = vandq_u8(vrhaddq_u8(Load<align>(lo + offset), Load<align>(hi + offset)), mask);
            return (int16x8_t)vsubq_u16(vpaddlq_u8(_src), vpaddlq_u8(avg));
        }

        template <bool align> void TextureGetDifferenceSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride, int64_t * sum)
        {
            assert(width >= A && sum != NULL);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t blockSize = A << 6;
            size_t blockCount = (alignedWidth >> 6) + 1;

            int64x2_t _sum = (int64x2_t)K64_0000000000000000;
            for (size_t row = 0; row < height; ++row)
            {
                int32x4_t rowSum = (int32x4_t)K32_00000000;
                for (size_t block = 0; block < blockCount; ++block)
                {
                    int16x8_t blockSum = (int16x8_t)K16_0000;
                    for (size_t col = block*blockSize, end = Min(col + blockSize, alignedWidth); col < end; col += A)
                        blockSum = vaddq_s16(blockSum, TextureGetDifferenceSum<align>(src, lo, hi, col));
                    rowSum = vaddq_s32(rowSum, vpaddlq_s16(blockSum));
                }
                if (alignedWidth != width)
                    rowSum = vaddq_s32(rowSum, vpaddlq_s16(TextureGetDifferenceSum<false>(src, lo, hi, width - A, tailMask)));
                _sum = vaddq_s64(_sum, vpaddlq_s32(rowSum));
                src += srcStride;
                lo += loStride;
                hi += hiStride;
            }
            *sum = ExtractSum64i(_sum);
        }

        void TextureGetDifferenceSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride, int64_t * sum)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(lo) && Aligned(loStride) && Aligned(hi) && Aligned(hiStride))
                TextureGetDifferenceSum<true>(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
            else
                TextureGetDifferenceSum<false>(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
        }

        template <bool align> void TexturePerformCompensation(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            int shift, uint8_t * dst, size_t dstStride)
        {
            assert(width >= A && shift > -0xFF && shift < 0xFF && shift != 0);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = AlignLo(width, A);
            uint8x16_t tailMask = src == dst ? ShiftLeft(K8_FF, A - width + alignedWidth) : K8_FF;
            if (shift > 0)
            {
                uint8x16_t _shift = vdupq_n_u8(shift);
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += A)
                    {
                        uint8x16_t _src = Load<align>(src + col);
                        Store<align>(dst + col, vqaddq_u8(_src, _shift));
                    }
                    if (width != alignedWidth)
                    {
                        uint8x16_t _src = Load<false>(src + width - A);
                        Store<false>(dst + width - A, vqaddq_u8(_src, vandq_u8(_shift, tailMask)));
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }
            if (shift < 0)
            {
                uint8x16_t _shift = vdupq_n_u8(-shift);
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += A)
                    {
                        uint8x16_t _src = Load<align>(src + col);
                        Store<align>(dst + col, vqsubq_u8(_src, _shift));
                    }
                    if (width != alignedWidth)
                    {
                        uint8x16_t _src = Load<false>(src + width - A);
                        Store<false>(dst + width - A, vqsubq_u8(_src, vandq_u8(_shift, tailMask)));
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }
        }

        void TexturePerformCompensation(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            int shift, uint8_t * dst, size_t dstStride)
        {
            if (shift == 0)
            {
                if (src != dst)
                    Base::Copy(src, srcStride, width, height, 1, dst, dstStride);
                return;
            }
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                TexturePerformCompensation<true>(src, srcStride, width, height, shift, dst, dstStride);
            else
                TexturePerformCompensation<false>(src, srcStride, width, height, shift, dst, dstStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}
