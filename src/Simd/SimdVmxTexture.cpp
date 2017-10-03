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
#include "Simd/SimdBase.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        SIMD_INLINE v128_s16 TextureBoostedSaturatedGradient16(v128_s16 a, v128_s16 b, const v128_s16 & saturation, const v128_s16 & boost)
        {
            return vec_mladd(vec_max((v128_s16)K16_0000, vec_add(saturation, vec_min(vec_sub(b, a), saturation))), boost, (v128_s16)K16_0000);
        }

        SIMD_INLINE v128_u8 TextureBoostedSaturatedGradient8(v128_u8 a, v128_u8 b, const v128_s16 & saturation, const v128_s16 & boost)
        {
            v128_s16 lo = TextureBoostedSaturatedGradient16((v128_s16)UnpackLoU8(a), (v128_s16)UnpackLoU8(b), saturation, boost);
            v128_s16 hi = TextureBoostedSaturatedGradient16((v128_s16)UnpackHiU8(a), (v128_s16)UnpackHiU8(b), saturation, boost);
            return vec_packsu(lo, hi);
        }

        template<bool align, bool first>
        SIMD_INLINE void TextureBoostedSaturatedGradient(const uint8_t * src, size_t stride, const v128_s16 & saturation, const v128_s16 & boost,
            Storer<align> & dx, Storer<align> & dy)
        {
            Store<align, first>(dx, TextureBoostedSaturatedGradient8(Load<false>(src - 1), Load<false>(src + 1), saturation, boost));
            Store<align, first>(dy, TextureBoostedSaturatedGradient8(Load<align>(src - stride), Load<align>(src + stride), saturation, boost));
        }

        template<bool align>
        SIMD_INLINE void TextureBoostedSaturatedGradient(const uint8_t * src, size_t stride, const v128_s16 & saturation, const v128_s16 & boost,
            uint8_t * dx, uint8_t * dy, size_t offset)
        {
            const uint8_t * s = src + offset;
            Store<align>(dx + offset, TextureBoostedSaturatedGradient8(Load<false>(s - 1), Load<false>(s + 1), saturation, boost));
            Store<align>(dy + offset, TextureBoostedSaturatedGradient8(Load<align>(s - stride), Load<align>(s + stride), saturation, boost));
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
            size_t fullAlignedWidth = AlignLo(width, DA);
            v128_s16 _saturation = SIMD_VEC_SET1_EPI16(saturation);
            v128_s16 _boost = SIMD_VEC_SET1_EPI16(boost);

            memset(dx, 0, width);
            memset(dy, 0, width);
            src += srcStride;
            dx += dxStride;
            dy += dyStride;
            for (size_t row = 2; row < height; ++row)
            {
                if (align)
                {
                    size_t col = 0;
                    for (; col < fullAlignedWidth; col += DA)
                    {
                        TextureBoostedSaturatedGradient<align>(src, srcStride, _saturation, _boost, dx, dy, col);
                        TextureBoostedSaturatedGradient<align>(src, srcStride, _saturation, _boost, dx, dy, col + A);
                    }
                    for (; col < alignedWidth; col += A)
                        TextureBoostedSaturatedGradient<align>(src, srcStride, _saturation, _boost, dx, dy, col);
                }
                else
                {
                    Storer<align> _dx(dx), _dy(dy);
                    TextureBoostedSaturatedGradient<align, true>(src, srcStride, _saturation, _boost, _dx, _dy);
                    for (size_t col = A; col < alignedWidth; col += A)
                        TextureBoostedSaturatedGradient<align, false>(src + col, srcStride, _saturation, _boost, _dx, _dy);
                    Flush(_dx, _dy);
                }
                if (width != alignedWidth)
                    TextureBoostedSaturatedGradient<false>(src, srcStride, _saturation, _boost, dx, dy, width - A);

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

        template<bool align, bool first>
        SIMD_INLINE void TextureBoostedUv(const uint8_t * src, const v128_u8 & min, const v128_u8 & max, const v128_u16 & boost, Storer<align> & dst)
        {
            const v128_u8 _src = Load<align>(src);
            const v128_u8 saturated = vec_subs(vec_min(_src, max), min);
            const v128_u16 lo = vec_mladd(UnpackLoU8(saturated), boost, K16_0000);
            const v128_u16 hi = vec_mladd(UnpackHiU8(saturated), boost, K16_0000);
            Store<align, first>(dst, vec_packsu(lo, hi));
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

            v128_u8 _min = SetU8(min);
            v128_u8 _max = SetU8(max);
            v128_u16 _boost = SetU16(boost);

            for (size_t row = 0; row < height; ++row)
            {
                Storer<align> _dst(dst);
                TextureBoostedUv<align, true>(src, _min, _max, _boost, _dst);
                for (size_t col = A; col < alignedWidth; col += A)
                    TextureBoostedUv<align, false>(src + col, _min, _max, _boost, _dst);
                Flush(_dst);

                if (width != alignedWidth)
                {
                    Storer<false> _dst(dst + width - A);
                    TextureBoostedUv<false, true>(src + width - A, _min, _max, _boost, _dst);
                    Flush(_dst);
                }

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

        template <bool align> SIMD_INLINE void TextureGetDifferenceSum(const uint8_t * src, const uint8_t * lo, const uint8_t * hi,
            v128_u32 & positive, v128_u32 & negative, const v128_u8 & mask)
        {
            const v128_u8 _src = Load<align>(src);
            const v128_u8 _lo = Load<align>(lo);
            const v128_u8 _hi = Load<align>(hi);
            const v128_u8 average = vec_and(mask, vec_avg(_lo, _hi));
            const v128_u8 current = vec_and(mask, _src);
            positive = vec_msum(vec_subs(current, average), K8_01, positive);
            negative = vec_msum(vec_subs(average, current), K8_01, negative);
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
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                v128_u32 positive = K32_00000000;
                v128_u32 negative = K32_00000000;

                for (size_t col = 0; col < alignedWidth; col += A)
                    TextureGetDifferenceSum<align>(src + col, lo + col, hi + col, positive, negative, K8_FF);
                if (width != alignedWidth)
                    TextureGetDifferenceSum<false>(src + width - A, lo + width - A, hi + width - A, positive, negative, tailMask);

                *sum += ExtractSum(positive);
                *sum -= ExtractSum(negative);

                src += srcStride;
                lo += loStride;
                hi += hiStride;
            }
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
            {
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = src == dst ? ShiftLeft(K8_FF, A - width + alignedWidth) : K8_FF;
            if (shift > 0)
            {
                v128_u8 _shift = SetU8(shift);
                for (size_t row = 0; row < height; ++row)
                {
                    Storer<align> _dst(dst);
                    _dst.First(vec_adds(Load<align>(src), _shift));
                    for (size_t col = A; col < alignedWidth; col += A)
                        _dst.Next(vec_adds(Load<align>(src + col), _shift));
                    Flush(_dst);
                    if (width != alignedWidth)
                    {
                        const v128_u8 _src = Load<false>(src + width - A);
                        Store<false>(dst + width - A, vec_adds(_src, vec_and(_shift, tailMask)));
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }
            if (shift < 0)
            {
                v128_u8 _shift = SetU8(-shift);
                for (size_t row = 0; row < height; ++row)
                {
                    Storer<align> _dst(dst);
                    _dst.First(vec_subs(Load<align>(src), _shift));
                    for (size_t col = A; col < alignedWidth; col += A)
                        _dst.Next(vec_subs(Load<align>(src + col), _shift));
                    Flush(_dst);
                    if (width != alignedWidth)
                    {
                        const v128_u8 _src = Load<false>(src + width - A);
                        Store<false>(dst + width - A, vec_subs(_src, vec_and(_shift, tailMask)));
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
#endif// SIMD_VMX_ENABLE
}
