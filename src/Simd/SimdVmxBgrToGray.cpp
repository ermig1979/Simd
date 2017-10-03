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
#include "Simd/SimdStore.h"
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        const v128_u16 K16_BLUE_RED = SIMD_VEC_SET2_EPI16(Base::BLUE_TO_GRAY_WEIGHT, Base::RED_TO_GRAY_WEIGHT);
        const v128_u16 K16_GREEN_0000 = SIMD_VEC_SET2_EPI16(Base::GREEN_TO_GRAY_WEIGHT, 0x0000);
        const v128_u32 K32_ROUND_TERM = SIMD_VEC_SET1_EPI32(Base::BGR_TO_GRAY_ROUND_TERM);
        const v128_u32 K32_SHIFT = SIMD_VEC_SET1_EPI32(Base::BGR_TO_GRAY_AVERAGING_SHIFT);

        const v128_u8 K8_PERM_0 = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x02, 0x00, 0x03, 0x04, 0x05, 0x00, 0x06, 0x07, 0x08, 0x00, 0x09, 0x0A, 0x0B, 0x00);
        const v128_u8 K8_PERM_1 = SIMD_VEC_SETR_EPI8(0x0C, 0x0D, 0x0E, 0x00, 0x0F, 0x10, 0x11, 0x00, 0x12, 0x13, 0x14, 0x00, 0x15, 0x16, 0x17, 0x00);
        const v128_u8 K8_PERM_2 = SIMD_VEC_SETR_EPI8(0x08, 0x09, 0x0A, 0x00, 0x0B, 0x0C, 0x0D, 0x00, 0x0E, 0x0F, 0x10, 0x00, 0x11, 0x12, 0x13, 0x00);
        const v128_u8 K8_PERM_3 = SIMD_VEC_SETR_EPI8(0x14, 0x15, 0x16, 0x00, 0x17, 0x18, 0x19, 0x00, 0x1A, 0x1B, 0x1C, 0x00, 0x1D, 0x1E, 0x1F, 0x00);

        SIMD_INLINE v128_u32 BgraToGray32(v128_u8 bgra)
        {
            const v128_u16 _b_r = vec_mule(bgra, K8_01);
            const v128_u16 _g_a = vec_mulo(bgra, K8_01);
            const v128_u32 weightedSum = vec_add(vec_mule(_g_a, K16_GREEN_0000),
                vec_add(vec_mule(_b_r, K16_BLUE_RED), vec_mulo(_b_r, K16_BLUE_RED)));
            return vec_sr(vec_add(weightedSum, K32_ROUND_TERM), K32_SHIFT);
        }

        template<bool align, bool first>
        SIMD_INLINE void BgrToGray(const Loader<align> & bgr, Storer<align> & gray)
        {
            v128_u8 _bgr[3];
            _bgr[0] = Load<align, first>(bgr);
            _bgr[1] = Load<align, false>(bgr);
            _bgr[2] = Load<align, false>(bgr);

            const v128_u16 lo = vec_packsu(
                BgraToGray32(vec_perm(_bgr[0], _bgr[1], K8_PERM_0)),
                BgraToGray32(vec_perm(_bgr[0], _bgr[1], K8_PERM_1)));
            const v128_u16 hi = vec_packsu(
                BgraToGray32(vec_perm(_bgr[1], _bgr[2], K8_PERM_2)),
                BgraToGray32(vec_perm(_bgr[1], _bgr[2], K8_PERM_3)));
            Store<align, first>(gray, vec_packsu(lo, hi));
        }

        template <bool align> void BgrToGray(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * gray, size_t grayStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(gray) && Aligned(grayStride));

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _bgr(bgr);
                Storer<align> _gray(gray);
                BgrToGray<align, true>(_bgr, _gray);
                for (size_t col = A; col < alignedWidth; col += A)
                    BgrToGray<align, false>(_bgr, _gray);
                Flush(_gray);

                if (alignedWidth != width)
                {
                    Loader<false> _bgr(bgr + 3 * (width - A));
                    Storer<false> _gray(gray + width - A);
                    BgrToGray<false, true>(_bgr, _gray);
                    Flush(_gray);
                }

                bgr += bgrStride;
                gray += grayStride;
            }
        }

        void BgrToGray(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * gray, size_t grayStride)
        {
            if (Aligned(bgr) && Aligned(gray) && Aligned(bgrStride) && Aligned(grayStride))
                BgrToGray<true>(bgr, width, height, bgrStride, gray, grayStride);
            else
                BgrToGray<false>(bgr, width, height, bgrStride, gray, grayStride);
        }
    }
#endif// SIMD_VMX_ENABLE
}
