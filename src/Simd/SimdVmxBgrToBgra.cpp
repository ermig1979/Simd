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
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        const v128_u8 K8_PERM_0 = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x02, 0x13, 0x03, 0x04, 0x05, 0x17, 0x06, 0x07, 0x08, 0x1B, 0x09, 0x0A, 0x0B, 0x1F);
        const v128_u8 K8_PERM_1 = SIMD_VEC_SETR_EPI8(0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B);
        const v128_u8 K8_PERM_2 = SIMD_VEC_SETR_EPI8(0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13);
        const v128_u8 K8_PERM_3 = SIMD_VEC_SETR_EPI8(0x04, 0x05, 0x06, 0x13, 0x07, 0x08, 0x09, 0x17, 0x0A, 0x0B, 0x0C, 0x1B, 0x0D, 0x0E, 0x0F, 0x1F);

        template <bool align, bool first> SIMD_INLINE void BgrToBgra(const Loader<align> & bgr, const v128_u8 & alpha, Storer<align> & bgra)
        {
            const v128_u8 bgr0 = Load<align, first>(bgr);
            const v128_u8 bgr1 = Load<align, false>(bgr);
            const v128_u8 bgr2 = Load<align, false>(bgr);
            Store<align, first>(bgra, vec_perm(bgr0, alpha, K8_PERM_0));
            Store<align, false>(bgra, vec_perm(vec_perm(bgr0, bgr1, K8_PERM_1), alpha, K8_PERM_0));
            Store<align, false>(bgra, vec_perm(vec_perm(bgr1, bgr2, K8_PERM_2), alpha, K8_PERM_3));
            Store<align, false>(bgra, vec_perm(bgr2, alpha, K8_PERM_3));
        }

        template <bool align> void BgrToBgra(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t alignedWidth = AlignLo(width, A);
            if (width == alignedWidth)
                alignedWidth -= A;

            const v128_u8 _alpha = SetU8(alpha);

            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _bgr(bgr);
                Storer<align> _bgra(bgra);
                BgrToBgra<align, true>(_bgr, _alpha, _bgra);
                for (size_t col = A; col < alignedWidth; col += A)
                    BgrToBgra<align, false>(_bgr, _alpha, _bgra);
                Flush(_bgra);

                if (width != alignedWidth)
                {
                    Loader<false> _bgr(bgr + 3 * (width - A));
                    Storer<false> _bgra(bgra + 4 * (width - A));
                    BgrToBgra<false, true>(_bgr, _alpha, _bgra);
                    Flush(_bgra);
                }

                bgra += bgraStride;
                bgr += bgrStride;
            }
        }

        void BgrToBgra(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToBgra<true>(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
            else
                BgrToBgra<false>(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
        }

        const v128_u8 K8_PERM_48 = SIMD_VEC_SETR_EPI8(0x01, 0x11, 0x03, 0x13, 0x05, 0x15, 0x07, 0x17, 0x09, 0x19, 0x0B, 0x1B, 0x0D, 0x1D, 0x0F, 0x1F);

        template <bool align, bool first>
        SIMD_INLINE void Bgr48pToBgra32(const uint8_t * blue, const uint8_t * green, const uint8_t * red, size_t offset,
            const v128_u8 & alpha, Storer<align> & bgra)
        {
            const v128_u8 _blue = Load<align>(blue + offset);
            const v128_u8 _green = Load<align>(green + offset);
            const v128_u8 _red = Load<align>(red + offset);

            v128_u16 bg = (v128_u16)vec_perm(_blue, _green, K8_PERM_48);
            v128_u16 ra = (v128_u16)vec_perm(_red, alpha, K8_PERM_48);

            Store<align, first>(bgra, (v128_u8)UnpackLoU16(ra, bg));
            Store<align, false>(bgra, (v128_u8)UnpackHiU16(ra, bg));
        }

        template <bool align> void Bgr48pToBgra32(const uint8_t * blue, size_t blueStride, size_t width, size_t height,
            const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= HA);
            if (align)
            {
                assert(Aligned(blue) && Aligned(blueStride));
                assert(Aligned(green) && Aligned(greenStride));
                assert(Aligned(red) && Aligned(redStride));
                assert(Aligned(bgra) && Aligned(bgraStride));
            }

            v128_u8 _alpha = SetU8(alpha);
            size_t alignedWidth = AlignLo(width, HA);
            for (size_t row = 0; row < height; ++row)
            {
                Storer<align> _bgra(bgra);
                Bgr48pToBgra32<align, true>(blue, green, red, 0, _alpha, _bgra);
                for (size_t col = HA; col < alignedWidth; col += HA)
                    Bgr48pToBgra32<align, false>(blue, green, red, col * 2, _alpha, _bgra);
                Flush(_bgra);

                if (width != alignedWidth)
                {
                    Storer<false> _bgra(bgra + (width - HA) * 4);
                    Bgr48pToBgra32<false, true>(blue, green, red, (width - HA) * 2, _alpha, _bgra);
                    Flush(_bgra);
                }

                blue += blueStride;
                green += greenStride;
                red += redStride;
                bgra += bgraStride;
            }
        }

        void Bgr48pToBgra32(const uint8_t * blue, size_t blueStride, size_t width, size_t height,
            const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(blue) && Aligned(blueStride) && Aligned(green) && Aligned(greenStride) &&
                Aligned(red) && Aligned(redStride) && Aligned(bgra) && Aligned(bgraStride))
                Bgr48pToBgra32<true>(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
            else
                Bgr48pToBgra32<false>(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
        }
    }
#endif// SIMD_VMX_ENABLE
}
