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

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        const v128_u8 K8_PERM_GR_0 = SIMD_VEC_SETR_EPI8(0x01, 0x05, 0x07, 0x0B, 0x0D, 0x11, 0x13, 0x17, 0x19, 0x1D, 0x1F, 0x00, 0x00, 0x00, 0x00, 0x00);
        const v128_u8 K8_PERM_GR_1 = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x13, 0x15, 0x19, 0x1B, 0x1F);

        const v128_u8 K8_PERM_BG_0 = SIMD_VEC_SETR_EPI8(0x00, 0x04, 0x06, 0x0A, 0x0C, 0x10, 0x12, 0x16, 0x18, 0x1C, 0x1E, 0x00, 0x00, 0x00, 0x00, 0x00);
        const v128_u8 K8_PERM_BG_1 = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x12, 0x14, 0x18, 0x1A, 0x1E);

        const v128_u8 K8_PERM_GB_0 = SIMD_VEC_SETR_EPI8(0x01, 0x03, 0x07, 0x09, 0x0D, 0x0F, 0x13, 0x15, 0x19, 0x1B, 0x1F, 0x00, 0x00, 0x00, 0x00, 0x00);
        const v128_u8 K8_PERM_GB_1 = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x11, 0x15, 0x17, 0x1B, 0x1D);

        const v128_u8 K8_PERM_RG_0 = SIMD_VEC_SETR_EPI8(0x02, 0x04, 0x08, 0x0A, 0x0E, 0x10, 0x14, 0x16, 0x1A, 0x1C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        const v128_u8 K8_PERM_RG_1 = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x10, 0x12, 0x16, 0x18, 0x1C, 0x1E);

        template <int format, int row, bool align, bool first>
        SIMD_INLINE void BgrToBayer(const Loader<align> & bgr, const v128_u8 perm[4][2][2], Storer<align> & bayer)
        {
            const v128_u8 bgr0 = Load<align, first>(bgr);
            const v128_u8 bgr1 = Load<align, false>(bgr);
            const v128_u8 bgr2 = Load<align, false>(bgr);
            Store<align, first>(bayer, vec_perm(vec_perm(bgr0, bgr1, perm[format][row][0]), bgr2, perm[format][row][1]));
        }

        template <int format, bool align>
        void BgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(bayer) && Aligned(bayerStride));

            size_t alignedWidth = AlignLo(width, A);

            const v128_u8 perm[4][2][2] =
            {
                {{K8_PERM_GR_0, K8_PERM_GR_1}, {K8_PERM_BG_0, K8_PERM_BG_1}},
                {{K8_PERM_GB_0, K8_PERM_GB_1}, {K8_PERM_RG_0, K8_PERM_RG_1}},
                {{K8_PERM_RG_0, K8_PERM_RG_1}, {K8_PERM_GB_0, K8_PERM_GB_1}},
                {{K8_PERM_BG_0, K8_PERM_BG_1}, {K8_PERM_GR_0, K8_PERM_GR_1}}
            };

            for (size_t row = 0; row < height; row += 2)
            {
                Loader<align> _bgr0(bgr);
                Storer<align> _bayer0(bayer);
                BgrToBayer<format, 0, align, true>(_bgr0, perm, _bayer0);
                for (size_t col = A; col < alignedWidth; col += A)
                    BgrToBayer<format, 0, align, false>(_bgr0, perm, _bayer0);
                Flush(_bayer0);

                if (width != alignedWidth)
                {
                    Loader<false> _bgr(bgr + 3 * (width - A));
                    Storer<false> _bayer(bayer + width - A);
                    BgrToBayer<format, 0, false, true>(_bgr, perm, _bayer);
                    Flush(_bayer);
                }

                bgr += bgrStride;
                bayer += bayerStride;

                Loader<align> _bgr1(bgr);
                Storer<align> _bayer1(bayer);
                BgrToBayer<format, 1, align, true>(_bgr1, perm, _bayer1);
                for (size_t col = A; col < alignedWidth; col += A)
                    BgrToBayer<format, 1, align, false>(_bgr1, perm, _bayer1);
                Flush(_bayer1);

                if (width != alignedWidth)
                {
                    Loader<false> _bgr(bgr + 3 * (width - A));
                    Storer<false> _bayer(bayer + width - A);
                    BgrToBayer<format, 1, false, true>(_bgr, perm, _bayer);
                    Flush(_bayer);
                }

                bgr += bgrStride;
                bayer += bayerStride;
            }
        }

        template<bool align>
        void BgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BgrToBayer<0, align>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BgrToBayer<1, align>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerRggb:
                BgrToBayer<2, align>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerBggr:
                BgrToBayer<3, align>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            default:
                assert(0);
            }
        }

        void BgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            if (Aligned(bgr) && Aligned(bgrStride) && Aligned(bayer) && Aligned(bayerStride))
                BgrToBayer<true>(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
            else
                BgrToBayer<false>(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
        }
    }
#endif// SIMD_VMX_ENABLE
}
