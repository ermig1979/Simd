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
        const v128_u8 K8_PERM_GR = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x00, 0x06, 0x00, 0x09, 0x00, 0x0E, 0x00, 0x11, 0x00, 0x16, 0x00, 0x19, 0x00, 0x1E);
        const v128_u8 K8_PERM_BG = SIMD_VEC_SETR_EPI8(0x00, 0x00, 0x00, 0x05, 0x00, 0x08, 0x00, 0x0D, 0x00, 0x10, 0x00, 0x15, 0x00, 0x18, 0x00, 0x1D);
        const v128_u8 K8_PERM_GB = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x00, 0x04, 0x00, 0x09, 0x00, 0x0C, 0x00, 0x11, 0x00, 0x14, 0x00, 0x19, 0x00, 0x1C);
        const v128_u8 K8_PERM_RG = SIMD_VEC_SETR_EPI8(0x00, 0x02, 0x00, 0x05, 0x00, 0x0A, 0x00, 0x0D, 0x00, 0x12, 0x00, 0x15, 0x00, 0x1A, 0x00, 0x1D);

        template <int format, int row, bool align, bool first>
        SIMD_INLINE void BgraToBayer(const Loader<align> & bgra, const v128_u8 perm[4][2], Storer<align> & bayer)
        {
            const v128_u16 lo = (v128_u16)vec_perm(Load<align, first>(bgra), Load<align, false>(bgra), perm[format][row]);
            const v128_u16 hi = (v128_u16)vec_perm(Load<align, false>(bgra), Load<align, false>(bgra), perm[format][row]);
            Store<align, first>(bayer, vec_pack(lo, hi));
        }

        template <int format, bool align>
        void BgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(bayer) && Aligned(bayerStride));

            size_t alignedWidth = AlignLo(width, A);

            const v128_u8 perm[4][2] =
            {
                {K8_PERM_GR, K8_PERM_BG},
                {K8_PERM_GB, K8_PERM_RG},
                {K8_PERM_RG, K8_PERM_GB},
                {K8_PERM_BG, K8_PERM_GR}
            };

            for (size_t row = 0; row < height; row += 2)
            {
                Loader<align> _bgra0(bgra);
                Storer<align> _bayer0(bayer);
                BgraToBayer<format, 0, align, true>(_bgra0, perm, _bayer0);
                for (size_t col = A; col < alignedWidth; col += A)
                    BgraToBayer<format, 0, align, false>(_bgra0, perm, _bayer0);
                Flush(_bayer0);

                if (width != alignedWidth)
                {
                    Loader<false> _bgra(bgra + 4 * (width - A));
                    Storer<false> _bayer(bayer + width - A);
                    BgraToBayer<format, 0, false, true>(_bgra, perm, _bayer);
                    Flush(_bayer);
                }

                bgra += bgraStride;
                bayer += bayerStride;

                Loader<align> _bgra1(bgra);
                Storer<align> _bayer1(bayer);
                BgraToBayer<format, 1, align, true>(_bgra1, perm, _bayer1);
                for (size_t col = A; col < alignedWidth; col += A)
                    BgraToBayer<format, 1, align, false>(_bgra1, perm, _bayer1);
                Flush(_bayer1);

                if (width != alignedWidth)
                {
                    Loader<false> _bgra(bgra + 4 * (width - A));
                    Storer<false> _bayer(bayer + width - A);
                    BgraToBayer<format, 1, false, true>(_bgra, perm, _bayer);
                    Flush(_bayer);
                }

                bgra += bgraStride;
                bayer += bayerStride;
            }
        }

        template<bool align>
        void BgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BgraToBayer<0, align>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BgraToBayer<1, align>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerRggb:
                BgraToBayer<2, align>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerBggr:
                BgraToBayer<3, align>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            default:
                assert(0);
            }
        }

        void BgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            if (Aligned(bgra) && Aligned(bgraStride) && Aligned(bayer) && Aligned(bayerStride))
                BgraToBayer<true>(bgra, width, height, bgraStride, bayer, bayerStride, bayerFormat);
            else
                BgraToBayer<false>(bgra, width, height, bgraStride, bayer, bayerStride, bayerFormat);
        }
    }
#endif// SIMD_VMX_ENABLE
}
