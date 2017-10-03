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
        const v128_u8 K8_PERM_GRAY_TO_BGRA_0 = SIMD_VEC_SETR_EPI8(0x00, 0x00, 0x00, 0x10, 0x01, 0x01, 0x01, 0x11, 0x02, 0x02, 0x02, 0x12, 0x03, 0x03, 0x03, 0x13);
        const v128_u8 K8_PERM_GRAY_TO_BGRA_1 = SIMD_VEC_SETR_EPI8(0x04, 0x04, 0x04, 0x14, 0x05, 0x05, 0x05, 0x15, 0x06, 0x06, 0x06, 0x16, 0x07, 0x07, 0x07, 0x17);
        const v128_u8 K8_PERM_GRAY_TO_BGRA_2 = SIMD_VEC_SETR_EPI8(0x08, 0x08, 0x08, 0x18, 0x09, 0x09, 0x09, 0x19, 0x0A, 0x0A, 0x0A, 0x1A, 0x0B, 0x0B, 0x0B, 0x1B);
        const v128_u8 K8_PERM_GRAY_TO_BGRA_3 = SIMD_VEC_SETR_EPI8(0x0C, 0x0C, 0x0C, 0x1C, 0x0D, 0x0D, 0x0D, 0x1D, 0x0E, 0x0E, 0x0E, 0x1E, 0x0F, 0x0F, 0x0F, 0x1F);

        template <bool align, bool first>
        SIMD_INLINE void GrayToBgra(const Loader<align> & gray, v128_u8 alpha, Storer<align> & bgra)
        {
            v128_u8 _gray = Load<align, first>(gray);

            Store<align, first>(bgra, vec_perm(_gray, alpha, K8_PERM_GRAY_TO_BGRA_0));
            Store<align, false>(bgra, vec_perm(_gray, alpha, K8_PERM_GRAY_TO_BGRA_1));
            Store<align, false>(bgra, vec_perm(_gray, alpha, K8_PERM_GRAY_TO_BGRA_2));
            Store<align, false>(bgra, vec_perm(_gray, alpha, K8_PERM_GRAY_TO_BGRA_3));
        }

        template <bool align> void GrayToBgra(const uint8_t *gray, size_t width, size_t height, size_t grayStride, uint8_t *bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(gray) && Aligned(grayStride));

            const v128_u8 _alpha = SIMD_VEC_SET1_EPI8(alpha);
            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _gray(gray);
                Storer<align> _bgra(bgra);
                GrayToBgra<align, true>(_gray, _alpha, _bgra);
                for (size_t col = A; col < alignedWidth; col += A)
                    GrayToBgra<align, false>(_gray, _alpha, _bgra);
                Flush(_bgra);

                if (alignedWidth != width)
                {
                    Loader<false> _gray(gray + width - A);
                    Storer<false> _bgra(bgra + 4 * (width - A));
                    GrayToBgra<false, true>(_gray, _alpha, _bgra);
                    Flush(_bgra);
                }

                gray += grayStride;
                bgra += bgraStride;
            }
        }

        void GrayToBgra(const uint8_t *gray, size_t width, size_t height, size_t grayStride, uint8_t *bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(bgra) && Aligned(gray) && Aligned(bgraStride) && Aligned(grayStride))
                GrayToBgra<true>(gray, width, height, grayStride, bgra, bgraStride, alpha);
            else
                GrayToBgra<false>(gray, width, height, grayStride, bgra, bgraStride, alpha);
        }
    }
#endif// SIMD_VMX_ENABLE
}
