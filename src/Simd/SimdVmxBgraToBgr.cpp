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
        const v128_u8 K8_PERM_0 = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x02, 0x04, 0x05, 0x06, 0x08, 0x09, 0x0A, 0x0C, 0x0D, 0x0E, 0x10, 0x11, 0x12, 0x14);
        const v128_u8 K8_PERM_1 = SIMD_VEC_SETR_EPI8(0x05, 0x06, 0x08, 0x09, 0x0A, 0x0C, 0x0D, 0x0E, 0x10, 0x11, 0x12, 0x14, 0x15, 0x16, 0x18, 0x19);
        const v128_u8 K8_PERM_2 = SIMD_VEC_SETR_EPI8(0x0A, 0x0C, 0x0D, 0x0E, 0x10, 0x11, 0x12, 0x14, 0x15, 0x16, 0x18, 0x19, 0x1A, 0x1C, 0x1D, 0x1E);

        template <bool align, bool first> SIMD_INLINE void BgraToBgr(const Loader<align> & bgra, Storer<align> & bgr)
        {
            const v128_u8 bgra0 = Load<align, first>(bgra);
            const v128_u8 bgra1 = Load<align, false>(bgra);
            const v128_u8 bgra2 = Load<align, false>(bgra);
            const v128_u8 bgra3 = Load<align, false>(bgra);
            Store<align, first>(bgr, vec_perm(bgra0, bgra1, K8_PERM_0));
            Store<align, false>(bgr, vec_perm(bgra1, bgra2, K8_PERM_1));
            Store<align, false>(bgr, vec_perm(bgra2, bgra3, K8_PERM_2));
        }

        template <bool align> void BgraToBgr(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bgr, size_t bgrStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t alignedWidth = AlignLo(width, A);
            if (width == alignedWidth)
                alignedWidth -= A;

            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _bgra(bgra);
                Storer<align> _bgr(bgr);
                BgraToBgr<align, true>(_bgra, _bgr);
                for (size_t col = A; col < alignedWidth; col += A)
                    BgraToBgr<align, false>(_bgra, _bgr);
                Flush(_bgr);

                if (width != alignedWidth)
                {
                    Loader<false> _bgra(bgra + 4 * (width - A));
                    Storer<false> _bgr(bgr + 3 * (width - A));
                    BgraToBgr<false, true>(_bgra, _bgr);
                    Flush(_bgr);
                }

                bgra += bgraStride;
                bgr += bgrStride;
            }
        }

        void BgraToBgr(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bgr, size_t bgrStride)
        {
            if (Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride))
                BgraToBgr<true>(bgra, width, height, bgraStride, bgr, bgrStride);
            else
                BgraToBgr<false>(bgra, width, height, bgraStride, bgr, bgrStride);
        }
    }
#endif// SIMD_VMX_ENABLE
}
