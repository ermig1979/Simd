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
        template <bool align, bool first>
        SIMD_INLINE void GrayToBgr(const Loader<align> & gray, Storer<align> & bgr)
        {
            v128_u8 _gray = Load<align, first>(gray);

            Store<align, first>(bgr, vec_perm(_gray, K8_00, K8_PERM_GRAY_TO_BGR_0));
            Store<align, false>(bgr, vec_perm(_gray, K8_00, K8_PERM_GRAY_TO_BGR_1));
            Store<align, false>(bgr, vec_perm(_gray, K8_00, K8_PERM_GRAY_TO_BGR_2));
        }

        template <bool align> void GrayToBgr(const uint8_t *gray, size_t width, size_t height, size_t grayStride, uint8_t *bgr, size_t bgrStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(gray) && Aligned(grayStride));

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _gray(gray);
                Storer<align> _bgr(bgr);
                GrayToBgr<align, true>(_gray, _bgr);
                for (size_t col = A; col < alignedWidth; col += A)
                    GrayToBgr<align, false>(_gray, _bgr);
                Flush(_bgr);

                if (alignedWidth != width)
                {
                    Loader<false> _gray(gray + width - A);
                    Storer<false> _bgr(bgr + 3 * (width - A));
                    GrayToBgr<false, true>(_gray, _bgr);
                    Flush(_bgr);
                }

                gray += grayStride;
                bgr += bgrStride;
            }
        }

        void GrayToBgr(const uint8_t *gray, size_t width, size_t height, size_t grayStride, uint8_t *bgr, size_t bgrStride)
        {
            if (Aligned(bgr) && Aligned(gray) && Aligned(bgrStride) && Aligned(grayStride))
                GrayToBgr<true>(gray, width, height, grayStride, bgr, bgrStride);
            else
                GrayToBgr<false>(gray, width, height, grayStride, bgr, bgrStride);
        }
    }
#endif// SIMD_VMX_ENABLE
}
