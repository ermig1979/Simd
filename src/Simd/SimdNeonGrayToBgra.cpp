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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE  
    namespace Neon
    {
        const size_t A4 = A * 4;

        template <bool align> SIMD_INLINE void GrayToBgra(const uint8_t * gray, uint8_t * bgra, uint8x16x4_t & _bgra)
        {
            _bgra.val[0] = Load<align>(gray);
            _bgra.val[1] = _bgra.val[0];
            _bgra.val[2] = _bgra.val[0];
            Store4<align>(bgra, _bgra);
        }

        template <bool align> void GrayToBgra(const uint8_t * gray, size_t width, size_t height, size_t grayStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(gray) && Aligned(grayStride));

            size_t alignedWidth = AlignLo(width, A);

            uint8x16x4_t _bgra;
            _bgra.val[3] = vdupq_n_u8(alpha);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgra = 0; col < alignedWidth; col += A, colBgra += A4)
                    GrayToBgra<align>(gray + col, bgra + colBgra, _bgra);
                if (width != alignedWidth)
                    GrayToBgra<false>(gray + width - A, bgra + 4 * (width - A), _bgra);
                gray += grayStride;
                bgra += bgraStride;
            }
        }

        void GrayToBgra(const uint8_t * gray, size_t width, size_t height, size_t grayStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(bgra) && Aligned(gray) && Aligned(bgraStride) && Aligned(grayStride))
                GrayToBgra<true>(gray, width, height, grayStride, bgra, bgraStride, alpha);
            else
                GrayToBgra<false>(gray, width, height, grayStride, bgra, bgraStride, alpha);
        }
    }
#endif// SIMD_NEON_ENABLE
}
