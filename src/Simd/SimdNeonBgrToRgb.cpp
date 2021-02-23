/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> SIMD_INLINE void BgrToRgb(const uint8_t * bgr, uint8_t * rgb)
        {
            uint8x16x3_t _bgr = Load3<align>(bgr);
            uint8x16_t tmp = _bgr.val[0];
            _bgr.val[0] = _bgr.val[2];
            _bgr.val[2] = tmp;
            Store3<align>(rgb, _bgr);
        }

        template <bool align> void BgrToRgb(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* rgb, size_t rgbStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(rgb) && Aligned(rgbStride));

            const size_t A3 = A * 3;
            size_t size = width * 3;
            size_t aligned = AlignLo(width, A)*3;

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t i = 0; i < aligned; i += A3)
                    BgrToRgb<align>(bgr + i, rgb + i);
                if (aligned < size)
                    BgrToRgb<false>(bgr + size - A3, rgb + size - A3);
                bgr += bgrStride;
                rgb += rgbStride;
            }
        }

        void BgrToRgb(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* rgb, size_t rgbStride)
        {
            if (Aligned(bgr) && Aligned(bgrStride) && Aligned(rgb) && Aligned(rgbStride))
                BgrToRgb<true>(bgr, width, height, bgrStride, rgb, rgbStride);
            else
                BgrToRgb<false>(bgr, width, height, bgrStride, rgb, rgbStride);
        }
    }
#endif//SIMD_NEON_ENABLE
}
