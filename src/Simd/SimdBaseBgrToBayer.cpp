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
#include "Simd/SimdDefs.h"

namespace Simd
{
    namespace Base
    {
        template <SimdPixelFormatType bayerFormat> void BgrToBayer(const uint8_t * bgr0, const uint8_t * bgr1, uint8_t * bayer0, uint8_t * bayer1);

        template <> SIMD_INLINE void BgrToBayer<SimdPixelFormatBayerGrbg>(const uint8_t * bgr0, const uint8_t * bgr1, uint8_t * bayer0, uint8_t * bayer1)
        {
            bayer0[0] = bgr0[1];
            bayer0[1] = bgr0[5];
            bayer1[0] = bgr1[0];
            bayer1[1] = bgr1[4];
        }

        template <> SIMD_INLINE void BgrToBayer<SimdPixelFormatBayerGbrg>(const uint8_t * bgr0, const uint8_t * bgr1, uint8_t * bayer0, uint8_t * bayer1)
        {
            bayer0[0] = bgr0[1];
            bayer0[1] = bgr0[3];
            bayer1[0] = bgr1[2];
            bayer1[1] = bgr1[4];
        }

        template <> SIMD_INLINE void BgrToBayer<SimdPixelFormatBayerRggb>(const uint8_t * bgr0, const uint8_t * bgr1, uint8_t * bayer0, uint8_t * bayer1)
        {
            bayer0[0] = bgr0[2];
            bayer0[1] = bgr0[4];
            bayer1[0] = bgr1[1];
            bayer1[1] = bgr1[3];
        }

        template <> SIMD_INLINE void BgrToBayer<SimdPixelFormatBayerBggr>(const uint8_t * bgr0, const uint8_t * bgr1, uint8_t * bayer0, uint8_t * bayer1)
        {
            bayer0[0] = bgr0[0];
            bayer0[1] = bgr0[4];
            bayer1[0] = bgr1[1];
            bayer1[1] = bgr1[5];
        }

        template <SimdPixelFormatType bayerFormat> SIMD_INLINE void BgrToBayer(const uint8_t * bgr, size_t bgrStride, uint8_t * bayer, size_t bayerStride)
        {
            BgrToBayer<bayerFormat>(bgr, bgr + bgrStride, bayer, bayer + bayerStride);
        }

        template <SimdPixelFormatType bayerFormat> void BgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride)
        {
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t col = 0, offset = 0; col < width; col += 2, offset += 6)
                    BgrToBayer<bayerFormat>(bgr + offset, bgrStride, bayer + col, bayerStride);
                bgr += 2 * bgrStride;
                bayer += 2 * bayerStride;
            }
        }

        void BgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BgrToBayer<SimdPixelFormatBayerGrbg>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BgrToBayer<SimdPixelFormatBayerGbrg>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerRggb:
                BgrToBayer<SimdPixelFormatBayerRggb>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerBggr:
                BgrToBayer<SimdPixelFormatBayerBggr>(bgr, width, height, bgrStride, bayer, bayerStride);
                break;
            default:
                assert(0);
            }
        }
    }
}
