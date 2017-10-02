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
        template <SimdPixelFormatType bayerFormat> void BgraToBayer(const uint8_t * bgra0, const uint8_t * bgra1, uint8_t * bayer0, uint8_t * bayer1);

        template <> SIMD_INLINE void BgraToBayer<SimdPixelFormatBayerGrbg>(const uint8_t * bgra0, const uint8_t * bgra1, uint8_t * bayer0, uint8_t * bayer1)
        {
            bayer0[0] = bgra0[1];
            bayer0[1] = bgra0[6];
            bayer1[0] = bgra1[0];
            bayer1[1] = bgra1[5];
        }

        template <> SIMD_INLINE void BgraToBayer<SimdPixelFormatBayerGbrg>(const uint8_t * bgra0, const uint8_t * bgra1, uint8_t * bayer0, uint8_t * bayer1)
        {
            bayer0[0] = bgra0[1];
            bayer0[1] = bgra0[4];
            bayer1[0] = bgra1[2];
            bayer1[1] = bgra1[5];
        }

        template <> SIMD_INLINE void BgraToBayer<SimdPixelFormatBayerRggb>(const uint8_t * bgra0, const uint8_t * bgra1, uint8_t * bayer0, uint8_t * bayer1)
        {
            bayer0[0] = bgra0[2];
            bayer0[1] = bgra0[5];
            bayer1[0] = bgra1[1];
            bayer1[1] = bgra1[4];
        }

        template <> SIMD_INLINE void BgraToBayer<SimdPixelFormatBayerBggr>(const uint8_t * bgra0, const uint8_t * bgra1, uint8_t * bayer0, uint8_t * bayer1)
        {
            bayer0[0] = bgra0[0];
            bayer0[1] = bgra0[5];
            bayer1[0] = bgra1[1];
            bayer1[1] = bgra1[6];
        }

        template <SimdPixelFormatType bayerFormat> SIMD_INLINE void BgraToBayer(const uint8_t * bgra, size_t bgraStride, uint8_t * bayer, size_t bayerStride)
        {
            BgraToBayer<bayerFormat>(bgra, bgra + bgraStride, bayer, bayer + bayerStride);
        }

        template <SimdPixelFormatType bayerFormat> void BgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride)
        {
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t col = 0, offset = 0; col < width; col += 2, offset += 8)
                    BgraToBayer<bayerFormat>(bgra + offset, bgraStride, bayer + col, bayerStride);
                bgra += 2 * bgraStride;
                bayer += 2 * bayerStride;
            }
        }

        void BgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BgraToBayer<SimdPixelFormatBayerGrbg>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BgraToBayer<SimdPixelFormatBayerGbrg>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerRggb:
                BgraToBayer<SimdPixelFormatBayerRggb>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            case SimdPixelFormatBayerBggr:
                BgraToBayer<SimdPixelFormatBayerBggr>(bgra, width, height, bgraStride, bayer, bayerStride);
                break;
            default:
                assert(0);
            }
        }
    }
}
