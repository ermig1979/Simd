/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#include "Simd/SimdBayer.h"

namespace Simd
{
    namespace Base
    {
        template <SimdPixelFormatType bayerFormat> void BayerToBgra(const uint8_t * src[6],
            size_t col0, size_t col2, size_t col4, uint8_t * dst0, size_t stride, uint8_t alpha)
        {
            uint8_t * dst1 = dst0 + stride;
            BayerToBgr<bayerFormat>(src, col0, col0 + 1, col2, col2 + 1, col4, col4 + 1, dst0, dst0 + 4, dst1, dst1 + 4);
            dst0[3] = alpha;
            dst0[7] = alpha;
            dst1[3] = alpha;
            dst1[7] = alpha;
        }

        template <SimdPixelFormatType bayerFormat> void BayerToBgra(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            const uint8_t * src[6];
            for (size_t row = 0; row < height; row += 2)
            {
                src[0] = (row == 0 ? bayer : bayer - 2 * bayerStride);
                src[1] = src[0] + bayerStride;
                src[2] = bayer;
                src[3] = src[2] + bayerStride;
                src[4] = (row == height - 2 ? bayer : bayer + 2 * bayerStride);
                src[5] = src[4] + bayerStride;

                BayerToBgra<bayerFormat>(src, 0, 0, 2, bgra, bgraStride, alpha);

                for (size_t col = 2; col < width - 2; col += 2)
                    BayerToBgra<bayerFormat>(src, col - 2, col, col + 2, bgra + 4 * col, bgraStride, alpha);

                BayerToBgra<bayerFormat>(src, width - 4, width - 2, width - 2, bgra + 4 * (width - 2), bgraStride, alpha);

                bayer += 2 * bayerStride;
                bgra += 2 * bgraStride;
            }
        }

        void BayerToBgra(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BayerToBgra<SimdPixelFormatBayerGrbg>(bayer, width, height, bayerStride, bgra, bgraStride, alpha);
                break;
            case SimdPixelFormatBayerGbrg:
                BayerToBgra<SimdPixelFormatBayerGbrg>(bayer, width, height, bayerStride, bgra, bgraStride, alpha);
                break;
            case SimdPixelFormatBayerRggb:
                BayerToBgra<SimdPixelFormatBayerRggb>(bayer, width, height, bayerStride, bgra, bgraStride, alpha);
                break;
            case SimdPixelFormatBayerBggr:
                BayerToBgra<SimdPixelFormatBayerBggr>(bayer, width, height, bayerStride, bgra, bgraStride, alpha);
                break;
            default:
                assert(0);
            }
        }
    }
}
