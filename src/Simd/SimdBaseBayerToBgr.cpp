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
        template <SimdPixelFormatType bayerFormat> void BayerToBgr(const uint8_t * src[6],
            size_t col0, size_t col2, size_t col4, uint8_t * dst, size_t stride)
        {
            BayerToBgr<bayerFormat>(src,
                col0, col0 + 1, col2, col2 + 1, col4, col4 + 1,
                dst, dst + 3, dst + stride, dst + stride + 3);
        }

        template <SimdPixelFormatType bayerFormat> void BayerToBgr(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, uint8_t * bgr, size_t bgrStride)
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

                BayerToBgr<bayerFormat>(src, 0, 0, 2, bgr, bgrStride);

                for (size_t col = 2; col < width - 2; col += 2)
                    BayerToBgr<bayerFormat>(src, col - 2, col, col + 2, bgr + 3 * col, bgrStride);

                BayerToBgr<bayerFormat>(src, width - 4, width - 2, width - 2, bgr + 3 * (width - 2), bgrStride);

                bayer += 2 * bayerStride;
                bgr += 2 * bgrStride;
            }
        }

        void BayerToBgr(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t * bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BayerToBgr<SimdPixelFormatBayerGrbg>(bayer, width, height, bayerStride, bgr, bgrStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BayerToBgr<SimdPixelFormatBayerGbrg>(bayer, width, height, bayerStride, bgr, bgrStride);
                break;
            case SimdPixelFormatBayerRggb:
                BayerToBgr<SimdPixelFormatBayerRggb>(bayer, width, height, bayerStride, bgr, bgrStride);
                break;
            case SimdPixelFormatBayerBggr:
                BayerToBgr<SimdPixelFormatBayerBggr>(bayer, width, height, bayerStride, bgr, bgrStride);
                break;
            default:
                assert(0);
            }
        }
    }
}
