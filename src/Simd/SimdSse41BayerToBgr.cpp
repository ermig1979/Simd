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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdBayer.h"
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template <bool align> SIMD_INLINE void SaveBgr(const __m128i src[3], uint8_t * dst)
        {
            Store<align>((__m128i*)dst + 0, InterleaveBgr<0>(src[0], src[1], src[2]));
            Store<align>((__m128i*)dst + 1, InterleaveBgr<1>(src[0], src[1], src[2]));
            Store<align>((__m128i*)dst + 2, InterleaveBgr<2>(src[0], src[1], src[2]));
        }

        template <bool align, SimdPixelFormatType bayerFormat> void BayerToBgr(const __m128i src[12], uint8_t * bgr, size_t stride)
        {
            __m128i _bgr[6];
            Sse2::BayerToBgr<bayerFormat>(src, _bgr);
            SaveBgr<align>(_bgr + 0, bgr);
            SaveBgr<align>(_bgr + 3, bgr + stride);
        }

        template <bool align, SimdPixelFormatType bayerFormat> void BayerToBgr(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, uint8_t * bgr, size_t bgrStride)
        {
            const uint8_t * src[3];
            __m128i _src[12];
            size_t body = AlignHi(width - 2, A) - A;
            for (size_t row = 0; row < height; row += 2)
            {
                src[0] = (row == 0 ? bayer : bayer - 2 * bayerStride);
                src[1] = bayer;
                src[2] = (row == height - 2 ? bayer : bayer + 2 * bayerStride);

                LoadBayerNose<align>(src, 0, bayerStride, _src);
                BayerToBgr<align, bayerFormat>(_src, bgr, bgrStride);
                for (size_t col = A; col < body; col += A)
                {
                    LoadBayerBody<align>(src, col, bayerStride, _src);
                    BayerToBgr<align, bayerFormat>(_src, bgr + 3 * col, bgrStride);
                }
                LoadBayerTail<false>(src, width - A, bayerStride, _src);
                BayerToBgr<false, bayerFormat>(_src, bgr + 3 * (width - A), bgrStride);

                bayer += 2 * bayerStride;
                bgr += 2 * bgrStride;
            }
        }

        template <bool align> void BayerToBgr(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t * bgr, size_t bgrStride)
        {
            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BayerToBgr<align, SimdPixelFormatBayerGrbg>(bayer, width, height, bayerStride, bgr, bgrStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BayerToBgr<align, SimdPixelFormatBayerGbrg>(bayer, width, height, bayerStride, bgr, bgrStride);
                break;
            case SimdPixelFormatBayerRggb:
                BayerToBgr<align, SimdPixelFormatBayerRggb>(bayer, width, height, bayerStride, bgr, bgrStride);
                break;
            case SimdPixelFormatBayerBggr:
                BayerToBgr<align, SimdPixelFormatBayerBggr>(bayer, width, height, bayerStride, bgr, bgrStride);
                break;
            default:
                assert(0);
            }
        }

        void BayerToBgr(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t * bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            if (Aligned(bayer) && Aligned(bgr) && Aligned(bayerStride) && Aligned(bgrStride))
                BayerToBgr<true>(bayer, width, height, bayerStride, bayerFormat, bgr, bgrStride);
            else
                BayerToBgr<false>(bayer, width, height, bayerStride, bayerFormat, bgr, bgrStride);
        }
    }
#endif
}
