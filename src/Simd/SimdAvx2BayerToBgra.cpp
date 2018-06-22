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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdBayer.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align> SIMD_INLINE void SaveBgra(const __m256i bgr[3], const __m256i & alpha, uint8_t * bgra)
        {
            __m256i bgLo = PermutedUnpackLoU8(bgr[0], bgr[1]);
            __m256i bgHi = PermutedUnpackHiU8(bgr[0], bgr[1]);
            __m256i raLo = PermutedUnpackLoU8(bgr[2], alpha);
            __m256i raHi = PermutedUnpackHiU8(bgr[2], alpha);
            Store<align>((__m256i*)bgra + 0, UnpackU16<0>(bgLo, raLo));
            Store<align>((__m256i*)bgra + 1, UnpackU16<0>(bgHi, raHi));
            Store<align>((__m256i*)bgra + 2, UnpackU16<1>(bgLo, raLo));
            Store<align>((__m256i*)bgra + 3, UnpackU16<1>(bgHi, raHi));
        }

        template <bool align, SimdPixelFormatType bayerFormat> void BayerToBgra(const __m256i src[12], const __m256i & alpha, uint8_t * bgra, size_t stride)
        {
            __m256i bgr[6];
            BayerToBgr<bayerFormat>(src, bgr);
            SaveBgra<align>(bgr + 0, alpha, bgra);
            SaveBgra<align>(bgr + 3, alpha, bgra + stride);
        }

        template <bool align, SimdPixelFormatType bayerFormat> void BayerToBgra(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            const uint8_t * src[3];
            __m256i _src[12];
            __m256i _alpha = _mm256_set1_epi8((char)alpha);
            size_t body = AlignHi(width - 2, A) - A;
            for (size_t row = 0; row < height; row += 2)
            {
                src[0] = (row == 0 ? bayer : bayer - 2 * bayerStride);
                src[1] = bayer;
                src[2] = (row == height - 2 ? bayer : bayer + 2 * bayerStride);

                LoadBayerNose<align>(src, 0, bayerStride, _src);
                BayerToBgra<align, bayerFormat>(_src, _alpha, bgra, bgraStride);
                for (size_t col = A; col < body; col += A)
                {
                    LoadBayerBody<align>(src, col, bayerStride, _src);
                    BayerToBgra<align, bayerFormat>(_src, _alpha, bgra + 4 * col, bgraStride);
                }
                LoadBayerTail<false>(src, width - A, bayerStride, _src);
                BayerToBgra<false, bayerFormat>(_src, _alpha, bgra + 4 * (width - A), bgraStride);

                bayer += 2 * bayerStride;
                bgra += 2 * bgraStride;
            }
        }

        template <bool align> void BayerToBgra(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BayerToBgra<align, SimdPixelFormatBayerGrbg>(bayer, width, height, bayerStride, bgra, bgraStride, alpha);
                break;
            case SimdPixelFormatBayerGbrg:
                BayerToBgra<align, SimdPixelFormatBayerGbrg>(bayer, width, height, bayerStride, bgra, bgraStride, alpha);
                break;
            case SimdPixelFormatBayerRggb:
                BayerToBgra<align, SimdPixelFormatBayerRggb>(bayer, width, height, bayerStride, bgra, bgraStride, alpha);
                break;
            case SimdPixelFormatBayerBggr:
                BayerToBgra<align, SimdPixelFormatBayerBggr>(bayer, width, height, bayerStride, bgra, bgraStride, alpha);
                break;
            default:
                assert(0);
            }
        }

        void BayerToBgra(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            if (Aligned(bayer) && Aligned(bgra) && Aligned(bayerStride) && Aligned(bgraStride))
                BayerToBgra<true>(bayer, width, height, bayerStride, bayerFormat, bgra, bgraStride, alpha);
            else
                BayerToBgra<false>(bayer, width, height, bayerStride, bayerFormat, bgra, bgraStride, alpha);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
