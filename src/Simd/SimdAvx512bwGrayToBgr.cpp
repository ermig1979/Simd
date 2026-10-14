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
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool mask> SIMD_INLINE void GrayToBgr(const uint8_t * gray, uint8_t * bgr, const __mmask64 tails[4])
        {
            const __m512i gray0 = mask ? _mm512_maskz_loadu_epi8(tails[0], gray + 0 * A) : _mm512_loadu_si512(gray + 0 * A);
            const __m512i bgr0 = GrayToBgr<0>(gray0);
            const __m512i bgr1 = GrayToBgr<1>(gray0);
            const __m512i bgr2 = GrayToBgr<2>(gray0);
            if (mask)
            {
                _mm512_mask_storeu_epi8(bgr + 0 * A, tails[1], bgr0);
                _mm512_mask_storeu_epi8(bgr + 1 * A, tails[2], bgr1);
                _mm512_mask_storeu_epi8(bgr + 2 * A, tails[3], bgr2);
            }
            else
            {
                _mm512_storeu_si512(bgr + 0 * A, bgr0);
                _mm512_storeu_si512(bgr + 1 * A, bgr1);
                _mm512_storeu_si512(bgr + 2 * A, bgr2);
            }
        }

        SIMD_INLINE void GrayToBgr2(const uint8_t * gray, uint8_t * bgr)
        {
            const __m512i gray0 = _mm512_loadu_si512(gray + 0 * A);
            _mm512_storeu_si512(bgr + 0 * A, GrayToBgr<0>(gray0));
            _mm512_storeu_si512(bgr + 1 * A, GrayToBgr<1>(gray0));
            _mm512_storeu_si512(bgr + 2 * A, GrayToBgr<2>(gray0));
            const __m512i gray1 = _mm512_loadu_si512(gray + 1 * A);
            _mm512_storeu_si512(bgr + 3 * A, GrayToBgr<0>(gray1));
            _mm512_storeu_si512(bgr + 4 * A, GrayToBgr<1>(gray1));
            _mm512_storeu_si512(bgr + 5 * A, GrayToBgr<2>(gray1));
        }

        void GrayToBgr(const uint8_t * gray, size_t width, size_t height, size_t grayStride, uint8_t *bgr, size_t bgrStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[4];
            tailMasks[0] = TailMask64(width - alignedWidth);
            for (size_t c = 0; c < 3; ++c)
                tailMasks[1 + c] = TailMask64((width - alignedWidth) * 3 - A*c);
            size_t fullAlignedWidth = AlignLo(width, DA);

            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += DA)
                    GrayToBgr2(gray + col, bgr + col * 3);
                for (; col < alignedWidth; col += A)
                    GrayToBgr<false>(gray + col, bgr + col * 3, tailMasks);
                if (col < width)
                    GrayToBgr<true>(gray + col, bgr + col * 3, tailMasks);
                gray += grayStride;
                bgr += bgrStride;
            }
        }
    }
#endif// SIMD_AVX2_ENABLE
}
