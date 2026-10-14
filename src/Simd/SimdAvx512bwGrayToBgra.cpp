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
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool mask> SIMD_INLINE void GrayToBgra(const uint8_t * gray, const __m512i & alpha, uint8_t * bgra, const __mmask64 tails[5])
        {
            __m512i gray0 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, mask ? _mm512_maskz_loadu_epi8(tails[0], gray + 0) : _mm512_loadu_si512(gray + 0));
            __m512i bg0 = _mm512_unpacklo_epi8(gray0, gray0);
            __m512i bg1 = _mm512_unpackhi_epi8(gray0, gray0);
            __m512i ra0 = _mm512_unpacklo_epi8(gray0, alpha);
            __m512i ra1 = _mm512_unpackhi_epi8(gray0, alpha);
            __m512i bgra0 = _mm512_unpacklo_epi16(bg0, ra0);
            __m512i bgra1 = _mm512_unpackhi_epi16(bg0, ra0);
            __m512i bgra2 = _mm512_unpacklo_epi16(bg1, ra1);
            __m512i bgra3 = _mm512_unpackhi_epi16(bg1, ra1);
            if (mask)
            {
                _mm512_mask_storeu_epi8(bgra + 0 * A, tails[1], bgra0);
                _mm512_mask_storeu_epi8(bgra + 1 * A, tails[2], bgra1);
                _mm512_mask_storeu_epi8(bgra + 2 * A, tails[3], bgra2);
                _mm512_mask_storeu_epi8(bgra + 3 * A, tails[4], bgra3);
            }
            else
            {
                _mm512_storeu_si512(bgra + 0 * A, bgra0);
                _mm512_storeu_si512(bgra + 1 * A, bgra1);
                _mm512_storeu_si512(bgra + 2 * A, bgra2);
                _mm512_storeu_si512(bgra + 3 * A, bgra3);
            }
        }

        SIMD_INLINE void GrayToBgra2(const uint8_t * gray, const __m512i & alpha, uint8_t * bgra)
        {
            __m512i gray0 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_loadu_si512(gray + 0));
            __m512i bg0 = _mm512_unpacklo_epi8(gray0, gray0);
            __m512i bg1 = _mm512_unpackhi_epi8(gray0, gray0);
            __m512i ra0 = _mm512_unpacklo_epi8(gray0, alpha);
            __m512i ra1 = _mm512_unpackhi_epi8(gray0, alpha);
            _mm512_storeu_si512(bgra + 0 * A, _mm512_unpacklo_epi16(bg0, ra0));
            _mm512_storeu_si512(bgra + 1 * A, _mm512_unpackhi_epi16(bg0, ra0));
            _mm512_storeu_si512(bgra + 2 * A, _mm512_unpacklo_epi16(bg1, ra1));
            _mm512_storeu_si512(bgra + 3 * A, _mm512_unpackhi_epi16(bg1, ra1));

            __m512i gray1 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_loadu_si512(gray + A));
            __m512i bg2 = _mm512_unpacklo_epi8(gray1, gray1);
            __m512i bg3 = _mm512_unpackhi_epi8(gray1, gray1);
            __m512i ra2 = _mm512_unpacklo_epi8(gray1, alpha);
            __m512i ra3 = _mm512_unpackhi_epi8(gray1, alpha);
            _mm512_storeu_si512(bgra + 4 * A, _mm512_unpacklo_epi16(bg2, ra2));
            _mm512_storeu_si512(bgra + 5 * A, _mm512_unpackhi_epi16(bg2, ra2));
            _mm512_storeu_si512(bgra + 6 * A, _mm512_unpacklo_epi16(bg3, ra3));
            _mm512_storeu_si512(bgra + 7 * A, _mm512_unpackhi_epi16(bg3, ra3));
        }

        void GrayToBgra(const uint8_t *gray, size_t width, size_t height, size_t grayStride, uint8_t *bgra, size_t bgraStride, uint8_t alpha)
        {
            __m512i _alpha = _mm512_set1_epi8(alpha);
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[5];
            tailMasks[0] = TailMask64(width - alignedWidth);
            for (size_t c = 0; c < 4; ++c)
                tailMasks[1 + c] = TailMask64((width - alignedWidth) * 4 - A*c);
            size_t fullAlignedWidth = AlignLo(width, DA);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += DA)
                    GrayToBgra2(gray + col, _alpha, bgra + col * 4);
                for (; col < alignedWidth; col += A)
                    GrayToBgra<false>(gray + col, _alpha, bgra + col * 4, tailMasks);
                if (col < width)
                    GrayToBgra<true>(gray + col, _alpha, bgra + col * 4, tailMasks);
                gray += grayStride;
                bgra += bgraStride;
            }
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
