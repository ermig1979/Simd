/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
        const __m512i K8_SUFFLE_BGRA_TO_BGR = SIMD_MM512_SETR_EPI8(
            0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
            0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
            0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
            0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1);

        const __m512i K32_PERMUTE_BGRA_TO_BGR = SIMD_MM512_SETR_EPI32(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1);

        const __m512i K32_PERMUTE_BGRA_TO_BGR_0 = SIMD_MM512_SETR_EPI32(0x00, 0x01, 0x02, 0x04, 0x05, 0x06, 0x08, 0x09, 0x0A, 0x0C, 0x0D, 0x0E, 0x10, 0x11, 0x12, 0x14);
        const __m512i K32_PERMUTE_BGRA_TO_BGR_1 = SIMD_MM512_SETR_EPI32(0x05, 0x06, 0x08, 0x09, 0x0A, 0x0C, 0x0D, 0x0E, 0x10, 0x11, 0x12, 0x14, 0x15, 0x16, 0x18, 0x19);
        const __m512i K32_PERMUTE_BGRA_TO_BGR_2 = SIMD_MM512_SETR_EPI32(0x0A, 0x0C, 0x0D, 0x0E, 0x10, 0x11, 0x12, 0x14, 0x15, 0x16, 0x18, 0x19, 0x1A, 0x1C, 0x1D, 0x1E);

        template <bool mask> SIMD_INLINE void BgraToBgr(const uint8_t * bgra, uint8_t * bgr, __mmask64 bgraMask = -1, __mmask64 bgrMask = 0x0000ffffffffffff)
        {
            __m512i _bgra = mask ? _mm512_maskz_loadu_epi8(bgraMask, bgra) : _mm512_loadu_si512(bgra);
            __m512i _bgr = _mm512_permutexvar_epi32(K32_PERMUTE_BGRA_TO_BGR, _mm512_shuffle_epi8(_bgra, K8_SUFFLE_BGRA_TO_BGR));
            _mm512_mask_storeu_epi8(bgr, bgrMask, _bgr);
        }

        SIMD_INLINE void BgraToBgr(const uint8_t * bgra, uint8_t * bgr)
        {
            __m512i bgr0 = _mm512_shuffle_epi8(_mm512_loadu_si512(bgra + 0 * A), K8_SUFFLE_BGRA_TO_BGR);
            __m512i bgr1 = _mm512_shuffle_epi8(_mm512_loadu_si512(bgra + 1 * A), K8_SUFFLE_BGRA_TO_BGR);
            __m512i bgr2 = _mm512_shuffle_epi8(_mm512_loadu_si512(bgra + 2 * A), K8_SUFFLE_BGRA_TO_BGR);
            __m512i bgr3 = _mm512_shuffle_epi8(_mm512_loadu_si512(bgra + 3 * A), K8_SUFFLE_BGRA_TO_BGR);
            _mm512_storeu_si512(bgr + 0 * A, _mm512_permutex2var_epi32(bgr0, K32_PERMUTE_BGRA_TO_BGR_0, bgr1));
            _mm512_storeu_si512(bgr + 1 * A, _mm512_permutex2var_epi32(bgr1, K32_PERMUTE_BGRA_TO_BGR_1, bgr2));
            _mm512_storeu_si512(bgr + 2 * A, _mm512_permutex2var_epi32(bgr2, K32_PERMUTE_BGRA_TO_BGR_2, bgr3));
        }

        void BgraToBgr(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bgr, size_t bgrStride)
        {
            size_t fullAlignedWidth = AlignLo(width, A);
            size_t alignedWidth = AlignLo(width, F);
            __mmask64 bgraTailMask = TailMask64((width - alignedWidth) * 4);
            __mmask64 bgrTailMask = TailMask64((width - alignedWidth) * 3);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += A)
                    BgraToBgr(bgra + 4 * col, bgr + 3 * col);
                for (; col < alignedWidth; col += F)
                    BgraToBgr<false>(bgra + 4 * col, bgr + 3 * col);
                if (col < width)
                    BgraToBgr<true>(bgra + 4 * col, bgr + 3 * col, bgraTailMask, bgrTailMask);
                bgra += bgraStride;
                bgr += bgrStride;
            }
        }

        //---------------------------------------------------------------------

        const __m512i K8_SUFFLE_BGRA_TO_RGB = SIMD_MM512_SETR_EPI8(
            0x2, 0x1, 0x0, 0x6, 0x5, 0x4, 0xA, 0x9, 0x8, 0xE, 0xD, 0xC, -1, -1, -1, -1,
            0x2, 0x1, 0x0, 0x6, 0x5, 0x4, 0xA, 0x9, 0x8, 0xE, 0xD, 0xC, -1, -1, -1, -1,
            0x2, 0x1, 0x0, 0x6, 0x5, 0x4, 0xA, 0x9, 0x8, 0xE, 0xD, 0xC, -1, -1, -1, -1,
            0x2, 0x1, 0x0, 0x6, 0x5, 0x4, 0xA, 0x9, 0x8, 0xE, 0xD, 0xC, -1, -1, -1, -1);

        template <bool mask> SIMD_INLINE void BgraToRgb(const uint8_t* bgra, uint8_t* rgb, __mmask64 bgraMask = -1, __mmask64 rgbMask = 0x0000ffffffffffff)
        {
            __m512i _bgra = mask ? _mm512_maskz_loadu_epi8(bgraMask, bgra) : _mm512_loadu_si512(bgra);
            __m512i _rgb = _mm512_permutexvar_epi32(K32_PERMUTE_BGRA_TO_BGR, _mm512_shuffle_epi8(_bgra, K8_SUFFLE_BGRA_TO_RGB));
            _mm512_mask_storeu_epi8(rgb, rgbMask, _rgb);
        }

        SIMD_INLINE void BgraToRgb(const uint8_t* bgra, uint8_t* rgb)
        {
            __m512i rgb0 = _mm512_shuffle_epi8(_mm512_loadu_si512(bgra + 0 * A), K8_SUFFLE_BGRA_TO_RGB);
            __m512i rgb1 = _mm512_shuffle_epi8(_mm512_loadu_si512(bgra + 1 * A), K8_SUFFLE_BGRA_TO_RGB);
            __m512i rgb2 = _mm512_shuffle_epi8(_mm512_loadu_si512(bgra + 2 * A), K8_SUFFLE_BGRA_TO_RGB);
            __m512i rgb3 = _mm512_shuffle_epi8(_mm512_loadu_si512(bgra + 3 * A), K8_SUFFLE_BGRA_TO_RGB);
            _mm512_storeu_si512(rgb + 0 * A, _mm512_permutex2var_epi32(rgb0, K32_PERMUTE_BGRA_TO_BGR_0, rgb1));
            _mm512_storeu_si512(rgb + 1 * A, _mm512_permutex2var_epi32(rgb1, K32_PERMUTE_BGRA_TO_BGR_1, rgb2));
            _mm512_storeu_si512(rgb + 2 * A, _mm512_permutex2var_epi32(rgb2, K32_PERMUTE_BGRA_TO_BGR_2, rgb3));
        }

        void BgraToRgb(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* bgr, size_t bgrStride)
        {
            size_t fullAlignedWidth = AlignLo(width, A);
            size_t alignedWidth = AlignLo(width, F);
            __mmask64 bgraTailMask = TailMask64((width - alignedWidth) * 4);
            __mmask64 rgbTailMask = TailMask64((width - alignedWidth) * 3);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += A)
                    BgraToRgb(bgra + 4 * col, bgr + 3 * col);
                for (; col < alignedWidth; col += F)
                    BgraToRgb<false>(bgra + 4 * col, bgr + 3 * col);
                if (col < width)
                    BgraToRgb<true>(bgra + 4 * col, bgr + 3 * col, bgraTailMask, rgbTailMask);
                bgra += bgraStride;
                bgr += bgrStride;
            }
        }

        //---------------------------------------------------------------------

        const __m512i K8_BGRA_TO_RGBA = SIMD_MM512_SETR_EPI8(
            0x2, 0x1, 0x0, 0x3, 0x6, 0x5, 0x4, 0x7, 0xA, 0x9, 0x8, 0xB, 0xE, 0xD, 0xC, 0xF,
            0x2, 0x1, 0x0, 0x3, 0x6, 0x5, 0x4, 0x7, 0xA, 0x9, 0x8, 0xB, 0xE, 0xD, 0xC, 0xF,
            0x2, 0x1, 0x0, 0x3, 0x6, 0x5, 0x4, 0x7, 0xA, 0x9, 0x8, 0xB, 0xE, 0xD, 0xC, 0xF,
            0x2, 0x1, 0x0, 0x3, 0x6, 0x5, 0x4, 0x7, 0xA, 0x9, 0x8, 0xB, 0xE, 0xD, 0xC, 0xF);

        template <bool mask> SIMD_INLINE void BgraToRgba(const uint8_t* bgra, uint8_t* rgba, __mmask64 tail = -1)
        {
            const __m512i _bgra = mask ? _mm512_maskz_loadu_epi8(tail, bgra) : _mm512_loadu_si512(bgra);
            const __m512i rgbaOut = _mm512_shuffle_epi8(_bgra, K8_BGRA_TO_RGBA);
            if (mask)
                _mm512_mask_storeu_epi8(rgba, tail, rgbaOut);
            else
                _mm512_storeu_si512(rgba, rgbaOut);
        }

        void BgraToRgba(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgba, size_t rgbaStride)
        {
            size_t size = width * 4;
            size_t sizeA = AlignLo(size, A);
            __mmask64 tail = TailMask64(size - sizeA);

            for (size_t row = 0; row < height; ++row)
            {
                size_t i = 0;
                for (; i < sizeA; i += A)
                    BgraToRgba<false>(bgra + i, rgba + i);
                if (i < size)
                    BgraToRgba<true>(bgra + i, rgba + i, tail);
                bgra += bgraStride;
                rgba += rgbaStride;
            }
        }
    }
#endif
}
