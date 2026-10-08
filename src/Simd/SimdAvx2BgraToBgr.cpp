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
#include "Simd/SimdConst.h"
#include "Simd/SimdSse41.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE  
    namespace Avx2
    {
        SIMD_INLINE __m256i BgraToBgr(const uint8_t* bgra)
        {
            __m256i _bgra = _mm256_loadu_si256((__m256i*)bgra);
            return _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_bgra, K8_SHUFFLE_BGRA_TO_BGR), K32_PERMUTE_BGRA_TO_BGR);
        }

        void BgraToBgr(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bgr, size_t bgrStride)
        {
            if (width < F)
            {
                Sse41::BgraToBgr(bgra, width, height, bgraStride, bgr, bgrStride);
                return;
            }

            assert(width >= F);

            size_t widthF = AlignLo(width, F) - F;

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < widthF; col += F)
                    _mm256_storeu_si256((__m256i*)(bgr + 3 * col), BgraToBgr(bgra + 4 * col));
                __m256i bgrF = BgraToBgr(bgra + 4 * widthF);
                _mm_storeu_si128((__m128i*)(bgr + 3 * widthF), _mm256_extractf128_si256(bgrF, 0));
                _mm_storel_epi64((__m128i*)(bgr + 3 * widthF) + 1, _mm256_extractf128_si256(bgrF, 1));
                if (widthF + F != width)
                {
                    __m256i bgrTail = BgraToBgr(bgra + 4 * (width - F));
                    _mm_storeu_si128((__m128i*)(bgr + 3 * (width - F)), _mm256_extractf128_si256(bgrTail, 0));
                    _mm_storel_epi64((__m128i*)(bgr + 3 * (width - F)) + 1, _mm256_extractf128_si256(bgrTail, 1));
                }
                bgra += bgraStride;
                bgr += bgrStride;
            }
        }

        //---------------------------------------------------------------------

        const __m256i K8_SHUFFLE_BGRA_TO_RGB = SIMD_MM256_SETR_EPI8(
            0x2, 0x1, 0x0, 0x6, 0x5, 0x4, 0xA, 0x9, 0x8, 0xE, 0xD, 0xC, -1, -1, -1, -1,
            0x2, 0x1, 0x0, 0x6, 0x5, 0x4, 0xA, 0x9, 0x8, 0xE, 0xD, 0xC, -1, -1, -1, -1);

        SIMD_INLINE __m256i BgraToRgb(const uint8_t* bgra)
        {
            __m256i _bgra = _mm256_loadu_si256((__m256i*)bgra);
            return _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_bgra, K8_SHUFFLE_BGRA_TO_RGB), K32_PERMUTE_BGRA_TO_BGR);
        }

        void BgraToRgb(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgb, size_t rgbStride)
        {
            if (width < F)
            {
                Sse41::BgraToRgb(bgra, width, height, bgraStride, rgb, rgbStride);
                return;
            }

            assert(width >= F);

            size_t widthF = AlignLo(width, F) - F;

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < widthF; col += F)
                    _mm256_storeu_si256((__m256i*)(rgb + 3 * col), BgraToRgb(bgra + 4 * col));
                __m256i rgbF = BgraToRgb(bgra + 4 * widthF);
                _mm_storeu_si128((__m128i*)(rgb + 3 * widthF), _mm256_extractf128_si256(rgbF, 0));
                _mm_storel_epi64((__m128i*)(rgb + 3 * widthF) + 1, _mm256_extractf128_si256(rgbF, 1));
                if (widthF + F != width)
                {
                    __m256i rgbTail = BgraToRgb(bgra + 4 * (width - F));
                    _mm_storeu_si128((__m128i*)(rgb + 3 * (width - F)), _mm256_extractf128_si256(rgbTail, 0));
                    _mm_storel_epi64((__m128i*)(rgb + 3 * (width - F)) + 1, _mm256_extractf128_si256(rgbTail, 1));
                }
                bgra += bgraStride;
                rgb += rgbStride;
            }
        }

        //---------------------------------------------------------------------

        const __m256i K8_BGRA_TO_RGBA = SIMD_MM256_SETR_EPI8(
            0x2, 0x1, 0x0, 0x3, 0x6, 0x5, 0x4, 0x7, 0xA, 0x9, 0x8, 0xB, 0xE, 0xD, 0xC, 0xF,
            0x2, 0x1, 0x0, 0x3, 0x6, 0x5, 0x4, 0x7, 0xA, 0x9, 0x8, 0xB, 0xE, 0xD, 0xC, 0xF);

        SIMD_INLINE void BgraToRgba(const uint8_t* bgra, uint8_t* rgba)
        {
            _mm256_storeu_si256((__m256i*)rgba, _mm256_shuffle_epi8(_mm256_loadu_si256((__m256i*)bgra), K8_BGRA_TO_RGBA));
        }

        void BgraToRgba(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgba, size_t rgbaStride)
        {
            if (width < F)
            {
                Sse41::BgraToRgba(bgra, width, height, bgraStride, rgba, rgbaStride);
                return;
            }

            assert(width >= F);

            size_t size = width * 4;
            size_t sizeA = AlignLo(size, A);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t i = 0; i < sizeA; i += A)
                    BgraToRgba(bgra + i, rgba + i);
                if (size != sizeA)
                    BgraToRgba(bgra + size - A, rgba + size - A);
                bgra += bgraStride;
                rgba += rgbaStride;
            }
        }
    }
#endif
}
