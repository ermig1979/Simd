/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdDeinterleave.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE void Uyvy422ToYuv420p(const uint8_t* uyvy0, size_t uyvyStride, uint8_t* y0, size_t yStride, 
            uint8_t* u, uint8_t* v, __mmask32 uyvyMask0, __mmask32 uyvyMask1, __mmask32 yuvMask)
        {
            static const __m512i SHFL = SIMD_MM512_SETR_EPI8(
                0x0, 0x4, 0x8, 0xC, 0x2, 0x6, 0xA, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF,
                0x0, 0x4, 0x8, 0xC, 0x2, 0x6, 0xA, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF,
                0x0, 0x4, 0x8, 0xC, 0x2, 0x6, 0xA, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF,
                0x0, 0x4, 0x8, 0xC, 0x2, 0x6, 0xA, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF);
            static const __m512i PRMYY = SIMD_MM512_SETR_EPI32(0x02, 0x03, 0x06, 0x07, 0x0A, 0x0B, 0x0E, 0x0F, 0x12, 0x13, 0x16, 0x17, 0x1A, 0x1B, 0x1E, 0x1F);
            static const __m512i PRMUV = SIMD_MM512_SETR_EPI32(0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x01, 0x05, 0x09, 0x0D, 0x11, 0x15, 0x19, 0x1D);

            __m512i uyvy00 = _mm512_shuffle_epi8(_mm512_maskz_loadu_epi16(uyvyMask0, uyvy0 + 0 * 64), SHFL);
            __m512i uyvy01 = _mm512_shuffle_epi8(_mm512_maskz_loadu_epi16(uyvyMask1, uyvy0 + 1 * 64), SHFL);

            const uint8_t* uyvy1 = uyvy0 + uyvyStride;
            __m512i uyvy10 = _mm512_shuffle_epi8(_mm512_maskz_loadu_epi16(uyvyMask0, uyvy1 + 0 * 64), SHFL);
            __m512i uyvy11 = _mm512_shuffle_epi8(_mm512_maskz_loadu_epi16(uyvyMask1, uyvy1 + 1 * 64), SHFL);

            uint8_t* y1 = y0 + yStride;
            _mm512_mask_storeu_epi16(y0, yuvMask, _mm512_permutex2var_epi32(uyvy00, PRMYY, uyvy01));
            _mm512_mask_storeu_epi16(y1, yuvMask, _mm512_permutex2var_epi32(uyvy10, PRMYY, uyvy11));

            __m512i uv = _mm512_avg_epu8(_mm512_permutex2var_epi32(uyvy00, PRMUV, uyvy01), _mm512_permutex2var_epi32(uyvy10, PRMUV, uyvy11));
            _mm256_mask_storeu_epi8(u, yuvMask, _mm512_extracti64x4_epi64(uv, 0));
            _mm256_mask_storeu_epi8(v, yuvMask, _mm512_extracti64x4_epi64(uv, 1));
        }

        void Uyvy422ToYuv420p(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && width >= 2 * A);

            assert((width % 2 == 0) && (height % 2 == 0) && width >= 2 * A);

            size_t size = width / 2;
            size_t size32 = AlignLo(size, 32);
            size_t tail = size - size32;
            __mmask32 yuvMask = TailMask32(tail);
            __mmask32 uyvyMask0 = TailMask32(tail * 2 - 32 * 0);
            __mmask32 uyvyMask1 = TailMask32(tail * 2 - 32 * 1);

            for (size_t row = 0; row < height; row += 2)
            {
                size_t colUyvy = 0, colY = 0, colUV = 0;
                for (; colUV < size32; colY += 64, colUV += 32, colUyvy += 128)
                    Uyvy422ToYuv420p(uyvy + colUyvy, uyvyStride, y + colY, yStride, u + colUV, v + colUV, __mmask32(-1), __mmask32(-1), __mmask32(-1));
                if (tail)
                    Uyvy422ToYuv420p(uyvy + colUyvy, uyvyStride, y + colY, yStride, u + colUV, v + colUV, uyvyMask0, uyvyMask1, yuvMask);
                uyvy += 2 * uyvyStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
            }
        }
    }
#endif
}
