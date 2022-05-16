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

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE void Yuv420pToUyvy422(const uint8_t* y0, size_t yStride, const uint8_t* u, const uint8_t* v, 
            uint8_t* uyvy0, size_t uyvyStride, __mmask32 yuvMask, __mmask32 uyvyMask0, __mmask32 uyvyMask1)
        {
            static const __m512i PRM0 = SIMD_MM512_SETR_EPI32(0x00, 0x08, 0x10, 0x11, 0x01, 0x09, 0x12, 0x13, 0x02, 0x0A, 0x14, 0x15, 0x03, 0x0B, 0x16, 0x17);
            static const __m512i PRM1 = SIMD_MM512_SETR_EPI32(0x04, 0x0C, 0x18, 0x19, 0x05, 0x0D, 0x1A, 0x1B, 0x06, 0x0E, 0x1C, 0x1D, 0x07, 0x0F, 0x1E, 0x1F);
            static const __m512i SHFL = SIMD_MM512_SETR_EPI8(
                0x0, 0x8, 0x4, 0x9, 0x1, 0xA, 0x5, 0xB, 0x2, 0xC, 0x6, 0xD, 0x3, 0xE, 0x7, 0xF,
                0x0, 0x8, 0x4, 0x9, 0x1, 0xA, 0x5, 0xB, 0x2, 0xC, 0x6, 0xD, 0x3, 0xE, 0x7, 0xF,
                0x0, 0x8, 0x4, 0x9, 0x1, 0xA, 0x5, 0xB, 0x2, 0xC, 0x6, 0xD, 0x3, 0xE, 0x7, 0xF,
                0x0, 0x8, 0x4, 0x9, 0x1, 0xA, 0x5, 0xB, 0x2, 0xC, 0x6, 0xD, 0x3, 0xE, 0x7, 0xF);
            __m512i uv = Load(u, v, yuvMask);
            __m512i _y0 = _mm512_maskz_loadu_epi16(yuvMask, y0);
            _mm512_mask_storeu_epi16(uyvy0 + 0 * 64, uyvyMask0, _mm512_shuffle_epi8(_mm512_permutex2var_epi32(uv, PRM0, _y0), SHFL));
            _mm512_mask_storeu_epi16(uyvy0 + 1 * 64, uyvyMask1, _mm512_shuffle_epi8(_mm512_permutex2var_epi32(uv, PRM1, _y0), SHFL));
            __m512i _y1 = _mm512_maskz_loadu_epi16(yuvMask, y0 + yStride);
            uint8_t* uyvy1 = uyvy0 + uyvyStride;
            _mm512_mask_storeu_epi16(uyvy1 + 0 * 64, uyvyMask0, _mm512_shuffle_epi8(_mm512_permutex2var_epi32(uv, PRM0, _y1), SHFL));
            _mm512_mask_storeu_epi16(uyvy1 + 1 * 64, uyvyMask1, _mm512_shuffle_epi8(_mm512_permutex2var_epi32(uv, PRM1, _y1), SHFL));
        }

        void Yuv420pToUyvy422(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, 
            const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* uyvy, size_t uyvyStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && width >= 2 * A);

            size_t size = width / 2;
            size_t size32 = AlignLo(size, 32);
            size_t tail = size - size32;
            __mmask32 yuvMask = TailMask32(tail);
            __mmask32 uyvyMask0 = TailMask32(tail * 2 - 32 * 0);
            __mmask32 uyvyMask1 = TailMask32(tail * 2 - 32 * 1);

            for (size_t row = 0; row < height; row += 2)
            {
                size_t colY = 0, colUV = 0, colUyvy = 0;
                for (; colUV < size32; colY += 64, colUV += 32, colUyvy += 128)
                    Yuv420pToUyvy422(y + colY, yStride, u + colUV, v + colUV, uyvy + colUyvy, uyvyStride, __mmask32(-1), __mmask32(-1), __mmask32(-1));
                if (tail)
                    Yuv420pToUyvy422(y + colY, yStride, u + colUV, v + colUV, uyvy + colUyvy, uyvyStride, yuvMask, uyvyMask0, uyvyMask1);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                uyvy += 2 * uyvyStride;
            }            
        }
    }
#endif
}
