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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        const __m128i K8_CVT_00 = SIMD_MM_SETR_EPI8(0x2, 0x1, 0x0, 0x5, 0x4, 0x3, 0x8, 0x7, 0x6, 0xB, 0xA, 0x9, 0xE, 0xD, 0xC, -1);
        const __m128i K8_CVT_01 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1);
        const __m128i K8_CVT_10 = SIMD_MM_SETR_EPI8(-1, 0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_CVT_11 = SIMD_MM_SETR_EPI8(0x0, -1, 0x4, 0x3, 0x2, 0x7, 0x6, 0x5, 0xA, 0x9, 0x8, 0xD, 0xC, 0xB, -1, 0xF);
        const __m128i K8_CVT_12 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0, -1);
        const __m128i K8_CVT_21 = SIMD_MM_SETR_EPI8(0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_CVT_22 = SIMD_MM_SETR_EPI8(-1, 0x3, 0x2, 0x1, 0x6, 0x5, 0x4, 0x9, 0x8, 0x7, 0xC, 0xB, 0xA, 0xF, 0xE, 0xD);

        template <bool align> SIMD_INLINE void BgrToRgb(const uint8_t * src, uint8_t * dst)
        {
            __m128i s0 = Load<align>((__m128i*)src + 0);
            __m128i s1 = Load<align>((__m128i*)src + 1);
            __m128i s2 = Load<align>((__m128i*)src + 2);
            Store<align>((__m128i*)dst + 0, _mm_or_si128(_mm_shuffle_epi8(s0, K8_CVT_00), _mm_shuffle_epi8(s1, K8_CVT_01)));
            Store<align>((__m128i*)dst + 1, _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(s0, K8_CVT_10), _mm_shuffle_epi8(s1, K8_CVT_11)), _mm_shuffle_epi8(s2, K8_CVT_12)));
            Store<align>((__m128i*)dst + 2, _mm_or_si128(_mm_shuffle_epi8(s1, K8_CVT_21), _mm_shuffle_epi8(s2, K8_CVT_22)));
        }

        template <bool align> void BgrToRgb(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* rgb, size_t rgbStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(rgb) && Aligned(rgbStride));

            const size_t A3 = A * 3;
            size_t size = width * 3;
            size_t aligned = AlignLo(width, A) * 3;

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t i = 0; i < aligned; i += A3)
                    BgrToRgb<align>(bgr + i, rgb + i);
                if (aligned < size)
                    BgrToRgb<false>(bgr + size - A3, rgb + size - A3);
                bgr += bgrStride;
                rgb += rgbStride;
            }
        }

        void BgrToRgb(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* rgb, size_t rgbStride)
        {
            if (Aligned(bgr) && Aligned(bgrStride) && Aligned(rgb) && Aligned(rgbStride))
                BgrToRgb<true>(bgr, width, height, bgrStride, rgb, rgbStride);
            else
                BgrToRgb<false>(bgr, width, height, bgrStride, rgb, rgbStride);
        }
    }
#endif
}
