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
#include "Simd/SimdSet.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE __m128i TextureBoostedSaturatedGradient16(__m128i difference, __m128i saturation, const __m128i & boost)
        {
            return _mm_mullo_epi16(_mm_max_epi16(K_ZERO, _mm_add_epi16(saturation, _mm_min_epi16(difference, saturation))), boost);
        }

        SIMD_INLINE __m128i TextureBoostedSaturatedGradient8(__m128i a, __m128i b, __m128i saturation, const __m128i & boost)
        {
            __m128i lo = TextureBoostedSaturatedGradient16(SubUnpackedU8<0>(b, a), saturation, boost);
            __m128i hi = TextureBoostedSaturatedGradient16(SubUnpackedU8<1>(b, a), saturation, boost);
            return _mm_packus_epi16(lo, hi);
        }

        template<bool align> SIMD_INLINE void TextureBoostedSaturatedGradient(const uint8_t * src, uint8_t * dx, uint8_t * dy,
            size_t stride, __m128i saturation, __m128i boost)
        {
            const __m128i s10 = Load<false>((__m128i*)(src - 1));
            const __m128i s12 = Load<false>((__m128i*)(src + 1));
            const __m128i s01 = Load<align>((__m128i*)(src - stride));
            const __m128i s21 = Load<align>((__m128i*)(src + stride));
            Store<align>((__m128i*)dx, TextureBoostedSaturatedGradient8(s10, s12, saturation, boost));
            Store<align>((__m128i*)dy, TextureBoostedSaturatedGradient8(s01, s21, saturation, boost));
        }

        template<bool align> void TextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride)
        {
            assert(width >= A && int(2)*saturation*boost <= 0xFF);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dx) && Aligned(dxStride) && Aligned(dy) && Aligned(dyStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __m128i _saturation = _mm_set1_epi16(saturation);
            __m128i _boost = _mm_set1_epi16(boost);

            memset(dx, 0, width);
            memset(dy, 0, width);
            src += srcStride;
            dx += dxStride;
            dy += dyStride;
            for (size_t row = 2; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    TextureBoostedSaturatedGradient<align>(src + col, dx + col, dy + col, srcStride, _saturation, _boost);
                if (width != alignedWidth)
                    TextureBoostedSaturatedGradient<false>(src + width - A, dx + width - A, dy + width - A, srcStride, _saturation, _boost);

                dx[0] = 0;
                dy[0] = 0;
                dx[width - 1] = 0;
                dy[width - 1] = 0;

                src += srcStride;
                dx += dxStride;
                dy += dyStride;
            }
            memset(dx, 0, width);
            memset(dy, 0, width);
        }

        void TextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dx) && Aligned(dxStride) && Aligned(dy) && Aligned(dyStride))
                TextureBoostedSaturatedGradient<true>(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
            else
                TextureBoostedSaturatedGradient<false>(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
        }
    }
#endif
}
