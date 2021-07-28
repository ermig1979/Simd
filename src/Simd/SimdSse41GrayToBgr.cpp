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

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template <bool align> SIMD_INLINE void GrayToBgr(uint8_t * bgr, __m128i gray)
        {
            Store<align>((__m128i*)bgr + 0, _mm_shuffle_epi8(gray, K8_SHUFFLE_GRAY_TO_BGR0));
            Store<align>((__m128i*)bgr + 1, _mm_shuffle_epi8(gray, K8_SHUFFLE_GRAY_TO_BGR1));
            Store<align>((__m128i*)bgr + 2, _mm_shuffle_epi8(gray, K8_SHUFFLE_GRAY_TO_BGR2));
        }

        template <bool align> void GrayToBgr(const uint8_t *gray, size_t width, size_t height, size_t grayStride, uint8_t *bgr, size_t bgrStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(gray) && Aligned(grayStride));

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    __m128i _gray = Load<align>((__m128i*)(gray + col));
                    GrayToBgr<align>(bgr + 3 * col, _gray);
                }
                if (alignedWidth != width)
                {
                    __m128i _gray = Load<false>((__m128i*)(gray + width - A));
                    GrayToBgr<false>(bgr + 3 * (width - A), _gray);
                }
                gray += grayStride;
                bgr += bgrStride;
            }
        }

        void GrayToBgr(const uint8_t *gray, size_t width, size_t height, size_t grayStride, uint8_t *bgr, size_t bgrStride)
        {
            if (Aligned(bgr) && Aligned(gray) && Aligned(bgrStride) && Aligned(grayStride))
                GrayToBgr<true>(gray, width, height, grayStride, bgr, bgrStride);
            else
                GrayToBgr<false>(gray, width, height, grayStride, bgr, bgrStride);
        }
    }
#endif
}
