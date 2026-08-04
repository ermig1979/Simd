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
#include "Simd/SimdStore.h"
#include "Simd/SimdAlphaBlending.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template <bool align> SIMD_INLINE void CreateMask(const __m128i & vertical, const uint8_t * horizontal, uint8_t * dst)
        {
            __m128i _horizontal = Load<align>((__m128i*)horizontal);
            __m128i lo = Divide16uBy255(_mm_mullo_epi16(vertical, _mm_unpacklo_epi8(_horizontal, K_ZERO)));
            __m128i hi = Divide16uBy255(_mm_mullo_epi16(vertical, _mm_unpackhi_epi8(_horizontal, K_ZERO)));
            Store<align>((__m128i*)dst, _mm_packus_epi16(lo, hi));
        }

        template <bool align> void CreateMask(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(horizontal) && Aligned(dst) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                __m128i _vertical = _mm_set1_epi16(vertical[row]);
                for (size_t col = 0; col < alignedWidth; col += A)
                    CreateMask<align>(_vertical, horizontal + col, dst + col);
                if (alignedWidth != width)
                    CreateMask<false>(_vertical, horizontal + width - A, dst + width - A);
                dst += stride;
            }
        }

        void CreateMask(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
        {
            if (Aligned(horizontal) && Aligned(dst) && Aligned(stride))
                CreateMask<true>(vertical, horizontal, dst, stride, width, height);
            else
                CreateMask<false>(vertical, horizontal, dst, stride, width, height);
        }
    }
#endif
}
