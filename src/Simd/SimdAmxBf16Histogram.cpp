/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_AMXBF16_ENABLE    
    namespace AmxBf16
    {
        template<bool align, bool mask> SIMD_INLINE void ChangeColors(const uint8_t* src, const __m512i colors[4], uint8_t* dst, __mmask64 tail = __mmask64(-1))
        {
            __m512i _src = Load<align, mask>(src, tail);
            __mmask64 blend = _mm512_cmpge_epu8_mask(_src, K8_80);
            __m512i perm0 = _mm512_permutex2var_epi8(colors[0], _src, colors[1]);
            __m512i perm1 = _mm512_permutex2var_epi8(colors[2], _src, colors[3]);
            Store<align, mask>(dst, _mm512_mask_blend_epi8(blend, perm0, perm1), tail);
        }

        template< bool align> void ChangeColors(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* colors, uint8_t* dst, size_t dstStride)
        {
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            __m512i _colors[4];
            _colors[0] = Load<false>(colors + 0 * A);
            _colors[1] = Load<false>(colors + 1 * A);
            _colors[2] = Load<false>(colors + 2 * A);
            _colors[3] = Load<false>(colors + 3 * A);

            size_t widthA = Simd::AlignLo(width, A);
            __mmask64 tail = TailMask64(width - widthA);

            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    ChangeColors<align, false>(src + col, _colors, dst + col);
                if (col < width)
                    ChangeColors<align, true>(src + col, _colors, dst + col, tail);
                src += srcStride;
                dst += dstStride;
            }
        }

        void ChangeColors(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* colors, uint8_t* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                ChangeColors<true>(src, srcStride, width, height, colors, dst, dstStride);
            else
                ChangeColors<false>(src, srcStride, width, height, colors, dst, dstStride);
        }

        //-------------------------------------------------------------------------------------------------

        void NormalizeHistogram(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            uint32_t histogram[HISTOGRAM_SIZE];
            Base::Histogram(src, width, height, srcStride, histogram);

            uint8_t colors[HISTOGRAM_SIZE];
            Base::NormalizedColors(histogram, colors);

            ChangeColors(src, srcStride, width, height, colors, dst, dstStride);
        }
    }
#endif
}
