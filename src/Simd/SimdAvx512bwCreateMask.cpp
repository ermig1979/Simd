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
#include "Simd/SimdUnpack.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE __m512i CreateMask(const __m512i & vertical, const __m512i & horizontalLo, const __m512i & horizontalHi)
        {
            __m512i lo = Divide16uBy255(_mm512_mullo_epi16(vertical, horizontalLo));
            __m512i hi = Divide16uBy255(_mm512_mullo_epi16(vertical, horizontalHi));
            return _mm512_packus_epi16(lo, hi);
        }

        template <bool align, bool mask> SIMD_INLINE void CreateMask2(const __m512i & vertical0, const __m512i & vertical1, const uint8_t * horizontal, uint8_t * dst, size_t stride, __mmask64 m = -1)
        {
            __m512i _horizontal = Load<align, mask>(horizontal, m);
            __m512i horizontalLo = UnpackU8<0>(_horizontal);
            __m512i horizontalHi = UnpackU8<1>(_horizontal);
            Store<align, mask>(dst + 0 * stride, CreateMask(vertical0, horizontalLo, horizontalHi), m);
            Store<align, mask>(dst + 1 * stride, CreateMask(vertical1, horizontalLo, horizontalHi), m);
        }

        template <bool align, bool mask> SIMD_INLINE void CreateMask1(const __m512i & vertical, const uint8_t * horizontal, uint8_t * dst, __mmask64 m = -1)
        {
            __m512i _horizontal = Load<align, mask>(horizontal, m);
            Store<align, mask>(dst, CreateMask(vertical, UnpackU8<0>(_horizontal), UnpackU8<1>(_horizontal)), m);
        }

        template <bool align> void CreateMask(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
        {
            if (align)
                assert(Aligned(horizontal) && Aligned(dst) && Aligned(stride));

            size_t alignedHeight = Simd::AlignLo(height, 2);
            size_t alignedWidth = Simd::AlignLo(width, A);
            __mmask64 tailMask = __mmask64(-1) >> (A + alignedWidth - width);
            size_t row = 0;
            for (; row < alignedHeight; row += 2)
            {
                __m512i vertical0 = _mm512_set1_epi16(vertical[row + 0]);
                __m512i vertical1 = _mm512_set1_epi16(vertical[row + 1]);
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    CreateMask2<align, false>(vertical0, vertical1, horizontal + col, dst + col, stride);
                if (col < width)
                    CreateMask2<false, true>(vertical0, vertical1, horizontal + col, dst + col, stride, tailMask);
                dst += 2 * stride;
            }
            for (; row < height; ++row)
            {
                __m512i _vertical = _mm512_set1_epi16(vertical[row]);
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    CreateMask1<align, false>(_vertical, horizontal + col, dst + col);
                if (col < width)
                    CreateMask1<false, true>(_vertical, horizontal + col, dst + col, tailMask);
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
#endif// SIMD_AVX512BW_ENABLE
}
