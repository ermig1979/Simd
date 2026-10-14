/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool mask> SIMD_INLINE void Int16ToGray(const int16_t * src, uint8_t * dst, __mmask64 tail = -1)
        {
            __m512i src0 = mask ? _mm512_maskz_loadu_epi16(__mmask32(tail >> 00), src + 00) : _mm512_loadu_si512(src + 00);
            __m512i src1 = mask ? _mm512_maskz_loadu_epi16(__mmask32(tail >> 32), src + HA) : _mm512_loadu_si512(src + HA);
            __m512i gray = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi16(src0, src1));
            if (mask)
                _mm512_mask_storeu_epi8(dst, tail, gray);
            else
                _mm512_storeu_si512(dst, gray);
        }

        SIMD_INLINE void Int16ToGray2(const int16_t * src, uint8_t * dst)
        {
            _mm512_storeu_si512(dst + 0 * A, _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi16(_mm512_loadu_si512(src + 0 * HA), _mm512_loadu_si512(src + 1 * HA))));
            _mm512_storeu_si512(dst + 1 * A, _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi16(_mm512_loadu_si512(src + 2 * HA), _mm512_loadu_si512(src + 3 * HA))));
        }

        void Int16ToGray(const int16_t * src, size_t width, size_t height, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            size_t fullAlignedWidth = AlignLo(width, DA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += DA)
                    Int16ToGray2(src + col, dst + col);
                for (; col < alignedWidth; col += A)
                    Int16ToGray<false>(src + col, dst + col);
                if (col < width)
                    Int16ToGray<true>(src + col, dst + col, tailMask);
                src += srcStride;
                dst += dstStride;
            }
        }

        void Int16ToGray(const uint8_t * src, size_t width, size_t height, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            Int16ToGray((const int16_t *)src, width, height, srcStride / sizeof(int16_t), dst, dstStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
