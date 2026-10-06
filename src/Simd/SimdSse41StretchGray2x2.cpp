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
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE void StoreUnpacked(__m128i value, uint8_t * dst)
        {
            _mm_storeu_si128((__m128i*)(dst + 0), _mm_unpacklo_epi8(value, value));
            _mm_storeu_si128((__m128i*)(dst + A), _mm_unpackhi_epi8(value, value));
        }

        void StretchGray2x2(
            const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert(srcWidth * 2 == dstWidth && srcHeight * 2 == dstHeight && srcWidth >= A);

            size_t alignedWidth = AlignLo(srcWidth, A);
            for (size_t row = 0; row < srcHeight; ++row)
            {
                uint8_t * dstEven = dst;
                uint8_t * dstOdd = dst + dstStride;
                for (size_t srcCol = 0, dstCol = 0; srcCol < alignedWidth; srcCol += A, dstCol += DA)
                {
                    __m128i value = _mm_loadu_si128((__m128i*)(src + srcCol));
                    StoreUnpacked(value, dstEven + dstCol);
                    StoreUnpacked(value, dstOdd + dstCol);
                }
                if (alignedWidth != srcWidth)
                {
                    __m128i value = _mm_loadu_si128((__m128i*)(src + srcWidth - A));
                    StoreUnpacked(value, dstEven + dstWidth - 2 * A);
                    StoreUnpacked(value, dstOdd + dstWidth - 2 * A);
                }
                src += srcStride;
                dst += 2 * dstStride;
            }
        }
    }
#endif
}
