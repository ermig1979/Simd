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
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE void GrayToBgr(uint8_t * bgr, __m256i gray)
        {
            _mm256_storeu_si256((__m256i*)bgr + 0, GrayToBgr<0>(gray));
            _mm256_storeu_si256((__m256i*)bgr + 1, GrayToBgr<1>(gray));
            _mm256_storeu_si256((__m256i*)bgr + 2, GrayToBgr<2>(gray));
        }

        void GrayToBgr(const uint8_t * gray, size_t width, size_t height, size_t grayStride, uint8_t *bgr, size_t bgrStride)
        {
            assert(width >= A);

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    __m256i _gray = _mm256_loadu_si256((__m256i*)(gray + col));
                    GrayToBgr(bgr + 3 * col, _gray);
                }
                if (alignedWidth != width)
                {
                    __m256i _gray = _mm256_loadu_si256((__m256i*)(gray + width - A));
                    GrayToBgr(bgr + 3 * (width - A), _gray);
                }
                gray += grayStride;
                bgr += bgrStride;
            }
        }
    }
#endif// SIMD_AVX2_ENABLE
}
