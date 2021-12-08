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
#include "Simd/SimdDeinterleave.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template<bool align> SIMD_INLINE void Uyvy422ToYuv420p(const uint8_t* uyvy0, size_t uyvyStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v)
        {
            __m256i uyvy00 = Load<align>((__m256i*)uyvy0 + 0);
            __m256i uyvy01 = Load<align>((__m256i*)uyvy0 + 1);
            __m256i uyvy02 = Load<align>((__m256i*)uyvy0 + 2);
            __m256i uyvy03 = Load<align>((__m256i*)uyvy0 + 3);

            Store<align>((__m256i*)y0 + 0, Avx2::Deinterleave8<1>(uyvy00, uyvy01));
            Store<align>((__m256i*)y0 + 1, Avx2::Deinterleave8<1>(uyvy02, uyvy03));

            const uint8_t* uyvy1 = uyvy0 + uyvyStride;
            __m256i uyvy10 = Load<align>((__m256i*)uyvy1 + 0);
            __m256i uyvy11 = Load<align>((__m256i*)uyvy1 + 1);
            __m256i uyvy12 = Load<align>((__m256i*)uyvy1 + 2);
            __m256i uyvy13 = Load<align>((__m256i*)uyvy1 + 3);

            uint8_t* y1 = y0 + yStride;
            Store<align>((__m256i*)y1 + 0, Avx2::Deinterleave8<1>(uyvy10, uyvy11));
            Store<align>((__m256i*)y1 + 1, Avx2::Deinterleave8<1>(uyvy12, uyvy13));

            __m256i uv0 = Deinterleave8<0>(_mm256_avg_epu8(uyvy00, uyvy10), _mm256_avg_epu8(uyvy01, uyvy11));
            __m256i uv1 = Deinterleave8<0>(_mm256_avg_epu8(uyvy02, uyvy12), _mm256_avg_epu8(uyvy03, uyvy13));

            Store<align>((__m256i*)u, Avx2::Deinterleave8<0>(uv0, uv1));
            Store<align>((__m256i*)v, Avx2::Deinterleave8<1>(uv0, uv1));
        }

        template<bool align> void Uyvy422ToYuv420p(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && width >= 2 * A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(uyvy) && Aligned(uyvyStride));
            }

            size_t width2A = AlignLo(width, 2 * A);
            size_t tailUyvy = width * 2 - 4 * A;
            size_t tailY = width - 2 * A;
            size_t tailUV = width / 2 - A;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUyvy = 0, colY = 0, colUV = 0; colY < width2A; colUyvy += 4 * A, colY += 2 * A, colUV += 1 * A)
                    Uyvy422ToYuv420p<align>(uyvy + colUyvy, uyvyStride, y + colY, yStride, u + colUV, v + colUV);
                if (width2A != width)
                    Uyvy422ToYuv420p<false>(uyvy + tailUyvy, uyvyStride, y + tailY, yStride, u + tailUV, v + tailUV);
                uyvy += 2 * uyvyStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
            }
        }

        void Uyvy422ToYuv420p(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(uyvy) && Aligned(uyvyStride))
                Uyvy422ToYuv420p<true>(uyvy, uyvyStride, width, height, y, yStride, u, uStride, v, vStride);
            else
                Uyvy422ToYuv420p<false>(uyvy, uyvyStride, width, height, y, yStride, u, uStride, v, vStride);
        }
    }
#endif
}
