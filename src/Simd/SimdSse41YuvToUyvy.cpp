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
        template<bool align> SIMD_INLINE void Yuv420pToUyvy422(const uint8_t* y0, size_t yStride, 
            const uint8_t* u, const uint8_t* v, uint8_t* uyvy0, size_t uyvyStride)
        {
            __m128i u0 = Load<align>((__m128i*)u);
            __m128i v0 = Load<align>((__m128i*)v);
            __m128i uv0 = _mm_unpacklo_epi8(u0, v0);
            __m128i uv1 = _mm_unpackhi_epi8(u0, v0);

            __m128i y00 = Load<align>((__m128i*)y0 + 0);
            __m128i y01 = Load<align>((__m128i*)y0 + 1);
            Store<align>((__m128i*)uyvy0 + 0, _mm_unpacklo_epi8(uv0, y00));
            Store<align>((__m128i*)uyvy0 + 1, _mm_unpackhi_epi8(uv0, y00));
            Store<align>((__m128i*)uyvy0 + 2, _mm_unpacklo_epi8(uv1, y01));
            Store<align>((__m128i*)uyvy0 + 3, _mm_unpackhi_epi8(uv1, y01));

            const uint8_t* y1 = y0 + yStride;
            __m128i y10 = Load<align>((__m128i*)y1 + 0);
            __m128i y11 = Load<align>((__m128i*)y1 + 1);
            uint8_t* uyvy1 = uyvy0 + uyvyStride;
            Store<align>((__m128i*)uyvy1 + 0, _mm_unpacklo_epi8(uv0, y10));
            Store<align>((__m128i*)uyvy1 + 1, _mm_unpackhi_epi8(uv0, y10));
            Store<align>((__m128i*)uyvy1 + 2, _mm_unpacklo_epi8(uv1, y11));
            Store<align>((__m128i*)uyvy1 + 3, _mm_unpackhi_epi8(uv1, y11));
        }

        template<bool align> void Yuv420pToUyvy422(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, 
            const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* uyvy, size_t uyvyStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && width >= 2 * A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(uyvy) && Aligned(uyvyStride));
            }

            size_t width2A = AlignLo(width, 2 * A);
            size_t tailY = width - 2 * A;
            size_t tailUV = width / 2 - A;
            size_t tailUyvy = width * 2 - 4 * A;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colY = 0, colUV = 0, colUyvy = 0; colY < width2A; colY += 2 * A, colUV += 1 * A, colUyvy += 4 * A)
                    Yuv420pToUyvy422<align>(y + colY, yStride, u + colUV, v + colUV, uyvy + colUyvy, uyvyStride);
                if (width2A != width)
                    Yuv420pToUyvy422<false>(y + tailY, yStride, u + tailUV, v + tailUV, uyvy + tailUyvy, uyvyStride);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                uyvy += 2 * uyvyStride;
            }
        }

        void Yuv420pToUyvy422(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, 
            const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* uyvy, size_t uyvyStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(uyvy) && Aligned(uyvyStride))
                Yuv420pToUyvy422<true>(y, yStride, u, uStride, v, vStride, width, height, uyvy, uyvyStride);
            else
                Yuv420pToUyvy422<false>(y, yStride, u, uStride, v, vStride, width, height, uyvy, uyvyStride);
        }
    }
#endif
}
