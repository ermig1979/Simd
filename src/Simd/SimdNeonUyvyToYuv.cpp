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
#include "Simd/SimdDeinterleave.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template<bool align> SIMD_INLINE void Uyvy422ToYuv420p(const uint8_t* uyvy0, size_t uyvyStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v)
        {
            uint8x16x2_t uyvy00 = Load2<align>(uyvy0 + 0 * A);
            uint8x16x2_t uyvy02 = Load2<align>(uyvy0 + 2 * A);
            Store<align>(y0 + 0 * A, uyvy00.val[1]);
            Store<align>(y0 + 1 * A, uyvy02.val[1]);

            const uint8_t* uyvy1 = uyvy0 + uyvyStride;
            uint8x16x2_t uyvy10 = Load2<align>(uyvy1 + 0 * A);
            uint8x16x2_t uyvy12 = Load2<align>(uyvy1 + 2 * A);
            uint8_t* y1 = y0 + yStride;
            Store<align>(y1 + 0 * A, uyvy10.val[1]);
            Store<align>(y1 + 1 * A, uyvy12.val[1]);

            uint8x16x2_t uv0 = vuzpq_u8(uyvy00.val[0], uyvy02.val[0]);
            uint8x16x2_t uv1 = vuzpq_u8(uyvy10.val[0], uyvy12.val[0]);
            Store<align>(u, vrhaddq_u8(uv0.val[0], uv1.val[0]));
            Store<align>(v, vrhaddq_u8(uv0.val[1], uv1.val[1]));
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
