/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_SSSE3_ENABLE    
    namespace Ssse3
    {
        template <bool align> SIMD_INLINE void InterleaveBgr(const uint8_t * b, const uint8_t * g, const uint8_t * r, size_t offset, uint8_t * bgr)
        {
            __m128i _b = Load<align>((__m128i*)(b + offset));
            __m128i _g = Load<align>((__m128i*)(g + offset));
            __m128i _r = Load<align>((__m128i*)(r + offset));
            Store<align>((__m128i*)bgr + 0, InterleaveBgr<0>(_b, _g, _r));
            Store<align>((__m128i*)bgr + 1, InterleaveBgr<1>(_b, _g, _r));
            Store<align>((__m128i*)bgr + 2, InterleaveBgr<2>(_b, _g, _r));
        }

        template <bool align> void InterleaveBgr(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(b) && Aligned(bStride) && Aligned(g) && Aligned(gStride));
                assert(Aligned(r) && Aligned(rStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            size_t A3 = A * 3;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += A3)
                    InterleaveBgr<align>(b, g, r, col, bgr + offset);
                if (tail)
                    InterleaveBgr<false>(b, g, r, width - A, bgr + 3*(width - A));
                b += bStride;
                g += gStride;
                r += rStride;
                bgr += bgrStride;
            }
        }

        void InterleaveBgr(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            if (Aligned(b) && Aligned(bStride) && Aligned(g) && Aligned(gStride)
                && Aligned(r) && Aligned(rStride) && Aligned(bgr) && Aligned(bgrStride))
                InterleaveBgr<true>(b, bStride, g, gStride, r, rStride, width, height, bgr, bgrStride);
            else
                InterleaveBgr<false>(b, bStride, g, gStride, r, rStride, width, height, bgr, bgrStride);
        }
    }
#endif// SIMD_SSSE3_ENABLE
}