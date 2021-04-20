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

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> void DeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride));
            }

            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < bodyWidth; col += A, offset += DA)
                {
                    uint8x16x2_t _uv = Load2<align>(uv + offset);
                    Store<align>(u + col, _uv.val[0]);
                    Store<align>(v + col, _uv.val[1]);
                }
                if (tail)
                {
                    size_t col = width - A;
                    size_t offset = 2 * col;
                    uint8x16x2_t _uv = Load2<false>(uv + offset);
                    Store<false>(u + col, _uv.val[0]);
                    Store<false>(v + col, _uv.val[1]);
                }
                uv += uvStride;
                u += uStride;
                v += vStride;
            }
        }

        void DeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride))
                DeinterleaveUv<true>(uv, uvStride, width, height, u, uStride, v, vStride);
            else
                DeinterleaveUv<false>(uv, uvStride, width, height, u, uStride, v, vStride);
        }

        //---------------------------------------------------------------------

        template <bool align> void DeinterleaveBgr(const uint8_t * bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(b) && Aligned(bStride));
                assert(Aligned(g) && Aligned(gStride) && Aligned(r) && Aligned(rStride));
            }

            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            size_t A3 = A * 3;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < bodyWidth; col += A, offset += A3)
                {
                    uint8x16x3_t _bgr = Load3<align>(bgr + offset);
                    Store<align>(b + col, _bgr.val[0]);
                    Store<align>(g + col, _bgr.val[1]);
                    Store<align>(r + col, _bgr.val[2]);
                }
                if (tail)
                {
                    size_t col = width - A;
                    size_t offset = 3 * col;
                    uint8x16x3_t _bgr = Load3<false>(bgr + offset);
                    Store<false>(b + col, _bgr.val[0]);
                    Store<false>(g + col, _bgr.val[1]);
                    Store<false>(r + col, _bgr.val[2]);
                }
                bgr += bgrStride;
                b += bStride;
                g += gStride;
                r += rStride;
            }
        }

        void DeinterleaveBgr(const uint8_t * bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride)
        {
            if (Aligned(bgr) && Aligned(bgrStride) && Aligned(b) && Aligned(bStride) && Aligned(g) && Aligned(gStride) && Aligned(r) && Aligned(rStride))
                DeinterleaveBgr<true>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else
                DeinterleaveBgr<false>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
        }

        //---------------------------------------------------------------------

        template <bool align> void DeinterleaveBgra(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride, uint8_t * a, size_t aStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(b) && Aligned(bStride));
                assert(Aligned(g) && Aligned(gStride) && Aligned(r) && Aligned(rStride) && Aligned(a) && (Aligned(aStride) || a == NULL));
            }

            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            if (a)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0, offset = 0; col < bodyWidth; col += A, offset += QA)
                    {
                        uint8x16x4_t _bgra = Load4<align>(bgra + offset);
                        Store<align>(b + col, _bgra.val[0]);
                        Store<align>(g + col, _bgra.val[1]);
                        Store<align>(r + col, _bgra.val[2]);
                        Store<align>(a + col, _bgra.val[3]);
                    }
                    if (tail)
                    {
                        size_t col = width - A;
                        size_t offset = 4 * col;
                        uint8x16x4_t _bgra = Load4<false>(bgra + offset);
                        Store<false>(b + col, _bgra.val[0]);
                        Store<false>(g + col, _bgra.val[1]);
                        Store<false>(r + col, _bgra.val[2]);
                        Store<false>(a + col, _bgra.val[3]);
                    }
                    bgra += bgraStride;
                    b += bStride;
                    g += gStride;
                    r += rStride;
                    a += aStride;
                }
            }
            else
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0, offset = 0; col < bodyWidth; col += A, offset += QA)
                    {
                        uint8x16x4_t _bgra = Load4<align>(bgra + offset);
                        Store<align>(b + col, _bgra.val[0]);
                        Store<align>(g + col, _bgra.val[1]);
                        Store<align>(r + col, _bgra.val[2]);
                    }
                    if (tail)
                    {
                        size_t col = width - A;
                        size_t offset = 4 * col;
                        uint8x16x4_t _bgra = Load4<false>(bgra + offset);
                        Store<false>(b + col, _bgra.val[0]);
                        Store<false>(g + col, _bgra.val[1]);
                        Store<false>(r + col, _bgra.val[2]);
                    }
                    bgra += bgraStride;
                    b += bStride;
                    g += gStride;
                    r += rStride;
                }
            }
        }

        void DeinterleaveBgra(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride, uint8_t * a, size_t aStride)
        {
            if (Aligned(bgra) && Aligned(bgraStride) && Aligned(b) && Aligned(bStride) &&
                Aligned(g) && Aligned(gStride) && Aligned(r) && Aligned(rStride) && Aligned(a) && (Aligned(aStride) || a == NULL))
                DeinterleaveBgra<true>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else
                DeinterleaveBgra<false>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}
