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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> void InterleaveUv(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * uv, size_t uvStride)
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
                    uint8x16x2_t _uv;
                    _uv.val[0] = Load<align>(u + col);
                    _uv.val[1] = Load<align>(v + col);
                    Store2<align>(uv + offset, _uv);
                }
                if (tail)
                {
                    size_t col = width - A;
                    size_t offset = 2 * col;
                    uint8x16x2_t _uv;
                    _uv.val[0] = Load<false>(u + col);
                    _uv.val[1] = Load<false>(v + col);
                    Store2<false>(uv + offset, _uv);
                }
                u += uStride;
                v += vStride;
                uv += uvStride;
            }
        }

        void InterleaveUv(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * uv, size_t uvStride)
        {
            if (Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride))
                InterleaveUv<true>(u, uStride, v, vStride, width, height, uv, uvStride);
            else
                InterleaveUv<false>(u, uStride, v, vStride, width, height, uv, uvStride);
        }

        template <bool align> void InterleaveBgr(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(b) && Aligned(bStride));
                assert(Aligned(g) && Aligned(gStride) && Aligned(r) && Aligned(rStride));
            }

            size_t A3 = A * 3;
            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < bodyWidth; col += A, offset += A3)
                {
                    uint8x16x3_t _bgr;
                    _bgr.val[0] = Load<align>(b + col);
                    _bgr.val[1] = Load<align>(g + col);
                    _bgr.val[2] = Load<align>(r + col);
                    Store3<align>(bgr + offset, _bgr);
                }
                if (tail)
                {
                    size_t col = width - A;
                    size_t offset = 3 * col;
                    uint8x16x3_t _bgr;
                    _bgr.val[0] = Load<false>(b + col);
                    _bgr.val[1] = Load<false>(g + col);
                    _bgr.val[2] = Load<false>(r + col);
                    Store3<align>(bgr + offset, _bgr);
                }
                b += bStride;
                g += gStride;
                r += rStride;
                bgr += bgrStride;
            }
        }

        void InterleaveBgr(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            if (Aligned(bgr) && Aligned(bgrStride) && Aligned(b) && Aligned(bStride) && Aligned(g) && Aligned(gStride) && Aligned(r) && Aligned(rStride))
                InterleaveBgr<true>(b, bStride, g, gStride, r, rStride, width, height, bgr, bgrStride);
            else
                InterleaveBgr<false>(b, bStride, g, gStride, r, rStride, width, height, bgr, bgrStride);
        }

        template <bool align> void InterleaveBgra(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, const uint8_t * a, size_t aStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(b) && Aligned(bStride));
                assert(Aligned(g) && Aligned(gStride) && Aligned(r) && Aligned(rStride) && Aligned(a) && Aligned(aStride));
            }

            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < bodyWidth; col += A, offset += QA)
                {
                    uint8x16x4_t _bgra;
                    _bgra.val[0] = Load<align>(b + col);
                    _bgra.val[1] = Load<align>(g + col);
                    _bgra.val[2] = Load<align>(r + col);
                    _bgra.val[3] = Load<align>(a + col);
                    Store4<align>(bgra + offset, _bgra);
                }
                if (tail)
                {
                    size_t col = width - A;
                    size_t offset = 4 * col;
                    uint8x16x4_t _bgra;
                    _bgra.val[0] = Load<false>(b + col);
                    _bgra.val[1] = Load<false>(g + col);
                    _bgra.val[2] = Load<false>(r + col);
                    _bgra.val[3] = Load<false>(a + col);
                    Store4<align>(bgra + offset, _bgra);
                }
                b += bStride;
                g += gStride;
                r += rStride;
                a += aStride;
                bgra += bgraStride;
            }
        }

        void InterleaveBgra(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, const uint8_t * a, size_t aStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride)
        {
            if (Aligned(bgra) && Aligned(bgraStride) && Aligned(b) && Aligned(bStride) && Aligned(g) && Aligned(gStride) && Aligned(r) && Aligned(rStride) && Aligned(a) && Aligned(aStride))
                InterleaveBgra<true>(b, bStride, g, gStride, r, rStride, a, aStride, width, height, bgra, bgraStride);
            else
                InterleaveBgra<false>(b, bStride, g, gStride, r, rStride, a, aStride, width, height, bgra, bgraStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}
