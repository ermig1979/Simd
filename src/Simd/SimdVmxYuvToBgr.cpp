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
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        template <bool align, bool first>
        SIMD_INLINE void YuvToBgr(const v128_u8 & y, const v128_u8 & u, const v128_u8 & v, Storer<align> & bgr)
        {
            const v128_u8 blue = YuvToBlue(y, u);
            const v128_u8 green = YuvToGreen(y, u, v);
            const v128_u8 red = YuvToRed(y, v);

            Store<align, first>(bgr, InterleaveBgr<0>(blue, green, red));
            Store<align, false>(bgr, InterleaveBgr<1>(blue, green, red));
            Store<align, false>(bgr, InterleaveBgr<2>(blue, green, red));
        }

        template <bool align, bool first>
        SIMD_INLINE void Yuv422pToBgr(const uint8_t * y, const v128_u8 & u, const v128_u8 & v, Storer<align> & bgr)
        {
            YuvToBgr<align, first>(Load<align>(y + 0), (v128_u8)UnpackLoU8(u, u), (v128_u8)UnpackLoU8(v, v), bgr);
            YuvToBgr<align, false>(Load<align>(y + A), (v128_u8)UnpackHiU8(u, u), (v128_u8)UnpackHiU8(v, v), bgr);
        }

        template <bool align> void Yuv420pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; row += 2)
            {
                Storer<align> _bgr0(bgr), _bgr1(bgr + bgrStride);
                v128_u8 _u = Load<align>(u);
                v128_u8 _v = Load<align>(v);
                Yuv422pToBgr<align, true>(y, _u, _v, _bgr0);
                Yuv422pToBgr<align, true>(y + yStride, _u, _v, _bgr1);
                for (size_t colUV = A, colY = DA; colY < bodyWidth; colY += DA, colUV += A)
                {
                    v128_u8 _u = Load<align>(u + colUV);
                    v128_u8 _v = Load<align>(v + colUV);
                    Yuv422pToBgr<align, false>(y + colY, _u, _v, _bgr0);
                    Yuv422pToBgr<align, false>(y + colY + yStride, _u, _v, _bgr1);
                }
                Flush(_bgr0, _bgr1);

                if (tail)
                {
                    size_t offset = width - DA;
                    Storer<false> _bgr0(bgr + 3 * offset), _bgr1(bgr + 3 * offset + bgrStride);
                    v128_u8 _u = Load<false>(u + offset / 2);
                    v128_u8 _v = Load<false>(v + offset / 2);
                    Yuv422pToBgr<false, true>(y + offset, _u, _v, _bgr0);
                    Yuv422pToBgr<false, true>(y + offset + yStride, _u, _v, _bgr1);
                    Flush(_bgr0, _bgr1);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgr += 2 * bgrStride;
            }
        }

        void Yuv420pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                Yuv420pToBgr<true>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
            else
                Yuv420pToBgr<false>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
        }

        template <bool align, bool first>
        SIMD_INLINE void Yuv422pToBgr(const uint8_t * y, const uint8_t * u, const uint8_t * v, Storer<align> & bgr)
        {
            Yuv422pToBgr<align, first>(y, Load<align>(u), Load<align>(v), bgr);
        }

        template <bool align> void Yuv422pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                Storer<align> _bgr(bgr);
                Yuv422pToBgr<align, true>(y, u, v, _bgr);
                for (size_t colUV = A, colY = DA; colY < bodyWidth; colY += DA, colUV += A)
                    Yuv422pToBgr<align, false>(y + colY, u + colUV, v + colUV, _bgr);
                Flush(_bgr);
                if (tail)
                {
                    size_t offset = width - DA;
                    Storer<false> _bgr(bgr + 3 * offset);
                    Yuv422pToBgr<false, true>(y + offset, u + offset / 2, v + offset / 2, _bgr);
                    Flush(_bgr);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void Yuv422pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                Yuv422pToBgr<true>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
            else
                Yuv422pToBgr<false>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
        }

        template <bool align, bool first>
        SIMD_INLINE void Yuv444pToBgr(const uint8_t * y, const uint8_t * u, const uint8_t * v, Storer<align> & bgr)
        {
            YuvToBgr<align, first>(Load<align>(y), Load<align>(u), Load<align>(v), bgr);
        }

        template <bool align> void Yuv444pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            size_t A3 = A * 3;
            for (size_t row = 0; row < height; ++row)
            {
                Storer<align> _bgr(bgr);
                Yuv444pToBgr<align, true>(y, u, v, _bgr);
                for (size_t col = A; col < bodyWidth; col += A)
                    Yuv444pToBgr<align, false>(y + col, u + col, v + col, _bgr);
                Flush(_bgr);
                if (tail)
                {
                    size_t col = width - A;
                    Storer<false> _bgr(bgr + 3 * col);
                    Yuv444pToBgr<false, true>(y + col, u + col, v + col, _bgr);
                    Flush(_bgr);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void Yuv444pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                Yuv444pToBgr<true>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
            else
                Yuv444pToBgr<false>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
        }
    }
#endif// SIMD_VMX_ENABLE
}
