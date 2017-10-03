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
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        const v128_u8 K8_PERM_MERGE = SIMD_VEC_SETR_EPI8(0x01, 0x11, 0x03, 0x13, 0x05, 0x15, 0x07, 0x17, 0x09, 0x19, 0x0B, 0x1B, 0x0D, 0x1D, 0x0F, 0x1F);

        template <bool align, bool first>
        SIMD_INLINE void AdjustedYuvToBgra(const v128_s16 & y, const v128_s16 & u, const v128_s16 & v, const v128_u16 & a, Storer<align> & bgra)
        {
            const v128_u16 b = AdjustedYuvToBlue(y, u);
            const v128_u16 g = AdjustedYuvToGreen(y, u, v);
            const v128_u16 r = AdjustedYuvToRed(y, v);
            const v128_u16 bg = vec_perm(b, g, K8_PERM_MERGE);
            const v128_u16 ra = vec_perm(r, a, K8_PERM_MERGE);
            Store<align, first>(bgra, (v128_u8)UnpackLoU16(ra, bg));
            Store<align, false>(bgra, (v128_u8)UnpackHiU16(ra, bg));
        }

        template <bool align, bool first>
        SIMD_INLINE void YuvToBgra(const v128_u8 & y, const v128_u8 & u, const v128_u8 & v, const v128_u16 & a, Storer<align> & bgra)
        {
            AdjustedYuvToBgra<align, first>(AdjustY(UnpackLoU8(y)), AdjustUV(UnpackLoU8(u)), AdjustUV(UnpackLoU8(v)), a, bgra);
            AdjustedYuvToBgra<align, false>(AdjustY(UnpackHiU8(y)), AdjustUV(UnpackHiU8(u)), AdjustUV(UnpackHiU8(v)), a, bgra);
        }

        template <bool align, bool first>
        SIMD_INLINE void Yuv422pToBgra(const uint8_t * y, const v128_u8 & u, const v128_u8 & v, const v128_u16 & a, Storer<align> & bgra)
        {
            YuvToBgra<align, first>(Load<align>(y + 0), (v128_u8)UnpackLoU8(u, u), (v128_u8)UnpackLoU8(v, v), a, bgra);
            YuvToBgra<align, false>(Load<align>(y + A), (v128_u8)UnpackHiU8(u, u), (v128_u8)UnpackHiU8(v, v), a, bgra);
        }

        template <bool align> void Yuv420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            v128_u16 a = SetU16(alpha);
            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; row += 2)
            {
                Storer<align> _bgra0(bgra), _bgra1(bgra + bgraStride);
                v128_u8 _u = Load<align>(u);
                v128_u8 _v = Load<align>(v);
                Yuv422pToBgra<align, true>(y, _u, _v, a, _bgra0);
                Yuv422pToBgra<align, true>(y + yStride, _u, _v, a, _bgra1);
                for (size_t colUV = A, colY = DA; colY < bodyWidth; colY += DA, colUV += A)
                {
                    v128_u8 _u = Load<align>(u + colUV);
                    v128_u8 _v = Load<align>(v + colUV);
                    Yuv422pToBgra<align, false>(y + colY, _u, _v, a, _bgra0);
                    Yuv422pToBgra<align, false>(y + colY + yStride, _u, _v, a, _bgra1);
                }
                Flush(_bgra0, _bgra1);

                if (tail)
                {
                    size_t offset = width - DA;
                    Storer<false> _bgra0(bgra + 4 * offset), _bgra1(bgra + 4 * offset + bgraStride);
                    v128_u8 _u = Load<false>(u + offset / 2);
                    v128_u8 _v = Load<false>(v + offset / 2);
                    Yuv422pToBgra<false, true>(y + offset, _u, _v, a, _bgra0);
                    Yuv422pToBgra<false, true>(y + offset + yStride, _u, _v, a, _bgra1);
                    Flush(_bgra0, _bgra1);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        void Yuv420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv420pToBgra<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
            else
                Yuv420pToBgra<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
        }

        template <bool align, bool first>
        SIMD_INLINE void Yuv422pToBgra(const uint8_t * y, const uint8_t * u, const uint8_t * v, const v128_u16 & a, Storer<align> & bgra)
        {
            Yuv422pToBgra<align, first>(y, Load<align>(u), Load<align>(v), a, bgra);
        }

        template <bool align> void Yuv422pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            v128_u16 a = SetU16(alpha);
            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                Storer<align> _bgra(bgra);
                Yuv422pToBgra<align, true>(y, u, v, a, _bgra);
                for (size_t colUV = A, colY = DA; colY < bodyWidth; colY += DA, colUV += A)
                    Yuv422pToBgra<align, false>(y + colY, u + colUV, v + colUV, a, _bgra);
                Flush(_bgra);

                if (tail)
                {
                    size_t offset = width - DA;
                    Storer<false> _bgra(bgra + 4 * offset);
                    Yuv422pToBgra<false, true>(y + offset, u + offset / 2, v + offset / 2, a, _bgra);
                    Flush(_bgra);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void Yuv422pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv422pToBgra<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
            else
                Yuv422pToBgra<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
        }

        template <bool align, bool first>
        SIMD_INLINE void Yuv444pToBgra(const uint8_t * y, const uint8_t * u, const uint8_t * v, const v128_u16 & a, Storer<align> & bgra)
        {
            YuvToBgra<align, first>(Load<align>(y), Load<align>(u), Load<align>(v), a, bgra);
        }

        template <bool align> void Yuv444pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            v128_u16 a = SetU16(alpha);
            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                Storer<align> _bgra(bgra);
                Yuv444pToBgra<align, true>(y, u, v, a, _bgra);
                for (size_t col = A; col < bodyWidth; col += A)
                    Yuv444pToBgra<align, false>(y + col, u + col, v + col, a, _bgra);
                Flush(_bgra);

                if (tail)
                {
                    size_t col = width - A;
                    Storer<false> _bgra(bgra + 4 * col);
                    Yuv444pToBgra<false, true>(y + col, u + col, v + col, a, _bgra);
                    Flush(_bgra);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void Yuv444pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv444pToBgra<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
            else
                Yuv444pToBgra<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
        }
    }
#endif// SIMD_VMX_ENABLE
}
