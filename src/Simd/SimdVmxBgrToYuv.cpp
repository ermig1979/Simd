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
        template <bool align> SIMD_INLINE void LoadBgr(const uint8_t * p, v128_u8 & blue, v128_u8 & green, v128_u8 & red)
        {
            v128_u8 bgr[3];
            bgr[0] = Load<align>(p);
            bgr[1] = Load<align>(p + A);
            bgr[2] = Load<align>(p + DA);
            blue = BgrToBlue(bgr);
            green = BgrToGreen(bgr);
            red = BgrToRed(bgr);
        }

        SIMD_INLINE v128_s16 Average(const v128_u8 & s0, const v128_u8 & s1)
        {
            return (v128_s16)vec_sr(vec_add(vec_add(
                vec_add(vec_mule(s0, K8_01), vec_mulo(s0, K8_01)),
                vec_add(vec_mule(s1, K8_01), vec_mulo(s1, K8_01))), K16_0002), K16_0002);
        }

        template <bool align, bool first> SIMD_INLINE void BgrToYuv420p(const uint8_t * bgr0, size_t bgrStride,
            Storer<align> & y0, Storer<align> & y1, Storer<align> & u, Storer<align> & v)
        {
            const uint8_t * bgr1 = bgr0 + bgrStride;
            v128_u8 blue[2][2], green[2][2], red[2][2];
            LoadBgr<align>(bgr0, blue[0][0], green[0][0], red[0][0]);
            Store<align, first>(y0, BgrToY(blue[0][0], green[0][0], red[0][0]));
            LoadBgr<align>(bgr0 + 3 * A, blue[0][1], green[0][1], red[0][1]);
            Store<align, false>(y0, BgrToY(blue[0][1], green[0][1], red[0][1]));
            LoadBgr<align>(bgr1, blue[1][0], green[1][0], red[1][0]);
            Store<align, first>(y1, BgrToY(blue[1][0], green[1][0], red[1][0]));
            LoadBgr<align>(bgr1 + 3 * A, blue[1][1], green[1][1], red[1][1]);
            Store<align, false>(y1, BgrToY(blue[1][1], green[1][1], red[1][1]));

            v128_s16 blueAvg[2], greenAvg[2], redAvg[2];
            blueAvg[0] = Average(blue[0][0], blue[1][0]);
            blueAvg[1] = Average(blue[0][1], blue[1][1]);
            greenAvg[0] = Average(green[0][0], green[1][0]);
            greenAvg[1] = Average(green[0][1], green[1][1]);
            redAvg[0] = Average(red[0][0], red[1][0]);
            redAvg[1] = Average(red[0][1], red[1][1]);
            Store<align, first>(u, vec_pack(BgrToU(blueAvg[0], greenAvg[0], redAvg[0]), BgrToU(blueAvg[1], greenAvg[1], redAvg[1])));
            Store<align, first>(v, vec_pack(BgrToV(blueAvg[0], greenAvg[0], redAvg[0]), BgrToV(blueAvg[1], greenAvg[1], redAvg[1])));
        }

        template <bool align> void BgrToYuv420p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t alignedWidth = AlignLo(width, DA);
            const size_t A6 = A * 6;
            for (size_t row = 0; row < height; row += 2)
            {
                Storer<align> _y0(y), _y1(y + yStride), _u(u), _v(v);
                BgrToYuv420p<align, true>(bgr, bgrStride, _y0, _y1, _u, _v);
                for (size_t col = DA, colBgr = A6; col < alignedWidth; col += DA, colBgr += A6)
                    BgrToYuv420p<align, false>(bgr + colBgr, bgrStride, _y0, _y1, _u, _v);
                Flush(_y0, _y1, _u, _v);
                if (width != alignedWidth)
                {
                    size_t offset = width - DA;
                    Storer<false> _y0(y + offset), _y1(y + offset + yStride), _u(u + offset / 2), _v(v + offset / 2);
                    BgrToYuv420p<false, true>(bgr + offset * 3, bgrStride, _y0, _y1, _u, _v);
                    Flush(_y0, _y1, _u, _v);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgr += 2 * bgrStride;
            }
        }

        void BgrToYuv420p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToYuv420p<true>(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
            else
                BgrToYuv420p<false>(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
        }

        SIMD_INLINE v128_s16 Average(const v128_u8 & s)
        {
            return (v128_s16)vec_sr(vec_add(vec_add(vec_mule(s, K8_01), vec_mulo(s, K8_01)), K16_0001), K16_0001);
        }

        template <bool align, bool first> SIMD_INLINE void BgrToYuv422p(const uint8_t * bgr, Storer<align> & y, Storer<align> & u, Storer<align> & v)
        {
            v128_u8 blue[2], green[2], red[2];
            LoadBgr<align>(bgr, blue[0], green[0], red[0]);
            Store<align, first>(y, BgrToY(blue[0], green[0], red[0]));
            LoadBgr<align>(bgr + 3 * A, blue[1], green[1], red[1]);
            Store<align, false>(y, BgrToY(blue[1], green[1], red[1]));

            v128_s16 blueAvg[2], greenAvg[2], redAvg[2];
            blueAvg[0] = Average(blue[0]);
            blueAvg[1] = Average(blue[1]);
            greenAvg[0] = Average(green[0]);
            greenAvg[1] = Average(green[1]);
            redAvg[0] = Average(red[0]);
            redAvg[1] = Average(red[1]);
            Store<align, first>(u, vec_pack(BgrToU(blueAvg[0], greenAvg[0], redAvg[0]), BgrToU(blueAvg[1], greenAvg[1], redAvg[1])));
            Store<align, first>(v, vec_pack(BgrToV(blueAvg[0], greenAvg[0], redAvg[0]), BgrToV(blueAvg[1], greenAvg[1], redAvg[1])));
        }

        template <bool align> void BgrToYuv422p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t alignedWidth = AlignLo(width, DA);
            const size_t A6 = A * 6;
            for (size_t row = 0; row < height; ++row)
            {
                Storer<align> _y(y), _u(u), _v(v);
                BgrToYuv422p<align, true>(bgr, _y, _u, _v);
                for (size_t col = DA, colBgr = A6; col < alignedWidth; col += DA, colBgr += A6)
                    BgrToYuv422p<align, false>(bgr + colBgr, _y, _u, _v);
                Flush(_y, _u, _v);
                if (width != alignedWidth)
                {
                    size_t offset = width - DA;
                    Storer<false> _y(y + offset), _u(u + offset / 2), _v(v + offset / 2);
                    BgrToYuv422p<false, true>(bgr + offset * 3, _y, _u, _v);
                    Flush(_y, _u, _v);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void BgrToYuv422p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToYuv422p<true>(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
            else
                BgrToYuv422p<false>(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
        }

        template <bool align, bool first> SIMD_INLINE void BgrToYuv444p(const uint8_t * bgr,
            Storer<align> & y, Storer<align> & u, Storer<align> & v)
        {
            v128_u8 blue, green, red;
            LoadBgr<align>(bgr, blue, green, red);
            Store<align, first>(y, BgrToY(blue, green, red));
            Store<align, first>(u, BgrToU(blue, green, red));
            Store<align, first>(v, BgrToV(blue, green, red));
        }

        template <bool align> void BgrToYuv444p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            const size_t A3 = A * 3;
            for (size_t row = 0; row < height; ++row)
            {
                Storer<align> _y(y), _u(u), _v(v);
                BgrToYuv444p<align, true>(bgr, _y, _u, _v);
                for (size_t col = A, colBgr = A3; col < alignedWidth; col += A, colBgr += A3)
                    BgrToYuv444p<align, false>(bgr + colBgr, _y, _u, _v);
                Flush(_y, _u, _v);
                if (width != alignedWidth)
                {
                    size_t col = width - A;
                    Storer<false> _y(y + col), _u(u + col), _v(v + col);
                    BgrToYuv444p<false, true>(bgr + col * 3, _y, _u, _v);
                    Flush(_y, _u, _v);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void BgrToYuv444p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToYuv444p<true>(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
            else
                BgrToYuv444p<false>(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
        }
    }
#endif// SIMD_VMX_ENABLE
}
