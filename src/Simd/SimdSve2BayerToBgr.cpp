/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdBayer.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        struct BayerPair
        {
            svuint8_t val[2];
        };

        SIMD_INLINE svuint8_t Average(const svuint8_t& a, const svuint8_t& b)
        {
            svuint16_t lo = svadd_n_u16_x(svptrue_b16(), svaddlb_u16(a, b), 1);
            svuint16_t hi = svadd_n_u16_x(svptrue_b16(), svaddlt_u16(a, b), 1);
            return svshrnt_n_u16(svshrnb_n_u16(lo, 1), hi, 1);
        }

        SIMD_INLINE svuint8_t Average(const svuint8_t& a, const svuint8_t& b, const svuint8_t& c, const svuint8_t& d)
        {
            svuint16_t lo = svadd_u16_x(svptrue_b16(), svaddlb_u16(a, b), svaddlb_u16(c, d));
            svuint16_t hi = svadd_u16_x(svptrue_b16(), svaddlt_u16(a, b), svaddlt_u16(c, d));
            lo = svadd_n_u16_x(svptrue_b16(), lo, 2);
            hi = svadd_n_u16_x(svptrue_b16(), hi, 2);
            return svshrnt_n_u16(svshrnb_n_u16(lo, 2), hi, 2);
        }

        SIMD_INLINE svuint8_t BayerToGreen(const svuint8_t& greenLeft, const svuint8_t& greenTop, const svuint8_t& greenRight, const svuint8_t& greenBottom,
            const svuint8_t& blueOrRedLeft, const svuint8_t& blueOrRedTop, const svuint8_t& blueOrRedRight, const svuint8_t& blueOrRedBottom)
        {
            const svbool_t pg = svptrue_b8();
            svuint8_t verticalAbsDifference = svabd_u8_x(pg, blueOrRedTop, blueOrRedBottom);
            svuint8_t horizontalAbsDifference = svabd_u8_x(pg, blueOrRedLeft, blueOrRedRight);
            svuint8_t green = Average(greenLeft, greenTop, greenRight, greenBottom);
            green = svsel_u8(svcmplt_u8(pg, verticalAbsDifference, horizontalAbsDifference), Average(greenTop, greenBottom), green);
            return svsel_u8(svcmpgt_u8(pg, verticalAbsDifference, horizontalAbsDifference), Average(greenRight, greenLeft), green);
        }

        SIMD_INLINE BayerPair Load2(const uint8_t* src, const svbool_t& mask)
        {
            svuint8x2_t _src = svld2_u8(mask, src);
            BayerPair dst;
            dst.val[0] = svget2(_src, 0);
            dst.val[1] = svget2(_src, 1);
            return dst;
        }

        SIMD_INLINE void LoadBayerBody(const uint8_t* src[6], size_t col, const svbool_t& mask, BayerPair dst[12])
        {
            dst[1] = Load2(src[0] + col, mask);
            dst[0] = Load2(src[1] + col - 1, mask);
            dst[2] = Load2(src[1] + col + 1, mask);
            dst[3] = Load2(src[2] + col - 2, mask);
            dst[4] = Load2(src[2] + col, mask);
            dst[5] = Load2(src[2] + col + 2, mask);
            dst[6] = Load2(src[3] + col - 2, mask);
            dst[7] = Load2(src[3] + col, mask);
            dst[8] = Load2(src[3] + col + 2, mask);
            dst[9] = Load2(src[4] + col - 1, mask);
            dst[10] = Load2(src[5] + col, mask);
            dst[11] = Load2(src[4] + col + 1, mask);
        }

        SIMD_INLINE void SaveBgr(BayerPair src[3], uint8_t* dst, size_t pairs, size_t A)
        {
            size_t pixels = 2 * pairs, A3 = A * 3;
            svuint8x3_t lo = svcreate3_u8(
                svzip1_u8(src[0].val[0], src[0].val[1]),
                svzip1_u8(src[1].val[0], src[1].val[1]),
                svzip1_u8(src[2].val[0], src[2].val[1]));
            svuint8x3_t hi = svcreate3_u8(
                svzip2_u8(src[0].val[0], src[0].val[1]),
                svzip2_u8(src[1].val[0], src[1].val[1]),
                svzip2_u8(src[2].val[0], src[2].val[1]));
            svst3_u8(svwhilelt_b8(size_t(0), pixels), dst, lo);
            if (pixels > A)
                svst3_u8(svwhilelt_b8(A, pixels), dst + A3, hi);
        }

        template <SimdPixelFormatType bayerFormat> void BayerToBgr(const BayerPair s[12], BayerPair d[6]);

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerGrbg>(const BayerPair s[12], BayerPair d[6])
        {
            d[0].val[0] = Average(s[0].val[1], s[7].val[0]);
            d[0].val[1] = Average(s[0].val[1], s[2].val[1], s[7].val[0], s[8].val[0]);
            d[1].val[0] = s[4].val[0];
            d[1].val[1] = BayerToGreen(s[4].val[0], s[2].val[0], s[5].val[0], s[7].val[1], s[3].val[1], s[1].val[1], s[5].val[1], s[11].val[0]);
            d[2].val[0] = Average(s[3].val[1], s[4].val[1]);
            d[2].val[1] = s[4].val[1];
            d[3].val[0] = s[7].val[0];
            d[3].val[1] = Average(s[7].val[0], s[8].val[0]);
            d[4].val[0] = BayerToGreen(s[6].val[1], s[4].val[0], s[7].val[1], s[9].val[1], s[6].val[0], s[0].val[1], s[8].val[0], s[10].val[0]);
            d[4].val[1] = s[7].val[1];
            d[5].val[0] = Average(s[3].val[1], s[4].val[1], s[9].val[0], s[11].val[0]);
            d[5].val[1] = Average(s[4].val[1], s[11].val[0]);
        }

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerGbrg>(const BayerPair s[12], BayerPair d[6])
        {
            d[0].val[0] = Average(s[3].val[1], s[4].val[1]);
            d[0].val[1] = s[4].val[1];
            d[1].val[0] = s[4].val[0];
            d[1].val[1] = BayerToGreen(s[4].val[0], s[2].val[0], s[5].val[0], s[7].val[1], s[3].val[1], s[1].val[1], s[5].val[1], s[11].val[0]);
            d[2].val[0] = Average(s[0].val[1], s[7].val[0]);
            d[2].val[1] = Average(s[0].val[1], s[2].val[1], s[7].val[0], s[8].val[0]);
            d[3].val[0] = Average(s[3].val[1], s[4].val[1], s[9].val[0], s[11].val[0]);
            d[3].val[1] = Average(s[4].val[1], s[11].val[0]);
            d[4].val[0] = BayerToGreen(s[6].val[1], s[4].val[0], s[7].val[1], s[9].val[1], s[6].val[0], s[0].val[1], s[8].val[0], s[10].val[0]);
            d[4].val[1] = s[7].val[1];
            d[5].val[0] = s[7].val[0];
            d[5].val[1] = Average(s[7].val[0], s[8].val[0]);
        }

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerRggb>(const BayerPair s[12], BayerPair d[6])
        {
            d[0].val[0] = Average(s[0].val[0], s[2].val[0], s[6].val[1], s[7].val[1]);
            d[0].val[1] = Average(s[2].val[0], s[7].val[1]);
            d[1].val[0] = BayerToGreen(s[3].val[1], s[0].val[1], s[4].val[1], s[7].val[0], s[3].val[0], s[1].val[0], s[5].val[0], s[9].val[1]);
            d[1].val[1] = s[4].val[1];
            d[2].val[0] = s[4].val[0];
            d[2].val[1] = Average(s[4].val[0], s[5].val[0]);
            d[3].val[0] = Average(s[6].val[1], s[7].val[1]);
            d[3].val[1] = s[7].val[1];
            d[4].val[0] = s[7].val[0];
            d[4].val[1] = BayerToGreen(s[7].val[0], s[4].val[1], s[8].val[0], s[11].val[0], s[6].val[1], s[2].val[0], s[8].val[1], s[10].val[1]);
            d[5].val[0] = Average(s[4].val[0], s[9].val[1]);
            d[5].val[1] = Average(s[4].val[0], s[5].val[0], s[9].val[1], s[11].val[1]);
        }

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerBggr>(const BayerPair s[12], BayerPair d[6])
        {
            d[0].val[0] = s[4].val[0];
            d[0].val[1] = Average(s[4].val[0], s[5].val[0]);
            d[1].val[0] = BayerToGreen(s[3].val[1], s[0].val[1], s[4].val[1], s[7].val[0], s[3].val[0], s[1].val[0], s[5].val[0], s[9].val[1]);
            d[1].val[1] = s[4].val[1];
            d[2].val[0] = Average(s[0].val[0], s[2].val[0], s[6].val[1], s[7].val[1]);
            d[2].val[1] = Average(s[2].val[0], s[7].val[1]);
            d[3].val[0] = Average(s[4].val[0], s[9].val[1]);
            d[3].val[1] = Average(s[4].val[0], s[5].val[0], s[9].val[1], s[11].val[1]);
            d[4].val[0] = s[7].val[0];
            d[4].val[1] = BayerToGreen(s[7].val[0], s[4].val[1], s[8].val[0], s[11].val[0], s[6].val[1], s[2].val[0], s[8].val[1], s[10].val[1]);
            d[5].val[0] = Average(s[6].val[1], s[7].val[1]);
            d[5].val[1] = s[7].val[1];
        }

        template <SimdPixelFormatType bayerFormat> SIMD_INLINE void BayerToBgr(const uint8_t* src[6], size_t col, uint8_t* bgr, size_t bgrStride, size_t pairs, size_t A)
        {
            BayerPair _src[12], _bgr[6];
            LoadBayerBody(src, col, svwhilelt_b8(size_t(0), pairs), _src);
            BayerToBgr<bayerFormat>(_src, _bgr);
            SaveBgr(_bgr + 0, bgr, pairs, A);
            SaveBgr(_bgr + 3, bgr + bgrStride, pairs, A);
        }

        template <SimdPixelFormatType bayerFormat> SIMD_INLINE void BayerToBgrEdge(const uint8_t* src[6],
            size_t col0, size_t col2, size_t col4, uint8_t* dst, size_t stride)
        {
            Base::BayerToBgr<bayerFormat>(src,
                col0, col0 + 1, col2, col2 + 1, col4, col4 + 1,
                dst, dst + 3, dst + stride, dst + stride + 3);
        }

        template <SimdPixelFormatType bayerFormat> void BayerToBgr(const uint8_t* bayer, size_t width, size_t height, size_t bayerStride, uint8_t* bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && width >= 4);

            size_t A = svlen(svuint8_t()), pairs = (width - 4) / 2;
            const uint8_t* src[6];
            for (size_t row = 0; row < height; row += 2)
            {
                src[0] = (row == 0 ? bayer : bayer - 2 * bayerStride);
                src[1] = src[0] + bayerStride;
                src[2] = bayer;
                src[3] = src[2] + bayerStride;
                src[4] = (row == height - 2 ? bayer : bayer + 2 * bayerStride);
                src[5] = src[4] + bayerStride;

                BayerToBgrEdge<bayerFormat>(src, 0, 0, 2, bgr, bgrStride);

                for (size_t pair = 0; pair < pairs; pair += A)
                {
                    size_t count = Simd::Min(A, pairs - pair);
                    size_t col = 2 + 2 * pair;
                    BayerToBgr<bayerFormat>(src, col, bgr + 3 * col, bgrStride, count, A);
                }

                BayerToBgrEdge<bayerFormat>(src, width - 4, width - 2, width - 2, bgr + 3 * (width - 2), bgrStride);

                bayer += 2 * bayerStride;
                bgr += 2 * bgrStride;
            }
        }

        void BayerToBgr(const uint8_t* bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t* bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            switch (bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                BayerToBgr<SimdPixelFormatBayerGrbg>(bayer, width, height, bayerStride, bgr, bgrStride);
                break;
            case SimdPixelFormatBayerGbrg:
                BayerToBgr<SimdPixelFormatBayerGbrg>(bayer, width, height, bayerStride, bgr, bgrStride);
                break;
            case SimdPixelFormatBayerRggb:
                BayerToBgr<SimdPixelFormatBayerRggb>(bayer, width, height, bayerStride, bgr, bgrStride);
                break;
            case SimdPixelFormatBayerBggr:
                BayerToBgr<SimdPixelFormatBayerBggr>(bayer, width, height, bayerStride, bgr, bgrStride);
                break;
            default:
                assert(0);
            }
        }
    }
#endif
}
