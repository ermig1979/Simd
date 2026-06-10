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

        SIMD_INLINE void Load2(const uint8_t* src, const svbool_t& mask, svuint8_t& even, svuint8_t& odd)
        {
            svuint8x2_t _src = svld2_u8(mask, src);
            even = svget2(_src, 0);
            odd = svget2(_src, 1);
        }

        SIMD_INLINE void SaveBgr(const svuint8_t& b0, const svuint8_t& b1, const svuint8_t& g0, const svuint8_t& g1,
            const svuint8_t& r0, const svuint8_t& r1, uint8_t* dst, size_t pairs, size_t A)
        {
            size_t pixels = 2 * pairs, A3 = A * 3;
            svuint8x3_t lo = svcreate3_u8(svzip1_u8(b0, b1), svzip1_u8(g0, g1), svzip1_u8(r0, r1));
            svuint8x3_t hi = svcreate3_u8(svzip2_u8(b0, b1), svzip2_u8(g0, g1), svzip2_u8(r0, r1));
            svst3_u8(svwhilelt_b8(size_t(0), pixels), dst, lo);
            if (pixels > A)
                svst3_u8(svwhilelt_b8(A, pixels), dst + A3, hi);
        }

#define SIMD_SVE2_LOAD_BAYER_BODY(src, col, mask) \
            svuint8_t s0e, s0o, s1e, s1o, s2e, s2o, s3e, s3o, s4e, s4o, s5e, s5o; \
            svuint8_t s6e, s6o, s7e, s7o, s8e, s8o, s9e, s9o, s10e, s10o, s11e, s11o; \
            Load2(src[0] + col, mask, s1e, s1o); \
            Load2(src[1] + col - 1, mask, s0e, s0o); \
            Load2(src[1] + col + 1, mask, s2e, s2o); \
            Load2(src[2] + col - 2, mask, s3e, s3o); \
            Load2(src[2] + col, mask, s4e, s4o); \
            Load2(src[2] + col + 2, mask, s5e, s5o); \
            Load2(src[3] + col - 2, mask, s6e, s6o); \
            Load2(src[3] + col, mask, s7e, s7o); \
            Load2(src[3] + col + 2, mask, s8e, s8o); \
            Load2(src[4] + col - 1, mask, s9e, s9o); \
            Load2(src[5] + col, mask, s10e, s10o); \
            Load2(src[4] + col + 1, mask, s11e, s11o)

        template <SimdPixelFormatType bayerFormat> void BayerToBgr(const uint8_t* src[6], size_t col, uint8_t* bgr, size_t bgrStride, size_t pairs, size_t A);

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerGrbg>(const uint8_t* src[6], size_t col, uint8_t* bgr, size_t bgrStride, size_t pairs, size_t A)
        {
            const svbool_t mask = svwhilelt_b8(size_t(0), pairs);
            SIMD_SVE2_LOAD_BAYER_BODY(src, col, mask);

            svuint8_t d0e = Average(s0o, s7e);
            svuint8_t d0o = Average(s0o, s2o, s7e, s8e);
            svuint8_t d1e = s4e;
            svuint8_t d1o = BayerToGreen(s4e, s2e, s5e, s7o, s3o, s1o, s5o, s11e);
            svuint8_t d2e = Average(s3o, s4o);
            svuint8_t d2o = s4o;
            svuint8_t d3e = s7e;
            svuint8_t d3o = Average(s7e, s8e);
            svuint8_t d4e = BayerToGreen(s6o, s4e, s7o, s9o, s6e, s0o, s8e, s10e);
            svuint8_t d4o = s7o;
            svuint8_t d5e = Average(s3o, s4o, s9e, s11e);
            svuint8_t d5o = Average(s4o, s11e);

            SaveBgr(d0e, d0o, d1e, d1o, d2e, d2o, bgr, pairs, A);
            SaveBgr(d3e, d3o, d4e, d4o, d5e, d5o, bgr + bgrStride, pairs, A);
        }

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerGbrg>(const uint8_t* src[6], size_t col, uint8_t* bgr, size_t bgrStride, size_t pairs, size_t A)
        {
            const svbool_t mask = svwhilelt_b8(size_t(0), pairs);
            SIMD_SVE2_LOAD_BAYER_BODY(src, col, mask);

            svuint8_t d0e = Average(s3o, s4o);
            svuint8_t d0o = s4o;
            svuint8_t d1e = s4e;
            svuint8_t d1o = BayerToGreen(s4e, s2e, s5e, s7o, s3o, s1o, s5o, s11e);
            svuint8_t d2e = Average(s0o, s7e);
            svuint8_t d2o = Average(s0o, s2o, s7e, s8e);
            svuint8_t d3e = Average(s3o, s4o, s9e, s11e);
            svuint8_t d3o = Average(s4o, s11e);
            svuint8_t d4e = BayerToGreen(s6o, s4e, s7o, s9o, s6e, s0o, s8e, s10e);
            svuint8_t d4o = s7o;
            svuint8_t d5e = s7e;
            svuint8_t d5o = Average(s7e, s8e);

            SaveBgr(d0e, d0o, d1e, d1o, d2e, d2o, bgr, pairs, A);
            SaveBgr(d3e, d3o, d4e, d4o, d5e, d5o, bgr + bgrStride, pairs, A);
        }

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerRggb>(const uint8_t* src[6], size_t col, uint8_t* bgr, size_t bgrStride, size_t pairs, size_t A)
        {
            const svbool_t mask = svwhilelt_b8(size_t(0), pairs);
            SIMD_SVE2_LOAD_BAYER_BODY(src, col, mask);

            svuint8_t d0e = Average(s0e, s2e, s6o, s7o);
            svuint8_t d0o = Average(s2e, s7o);
            svuint8_t d1e = BayerToGreen(s3o, s0o, s4o, s7e, s3e, s1e, s5e, s9o);
            svuint8_t d1o = s4o;
            svuint8_t d2e = s4e;
            svuint8_t d2o = Average(s4e, s5e);
            svuint8_t d3e = Average(s6o, s7o);
            svuint8_t d3o = s7o;
            svuint8_t d4e = s7e;
            svuint8_t d4o = BayerToGreen(s7e, s4o, s8e, s11e, s6o, s2e, s8o, s10o);
            svuint8_t d5e = Average(s4e, s9o);
            svuint8_t d5o = Average(s4e, s5e, s9o, s11o);

            SaveBgr(d0e, d0o, d1e, d1o, d2e, d2o, bgr, pairs, A);
            SaveBgr(d3e, d3o, d4e, d4o, d5e, d5o, bgr + bgrStride, pairs, A);
        }

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerBggr>(const uint8_t* src[6], size_t col, uint8_t* bgr, size_t bgrStride, size_t pairs, size_t A)
        {
            const svbool_t mask = svwhilelt_b8(size_t(0), pairs);
            SIMD_SVE2_LOAD_BAYER_BODY(src, col, mask);

            svuint8_t d0e = s4e;
            svuint8_t d0o = Average(s4e, s5e);
            svuint8_t d1e = BayerToGreen(s3o, s0o, s4o, s7e, s3e, s1e, s5e, s9o);
            svuint8_t d1o = s4o;
            svuint8_t d2e = Average(s0e, s2e, s6o, s7o);
            svuint8_t d2o = Average(s2e, s7o);
            svuint8_t d3e = Average(s4e, s9o);
            svuint8_t d3o = Average(s4e, s5e, s9o, s11o);
            svuint8_t d4e = s7e;
            svuint8_t d4o = BayerToGreen(s7e, s4o, s8e, s11e, s6o, s2e, s8o, s10o);
            svuint8_t d5e = Average(s6o, s7o);
            svuint8_t d5o = s7o;

            SaveBgr(d0e, d0o, d1e, d1o, d2e, d2o, bgr, pairs, A);
            SaveBgr(d3e, d3o, d4e, d4o, d5e, d5o, bgr + bgrStride, pairs, A);
        }

#undef SIMD_SVE2_LOAD_BAYER_BODY

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
