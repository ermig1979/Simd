/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdYuvToBgr.h"

#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        SIMD_INLINE uint16x8_t Average(uint8x16_t a, uint8x16_t b)
        {
            return vshrq_n_u16(vpadalq_u8(vpadalq_u8(K16_0002, a), b), 2);
        }

        SIMD_INLINE uint16x8_t Average(uint8x16_t value)
        {
            return vshrq_n_u16(vpadalq_u8(K16_0001, value), 1);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool align> SIMD_INLINE void BgrToYuv420pV2(const uint8_t* bgr0, size_t bgrStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v)
        {
            const uint8_t* bgr1 = bgr0 + bgrStride;
            uint8_t* y1 = y0 + yStride;

            uint8x16x3_t bgr00 = Load3<align>(bgr0 + 0 * A);
            Store<align>(y0 + 0, BgrToY8<T>(bgr00.val[0], bgr00.val[1], bgr00.val[2]));

            uint8x16x3_t bgr01 = Load3<align>(bgr0 + 3 * A);
            Store<align>(y0 + A, BgrToY8<T>(bgr01.val[0], bgr01.val[1], bgr01.val[2]));

            uint8x16x3_t bgr10 = Load3<align>(bgr1 + 0 * A);
            Store<align>(y1 + 0, BgrToY8<T>(bgr10.val[0], bgr10.val[1], bgr10.val[2]));

            uint8x16x3_t bgr11 = Load3<align>(bgr1 + 3 * A);
            Store<align>(y1 + A, BgrToY8<T>(bgr11.val[0], bgr11.val[1], bgr11.val[2]));

            uint16x8_t b0 = Average(bgr00.val[0], bgr10.val[0]);
            uint16x8_t g0 = Average(bgr00.val[1], bgr10.val[1]);
            uint16x8_t r0 = Average(bgr00.val[2], bgr10.val[2]);

            uint16x8_t b1 = Average(bgr01.val[0], bgr11.val[0]);
            uint16x8_t g1 = Average(bgr01.val[1], bgr11.val[1]);
            uint16x8_t r1 = Average(bgr01.val[2], bgr11.val[2]);

            Store<align>(u, PackSaturatedI16(BgrToU16<T>(b0, g0, r0), BgrToU16<T>(b1, g1, r1)));
            Store<align>(v, PackSaturatedI16(BgrToV16<T>(b0, g0, r0), BgrToV16<T>(b1, g1, r1)));
        }

        template <class T, bool align> void BgrToYuv420pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((height % 2 == 0) && (width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t widthDA = AlignLo(width, DA);
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0; colY < widthDA; colY += DA, colUV += A)
                    BgrToYuv420pV2<T, align>(bgr + colY * 3, bgrStride, y + colY, yStride, u + colUV, v + colUV);
                if (width != widthDA)
                {
                    size_t col = width - DA;
                    BgrToYuv420pV2<T, false>(bgr + col * 3, bgrStride, y + col, yStride, u + col / 2, v + col / 2);
                }
                bgr += 2 * bgrStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
            }
        }

        template <bool align> void BgrToYuv420pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv420pV2<Base::Bt601, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv420pV2<Base::Bt709, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv420pV2<Base::Bt2020, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv420pV2<Base::Trect871, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        void BgrToYuv420pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToYuv420pV2<true>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
            else
                BgrToYuv420pV2<false>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool align> SIMD_INLINE void BgrToYuv422pV2(const uint8_t* bgr, uint8_t* y, uint8_t* u, uint8_t* v)
        {
            uint8x16x3_t bgr0 = Load3<align>(bgr + 0 * A);
            Store<align>(y + 0, BgrToY8<T>(bgr0.val[0], bgr0.val[1], bgr0.val[2]));

            uint16x8_t b0 = Average(bgr0.val[0]);
            uint16x8_t g0 = Average(bgr0.val[1]);
            uint16x8_t r0 = Average(bgr0.val[2]);

            uint8x16x3_t bgr1 = Load3<align>(bgr + 3 * A);
            Store<align>(y + A, BgrToY8<T>(bgr1.val[0], bgr1.val[1], bgr1.val[2]));

            uint16x8_t b1 = Average(bgr1.val[0]);
            uint16x8_t g1 = Average(bgr1.val[1]);
            uint16x8_t r1 = Average(bgr1.val[2]);

            Store<align>(u, PackSaturatedI16(BgrToU16<T>(b0, g0, r0), BgrToU16<T>(b1, g1, r1)));
            Store<align>(v, PackSaturatedI16(BgrToV16<T>(b0, g0, r0), BgrToV16<T>(b1, g1, r1)));
        }

        template <class T, bool align> void BgrToYuv422pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t widthDA = AlignLo(width, DA);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colY = 0; colY < widthDA; colY += DA, colUV += A)
                    BgrToYuv422pV2<T, align>(bgr + colY * 3, y + colY, u + colUV, v + colUV);
                if (width != widthDA)
                {
                    size_t col = width - DA;
                    BgrToYuv422pV2<T, false>(bgr + col * 3, y + col, u + col / 2, v + col / 2);
                }
                bgr += bgrStride;
                y += yStride;
                u += uStride;
                v += vStride;
            }
        }

        template <bool align> void BgrToYuv422pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv422pV2<Base::Bt601, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv422pV2<Base::Bt709, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv422pV2<Base::Bt2020, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv422pV2<Base::Trect871, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        void BgrToYuv422pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToYuv422pV2<true>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
            else
                BgrToYuv422pV2<false>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool align> SIMD_INLINE void BgrToYuv444pV2(const uint8_t* bgr, uint8_t* y, uint8_t* u, uint8_t* v)
        {
            uint8x16x3_t _bgr = Load3<align>(bgr);
            Store<align>(y, BgrToY8<T>(_bgr.val[0], _bgr.val[1], _bgr.val[2]));
            Store<align>(u, BgrToU8<T>(_bgr.val[0], _bgr.val[1], _bgr.val[2]));
            Store<align>(v, BgrToV8<T>(_bgr.val[0], _bgr.val[1], _bgr.val[2]));
        }

        template <class T, bool align> void BgrToYuv444pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t widthA = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < widthA; col += A)
                    BgrToYuv444pV2<T, align>(bgr + col * 3, y + col, u + col, v + col);
                if (width != widthA)
                {
                    size_t col = width - A;
                    BgrToYuv444pV2<T, false>(bgr + col * 3, y + col, u + col, v + col);
                }
                bgr += bgrStride;
                y += yStride;
                u += uStride;
                v += vStride;
            }
        }

        template <bool align> void BgrToYuv444pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv444pV2<Base::Bt601, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv444pV2<Base::Bt709, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv444pV2<Base::Bt2020, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv444pV2<Base::Trect871, align>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        void BgrToYuv444pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                BgrToYuv444pV2<true>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
            else
                BgrToYuv444pV2<false>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
        }
    }
#endif// SIMD_NEON_ENABLE
}
