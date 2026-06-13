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
#include "Simd/SimdMemory.h"
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svint16_t PackI32ToI16(const svint32_t& lo, const svint32_t& hi)
        {
            return svqxtnt_s32(svqxtnb_s32(lo), hi);
        }

        SIMD_INLINE svuint8_t PackSaturatedI16ToU8(const svint16_t& lo, const svint16_t& hi)
        {
            return svqxtunt_s16(svqxtunb_s16(lo), hi);
        }

        SIMD_INLINE svuint8_t PackSequentialI16ToU8(const svint16_t& value)
        {
            return PackSaturatedI16ToU8(svuzp1_s16(value, value), svuzp2_s16(value, value));
        }

        template<class T> SIMD_INLINE svint32_t BgrToY32(const svuint32_t& blue, const svuint32_t& green, const svuint32_t& red)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t y = svdup_n_s32(T::B_ROUND);
            y = svadd_s32_x(mask, y, svmul_n_s32_x(mask, svreinterpret_s32_u32(blue), T::B_2_Y));
            y = svadd_s32_x(mask, y, svmul_n_s32_x(mask, svreinterpret_s32_u32(green), T::G_2_Y));
            y = svadd_s32_x(mask, y, svmul_n_s32_x(mask, svreinterpret_s32_u32(red), T::R_2_Y));
            return svasr_n_s32_x(mask, y, T::B_SHIFT);
        }

        template<class T> SIMD_INLINE svint16_t BgrToY16(const svuint16_t& blue, const svuint16_t& green, const svuint16_t& red)
        {
            const svbool_t mask = svptrue_b16();
            return svadd_n_s16_x(mask, PackI32ToI16(
                BgrToY32<T>(svmovlb_u32(blue), svmovlb_u32(green), svmovlb_u32(red)),
                BgrToY32<T>(svmovlt_u32(blue), svmovlt_u32(green), svmovlt_u32(red))), T::Y_LO);
        }

        template<class T> SIMD_INLINE svuint8_t BgrToY8(const svuint8_t& blue, const svuint8_t& green, const svuint8_t& red)
        {
            return PackSaturatedI16ToU8(
                BgrToY16<T>(svmovlb_u16(blue), svmovlb_u16(green), svmovlb_u16(red)),
                BgrToY16<T>(svmovlt_u16(blue), svmovlt_u16(green), svmovlt_u16(red)));
        }

        template<class T> SIMD_INLINE svint32_t BgrToU32(const svuint32_t& blue, const svuint32_t& green, const svuint32_t& red)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t u = svdup_n_s32(T::B_ROUND);
            u = svadd_s32_x(mask, u, svmul_n_s32_x(mask, svreinterpret_s32_u32(blue), T::B_2_U));
            u = svadd_s32_x(mask, u, svmul_n_s32_x(mask, svreinterpret_s32_u32(green), T::G_2_U));
            u = svadd_s32_x(mask, u, svmul_n_s32_x(mask, svreinterpret_s32_u32(red), T::R_2_U));
            return svasr_n_s32_x(mask, u, T::B_SHIFT);
        }

        template<class T> SIMD_INLINE svint16_t BgrToU16(const svuint16_t& blue, const svuint16_t& green, const svuint16_t& red)
        {
            const svbool_t mask = svptrue_b16();
            return svadd_n_s16_x(mask, PackI32ToI16(
                BgrToU32<T>(svmovlb_u32(blue), svmovlb_u32(green), svmovlb_u32(red)),
                BgrToU32<T>(svmovlt_u32(blue), svmovlt_u32(green), svmovlt_u32(red))), T::UV_Z);
        }

        template<class T> SIMD_INLINE svint32_t BgrToV32(const svuint32_t& blue, const svuint32_t& green, const svuint32_t& red)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t v = svdup_n_s32(T::B_ROUND);
            v = svadd_s32_x(mask, v, svmul_n_s32_x(mask, svreinterpret_s32_u32(blue), T::B_2_V));
            v = svadd_s32_x(mask, v, svmul_n_s32_x(mask, svreinterpret_s32_u32(green), T::G_2_V));
            v = svadd_s32_x(mask, v, svmul_n_s32_x(mask, svreinterpret_s32_u32(red), T::R_2_V));
            return svasr_n_s32_x(mask, v, T::B_SHIFT);
        }

        template<class T> SIMD_INLINE svint16_t BgrToV16(const svuint16_t& blue, const svuint16_t& green, const svuint16_t& red)
        {
            const svbool_t mask = svptrue_b16();
            return svadd_n_s16_x(mask, PackI32ToI16(
                BgrToV32<T>(svmovlb_u32(blue), svmovlb_u32(green), svmovlb_u32(red)),
                BgrToV32<T>(svmovlt_u32(blue), svmovlt_u32(green), svmovlt_u32(red))), T::UV_Z);
        }

        SIMD_INLINE svuint16_t Average(const svuint8_t& row0, const svuint8_t& row1)
        {
            const svbool_t mask = svptrue_b16();
            svuint16_t sum = svadd_u16_x(mask, svmovlb_u16(svuzp1_u8(row0, row0)), svmovlb_u16(svuzp2_u8(row0, row0)));
            sum = svadd_u16_x(mask, sum, svmovlb_u16(svuzp1_u8(row1, row1)));
            sum = svadd_u16_x(mask, sum, svmovlb_u16(svuzp2_u8(row1, row1)));
            return svlsr_n_u16_x(mask, svadd_n_u16_x(mask, sum, 2), 2);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE void BgrToYuv420pV2(const uint8_t* bgr0, size_t bgrStride, uint8_t* y0, size_t yStride,
            uint8_t* u, uint8_t* v, const svbool_t& maskY, const svbool_t& maskUv)
        {
            const uint8_t* bgr1 = bgr0 + bgrStride;
            uint8_t* y1 = y0 + yStride;

            svuint8x3_t bgr00 = svld3_u8(maskY, bgr0);
            svuint8x3_t bgr10 = svld3_u8(maskY, bgr1);

            svst1_u8(maskY, y0, BgrToY8<T>(svget3(bgr00, 0), svget3(bgr00, 1), svget3(bgr00, 2)));
            svst1_u8(maskY, y1, BgrToY8<T>(svget3(bgr10, 0), svget3(bgr10, 1), svget3(bgr10, 2)));

            svuint16_t blue = Average(svget3(bgr00, 0), svget3(bgr10, 0));
            svuint16_t green = Average(svget3(bgr00, 1), svget3(bgr10, 1));
            svuint16_t red = Average(svget3(bgr00, 2), svget3(bgr10, 2));

            svst1_u8(maskUv, u, PackSequentialI16ToU8(BgrToU16<T>(blue, green, red)));
            svst1_u8(maskUv, v, PackSequentialI16ToU8(BgrToV16<T>(blue, green, red)));
        }

        template <class T> void BgrToYuv420pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            size_t A = svlen(svuint8_t());
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t col = 0; col < width; col += A)
                {
                    size_t block = Simd::Min(A, width - col);
                    BgrToYuv420pV2<T>(bgr + col * 3, bgrStride, y + col, yStride, u + col / 2, v + col / 2,
                        svwhilelt_b8(size_t(0), block), svwhilelt_b8(size_t(0), block / 2));
                }
                bgr += 2 * bgrStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
            }
        }

        void BgrToYuv420pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv420pV2<Base::Bt601>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv420pV2<Base::Bt709>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv420pV2<Base::Bt2020>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv420pV2<Base::Trect871>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }
    }
#endif
}
